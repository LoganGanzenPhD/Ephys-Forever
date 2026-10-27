"""Exponential fits on the filtered IV steps recording against the commercial tool's tables."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ephys_forever.core.analysis.fit import FitResult, exponential, fit_region
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.io.atf import read_atf
from ephys_forever.core.model import Recording

pytestmark = [pytest.mark.reference, pytest.mark.realdata]

CLAMPFIT = Path(__file__).resolve().parents[1] / "reference" / "clampfit"
CASES = ["vclamp_iv_steps_001_exp_fits", "vclamp_iv_steps_001_exp_fits_simplex"]
PA, MS = 1e-12, 1e-3
TERMS = {"single": 1, "double": 2}


@pytest.fixture(scope="module", params=CASES)
def settings(request: pytest.FixtureRequest) -> dict[str, Any]:
    with (CLAMPFIT / request.param / "settings.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    data["case_dir"] = CLAMPFIT / request.param
    return data


@pytest.fixture(scope="module")
def recording(settings: dict[str, Any], real_data_file: Callable[[str], Path]) -> Recording:
    return read_atf(real_data_file(settings["source_file"]))


def reference_rows(settings: dict[str, Any], fit: str) -> list[dict[str, str]]:
    """Rows of one table, keyed by header name; repeated "S.E." headers get the column.

    Tables saved as ATF start with two ATF header lines, which are skipped.
    """
    with (settings["case_dir"] / settings["outputs"][fit]).open(encoding="ascii", newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if rows[0][0] == "ATF":
        rows = rows[2:]
    header = [f"{h}@{i}" if h == "S.E." else h for i, h in enumerate(rows[0])]
    return [dict(zip(header, row, strict=True)) for row in rows[1:]]


def parameters(settings: dict[str, Any], fit: str, row: dict[str, str]) -> dict[str, float]:
    """Reference parameters in SI, by Ephys-Forever's names (A1, tau1, ..., C)."""
    values = list(row.values())
    out = {}
    for name, column in settings["columns"][fit].items():
        if name.endswith("_se"):
            continue
        key = {"A": "A1", "tau": "tau1"}.get(name, name)
        out[key] = float(values[column]) * (MS if key.startswith("tau") else PA)
    return out


def printed_rtol(text: str) -> float:
    """Half a unit in the last printed significant digit, relative (``4.39e+06``: 3 digits)."""
    mantissa = text.lower().split("e")[0].lstrip("-+").replace(".", "").lstrip("0")
    return 0.5 * 10.0 ** (1 - len(mantissa))


def region(settings: dict[str, Any]) -> Region:
    start, stop = settings["region"]["region_ms"]
    return Region(start * MS, stop * MS)


@pytest.fixture(scope="module")
def ours(settings: dict[str, Any], recording: Recording) -> dict[str, list[FitResult]]:
    ch = recording.channel_index(settings["channel"])
    return {
        fit: [
            fit_region(exponential(n), s.data[ch], recording.sample_rate, region(settings))
            for s in recording.sweeps
        ]
        for fit, n in TERMS.items()
    }


@pytest.mark.parametrize("fit", TERMS)
def test_window_and_x_origin_reproduce_the_reference_sums(
    settings: dict[str, Any], recording: Recording, fit: str, tolerances: dict[str, Any]
) -> None:
    """The reference parameters, evaluated on our window, give the reference's own SSE."""
    rtol = tolerances["fit"]["sse_vs_clampfit"]["rtol"]
    ch = recording.channel_index(settings["channel"])
    first, last = settings["region"]["first_sample"], settings["region"]["last_sample"]
    model = exponential(TERMS[fit])
    coarse = {(c["fit"], c["sweep"]) for c in settings.get("coarse_parameters", [])}
    for row, sweep in zip(reference_rows(settings, fit), recording.sweeps, strict=True):
        assert int(row["Points Fitted"]) == last - first + 1
        if (fit, int(row["Trace"])) in coarse:
            continue
        y = sweep.data[ch][first : last + 1] / PA
        p = parameters(settings, fit, row)
        x = np.arange(y.size) / recording.sample_rate
        f = model.function(x, np.array([p[n] for n in model.parameters])) / PA
        sum_sq = row["Sum of Data Squared"]
        assert np.sum(y**2) == pytest.approx(float(sum_sq), rel=printed_rtol(sum_sq))
        sse = float(np.sum((y - f) ** 2))
        assert sse == pytest.approx(float(row["Sum of Squared Errors"]), rel=rtol)
        sd = math.sqrt(sse / (y.size - 1))  # the tool's "Standard Deviation"
        assert sd == pytest.approx(float(row["Standard Deviation"]), rel=rtol)


@pytest.mark.parametrize("fit", TERMS)
def test_least_squares_is_at_least_as_good_as_the_reference(
    settings: dict[str, Any], ours: dict[str, list[FitResult]], fit: str, tolerances: dict[str, Any]
) -> None:
    rtol = tolerances["fit"]["sse_vs_clampfit"]["rtol"]
    excluded = {(e["fit"], e["sweep"]) for e in settings["excluded"]}
    compared = 0
    for row, result in zip(reference_rows(settings, fit), ours[fit], strict=True):
        sweep = int(row["Trace"])
        assert result.n_points == int(row["Points Fitted"])
        if (fit, sweep) in excluded:
            continue
        reference_sse = float(row["Sum of Squared Errors"]) * PA**2
        assert result.sse <= reference_sse * (1 + rtol), f"{fit} sweep {sweep}"
        compared += 1
    assert compared == 9 - sum(1 for e in settings["excluded"] if e["fit"] == fit)


@pytest.mark.parametrize("fit", TERMS)
def test_parameters_match_a_least_squares_reference(
    settings: dict[str, Any], ours: dict[str, list[FitResult]], fit: str, tolerances: dict[str, Any]
) -> None:
    """Where the reference is itself a least-squares fit, the parameters agree."""
    sweeps = settings.get("compare_parameters", {}).get(fit, [])
    if not sweeps:
        pytest.skip("this reference is compared by SSE only")
    rtol = tolerances["fit"]["params_vs_clampfit_least_squares"]["rtol"]
    rows = {int(row["Trace"]): row for row in reference_rows(settings, fit)}
    for sweep in sweeps:
        ref = parameters(settings, fit, rows[sweep])
        # The reference does not order terms; ours are sorted by increasing tau.
        n = TERMS[fit]
        terms = sorted(
            ((ref[f"A{i}"], ref[f"tau{i}"]) for i in range(1, n + 1)), key=lambda t: t[1]
        )
        expected = {"C": ref["C"]}
        for i, (a, tau) in enumerate(terms, start=1):
            expected[f"A{i}"], expected[f"tau{i}"] = a, tau
        got = ours[fit][sweep - 1].params
        for name, value in expected.items():
            assert got[name] == pytest.approx(value, rel=rtol), f"{fit} sweep {sweep} {name}"
