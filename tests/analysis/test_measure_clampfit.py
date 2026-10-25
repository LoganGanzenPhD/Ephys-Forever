"""Cursor statistics on the IV steps recording against the commercial tool's export."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from ephys_forever.core.analysis.measure import RegionStats, measure_sweeps
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.io.abf2 import read_abf2
from ephys_forever.core.units import parse_unit

pytestmark = [pytest.mark.reference, pytest.mark.realdata]

CASE = Path(__file__).resolve().parents[1] / "reference" / "clampfit" / "vclamp_iv_steps_001_stats"
MS = 1e-3


@pytest.fixture(scope="module")
def settings() -> dict[str, Any]:
    with (CASE / "settings.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


@pytest.fixture(scope="module")
def reference_rows(settings: dict[str, Any]) -> list[tuple[Any, ...]]:
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.load_workbook(CASE / settings["output_file"], read_only=True)
    return [tuple(row) for row in book.active.iter_rows(values_only=True)]


@pytest.fixture(scope="module")
def ours(settings: dict[str, Any], real_data_file: Callable[[str], Path]) -> list[RegionStats]:
    rec = read_abf2(real_data_file(settings["source_file"]))
    cursors = settings["cursors_ms"]
    return measure_sweeps(
        rec,
        rec.channel_index(settings["channel"]),
        Region(cursors["c3"] * MS, cursors["c4"] * MS),
        polarity=settings["peak_polarity"],
        baseline_region=Region(*(t * MS for t in settings["baseline"]["region_ms"])),
        baseline_method=settings["baseline"]["method"],
        rise_fractions=tuple(settings["rise_fractions"]),
    )


def si_factor(unit: str) -> float:
    """SI factor of a unit such as ``pA`` or ``pA*ms``."""
    return math.prod(parse_unit(part).factor for part in unit.split("*"))


def test_one_row_per_sweep_with_the_recorded_settings(
    settings: dict[str, Any], reference_rows: list[tuple[Any, ...]], ours: list[RegionStats]
) -> None:
    assert [row[1] for row in reference_rows] == list(range(1, len(ours) + 1))
    assert {(row[10], row[11]) for row in reference_rows} == {(10, 90)}
    assert all(s.n_samples == 1001 for s in ours)


def test_statistics_match_the_export(
    settings: dict[str, Any],
    reference_rows: list[tuple[Any, ...]],
    ours: list[RegionStats],
    tolerances: dict[str, Any],
) -> None:
    rtol = tolerances["measurements"]["vs_clampfit"]["rtol"]
    bad = {(c["sweep"], c["field"]) for c in settings["known_bad_cells"]}
    compared = 0
    for column, spec in settings["column_map"].items():
        if not isinstance(spec, dict):
            continue
        field, factor = spec["field"], si_factor(spec["unit"])
        for row, stats in zip(reference_rows, ours, strict=True):
            sweep, expected, value = row[1], row[column], getattr(stats, field)
            if (sweep, field) in bad:
                continue
            if expected == "Not found":
                assert math.isnan(value), f"sweep {sweep} {field}: reference has none"
                continue
            assert value == pytest.approx(expected * factor, rel=rtol), f"sweep {sweep} {field}"
            compared += 1
    assert compared == 7 * 9 - 2  # one known-bad cell and one "Not found"


def test_sweep_six_has_no_rise_or_half_width(ours: list[RegionStats]) -> None:
    assert math.isnan(ours[5].rise_time) and math.isnan(ours[5].half_width)
