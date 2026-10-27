"""Curve fits against the analytic truth of the synthetic exponential, Boltzmann and Hill cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ephys_forever.core.analysis.fit import (
    FitResult,
    boltzmann,
    custom,
    exponential,
    fit_curve,
    fit_region,
    hill,
)
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.io.atf import read_atf

pytestmark = pytest.mark.reference

STDERR_BOUND = 4.0
"""Noisy fits must land within this many standard errors of the truth."""


def load_truth(directory: Path, case: str) -> dict[str, Any]:
    with (directory / f"{case}.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def read_table(path: Path) -> np.ndarray:
    """Independent parse of a table ATF: the numeric rows below the column titles."""
    lines = path.read_text(encoding="latin-1").splitlines()
    n_records = int(lines[1].split("\t")[0])
    return np.array([[float(v) for v in ln.split("\t")] for ln in lines[3 + n_records :]])


def exp_truth(case: dict[str, Any], sweep: int) -> dict[str, float]:
    params = dict(case["sweeps"][sweep].get("params") or case["params"])
    if "tau" in params:  # single-exponential truth uses A, tau, C
        return {"A1": params["A"], "tau1": params["tau"], "C": params["C"]}
    return params


def check(
    result: FitResult, truth: dict[str, float], noise: float, tolerances: dict[str, Any]
) -> None:
    assert result.converged, result.message
    if noise == 0.0:
        for name, value in truth.items():
            assert result.params[name] == pytest.approx(value, rel=tolerances["default"]["rtol"])
        return
    tau_rtol = tolerances["fit"]["tau_vs_synthetic"]["rtol"]
    for name, value in truth.items():
        got, err = result.params[name], result.stderr[name]
        assert abs(got - value) < STDERR_BOUND * err, f"{name}: {got} vs {value} (se {err})"
        if name.startswith("tau"):
            assert got == pytest.approx(value, rel=tau_rtol)


@pytest.mark.parametrize(
    ("case", "n_terms"),
    [("exp_decay_single", 1), ("exp_decay_double", 2), ("exp_decay_triple", 3)],
)
def test_exponential_fits_recover_the_truth_without_start_values(
    case: str,
    n_terms: int,
    synthetic_reference_dir: Path,
    synthetic_data_dir: Path,
    tolerances: dict[str, Any],
) -> None:
    truth = load_truth(synthetic_reference_dir, case)
    rec = read_atf(synthetic_data_dir / truth["file"])
    for k, sweep in enumerate(rec.sweeps):
        t = np.arange(sweep.n_samples) / sweep.sample_rate
        result = fit_curve(exponential(n_terms), t, sweep.data[0])
        check(result, exp_truth(truth, k), truth["sweeps"][k]["noise_sigma"], tolerances)


def test_double_exponential_ignores_poor_start_values(
    synthetic_reference_dir: Path, synthetic_data_dir: Path, tolerances: dict[str, Any]
) -> None:
    truth = load_truth(synthetic_reference_dir, "exp_decay_double")
    sweep = read_atf(synthetic_data_dir / truth["file"]).sweeps[2]  # tau ratio 20
    t = np.arange(sweep.n_samples) / sweep.sample_rate
    for start in ([1e-12, 1e-4, 1e-12, 0.2, 0.0], [-5e-11, 0.03, 5e-11, 0.001, 1e-11]):
        result = fit_curve(exponential(2), t, sweep.data[0], start)
        for name, value in exp_truth(truth, 2).items():
            assert result.params[name] == pytest.approx(value, rel=1e-4), (start, name)


def test_fit_region_reports_amplitude_at_the_first_fitted_sample(
    synthetic_reference_dir: Path, synthetic_data_dir: Path
) -> None:
    truth = load_truth(synthetic_reference_dir, "exp_decay_single")
    sweep = read_atf(synthetic_data_dir / truth["file"]).sweeps[0]
    p = truth["params"]
    result = fit_region(exponential(1), sweep.data[0], sweep.sample_rate, Region(0.01, 0.2))
    assert result.x_offset == pytest.approx(0.01)
    assert result.n_points == 3801
    assert result.params["A1"] == pytest.approx(p["A"] * np.exp(-0.01 / p["tau"]), rel=1e-6)
    assert result.params["tau1"] == pytest.approx(p["tau"], rel=1e-6)


def test_custom_expression_matches_the_built_in_exponential(
    synthetic_reference_dir: Path, synthetic_data_dir: Path
) -> None:
    truth = load_truth(synthetic_reference_dir, "exp_decay_single")
    sweep = read_atf(synthetic_data_dir / truth["file"]).sweeps[1]
    t = np.arange(sweep.n_samples) / sweep.sample_rate
    built_in = fit_curve(exponential(1), t, sweep.data[0])
    typed = fit_curve(
        custom("A * exp(-x / tau) + C"), t, sweep.data[0], {"A": 5e-11, "tau": 0.05, "C": 0.0}
    )
    for ours, theirs in (("A", "A1"), ("tau", "tau1"), ("C", "C")):
        assert typed.params[ours] == pytest.approx(built_in.params[theirs], rel=1e-6)
        assert typed.stderr[ours] == pytest.approx(built_in.stderr[theirs], rel=1e-3)


@pytest.mark.parametrize(
    ("case", "make"), [("boltzmann_activation", boltzmann), ("hill_dose_response", hill)]
)
def test_table_fits_recover_the_truth(
    case: str,
    make: Any,
    synthetic_reference_dir: Path,
    synthetic_data_dir: Path,
    tolerances: dict[str, Any],
) -> None:
    truth = load_truth(synthetic_reference_dir, case)
    table = read_table(synthetic_data_dir / truth["file"])
    names = {"Gmax": "top"}
    params = {names.get(k, k): v for k, v in truth["params"].items()}
    clean = fit_curve(make(), table[:, 0], table[:, 1])
    check(clean, params, 0.0, tolerances)
    noisy = fit_curve(make(), table[:, 0], table[:, 2])
    check(noisy, params, truth["noise"]["sigma"], tolerances)
