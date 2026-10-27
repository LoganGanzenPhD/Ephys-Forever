"""Filters against the analytic truth of ``bessel_step_response`` and ``line_noise_60hz``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ephys_forever.core.analysis.filters import bessel_lowpass, notch
from ephys_forever.core.io.atf import read_atf

pytestmark = pytest.mark.reference


def load(directory: Path, case: str) -> dict[str, Any]:
    with (directory / f"{case}.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def crossing(v: np.ndarray, level: float) -> float:
    """Fractional index where ``v`` first rises through ``level``."""
    i = int(np.argmax(v >= level))
    return i - 1 + (level - v[i - 1]) / (v[i] - v[i - 1])


def test_bessel_step_matches_the_analog_filter(
    synthetic_reference_dir: Path, synthetic_data_dir: Path, tolerances: dict[str, Any]
) -> None:
    truth = load(synthetic_reference_dir, "bessel_step_response")
    tol = tolerances["filters"]
    rec = read_atf(synthetic_data_dir / truth["file"])
    fs = rec.sample_rate
    on = truth["input"]["step_sample"]
    step = np.zeros(truth["n_samples"])
    step[on:] = 1.0
    for expect in truth["sweeps"]:
        y = bessel_lowpass(step, fs, expect["cutoff_hz"], truth["filter"]["order"])
        rise = (crossing(y, 0.9) - crossing(y, 0.1)) / fs
        assert rise == pytest.approx(
            expect["rise_time_10_90_s"], rel=tol["rise_time_vs_synthetic"]["rtol"]
        )
        # A bilinear-transform filter responds half a sample earlier than the analog
        # filter to a sampled step (the trapezoid rule treats the step as starting
        # half-way between samples).
        delay = (crossing(y, 0.5) - on + 0.5) / fs
        assert delay == pytest.approx(
            expect["delay_to_50_percent_s"], rel=tol["delay_vs_synthetic"]["rtol"]
        )
        assert y.max() - 1.0 == pytest.approx(
            expect["overshoot_fraction"], abs=tol["overshoot_vs_synthetic"]["atol"]
        )


def test_notch_removes_line_noise_and_keeps_the_signal(
    synthetic_reference_dir: Path, synthetic_data_dir: Path
) -> None:
    truth = load(synthetic_reference_dir, "line_noise_60hz")
    rec = read_atf(synthetic_data_dir / truth["file"])
    fs = rec.sample_rate
    base = truth["base_signal"]
    t = np.arange(truth["n_samples"]) / fs
    clean = base["amplitude"] * np.sin(2 * np.pi * base["frequency_hz"] * t)
    line_peak = sum(c["amplitude"] for c in truth["line_components"])
    for sweep, info in zip(rec.sweeps, truth["sweeps"], strict=True):
        out = notch(sweep.data[0], fs, 60.0, harmonics=len(truth["line_components"]))
        mid = slice(out.size // 4, 3 * out.size // 4)  # away from the notch's edge ringing
        residual = out[mid] - clean[mid]
        # What is left is the added white noise (when present), not line noise.
        assert np.max(np.abs(residual)) < 0.02 * line_peak + 5 * info["noise_sigma"]
