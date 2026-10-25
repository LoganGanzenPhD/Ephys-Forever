"""Cursor-region statistics against the analytic truth of ``step_known_amplitude``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ephys_forever.core.analysis.measure import measure_sweeps
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.io.atf import read_atf

pytestmark = pytest.mark.reference


@pytest.fixture(scope="module")
def truth(synthetic_reference_dir: Path) -> dict[str, Any]:
    with (synthetic_reference_dir / "step_known_amplitude.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def test_step_baseline_mean_peak_and_area(
    truth: dict[str, Any], synthetic_data_dir: Path, tolerances: dict[str, Any]
) -> None:
    rtol = tolerances["measurements"]["vs_synthetic"]["rtol"]
    rec = read_atf(synthetic_data_dir / truth["file"])
    signal = truth["signal"]
    dt = 1.0 / rec.sample_rate  # truth windows are [start, end); regions include both ends
    stats = measure_sweeps(
        rec,
        rec.channel_index(truth["channel"]["name"]),
        Region(signal["step_start_s"], signal["step_end_s"] - dt),
        polarity="negative",
        baseline_region=Region(0.0, signal["step_start_s"] - dt),
    )
    i_on, i_off = signal["step_samples"]
    assert len(stats) == truth["n_sweeps"]
    for s, expect in zip(stats, truth["realized_per_sweep"], strict=True):
        assert s.n_samples == i_off - i_on
        assert s.baseline == pytest.approx(expect["baseline_mean"], rel=rtol)
        assert s.mean == pytest.approx(expect["step_mean"], rel=rtol)
        assert s.peak == pytest.approx(expect["step_min"], rel=rtol)
        assert s.area == pytest.approx(expect["step_area_vs_baseline_mean"], rel=rtol)
        # Noise (sigma 2 pA over 2000 samples) moves the mean by well under 1 %.
        assert s.mean_amplitude == pytest.approx(signal["step_amplitude"], rel=0.01)
