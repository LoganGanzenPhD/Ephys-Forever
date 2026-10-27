"""8-pole Bessel on the IV steps recording against the commercial tool's filtered exports."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ephys_forever.core.analysis.filters import bessel_lowpass
from ephys_forever.core.io.abf2 import read_abf2
from ephys_forever.core.io.atf import read_atf
from ephys_forever.core.model import Recording

pytestmark = [pytest.mark.reference, pytest.mark.realdata]

CLAMPFIT = Path(__file__).resolve().parents[1] / "reference" / "clampfit"
CASES = [
    "vclamp_iv_steps_001_bessel",
    "vclamp_iv_steps_001_bessel_1000hz",
    "vclamp_iv_steps_001_bessel_2000hz",
]


@pytest.fixture(scope="module", params=CASES)
def settings(request: pytest.FixtureRequest) -> dict[str, Any]:
    with (CLAMPFIT / request.param / "settings.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


@pytest.fixture(scope="module")
def recordings(
    settings: dict[str, Any], real_data_file: Callable[[str], Path]
) -> tuple[Recording, Recording]:
    raw = read_abf2(real_data_file(settings["source_file"]))
    return raw, read_atf(real_data_file(settings["output_file"]))


def filtered(raw: Recording, settings: dict[str, Any], cutoff_hz: float) -> list[np.ndarray]:
    ch = raw.channel_index(settings["channel"])
    return [
        bessel_lowpass(
            s.data[ch],
            raw.sample_rate,
            cutoff_hz,
            settings["filter"]["poles"],
            design=settings["reproduced_by"]["design"],
        )
        for s in raw.sweeps
    ]


def worst_error(ours: list[np.ndarray], ref: Recording, ch: int, start: int) -> float:
    """Largest |ours - ref| after ``start``, as a fraction of each sweep's largest |ref|."""
    return max(
        float(np.max(np.abs(y - s.data[ch])[start:]) / np.max(np.abs(s.data[ch])))
        for y, s in zip(ours, ref.sweeps, strict=True)
    )


def test_only_the_filtered_channel_changed(
    settings: dict[str, Any], recordings: tuple[Recording, Recording]
) -> None:
    raw, ref = recordings
    assert ref.n_sweeps == raw.n_sweeps and ref.sample_rate == raw.sample_rate
    ch = raw.channel_index(settings["channel"])
    k = raw.channel_index("IN 0")
    for a, b in zip(raw.sweeps, ref.sweeps, strict=True):
        np.testing.assert_allclose(b.data[k], a.data[k], rtol=5e-6, atol=1e-16)
        assert np.max(np.abs(b.data[ch] - a.data[ch])) > 1e-12


def test_matched_z_design_reproduces_the_export(
    settings: dict[str, Any], recordings: tuple[Recording, Recording], tolerances: dict[str, Any]
) -> None:
    raw, ref = recordings
    spec = settings["reproduced_by"]
    cutoff = settings["filter"]["cutoff_hz"] * spec["cutoff_scale"]
    ch = ref.channel_index(settings["channel"])
    error = worst_error(filtered(raw, settings, cutoff), ref, ch, spec["start_samples_excluded"])
    assert error <= tolerances["filters"]["bessel_vs_clampfit"]["rtol"]


def test_nominal_cutoff_differs_by_the_documented_amount(
    settings: dict[str, Any], recordings: tuple[Recording, Recording]
) -> None:
    """At the cutoff as entered the design is right but not exact (docs/differences.md)."""
    raw, ref = recordings
    ch = ref.channel_index(settings["channel"])
    nominal = filtered(raw, settings, settings["filter"]["cutoff_hz"])
    start = settings["reproduced_by"]["start_samples_excluded"]
    assert 1e-3 < worst_error(nominal, ref, ch, start) < 1e-2
