"""Cursor-region statistics, regions and baseline on shapes with exact answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ephys_forever.core.analysis.baseline import baseline_level, subtract_baseline
from ephys_forever.core.analysis.measure import measure_region, measure_sweeps
from ephys_forever.core.analysis.regions import Region, RegionError
from ephys_forever.core.model import Channel, Recording, Sweep

FS = 1000.0


def triangle(rise: int = 100, fall: int = 100, onset: int = 100, n: int = 600) -> np.ndarray:
    """Zero, a linear rise to 1 over ``rise`` samples, a linear fall over ``fall``."""
    y = np.zeros(n)
    y[onset : onset + rise + 1] = np.linspace(0.0, 1.0, rise + 1)
    y[onset + rise : onset + rise + fall + 1] = np.linspace(1.0, 0.0, fall + 1)
    return y


# -- regions --------------------------------------------------------------------------
def test_region_includes_samples_on_both_cursors_despite_float_products() -> None:
    assert Region(0.1, 0.3).indices(20_000.0, 10_000) == (2000, 6001)
    assert Region(0.10465, 0.15465).indices(20_000.0, 6000) == (2093, 3094)
    assert Region(0.0, 0.0005).indices(20_000.0, 10_000) == (0, 11)
    assert Region(0.00001, 0.0001).indices(20_000.0, 10_000) == (1, 3)
    assert Region(0.0, 0.49995).indices(20_000.0, 10_000) == (0, 10_000)


@pytest.mark.parametrize(
    ("start", "stop", "message"),
    [
        (0.2, 0.1, "precedes"),
        (0.1, 0.1, "precedes"),
        (-0.01, 0.1, "before the sweep"),
        (0.1, 0.6, "after the sweep's last sample"),
        (0.1001, 0.1002, "contains no sample"),
        (math.nan, 0.1, "finite"),
    ],
)
def test_bad_regions_raise_with_a_reason(start: float, stop: float, message: str) -> None:
    with pytest.raises(RegionError, match=message):
        Region(start, stop).indices(FS, 600)


# -- baseline -------------------------------------------------------------------------
def test_baseline_mean_and_median() -> None:
    y = np.array([1.0, 2.0, 3.0, 100.0, 5.0, 5.0])
    assert baseline_level(y, FS, Region(0.0, 0.003)) == 26.5
    assert baseline_level(y, FS, Region(0.0, 0.003), "median") == 2.5
    np.testing.assert_array_equal(subtract_baseline(y, FS, Region(0.004, 0.005)), y - 5.0)
    with pytest.raises(ValueError, match="'mean' or 'median'"):
        baseline_level(y, FS, Region(0.0, 0.003), "mode")  # type: ignore[arg-type]


# -- measurements ---------------------------------------------------------------------
def test_triangle_rise_time_half_width_area_and_peak() -> None:
    stats = measure_region(triangle(), FS, Region(0.0, 0.599), polarity="positive")
    assert stats.peak == 1.0
    assert stats.peak_time == pytest.approx(0.2)
    assert stats.rise_low_time == pytest.approx(0.11)
    assert stats.rise_high_time == pytest.approx(0.19)
    assert stats.rise_time == pytest.approx(0.08)
    assert stats.half_width == pytest.approx(0.1)
    assert stats.area == pytest.approx(0.1)  # 100 samples of mean height 1 / FS
    assert (stats.n_samples, stats.start_time, stats.stop_time) == (600, 0.0, 0.599)


def test_inward_peak_keeps_its_sign_and_mirrors_the_outward_one() -> None:
    base = -20e-12
    y = base - 100e-12 * triangle(rise=40, fall=160)
    up = measure_region(-y, FS, Region(0.05, 0.5), polarity="positive", baseline=-base)
    down = measure_region(y, FS, Region(0.05, 0.5), polarity="negative", baseline=base)
    assert down.peak_amplitude == pytest.approx(-100e-12)
    assert down.peak == pytest.approx(-120e-12)
    assert down.mean_amplitude < 0 and down.area < 0
    for name in ("peak_time", "rise_time", "half_width", "rise_low_time"):
        assert getattr(down, name) == pytest.approx(getattr(up, name))
    assert down.rise_time == pytest.approx(0.032)
    assert down.half_width == pytest.approx(0.1)


def test_absolute_polarity_picks_the_larger_excursion() -> None:
    y = np.zeros(100)
    y[10], y[50] = 3.0, -5.0
    stats = measure_region(y, FS, Region(0.0, 0.099), polarity="absolute", baseline=0.0)
    assert (stats.peak, stats.peak_time) == (-5.0, 0.05)
    assert (stats.maximum, stats.max_time, stats.minimum, stats.min_time) == (3.0, 0.01, -5.0, 0.05)


def test_mean_sd_and_area_of_a_constant_offset() -> None:
    y = np.full(200, 2.0)
    stats = measure_region(y, FS, Region(0.05, 0.149), polarity="positive", baseline=0.5)
    assert (stats.mean, stats.sd, stats.mean_amplitude) == (2.0, 0.0, 1.5)
    assert stats.area == pytest.approx(1.5 * 100 / FS)


def test_crossings_outside_the_region_are_nan() -> None:
    y = triangle()
    # Region starts after the 10 % and 90 % points and ends before the trace falls.
    stats = measure_region(y, FS, Region(0.195, 0.25), polarity="positive")
    assert math.isnan(stats.rise_low_time) and math.isnan(stats.rise_time)
    assert math.isnan(stats.half_width)
    flat = measure_region(np.zeros(50), FS, Region(0.0, 0.049), polarity="positive")
    assert math.isnan(flat.rise_time) and flat.peak_amplitude == 0.0


def test_single_sample_region_has_nan_sd() -> None:
    stats = measure_region(triangle(), FS, Region(0.2, 0.2005), polarity="positive")
    assert stats.n_samples == 1 and math.isnan(stats.sd)


def test_invalid_options_raise() -> None:
    with pytest.raises(ValueError, match="polarity"):
        measure_region(triangle(), FS, Region(0.0, 0.1), polarity="up")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rise fractions"):
        measure_region(
            triangle(), FS, Region(0.0, 0.1), polarity="positive", rise_fractions=(0.9, 0.1)
        )


@settings(max_examples=50, deadline=None)
@given(offset=st.floats(-1e-9, 1e-9), scale=st.floats(1e-12, 1e-9))
def test_relative_measurements_ignore_a_baseline_offset(offset: float, scale: float) -> None:
    y = triangle(rise=37, fall=81) * scale
    ref = measure_region(y, FS, Region(0.0, 0.599), polarity="positive")
    moved = measure_region(y + offset, FS, Region(0.0, 0.599), polarity="positive", baseline=offset)
    assert moved.peak_amplitude == pytest.approx(ref.peak_amplitude, rel=1e-6)
    assert moved.rise_time == pytest.approx(ref.rise_time, rel=1e-6)
    assert moved.half_width == pytest.approx(ref.half_width, rel=1e-6)


def test_measure_sweeps_uses_each_sweeps_own_baseline() -> None:
    sweeps = tuple(
        Sweep(index=k, data=(triangle() * (k + 1) + k)[None, :], sample_rate=FS, t0=0.6 * k)
        for k in range(3)
    )
    rec = Recording(channels=(Channel(name="IN0", unit="A", kind="current"),), sweeps=sweeps)
    stats = measure_sweeps(
        rec, 0, Region(0.0, 0.599), polarity="positive", baseline_region=Region(0.0, 0.05)
    )
    assert [s.baseline for s in stats] == [0.0, 1.0, 2.0]
    assert [s.peak_amplitude for s in stats] == pytest.approx([1.0, 2.0, 3.0])
    with pytest.raises(IndexError, match="channel 1"):
        measure_sweeps(rec, 1, Region(0.0, 0.599), polarity="positive")
