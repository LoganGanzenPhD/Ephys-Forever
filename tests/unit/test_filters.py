"""Filters: cutoff definitions, edge behaviour and argument checks on exact signals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal

from ephys_forever.core.analysis.filters import (
    GAUSSIAN_SIGMA_FACTOR,
    bessel_lowpass,
    boxcar,
    gaussian_lowpass,
    gaussian_rise_time,
    notch,
)

FS = 50_000.0


def sine(freq: float, n: int = 50_000, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / FS)


def gain(filtered: np.ndarray, freq: float) -> float:
    """Amplitude at ``freq`` in the middle half of a filtered unit sine, by projection."""
    n = filtered.size
    t = np.arange(n) / FS
    mid = slice(n // 4, 3 * n // 4)
    basis = np.vstack([np.sin(2 * np.pi * freq * t[mid]), np.cos(2 * np.pi * freq * t[mid])])
    coef, *_ = np.linalg.lstsq(basis.T, filtered[mid], rcond=None)
    return float(np.hypot(*coef))


def crossing(v: np.ndarray, level: float) -> float:
    i = int(np.argmax(v >= level))
    return i - 1 + (level - v[i - 1]) / (v[i] - v[i - 1])


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("cutoff", [100.0, 1000.0, 5000.0])
def test_bessel_is_minus_3_db_at_the_cutoff(order: int, cutoff: float) -> None:
    out = bessel_lowpass(sine(cutoff), FS, cutoff, order)
    assert gain(out, cutoff) == pytest.approx(1 / math.sqrt(2), rel=1e-3)
    assert gain(bessel_lowpass(sine(cutoff / 20), FS, cutoff, order), cutoff / 20) == (
        pytest.approx(1.0, rel=1e-3)
    )


def test_zero_phase_bessel_squares_the_response_and_has_no_delay() -> None:
    out = bessel_lowpass(sine(1000.0), FS, 1000.0, 8, zero_phase=True)
    assert gain(out, 1000.0) == pytest.approx(0.5, rel=1e-3)
    step = np.r_[np.zeros(500), np.ones(500)]
    sym = bessel_lowpass(step, FS, 1000.0, 8, zero_phase=True)
    assert crossing(sym, 0.5) == pytest.approx(499.5, abs=0.05)


def test_bessel_starts_settled_at_the_first_value() -> None:
    flat = np.full(1000, -60e-12)
    np.testing.assert_allclose(bessel_lowpass(flat, FS, 1000.0), flat, rtol=1e-9)


@pytest.mark.parametrize("order", [1, 3, 8])
def test_matched_z_bessel_maps_the_analog_poles_with_unit_dc_gain(order: int) -> None:
    """Impulse response of exp(s_k T) all-pole sections; settles at 1 for a unit step."""
    cutoff = 1000.0
    _, poles, _ = signal.besselap(order, norm="mag")
    a = np.real(np.poly(np.exp(poles * 2 * np.pi * cutoff / FS)))
    impulse = np.r_[1.0, np.zeros(399)]
    expected = signal.lfilter([a.sum()], a, impulse)
    out = bessel_lowpass(np.r_[np.zeros(100), impulse], FS, cutoff, order, design="matched_z")
    np.testing.assert_allclose(out[100:], expected, atol=1e-10)  # direct-form rounding
    step = bessel_lowpass(np.r_[np.zeros(10), np.ones(2000)], FS, cutoff, order, design="matched_z")
    assert step[-1] == pytest.approx(1.0, rel=1e-12)


def test_matched_z_bessel_is_close_to_but_above_the_cutoff() -> None:
    out = bessel_lowpass(sine(1000.0), FS, 1000.0, 8, design="matched_z")
    assert 1 / math.sqrt(2) < gain(out, 1000.0) < 0.75


def test_gaussian_is_minus_3_db_at_the_cutoff_with_the_documented_rise_time() -> None:
    assert gain(gaussian_lowpass(sine(500.0), FS, 500.0), 500.0) == pytest.approx(
        1 / math.sqrt(2), rel=2e-3
    )
    step = np.r_[np.zeros(2000), np.ones(2000)]
    out = gaussian_lowpass(step, FS, 1000.0)
    rise = (crossing(out, 0.9) - crossing(out, 0.1)) / FS
    assert rise == pytest.approx(gaussian_rise_time(1000.0), rel=1e-2)
    assert gaussian_rise_time(1000.0) == pytest.approx(0.3399e-3, rel=1e-3)
    assert out.max() <= 1.0 + 1e-12 and out.min() >= -1e-12  # no overshoot
    assert GAUSSIAN_SIGMA_FACTOR == 0.1325


def test_notch_removes_the_line_and_its_harmonics_and_keeps_the_rest() -> None:
    n = 400_000  # 8 s: the middle half is well clear of the notch's edge ringing
    line = sine(60.0, n) + 0.4 * sine(120.0, n) + 0.2 * sine(180.0, n)
    signal_ = sine(3.0, n) + sine(500.0, n)
    out = notch(signal_ + line, FS, 60.0, harmonics=3)
    for f in (60.0, 120.0, 180.0):
        assert gain(out, f) < 1e-3
    for f in (3.0, 500.0):
        assert gain(out, f) == pytest.approx(1.0, rel=1e-3)


def test_notch_skips_harmonics_above_nyquist() -> None:
    out = notch(sine(60.0), 200.0, 60.0, harmonics=5)  # only 60 Hz is below 100 Hz
    assert out.shape == (50_000,)


def test_boxcar_is_a_centred_moving_average_with_edge_padding() -> None:
    y = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 6.0])
    np.testing.assert_allclose(boxcar(y, 3), [0.0, 1.0, 1.0, 1.0, 2.0, 4.0])
    np.testing.assert_array_equal(boxcar(y, 1), y)
    _, h = signal.freqz(np.ones(9) / 9, worN=[0.443 * FS / 9], fs=FS)
    assert abs(h[0]) == pytest.approx(1 / math.sqrt(2), rel=1e-2)


def test_filters_do_not_modify_their_input() -> None:
    y = sine(100.0, 2000)
    before = y.copy()
    for out in (
        bessel_lowpass(y, FS, 1000.0),
        gaussian_lowpass(y, FS, 1000.0),
        notch(y, FS),
        boxcar(y, 5),
    ):
        assert out.dtype == np.float64 and out.shape == y.shape
    np.testing.assert_array_equal(y, before)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: bessel_lowpass(np.zeros(10), FS, 30_000.0), "Nyquist"),
        (lambda: bessel_lowpass(np.zeros(10), FS, 0.0), "above 0"),
        (lambda: bessel_lowpass(np.zeros(10), FS, 1000.0, order=0), "order"),
        (lambda: bessel_lowpass(np.zeros((2, 10)), FS, 1000.0), "1-D"),
        (lambda: bessel_lowpass(np.zeros(10), FS, 1000.0, design="analog"), "design"),
        (lambda: gaussian_lowpass(np.zeros(0), FS, 1000.0), "empty"),
        (lambda: notch(np.zeros(10), FS, -60.0), "positive"),
        (lambda: notch(np.zeros(10), 100.0, 60.0), "Nyquist"),
        (lambda: notch(np.zeros(10), FS, harmonics=0), "harmonics"),
        (lambda: boxcar(np.zeros(10), 4), "odd"),
    ],
)
def test_bad_arguments_raise(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


def test_filter_recording_filters_one_channel_per_sweep_and_records_it() -> None:
    from ephys_forever.core.analysis.filters import filter_recording
    from ephys_forever.core.model import Channel, Recording, Sweep

    rng = np.random.default_rng(0)
    sweeps = tuple(
        Sweep(index=k, data=rng.normal(size=(2, 1000)), sample_rate=FS) for k in range(2)
    )
    rec = Recording(
        channels=(
            Channel(name="I", unit="A", kind="current"),
            Channel(name="V", unit="V", kind="voltage"),
        ),
        sweeps=sweeps,
        metadata={"sha256": "x"},
    )
    once = filter_recording(rec, 0, "boxcar", points=5)
    twice = filter_recording(once, 0, "bessel_lowpass", cutoff_hz=1000.0, order=4)
    for k in range(2):
        np.testing.assert_array_equal(once.sweeps[k].data[0], boxcar(sweeps[k].data[0], 5))
        np.testing.assert_array_equal(twice.sweeps[k].data[1], sweeps[k].data[1])
    assert [f["name"] for f in twice.metadata["filters"]] == ["boxcar", "bessel_lowpass"]
    assert twice.metadata["filters"][1]["params"] == {"cutoff_hz": 1000.0, "order": 4}
    assert twice.metadata["sha256"] == "x" and "filters" not in rec.metadata
    with pytest.raises(KeyError, match="no filter 'median'"):
        filter_recording(rec, 0, "median")
    with pytest.raises(IndexError, match="channel 2"):
        filter_recording(rec, 2, "boxcar", points=3)
