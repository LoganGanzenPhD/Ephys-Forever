"""Statistics of a trace between two cursors.

All values are in the units of the input (SI) and all times are seconds from the start
of the sweep. Amplitudes, area, rise time and half-width are taken relative to a
baseline level that the caller supplies, usually from
:func:`ephys_forever.core.analysis.baseline.baseline_level`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ephys_forever.core.analysis.baseline import BaselineMethod, baseline_level
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.model import FloatArray, Recording

Polarity = Literal["negative", "positive", "absolute"]


@dataclass(frozen=True, slots=True)
class RegionStats:
    """Measurements of one channel of one sweep over one region.

    Attributes
    ----------
    n_samples : int
        Samples in the region.
    start_time, stop_time : float
        Time of the first and of the last sample in the region, s.
    baseline : float
        Baseline level the relative values below are measured from.
    mean, sd : float
        Mean and sample standard deviation (``ddof=1``; NaN for one sample).
    minimum, maximum : float
        Most negative and most positive sample.
    min_time, max_time : float
        Time of the first sample equal to ``minimum`` and to ``maximum``.
    peak, peak_time : float
        The sample furthest from the baseline in the requested direction, and its time.
    peak_amplitude : float
        ``peak - baseline``. Keeps its sign: an inward current gives a negative value.
    mean_amplitude : float
        ``mean - baseline``.
    area : float
        ``sum(y - baseline) / sample_rate`` over the region, in units times seconds.
    rise_low_time, rise_high_time : float
        Interpolated times at which the trace, approaching the peak, last crosses the
        low and high fractions of ``peak_amplitude``. NaN when the trace is already past
        the level at the region start.
    rise_time : float
        ``rise_high_time - rise_low_time``.
    half_width : float
        Time between the crossings of half the peak amplitude before and after the peak.
        NaN when the trace does not return below half amplitude inside the region.
    """

    n_samples: int
    start_time: float
    stop_time: float
    baseline: float
    mean: float
    sd: float
    minimum: float
    maximum: float
    min_time: float
    max_time: float
    peak: float
    peak_time: float
    peak_amplitude: float
    mean_amplitude: float
    area: float
    rise_low_time: float
    rise_high_time: float
    rise_time: float
    half_width: float


def _crossing_before(d: FloatArray, p: int, level: float) -> float:
    """Fractional index where ``d`` last rises through ``level`` before index ``p``."""
    below = np.flatnonzero(d[: p + 1] < level)
    if below.size == 0:
        return math.nan
    k = int(below[-1])
    return float(k + (level - d[k]) / (d[k + 1] - d[k]))


def _crossing_after(d: FloatArray, p: int, level: float) -> float:
    """Fractional index where ``d`` first falls through ``level`` after index ``p``."""
    below = np.flatnonzero(d[p:] < level)
    if below.size == 0:
        return math.nan
    m = p + int(below[0])
    return float(m - 1 + (d[m - 1] - level) / (d[m - 1] - d[m]))


def measure_region(
    y: FloatArray,
    sample_rate: float,
    region: Region,
    *,
    polarity: Polarity,
    baseline: float = 0.0,
    rise_fractions: tuple[float, float] = (0.1, 0.9),
) -> RegionStats:
    """Measure one channel of one sweep between two cursors.

    Parameters
    ----------
    y : numpy.ndarray
        One channel of one sweep, 1-D, SI units.
    sample_rate : float
        Samples per second.
    region : Region
        Cursor region, seconds from the sweep start; samples on either cursor count.
    polarity : {"negative", "positive", "absolute"}
        Direction of the peak: the most negative sample, the most positive sample, or
        the sample furthest from ``baseline`` either way. Ties go to the earliest sample.
    baseline : float, optional
        Level that amplitudes, area, rise time and half-width are measured from, in the
        units of ``y``. Default 0, which measures from zero.
    rise_fractions : (float, float), optional
        Low and high fractions of the peak amplitude for the rise time. Default
        ``(0.1, 0.9)``, the 10-90 % rise time.

    Returns
    -------
    RegionStats

    Notes
    -----
    Rise time and half-width use the trace measured towards the peak,
    ``d = (y - baseline) * sign(peak_amplitude)``. Walking back from the peak, the low and
    high levels are placed where ``d`` last rises through them; walking forward from the
    peak, the second half-amplitude point is where ``d`` first falls below half. Each
    crossing time is linearly interpolated between the two samples on either side, so it
    is not limited to the sample grid. Searching back from the peak, rather than forward
    from the region start, keeps baseline noise from triggering an early crossing.

    The area is a rectangle sum (each sample stands for ``1 / sample_rate`` seconds), not
    a trapezoid.

    Raises
    ------
    RegionError
        If the region is outside the sweep or contains no sample.
    ValueError
        If ``polarity`` or ``rise_fractions`` is invalid.
    """
    low, high = rise_fractions
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"rise fractions must satisfy 0 < low < high < 1, got {rise_fractions}")
    i0, i1 = region.indices(sample_rate, y.size)
    w = y[i0:i1]
    rel = w - baseline
    if polarity == "negative":
        p = int(np.argmin(w))
    elif polarity == "positive":
        p = int(np.argmax(w))
    elif polarity == "absolute":
        p = int(np.argmax(np.abs(rel)))
    else:
        raise ValueError(f"polarity must be 'negative', 'positive' or 'absolute', got {polarity!r}")
    amplitude = float(rel[p])

    rise_low = rise_high = half_width = math.nan
    if amplitude != 0.0:
        d = rel * math.copysign(1.0, amplitude)
        size = abs(amplitude)
        rise_low = _crossing_before(d, p, low * size)
        rise_high = _crossing_before(d, p, high * size)
        half_width = _crossing_after(d, p, 0.5 * size) - _crossing_before(d, p, 0.5 * size)

    def at(index: float) -> float:
        return (i0 + index) / sample_rate

    mean = float(np.mean(w))
    return RegionStats(
        n_samples=int(w.size),
        start_time=i0 / sample_rate,
        stop_time=(i1 - 1) / sample_rate,
        baseline=float(baseline),
        mean=mean,
        sd=float(np.std(w, ddof=1)) if w.size > 1 else math.nan,
        minimum=float(np.min(w)),
        maximum=float(np.max(w)),
        min_time=at(int(np.argmin(w))),
        max_time=at(int(np.argmax(w))),
        peak=float(w[p]),
        peak_time=at(p),
        peak_amplitude=amplitude,
        mean_amplitude=mean - baseline,
        area=float(np.sum(rel)) / sample_rate,
        rise_low_time=at(rise_low),
        rise_high_time=at(rise_high),
        rise_time=(rise_high - rise_low) / sample_rate,
        half_width=half_width / sample_rate,
    )


def measure_sweeps(
    recording: Recording,
    channel: int,
    region: Region,
    *,
    polarity: Polarity,
    baseline_region: Region | None = None,
    baseline_method: BaselineMethod = "mean",
    rise_fractions: tuple[float, float] = (0.1, 0.9),
) -> list[RegionStats]:
    """Measure ``region`` of one channel in every sweep of ``recording``.

    Parameters
    ----------
    recording : Recording
        Recording to measure.
    channel : int
        Channel index.
    region : Region
        Cursor region, seconds from each sweep's start.
    polarity : {"negative", "positive", "absolute"}
        See :func:`measure_region`.
    baseline_region : Region or None, optional
        Window for each sweep's own baseline (see
        :func:`~ephys_forever.core.analysis.baseline.baseline_level`). ``None``, the
        default, measures from zero.
    baseline_method : {"mean", "median"}, optional
        How the baseline window is reduced to one level. Default ``"mean"``.
    rise_fractions : (float, float), optional
        See :func:`measure_region`.

    Returns
    -------
    list of RegionStats
        One entry per sweep, in sweep order.

    Raises
    ------
    IndexError
        If ``channel`` is out of range.
    """
    if not 0 <= channel < recording.n_channels:
        raise IndexError(f"channel {channel} out of range 0..{recording.n_channels - 1}")
    results = []
    for sweep in recording.sweeps:
        y = sweep.data[channel]
        base = (
            0.0
            if baseline_region is None
            else baseline_level(y, sweep.sample_rate, baseline_region, baseline_method)
        )
        results.append(
            measure_region(
                y,
                sweep.sample_rate,
                region,
                polarity=polarity,
                baseline=base,
                rise_fractions=rise_fractions,
            )
        )
    return results
