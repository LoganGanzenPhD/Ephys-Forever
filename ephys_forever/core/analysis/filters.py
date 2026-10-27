"""Digital filters for sampled traces: Bessel and Gaussian low-pass, notch, boxcar.

Every filter takes one 1-D trace in SI units and returns a new float64 array of the
same length; the input is never modified. Cutoff frequencies are in hertz and follow
the -3 dB convention unless a docstring says otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
from scipy import ndimage, signal
from scipy.special import erfinv

from ephys_forever.core.model import FloatArray, Recording

GAUSSIAN_SIGMA_FACTOR = 0.1325
"""``sigma_t = 0.1325 / fc``: the Gaussian whose response is -3 dB at ``fc``
(Colquhoun & Sigworth 1995, in Single-Channel Recording, 2nd ed., ch. 19)."""
GAUSSIAN_TRUNCATE = 4.0
"""Gaussian kernel half-width in standard deviations."""


def _check_trace(y: FloatArray) -> FloatArray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"filters take a 1-D trace, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("cannot filter an empty trace")
    return arr


def _check_cutoff(cutoff_hz: float, sample_rate: float) -> None:
    nyquist = sample_rate / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"cutoff {cutoff_hz:g} Hz must be above 0 and below the Nyquist frequency "
            f"{nyquist:g} Hz (half the {sample_rate:g} Hz sample rate)"
        )


def bessel_lowpass(
    y: FloatArray,
    sample_rate: float,
    cutoff_hz: float,
    order: int = 8,
    *,
    zero_phase: bool = False,
    design: str = "bilinear",
) -> FloatArray:
    """Low-pass Bessel filter.

    Parameters
    ----------
    y : numpy.ndarray
        Trace, 1-D.
    sample_rate : float
        Samples per second.
    cutoff_hz : float
        Frequency where the magnitude response is -3 dB. Must be below half the
        sample rate.
    order : int, optional
        Number of poles. Default 8, the usual choice for single-channel and synaptic
        recordings.
    zero_phase : bool, optional
        ``False`` (default) filters forward once, like an analog filter: the output is
        delayed and the -3 dB point is at ``cutoff_hz``. ``True`` filters forward and
        backward, which removes the delay but squares the magnitude response, so the
        -3 dB point moves below ``cutoff_hz``.
    design : {"bilinear", "matched_z"}, optional
        How the analog prototype becomes a digital filter. ``"bilinear"`` (default)
        keeps the -3 dB point exactly at ``cutoff_hz``. ``"matched_z"`` maps each analog
        pole ``s`` to ``exp(s / sample_rate)`` with no zeros and unit gain at 0 Hz;
        this is the design that reproduces the 8-pole Bessel of the commercial
        reference tool (see ``docs/differences.md``). Its -3 dB point lies a few percent
        above ``cutoff_hz`` (about 4.5 % at ``sample_rate / 20``) and its attenuation
        near the Nyquist frequency is weaker.

    Returns
    -------
    numpy.ndarray
        Filtered trace, same length.

    Notes
    -----
    The digital filter is the analog Bessel prototype normalised to -3 dB at the cutoff
    (``scipy.signal.bessel(norm="mag")``), mapped with the bilinear transform prewarped
    at the cutoff, or with the matched z-transform (``design="matched_z"``). The filter
    starts in its steady state for the first sample's value, so a trace that begins at
    a constant level shows no start-up transient.

    References
    ----------
    Oppenheim, A. V. & Schafer, R. W. (2010). Discrete-Time Signal Processing, 3rd ed.,
    sec. 7.1 and 7.2 (bilinear transform; impulse invariance and pole mapping).
    """
    arr = _check_trace(y)
    _check_cutoff(cutoff_hz, sample_rate)
    if order < 1:
        raise ValueError(f"filter order must be at least 1, got {order}")
    if design == "bilinear":
        sos = signal.bessel(order, cutoff_hz, btype="low", norm="mag", output="sos", fs=sample_rate)
    elif design == "matched_z":
        sos = _matched_z_bessel(order, cutoff_hz, sample_rate)
    else:
        raise ValueError(f"unknown Bessel design {design!r}; use 'bilinear' or 'matched_z'")
    if zero_phase:
        return np.asarray(signal.sosfiltfilt(sos, arr), dtype=np.float64)
    zi = signal.sosfilt_zi(sos) * arr[0]
    out, _ = signal.sosfilt(sos, arr, zi=zi)
    return np.asarray(out, dtype=np.float64)


def _matched_z_bessel(order: int, cutoff_hz: float, sample_rate: float) -> FloatArray:
    """All-pole sections with poles ``exp(s_k / fs)``, each with unit gain at 0 Hz."""
    _, poles, _ = signal.besselap(order, norm="mag")
    z = np.exp(poles * (2.0 * math.pi * cutoff_hz / sample_rate))
    sections = []
    for pole in z[z.imag > 0]:
        a = np.array([1.0, -2.0 * pole.real, abs(pole) ** 2])
        sections.append([a.sum(), 0.0, 0.0, *a])
    for pole in z[z.imag == 0]:
        a = np.array([1.0, -pole.real, 0.0])
        sections.append([a.sum(), 0.0, 0.0, *a])
    return np.asarray(sections, dtype=np.float64)


def gaussian_lowpass(y: FloatArray, sample_rate: float, cutoff_hz: float) -> FloatArray:
    """Low-pass Gaussian filter (zero phase).

    Parameters
    ----------
    y : numpy.ndarray
        Trace, 1-D.
    sample_rate : float
        Samples per second.
    cutoff_hz : float
        -3 dB frequency. The kernel's standard deviation is
        ``0.1325 / cutoff_hz`` seconds (:data:`GAUSSIAN_SIGMA_FACTOR`).

    Returns
    -------
    numpy.ndarray
        Filtered trace, same length. Samples beyond the ends are taken equal to the
        end samples.

    Notes
    -----
    A Gaussian filter has no overshoot and a 10-90 % step rise time of
    ``0.3399 / cutoff_hz``. The kernel is truncated at ±4 standard deviations. When the
    standard deviation is shorter than about half a sample the discrete kernel can no
    longer reach the requested cutoff, and the result is close to the unfiltered trace.

    References
    ----------
    Colquhoun, D. & Sigworth, F. J. (1995). Fitting and statistical analysis of
    single-channel records. In Sakmann & Neher (eds.), Single-Channel Recording,
    2nd ed., pp. 483-587. Plenum.
    """
    arr = _check_trace(y)
    _check_cutoff(cutoff_hz, sample_rate)
    sigma = GAUSSIAN_SIGMA_FACTOR / cutoff_hz * sample_rate
    out = ndimage.gaussian_filter1d(arr, sigma, mode="nearest", truncate=GAUSSIAN_TRUNCATE)
    return np.asarray(out, dtype=np.float64)


def notch(
    y: FloatArray,
    sample_rate: float,
    frequency_hz: float = 60.0,
    *,
    harmonics: int = 1,
    quality: float = 30.0,
) -> FloatArray:
    """Remove line noise with zero-phase notch filters at a frequency and its harmonics.

    Parameters
    ----------
    y : numpy.ndarray
        Trace, 1-D.
    sample_rate : float
        Samples per second.
    frequency_hz : float, optional
        Fundamental to remove. Default 60 Hz; use 50 Hz where mains is 50 Hz.
    harmonics : int, optional
        Number of frequencies removed: ``frequency_hz * k`` for ``k = 1 .. harmonics``.
        Harmonics at or above the Nyquist frequency are skipped. Default 1.
    quality : float, optional
        Quality factor of each notch: centre frequency divided by the -3 dB width.
        Default 30 (a 2 Hz wide notch at 60 Hz).

    Returns
    -------
    numpy.ndarray
        Filtered trace, same length.

    Notes
    -----
    Each notch is a second-order IIR notch (``scipy.signal.iirnotch``) applied forward
    and backward, so there is no phase shift and the attenuation at each notch is
    squared. A notch rings for roughly ``quality / (pi * f)`` seconds after an abrupt
    change, including the start and end of the trace.
    """
    arr = _check_trace(y)
    if frequency_hz <= 0:
        raise ValueError(f"notch frequency must be positive, got {frequency_hz:g} Hz")
    if harmonics < 1:
        raise ValueError(f"harmonics must be at least 1, got {harmonics}")
    if quality <= 0:
        raise ValueError(f"quality factor must be positive, got {quality:g}")
    nyquist = sample_rate / 2.0
    freqs = [frequency_hz * k for k in range(1, harmonics + 1) if frequency_hz * k < nyquist]
    if not freqs:
        raise ValueError(f"{frequency_hz:g} Hz is at or above the Nyquist frequency {nyquist:g} Hz")
    sos = np.vstack([signal.tf2sos(*signal.iirnotch(f, quality, fs=sample_rate)) for f in freqs])
    return np.asarray(signal.sosfiltfilt(sos, arr), dtype=np.float64)


def boxcar(y: FloatArray, points: int) -> FloatArray:
    """Centred moving average over ``points`` samples.

    Parameters
    ----------
    y : numpy.ndarray
        Trace, 1-D.
    points : int
        Window length in samples, odd so the window is centred on each sample.

    Returns
    -------
    numpy.ndarray
        Smoothed trace, same length. Samples beyond the ends are taken equal to the end
        samples.

    Notes
    -----
    The -3 dB frequency of an ``N``-point boxcar is about ``0.443 * sample_rate / N``.
    """
    arr = _check_trace(y)
    if points < 1 or points % 2 == 0:
        raise ValueError(f"boxcar length must be a positive odd number, got {points}")
    half = points // 2
    padded = np.pad(arr, half, mode="edge")
    csum = np.concatenate(([0.0], np.cumsum(padded)))
    return (csum[points:] - csum[:-points]) / points


def gaussian_rise_time(cutoff_hz: float) -> float:
    """10-90 % step rise time of :func:`gaussian_lowpass` at ``cutoff_hz``, in seconds.

    ``2 * erfinv(0.8) * sqrt(2) * 0.1325 / fc``, about ``0.3399 / fc``.
    """
    return float(2.0 * math.sqrt(2.0) * erfinv(0.8) * GAUSSIAN_SIGMA_FACTOR / cutoff_hz)


FILTERS: dict[str, Callable[..., FloatArray]] = {
    "bessel_lowpass": bessel_lowpass,
    "gaussian_lowpass": gaussian_lowpass,
    "notch": notch,
    "boxcar": lambda y, sample_rate, **params: boxcar(y, **params),
}
"""Filters by name, each called as ``f(y, sample_rate, **params)``."""


def filter_recording(recording: Recording, channel: int, name: str, **params: Any) -> Recording:
    """Return a copy of ``recording`` with one channel filtered in every sweep.

    Parameters
    ----------
    recording : Recording
        Source recording; not modified.
    channel : int
        Channel index to filter. Other channels are copied unchanged.
    name : str
        Filter name, a key of :data:`FILTERS`.
    **params
        Filter parameters other than the trace and sample rate, for example
        ``cutoff_hz=1000.0, order=8``.

    Returns
    -------
    Recording
        Same channels and sweeps. Each sweep is filtered on its own, from its first
        sample. ``metadata["filters"]`` lists every filter applied so far as
        ``{"name", "channel", "params"}`` records, oldest first.

    Raises
    ------
    KeyError
        If ``name`` is not a known filter.
    IndexError
        If ``channel`` is out of range.
    """
    if name not in FILTERS:
        raise KeyError(f"no filter {name!r}; filters are {sorted(FILTERS)}")
    if not 0 <= channel < recording.n_channels:
        raise IndexError(f"channel {channel} out of range 0..{recording.n_channels - 1}")
    fn = FILTERS[name]
    sweeps = []
    for sweep in recording.sweeps:
        data = np.array(sweep.data)
        data[channel] = fn(sweep.data[channel], sweep.sample_rate, **params)
        sweeps.append(replace(sweep, data=data))
    history = [
        *recording.metadata.get("filters", ()),
        {"name": name, "channel": channel, "params": dict(params)},
    ]
    return Recording(
        channels=recording.channels,
        sweeps=tuple(sweeps),
        metadata={**recording.metadata, "filters": history},
    )
