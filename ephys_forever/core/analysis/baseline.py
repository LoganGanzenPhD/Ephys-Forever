"""Baseline level of a trace and baseline subtraction."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.model import FloatArray

BaselineMethod = Literal["mean", "median"]


def baseline_level(
    y: FloatArray,
    sample_rate: float,
    region: Region,
    method: BaselineMethod = "mean",
) -> float:
    """Baseline level of ``y`` over ``region``.

    Parameters
    ----------
    y : numpy.ndarray
        One channel of one sweep, 1-D, SI units.
    sample_rate : float
        Samples per second.
    region : Region
        Baseline window, seconds from the sweep start. The caller chooses it; there is
        no default window.
    method : {"mean", "median"}
        ``"mean"`` is the arithmetic mean of the samples in the window. ``"median"`` is
        less affected by events or artefacts inside the window.

    Returns
    -------
    float
        Baseline level in the units of ``y``.

    Raises
    ------
    RegionError
        If the window lies outside the sweep or contains no sample.
    ValueError
        If ``method`` is not recognised.
    """
    i0, i1 = region.indices(sample_rate, y.size)
    window = y[i0:i1]
    if method == "mean":
        return float(np.mean(window))
    if method == "median":
        return float(np.median(window))
    raise ValueError(f"baseline method must be 'mean' or 'median', got {method!r}")


def subtract_baseline(
    y: FloatArray,
    sample_rate: float,
    region: Region,
    method: BaselineMethod = "mean",
) -> FloatArray:
    """Return ``y`` minus its baseline level over ``region`` (a new array).

    See :func:`baseline_level` for the parameters.
    """
    return np.asarray(y - baseline_level(y, sample_rate, region, method), dtype=np.float64)
