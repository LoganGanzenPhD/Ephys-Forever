"""Min/max envelope for drawing long traces at screen resolution.

A 30-minute recording at 20 kHz has 36 million samples per channel, far more than a
screen can show. :class:`Envelope` answers each zoom or pan with at most ``max_points``
points for the visible range: the raw samples when they fit, otherwise the minimum and
maximum of each bin. Drawing both keeps every peak visible, the same picture as plotting
every sample.

Bins come from two sources, chosen for speed:

- Bins smaller than :data:`BASE_BIN` are computed directly from the visible samples.
  Such views cover at most ``BASE_BIN * max_points / 2`` samples, so this stays cheap.
- Larger bins come from a pyramid (bin sizes ``BASE_BIN * FACTOR**k``) built on first
  use and cached. Its base level costs one pass over the signal with long contiguous
  bins, which NumPy reduces near memory speed.

No Qt imports, so this is testable on its own.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

BASE_BIN = 4096
"""Bin size of the pyramid's finest level."""
FACTOR = 2
"""Bin size ratio between consecutive pyramid levels."""


def _bin_extrema(y: FloatArray, size: int) -> tuple[FloatArray, FloatArray]:
    """Min and max of consecutive ``size``-sample bins; a short last bin is kept."""
    n_full = y.size // size
    head = n_full * size
    blocks = y[:head].reshape(n_full, size)
    lo, hi = blocks.min(axis=1), blocks.max(axis=1)
    if head < y.size:
        lo = np.append(lo, y[head:].min())
        hi = np.append(hi, y[head:].max())
    return lo, hi


class Envelope:
    """Multi-resolution min/max view of a 1-D signal.

    Parameters
    ----------
    y : numpy.ndarray
        Samples, 1-D float64. Kept by reference, not copied.
    """

    def __init__(self, y: FloatArray) -> None:
        if y.ndim != 1:
            raise ValueError(f"envelope needs a 1-D signal, got shape {y.shape}")
        self.y = y
        self._levels: dict[int, tuple[FloatArray, FloatArray]] = {}
        self._extent: tuple[float, float] | None = None

    def __len__(self) -> int:
        return int(self.y.size)

    def extent(self) -> tuple[float, float]:
        """Minimum and maximum of the whole signal (NaN for an empty one)."""
        if self._extent is None:
            if self.y.size == 0:
                self._extent = (math.nan, math.nan)
            elif self.y.size <= BASE_BIN * FACTOR:
                self._extent = (float(self.y.min()), float(self.y.max()))
            else:
                lo, hi = self._level(BASE_BIN)
                self._extent = (float(lo.min()), float(hi.max()))
        return self._extent

    def _level(self, size: int) -> tuple[FloatArray, FloatArray]:
        level = self._levels.get(size)
        if level is None:
            if size == BASE_BIN:
                level = _bin_extrema(self.y, size)
            else:
                lo, hi = self._level(size // FACTOR)
                level = (_bin_extrema(lo, FACTOR)[0], _bin_extrema(hi, FACTOR)[1])
            self._levels[size] = level
        return level

    def view(self, start: float, stop: float, max_points: int) -> tuple[FloatArray, FloatArray]:
        """Points to draw for samples ``[start, stop)``.

        Parameters
        ----------
        start, stop : float
            Visible range in sample indices; clipped to the signal and widened by one
            sample each side so lines run to the plot edges.
        max_points : int
            Upper bound on returned points; at least 8.

        Returns
        -------
        x, y : numpy.ndarray
            ``x`` in sample indices, non-decreasing. When the raw samples fit, they are
            returned as is. Otherwise each bin gives two points at its centre, its
            minimum then its maximum.
        """
        if max_points < 8:
            raise ValueError(f"max_points must be at least 8, got {max_points}")
        n = self.y.size
        i0 = max(math.floor(start) - 1, 0)
        i1 = min(math.ceil(stop) + 1, n)
        if i1 <= i0:
            return np.empty(0), np.empty(0)
        if i1 - i0 <= max_points:
            return np.arange(i0, i1, dtype=np.float64), self.y[i0:i1]
        half = max_points // 2 - 2  # bins; alignment adds at most two
        needed = -(-(i1 - i0) // half)
        if needed < BASE_BIN:
            size = 1 << (needed - 1).bit_length()
            j0, j1 = i0 // size, -(-i1 // size)
            lo, hi = _bin_extrema(self.y[j0 * size : min(j1 * size, n)], size)
            first = j0
        else:
            size = BASE_BIN
            while size < needed:
                size *= FACTOR
            j0, j1 = i0 // size, -(-i1 // size)
            all_lo, all_hi = self._level(size)
            lo, hi = all_lo[j0:j1], all_hi[j0:j1]
            first = j0
        centres = np.minimum((np.arange(lo.size, dtype=np.float64) + first + 0.5) * size, n - 1)
        x = np.repeat(centres, 2)
        y = np.empty(x.size)
        y[0::2] = lo
        y[1::2] = hi
        return x, y
