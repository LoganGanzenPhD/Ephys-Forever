"""Time regions of a sweep, converted to sample index ranges.

A region is a closed time interval ``[start, stop]`` in seconds from the start of the
sweep, the same time axis the viewer shows. It covers sample ``i`` (at time
``i / sample_rate``) when ``start <= i / sample_rate <= stop``, so samples that sit
exactly on either cursor are included. This matches the commercial tool's cursor
statistics: cursors at 104.65 and 154.65 ms at 20 kHz cover 1001 samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

INDEX_GUARD = 1e-6
"""Samples of slack when turning times into indices, so ``0.1 s * 20 kHz`` gives 2000
even though the floating-point product is 2000.0000000000002."""


class RegionError(ValueError):
    """Raised for a region that is reversed, empty, or outside the sweep."""


@dataclass(frozen=True, slots=True)
class Region:
    """Closed time interval ``[start, stop]`` in seconds from the sweep start.

    Attributes
    ----------
    start, stop : float
        Region bounds in seconds (cursor 1 and cursor 2). ``stop`` must be greater
        than ``start``.
    """

    start: float
    stop: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.stop)):
            raise RegionError(f"region bounds must be finite, got {self.start}, {self.stop}")
        if self.stop <= self.start:
            raise RegionError(
                f"region end {self.stop:g} s precedes or equals its start {self.start:g} s"
            )

    def indices(self, sample_rate: float, n_samples: int) -> tuple[int, int]:
        """Sample index range covered by the region, as a slice ``[i0, i1)``.

        Parameters
        ----------
        sample_rate : float
            Samples per second.
        n_samples : int
            Samples in the sweep. The region must lie within the sweep, from 0 s to the
            time of the last sample, ``(n_samples - 1) / sample_rate``.

        Returns
        -------
        i0, i1 : int
            ``i0`` is the first sample at or after ``start``; ``i1 - 1`` is the last
            sample at or before ``stop``. Use as ``y[i0:i1]``.

        Raises
        ------
        RegionError
            If the region starts before the sweep, ends after its last sample, or
            contains no sample.
        """
        last = (n_samples - 1) / sample_rate
        i0 = math.ceil(self.start * sample_rate - INDEX_GUARD)
        i1 = math.floor(self.stop * sample_rate + INDEX_GUARD) + 1
        if i0 < 0:
            raise RegionError(f"region starts at {self.start:g} s, before the sweep starts at 0 s")
        if i1 > n_samples:
            raise RegionError(
                f"region ends at {self.stop:g} s, after the sweep's last sample at {last:g} s"
            )
        if i1 <= i0:
            raise RegionError(
                f"region [{self.start:g}, {self.stop:g}] s contains no sample at {sample_rate:g} Hz"
            )
        return i0, i1
