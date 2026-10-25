"""Recording, Sweep, Channel and Epoch: the data model every reader produces.

Conventions
-----------
- ``Sweep.data`` is a read-only float64 array shaped ``(n_channels, n_samples)`` in SI
  units. Readers convert once, at load time, and keep the raw scaling in metadata.
- Time is never stored per sample. A sample ``i`` of a sweep is at ``i / sample_rate``
  seconds from the sweep start, and the sweep starts ``t0`` seconds after the first
  sweep of the recording.
- Every channel has an SI ``unit`` and a ``kind`` derived from it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ephys_forever.core.units import SI_BASE_UNITS, ChannelKind, channel_kind

FloatArray = NDArray[np.float64]


def _frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class Channel:
    """One recorded signal.

    Attributes
    ----------
    name : str
        Name as stored in the file, for example ``"IN 0"``.
    unit : str
        SI symbol of the values in ``Sweep.data`` (``"A"``, ``"V"``, ...), or the file's own
        unit text when it is not an SI unit (then ``kind`` is ``"other"`` and values are
        left unscaled; see the reader's metadata).
    kind : {"current", "voltage", "other"}
        What the channel measures. ``"current"`` requires unit ``A`` and ``"voltage"``
        requires ``V``; :func:`ephys_forever.core.units.channel_kind` gives the default.
    metadata : Mapping[str, Any]
        Reader-specific details such as raw scaling factors and amplifier telegraphs.
    """

    name: str
    unit: str
    kind: ChannelKind
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind != "other" and self.kind != channel_kind(self.unit):
            raise ValueError(
                f"channel {self.name!r}: kind {self.kind!r} does not match unit {self.unit!r}"
            )
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    @property
    def is_si(self) -> bool:
        """True when ``unit`` is an SI base unit."""
        return self.unit in SI_BASE_UNITS


@dataclass(frozen=True, slots=True)
class Epoch:
    """One segment of a command waveform within a sweep.

    Attributes
    ----------
    label : str
        Epoch name, ``"A"``, ``"B"``, ... in protocol order.
    kind : str
        Waveform type: ``"step"``, ``"ramp"``, ``"pulse"``, ``"triangle"``, ``"cosine"``
        or ``"biphasic"``.
    start, stop : int
        Sample indices in the sweep, ``start`` inclusive and ``stop`` exclusive.
    level : float
        Command level for this sweep in SI units. For a ramp, the level reached at the
        end of the epoch.
    unit : str
        SI unit of ``level``.
    output : str
        Name of the command output that plays this epoch, for example ``"Cmd 0"``.
    """

    label: str
    kind: str
    start: int
    stop: int
    level: float
    unit: str
    output: str

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.stop:
            raise ValueError(
                f"epoch {self.label}: need 0 <= start <= stop, got {self.start}, {self.stop}"
            )

    @property
    def n_samples(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True, eq=False)
class Sweep:
    """One sweep: all channels sampled over the same interval.

    Attributes
    ----------
    index : int
        Zero-based sweep number in the recording.
    data : numpy.ndarray
        Float64 array shaped ``(n_channels, n_samples)`` in SI units. Stored read-only.
    sample_rate : float
        Samples per second per channel, in Hz.
    t0 : float
        Start of this sweep in seconds, relative to the start of the first sweep.
    epochs : tuple of Epoch
        Command epochs played during this sweep, in time order.
    """

    index: int
    data: FloatArray
    sample_rate: float
    t0: float = 0.0
    epochs: tuple[Epoch, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray) or self.data.dtype != np.float64:
            raise TypeError(f"sweep {self.index}: data must be a float64 numpy array")
        if self.data.ndim != 2:
            raise ValueError(
                f"sweep {self.index}: data must be 2D (n_channels, n_samples), "
                f"got shape {self.data.shape}"
            )
        if not (math.isfinite(self.sample_rate) and self.sample_rate > 0):
            raise ValueError(f"sweep {self.index}: sample_rate must be positive and finite")
        if self.data.flags.writeable:
            view = self.data.view()
            view.flags.writeable = False
            object.__setattr__(self, "data", view)
        object.__setattr__(self, "epochs", tuple(self.epochs))

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        """Length of the sweep in seconds, ``n_samples / sample_rate``."""
        return self.n_samples / self.sample_rate

    def time(self, *, absolute: bool = False) -> FloatArray:
        """Sample times in seconds.

        Parameters
        ----------
        absolute : bool, default False
            False gives times from the start of this sweep (``i / sample_rate``). True adds
            ``t0``, giving times from the start of the first sweep.
        """
        t = np.arange(self.n_samples, dtype=np.float64) / self.sample_rate
        return t + self.t0 if absolute else t


@dataclass(frozen=True, slots=True, eq=False)
class Recording:
    """A file's worth of sweeps that share channels and sample rate.

    Attributes
    ----------
    channels : tuple of Channel
        Channel definitions, in the row order of every ``Sweep.data``.
    sweeps : tuple of Sweep
        Sweeps in acquisition order. Gap-free recordings have a single sweep.
    metadata : Mapping[str, Any]
        File-level details: format and version, creation time, protocol, comment, the
        source file's SHA-256, and the reader's raw header values.
    """

    channels: tuple[Channel, ...]
    sweeps: tuple[Sweep, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "sweeps", tuple(self.sweeps))
        object.__setattr__(self, "metadata", _frozen(self.metadata))
        if not self.sweeps:
            raise ValueError("a recording needs at least one sweep")
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be unique, got {names}")
        rate = self.sweeps[0].sample_rate
        for i, sweep in enumerate(self.sweeps):
            if sweep.index != i:
                raise ValueError(f"sweep at position {i} has index {sweep.index}")
            if sweep.n_channels != len(self.channels):
                raise ValueError(
                    f"sweep {i} has {sweep.n_channels} channels, recording defines "
                    f"{len(self.channels)}"
                )
            if sweep.sample_rate != rate:
                raise ValueError(
                    f"sweep {i} sample rate {sweep.sample_rate} Hz differs from sweep 0 ({rate} Hz)"
                )

    @property
    def sample_rate(self) -> float:
        """Samples per second per channel, in Hz."""
        return self.sweeps[0].sample_rate

    @property
    def n_sweeps(self) -> int:
        return len(self.sweeps)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def channel_index(self, name: str) -> int:
        """Row of ``Sweep.data`` that holds channel ``name``.

        Raises
        ------
        KeyError
            If no channel has that name; the message lists the available names.
        """
        for i, channel in enumerate(self.channels):
            if channel.name == name:
                return i
        raise KeyError(f"no channel {name!r}; channels are {[c.name for c in self.channels]}")
