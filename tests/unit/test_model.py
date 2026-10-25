"""Recording, Sweep, Channel and Epoch invariants."""

from __future__ import annotations

import numpy as np
import pytest

from ephys_forever.core.model import Channel, Epoch, Recording, Sweep


def make_sweep(index: int = 0, n_channels: int = 2, n: int = 10, rate: float = 1e3) -> Sweep:
    return Sweep(index=index, data=np.zeros((n_channels, n)), sample_rate=rate)


def channels() -> tuple[Channel, ...]:
    return (Channel("IN 0", "A", "current"), Channel("IN 1", "V", "voltage"))


def test_sweep_time_is_derived_from_rate_and_t0() -> None:
    sweep = Sweep(index=0, data=np.zeros((1, 4)), sample_rate=2.0, t0=10.0)
    np.testing.assert_array_equal(sweep.time(), [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(sweep.time(absolute=True), [10.0, 10.5, 11.0, 11.5])
    assert sweep.duration == 2.0
    assert (sweep.n_channels, sweep.n_samples) == (1, 4)


def test_sweep_data_is_read_only_and_caller_array_untouched() -> None:
    source = np.zeros((1, 3))
    sweep = Sweep(index=0, data=source, sample_rate=1.0)
    with pytest.raises(ValueError, match="read-only"):
        sweep.data[0, 0] = 1.0
    source[0, 0] = 2.0  # the caller's array stays writable
    assert source.flags.writeable


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (np.zeros((2, 3), dtype=np.float32), TypeError),
        ([[0.0, 1.0]], TypeError),
        (np.zeros(3), ValueError),
    ],
)
def test_sweep_rejects_bad_data(data: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        Sweep(index=0, data=data, sample_rate=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_sweep_rejects_bad_sample_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        make_sweep(rate=rate)


def test_channel_kind_must_match_unit() -> None:
    with pytest.raises(ValueError, match="does not match"):
        Channel("IN 0", "V", "current")
    assert Channel("Temp", "Degrees", "other").is_si is False
    assert Channel("IN 0", "A", "current").is_si


def test_metadata_is_read_only() -> None:
    ch = Channel("IN 0", "A", "current", metadata={"gain": 1.0})
    with pytest.raises(TypeError):
        ch.metadata["gain"] = 2.0  # type: ignore[index]


def test_epoch_bounds() -> None:
    epoch = Epoch("B", "step", 2093, 3093, 0.01, "V", "Cmd 0")
    assert epoch.n_samples == 1000
    with pytest.raises(ValueError, match="start <= stop"):
        Epoch("B", "step", 10, 5, 0.0, "V", "Cmd 0")


def test_recording_properties_and_channel_lookup() -> None:
    rec = Recording(channels(), (make_sweep(0), make_sweep(1)), {"format": "test"})
    assert (rec.n_sweeps, rec.n_channels, rec.sample_rate) == (2, 2, 1e3)
    assert rec.channel_index("IN 1") == 1
    with pytest.raises(KeyError, match="IN 0"):
        rec.channel_index("IN 9")


def test_recording_validation() -> None:
    with pytest.raises(ValueError, match="at least one sweep"):
        Recording(channels(), ())
    with pytest.raises(ValueError, match="channels"):
        Recording(channels(), (make_sweep(n_channels=3),))
    with pytest.raises(ValueError, match="sample rate"):
        Recording(channels(), (make_sweep(0), make_sweep(1, rate=2e3)))
    with pytest.raises(ValueError, match="index"):
        Recording(channels(), (make_sweep(1),))
    with pytest.raises(ValueError, match="unique"):
        Recording((channels()[0], channels()[0]), (make_sweep(),))
