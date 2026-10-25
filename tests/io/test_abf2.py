"""ABF2 reader against the voltage-clamp IV steps recording.

Expected values come from ``tests/reference/clampfit/vclamp_iv_steps_001/``: the
acquisition software's File Properties export and ``expected.json`` transcribed from it.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ephys_forever.core.io.abf2 import read_abf2
from ephys_forever.core.io.errors import ReaderError
from ephys_forever.core.model import Recording
from ephys_forever.core.units import parse_unit

REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "clampfit" / "vclamp_iv_steps_001"
FILE = "vclamp_iv_steps_001.abf"

pytestmark = [pytest.mark.realdata, pytest.mark.reference]


@pytest.fixture(scope="module")
def expected() -> dict[str, Any]:
    with (REFERENCE / "expected.json").open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@pytest.fixture(scope="module")
def path(real_data_file: Callable[[str], Path]) -> Path:
    return real_data_file(FILE)


@pytest.fixture(scope="module")
def rec(path: Path) -> Recording:
    return read_abf2(path)


def test_file_identity(rec: Recording, expected: dict[str, Any]) -> None:
    md = rec.metadata
    assert md["sha256"] == expected["sha256"]
    assert md["format"] == "abf2"
    assert md["format_version"] == expected["abf_version"]
    assert md["created"] == expected["created"]
    assert f"{md['creator']} {md['creator_version']}".startswith(expected["created_by"])
    assert md["protocol"] == expected["protocol_name"]
    assert md["comment"] == expected["comment"]
    assert md["operation_mode"] == expected["acquisition_mode"]
    assert md["guid"] == "{10540596-0FE0-4C65-98E8-724A535602A8}"  # File Properties export


def test_shape_and_timing(rec: Recording, expected: dict[str, Any]) -> None:
    assert rec.metadata["total_samples"] == expected["total_samples_in_file"]
    assert rec.n_sweeps == expected["sweep_count"]
    assert rec.n_channels == expected["channel_count"]
    assert rec.sample_rate == expected["sampling_rate_hz"]
    assert rec.metadata["sample_interval_us"] == expected["sample_interval_us"]
    for sweep in rec.sweeps:
        assert sweep.data.shape == (3, expected["samples_per_sweep_per_channel"])
        assert sweep.data.dtype == np.float64
        assert sweep.duration * 1e3 == pytest.approx(expected["sweep_duration_ms"], rel=1e-12)
    starts = np.diff([s.t0 for s in rec.sweeps])
    np.testing.assert_allclose(starts, expected["sweep_start_interval_s"], rtol=1e-12)
    assert rec.metadata["sweep_start_source"] == "synch_array"


def test_channels(rec: Recording, expected: dict[str, Any]) -> None:
    for channel, exp in zip(rec.channels, expected["channels"], strict=True):
        assert channel.name == exp["name"]
        assert channel.metadata["file_unit"] == exp["units"]
        assert channel.unit == parse_unit(exp["units"]).si_unit
        assert channel.kind == {"A": "current", "V": "voltage"}[channel.unit]
        assert channel.metadata["telegraph"]["lowpass_hz"] == exp["lowpass_hz"]


def test_samples_rebuild_exactly_from_raw_integers(rec: Recording, path: Path) -> None:
    md = rec.metadata
    data = md["sections"]["Data"]
    raw = np.fromfile(path, dtype="<i2", count=data["n_entries"], offset=data["block"] * 512)
    raw = raw.reshape(rec.n_sweeps, -1, rec.n_channels)
    for c, channel in enumerate(rec.channels):
        s = channel.metadata["scaling"]
        rebuilt = (raw[:, :, c].astype(np.float64) * s["file_scale"] + s["file_offset"]) * s[
            "si_factor"
        ]
        actual = np.stack([sweep.data[c] for sweep in rec.sweeps])
        np.testing.assert_array_equal(actual, rebuilt)


def test_scale_factor_matches_header_formula(rec: Recording) -> None:
    in0 = rec.channels[0].metadata
    header = in0["header"]
    proto = rec.metadata["protocol_section"]
    gain = header["fInstrumentScaleFactor"] * header["fSignalGain"]
    gain *= header["fADCProgrammableGain"] * header["fTelegraphAdditGain"]
    assert in0["scaling"]["file_scale"] == proto["fADCRange"] / proto["lADCResolution"] / gain
    assert in0["scaling"]["si_factor"] == 1e-12


def test_command_epochs(rec: Recording, expected: dict[str, Any]) -> None:
    cmd = expected["command"]
    dac0 = rec.metadata["dacs"][0]
    assert dac0["name"] == cmd["dac"]
    assert dac0["file_unit"] == cmd["units"]
    assert dac0["holding_level"] == cmd["holding_level_in_file_mv"] * 1e-3
    assert rec.metadata["pre_epoch_samples"] == cmd["pre_epoch_holding_samples"]
    for sweep, level_mv in zip(rec.sweeps, cmd["step_levels_mv_by_sweep"], strict=True):
        labels = [e.label for e in sweep.epochs]
        assert labels == [e["label"] for e in cmd["epochs"]]
        for epoch, exp in zip(sweep.epochs, cmd["epochs"], strict=True):
            assert epoch.kind == exp["type"]
            assert epoch.n_samples == exp["duration_samples"]
            assert epoch.output == "Cmd 0"
            assert epoch.unit == "V"
        step = sweep.epochs[1]
        assert (step.start, step.stop) == (cmd["step_start_sample"], cmd["step_end_sample"])
        assert step.level == pytest.approx(level_mv * 1e-3, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    ("key", "channel", "sweep", "window"),
    [
        ("IN 1 sweep 0 mean samples 0..2000 (mV)", "IN 1", 0, (0, 2000)),
        ("IN 1 sweep 8 mean samples 2000..3000 (mV)", "IN 1", 8, (2000, 3000)),
        ("IN 0 sweep 4 mean samples 2000..3000 (pA)", "IN 0", 4, (2000, 3000)),
        ("IN 0C sweep 4 mean samples 2000..3000 (pA)", "IN 0C", 4, (2000, 3000)),
    ],
)
def test_spot_check_means(
    rec: Recording,
    expected: dict[str, Any],
    key: str,
    channel: str,
    sweep: int,
    window: tuple[int, int],
) -> None:
    """Means from an independent open-source reader, rounded to 3 decimals in file units."""
    c = rec.channel_index(channel)
    factor = rec.channels[c].metadata["scaling"]["si_factor"]
    mean = float(np.mean(rec.sweeps[sweep].data[c, window[0] : window[1]])) / factor
    assert mean == pytest.approx(expected["spot_checks_from_pyabf"][key], abs=5e-4)


def test_rejects_non_abf2(tmp_path: Path) -> None:
    bad = tmp_path / "old.abf"
    bad.write_bytes(b"ABF " + bytes(600))
    with pytest.raises(ReaderError, match="ABF1 reader"):
        read_abf2(bad)


def test_rejects_truncated_data(path: Path, tmp_path: Path) -> None:
    buf = path.read_bytes()
    cut = tmp_path / "cut.abf"
    cut.write_bytes(buf[: len(buf) // 2])
    with pytest.raises(ReaderError, match=r"file ends before|past end of file"):
        read_abf2(cut)


def test_rejects_inconsistent_sample_count(path: Path, tmp_path: Path) -> None:
    buf = bytearray(path.read_bytes())
    data_entry = 76 + 16 * 10  # Data is the 11th section-map entry
    struct.pack_into("<q", buf, data_entry + 8, 161_999)
    odd = tmp_path / "odd.abf"
    odd.write_bytes(bytes(buf))
    with pytest.raises(ReaderError, match="do not fill"):
        read_abf2(odd)
