"""Reader for Axon Binary Format 2.x (``.abf`` files that start with ``ABF2``).

Implemented from the publicly documented ABF2 layout: a 512-byte header holding a
section map, with each section stored at ``block_index * 512``. Integer samples are
interleaved by channel and converted to SI with the per-channel scaling in the ADC
section::

    gain   = fInstrumentScaleFactor * fSignalGain * fADCProgrammableGain
             * (fTelegraphAdditGain if nTelegraphEnable else 1)
    value  = raw * (fADCRange / lADCResolution) / gain + fInstrumentOffset - fSignalOffset
    si     = value * factor(unit)          # e.g. 1e-12 for "pA"

The three float64 multiplications and the addition are done in that order, so a reader
test can rebuild every sample exactly from the raw integers and the factors kept in
``Channel.metadata["scaling"]``.

Command epochs start after a holding period of ``samples_per_sweep_per_channel // 64``
samples, which the acquisition software inserts before the first epoch.

Supported: episodic stimulation, fixed-length event-driven and gap-free files, int16 or
float32 samples. Variable-length event-driven files raise
:class:`~ephys_forever.core.io.errors.UnsupportedFeatureError`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import struct
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from ephys_forever.core.io.errors import ReaderError, UnsupportedFeatureError
from ephys_forever.core.model import Channel, Epoch, Recording, Sweep
from ephys_forever.core.units import UnknownUnitError, channel_kind, parse_unit

logger = logging.getLogger(__name__)

BLOCK_SIZE = 512
SIGNATURE = b"ABF2"

SECTION_NAMES: tuple[str, ...] = (
    "Protocol",
    "ADC",
    "DAC",
    "Epoch",
    "ADCPerDAC",
    "EpochPerDAC",
    "UserList",
    "StatsRegion",
    "Math",
    "Strings",
    "Data",
    "Tag",
    "Scope",
    "Delta",
    "VoiceTag",
    "SynchArray",
    "Annotation",
    "Stats",
)
SECTION_MAP_OFFSET = 76

# (field name, byte offset, struct format), little-endian.
HEADER_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("uFileInfoSize", 8, "I"),
    ("uActualEpisodes", 12, "I"),
    ("uFileStartDate", 16, "I"),
    ("uFileStartTimeMS", 20, "I"),
    ("uStopwatchTime", 24, "I"),
    ("nFileType", 28, "h"),
    ("nDataFormat", 30, "h"),
    ("nSimultaneousScan", 32, "h"),
    ("nCRCEnable", 34, "h"),
    ("uFileCRC", 36, "I"),
    ("uCreatorNameIndex", 60, "I"),
    ("uModifierVersion", 64, "I"),
    ("uModifierNameIndex", 68, "I"),
    ("uProtocolPathIndex", 72, "I"),
)
PROTOCOL_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("nOperationMode", 0, "h"),
    ("fADCSequenceInterval", 2, "f"),
    ("fSynchTimeUnit", 14, "f"),
    ("fSecondsPerRun", 18, "f"),
    ("lNumSamplesPerEpisode", 22, "i"),
    ("lPreTriggerSamples", 26, "i"),
    ("lEpisodesPerRun", 30, "i"),
    ("lRunsPerTrial", 34, "i"),
    ("lNumberOfTrials", 38, "i"),
    ("fEpisodeStartToStart", 62, "f"),
    ("fRunStartToStart", 66, "f"),
    ("fTrialStartToStart", 74, "f"),
    ("fADCRange", 110, "f"),
    ("fDACRange", 114, "f"),
    ("lADCResolution", 118, "i"),
    ("lDACResolution", 122, "i"),
    ("nExperimentType", 126, "h"),
    ("lFileCommentIndex", 132, "i"),
)
ADC_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("nADCNum", 0, "h"),
    ("nTelegraphEnable", 2, "h"),
    ("nTelegraphInstrument", 4, "h"),
    ("fTelegraphAdditGain", 6, "f"),
    ("fTelegraphFilter", 10, "f"),
    ("fTelegraphMembraneCap", 14, "f"),
    ("nTelegraphMode", 18, "h"),
    ("fTelegraphAccessResistance", 20, "f"),
    ("nADCPtoLChannelMap", 24, "h"),
    ("nADCSamplingSeq", 26, "h"),
    ("fADCProgrammableGain", 28, "f"),
    ("fADCDisplayAmplification", 32, "f"),
    ("fADCDisplayOffset", 36, "f"),
    ("fInstrumentScaleFactor", 40, "f"),
    ("fInstrumentOffset", 44, "f"),
    ("fSignalGain", 48, "f"),
    ("fSignalOffset", 52, "f"),
    ("fSignalLowpassFilter", 56, "f"),
    ("fSignalHighpassFilter", 60, "f"),
    ("lADCChannelNameIndex", 74, "i"),
    ("lADCUnitsIndex", 78, "i"),
)
DAC_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("nDACNum", 0, "h"),
    ("nTelegraphDACScaleFactorEnable", 2, "h"),
    ("fInstrumentHoldingLevel", 4, "f"),
    ("fDACScaleFactor", 8, "f"),
    ("fDACHoldingLevel", 12, "f"),
    ("fDACCalibrationFactor", 16, "f"),
    ("fDACCalibrationOffset", 20, "f"),
    ("lDACChannelNameIndex", 24, "i"),
    ("lDACChannelUnitsIndex", 28, "i"),
    ("nWaveformEnable", 40, "h"),
    ("nWaveformSource", 42, "h"),
    ("nInterEpisodeLevel", 44, "h"),
)
EPOCH_PER_DAC_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("nEpochNum", 0, "h"),
    ("nDACNum", 2, "h"),
    ("nEpochType", 4, "h"),
    ("fEpochInitLevel", 6, "f"),
    ("fEpochLevelInc", 10, "f"),
    ("lEpochInitDuration", 14, "i"),
    ("lEpochDurationInc", 18, "i"),
    ("lEpochPulsePeriod", 22, "i"),
    ("lEpochPulseWidth", 26, "i"),
)
SYNCH_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("lStart", 0, "i"),
    ("lLength", 4, "i"),
)

OPERATION_MODES = {
    1: "event_driven_variable",
    2: "event_driven_fixed",
    3: "gap_free",
    4: "high_speed_oscilloscope",
    5: "episodic",
}
EPOCH_TYPES = {1: "step", 2: "ramp", 3: "pulse", 4: "triangle", 5: "cosine", 7: "biphasic"}
DATA_FORMATS: dict[int, np.dtype[Any]] = {0: np.dtype("<i2"), 1: np.dtype("<f4")}
WAVEFORM_SOURCE_EPOCHS = 1
STRINGS_SIGNATURE = b"SSCH"
STRINGS_HEADER_SIZE = 44


@dataclass(frozen=True, slots=True)
class Section:
    """One entry of the ABF2 section map."""

    block: int
    entry_bytes: int
    n_entries: int

    @property
    def offset(self) -> int:
        return self.block * BLOCK_SIZE


def _unpack(
    buf: bytes, base: int, fields: Sequence[tuple[str, int, str]], where: str
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, off, fmt in fields:
        try:
            out[name] = struct.unpack_from("<" + fmt, buf, base + off)[0]
        except struct.error as exc:
            raise ReaderError(f"{where}: file ends before field {name}") from exc
    return out


def _entries(
    buf: bytes, section: Section, fields: Sequence[tuple[str, int, str]], name: str
) -> list[dict[str, Any]]:
    return [
        _unpack(buf, section.offset + i * section.entry_bytes, fields, f"{name}[{i}]")
        for i in range(section.n_entries)
    ]


def _strings(buf: bytes, section: Section) -> list[str]:
    """Strings section: 44-byte header (``SSCH``, version, count, ...) then NUL-separated
    strings. Header fields refer to them with 1-based indices."""
    if section.n_entries == 0:
        return []
    start = section.offset
    if buf[start : start + 4] != STRINGS_SIGNATURE:
        raise ReaderError(f"strings section at byte {start} does not start with SSCH")
    count = struct.unpack_from("<I", buf, start + 8)[0]
    body = buf[start + STRINGS_HEADER_SIZE : start + section.entry_bytes]
    parts = body.split(b"\x00")
    if len(parts) < count:
        raise ReaderError(f"strings section holds {len(parts)} strings, header says {count}")
    return [p.decode("latin-1").strip() for p in parts[:count]]


def _string(strings: Sequence[str], index: int) -> str:
    return strings[index - 1] if 1 <= index <= len(strings) else ""


def _version(raw: bytes) -> tuple[int, int, int, int]:
    """Version bytes are stored least significant first."""
    return raw[3], raw[2], raw[1], raw[0]


def _unit(text: str, owner: str) -> tuple[str, float]:
    """SI symbol and factor for a unit string, or the text itself and 1.0 if not SI."""
    try:
        parsed = parse_unit(text)
    except UnknownUnitError:
        logger.warning("%s: unit %r is not SI; values are kept in file units", owner, text)
        return text, 1.0
    return parsed.si_unit, parsed.factor


def read_abf2(path: str | PathLike[str]) -> Recording:
    """Read an ABF 2.x file into a :class:`~ephys_forever.core.model.Recording`.

    Parameters
    ----------
    path : str or path-like
        File to read.

    Returns
    -------
    Recording
        Channels in ADC-section order, values in SI units (float64). Sweep ``t0`` comes
        from the synch array when present, otherwise from the protocol's sweep
        start-to-start interval. ``metadata`` keeps the raw header sections.

    Raises
    ------
    ReaderError
        If the file is not ABF2 or its sections are inconsistent.
    UnsupportedFeatureError
        For variable-length event-driven recordings or an unknown sample format.
    """
    path = Path(path)
    buf = path.read_bytes()
    if buf[:4] != SIGNATURE:
        raise ReaderError(
            f"{path.name}: signature {buf[:4]!r} is not {SIGNATURE!r}; "
            "ABF 1.x files need the ABF1 reader"
        )
    if len(buf) < BLOCK_SIZE:
        raise ReaderError(f"{path.name}: {len(buf)} bytes is shorter than the ABF2 header")

    header = _unpack(buf, 0, HEADER_FIELDS, "header")
    file_version = _version(buf[4:8])
    creator_version = _version(buf[56:60])
    sections: dict[str, Section] = {}
    for i, name in enumerate(SECTION_NAMES):
        block, entry_bytes, n_entries = struct.unpack_from("<IIq", buf, SECTION_MAP_OFFSET + 16 * i)
        sections[name] = Section(block, entry_bytes, n_entries)

    strings = _strings(buf, sections["Strings"])
    protocol = _unpack(buf, sections["Protocol"].offset, PROTOCOL_FIELDS, "protocol")
    adcs = _entries(buf, sections["ADC"], ADC_FIELDS, "ADC")
    dacs = _entries(buf, sections["DAC"], DAC_FIELDS, "DAC")
    epochs_raw = _entries(buf, sections["EpochPerDAC"], EPOCH_PER_DAC_FIELDS, "EpochPerDAC")
    synch = _entries(buf, sections["SynchArray"], SYNCH_FIELDS, "SynchArray")

    mode_code = protocol["nOperationMode"]
    mode = OPERATION_MODES.get(mode_code, f"unknown_{mode_code}")
    if mode_code == 1:
        raise UnsupportedFeatureError(
            f"{path.name}: variable-length event-driven recordings are not supported yet"
        )
    if mode_code not in OPERATION_MODES:
        raise ReaderError(f"{path.name}: unknown operation mode {mode_code}")
    if header["nDataFormat"] not in DATA_FORMATS:
        raise UnsupportedFeatureError(
            f"{path.name}: unknown sample format code {header['nDataFormat']}"
        )
    dtype = DATA_FORMATS[header["nDataFormat"]]

    n_channels = len(adcs)
    data_section = sections["Data"]
    total = data_section.n_entries
    if n_channels == 0:
        raise ReaderError(f"{path.name}: ADC section lists no channels")
    if data_section.entry_bytes != dtype.itemsize:
        raise ReaderError(
            f"{path.name}: data entries are {data_section.entry_bytes} bytes, "
            f"format {dtype} needs {dtype.itemsize}"
        )
    if mode == "gap_free":
        n_sweeps = 1
        n_per = total // n_channels
    else:
        n_sweeps = header["uActualEpisodes"]
        n_per = protocol["lNumSamplesPerEpisode"] // n_channels
    if n_sweeps * n_per * n_channels != total:
        raise ReaderError(
            f"{path.name}: {total} samples do not fill {n_sweeps} sweeps of {n_per} samples "
            f"x {n_channels} channels"
        )
    end = data_section.offset + total * dtype.itemsize
    if end > len(buf):
        raise ReaderError(f"{path.name}: data section runs past end of file ({end} bytes)")

    raw = np.frombuffer(buf, dtype=dtype, count=total, offset=data_section.offset)
    raw_by_sweep = raw.reshape(n_sweeps, n_per, n_channels).transpose(0, 2, 1)

    channels: list[Channel] = []
    file_scale = np.ones(n_channels)
    file_offset = np.zeros(n_channels)
    si_factor = np.ones(n_channels)
    for i, adc in enumerate(adcs):
        name = _string(strings, adc["lADCChannelNameIndex"])
        file_unit = _string(strings, adc["lADCUnitsIndex"])
        unit, factor = _unit(file_unit, f"{path.name} channel {name!r}")
        add_gain = adc["fTelegraphAdditGain"] if adc["nTelegraphEnable"] else 1.0
        gain = (
            adc["fInstrumentScaleFactor"]
            * adc["fSignalGain"]
            * adc["fADCProgrammableGain"]
            * add_gain
        )
        if dtype.kind == "i":
            if gain == 0:
                raise ReaderError(f"{path.name}: channel {name!r} has zero total gain")
            file_scale[i] = protocol["fADCRange"] / protocol["lADCResolution"] / gain
            file_offset[i] = adc["fInstrumentOffset"] - adc["fSignalOffset"]
        si_factor[i] = factor
        channels.append(
            Channel(
                name=name,
                unit=unit,
                kind=channel_kind(unit),
                metadata={
                    "adc_num": adc["nADCNum"],
                    "file_unit": file_unit,
                    "scaling": {
                        "raw_dtype": dtype.str,
                        "file_scale": float(file_scale[i]),
                        "file_offset": float(file_offset[i]),
                        "si_factor": float(si_factor[i]),
                        "formula": "(raw * file_scale + file_offset) * si_factor",
                    },
                    "telegraph": {
                        "enabled": bool(adc["nTelegraphEnable"]),
                        "instrument": adc["nTelegraphInstrument"],
                        "additional_gain": adc["fTelegraphAdditGain"],
                        "lowpass_hz": adc["fTelegraphFilter"],
                        "membrane_capacitance_f": adc["fTelegraphMembraneCap"] * 1e-12,
                        "mode": adc["nTelegraphMode"],
                    },
                    "header": adc,
                },
            )
        )

    # De-interleave one channel at a time straight into the output: reading the strided
    # int16 view is far cheaper than transposing a float64 copy of the whole file.
    scaled = np.empty((n_sweeps, n_channels, n_per), dtype=np.float64)
    for c in range(n_channels):
        out = scaled[:, c, :]
        np.multiply(raw_by_sweep[:, c, :], file_scale[c], out=out)
        if file_offset[c] != 0.0:
            out += file_offset[c]
        if si_factor[c] != 1.0:
            out *= si_factor[c]

    sample_interval_us = protocol["fADCSequenceInterval"]
    if not sample_interval_us > 0:
        raise ReaderError(f"{path.name}: sample interval {sample_interval_us} us is not positive")
    sample_rate = 1e6 / sample_interval_us

    t0s, t0_source = _sweep_starts(protocol, synch, n_sweeps, mode)
    dac_info = [_dac_info(strings, dac) for dac in dacs]
    sweeps = [
        Sweep(
            index=k,
            data=scaled[k],
            sample_rate=sample_rate,
            t0=t0s[k],
            epochs=() if mode == "gap_free" else _epochs(dacs, dac_info, epochs_raw, k, n_per),
        )
        for k in range(n_sweeps)
    ]

    protocol_path = _string(strings, header["uProtocolPathIndex"])
    metadata: dict[str, Any] = {
        "format": "abf2",
        "format_version": f"{file_version[0]}.{file_version[1]:02d}",
        "format_version_full": ".".join(map(str, file_version)),
        "source_path": str(path),
        "sha256": hashlib.sha256(buf).hexdigest(),
        "creator": _string(strings, header["uCreatorNameIndex"]),
        "creator_version": ".".join(map(str, creator_version)),
        "created": _created(header["uFileStartDate"], header["uFileStartTimeMS"]),
        "guid": "{" + str(uuid.UUID(bytes_le=buf[40:56])).upper() + "}",
        "protocol_path": protocol_path,
        "protocol": PureWindowsPath(protocol_path).stem if protocol_path else "",
        "comment": _string(strings, protocol["lFileCommentIndex"]),
        "operation_mode": mode,
        "data_format": dtype.str,
        "sample_interval_us": sample_interval_us,
        "total_samples": total,
        "samples_per_sweep_per_channel": n_per,
        "pre_epoch_samples": n_per // 64,
        "sweep_start_source": t0_source,
        "dacs": dac_info,
        "header": header,
        "protocol_section": protocol,
        "sections": {
            k: {"block": s.block, "entry_bytes": s.entry_bytes, "n_entries": s.n_entries}
            for k, s in sections.items()
        },
        "synch_array": synch,
        "strings": strings,
    }
    return Recording(channels=tuple(channels), sweeps=tuple(sweeps), metadata=metadata)


def _created(date: int, time_ms: int) -> str:
    """ISO timestamp from a ``YYYYMMDD`` integer and milliseconds after midnight."""
    if date == 0:
        return ""
    day = dt.datetime(date // 10000, date // 100 % 100, date % 100)
    return (day + dt.timedelta(milliseconds=time_ms)).isoformat(timespec="milliseconds")


def _sweep_starts(
    protocol: dict[str, Any], synch: list[dict[str, Any]], n_sweeps: int, mode: str
) -> tuple[list[float], str]:
    """Sweep start times relative to the first sweep, and where they came from."""
    unit_us = protocol["fSynchTimeUnit"]
    if mode != "gap_free" and len(synch) == n_sweeps and unit_us > 0:
        first = synch[0]["lStart"]
        return [(s["lStart"] - first) * unit_us * 1e-6 for s in synch], "synch_array"
    if mode == "gap_free":
        return [0.0], "gap_free"
    step = protocol["fEpisodeStartToStart"]
    return [k * step for k in range(n_sweeps)], "episode_start_to_start"


def _dac_info(strings: Sequence[str], dac: dict[str, Any]) -> dict[str, Any]:
    name = _string(strings, dac["lDACChannelNameIndex"])
    file_unit = _string(strings, dac["lDACChannelUnitsIndex"])
    unit, factor = _unit(file_unit, f"command output {name!r}")
    return {
        "dac_num": dac["nDACNum"],
        "name": name,
        "unit": unit,
        "file_unit": file_unit,
        "si_factor": factor,
        "holding_level": dac["fDACHoldingLevel"] * factor,
        "waveform_from_epochs": bool(dac["nWaveformEnable"])
        and dac["nWaveformSource"] == WAVEFORM_SOURCE_EPOCHS,
        "header": dac,
    }


def _epochs(
    dacs: list[dict[str, Any]],
    dac_info: list[dict[str, Any]],
    epochs_raw: list[dict[str, Any]],
    sweep: int,
    n_per: int,
) -> tuple[Epoch, ...]:
    """Epochs of every DAC whose waveform comes from the epoch table, for one sweep."""
    out: list[Epoch] = []
    for dac, info in zip(dacs, dac_info, strict=True):
        if not info["waveform_from_epochs"]:
            continue
        rows = sorted(
            (e for e in epochs_raw if e["nDACNum"] == dac["nDACNum"]),
            key=lambda e: int(e["nEpochNum"]),
        )
        position = n_per // 64
        for e in rows:
            if e["nEpochType"] == 0:
                continue
            duration = e["lEpochInitDuration"] + e["lEpochDurationInc"] * sweep
            if duration < 0:
                raise ReaderError(
                    f"epoch {e['nEpochNum']} on {info['name']} has negative duration "
                    f"{duration} in sweep {sweep}"
                )
            stop = position + duration
            if stop > n_per:
                raise ReaderError(
                    f"epoch {e['nEpochNum']} on {info['name']} ends at sample {stop}, "
                    f"past the sweep's {n_per} samples"
                )
            level = (e["fEpochInitLevel"] + e["fEpochLevelInc"] * sweep) * info["si_factor"]
            out.append(
                Epoch(
                    label=chr(ord("A") + int(e["nEpochNum"])),
                    kind=EPOCH_TYPES.get(e["nEpochType"], f"type_{e['nEpochType']}"),
                    start=position,
                    stop=stop,
                    level=level,
                    unit=info["unit"],
                    output=info["name"],
                )
            )
            position = stop
    return tuple(out)
