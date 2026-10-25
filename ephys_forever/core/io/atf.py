"""Reader for Axon Text Files (ATF 1.0), the tab-separated text format.

Layout, from the public ATF description::

    ATF<TAB>1.0
    <n_header_records><TAB><n_columns>
    "Key=value"                       x n_header_records, quoted
    "Title (unit)"<TAB>...            column titles
    value<TAB>value...                data rows, one per sample

Episodic exports put time in the first column and then one column per channel per
sweep, titled ``"Trace #k (unit)"``. The ``Signals=`` record names the channel of
each data column. Columns not titled ``Trace #k`` are read as one channel each in a
single sweep (gap-free exports).

The time column is written with about float32 precision, so the sample interval is
taken from a least-squares line through the times and rounded to 6 significant
digits, and the sample rate is its reciprocal rounded to 6 significant digits. Times
that stray from a uniform grid raise :class:`ReaderError`.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ephys_forever.core.io.errors import ReaderError, UnsupportedFeatureError
from ephys_forever.core.model import Channel, FloatArray, Recording, Sweep
from ephys_forever.core.units import UnknownUnitError, channel_kind, parse_unit

SIGNATURE = "ATF"
TITLE_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<unit>[^()]*)\)\s*$")
TRACE_RE = re.compile(r"^Trace #(?P<n>\d+)$")
INTERVAL_SIGNIFICANT_DIGITS = 6
TIME_GRID_RTOL = 1e-6


def _unquote(field: str) -> str:
    field = field.strip()
    if len(field) >= 2 and field[0] == field[-1] == '"':
        return field[1:-1]
    return field


def _split_title(title: str) -> tuple[str, str]:
    match = TITLE_RE.match(title)
    if match is None:
        return title.strip(), ""
    return match["name"], match["unit"].strip()


def _records(lines: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Header records as ``key -> value``, plus the ``Signals=`` list."""
    records: dict[str, str] = {}
    signals: list[str] = []
    for line in lines:
        fields = [_unquote(f) for f in line.split("\t")]
        key, sep, value = fields[0].partition("=")
        if not sep:
            raise ReaderError(f"header record {line!r} is not key=value")
        records[key] = value
        if key == "Signals":
            signals = [f for f in fields[1:] if f]
    return records, signals


def _round_significant(x: float, digits: int) -> float:
    if x == 0:
        return 0.0
    return round(x, digits - 1 - math.floor(math.log10(abs(x))))


def _sample_interval(t: FloatArray, name: str) -> float:
    """Sample interval from a uniform-grid fit of the time column, in time-column units."""
    if t.size < 2:
        raise ReaderError(f"{name}: need at least 2 samples to find the sample interval")
    i = np.arange(t.size, dtype=np.float64)
    slope = np.polyfit(i, t, 1)[0]
    dt = _round_significant(float(slope), INTERVAL_SIGNIFICANT_DIGITS)
    if not dt > 0:
        raise ReaderError(f"{name}: time column does not increase")
    worst = float(np.max(np.abs(t - (t[0] + i * dt))))
    span = float(abs(t[-1] - t[0]))
    if worst > TIME_GRID_RTOL * span + abs(dt) * 1e-3:
        raise ReaderError(
            f"{name}: time column is not uniformly sampled (max deviation {worst:g} from a "
            f"{dt:g} grid)"
        )
    return dt


def _unit(text: str) -> tuple[str, float]:
    try:
        parsed = parse_unit(text)
    except UnknownUnitError:
        return text, 1.0
    return parsed.si_unit, parsed.factor


def read_atf(path: str | PathLike[str]) -> Recording:
    """Read an ATF 1.0 file of time-series data into a Recording.

    Parameters
    ----------
    path : str or path-like
        File to read.

    Returns
    -------
    Recording
        Values converted to SI (float64) using the unit in each column title. Sweep
        ``t0`` comes from the ``SweepStartTimesMS`` record when it lists every sweep,
        otherwise sweeps are assumed back to back. ``metadata["records"]`` keeps every
        header record.

    Raises
    ------
    ReaderError
        If the file is not ATF 1.0, the counts in the header do not match the content,
        or the time column is not uniform.
    UnsupportedFeatureError
        If the first column is not time (for example a results table), or sweeps do not
        all contain the same channels.
    """
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("latin-1")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ReaderError(f"{path.name}: too short to be an ATF file")
    first = lines[0].split("\t")
    if first[0].strip() != SIGNATURE:
        raise ReaderError(f"{path.name}: first line {lines[0]!r} does not start with ATF")
    version = first[1].strip() if len(first) > 1 else ""
    try:
        n_records, n_columns = (int(f) for f in lines[1].split("\t")[:2])
    except ValueError as exc:
        raise ReaderError(f"{path.name}: line 2 {lines[1]!r} is not two counts") from exc
    if len(lines) < 3 + n_records:
        raise ReaderError(f"{path.name}: header says {n_records} records, file ends early")
    records, signals = _records(lines[2 : 2 + n_records])
    titles = [_unquote(f) for f in lines[2 + n_records].split("\t")]
    titles = [t for t in titles if t] if len(titles) > n_columns else titles
    if len(titles) != n_columns:
        raise ReaderError(f"{path.name}: {len(titles)} column titles, header says {n_columns}")

    body = "\n".join(lines[3 + n_records :])
    try:
        table = np.loadtxt(io.StringIO(body), dtype=np.float64, delimiter="\t", ndmin=2)
    except ValueError as exc:
        raise ReaderError(f"{path.name}: data rows are not all numeric: {exc}") from exc
    if table.shape[0] == 0 or table.shape[1] != n_columns:
        raise ReaderError(
            f"{path.name}: data has {table.shape[1]} columns, header says {n_columns}"
        )

    time_name, time_unit_text = _split_title(titles[0])
    time_unit, time_factor = _unit(time_unit_text)
    if time_name.lower() != "time" or time_unit != "s":
        raise UnsupportedFeatureError(
            f"{path.name}: first column {titles[0]!r} is not time; only time-series ATF "
            "files are supported"
        )
    sample_rate = _round_significant(
        1.0 / (_sample_interval(table[:, 0], path.name) * time_factor),
        INTERVAL_SIGNIFICANT_DIGITS,
    )
    dt = 1.0 / sample_rate

    # Group data columns into (sweep, channel).
    data_titles = titles[1:]
    if signals and len(signals) != len(data_titles):
        raise ReaderError(
            f"{path.name}: Signals= lists {len(signals)} names for {len(data_titles)} columns"
        )
    layout: list[tuple[int, str, str]] = []  # (sweep, channel name, unit text) per column
    for c, title in enumerate(data_titles):
        name, unit_text = _split_title(title)
        trace = TRACE_RE.match(name)
        if trace is not None:
            channel_name = signals[c] if signals else f"Signal {c}"
            layout.append((int(trace["n"]) - 1, channel_name, unit_text))
        else:
            layout.append((0, name, unit_text))
    sweep_ids = sorted({s for s, _, _ in layout})
    channel_order: list[str] = []
    channel_units: dict[str, str] = {}
    for _, name, unit_text in layout:
        if name not in channel_units:
            channel_order.append(name)
            channel_units[name] = unit_text
        elif channel_units[name] != unit_text:
            raise ReaderError(
                f"{path.name}: channel {name!r} has units {channel_units[name]!r} and "
                f"{unit_text!r} in different columns"
            )
    if sweep_ids != list(range(len(sweep_ids))):
        raise ReaderError(f"{path.name}: trace numbers {[s + 1 for s in sweep_ids]} have gaps")
    index: dict[tuple[int, str], int] = {}
    for c, (s, name, _) in enumerate(layout):
        if (s, name) in index:
            raise ReaderError(f"{path.name}: sweep {s + 1} has channel {name!r} twice")
        index[(s, name)] = c + 1
    for s in sweep_ids:
        missing = [n for n in channel_order if (s, n) not in index]
        if missing:
            raise UnsupportedFeatureError(
                f"{path.name}: sweep {s + 1} lacks channels {missing}; ragged exports are "
                "not supported"
            )

    channels: list[Channel] = []
    factors = np.ones(len(channel_order))
    for k, name in enumerate(channel_order):
        unit, factor = _unit(channel_units[name])
        factors[k] = factor
        channels.append(
            Channel(
                name=name,
                unit=unit,
                kind=channel_kind(unit),
                metadata={
                    "file_unit": channel_units[name],
                    "scaling": {"si_factor": factor, "formula": "text_value * si_factor"},
                },
            )
        )

    n_samples = table.shape[0]
    t0s, t0_source = _sweep_starts(records, len(sweep_ids), n_samples * dt)
    sweeps = []
    for s in sweep_ids:
        cols = [index[(s, name)] for name in channel_order]
        data = np.ascontiguousarray(table[:, cols].T) * factors[:, None]
        sweeps.append(Sweep(index=s, data=data, sample_rate=sample_rate, t0=t0s[s]))

    metadata: dict[str, Any] = {
        "format": "atf",
        "format_version": version,
        "source_path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "acquisition_mode": records.get("AcquisitionMode", ""),
        "comment": records.get("Comment", ""),
        "sample_interval_s": dt,
        "time_column_first_s": float(table[0, 0]) * time_factor,
        "total_samples": n_samples * (n_columns - 1),
        "sweep_start_source": t0_source,
        "records": records,
        "signals": signals,
        "column_titles": titles,
    }
    return Recording(channels=tuple(channels), sweeps=tuple(sweeps), metadata=metadata)


def _sweep_starts(
    records: dict[str, str], n_sweeps: int, duration: float
) -> tuple[list[float], str]:
    """Sweep starts relative to the first sweep, from ``SweepStartTimesMS`` when usable."""
    text = records.get("SweepStartTimesMS", "")
    if text:
        try:
            starts_ms = [float(v) for v in text.split(",")]
        except ValueError as exc:
            raise ReaderError(f"SweepStartTimesMS={text!r} is not a list of numbers") from exc
        if len(starts_ms) == n_sweeps:
            return [(v - starts_ms[0]) * 1e-3 for v in starts_ms], "SweepStartTimesMS"
    return [k * duration for k in range(n_sweeps)], "assumed_contiguous"
