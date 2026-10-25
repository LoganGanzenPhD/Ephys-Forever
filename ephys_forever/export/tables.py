"""Results tables of region measurements, in SI or display units, and file output.

Each column header carries its unit, for example ``peak (pA)`` or ``rise_time (ms)``,
so a table is never ambiguous about scale. Values are converted only here; the
measurements themselves stay in SI.

CSV needs nothing extra. Excel (.xlsx) and Parquet need the optional ``export`` extra
(openpyxl and pyarrow), imported only when used.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Literal

from ephys_forever.core.analysis.measure import RegionStats
from ephys_forever.core.model import Channel

UnitSystem = Literal["display", "si"]

DISPLAY_PREFIX: dict[str, tuple[str, float]] = {"A": ("p", 1e-12), "V": ("m", 1e-3)}
"""Display prefix for each SI signal unit; other units are shown in SI."""
TIME_DISPLAY: tuple[str, float] = ("ms", 1e-3)

TIME_FIELDS = frozenset(
    {
        "start_time",
        "stop_time",
        "min_time",
        "max_time",
        "peak_time",
        "rise_low_time",
        "rise_high_time",
        "rise_time",
        "half_width",
    }
)
COUNT_FIELDS = frozenset({"n_samples"})
AREA_FIELDS = frozenset({"area"})
FIELDS: tuple[str, ...] = (
    "peak_amplitude",
    "peak_time",
    "mean_amplitude",
    "area",
    "rise_time",
    "half_width",
    "baseline",
    "peak",
    "mean",
    "sd",
    "minimum",
    "min_time",
    "maximum",
    "max_time",
    "rise_low_time",
    "rise_high_time",
    "start_time",
    "stop_time",
    "n_samples",
)
"""Measurement fields in table order: the usual results first, bookkeeping last."""


def field_unit(field: str, channel: Channel, units: UnitSystem) -> tuple[str, float]:
    """Unit label and the factor that converts a field's SI value into it.

    Parameters
    ----------
    field : str
        A :class:`~ephys_forever.core.analysis.measure.RegionStats` field name.
    channel : Channel
        Channel the measurement came from; gives the signal unit.
    units : {"display", "si"}
        ``"si"`` keeps SI (A, V, s). ``"display"`` uses pA, mV and ms.

    Returns
    -------
    label, divisor : str, float
        Write ``value / divisor`` under a header labelled with ``label`` (empty for
        counts).
    """
    if field in COUNT_FIELDS:
        return "", 1.0
    display = units == "display"
    time_label, time_factor = TIME_DISPLAY if display else ("s", 1.0)
    prefix, factor = DISPLAY_PREFIX.get(channel.unit, ("", 1.0)) if display else ("", 1.0)
    signal_label = f"{prefix}{channel.unit}"
    if field in TIME_FIELDS:
        return time_label, time_factor
    if field in AREA_FIELDS:
        return f"{signal_label}*{time_label}", factor * time_factor
    return signal_label, factor


def results_table(
    stats: Sequence[RegionStats],
    channel: Channel,
    *,
    units: UnitSystem = "display",
    sweeps: Sequence[int] | None = None,
    source: str = "",
) -> tuple[list[str], list[list[str | int | float]]]:
    """Header and rows for a list of region measurements, one row per sweep.

    Parameters
    ----------
    stats : sequence of RegionStats
        Measurements, usually from
        :func:`~ephys_forever.core.analysis.measure.measure_sweeps`.
    channel : Channel
        Channel measured; its name goes in every row and its unit sets the headers.
    units : {"display", "si"}, optional
        Unit system for the values. Default ``"display"`` (pA, mV, ms).
    sweeps : sequence of int or None, optional
        Zero-based sweep index of each entry. Default ``0, 1, 2, ...``. The table shows
        them one-based, as the viewer does.
    source : str, optional
        File name for the ``file`` column.

    Returns
    -------
    header, rows : list of str, list of list
        ``file``, ``sweep``, ``channel`` and then every field of RegionStats with its
        unit in the header. Values are floats (NaN where undefined) or ints.
    """
    indices = list(range(len(stats))) if sweeps is None else list(sweeps)
    if len(indices) != len(stats):
        raise ValueError(f"{len(indices)} sweep numbers for {len(stats)} measurements")
    columns = [(f, *field_unit(f, channel, units)) for f in FIELDS]
    header = ["file", "sweep", "channel"]
    header += [f"{f} ({label})" if label else f for f, label, _ in columns]
    rows: list[list[str | int | float]] = []
    for sweep, s in zip(indices, stats, strict=True):
        row: list[str | int | float] = [source, sweep + 1, channel.name]
        for f, _, divisor in columns:
            value = getattr(s, f)
            row.append(value if f in COUNT_FIELDS else value / divisor)
        rows.append(row)
    return header, rows


def write_csv(
    path: str | PathLike[str],
    header: Sequence[str],
    rows: Sequence[Sequence[str | int | float]],
) -> None:
    """Write a table as UTF-8 CSV. NaN is written as an empty cell.

    Floats are written with ``repr`` precision, so reading the file back gives the same
    numbers.
    """

    def cell(value: str | int | float) -> str | int | float:
        if isinstance(value, float):
            return "" if math.isnan(value) else repr(value)
        return value

    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows([cell(v) for v in row] for row in rows)


EXPORT_HINT = 'install the export extra: pip install -e ".[export]"'


def write_excel(
    path: str | PathLike[str],
    header: Sequence[str],
    rows: Sequence[Sequence[str | int | float]],
    sheet: str = "Measurements",
) -> None:
    """Write a table as an Excel workbook (.xlsx) with one sheet. NaN cells are left empty.

    Raises
    ------
    ImportError
        If openpyxl (the ``export`` extra) is not installed.
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(f"Excel export needs openpyxl; {EXPORT_HINT}") from exc
    book = openpyxl.Workbook()
    ws = book.active
    ws.title = sheet
    ws.append(list(header))
    for row in rows:
        ws.append([None if isinstance(v, float) and math.isnan(v) else v for v in row])
    ws.freeze_panes = "A2"
    book.save(Path(path))


def write_parquet(
    path: str | PathLike[str],
    header: Sequence[str],
    rows: Sequence[Sequence[str | int | float]],
) -> None:
    """Write a table as Parquet, one typed column per header. NaN is stored as null.

    Raises
    ------
    ImportError
        If pyarrow (the ``export`` extra) is not installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(f"Parquet export needs pyarrow; {EXPORT_HINT}") from exc
    columns = {
        name: pa.array([row[c] for row in rows], from_pandas=True) for c, name in enumerate(header)
    }
    pq.write_table(pa.table(columns), Path(path))


WRITERS = {".csv": write_csv, ".xlsx": write_excel, ".parquet": write_parquet}
"""Table writers by file extension."""


def write_table(
    path: str | PathLike[str],
    header: Sequence[str],
    rows: Sequence[Sequence[str | int | float]],
) -> None:
    """Write a table in the format given by the file extension (.csv, .xlsx, .parquet).

    Raises
    ------
    ValueError
        If the extension is not one of :data:`WRITERS`.
    """
    suffix = Path(path).suffix.lower()
    writer = WRITERS.get(suffix)
    if writer is None:
        raise ValueError(
            f"cannot export to {suffix or 'a file without an extension'}; "
            f"use one of {', '.join(WRITERS)}"
        )
    writer(path, header, rows)
