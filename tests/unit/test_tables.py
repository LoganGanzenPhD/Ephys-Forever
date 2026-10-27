"""Results tables: unit headers, display conversion, CSV round trip."""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from ephys_forever.core.analysis.measure import RegionStats, measure_region
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.model import Channel
from ephys_forever.export.tables import (
    FIELDS,
    UnitSystem,
    field_unit,
    results_table,
    write_csv,
    write_table,
)

CURRENT = Channel(name="IN 0", unit="A", kind="current")
VOLTAGE = Channel(name="IN 1", unit="V", kind="voltage")


def stats_of(y: np.ndarray) -> list[RegionStats]:
    return [measure_region(y, 1000.0, Region(0.0, 0.099), polarity="negative")]


@pytest.mark.parametrize(
    ("field", "channel", "units", "expected"),
    [
        ("peak", CURRENT, "display", ("pA", 1e-12)),
        ("peak", VOLTAGE, "display", ("mV", 1e-3)),
        ("peak", CURRENT, "si", ("A", 1.0)),
        ("rise_time", CURRENT, "display", ("ms", 1e-3)),
        ("area", CURRENT, "display", ("pA*ms", pytest.approx(1e-15))),
        ("area", VOLTAGE, "si", ("V*s", 1.0)),
        ("n_samples", CURRENT, "display", ("", 1.0)),
    ],
)
def test_field_units(field: str, channel: Channel, units: UnitSystem, expected: object) -> None:
    assert field_unit(field, channel, units) == expected


def test_fields_cover_every_measurement_once() -> None:
    assert sorted(FIELDS) == sorted(f.name for f in dataclasses.fields(RegionStats))


def test_display_table_converts_values_and_labels_headers() -> None:
    y = np.zeros(100)
    y[40:60] = -50e-12
    header, rows = results_table(stats_of(y), CURRENT, sweeps=[3], source="a.abf")
    assert header[:5] == ["file", "sweep", "channel", "peak_amplitude (pA)", "peak_time (ms)"]
    assert len(header) == 3 + len(FIELDS)
    row = dict(zip(header, rows[0], strict=True))
    assert (row["file"], row["sweep"], row["channel"]) == ("a.abf", 4, "IN 0")
    assert row["peak (pA)"] == pytest.approx(-50.0)
    assert row["peak_time (ms)"] == pytest.approx(40.0)
    assert row["area (pA*ms)"] == pytest.approx(-50.0 * 20)
    assert row["n_samples"] == 100


def test_csv_round_trips_values_and_blanks_nan(tmp_path: Path) -> None:
    y = np.linspace(0.0, -1e-10, 100)  # never returns below half, so half-width is NaN
    header, rows = results_table(stats_of(y), CURRENT, units="si")
    out = tmp_path / "results.csv"
    write_csv(out, header, rows)
    with out.open(encoding="utf-8", newline="") as fh:
        back = list(csv.reader(fh))
    assert back[0] == header
    values = dict(zip(header, back[1], strict=True))
    assert values["half_width (s)"] == ""
    assert math.isnan(float(rows[0][header.index("half_width (s)")]))
    assert float(values["peak (A)"]) == rows[0][header.index("peak (A)")]


def test_sweep_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="2 sweep numbers for 1"):
        results_table(stats_of(np.zeros(100)), CURRENT, sweeps=[0, 1])


def nan_table() -> tuple[list[str], list[list[str | int | float]]]:
    y = np.linspace(0.0, -1e-10, 100)  # half-width is NaN
    return results_table(stats_of(y), CURRENT, source="a.abf")


def test_excel_round_trips_values_and_blanks_nan(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    header, rows = nan_table()
    out = tmp_path / "results.xlsx"
    write_table(out, header, rows)
    back = list(openpyxl.load_workbook(out).active.iter_rows(values_only=True))
    assert list(back[0]) == header
    values = dict(zip(header, back[1], strict=True))
    assert values["half_width (ms)"] is None
    assert values["peak (pA)"] == rows[0][header.index("peak (pA)")]
    assert (values["file"], values["sweep"]) == ("a.abf", 1)


def test_parquet_round_trips_with_types_and_nulls(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    header, rows = nan_table()
    out = tmp_path / "results.parquet"
    write_table(out, header, rows)
    table = pq.read_table(out)
    assert table.column_names == header
    assert table.column("half_width (ms)").null_count == 1
    assert table.column("sweep").to_pylist() == [1]
    assert table.column("peak (pA)").to_pylist() == [rows[0][header.index("peak (pA)")]]


def test_unknown_extension_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.txt.*\.csv, \.xlsx, \.parquet"):
        write_table(tmp_path / "x.txt", *nan_table())


def test_fit_table_labels_exponential_units_and_standard_errors() -> None:
    from ephys_forever.core.analysis.fit import exponential, fit_curve
    from ephys_forever.export.tables import fit_results_table

    t = np.arange(2000) / 20_000.0
    y = exponential(1)(t, [-80e-12, 0.01, -5e-12]) + np.random.default_rng(1).normal(
        0, 1e-13, t.size
    )
    result = fit_curve(exponential(1), t + 0.105, y, x_offset=0.105)
    header, rows = fit_results_table([result], CURRENT, sweeps=[4], source="a.abf")
    assert header[:8] == [
        "file",
        "sweep",
        "channel",
        "model",
        "A1 (pA)",
        "A1_se (pA)",
        "tau1 (ms)",
        "tau1_se (ms)",
    ]
    row = dict(zip(header, rows[0], strict=True))
    assert row["sweep"] == 5 and row["model"] == "exponential_1"
    assert row["tau1 (ms)"] == pytest.approx(10.0, rel=1e-2)
    assert row["A1 (pA)"] == pytest.approx(-80.0, rel=1e-2)
    assert 0 < row["tau1_se (ms)"] < 0.1
    assert row["fit_start (ms)"] == pytest.approx(105.0)
    assert row["converged"] == "yes"


def test_fit_table_refuses_mixed_models() -> None:
    from ephys_forever.core.analysis.fit import exponential, fit_curve
    from ephys_forever.export.tables import fit_results_table

    t = np.linspace(0, 1, 50)
    a = fit_curve(exponential(1), t, np.exp(-t / 0.2))
    b = fit_curve(exponential(2), t, np.exp(-t / 0.2) + np.exp(-t / 0.02))
    with pytest.raises(ValueError, match="one model"):
        fit_results_table([a, b], CURRENT)
