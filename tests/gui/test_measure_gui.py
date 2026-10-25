"""Cursor regions, the measure panel, results table, CSV export and the operation log."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ephys_forever.core.analysis.measure import measure_sweeps
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.io.atf import read_atf
from ephys_forever.core.model import Channel, Recording, Sweep
from ephys_forever.gui.app import MainWindow

pytestmark = pytest.mark.gui

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
STEP = SYNTHETIC / "step_known_amplitude.atf"


def two_channels() -> Recording:
    rec = read_atf(STEP)
    channels = (rec.channels[0], Channel(name="IN1", unit="V", kind="voltage"))
    sweeps = tuple(
        Sweep(index=s.index, data=np.vstack([s.data, s.data]), sample_rate=s.sample_rate, t0=s.t0)
        for s in rec.sweeps
    )
    return Recording(channels=channels, sweeps=sweeps, metadata=dict(rec.metadata))


@pytest.fixture
def window(qapp: Any) -> Iterator[MainWindow]:
    win = MainWindow()
    win.show()
    win.open_path(STEP)
    qapp.processEvents()
    yield win
    win.close()


def test_regions_start_inside_the_sweep_and_fill_the_panel(window: MainWindow) -> None:
    regions = window.view.regions
    assert regions["baseline"] == Region(0.0, 0.05)
    assert regions["measure"] == Region(0.2, 0.3)
    box = window.measure_panel.region_boxes["measure"]
    assert (box.start.value(), box.stop.value()) == pytest.approx((200.0, 300.0))


def test_dragging_a_region_in_one_row_moves_it_everywhere(window: MainWindow) -> None:
    window.set_recording(two_channels(), "two")
    item = window.view.region_items[0]["measure"]
    item.setRegion((0.11, 0.2))
    moved = window.view.regions["measure"]
    assert (moved.start, moved.stop) == pytest.approx((0.11, 0.2))
    assert window.view.region_items[1]["measure"].getRegion() == pytest.approx((0.11, 0.2))
    box = window.measure_panel.region_boxes["measure"]
    assert box.start.value() == pytest.approx(110.0)


def test_typed_region_moves_the_cursors(window: MainWindow) -> None:
    box = window.measure_panel.region_boxes["baseline"]
    box.start.setValue(10.0)
    box.stop.setValue(90.0)
    box.edited.emit(box.region())
    assert window.view.regions["baseline"] == Region(0.01, 0.09)
    lo, hi = window.view.region_items[0]["baseline"].getRegion()
    assert (lo, hi) == pytest.approx((0.01, 0.09))


def test_measure_fills_the_table_logs_and_exports(window: MainWindow, tmp_path: Path) -> None:
    window.view.set_region("baseline", Region(0.0, 0.09995))
    window.view.set_region("measure", Region(0.1, 0.29995))
    assert window.run_measurement()
    rec = window.recording
    assert rec is not None
    expected = measure_sweeps(
        rec, 0, Region(0.1, 0.29995), polarity="negative", baseline_region=Region(0.0, 0.09995)
    )
    table = window.measure_panel.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert table.rowCount() == 5
    peak = headers.index("peak_amplitude (pA)")
    for r, s in enumerate(expected):
        assert float(table.item(r, peak).text()) == pytest.approx(s.peak_amplitude / 1e-12, 1e-5)

    entry = window.log.entries[-1]
    assert entry.function.endswith("measure_sweeps")
    assert entry.params["region"] == [0.1, 0.29995]
    assert entry.params["baseline_region"] == [0.0, 0.09995]
    assert entry.input_sha256 == rec.metadata["sha256"]

    out = tmp_path / "m.csv"
    assert window.export_results(out)
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and rows[0]["file"] == STEP.name
    assert float(rows[2]["mean (pA)"]) == pytest.approx(expected[2].mean / 1e-12)
    for suffix in (".xlsx", ".parquet"):
        assert window.export_results(tmp_path / f"m{suffix}")
        assert (tmp_path / f"m{suffix}").stat().st_size > 0

    log_path = tmp_path / "ops.json"
    window.log.save(log_path)
    assert "measure_sweeps" in log_path.read_text(encoding="utf-8")


def test_without_baseline_measures_from_zero_and_hides_the_region(window: MainWindow) -> None:
    window.measure_panel.baseline_check.setChecked(False)
    assert not window.view.region_items[0]["baseline"].isVisible()
    assert window.run_measurement()
    assert window.log.entries[-1].params["baseline_region"] is None


def test_region_outside_the_sweep_reports_an_error(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
    window.view.set_region("measure", Region(0.4, 0.9))
    assert not window.run_measurement()
    assert errors and "after the sweep's last sample" in errors[0][1]
    assert len(window.log) == 0


def test_export_to_an_unknown_format_reports_an_error(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
    assert window.run_measurement()
    assert not window.export_results(tmp_path / "m.txt")
    assert errors and ".txt" in errors[0][1]
