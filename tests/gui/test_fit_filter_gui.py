"""Fit and Filter tabs: fitting the blue region, drawing fits, filtering, logging."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ephys_forever.core.analysis.filters import boxcar
from ephys_forever.core.analysis.regions import Region
from ephys_forever.gui.app import MainWindow
from ephys_forever.gui.widgets.fit_panel import StartValueError, parse_start_values

pytestmark = pytest.mark.gui

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.fixture
def window(qapp: Any) -> Iterator[MainWindow]:
    win = MainWindow()
    win.show()
    win.open_path(SYNTHETIC / "exp_decay_single.atf")
    win.view.set_region("measure", Region(0.0, 0.2))
    qapp.processEvents()
    yield win
    win.close()


def table_column(window: MainWindow, name: str) -> list[str]:
    table = window.fit_panel.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    c = headers.index(name)
    return [table.item(r, c).text() for r in range(table.rowCount())]


def test_fit_all_sweeps_fills_the_table_draws_and_logs(window: MainWindow, tmp_path: Path) -> None:
    assert window.run_fit(all_sweeps=True)
    taus = [float(v) for v in table_column(window, "tau1 (ms)")]
    assert taus == pytest.approx([20.0, 20.0], rel=1e-2)
    assert set(window.view.fits[0]) == {0, 1}
    assert len(window.view.fit_items) == 1  # only the shown sweep is drawn
    window.navigator.overlay_box.setChecked(True)
    assert len(window.view.fit_items) == 2
    entry = window.log.entries[-1]
    assert entry.function.endswith("fit_region")
    assert entry.params["model"] == "exponential_1" and entry.params["sweeps"] == [0, 1]
    out = tmp_path / "fits.csv"
    assert window.export_results(out, "fit")
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[0]["tau1 (ms)"]) == pytest.approx(20.0, rel=1e-6)


def test_fit_this_sweep_and_clear(window: MainWindow) -> None:
    window.navigator.spin.setValue(2)
    assert window.run_fit(all_sweeps=False)
    assert list(window.view.fits[0]) == [1]
    assert table_column(window, "sweep") == ["2"]
    window.clear_fits()
    assert window.view.fit_items == [] and window.fit_panel.table.rowCount() == 0


def test_custom_expression_fit_and_its_errors(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
    panel = window.fit_panel
    panel.model_box.setCurrentIndex(panel.model_box.findData("custom"))
    panel.expression.setText("A * exp(-x / tau) + C")
    panel.start_values.setText("A=5e-11, tau=0.05, C=0")
    assert window.run_fit(all_sweeps=False)
    assert float(table_column(window, "tau")[0]) == pytest.approx(0.02, rel=1e-6)  # SI, unlabelled
    panel.expression.setText("A * exp(-x / tau")
    assert not window.run_fit(all_sweeps=False)
    panel.expression.setText("A * exp(-x / tau) + C")
    panel.start_values.setText("A=5e-11, tau")
    assert not window.run_fit(all_sweeps=False)
    assert [t for t, _ in errors] == ["Cannot fit", "Cannot fit"]
    assert "cannot parse" in errors[0][1] and "name=value" in errors[1][1]


def test_parse_start_values() -> None:
    assert parse_start_values(" A=1e-10; tau = 0.01,, ") == {"A": 1e-10, "tau": 0.01}
    with pytest.raises(StartValueError, match="not a number"):
        parse_start_values("A=x")


def test_filter_applies_to_every_sweep_logs_and_can_be_removed(window: MainWindow) -> None:
    raw = window.recording
    assert raw is not None
    assert window.apply_filter(0, "boxcar", {"points": 9})
    rec = window.recording
    assert rec is not None and rec is not raw and window.raw_recording is raw
    for k in range(raw.n_sweeps):
        np.testing.assert_array_equal(rec.sweeps[k].data[0], boxcar(raw.sweeps[k].data[0], 9))
    assert "boxcar points=9" in window.filter_panel.applied.text()
    assert window.log.entries[-1].function.endswith("filter_recording")

    assert window.run_measurement()
    assert window.log.entries[-1].params["filters"][0]["name"] == "boxcar"

    window.remove_filters()
    assert window.recording is raw
    assert window.filter_panel.applied.text() == "No filters applied."
    assert window.results is None  # results from filtered data are not kept


def test_bad_filter_settings_report_an_error(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
    assert not window.apply_filter(0, "bessel_lowpass", {"cutoff_hz": 15_000.0, "order": 8})
    assert errors and "Nyquist" in errors[0][1]
    assert window.recording is window.raw_recording


def test_filter_panel_params_follow_the_chosen_filter(window: MainWindow) -> None:
    panel = window.filter_panel
    assert panel.name() == "bessel_lowpass"
    assert panel.params() == {
        "cutoff_hz": 1000.0,
        "order": 8,
        "zero_phase": False,
        "design": "bilinear",
    }
    panel.type_box.setCurrentIndex(panel.type_box.findData("notch"))
    assert panel.params() == {"frequency_hz": 60.0, "harmonics": 3, "quality": 30.0}
    emitted: list[tuple[int, str, dict[str, Any]]] = []
    panel.applyRequested.connect(lambda c, n, p: emitted.append((c, n, p)))
    panel.apply_button.click()
    assert emitted[0][1] == "notch"
