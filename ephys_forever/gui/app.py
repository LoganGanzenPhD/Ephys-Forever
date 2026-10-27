"""Desktop viewer: open a recording, browse sweeps and channels, zoom, pan, overlay,
measure between cursors, filter, and fit curves.

Run with ``python -m ephys_forever.gui.app [file]``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ephys_forever import __version__
from ephys_forever.core.analysis.filters import filter_recording
from ephys_forever.core.analysis.fit import (
    BUILT_IN,
    ExpressionError,
    FitError,
    FitResult,
    Model,
    curve,
    custom,
    fit_region,
)
from ephys_forever.core.analysis.measure import Polarity, measure_sweeps
from ephys_forever.core.analysis.regions import RegionError
from ephys_forever.core.io.errors import ReaderError
from ephys_forever.core.io.registry import FILE_FILTERS, read
from ephys_forever.core.log import OperationLog
from ephys_forever.core.model import Recording
from ephys_forever.core.units import convert
from ephys_forever.export.tables import fit_results_table, results_table, write_table
from ephys_forever.gui.views.trace_view import TraceView
from ephys_forever.gui.widgets.filter_panel import FilterPanel
from ephys_forever.gui.widgets.fit_panel import FitPanel, StartValueError, parse_start_values
from ephys_forever.gui.widgets.measure_panel import MeasurePanel
from ephys_forever.gui.widgets.navigator import ChannelList, SweepNavigator

APP_NAME = "Ephys-Forever"
EXPORT_FILTERS = (
    ("CSV (*.csv)", ".csv"),
    ("Excel workbook (*.xlsx)", ".xlsx"),
    ("Parquet (*.parquet)", ".parquet"),
)


def file_dialog_filter() -> str:
    every = " ".join(f"*{ext}" for _, exts in FILE_FILTERS for ext in exts)
    return ";;".join([f"Recordings ({every})", *(name for name, _ in FILE_FILTERS)])


class MainWindow(QtWidgets.QMainWindow):
    """Main window: trace plots in the centre, channels on the left, analysis tabs right."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 800)
        self.recording: Recording | None = None
        self.raw_recording: Recording | None = None
        self.file_name = ""
        self.log = OperationLog()
        self.results: tuple[list[str], list[list[str | int | float]]] | None = None
        self.fit_results: tuple[list[str], list[list[str | int | float]]] | None = None
        self.view = TraceView()
        self.setCentralWidget(self.view)

        self.navigator = SweepNavigator()
        toolbar = self.addToolBar("Sweeps")
        toolbar.setObjectName("sweeps")
        toolbar.addWidget(self.navigator)
        autoscale = toolbar.addAction("Autoscale")
        autoscale.setToolTip("Fit all plots to their data (A)")
        autoscale.setShortcut(QtGui.QKeySequence("A"))
        autoscale.triggered.connect(self.view.auto_range)

        self.channel_list = ChannelList()
        dock = QtWidgets.QDockWidget("Channels", self)
        dock.setObjectName("channels")
        dock.setWidget(self.channel_list)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self.measure_panel = MeasurePanel()
        self.fit_panel = FitPanel()
        self.filter_panel = FilterPanel()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.measure_panel, "Measure")
        self.tabs.addTab(self.fit_panel, "Fit")
        self.tabs.addTab(self.filter_panel, "Filter")
        analysis_dock = QtWidgets.QDockWidget("Analysis", self)
        analysis_dock.setObjectName("analysis")
        analysis_dock.setWidget(self.tabs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, analysis_dock)

        menu = self.menuBar().addMenu("&File")
        open_action = menu.addAction("&Open...")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.choose_file)
        save_log = menu.addAction("Save operation &log...")
        save_log.triggered.connect(self.choose_log_file)
        quit_action = menu.addAction("&Quit")
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        for key, step in (
            (QtCore.Qt.Key.Key_Right, 1),
            (QtCore.Qt.Key.Key_Left, -1),
            (QtCore.Qt.Key.Key_PageDown, 1),
            (QtCore.Qt.Key.Key_PageUp, -1),
        ):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(lambda s=step: self.navigator.spin.stepBy(s))

        self.navigator.sweepChanged.connect(self._on_sweep)
        self.navigator.overlayChanged.connect(self._on_overlay)
        self.channel_list.channelsChanged.connect(self._on_channels)
        self.view.regionsChanged.connect(
            lambda name: self.measure_panel.set_region(name, self.view.regions[name])
        )
        self.measure_panel.regionEdited.connect(self.view.set_region)
        self.measure_panel.baselineToggled.connect(self.view.set_baseline_visible)
        self.measure_panel.measureRequested.connect(self.run_measurement)
        self.measure_panel.exportRequested.connect(lambda: self.choose_export_file("measure"))
        self.fit_panel.fitRequested.connect(self.run_fit)
        self.fit_panel.clearRequested.connect(self.clear_fits)
        self.fit_panel.exportRequested.connect(lambda: self.choose_export_file("fit"))
        self.filter_panel.applyRequested.connect(self.apply_filter)
        self.filter_panel.resetRequested.connect(self.remove_filters)
        self.statusBar().showMessage("Open a recording with File > Open (Ctrl+O)")

    def choose_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open recording", "", file_dialog_filter()
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        """Load ``path``; on failure show the reader's message and keep the current file."""
        try:
            recording = read(path)
        except (OSError, ReaderError) as exc:
            self.show_error(f"Could not open {Path(path).name}", str(exc))
            return False
        self.set_recording(recording, Path(path).name)
        return True

    def set_recording(self, recording: Recording, title: str = "") -> None:
        self.recording = recording
        self.raw_recording = recording
        self.file_name = title
        self.results = None
        self.fit_results = None
        self.view.set_recording(recording)
        self.navigator.set_count(recording.n_sweeps)
        labels = [f"{c.name} ({c.metadata.get('file_unit') or c.unit})" for c in recording.channels]
        self.channel_list.set_channels(labels)
        currents = [i for i, c in enumerate(recording.channels) if c.kind == "current"]
        default = currents[0] if currents else 0
        for panel in (self.measure_panel, self.fit_panel, self.filter_panel):
            panel.set_channels(labels, default)
        for name, region in self.view.regions.items():
            self.measure_panel.set_region(name, region)
        if title:
            self.setWindowTitle(f"{title} - {APP_NAME}")
        rate_khz = convert(recording.sample_rate, "Hz", "kHz")
        self.statusBar().showMessage(
            f"{recording.n_sweeps} sweep{'s' * (recording.n_sweeps != 1)}, "
            f"{recording.n_channels} channel{'s' * (recording.n_channels != 1)}, "
            f"{rate_khz:g} kHz, {recording.sweeps[0].duration:g} s per sweep"
        )

    def run_measurement(self) -> bool:
        """Measure the cursor region in every sweep, log the run, and show the results."""
        rec = self.recording
        if rec is None:
            return False
        panel = self.measure_panel
        measure = self.view.regions["measure"]
        baseline = self.view.regions["baseline"] if panel.use_baseline() else None
        polarity = cast(Polarity, panel.polarity())
        method = "median" if panel.method() == "median" else "mean"
        channel = panel.channel()
        try:
            stats = measure_sweeps(
                rec,
                channel,
                measure,
                polarity=polarity,
                baseline_region=baseline,
                baseline_method=method,
            )
        except RegionError as exc:
            self.show_error("Cannot measure", str(exc))
            return False
        self.log.record(
            "ephys_forever.core.analysis.measure.measure_sweeps",
            {
                "channel": channel,
                "channel_name": rec.channels[channel].name,
                "region": [measure.start, measure.stop],
                "polarity": polarity,
                "baseline_region": None if baseline is None else [baseline.start, baseline.stop],
                "baseline_method": method,
                "rise_fractions": [0.1, 0.9],
                "filters": list(rec.metadata.get("filters", ())),
            },
            input_sha256=str(rec.metadata.get("sha256", "")),
            input_name=self.file_name,
        )
        self.results = results_table(stats, rec.channels[channel], source=self.file_name)
        panel.show_results(*self.results)
        return True

    # -- fitting ----------------------------------------------------------------------
    def _fit_model(self) -> tuple[Model, dict[str, float] | None]:
        """The chosen model and, for custom expressions, the typed start values."""
        name = self.fit_panel.model()
        if name != "custom":
            return BUILT_IN[name](), None
        model = custom(self.fit_panel.expression.text())
        return model, parse_start_values(self.fit_panel.start_values.text())

    def run_fit(self, all_sweeps: bool) -> bool:
        """Fit the blue region of the selected sweep (or every sweep), log, and draw."""
        rec = self.recording
        if rec is None:
            return False
        channel = self.fit_panel.channel()
        region = self.view.regions["measure"]
        sweeps = list(range(rec.n_sweeps)) if all_sweeps else [self.view.sweep]
        try:
            model, p0 = self._fit_model()
            results: list[FitResult] = [
                fit_region(model, rec.sweeps[s].data[channel], rec.sample_rate, region, p0)
                for s in sweeps
            ]
        except (ExpressionError, StartValueError, FitError, RegionError, ValueError) as exc:
            self.show_error("Cannot fit", str(exc))
            return False
        self.log.record(
            "ephys_forever.core.analysis.fit.fit_region",
            {
                "model": model.name,
                "formula": model.formula,
                "p0": p0,
                "channel": channel,
                "channel_name": rec.channels[channel].name,
                "region": [region.start, region.stop],
                "sweeps": sweeps,
                "filters": list(rec.metadata.get("filters", ())),
            },
            input_sha256=str(rec.metadata.get("sha256", "")),
            input_name=self.file_name,
        )
        fits = {}
        for s, result in zip(sweeps, results, strict=True):
            i0, i1 = region.indices(rec.sample_rate, rec.sweeps[s].n_samples)
            t = np.arange(i0, i1, dtype=np.float64) / rec.sample_rate
            fits[s] = (t, curve(model, result, t))
        self.view.set_fits(channel, fits)
        self.fit_results = fit_results_table(
            results, rec.channels[channel], sweeps=sweeps, source=self.file_name
        )
        self.fit_panel.show_results(*self.fit_results)
        if not all(r.converged for r in results):
            self.statusBar().showMessage("Some fits did not converge; see the converged column")
        return True

    def clear_fits(self) -> None:
        self.view.clear_fits()
        self.fit_results = None
        self.fit_panel.clear_results()

    # -- filtering ----------------------------------------------------------------------
    def apply_filter(self, channel: int, name: str, params: dict[str, object]) -> bool:
        """Filter one channel of every sweep of the displayed data, and log it."""
        rec = self.recording
        if rec is None:
            return False
        try:
            filtered = filter_recording(rec, channel, name, **params)
        except (ValueError, KeyError, IndexError) as exc:
            self.show_error("Cannot filter", str(exc))
            return False
        self.log.record(
            "ephys_forever.core.analysis.filters.filter_recording",
            {
                "channel": channel,
                "channel_name": rec.channels[channel].name,
                "name": name,
                "params": params,
            },
            input_sha256=str(rec.metadata.get("sha256", "")),
            input_name=self.file_name,
        )
        self._show_filtered(filtered)
        return True

    def remove_filters(self) -> None:
        """Go back to the data as read from the file. Not logged: every logged analysis
        lists the filters it ran on, so the log already shows the data each one used."""
        if self.raw_recording is not None and self.recording is not self.raw_recording:
            self._show_filtered(self.raw_recording)

    def _show_filtered(self, recording: Recording) -> None:
        self.recording = recording
        self.view.replace_data(recording)
        self.clear_fits()
        self.results = None
        self.measure_panel.clear_results()
        self.filter_panel.show_applied(
            [
                f"{recording.channels[f['channel']].name}: {f['name']} "
                + ", ".join(f"{k}={v}" for k, v in f["params"].items())
                for f in recording.metadata.get("filters", ())
            ]
        )

    # -- export -------------------------------------------------------------------------
    def choose_export_file(self, which: str = "measure") -> None:
        stem = Path(self.file_name).stem or "results"
        suffix = "fits" if which == "fit" else "measurements"
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export results",
            f"{stem}_{suffix}.csv",
            ";;".join(name for name, _ in EXPORT_FILTERS),
        )
        if not path:
            return
        if not Path(path).suffix:
            path += dict(EXPORT_FILTERS).get(chosen, ".csv")
        self.export_results(path, which)

    def export_results(self, path: str | Path, which: str = "measure") -> bool:
        """Write the measurement (``"measure"``) or fit (``"fit"``) results to ``path``."""
        table = self.fit_results if which == "fit" else self.results
        if table is None:
            return False
        try:
            write_table(path, *table)
        except (OSError, ImportError, ValueError) as exc:
            self.show_error("Could not export results", str(exc))
            return False
        self.statusBar().showMessage(f"Results written to {path}")
        return True

    def choose_log_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save operation log", "operations.json", "Operation log (*.json)"
        )
        if path:
            try:
                self.log.save(path)
            except OSError as exc:
                self.show_error("Could not save the operation log", str(exc))

    def show_error(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, title, message)

    def _on_sweep(self, index: int) -> None:
        if self.recording is not None:
            self.view.show_sweep(index)

    def _on_overlay(self, on: bool) -> None:
        if self.recording is not None:
            self.view.set_overlay(on)

    def _on_channels(self, channels: list[int]) -> None:
        if self.recording is not None:
            self.view.set_channels(channels)


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(background="w", foreground="k", antialias=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(args)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    configure_pyqtgraph()
    window = MainWindow()
    window.show()
    if len(args) > 1:
        window.open_path(args[1])
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
