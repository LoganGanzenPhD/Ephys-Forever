"""Cursor-region measurement controls and the results table."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6 import QtCore, QtWidgets

from ephys_forever.core.analysis.regions import Region
from ephys_forever.gui.widgets.results import ResultsTable

POLARITIES = (
    ("Negative (inward current)", "negative"),
    ("Positive", "positive"),
    ("Either direction", "absolute"),
)
BASELINE_METHODS = (("Mean", "mean"), ("Median", "median"))
MS = 1e-3
"""Region boxes show milliseconds; regions are stored in seconds."""


class RegionBox(QtWidgets.QWidget):
    """Start and end of one cursor region, in ms."""

    edited = QtCore.Signal(object)  # Region

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.start = self._spin()
        self.stop = self._spin()
        layout.addWidget(self.start)
        layout.addWidget(QtWidgets.QLabel("to"))
        layout.addWidget(self.stop)
        for spin in (self.start, self.stop):
            spin.editingFinished.connect(self._emit)

    @staticmethod
    def _spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(0.0, 1e9)
        spin.setSuffix(" ms")
        spin.setKeyboardTracking(False)
        return spin

    def set_region(self, region: Region) -> None:
        for spin, value in ((self.start, region.start), (self.stop, region.stop)):
            spin.blockSignals(True)
            spin.setValue(value / MS)
            spin.blockSignals(False)

    def region(self) -> Region:
        return Region(self.start.value() * MS, self.stop.value() * MS)

    def _emit(self) -> None:
        if self.stop.value() > self.start.value():
            self.edited.emit(self.region())


class MeasurePanel(QtWidgets.QWidget):
    """Measurement settings, a Measure button, and the results of the last run."""

    regionEdited = QtCore.Signal(str, object)  # name, Region
    baselineToggled = QtCore.Signal(bool)
    measureRequested = QtCore.Signal()
    exportRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.channel_box = QtWidgets.QComboBox()
        self.polarity_box = QtWidgets.QComboBox()
        for label, value in POLARITIES:
            self.polarity_box.addItem(label, value)
        self.baseline_check = QtWidgets.QCheckBox("Measure from baseline")
        self.baseline_check.setChecked(True)
        self.baseline_method = QtWidgets.QComboBox()
        for label, value in BASELINE_METHODS:
            self.baseline_method.addItem(label, value)
        self.region_boxes = {"baseline": RegionBox(), "measure": RegionBox()}
        self.measure_button = QtWidgets.QPushButton("Measure all sweeps")
        self.export_button = QtWidgets.QPushButton("Export...")
        self.export_button.setEnabled(False)
        self.table = ResultsTable()

        form = QtWidgets.QFormLayout()
        form.addRow("Channel", self.channel_box)
        form.addRow("Peak", self.polarity_box)
        form.addRow(self.baseline_check)
        form.addRow("Baseline", self.region_boxes["baseline"])
        form.addRow("Baseline level", self.baseline_method)
        form.addRow("Region", self.region_boxes["measure"])
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.measure_button)
        buttons.addWidget(self.export_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.table, stretch=1)

        for name, box in self.region_boxes.items():
            box.edited.connect(lambda r, n=name: self.regionEdited.emit(n, r))
        self.baseline_check.toggled.connect(self._on_baseline_toggled)
        self.measure_button.clicked.connect(self.measureRequested.emit)
        self.export_button.clicked.connect(self.exportRequested.emit)

    # -- settings -------------------------------------------------------------------
    def set_channels(self, labels: Sequence[str], default: int = 0) -> None:
        self.channel_box.clear()
        self.channel_box.addItems(list(labels))
        self.channel_box.setCurrentIndex(default)
        self.clear_results()

    def set_region(self, name: str, region: Region) -> None:
        self.region_boxes[name].set_region(region)

    def channel(self) -> int:
        return int(self.channel_box.currentIndex())

    def polarity(self) -> str:
        return str(self.polarity_box.currentData())

    def use_baseline(self) -> bool:
        return bool(self.baseline_check.isChecked())

    def method(self) -> str:
        return str(self.baseline_method.currentData())

    def _on_baseline_toggled(self, on: bool) -> None:
        self.region_boxes["baseline"].setEnabled(on)
        self.baseline_method.setEnabled(on)
        self.baselineToggled.emit(on)

    # -- results --------------------------------------------------------------------
    def clear_results(self) -> None:
        self.table.clear_results()
        self.export_button.setEnabled(False)

    def show_results(self, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        self.table.show_results(header, rows)
        self.export_button.setEnabled(bool(rows))
