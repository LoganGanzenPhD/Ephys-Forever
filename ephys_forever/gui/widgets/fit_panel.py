"""Curve-fit controls for traces and the table of fit results."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6 import QtCore, QtWidgets

from ephys_forever.gui.widgets.results import ResultsTable

MODELS = (
    ("Exponential, 1 term", "exponential_1"),
    ("Exponential, 2 terms", "exponential_2"),
    ("Exponential, 3 terms", "exponential_3"),
    ("Custom expression", "custom"),
)


class StartValueError(ValueError):
    """Raised for start values that are not ``name=value`` pairs."""


def parse_start_values(text: str) -> dict[str, float]:
    """Parse ``"A=-1e-10, tau=0.01, C=0"`` into a dict. Values are SI."""
    values: dict[str, float] = {}
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        if not sep or not name.strip():
            raise StartValueError(f"start value {part.strip()!r} is not name=value")
        try:
            values[name.strip()] = float(value)
        except ValueError as exc:
            raise StartValueError(f"start value for {name.strip()} is not a number") from exc
    return values


class FitPanel(QtWidgets.QWidget):
    """Model choice, Fit buttons, and the results of the last fit."""

    fitRequested = QtCore.Signal(bool)  # True: every sweep, False: the selected sweep
    exportRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.channel_box = QtWidgets.QComboBox()
        self.model_box = QtWidgets.QComboBox()
        for label, name in MODELS:
            self.model_box.addItem(label, name)
        self.expression = QtWidgets.QLineEdit("A * exp(-x / tau) + C")
        self.expression.setToolTip(
            "x is seconds from the first fitted sample. Allowed: + - * / ^, exp, log, "
            "log10, sqrt, abs, sin, cos, tan, tanh, arctan, pi."
        )
        self.start_values = QtWidgets.QLineEdit("A=-1e-10, tau=0.01, C=0")
        self.start_values.setToolTip("Start values in SI units (A, V, s), as name=value pairs.")
        self.note = QtWidgets.QLabel("Fits the samples in the blue region, both cursors included.")
        self.note.setWordWrap(True)
        self.fit_one = QtWidgets.QPushButton("Fit this sweep")
        self.fit_all = QtWidgets.QPushButton("Fit all sweeps")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.export_button = QtWidgets.QPushButton("Export...")
        self.export_button.setEnabled(False)
        self.table = ResultsTable()

        form = QtWidgets.QFormLayout()
        form.addRow("Channel", self.channel_box)
        form.addRow("Model", self.model_box)
        form.addRow("Expression", self.expression)
        form.addRow("Start values", self.start_values)
        self._custom_rows = (self.expression, self.start_values)
        buttons = QtWidgets.QHBoxLayout()
        for button in (self.fit_one, self.fit_all, self.clear_button, self.export_button):
            buttons.addWidget(button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.note)
        layout.addLayout(buttons)
        layout.addWidget(self.table, stretch=1)
        self._form = form

        self.model_box.currentIndexChanged.connect(self._on_model)
        self.fit_one.clicked.connect(lambda: self.fitRequested.emit(False))
        self.fit_all.clicked.connect(lambda: self.fitRequested.emit(True))
        self.clear_button.clicked.connect(self.clearRequested.emit)
        self.export_button.clicked.connect(self.exportRequested.emit)
        self._on_model()

    def _on_model(self) -> None:
        custom = self.model() == "custom"
        for widget in self._custom_rows:
            widget.setVisible(custom)
            label = self._form.labelForField(widget)
            if label is not None:
                label.setVisible(custom)

    def set_channels(self, labels: Sequence[str], default: int = 0) -> None:
        self.channel_box.clear()
        self.channel_box.addItems(list(labels))
        self.channel_box.setCurrentIndex(default)
        self.clear_results()

    def channel(self) -> int:
        return int(self.channel_box.currentIndex())

    def model(self) -> str:
        return str(self.model_box.currentData())

    def clear_results(self) -> None:
        self.table.clear_results()
        self.export_button.setEnabled(False)

    def show_results(self, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        self.table.show_results(header, rows)
        self.export_button.setEnabled(bool(rows))
