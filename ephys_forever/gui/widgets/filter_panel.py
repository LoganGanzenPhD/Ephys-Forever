"""Filter controls: pick a filter and its settings, apply it to one channel."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6 import QtCore, QtWidgets


def _spin(
    lo: float, hi: float, value: float, suffix: str, decimals: int = 1
) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


class FilterPanel(QtWidgets.QWidget):
    """Filter type, its parameters, Apply and Remove buttons, and the applied list."""

    applyRequested = QtCore.Signal(int, str, dict)  # channel, filter name, params
    resetRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.channel_box = QtWidgets.QComboBox()
        self.type_box = QtWidgets.QComboBox()
        self.pages = QtWidgets.QStackedWidget()

        self.bessel_cutoff = _spin(1.0, 1e6, 1000.0, " Hz")
        self.bessel_order = QtWidgets.QSpinBox()
        self.bessel_order.setRange(1, 10)
        self.bessel_order.setValue(8)
        self.bessel_zero_phase = QtWidgets.QCheckBox("Zero phase (forward and backward)")
        self.bessel_design = QtWidgets.QComboBox()
        self.bessel_design.addItem("Bilinear (-3 dB exactly at cutoff)", "bilinear")
        self.bessel_design.addItem("Matched z (as in common commercial software)", "matched_z")
        self.gaussian_cutoff = _spin(1.0, 1e6, 1000.0, " Hz")
        self.notch_frequency = _spin(1.0, 1e5, 60.0, " Hz")
        self.notch_harmonics = QtWidgets.QSpinBox()
        self.notch_harmonics.setRange(1, 20)
        self.notch_harmonics.setValue(3)
        self.notch_quality = _spin(1.0, 1000.0, 30.0, "")
        self.boxcar_points = QtWidgets.QSpinBox()
        self.boxcar_points.setRange(1, 10001)
        self.boxcar_points.setSingleStep(2)
        self.boxcar_points.setValue(5)

        pages: list[tuple[str, str, list[tuple[str, QtWidgets.QWidget]]]] = [
            (
                "Bessel low-pass",
                "bessel_lowpass",
                [
                    ("-3 dB cutoff", self.bessel_cutoff),
                    ("Poles", self.bessel_order),
                    ("Design", self.bessel_design),
                    ("", self.bessel_zero_phase),
                ],
            ),
            ("Gaussian low-pass", "gaussian_lowpass", [("-3 dB cutoff", self.gaussian_cutoff)]),
            (
                "Notch (line noise)",
                "notch",
                [
                    ("Frequency", self.notch_frequency),
                    ("Harmonics", self.notch_harmonics),
                    ("Quality factor", self.notch_quality),
                ],
            ),
            ("Boxcar average", "boxcar", [("Points (odd)", self.boxcar_points)]),
        ]
        for label, name, rows in pages:
            self.type_box.addItem(label, name)
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.setContentsMargins(0, 0, 0, 0)
            for text, widget in rows:
                form.addRow(text, widget) if text else form.addRow(widget)
            self.pages.addWidget(page)

        self.apply_button = QtWidgets.QPushButton("Apply to all sweeps")
        self.reset_button = QtWidgets.QPushButton("Remove all filters")
        self.reset_button.setEnabled(False)
        self.applied = QtWidgets.QLabel("No filters applied.")
        self.applied.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.addRow("Channel", self.channel_box)
        form.addRow("Filter", self.type_box)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.reset_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.pages)
        layout.addLayout(buttons)
        layout.addWidget(self.applied)
        layout.addStretch(1)

        self.type_box.currentIndexChanged.connect(self.pages.setCurrentIndex)
        self.apply_button.clicked.connect(
            lambda: self.applyRequested.emit(self.channel(), self.name(), self.params())
        )
        self.reset_button.clicked.connect(self.resetRequested.emit)

    def set_channels(self, labels: Sequence[str], default: int = 0) -> None:
        self.channel_box.clear()
        self.channel_box.addItems(list(labels))
        self.channel_box.setCurrentIndex(default)
        self.show_applied([])

    def channel(self) -> int:
        return int(self.channel_box.currentIndex())

    def name(self) -> str:
        return str(self.type_box.currentData())

    def params(self) -> dict[str, Any]:
        """Parameters of the selected filter, as the core filter functions take them."""
        name = self.name()
        if name == "bessel_lowpass":
            return {
                "cutoff_hz": self.bessel_cutoff.value(),
                "order": self.bessel_order.value(),
                "zero_phase": self.bessel_zero_phase.isChecked(),
                "design": str(self.bessel_design.currentData()),
            }
        if name == "gaussian_lowpass":
            return {"cutoff_hz": self.gaussian_cutoff.value()}
        if name == "notch":
            return {
                "frequency_hz": self.notch_frequency.value(),
                "harmonics": self.notch_harmonics.value(),
                "quality": self.notch_quality.value(),
            }
        return {"points": self.boxcar_points.value()}

    def show_applied(self, descriptions: Sequence[str]) -> None:
        text = "\n".join(["Applied, in order:", *descriptions])
        self.applied.setText(text if descriptions else "No filters applied.")
        self.reset_button.setEnabled(bool(descriptions))
