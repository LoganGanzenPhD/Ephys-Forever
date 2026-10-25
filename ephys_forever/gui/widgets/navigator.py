"""Sweep navigation and channel visibility controls."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6 import QtCore, QtWidgets


class SweepNavigator(QtWidgets.QWidget):
    """Previous/next buttons, a sweep number box and an overlay switch."""

    sweepChanged = QtCore.Signal(int)
    overlayChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.prev_button = QtWidgets.QToolButton(text="◀")
        self.prev_button.setToolTip("Previous sweep")
        self.next_button = QtWidgets.QToolButton(text="▶")
        self.next_button.setToolTip("Next sweep")
        self.spin = QtWidgets.QSpinBox()
        self.spin.setPrefix("Sweep ")
        self.count_label = QtWidgets.QLabel()
        self.overlay_box = QtWidgets.QCheckBox("Overlay all sweeps")
        for w in (self.prev_button, self.spin, self.next_button, self.count_label):
            layout.addWidget(w)
        layout.addSpacing(12)
        layout.addWidget(self.overlay_box)
        self.prev_button.clicked.connect(lambda: self.spin.stepBy(-1))
        self.next_button.clicked.connect(lambda: self.spin.stepBy(1))
        self.spin.valueChanged.connect(lambda v: self.sweepChanged.emit(v - 1))
        self.overlay_box.toggled.connect(self.overlayChanged.emit)
        self.set_count(0)

    def set_count(self, n_sweeps: int) -> None:
        """Reset to sweep 1 of ``n_sweeps``; disables the controls for one sweep or none."""
        self.spin.blockSignals(True)
        self.spin.setRange(1, max(n_sweeps, 1))
        self.spin.setValue(1)
        self.spin.blockSignals(False)
        self.overlay_box.blockSignals(True)
        self.overlay_box.setChecked(False)
        self.overlay_box.blockSignals(False)
        self.count_label.setText(f"of {n_sweeps}")
        many = n_sweeps > 1
        for w in (self.prev_button, self.next_button, self.spin, self.overlay_box):
            w.setEnabled(many)

    def current(self) -> int:
        """Zero-based selected sweep."""
        return int(self.spin.value()) - 1


class ChannelList(QtWidgets.QListWidget):
    """Checkable list of channels; emits the checked indices when they change."""

    channelsChanged = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.itemChanged.connect(lambda _item: self.channelsChanged.emit(self.checked()))

    def set_channels(self, labels: Sequence[str]) -> None:
        self.blockSignals(True)
        self.clear()
        for label in labels:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.addItem(item)
        self.blockSignals(False)

    def checked(self) -> list[int]:
        return [
            i
            for i in range(self.count())
            if self.item(i).checkState() == QtCore.Qt.CheckState.Checked
        ]

    def set_checked(self, index: int, on: bool) -> None:
        state = QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked
        self.item(index).setCheckState(state)
