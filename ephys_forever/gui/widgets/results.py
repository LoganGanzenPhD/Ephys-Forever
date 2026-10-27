"""Read-only table of analysis results."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6 import QtWidgets

HIDDEN_COLUMNS = ("file", "channel", "model")
"""Columns the table leaves out because every row has the same value; export keeps them."""


class ResultsTable(QtWidgets.QTableWidget):
    """Rows of results; floats shown to 6 significant digits and NaN as blank."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)

    def clear_results(self) -> None:
        self.clear()
        self.setRowCount(0)
        self.setColumnCount(0)

    def show_results(self, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        self.clear()
        self.setColumnCount(len(header))
        self.setRowCount(len(rows))
        self.setHorizontalHeaderLabels(list(header))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if isinstance(value, float):
                    text = "" if math.isnan(value) else f"{value:.6g}"
                else:
                    text = str(value)
                self.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        for c, name in enumerate(header):
            self.setColumnHidden(c, name in HIDDEN_COLUMNS)
        self.resizeColumnsToContents()
