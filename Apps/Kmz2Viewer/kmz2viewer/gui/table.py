"""Sortable table of MS2 scans."""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import QTableView

from kmz2viewer.model import Ms2Scan

COLUMNS = (
    ("File", "file_name"),
    ("Scan", "scan_id"),
    ("RT (min)", "rt_min"),
    ("Precursor m/z", "precursor_mz"),
    ("Charge", "charge"),
    ("Precursor intensity", "precursor_intensity"),
    ("Collision energy", "collision_energy"),
    ("Base peak m/z", "base_peak_mz"),
    ("Base peak intensity", "base_peak_intensity"),
    ("TIC", "tic"),
    ("Peaks", "n_peaks"),
)


def _display(scan: Ms2Scan, field: str) -> str:
    value = getattr(scan, field)
    if value is None:
        return ""
    if field == "rt_min":
        return "" if value != value else f"{value:.4f}"
    if field in {"precursor_mz", "base_peak_mz"}:
        return f"{float(value):.4f}"
    if field in {"precursor_intensity", "base_peak_intensity", "tic", "collision_energy"}:
        return f"{float(value):.6g}"
    return str(value)


def _sort_value(scan: Ms2Scan, field: str):
    value = getattr(scan, field)
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value


class ScanTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[Ms2Scan] = []

    def set_rows(self, rows: list[Ms2Scan]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def scan_at(self, row: int) -> Ms2Scan | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        if 0 <= section < len(COLUMNS):
            return COLUMNS[section][0]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        scan = self._rows[index.row()]
        field = COLUMNS[index.column()][1]
        if role == Qt.ItemDataRole.DisplayRole:
            return _display(scan, field)
        if role == Qt.ItemDataRole.UserRole:
            return _sort_value(scan, field)
        return None


class ScanTable(QTableView):
    open_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.doubleClicked.connect(lambda _index: self.open_requested.emit())

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.open_requested.emit()
            return
        super().keyPressEvent(event)
