"""Compact Apply/Clear controls above the chromatograms."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton

from kmzviewer.mz_filter import mz_range


class MzFilterBar(QWidget):
    apply_requested = Signal(float, float)
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start = QLineEdit()
        self.stop = QLineEdit()
        self.apply_button = QPushButton("Apply")
        self.clear_button = QPushButton("Clear")
        row = QHBoxLayout()
        row.addWidget(QLabel("m/z filter"))
        row.addSpacing(8)
        for label, edit in (("Start", self.start), ("Stop", self.stop)):
            caption = QLabel(label)
            caption.setBuddy(edit)
            edit.setAccessibleName(f"m/z filter {label.lower()}")
            edit.setMaximumWidth(110)
            edit.setPlaceholderText(label + " m/z")
            edit.returnPressed.connect(self._apply)
            row.addWidget(caption)
            row.addWidget(edit)
        row.addWidget(self.apply_button)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        self.status = QLabel("No m/z filter.")
        self.status.setWordWrap(True)
        self.setToolTip("Inclusive, display-only ion range. TIC sums, BPC maxima, and EIC window intersections; "
                        "wavelength traces and spectra are unchanged. Clear restores the original traces.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(2)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.apply_button.clicked.connect(self._apply)
        self.clear_button.clicked.connect(self.clear_requested.emit)

    def _apply(self):
        if not self.apply_button.isEnabled():
            return
        try:
            bounds = mz_range(self.start.text(), self.stop.text())
        except (ValueError, OverflowError):
            self.status.setText("Enter finite m/z values with 0 ≤ Start ≤ Stop. The current view is unchanged.")
            return
        self.apply_requested.emit(bounds.lower, bounds.upper)
