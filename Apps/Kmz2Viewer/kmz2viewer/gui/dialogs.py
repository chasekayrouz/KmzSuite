"""Formula limits for mass-difference annotations."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from kmz2viewer.formula import FormulaSettings


class FormulaDialog(QDialog):
    def __init__(self, settings: FormulaSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Formula settings")
        self.ppm = self._spin(1, 500, settings.ppm)
        self.carbon = self._spin(0, 40, settings.c_max)
        self.hydrogen = self._spin(0, 80, settings.h_max)
        self.nitrogen = self._spin(0, 20, settings.n_max)
        self.oxygen = self._spin(0, 30, settings.o_max)
        self.sulfur = self._spin(0, 6, settings.s_max)
        self.dbe_min = self._spin(0, 30, settings.dbe_min)
        self.dbe_max = self._spin(1, 40, settings.dbe_max)
        form = QFormLayout()
        form.addRow("ppm tolerance", self.ppm)
        form.addRow("C max", self.carbon)
        form.addRow("H max", self.hydrogen)
        form.addRow("N max", self.nitrogen)
        form.addRow("O max", self.oxygen)
        form.addRow("S max", self.sulfur)
        form.addRow("DBE min", self.dbe_min)
        form.addRow("DBE max", self.dbe_max)
        note = QLabel("DBE runs from the minimum, inclusive, up to but not including the maximum. Default 0 and 15 keeps DBE 0–14.")
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _spin(self, low: int, high: int, value: float) -> QSpinBox:
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(int(value))
        return box

    def settings(self) -> FormulaSettings:
        return FormulaSettings(
            ppm=float(self.ppm.value()),
            c_max=self.carbon.value(),
            h_max=self.hydrogen.value(),
            n_max=self.nitrogen.value(),
            o_max=self.oxygen.value(),
            s_max=self.sulfur.value(),
            dbe_min=self.dbe_min.value(),
            dbe_max=self.dbe_max.value(),
        )
