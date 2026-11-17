"""Extract and SVG export dialogs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from kmzviewer.extract import MzRange, describe_targets, parse_targets
from kmzviewer.dad import WavelengthBand, WavelengthExtraction


class Ms2FilesDialog(QDialog):
    """Choose source files independently of chromatogram/spectrum visibility."""
    def __init__(self, files, parent=None, *, selected_file_ids=()):
        super().__init__(parent)
        self.setWindowTitle("Open current files in Kmz2Viewer")
        self.resize(560, 340)
        self.file_list = QListWidget()
        selected = set(selected_file_ids)
        for loaded in files:
            item = QListWidgetItem(loaded.path)
            item.setData(Qt.ItemDataRole.UserRole, loaded.file_id)
            item.setToolTip(loaded.path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if not selected or loaded.file_id in selected
                               else Qt.CheckState.Unchecked)
            self.file_list.addItem(item)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        self.open_button = buttons.button(QDialogButtonBox.StandardButton.Open)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.file_list.itemChanged.connect(lambda _: self._sync_open())
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose one or more files. The new MS/MS session starts with no filters."))
        layout.addWidget(self.file_list)
        layout.addWidget(buttons)
        self._sync_open()

    def checked_file_ids(self):
        return [self.file_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.file_list.count())
                if self.file_list.item(i).checkState() == Qt.CheckState.Checked]

    def _sync_open(self):
        self.open_button.setEnabled(bool(self.checked_file_ids()))

    def accept(self):
        if self.checked_file_ids():
            super().accept()


class ExtractDialog(QDialog):
    def __init__(self, files: list[tuple[int, str]], parent=None, *, selected_file_ids=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract EIC")
        self.setMinimumWidth(460)
        self._target = None

        self.mz_edit = QLineEdit()
        self.mz_edit.setPlaceholderText("273.1559, 301.1412    or    250-300")
        self.ppm = QDoubleSpinBox()
        self.ppm.setRange(0.0, 1_000_000.0)
        self.ppm.setDecimals(4)
        self.ppm.setValue(10.0)
        self.ppm.setSuffix(" ppm")
        self.polarity = QComboBox()
        self.polarity.addItem("All", "all")
        self.polarity.addItem("Positive", "positive")
        self.polarity.addItem("Negative", "negative")
        self.merge = QCheckBox("Merge multiple masses into one chromatogram")
        self.file_list = QListWidget()
        for file_id, name in files:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, file_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            selected = selected_file_ids is None or file_id in selected_file_ids
            item.setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
            self.file_list.addItem(item)

        all_button = QPushButton("All files")
        none_button = QPushButton("No files")
        all_button.clicked.connect(lambda: self._check_files(Qt.CheckState.Checked))
        none_button.clicked.connect(lambda: self._check_files(Qt.CheckState.Unchecked))
        file_buttons = QHBoxLayout()
        file_buttons.addWidget(all_button)
        file_buttons.addWidget(none_button)
        file_buttons.addStretch(1)

        self.note = QLabel(
            "ppm applies to one m/z or a comma-separated list. "
            "An explicit range such as 250-300 sums that interval and does not use ppm. "
            "MS level is MS1."
        )
        self.note.setWordWrap(True)
        self.error = QLabel("")
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #9b2c2c;")

        form = QFormLayout()
        form.addRow("m/z", self.mz_edit)
        form.addRow("Tolerance", self.ppm)
        form.addRow("Polarity", self.polarity)
        form.addRow("MS level", QLabel("MS1"))
        form.addRow("", self.merge)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.mz_edit.returnPressed.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Files"))
        layout.addWidget(self.file_list)
        layout.addLayout(file_buttons)
        layout.addWidget(self.note)
        layout.addWidget(self.error)
        layout.addWidget(buttons)

        self.mz_edit.textChanged.connect(self._refresh_note)
        self.ppm.valueChanged.connect(self._refresh_note)
        self.polarity.currentIndexChanged.connect(self._refresh_note)
        self.merge.toggled.connect(self._refresh_note)
        self.file_list.itemChanged.connect(lambda _item: self._refresh_note())
        self._refresh_note()

    def _check_files(self, state: Qt.CheckState) -> None:
        for index in range(self.file_list.count()):
            self.file_list.item(index).setCheckState(state)

    def checked_file_ids(self) -> list[int]:
        chosen = []
        for index in range(self.file_list.count()):
            item = self.file_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return chosen

    def _refresh_note(self) -> None:
        text = self.mz_edit.text().strip()
        self.error.setText("")
        if not text:
            self.ppm.setEnabled(True)
            self.merge.setEnabled(False)
            self.note.setText(
                "ppm applies to one m/z or a comma-separated list. "
                "An explicit range such as 250-300 sums that interval and does not use ppm. "
                "MS level is MS1."
            )
            return
        try:
            target = parse_targets(text, float(self.ppm.value()))
        except ValueError:
            self.ppm.setEnabled(True)
            self.merge.setEnabled(False)
            return
        if isinstance(target, MzRange):
            self.ppm.setEnabled(False)
            self.merge.setEnabled(False)
            self.merge.setChecked(False)
        else:
            self.ppm.setEnabled(True)
            self.merge.setEnabled(len(target.masses) > 1)
            if len(target.masses) < 2 and self.merge.isChecked():
                self.merge.setChecked(False)
        self.note.setText(
            describe_targets(
                target,
                self.polarity.currentData(),
                self.merge.isChecked(),
                len(self.checked_file_ids()),
            )
        )

    def accept(self) -> None:
        try:
            self._target = parse_targets(self.mz_edit.text(), float(self.ppm.value()))
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        if not self.checked_file_ids():
            self.error.setText("Choose at least one file.")
            return
        super().accept()

    def request(self) -> dict:
        assert self._target is not None
        return {
            "text": self.mz_edit.text(),
            "ppm": float(self.ppm.value()),
            "polarity": self.polarity.currentData(),
            "merge": self.merge.isChecked() and not isinstance(self._target, MzRange),
            "file_ids": self.checked_file_ids(),
        }


class BpcDialog(QDialog):
    def __init__(self, files, parent=None, *, selected_file_ids=None):
        super().__init__(parent)
        self.setWindowTitle("Extract BPC")
        self.setMinimumWidth(460)
        self.polarity = QComboBox()
        for label, value in (("All", "all"), ("Positive", "positive"), ("Negative", "negative")):
            self.polarity.addItem(label, value)
        self.file_list = QListWidget()
        for file_id, name in files:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, file_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if selected_file_ids is None or file_id in selected_file_ids
                               else Qt.CheckState.Unchecked)
            self.file_list.addItem(item)
        file_buttons = QHBoxLayout()
        for label, state in (("All files", Qt.CheckState.Checked), ("No files", Qt.CheckState.Unchecked)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, state=state: self._check_files(state))
            file_buttons.addWidget(button)
        file_buttons.addStretch(1)
        form = QFormLayout()
        form.addRow("Polarity", self.polarity)
        form.addRow("MS level", QLabel("MS1"))
        note = QLabel("One BPC per file: the highest native ion intensity in each selected MS1 scan. "
                      "Empty scans contribute zero. No centroiding or smoothing is applied.")
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.file_list.itemChanged.connect(lambda _: self.ok_button.setEnabled(bool(self.checked_file_ids())))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Files"))
        layout.addWidget(self.file_list)
        layout.addLayout(file_buttons)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.ok_button.setEnabled(bool(self.checked_file_ids()))

    checked_file_ids = ExtractDialog.checked_file_ids
    _check_files = ExtractDialog._check_files

    def accept(self):
        if self.checked_file_ids():
            super().accept()


class WavelengthDialog(QDialog):
    def __init__(self, files, parent=None, *, total=False, selected_file_ids=None):
        super().__init__(parent)
        self.total = total
        self.setWindowTitle('Extract TWC' if total else 'Extract EWC')
        self.setMinimumWidth(460)
        self.wavelength = QLineEdit('254')
        self.wavelength.setPlaceholderText('254, 280')
        def spin(value):
            control = QDoubleSpinBox()
            control.setDecimals(3)
            control.setRange(0, 100000)
            control.setValue(value)
            control.setSuffix(' nm')
            return control
        self.bandwidth = spin(4)
        self.reference = QCheckBox('Subtract reference band')
        self.reference_wavelength = spin(360)
        self.reference_bandwidth = spin(100)
        self.reference_wavelength.setEnabled(False)
        self.reference_bandwidth.setEnabled(False)
        self.reference.toggled.connect(self.reference_wavelength.setEnabled)
        self.reference.toggled.connect(self.reference_bandwidth.setEnabled)
        form = QFormLayout()
        if not total:
            form.addRow('Wavelength(s)', self.wavelength)
            form.addRow('Full bandwidth', self.bandwidth)
            form.addRow(self.reference)
            form.addRow('Reference wavelength', self.reference_wavelength)
            form.addRow('Reference full bandwidth', self.reference_bandwidth)
        self.file_list = QListWidget()
        for file_id, name in files:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, file_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if selected_file_ids is None or file_id in selected_file_ids else Qt.CheckState.Unchecked)
            self.file_list.addItem(item)
        all_button, none_button = QPushButton('All files'), QPushButton('No files')
        all_button.clicked.connect(lambda: self._check_files(Qt.CheckState.Checked))
        none_button.clicked.connect(lambda: self._check_files(Qt.CheckState.Unchecked))
        file_buttons = QHBoxLayout()
        file_buttons.addWidget(all_button)
        file_buttons.addWidget(none_button)
        file_buttons.addStretch()
        note = QLabel('Sum absorbance over all acquired wavelengths.' if total else
                      'Mean absorbance within wavelength ± half bandwidth, minus the optional reference mean. '
                      'Separate wavelengths with commas. Band edges are inclusive; no interpolation. '
                      'Zero bandwidth requires an exact acquired wavelength.')
        note.setWordWrap(True)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet('color: #9b2c2c;')
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel('Files'))
        layout.addWidget(self.file_list)
        layout.addLayout(file_buttons)
        layout.addWidget(note)
        layout.addWidget(self.error)
        layout.addWidget(buttons)

    checked_file_ids = ExtractDialog.checked_file_ids
    _check_files = ExtractDialog._check_files

    def settings(self):
        if self.total:
            return [WavelengthExtraction()]
        reference = WavelengthBand(self.reference_wavelength.value(), self.reference_bandwidth.value()) if self.reference.isChecked() else None
        try:
            wavelengths = list(dict.fromkeys(float(x.strip()) for x in self.wavelength.text().split(',')))
        except ValueError:
            raise ValueError('Enter one or more numeric wavelengths separated by commas.') from None
        return [WavelengthExtraction(WavelengthBand(x, self.bandwidth.value()), reference) for x in wavelengths]

    def accept(self):
        try:
            self.settings()
            if not self.checked_file_ids():
                raise ValueError('Choose at least one file.')
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()


# Re-export for callers of the original dialogs module.
from kmzviewer.gui.svg_preview import ExportDialog
