"""Secondary settings and analysis detail views."""
from dataclasses import asdict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QLabel, QMessageBox, QSpinBox, QTabWidget, QVBoxLayout)

from kmzcomet.model import Settings
from .table import MatrixModel, MeasurementsModel, RowsModel, Table, run_rows


def benchmark_settings():
    return Settings(mz_ppm=30, discovery_window=.35, rt_window=.15, min_contrast=2, min_height=100)


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Analysis Settings')
        self.widgets = {}
        layout = QVBoxLayout(self)
        note = QLabel('Settings apply to the next run. Saved analyses retain their original settings.')
        note.setWordWrap(True); layout.addWidget(note)
        form = QFormLayout(); layout.addLayout(form)
        labels = {'mz_ppm':'Mass tolerance (±ppm)', 'rt_tolerance':'RT matching tolerance (min)',
                  'rt_window':'Measurement half-window (min)', 'discovery_window':'Discovery half-window (min)',
                  'discovery_height':'Discovery height', 'min_height':'Evidence height floor',
                  'min_points':'Minimum supporting scans', 'min_snr':'Minimum SNR',
                  'min_contrast':'Minimum contrast', 'max_scan_gap':'Maximum scan gap (min)',
                  'max_rt_shift':'Maximum RT shift (min)', 'min_anchors':'Minimum RT anchors',
                  'alignment':'RT alignment'}
        for key,value in asdict(settings).items():
            if key == 'alignment':
                widget = QComboBox(); widget.addItems(['shift','none']); widget.setCurrentText(value)
            elif key in {'min_points','min_anchors'}:
                widget = QSpinBox(); widget.setRange(3,1000000); widget.setValue(value)
            else:
                widget = QDoubleSpinBox(); widget.setRange(0,1e12); widget.setDecimals(5); widget.setValue(value)
            self.widgets[key] = widget; form.addRow(labels[key], widget)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.settings = settings

    def accept(self):
        try:
            self.settings = Settings(**{key:widget.currentText() if key == 'alignment' else widget.value()
                                       for key,widget in self.widgets.items()})
        except ValueError as error:
            QMessageBox.warning(self, 'Check settings', str(error)); return
        super().accept()


class DetailsDialog(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Analysis Details')
        self.resize(1100,650)
        layout = QVBoxLayout(self); tabs = QTabWidget(); layout.addWidget(tabs)
        self.models = [MatrixModel(result, evidence=True), MeasurementsModel(result),
                       RowsModel(['Record','Field','Value'], run_rows(result))]
        for name,model in zip(['Evidence','Measurements','Run'],self.models):
            table = Table(); table.setSortingEnabled(False); table.setModel(model)
            table.horizontalHeader().setDefaultSectionSize(170)
            if name == 'Run':
                table.setColumnWidth(0,120); table.setColumnWidth(1,210); table.setColumnWidth(2,700)
            tabs.addTab(table,name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject); layout.addWidget(buttons)
