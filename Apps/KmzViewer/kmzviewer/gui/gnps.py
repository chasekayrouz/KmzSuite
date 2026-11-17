"""Independent GNPS inspection windows; no automatic submissions or history."""
from __future__ import annotations

import base64
from pathlib import Path
from threading import Event

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)
from PySide6.QtCore import QObject

from kmzcore.cancel import Cancelled
from kmzcore.runtime.qt import WindowRuntime
from kmzcore.theme import INK
from kmzviewer.gnps import (
    SearchSettings, check_cancel, finite, load_analysis, matching_scans,
    new_analysis, prepare_query, save_analysis, timestamp,
)
from kmzviewer.gnps_client import GnpsClient, public_url

TITLES = {"standard": "Standard matches", "analog": "Possible analogs", "datasets": "Public datasets"}


class GnpsWorker(QObject):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self._cancel = Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            result = self.operation(self._cancel.is_set, lambda message: self.progress.emit(0, 0, message))
            check_cancel(self._cancel.is_set)
            self.succeeded.emit(result)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


def show_peaks(plot, peaks, color, sign=1):
    if not peaks:
        return
    a = np.asarray(peaks, dtype=float)
    heights = a[:, 1] / np.max(a[:, 1]) * 100 * sign
    x = np.repeat(a[:, 0], 2)
    y = np.column_stack((np.zeros(len(a)), heights)).ravel()
    plot.plot(x, y, connect="pairs", pen=pg.mkPen(color, width=1.4))


class StructureView(QLabel):
    """Fit the entire reference structure as the split/window is resized."""
    _image = None

    def clear(self):
        self._image = None
        super().clear()

    def set_image(self, image):
        self._image = image
        self.update()

    def paintEvent(self, event):  # noqa: N802
        if self._image is None:
            return super().paintEvent(event)
        rect = self.contentsRect()
        image = self._image.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(self)
        painter.drawPixmap(rect.center().x() - image.width() // 2, rect.center().y() - image.height() // 2, image)


class GnpsWindow(QMainWindow):
    closed = Signal()

    def __init__(self, parent=None, *, client=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.resize(1180, 800)
        self.setMinimumSize(1000, 740)
        self.client = client or GnpsClient()
        self.analysis = None
        self.query = None
        self.scans = []
        self.target_mz = self.target_rt = None
        self.saved_path = None
        self.offline = False
        self._busy = False
        self._updating = False
        self._worker = self._thread = None
        self._active_hit = None
        self._reference_attempts = set()
        self._tables = {}
        self._summaries = {}
        self.runtime = WindowRuntime(self, "ms1")
        self.setWindowTitle("GNPS analysis · KmzViewer")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.source = QLabel("Choose an MS1 peak in KmzViewer, or open a saved GNPS analysis.")
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.source.setWordWrap(True)
        layout.addWidget(self.source)
        row = QHBoxLayout()
        row.addWidget(QLabel("MS2 scan"))
        self.scan_picker = QComboBox()
        self.scan_picker.setMinimumContentsLength(30)
        self.scan_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.scan_picker.setAccessibleName("MS2 scan for GNPS search")
        self.scan_picker.currentIndexChanged.connect(self._scan_changed)
        row.addWidget(self.scan_picker, 1)
        self.open_button = QPushButton("Open analysis…")
        self.open_button.clicked.connect(self.open_dialog)
        self.save_button = QPushButton("Save analysis…")
        self.save_button.clicked.connect(self.save_dialog)
        row.addWidget(self.open_button)
        row.addWidget(self.save_button)
        layout.addLayout(row)
        self.scan_details = QLabel()
        self.scan_details.setTextFormat(Qt.TextFormat.PlainText)
        self.scan_details.setWordWrap(True)
        layout.addWidget(self.scan_details)

        controls = QGridLayout()
        self.fields = {}
        defaults = SearchSettings()
        specifications = [
            ("precursor_da", "Precursor ± (Da)", .00001, 1., 5),
            ("fragment_da", "Fragment ± (Da)", .00001, 1., 5),
            ("min_cosine", "Min cosine", 0., 1., 2),
            ("min_peaks", "Min matched peaks", 1, 1000, None),
            ("analog_below", "Analog below (Da)", 0., 300., 2),
            ("analog_above", "Analog above (Da)", 0., 300., 2),
        ]
        for column, (name, title, minimum, maximum, decimals) in enumerate(specifications):
            widget = QSpinBox() if decimals is None else QDoubleSpinBox()
            if decimals is not None:
                widget.setDecimals(decimals)
                widget.setSingleStep(.01 if maximum <= 1 else 1.)
            widget.setRange(minimum, maximum)
            widget.setValue(getattr(defaults, name))
            widget.setAccessibleName(title)
            widget.setKeyboardTracking(False)
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.valueChanged.connect(self._settings_changed)
            self.fields[name] = widget
            controls.addWidget(QLabel(title), 0, column)
            controls.addWidget(widget, 1, column)
        self.search_button = QPushButton("Search GNPS")
        self.search_button.clicked.connect(self.search)
        controls.addWidget(self.search_button, 0, 6, 2, 1)
        layout.addLayout(controls)
        actions = QHBoxLayout()
        self.dataset_button = QPushButton("Find in public datasets")
        self.dataset_button.clicked.connect(self.search_datasets)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.runtime.cancel)
        actions.addWidget(self.dataset_button)
        self.privacy = QLabel("Search sends this MS2 peak list and precursor metadata to GNPS. File names and paths stay local.")
        self.privacy.setWordWrap(True)
        actions.addWidget(self.privacy, 1)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

        split = QSplitter()
        self.tabs = QTabWidget()
        for mode, title in TITLES.items():
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            summary = QLabel("Not searched.")
            summary.setTextFormat(Qt.TextFormat.PlainText)
            summary.setWordWrap(True)
            page_layout.addWidget(summary)
            table = QTableWidget(0, 5)
            table.setHorizontalHeaderLabels(["Dataset / spectrum" if mode == "datasets" else "Candidate", "Cosine", "Peaks", "Δ mass", "Adduct"])
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.verticalHeader().hide()
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for col in range(1, 5):
                table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
            table.itemSelectionChanged.connect(lambda mode=mode: self._select_hit(mode))
            page_layout.addWidget(table)
            self.tabs.addTab(page, title)
            self._tables[mode], self._summaries[mode] = table, summary
        split.addWidget(self.tabs)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        self.candidate = QLabel("Query spectrum")
        self.candidate.setTextFormat(Qt.TextFormat.PlainText)
        self.candidate.setWordWrap(True)
        self.candidate.setStyleSheet("font-weight: 600;")
        right_layout.addWidget(self.candidate)
        self.evidence = QLabel("Select a candidate to compare its reference spectrum.")
        self.evidence.setTextFormat(Qt.TextFormat.PlainText)
        self.evidence.setWordWrap(True)
        right_layout.addWidget(self.evidence)
        self.structure = StructureView("Structure appears when available")
        self.structure.setTextFormat(Qt.TextFormat.PlainText)
        self.structure.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.structure.setMinimumHeight(100)
        self.structure.setMaximumHeight(200)
        self.structure.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.structure, 1)
        self.mirror = pg.PlotWidget()
        self.mirror.setMinimumHeight(200)
        self.mirror.setBackground("#ffffff")
        self.mirror.setLabel("bottom", "m/z")
        self.mirror.setLabel("left", "Relative intensity (%)")
        self.mirror.setMenuEnabled(False)
        self.mirror.getPlotItem().hideButtons()
        for name in ("left", "bottom"):
            self.mirror.getAxis(name).setTextPen(INK)
            self.mirror.getAxis(name).setPen(INK)
        right_layout.addWidget(self.mirror, 2)
        self.legend = QLabel("Query above zero · Reference below zero · Each spectrum scaled to its own base peak")
        self.legend.setWordWrap(True)
        right_layout.addWidget(self.legend)
        detail_actions = QHBoxLayout()
        self.reference_button = QPushButton("Load reference from GNPS")
        self.reference_button.clicked.connect(self.load_reference)
        self.link_button = QPushButton("Open public record")
        self.link_button.clicked.connect(self.open_public_record)
        self.full_button = QPushButton("Full m/z")
        self.full_button.clicked.connect(self.mirror.autoRange)
        detail_actions.addWidget(self.reference_button)
        detail_actions.addWidget(self.link_button)
        detail_actions.addWidget(self.full_button)
        right_layout.addLayout(detail_actions)
        split.addWidget(right)
        split.setSizes([540, 600])
        layout.addWidget(split, 1)
        footer = QLabel("Spectral similarity supports candidate annotations, not confirmed identities. Analog matches may represent related compounds.")
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.tabs.currentChanged.connect(self._tab_changed)
        self._set_busy(False)

    def settings(self):
        return SearchSettings(**{name: widget.value() for name, widget in self.fields.items()}).validate()

    def load_source(self, path, mass, rt, polarity):
        self.target_mz, self.target_rt = mass, rt
        self.source.setText(f"{Path(path).name} · MS1 m/z {mass:.7f} · matching MS2 precursors ±20 ppm")
        def read(cancel, progress):
            return matching_scans(path, mass, rt, polarity, cancel=cancel,
                                  progress=lambda done, total: progress(f"Reading matching MS2 scans · {done * 100 // max(total, 1)}%"))
        self.runtime.start(GnpsWorker(read), self._loaded_scans)

    def load_query(self, query):
        """Review an existing local query; submission still requires Search."""
        from copy import deepcopy
        self._updating = True
        self.scans = []
        self.scan_picker.clear()
        self.scan_picker.addItem("Selected MS2 · " + str(query.get("scan_id", "")))
        self._updating = False
        self.query = deepcopy(query)
        self.source.setText("Selected MS2 · " + self.query.get("source_name", ""))
        self._clear_results()
        self._show_query()
        self._sync()

    def _loaded_scans(self, scans):
        self.scans = scans
        self._updating = True
        self.scan_picker.clear()
        for scan in scans:
            rt = f"{scan.rt_min:.4f} min" if finite(scan.rt_min) else "RT unknown"
            self.scan_picker.addItem(f"{rt} · m/z {scan.precursor_mz:.6f} · {scan.scan_id}")
        self._updating = False
        if scans:
            self._scan_changed(0)
        else:
            self.scan_details.setText("No matching MS2 scans found in this file within ±20 ppm and compatible polarity. GNPS requires an acquired MS2 spectrum.")
            self.statusBar().showMessage("No matching MS2 spectrum. No data was sent.")

    def _scan_changed(self, index):
        if self._updating or not 0 <= index < len(self.scans):
            return
        try:
            self.query = prepare_query(self.scans[index], target_mz=self.target_mz, target_rt=self.target_rt)
            self._clear_results()
            self._show_query()
            self.statusBar().showMessage("Review this scan, then Search. Use Save before switching scans or settings.")
        except ValueError as exc:
            self.query = None
            self._clear_results()
            self.scan_details.setText(str(exc))
        self._sync()

    def _settings_changed(self):
        if not self._updating:
            self._clear_results()
            self.statusBar().showMessage("Settings changed. Search again to obtain results with these settings.")
            self._sync()

    def _clear_results(self):
        self.analysis = None
        self.saved_path = None
        self.offline = False
        self._active_hit = None
        self._reference_attempts.clear()
        for mode, table in self._tables.items():
            table.setRowCount(0)
            self._summaries[mode].setText("Not searched.")
        self._show_reference(None)

    def _show_query(self):
        q = self.query
        if not q:
            return
        polarity = {1: "positive", -1: "negative", 0: "unknown polarity"}[q.get("polarity", 0)]
        charge = q.get("charge")
        charge_text = str(charge) if charge else "unknown (sent as unspecified)"
        rt = f"{q['rt_min']:.4f} min" if finite(q.get("rt_min")) else "unknown RT"
        ce = str(q["collision_energy"]) if q.get("collision_energy") is not None else "unknown"
        self.scan_details.setText(f"{rt} · {polarity} · charge {charge_text} · collision energy {ce} · precursor from {q.get('precursor_source', 'unknown')}\n{q['preparation']} · {len(q['peaks'])} query peaks")
        self._show_reference(None)

    def _run_search(self, modes):
        if self._busy or not self.query:
            return
        query, settings = self.query, self.settings()
        if self.analysis is None:
            self.analysis = new_analysis(query, settings)
        self.saved_path = None
        self.offline = False
        for mode in modes:
            self._summaries[mode].setText("Searching…")

        def search(cancel, progress):
            results = {}
            for mode in modes:
                progress(f"Searching {TITLES[mode].lower()}…")
                try:
                    results[mode] = self.client.search(query, settings, mode, cancel=cancel, progress=progress)
                except Cancelled:
                    raise
                except Exception as exc:
                    results[mode] = dict(hits=[], error=str(exc), created=timestamp())
            return results
        self.runtime.start(GnpsWorker(search), self._searched)

    def search(self):
        if self._busy or not self.query:
            return
        self._clear_results()
        self._run_search(("standard", "analog"))

    def search_datasets(self):
        self._run_search(("datasets",))

    def _searched(self, results):
        self.analysis["searches"].update(results)
        self._fill_tables()
        self.statusBar().showMessage("Search finished. Results are unsaved; use Save analysis to keep them.")
        preferred = "datasets" if tuple(results) == ("datasets",) else "standard"
        self.tabs.setCurrentIndex(list(TITLES).index(preferred))
        table = self._tables[preferred]
        if table.rowCount():
            table.selectRow(0)

    def _fill_tables(self):
        for mode, table in self._tables.items():
            result = self.analysis["searches"].get(mode)
            table.blockSignals(True)
            table.setRowCount(0)
            if result is not None:
                hits = result["hits"]
                # Keep all accepted matches in the snapshot, cap widget work.
                shown = hits[:1000]
                table.setRowCount(len(shown))
                for row, hit in enumerate(shown):
                    title = f"{hit['dataset']} · {hit['name']}" if mode == "datasets" else hit["name"]
                    cells = [title, f"{hit['score']:.3f}", str(hit["matched_peaks"]), f"{hit.get('delta_mass', 0):+.4f}", hit.get("adduct", "")]
                    for col, text in enumerate(cells):
                        item = QTableWidgetItem(text)
                        item.setToolTip(hit["usi"] if col == 0 else text)
                        if col == 0:
                            item.setData(Qt.ItemDataRole.UserRole, hit)
                        table.setItem(row, col, item)
                if result.get("error"):
                    summary = "Search failed: " + result["error"]
                else:
                    summary = f"{len(hits)} matches · {result.get('created', '')[:19]} UTC"
                    if not hits:
                        summary = "No matches passed these settings. " + summary
                    if len(hits) > 1000:
                        summary += " · displaying the top 1,000; all matches are saved"
                    if mode == "datasets":
                        summary += f" · {len({h['dataset'] for h in hits})} datasets"
                self._summaries[mode].setText(summary)
            table.blockSignals(False)
        self._sync()

    def _tab_changed(self, _index):
        self._select_hit(list(TITLES)[self.tabs.currentIndex()])

    def _select_hit(self, mode):
        table = self._tables[mode]
        row = table.currentRow()
        if row < 0 or table.item(row, 0) is None:
            self._active_hit = None
            self._show_reference(None)
        else:
            self._active_hit = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            reference = self.analysis["references"].get(self._active_hit["usi"]) if self.analysis else None
            self._show_reference(reference)
        self._sync()

    def load_reference(self):
        if self._busy or not self._active_hit or not self.analysis:
            return
        hit = self._active_hit
        self._reference_attempts.add(hit["usi"])
        def receive(reference):
            self.analysis["references"][hit["usi"]] = reference
            self.saved_path = None
            if self._active_hit and self._active_hit["usi"] == hit["usi"]:
                self._show_reference(reference)
            self.statusBar().showMessage("Reference loaded. Save analysis includes this spectrum and available structure.")
        self.runtime.start(GnpsWorker(lambda cancel, progress: self.client.reference(hit, cancel=cancel)), receive)

    def _auto_reference(self):
        if (not self._busy and not self.offline and self._active_hit and self.analysis
                and self._active_hit["usi"] not in self._reference_attempts
                and self._active_hit["usi"] not in self.analysis["references"]
                and not self.runtime.closing):
            self.load_reference()

    def _show_reference(self, reference):
        self.mirror.clear()
        self.mirror.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#b8c7cc", width=1)))
        if self.query:
            show_peaks(self.mirror, self.query["peaks"], "#167c91")
        self.structure.clear()
        self.structure.setText("Structure unavailable" if reference else "Load a reference to view its structure")
        hit = self._active_hit
        self.candidate.setText(hit["name"] if hit else "Query spectrum")
        if hit:
            self.evidence.setText(f"Cosine {hit['score']:.3f} · {hit['matched_peaks']} matched peaks · Δ mass {hit.get('delta_mass', 0):+.4f} Da\n{hit['usi']}")
        else:
            self.evidence.setText("Select a candidate to inspect its evidence and load a comparison.")
        if reference:
            show_peaks(self.mirror, reference["peaks"], "#b06c24", -1)
            metadata = reference.get("metadata", {})
            detail = f"\n{reference.get('library', '')} · {metadata.get('Ion_Mode', 'unknown ion mode')} · {metadata.get('Adduct', '')}"
            self.evidence.setText(self.evidence.text() + detail + "\n" + " ".join(reference.get("warnings", [])))
            mode = metadata.get("Ion_Mode", "").strip().lower()
            polarity = self.query.get("polarity", 0) if self.query else 0
            if (polarity == 1 and mode.startswith("negative")) or (polarity == -1 and mode.startswith("positive")):
                self.evidence.setText(self.evidence.text() + "\nPolarity differs from the query. This is not a compatible ion annotation.")
            encoded = reference.get("structure_png")
            if encoded:
                try:
                    raw = base64.b64decode(encoded, validate=True)
                    buffer = QBuffer()
                    buffer.setData(QByteArray(raw))
                    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
                    reader = QImageReader(buffer, b"png")
                    size = reader.size()
                    if size.width() > 0 and size.height() > 0 and size.width() * size.height() <= 8_000_000:
                        pixmap = QPixmap.fromImage(reader.read())
                        if not pixmap.isNull():
                            self.structure.set_image(pixmap)
                except (ValueError, TypeError):
                    pass
        self.mirror.autoRange()

    def open_public_record(self):
        if self._active_hit:
            QDesktopServices.openUrl(QUrl(public_url(self._active_hit, dataset=self.tabs.currentIndex() == 2)))

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open GNPS analysis", "", "KmzViewer GNPS analysis (*.kmzgnps)")
        if path:
            try:
                self.open_analysis(path)
            except Exception as exc:
                QMessageBox.warning(self, "Could not open analysis", str(exc))

    def open_analysis(self, path):
        if self._busy:
            raise ValueError("Wait for or cancel the current operation before opening an analysis.")
        data = load_analysis(path)
        self._updating = True
        self.scans = []
        self.scan_picker.clear()
        self.scan_picker.addItem("Saved query · " + str(data["query"].get("scan_id", "")))
        for name, widget in self.fields.items():
            widget.setValue(data["settings"][name])
        self._updating = False
        self.analysis = data
        self.query = data["query"]
        self.saved_path = str(path)
        self.offline = True
        self._active_hit = None
        self.source.setText("Saved GNPS analysis · " + str(self.query.get("source_name", "")))
        self._show_query()
        self._fill_tables()
        for mode, table in self._tables.items():
            if table.rowCount():
                self.tabs.setCurrentIndex(list(TITLES).index(mode))
                table.selectRow(0)
                break
        self.statusBar().showMessage("Opened saved results locally. No network request was made.")
        self._sync()

    def save_dialog(self):
        if self.analysis is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save GNPS analysis", self.saved_path or "GNPS analysis.kmzgnps", "KmzViewer GNPS analysis (*.kmzgnps)")
        if path:
            if not path.lower().endswith(".kmzgnps"):
                path += ".kmzgnps"
            try:
                save_analysis(path, self.analysis)
                self.saved_path = path
                self.statusBar().showMessage("Analysis saved, including all loaded reference spectra and structures.")
            except Exception as exc:
                QMessageBox.warning(self, "Could not save analysis", str(exc))

    def _sync(self):
        idle = not self._busy and not self.runtime.closing
        self.search_button.setEnabled(idle and self.query is not None)
        self.dataset_button.setEnabled(idle and self.query is not None)
        self.open_button.setEnabled(idle)
        self.save_button.setEnabled(idle and self.analysis is not None and bool(self.analysis["searches"]))
        self.scan_picker.setEnabled(idle and bool(self.scans))
        for widget in self.fields.values():
            widget.setEnabled(idle)
        self.cancel_button.setVisible(self._busy)
        hit = self._active_hit
        loaded = bool(hit and self.analysis and hit["usi"] in self.analysis["references"])
        self.reference_button.setEnabled(idle and bool(hit) and not loaded)
        self.reference_button.setText("Reference loaded" if loaded else "Load reference from GNPS")
        self.link_button.setEnabled(bool(hit))
        if idle and hit and not loaded and not self.offline and hit["usi"] not in self._reference_attempts:
            QTimer.singleShot(0, self._auto_reference)

    def _set_busy(self, busy, message=""):
        self._busy = busy
        if message:
            self.statusBar().showMessage(message)
        if not busy and self.analysis:
            for mode, label in self._summaries.items():
                if label.text() == "Searching…":
                    label.setText("Search stopped. No result was accepted.")
        self._sync()

    def _on_progress(self, done, total, message):
        self.statusBar().showMessage(message)

    def _on_failed(self, message):
        self.statusBar().showMessage(message)
        self.evidence.setText(message)

    def closeEvent(self, event):  # noqa: N802
        already_closed = self.runtime.closed
        self.runtime.close_event(event)
        if self.runtime.closed and not already_closed:
            self.closed.emit()
            self.deleteLater()
