"""Kmz2Viewer main window."""

from __future__ import annotations

from kmzcore.plot_paint import finish_pending
from kmzcore.theme import apply_theme

from kmzcore.runtime.qt import WindowRuntime

from PySide6.QtCore import QEvent, QSortFilterProxyModel, Qt, QThread
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from kmz2viewer.differences import SpectrumAnalysis, analyze_spectrum
from kmz2viewer.filter import ScanFilter, apply_filter
from kmz2viewer.formula import FormulaIndex, FormulaSettings
from kmz2viewer.gui.dialogs import FormulaDialog
from kmz2viewer.gui.spectrum import SpectrumView
from kmz2viewer.gui.table import ScanTable, ScanTableModel
from kmz2viewer.model import Library, Ms2Scan
from kmz2viewer.workers import LoadWorker


class MainWindow(QMainWindow):
    def __init__(self, *, precursor_mz: float | None = None) -> None:
        super().__init__()
        from PySide6.QtWidgets import QApplication
        apply_theme(QApplication.instance())
        self.setWindowTitle("Kmz2Viewer")
        self.resize(1280, 860)
        self.library = Library()
        self.formulas = FormulaIndex()
        self.settings = FormulaSettings()
        self._matches: list[Ms2Scan] = []
        self._analysis_cache: dict[tuple, SpectrumAnalysis] = {}
        self._shown_key: tuple[str, int] | None = None
        self._busy = False
        self._worker = None
        self._thread: QThread | None = None

        self.precursor = QLineEdit()
        self.precursor.setPlaceholderText("250.0000")
        if precursor_mz is not None:
            from kmzcore.runtime.handoff import precursor_mass
            self.precursor.setText(repr(precursor_mass(str(precursor_mz))))
        self.precursor_tol = self._tolerance(20.0)
        self.precursor_unit = self._unit_box()
        self.fragment = QLineEdit()
        self.fragment.setPlaceholderText("132.0760")
        self.fragment_tol = self._tolerance(20.0)
        self.fragment_unit = self._unit_box()
        self.fragment_percent = QDoubleSpinBox()
        self.fragment_percent.setRange(0.0, 100.0)
        self.fragment_percent.setDecimals(1)
        self.fragment_percent.setSuffix(" %")
        self.rt_min = QLineEdit()
        self.rt_min.setPlaceholderText("min")
        self.rt_max = QLineEdit()
        self.rt_max.setPlaceholderText("max")
        self.filter_note = QLabel("")
        self.filter_note.setWordWrap(True)
        clear_filters = QPushButton("Clear filters")
        clear_filters.clicked.connect(self.clear_filters)

        filters = QGroupBox("Filters")
        form = QFormLayout(filters)
        form.addRow("Precursor m/z", self._row(self.precursor, self.precursor_tol, self.precursor_unit))
        form.addRow("Fragment m/z", self._row(self.fragment, self.fragment_tol, self.fragment_unit))
        form.addRow("Fragment ≥ % base peak", self.fragment_percent)
        form.addRow("RT min (min)", self.rt_min)
        form.addRow("RT max (min)", self.rt_max)
        form.addRow("", clear_filters)
        form.addRow("", self.filter_note)

        self.model = ScanTableModel()
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.table = ScanTable()
        self.table.setModel(self.proxy)
        self.table.open_requested.connect(self.open_selected)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._sync_actions())
        self.table.horizontalHeader().setStretchLastSection(True)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.addWidget(filters)
        center_layout.addWidget(self.table, stretch=1)
        self.setCentralWidget(center)

        self.spectrum = SpectrumView()
        self.spectrum.previous_requested.connect(lambda: self._step_spectrum(-1))
        self.spectrum.next_requested.connect(lambda: self._step_spectrum(1))
        self.spectrum.settings_requested.connect(self.edit_formula_settings)
        self.spectrum.losses.toggled.connect(self._losses_toggled)
        dock = self._dock("MS2 Spectrum", self.spectrum)
        dock.setMinimumHeight(280)

        self.count_label = QLabel("0 MS2 scans")
        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setTextVisible(False)
        self._progress.hide()
        self.statusBar().addPermanentWidget(self.count_label)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Open an mzML file.")

        self._build_actions()
        self.runtime = WindowRuntime(self, "ms2")
        self._install_drops()
        self._apply_filters()

    def _dock(self, title: str, widget):
        from PySide6.QtWidgets import QDockWidget

        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(title)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        return dock

    def _tolerance(self, value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 1_000_000.0)
        box.setDecimals(4)
        box.setValue(value)
        return box

    def _unit_box(self) -> QComboBox:
        box = QComboBox()
        box.addItem("ppm", "ppm")
        box.addItem("Da", "Da")
        return box

    def _row(self, *widgets) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return holder

    def _build_actions(self) -> None:
        self.open_action = QAction("Open mzML", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_dialog)
        self.clear_action = QAction("Clear files", self)
        self.clear_action.triggered.connect(self.clear_files)
        self.open_spectrum_action = QAction("Open spectrum", self)
        self.open_spectrum_action.triggered.connect(self.open_selected)
        quit_action = QAction("Quit", self)
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        self.cancel_action = QAction("Cancel loading", self)
        self.cancel_action.triggered.connect(lambda: self.runtime.cancel())
        self.cancel_action.setVisible(False)
        file_menu.addAction(self.cancel_action)
        file_menu.addAction(quit_action)
        spectrum_menu = self.menuBar().addMenu("Spectrum")
        spectrum_menu.addAction(self.open_spectrum_action)
        toolbar = self.addToolBar("Tools")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.clear_action)
        toolbar.addAction(self.open_spectrum_action)
        for widget in (
            self.precursor,
            self.precursor_tol,
            self.precursor_unit,
            self.fragment,
            self.fragment_tol,
            self.fragment_unit,
            self.fragment_percent,
            self.rt_min,
            self.rt_max,
        ):
            if isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self._apply_filters)
            else:
                widget.valueChanged.connect(lambda *_: self._apply_filters()) if hasattr(widget, "valueChanged") else widget.currentIndexChanged.connect(self._apply_filters)
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self.spectrum)
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self.spectrum)
        self.left_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.right_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.left_shortcut.activated.connect(lambda: self._step_spectrum(-1))
        self.right_shortcut.activated.connect(lambda: self._step_spectrum(1))
        self.full_shortcuts = []
        for widget in (self.spectrum, self.table):
            shortcut = QShortcut(QKeySequence("F"), widget)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(self.spectrum.reset_range)
            self.full_shortcuts.append(shortcut)
        self.spectrum.plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.spectrum.full_button.setToolTip("Restore the full m/z and intensity bounds. Shortcut: F.")

    def _install_drops(self) -> None:
        holder = _DropFilter(self.open_paths)
        self._drop_filter = holder.filter
        for widget in (self, self.table, self.spectrum, self.spectrum.plot):
            widget.setAcceptDrops(True)
            widget.installEventFilter(self._drop_filter)

    def open_dialog(self) -> None:
        paths, _selected = QFileDialog.getOpenFileNames(self, "Open mzML", self.runtime.directory, "mzML files (*.mzML *.mzml)")
        if paths:
            self.open_paths(paths)

    def open_paths(self, paths: list[str]) -> None:
        self.runtime.request(paths)

    def _open_paths_now(self, paths: list[str]) -> None:
        self.runtime.start(LoadWorker(paths), self._on_loaded)

    def load_paths_sync(self, paths: list[str]) -> list[str]:
        from kmz2viewer.index import index_ms2

        errors = []
        collected = []
        have = {scan.path for scan in self.library.scans}
        for path in paths:
            if path in have:
                continue
            try:
                collected.extend(index_ms2(path))
                have.add(path)
            except Exception as exc:
                errors.append(str(exc))
        self.library.add(collected)
        self._analysis_cache.clear()
        self._apply_filters()
        return errors

    def clear_files(self) -> None:
        self.runtime.cancel()
        self.runtime.opened.clear()
        self.library.clear()
        self._analysis_cache.clear()
        self._shown_key = None
        self.spectrum.caption.setText("Open a row to inspect an MS2 scan.")
        self.spectrum.detail.setText("")
        self.spectrum.plot.clear()
        self._apply_filters()

    def clear_filters(self) -> None:
        for box in (self.precursor, self.fragment, self.rt_min, self.rt_max):
            box.clear()
        self.precursor_tol.setValue(20.0)
        self.fragment_tol.setValue(20.0)
        self.precursor_unit.setCurrentIndex(0)
        self.fragment_unit.setCurrentIndex(0)
        self.fragment_percent.setValue(0.0)
        self._apply_filters()

    def _apply_filters(self) -> None:
        spec, note = _read_filter(self)
        self.filter_note.setText(note)
        self._matches = apply_filter(self.library.scans, spec)
        self.model.set_rows(self._matches)
        self._update_count()
        self._sync_actions()
        self._refresh_navigation()

    def _update_count(self) -> None:
        total = len(self.library.scans)
        matched = len(self._matches)
        noun = "spectrum" if matched == 1 else "spectra"
        self.count_label.setText(f"{total} MS2 scans · {matched} matching {noun}")

    def open_selected(self) -> None:
        scan = self.current_scan()
        if scan is None:
            return
        self._show_scan(scan, keep_window=False)

    def current_scan(self) -> Ms2Scan | None:
        index = self.table.currentIndex()
        if not index.isValid():
            return None
        source = self.proxy.mapToSource(index)
        return self.model.scan_at(source.row())

    def _show_scan(self, scan: Ms2Scan, keep_window: bool) -> None:
        analysis = self._analysis_for(scan)
        same_window = keep_window and self._shown_key is not None
        self.spectrum.show_scan(scan, analysis, keep_window=same_window)
        self._shown_key = scan.key
        self._select_scan(scan)
        self._refresh_navigation()
        assigned = len(analysis.assigned_pairs)
        self.statusBar().showMessage(
            f"{len(analysis.anchors)} anchors, {len(analysis.pairs)} differences, "
            f"{assigned} formulae, {analysis.seconds * 1000:.1f} ms"
        )

    def _analysis_for(self, scan: Ms2Scan) -> SpectrumAnalysis:
        key = (scan.path, scan.spectrum_index, self.settings.table_key(), self.settings.ppm)
        cached = self._analysis_cache.get(key)
        if cached is not None:
            return cached
        analysis = analyze_spectrum(
            scan.mz,
            scan.intensity,
            centroid=scan.centroid,
            precursor_mz=scan.precursor_mz,
            charge=scan.charge,
            formulas=self.formulas,
            settings=self.settings,
        )
        self._analysis_cache[key] = analysis
        return analysis

    def _step_spectrum(self, step: int) -> None:
        if not self._matches:
            return
        current = self._proxy_row()
        if current is None:
            target = 0 if step > 0 else len(self._matches) - 1
        else:
            target = min(max(current + step, 0), len(self._matches) - 1)
        index = self.proxy.index(target, 0)
        self.table.setCurrentIndex(index)
        scan = self.model.scan_at(self.proxy.mapToSource(index).row())
        if scan is not None:
            self._show_scan(scan, keep_window=True)

    def _proxy_row(self) -> int | None:
        index = self.table.currentIndex()
        if index.isValid():
            return index.row()
        if self._shown_key is None:
            return None
        for row, scan in enumerate(self._matches):
            if scan.key == self._shown_key:
                source = self.model.index(row, 0)
                mapped = self.proxy.mapFromSource(source)
                if mapped.isValid():
                    return mapped.row()
        return None

    def _select_scan(self, scan: Ms2Scan) -> None:
        for row in range(self.model.rowCount()):
            candidate = self.model.scan_at(row)
            if candidate is not None and candidate.key == scan.key:
                mapped = self.proxy.mapFromSource(self.model.index(row, 0))
                if mapped.isValid():
                    self.table.setCurrentIndex(mapped)
                    self.table.scrollTo(mapped)
                return

    def _refresh_navigation(self) -> None:
        row = self._proxy_row()
        count = self.proxy.rowCount()
        self.spectrum.set_navigation(row is not None and row > 0, row is not None and row < count - 1)

    def edit_formula_settings(self) -> None:
        dialog = FormulaDialog(self.settings, self)
        if dialog.exec() != FormulaDialog.DialogCode.Accepted:
            return
        self.settings = dialog.settings()
        self._analysis_cache.clear()
        scan = self.current_scan()
        if scan is not None and self._shown_key == scan.key:
            self._show_scan(scan, keep_window=True)

    def _losses_toggled(self, _enabled: bool) -> None:
        scan = self.current_scan()
        if scan is not None and self._shown_key == scan.key:
            self.spectrum.set_show_losses(self.spectrum.losses.isChecked())

    def _on_loaded(self, payload: dict) -> None:
        self.runtime.opened.update(payload["paths"])
        self.library.add(payload["scans"])
        self._apply_filters()
        if payload["errors"]:
            QMessageBox.warning(self, "Some files were not opened", "\n".join(payload["errors"]))
        names = ", ".join(payload["names"])
        if names:
            self.statusBar().showMessage(f"Opened {names}")

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.statusBar().showMessage(message)
        if total <= 0:
            self._progress.setRange(0, 0)
            return
        self._progress.setRange(0, 1000)
        self._progress.setValue(int(1000 * done / total))

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Kmz2Viewer", message)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        self.open_action.setEnabled(not self.runtime.closing)
        self.cancel_action.setVisible(busy)
        self._progress.setVisible(busy)
        if not busy:
            self._progress.setRange(0, 1000)
            self._progress.setValue(0)
        if message:
            self.statusBar().showMessage(message)
        self._sync_actions()

    def _sync_actions(self) -> None:
        self.open_spectrum_action.setEnabled(self.current_scan() is not None and not self._busy)
        self.clear_action.setEnabled(bool(self.library.scans) and not self._busy)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.UpdateRequest:
            plot = self.spectrum.plot
            finish_pending(plot, (plot.getAxis("left"), plot.getAxis("bottom")))
        return super().event(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.runtime.close_event(event)



def _optional_float(text: str) -> tuple[float | None, str]:
    raw = text.strip()
    if not raw:
        return None, ""
    try:
        return float(raw), ""
    except ValueError:
        return None, f"Could not read '{raw}'."


def _read_filter(window: MainWindow) -> tuple[ScanFilter, str]:
    notes = []
    precursor, note = _optional_float(window.precursor.text())
    if note:
        notes.append("Precursor m/z " + note)
    fragment, note = _optional_float(window.fragment.text())
    if note:
        notes.append("Fragment m/z " + note)
    rt_min, note = _optional_float(window.rt_min.text())
    if note:
        notes.append("RT min " + note)
    rt_max, note = _optional_float(window.rt_max.text())
    if note:
        notes.append("RT max " + note)
    spec = ScanFilter(
        precursor_mz=precursor,
        precursor_tolerance=float(window.precursor_tol.value()),
        precursor_unit=window.precursor_unit.currentData(),
        fragment_mz=fragment,
        fragment_tolerance=float(window.fragment_tol.value()),
        fragment_unit=window.fragment_unit.currentData(),
        fragment_min_percent=float(window.fragment_percent.value()),
        rt_min=rt_min,
        rt_max=rt_max,
    )
    return spec, " ".join(notes)


class _DropFilter:
    def __init__(self, callback) -> None:
        from PySide6.QtCore import QObject

        class _Filter(QObject):
            def eventFilter(self, watched, event):  # noqa: N802
                if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                    if _mzml_paths(event):
                        event.acceptProposedAction()
                        return True
                elif event.type() == QEvent.Type.Drop:
                    paths = _mzml_paths(event)
                    if paths:
                        event.acceptProposedAction()
                        callback(paths)
                        return True
                return False

        self.filter = _Filter()


def _mzml_paths(event) -> list[str]:
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    return [url.toLocalFile() for url in mime.urls() if url.toLocalFile().lower().endswith(".mzml")]
