"""Main window. Menus call into the session; file work runs off the GUI thread."""

from __future__ import annotations

from kmzcore.runtime.qt import WindowRuntime
from kmzcore.runtime.handoff import open_ms2
from kmzcore.sparse_profile import HELP as SP_HELP
from kmzcore.theme import apply_theme, color_icon

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QItemSelectionModel, QSignalBlocker, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QColorDialog,
    QApplication,
    QComboBox,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QTreeWidgetItem,
    QHeaderView,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kmzviewer.export_svg import ExportOptions, ExportTrace, export_chromatograms
from kmzviewer.extract import extract_bpc, extract_eics, make_tic_trace
from kmzviewer.gui.dialogs import BpcDialog, ExportDialog, ExtractDialog, Ms2FilesDialog, WavelengthDialog
from kmzviewer.gui.mz_filter import MzFilterBar
from kmzviewer.gui.manage_chromatograms import ManageChromatogramsDialog
from kmzviewer.mz_filter import FilterCache, mz_range, range_label
from kmzviewer.dad import extract_wavelengths
from kmzviewer.gui.navigator import DataNavigator
from kmzviewer.gui.plots import ChromatogramPanel, SpectrumPanel
from kmzviewer.gui.workers import ExtractWorker, FilterWorker, IndexWorker, SpectrumWorker, plan_bpc_jobs, plan_jobs
from kmzviewer.model import Session
from kmzviewer.comparison import SpectrumFrame, match_scan
from kmzviewer.gui.spectrum_visibility import SpectrumEye
from kmzviewer.mzml_index import load_ms1
from kmzviewer.spectra import ArrayCache


class _DropFilter:
    """Accepts mzML drops on the window and the plot widgets."""

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
    paths = []
    for url in mime.urls():
        local = url.toLocalFile()
        if local.lower().endswith(".mzml"):
            paths.append(local)
    return paths


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        apply_theme(QApplication.instance())
        self.setWindowTitle("KmzViewer")
        self.resize(1280, 820)
        self.session = Session()
        self.active_trace_id: int | None = None
        self._last_input_path: Path | None = None
        self.walk_mode = False
        self.walk_trace_id: int | None = None
        self.walk_i: int | None = None
        self._token = 0
        self.spectrum_file_ids: set[int] = set()
        self.dad_file_ids: set[int] = set()
        self._dad_enabled: set[int] = set()
        self._dad_eyes = {}
        self._reference_dad_scan = None
        self._selected_rt: float | None = None
        self._selected_polarity = 0
        self._reference_scan = None
        self._inflight_comparison = None
        self._queued_comparison = None
        self._requested_comparison_key = None
        self._spectrum_eyes = {}
        self._all_spectrum_eyes = []
        self._all_dad_eyes = []
        self._navigator_expanded = {}
        self._busy = False
        self._gnps_windows = []
        self._worker = None
        self._thread: QThread | None = None
        self._array_cache = ArrayCache()
        self._cache_hits_before = 0
        self._cache_misses_before = 0
        self._mz_range = None
        self._filter_values = {}
        self._filter_cache = FilterCache()
        self._filter_token = 0
        self._filter_running = False
        self._filter_refresh_pending = False

        self.tree = DataNavigator()
        self.tree.setColumnCount(3)
        self.tree.setHeaderHidden(True)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 32)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(2, 32)
        self.tree.setColumnHidden(2, True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._on_selection)
        self.tree.files_moved.connect(self.move_files)
        self.tree.customContextMenuRequested.connect(self._context_menu)

        self.plots = ChromatogramPanel(normalize_overlay=True)
        self.plots.emphasis_requested.connect(self._toggle_emphasis)
        self.spectrum = SpectrumPanel()
        self.dad_spectrum = SpectrumPanel(optical=True)
        self.mz_filter = MzFilterBar()
        self.mz_filter.apply_requested.connect(self.apply_mz_filter)
        self.mz_filter.clear_requested.connect(self.clear_mz_filter)
        self.mz_filter.start.textChanged.connect(lambda: self._sync_actions())
        self.mz_filter.stop.textChanged.connect(lambda: self._sync_actions())
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.mz_filter)
        central_layout.addWidget(self.plots, 1)
        self.setCentralWidget(central)

        navigator_widget = QWidget()
        navigator_layout = QVBoxLayout(navigator_widget)
        navigator_layout.setContentsMargins(3, 3, 3, 3)
        self.group_by = QComboBox()
        self.group_by.addItem("Group by file", "file")
        self.group_by.addItem("Group by chromatogram type", "type")
        self.group_by.setAccessibleName("Navigator grouping")
        self.group_by.setToolTip("Group matching extraction settings across files and reorder chromatograms. Spectra retain file order.")
        self.group_by.currentIndexChanged.connect(lambda: self.set_grouping(self.group_by.currentData()))
        navigator_layout.addWidget(self.group_by)
        self.sort_button = QPushButton("Sort files A–Z")
        self.sort_button.setToolTip("Sort file names alphanumerically (2 before 10). Drag names to arrange them manually.")
        self.sort_button.clicked.connect(self.sort_files)
        navigator_layout.addWidget(self.sort_button)
        self.manage_chromatograms_button = QPushButton("Remove/Show/Hide…")
        self.manage_chromatograms_button.setToolTip("Remove chromatograms, show/hide plots, or control MS1 and DAD spectrum eyes.")
        self.manage_chromatograms_button.clicked.connect(self.open_manage_chromatograms)
        navigator_layout.addWidget(self.manage_chromatograms_button)
        navigator_layout.addWidget(self.tree)
        navigator = self._dock("Data Navigator", navigator_widget, Qt.DockWidgetArea.LeftDockWidgetArea)
        navigator.setMinimumWidth(180)
        self.navigator_dock = navigator
        self.spectrum_dock = self._dock(
            "MS1 Spectra", self.spectrum, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.spectrum_dock.setMinimumWidth(320)
        self.dad_dock = self._dock('DAD Spectra', self.dad_spectrum, Qt.DockWidgetArea.RightDockWidgetArea)
        self.dad_dock.setMinimumWidth(320)
        self.splitDockWidget(self.spectrum_dock, self.dad_dock, Qt.Orientation.Vertical)
        self.dad_dock.hide()
        self.resizeDocks([navigator, self.spectrum_dock], [220, 520], Qt.Orientation.Horizontal)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setTextVisible(False)
        self._progress.hide()
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Open an mzML file.")

        self._build_actions()
        self.runtime = WindowRuntime(self, "ms1")
        self._install_drops()
        self._start_spectrum_worker()
        self._sync_actions()

    def _dock(self, title: str, widget, area):
        from PySide6.QtWidgets import QDockWidget

        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(title)
        self.addDockWidget(area, dock)
        return dock

    def _build_actions(self) -> None:
        self.open_action = QAction("Open mzML…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_dialog)

        self.ms2_menu = QMenu("Open Kmz2Viewer for MS/MS", self)
        self.ms2_files_action = self.ms2_menu.addAction("Open files from current session…", self.open_ms2_files)
        self.ms2_blank_action = self.ms2_menu.addAction("Open blank session", lambda: self._launch_ms2([]))
        self.ms2_button = QToolButton()
        self.ms2_button.setText(self.ms2_menu.title())
        self.ms2_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.ms2_button.setMenu(self.ms2_menu)
        self.ms2_button.setToolTip("Start a new Kmz2Viewer session with selected files or no files.")
        self.spectrum.msms_requested.connect(self.open_ms2_peak)
        self.spectrum.gnps_requested.connect(self.open_gnps_peak)
        self.gnps_open_action = QAction("Open spectral analysis…", self)
        self.gnps_open_action.triggered.connect(self.open_gnps_analysis)

        self.extract_action = QAction("Extract EIC…", self)
        self.extract_action.setShortcut(QKeySequence("Ctrl+E"))
        self.extract_action.triggered.connect(self.open_extract_dialog)
        self.bpc_action = QAction("Extract BPC…", self)
        self.bpc_action.triggered.connect(self.open_bpc_dialog)
        self.manage_chromatograms_action = QAction("Remove/Show/Hide…", self)
        self.manage_chromatograms_action.triggered.connect(self.open_manage_chromatograms)
        self.twc_action = QAction('Extract TWC…', self)
        self.twc_action.triggered.connect(lambda: self.open_wavelength_dialog(total=True))
        self.ewc_action = QAction('Extract EWC…', self)
        self.ewc_action.triggered.connect(lambda: self.open_wavelength_dialog(total=False))

        self.walk_action = QAction("Walk", self)
        self.walk_action.setCheckable(True)
        self.walk_action.setToolTip("Click or drag across a chromatogram to inspect scans. Left/Right steps along the active trace.")
        self.walk_action.toggled.connect(self._set_walk)

        self.pan_action = QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setChecked(True)
        self.zoom_action = QAction("Zoom box", self)
        self.zoom_action.setCheckable(True)
        mouse_group = QActionGroup(self)
        mouse_group.setExclusive(True)
        mouse_group.addAction(self.walk_action)
        mouse_group.addAction(self.pan_action)
        mouse_group.addAction(self.zoom_action)
        self.pan_action.triggered.connect(lambda: self.plots.set_mouse_mode("pan"))
        self.zoom_action.triggered.connect(lambda: self.plots.set_mouse_mode("zoom"))

        self.full_action = QAction("Full view", self)
        self.full_action.setShortcuts([QKeySequence("Ctrl+F"), QKeySequence("Ctrl+0")])
        self.full_action.setToolTip("Show the full retention-time range of the chromatograms.")
        self.full_action.triggered.connect(self.plots.reset_view)

        self.autoscale_action = QAction("Autoscale Y", self)
        self.autoscale_action.setCheckable(True)
        self.autoscale_action.setChecked(True)
        self.autoscale_action.setToolTip(
            "When on, Y is refit to the points inside the visible retention-time window after every X change."
        )
        self.autoscale_action.toggled.connect(self.plots.set_autoscale_y)

        self.match_action = QAction("Match Y axes", self)
        self.match_action.setCheckable(True)
        self.match_action.setChecked(False)
        self.match_action.setToolTip(
            "When on, share absolute Y scales within each unit. When off, Overlay scales each trace's visible peak to 100%."
        )
        self.match_action.toggled.connect(self.plots.set_match_y)

        self.overlay_action = QAction("Overlay", self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.toggled.connect(self._set_overlay)

        self.export_action = QAction("Export SVG…", self)
        self.export_action.triggered.connect(lambda: self.open_export_dialog(single=False))
        self.export_current_action = QAction("Export current chromatogram…", self)
        self.export_current_action.triggered.connect(lambda: self.open_export_dialog(single=True))
        self.export_ms1_action = QAction('Export MS1 spectra…', self)
        self.export_ms1_action.triggered.connect(lambda: self.open_spectrum_export(False))
        self.export_dad_action = QAction('Export DAD spectra…', self)
        self.export_dad_action.triggered.connect(lambda: self.open_spectrum_export(True))
        for panel, action in ((self.spectrum, self.export_ms1_action), (self.dad_spectrum, self.export_dad_action)):
            button = QPushButton('Export…')
            button.clicked.connect(action.trigger)
            panel.layout().itemAt(0).layout().addWidget(button)

        self.close_action = QAction("Close selected files", self)
        self.close_action.triggered.connect(self.close_selected_files)

        self.cancel_action = QAction("Cancel", self)
        self.cancel_action.setVisible(False)
        self.cancel_action.triggered.connect(self._cancel_job)

        quit_action = QAction("Quit", self)
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addMenu(self.ms2_menu)
        file_menu.addAction(self.gnps_open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_current_action)
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.export_ms1_action)
        file_menu.addAction(self.export_dad_action)
        file_menu.addSeparator()
        file_menu.addAction(self.close_action)
        file_menu.addAction(quit_action)

        chrom_menu = self.menuBar().addMenu("Chromatogram")
        chrom_menu.addAction(self.extract_action)
        chrom_menu.addAction(self.bpc_action)
        chrom_menu.addAction(self.twc_action)
        chrom_menu.addAction(self.ewc_action)
        chrom_menu.addSeparator()
        chrom_menu.addAction(self.manage_chromatograms_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.walk_action)
        view_menu.addAction(self.pan_action)
        view_menu.addAction(self.zoom_action)
        view_menu.addSeparator()
        view_menu.addAction(self.full_action)
        view_menu.addAction(self.autoscale_action)
        view_menu.addAction(self.match_action)
        view_menu.addAction(self.overlay_action)
        view_menu.addSeparator()
        view_menu.addAction(self.spectrum_dock.toggleViewAction())
        view_menu.addAction(self.dad_dock.toggleViewAction())
        view_menu.addAction(self.navigator_dock.toggleViewAction())

        toolbar = self.addToolBar("Tools")
        toolbar.setMovable(False)
        for action in (
            self.open_action,
            self.extract_action,
            self.bpc_action,
            self.walk_action,
            self.pan_action,
            self.zoom_action,
            self.full_action,
            self.autoscale_action,
            self.match_action,
            self.overlay_action,
            self.export_action,
            self.cancel_action,
        ):
            toolbar.addAction(action)
            if action is self.bpc_action:
                wavelength_button = QToolButton()
                wavelength_button.setText('Wavelength')
                wavelength_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
                wavelength_menu = QMenu(wavelength_button)
                wavelength_menu.addAction(self.twc_action)
                wavelength_menu.addAction(self.ewc_action)
                wavelength_button.setMenu(wavelength_menu)
                toolbar.addWidget(wavelength_button)
            if action is self.open_action:
                toolbar.addWidget(self.ms2_button)

        self.left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.left_shortcut.setEnabled(False)
        self.right_shortcut.setEnabled(False)
        self.left_shortcut.activated.connect(lambda: self.walk_by(-1))
        self.right_shortcut.activated.connect(lambda: self.walk_by(1))

        self.full_shortcuts = []
        for widget, callback in ((self.plots, self.plots.reset_view),
                                 (self.spectrum, self.spectrum.reset_range),
                                 (self.dad_spectrum, self.dad_spectrum.reset_range),
                                 (self.navigator_dock, self.plots.reset_view)):
            shortcut = QShortcut(QKeySequence("F"), widget)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
            self.full_shortcuts.append(shortcut)
        self.plots.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.spectrum.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.dad_spectrum.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Window-scoped shortcuts also work after detaching either spectrum dock.
        for dock in (self.spectrum_dock, self.dad_dock):
            for key, step in ((Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1)):
                shortcut = QShortcut(QKeySequence(key), dock)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.setEnabled(dock.isFloating())
                dock.topLevelChanged.connect(shortcut.setEnabled)
                shortcut.activated.connect(lambda step=step: self.walk_by(step))
                self.full_shortcuts.append(shortcut)
        self.full_action.setToolTip("Restore full retention-time and Y bounds. F resets the active plot area.")
        self.spectrum.reset_button.setToolTip("Restore the full m/z range. Shortcut: F while working in spectra.")

        self.plots.trace_activated.connect(lambda trace_id: self.activate_trace(trace_id, retarget=False))
        self.plots.scan_requested.connect(self._navigate_rt)
        self.plots.spectrum_added.connect(self.inspect_rt)
        self.plots.readout.connect(self.statusBar().showMessage)
        self.spectrum.readout.connect(self.statusBar().showMessage)
        self.dad_spectrum.readout.connect(self.statusBar().showMessage)

    def _install_drops(self) -> None:
        holder = _DropFilter(self.open_paths)
        self._drop_filter = holder.filter
        for widget in (self, self.tree, self.plots, self.plots.view, self.spectrum, self.spectrum.view, self.dad_spectrum, self.dad_spectrum.view):
            widget.setAcceptDrops(True)
            widget.installEventFilter(self._drop_filter)

    def _start_spectrum_worker(self) -> None:
        self._spectrum_worker = SpectrumWorker(self._array_cache)
        self._spectrum_worker.batch_result.connect(self._on_comparison, Qt.ConnectionType.QueuedConnection)
        self._spectrum_thread = threading.Thread(target=self._spectrum_worker.run, name="spectrum-reader", daemon=False)
        self._spectrum_thread.start()

    def open_dialog(self) -> None:
        paths, _selected = QFileDialog.getOpenFileNames(
            self,
            "Open mzML",
            self.runtime.directory,
            "mzML files (*.mzML *.mzml)",
        )
        if paths:
            self.open_paths(paths)

    def open_paths(self, paths: list[str]) -> None:
        self.runtime.request(paths)

    def open_ms2_files(self) -> None:
        dialog = Ms2FilesDialog(self.session.files.values(), self,
                                selected_file_ids=self._selected_file_ids())
        if dialog.exec():
            paths = [self.session.files[fid].path for fid in dialog.checked_file_ids()
                     if fid in self.session.files]
            if paths:
                self._launch_ms2(paths)
        dialog.deleteLater()

    def open_ms2_peak(self, file_id: int, mz: float) -> None:
        loaded = self.session.files.get(file_id)
        if loaded is not None:
            self._launch_ms2([loaded.path], precursor_mz=mz)

    def _new_gnps_window(self):
        from kmzviewer.gui.gnps import GnpsWindow
        window = GnpsWindow(self)
        self._gnps_windows.append(window)
        window.closed.connect(lambda: self._gnps_windows.remove(window) if window in self._gnps_windows else None)
        return window

    def _new_spectral_window(self):
        from kmzviewer.gui.spectral import SpectralWindow
        window = SpectralWindow(self)
        self._gnps_windows.append(window)
        window.closed.connect(lambda: self._gnps_windows.remove(window) if window in self._gnps_windows else None)
        return window

    def open_gnps_peak(self, file_id, mz, rt, polarity):
        if self.runtime.closing:
            return
        loaded = self.session.files.get(file_id)
        if loaded is not None:
            window = self._new_spectral_window()
            window.show()
            window.load_source(loaded.path, mz, rt, polarity)

    def open_gnps_analysis(self):
        if self.runtime.closing:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open spectral analysis", self.runtime.directory,
                                            "KmzViewer analysis (*.kmzspectra *.kmzgnps)")
        if path:
            window = self._new_gnps_window() if path.lower().endswith(".kmzgnps") else self._new_spectral_window()
            try:
                window.open_analysis(path)
                window.show()
            except Exception as exc:
                window.close()
                QMessageBox.warning(self, "Could not open spectral analysis", str(exc))

    def _launch_ms2(self, paths, *, precursor_mz=None) -> None:
        try:
            open_ms2(paths, precursor_mz=precursor_mz)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open Kmz2Viewer", str(exc))
            return
        self.statusBar().showMessage("Opened a new Kmz2Viewer session.", 5000)

    def _open_paths_now(self, paths: list[str]) -> None:
        self._start_worker(IndexWorker(paths), self._on_indexed)

    def load_paths_sync(self, paths: list[str]) -> list[str]:
        """Open files on the calling thread. Used by tests and small files."""
        errors = []
        for path in paths:
            if self.session.__contains__(path):
                continue
            try:
                self._adopt(path, load_ms1(path))
            except Exception as exc:
                errors.append(str(exc))
        self._refresh_all()
        return errors

    def _on_indexed(self, payload: dict) -> None:
        self.runtime.opened.update(path for path, _ in payload["files"])
        for path, table in payload["files"]:
            self._adopt(path, table)
        self._refresh_all()
        if payload["errors"]:
            QMessageBox.warning(self, "Some files were not opened", "\n".join(payload["errors"]))
        names = [Path(path).name for path, _table in payload["files"]]
        if names:
            self.statusBar().showMessage("Opened " + ", ".join(names))

    def _adopt(self, path: str, table) -> None:
        tic = make_tic_trace(self.session.new_trace_id(), self.session.next_color(), table)
        self.session.add_file(path, table, tic)
        self._filter_refresh_pending = self._mz_range is not None
        self._last_input_path = Path(path).resolve()
        if self.active_trace_id is None:
            self.active_trace_id = tic.trace_id

    def open_extract_dialog(self, *, file_ids=None) -> None:
        if self._busy or not self.session.files:
            return
        files = [(loaded.file_id, loaded.name) for loaded in self.session.files.values()]
        if file_ids is None:
            file_ids = self._selected_file_ids() or None
        dialog = ExtractDialog(files, self, selected_file_ids=file_ids)
        if dialog.exec() != ExtractDialog.DialogCode.Accepted:
            return
        request = dialog.request()
        self.extract_request(
            request["text"],
            request["ppm"],
            request["polarity"],
            request["merge"],
            request["file_ids"],
        )

    def extract_request(self, text: str, ppm: float, polarity: str, merge: bool, file_ids: list[int]) -> None:
        if self._busy or self.runtime.closing:
            return
        files = [self.session.files[file_id] for file_id in file_ids if file_id in self.session.files]
        if not files:
            return
        jobs = plan_jobs(files, text, ppm, polarity, merge, self.session.new_trace_id)
        self._cache_hits_before = self._array_cache.hits
        self._cache_misses_before = self._array_cache.misses
        self._start_worker(ExtractWorker(jobs, self._array_cache), self._on_extracted)

    def extract_sync(self, text: str, ppm: float, polarity: str, merge: bool, file_ids: list[int]) -> None:
        files = [self.session.files[file_id] for file_id in file_ids if file_id in self.session.files]
        jobs = plan_jobs(files, text, ppm, polarity, merge, self.session.new_trace_id)
        if not jobs:
            return
        target = jobs[0]["target"]
        for job in jobs:
            traces = extract_eics(
                job["path"],
                job["scans"],
                target,
                job["polarity"],
                job["merge"],
                job["trace_ids"],
                job["colors"],
                cache=self._array_cache,
            )
            self.session.add_traces(job["file_id"], traces)
        self._filter_refresh_pending = self._mz_range is not None
        self._refresh_all()

    def open_bpc_dialog(self, *, file_ids=None):
        if self._busy or not self.session.files:
            return
        dialog = BpcDialog([(f.file_id, f.name) for f in self.session.files.values()], self,
                           selected_file_ids=file_ids if file_ids is not None else self._selected_file_ids() or None)
        if dialog.exec():
            self.extract_bpc_request(dialog.polarity.currentData(), dialog.checked_file_ids())
        dialog.deleteLater()

    def extract_bpc_request(self, polarity, file_ids):
        if self._busy or self.runtime.closing:
            return
        files = [self.session.files[fid] for fid in file_ids if fid in self.session.files]
        jobs = plan_bpc_jobs(files, polarity, self.session.new_trace_id)
        if jobs:
            self._cache_hits_before, self._cache_misses_before = self._array_cache.hits, self._array_cache.misses
            self._start_worker(ExtractWorker(jobs, self._array_cache), self._on_extracted)

    def extract_bpc_sync(self, polarity, file_ids):
        results = []
        for fid in file_ids:
            loaded = self.session.files[fid]
            trace = extract_bpc(loaded.path, loaded.scans, polarity, self.session.new_trace_id(), loaded.color,
                                cache=self._array_cache)
            results.append((fid, [trace]))
        self._on_extracted(results)

    def _on_extracted(self, results) -> None:
        count = 0
        for file_id, traces in results:
            loaded = self.session.files.get(file_id)
            if loaded is None:
                continue
            self.session.add_traces(file_id, traces)
            if any(trace.optical for trace in traces):
                self._dad_enabled.add(file_id)
                self.dad_file_ids.add(file_id)
                self.dad_dock.show()
                if self._selected_rt is None:
                    self._selected_rt = float(loaded.scans.dad.rt_min[0])
                    self._reference_dad_scan = (file_id, 0)
                    self._select_initial_ms_polarity(loaded)
                    if len(loaded.scans):
                        self.spectrum_file_ids.add(file_id)
                self._invalidate_comparison()
            count += len(traces)
        self._filter_refresh_pending = self._mz_range is not None
        self._refresh_all()
        self._queue_comparison()
        self.statusBar().showMessage(self._cache_message(f"Added {count} chromatograms."))

    def _display_trace(self, trace):
        if self._mz_range is None or trace.optical:
            return trace
        values = self._filter_values.get(trace.trace_id)
        if values is None:
            return None
        description = range_label(self._mz_range)
        return replace(trace, y=values, label=f"{trace.label} [{description}]",
                       detail=f"{trace.detail} Display filter: {description}, inclusive.")

    def apply_mz_filter(self, lower, upper):
        if self._busy or self.runtime.closing:
            return
        bounds = mz_range(lower, upper)
        self._filter_refresh_pending = False
        values, jobs = {}, []
        for loaded in self.session.files.values():
            missing = []
            for trace in loaded.traces:
                if trace.optical:
                    continue
                cached = self._filter_values.get(trace.trace_id) if bounds == self._mz_range else None
                if cached is None:
                    cached = self._filter_cache.get(trace.trace_id, bounds)
                if cached is None:
                    missing.append(trace)
                else:
                    values[trace.trace_id] = cached
            if missing:
                jobs.append(dict(path=loaded.path, name=loaded.name, scans=loaded.scans, traces=missing))
        self._filter_token += 1
        token = self._filter_token
        if not jobs:
            self._on_filtered(token, bounds, values, True)
            return
        self._filter_running = True
        previous = range_label(self._mz_range) if self._mz_range else "unfiltered"
        self.mz_filter.status.setText(f"Applying {range_label(bounds)}… Current view: {previous}.")
        self._cache_hits_before, self._cache_misses_before = self._array_cache.hits, self._array_cache.misses
        self._start_worker(FilterWorker(jobs, bounds, self._array_cache),
                           lambda result: self._on_filtered(token, bounds, values | result, False))

    def _on_filtered(self, token, bounds, values, cached):
        if token != self._filter_token or self.runtime.closing:
            return
        existing = {trace.trace_id for _, trace in self.session.ordered_traces()}
        self._mz_range = bounds
        self._filter_values = {tid: y for tid, y in values.items() if tid in existing}
        for tid, y in self._filter_values.items():
            self._filter_cache.put(tid, bounds, y)
        self._filter_running = False
        self.mz_filter.start.setText(str(bounds.lower))
        self.mz_filter.stop.setText(str(bounds.upper))
        self.mz_filter.status.setText(f"Active: {range_label(bounds)} (inclusive). Wavelength traces and spectra are unchanged."
                                      + (" Reused cached results." if cached else ""))
        self._refresh_all()
        self.statusBar().showMessage(self._cache_message("Applied m/z filter.") if not cached else "Applied cached m/z filter.")

    def clear_mz_filter(self):
        self._filter_token += 1
        if self._filter_running:
            self.runtime.cancel(clear_queue=False)
        self._filter_running = False
        self._filter_refresh_pending = False
        self._mz_range = None
        self._filter_values.clear()
        self.mz_filter.start.clear()
        self.mz_filter.stop.clear()
        self.mz_filter.status.setText("No m/z filter.")
        self._refresh_all()

    def _ensure_mz_filter(self):
        if self._filter_refresh_pending and self._mz_range is not None and not self._busy and not self.runtime.closing:
            self.apply_mz_filter(self._mz_range.lower, self._mz_range.upper)

    def _drop_filter_traces(self, trace_ids):
        self._filter_cache.drop_traces(trace_ids)
        for tid in trace_ids:
            self._filter_values.pop(tid, None)

    def open_wavelength_dialog(self, *, total=False, file_ids=None):
        if self._busy or not self.session.files:
            return
        dialog = WavelengthDialog([(f.file_id, f.name) for f in self.session.files.values()], self,
                                  total=total, selected_file_ids=file_ids or self._selected_file_ids() or None)
        if dialog.exec():
            self.extract_wavelength_request(dialog.settings(), dialog.checked_file_ids())
        dialog.deleteLater()

    def _wavelength_jobs(self, settings, file_ids):
        jobs, missing = [], []
        for fid in file_ids:
            loaded = self.session.files.get(fid)
            if loaded is None:
                continue
            if loaded.scans.dad is None or loaded.scans.dad_error:
                missing.append(loaded.name + (': ' + loaded.scans.dad_error if loaded.scans.dad_error else ''))
                continue
            jobs.append(dict(path=loaded.path, name=loaded.name, file_id=fid, scans=loaded.scans.dad,
                             wavelengths=settings, trace_ids=[self.session.new_trace_id() for _ in settings], color=loaded.color))
        return jobs, missing

    def extract_wavelength_request(self, settings, file_ids):
        if self._busy or self.runtime.closing:
            return
        jobs, missing = self._wavelength_jobs(settings, file_ids)
        if missing:
            QMessageBox.information(self, 'DAD data unavailable', 'No usable DAD spectra in:\n' + '\n'.join(missing) +
                                    '\n\nReconvert with the DAD sidecars to include them. Other selected files will be extracted.')
        if jobs:
            self._cache_hits_before, self._cache_misses_before = self._array_cache.hits, self._array_cache.misses
            self._start_worker(ExtractWorker(jobs, self._array_cache), self._on_extracted)

    def extract_wavelength_sync(self, settings, file_ids):
        jobs, missing = self._wavelength_jobs(settings, file_ids)
        if missing:
            raise ValueError('No usable DAD spectra: ' + ', '.join(missing))
        results = [(job['file_id'], extract_wavelengths(job['path'], job['scans'], settings, job['trace_ids'], job['color'],
                                                       cache=self._array_cache)) for job in jobs]
        self._on_extracted(results)

    def _select_initial_ms_polarity(self, loaded):
        finite = np.flatnonzero(np.isfinite(loaded.scans.rt_min))
        if finite.size:
            row = int(finite[np.argmin(np.abs(loaded.scans.rt_min[finite] - self._selected_rt))])
            self._selected_polarity = int(loaded.scans.polarity[row])

    def open_export_dialog(self, single: bool) -> None:
        if not self._filter_complete():
            QMessageBox.information(self, "Export SVG", "Apply or clear the m/z filter before exporting newly added traces.")
            return
        if single:
            found = None if self.active_trace_id is None else self.session.find_trace(self.active_trace_id)
            if found is None:
                QMessageBox.information(self, "Export SVG", "Select a chromatogram first.")
                return
            loaded, trace = found
            trace = self._display_trace(trace)
            rows = [(trace.trace_id, f"{loaded.name} — {trace.label}", True)]
        else:
            rows = [
                (trace.trace_id, f"{loaded.name} — {trace.label}", trace.visible)
                for loaded, original in self.session.ordered_traces(visible_only=False)
                if (trace := self._display_trace(original)) is not None
            ]
        if not rows:
            QMessageBox.information(self, "Export SVG", "Open an mzML file first.")
            return
        current = self.plots.current_x_range() or self.plots.data_x_range() or (0.0, 1.0)
        data = {}
        for trace_id, _label, _checked in rows:
            loaded, trace = self.session.find_trace(trace_id)
            trace = self._display_trace(trace)
            data[trace_id] = ExportTrace(loaded.name, trace.label, trace.color, trace.rt, trace.y, trace.unit,
                mz_filter=(self._mz_range.lower, self._mz_range.upper) if self._mz_range and not trace.optical else None)
        default_path = self._last_input_path.with_suffix(".svg") if self._last_input_path else Path("chromatograms.svg")
        dialog = ExportDialog(rows, current, trace_data=data, single_trace=single,
                              default_path=default_path, parent=self)
        dialog.exec()
        if dialog.saved_path:
            self.statusBar().showMessage(f"Exported {dialog.saved_path}")

    def export_trace_ids(self, trace_ids: list[int], path: str, options: ExportOptions) -> None:
        rows = []
        for trace_id in trace_ids:
            found = self.session.find_trace(trace_id)
            if found is None:
                continue
            loaded, trace = found
            trace = self._display_trace(trace)
            if trace is None:
                raise ValueError("Apply or clear the m/z filter before exporting newly added traces.")
            rows.append(ExportTrace(loaded.name, trace.label, trace.color, trace.rt, trace.y, trace.unit,
                mz_filter=(self._mz_range.lower, self._mz_range.upper) if self._mz_range and not trace.optical else None))
        export_chromatograms(path, rows, options)

    def spectrum_export_dialog(self, optical=False):
        panel = self.dad_spectrum if optical else self.spectrum
        data, rows = {}, []
        for frame in panel._frames:
            if not frame.mz.size or frame.message:
                continue
            x, y = (frame.mz, frame.intensity) if frame.centroid else frame.profile_curve
            label = f'{"DAD" if optical else "MS1"} {frame.rt:.4f} min' if frame.rt is not None else ('DAD' if optical else 'MS1')
            data[frame.file_id] = ExportTrace(frame.name, label, frame.color, x.copy(), y.copy(), 'mAU' if optical else 'counts', frame.centroid)
            rows.append((frame.file_id, f'{frame.name} ({label})', True))
        if not rows:
            return None
        bounds = tuple(panel.plot.viewRange()[0])
        default = self._last_input_path.with_suffix('.dad-spectra.svg' if optical else '.ms1-spectra.svg') if self._last_input_path else Path('spectra.svg')
        return ExportDialog(rows, bounds, trace_data=data, default_path=default, parent=self,
                            spectrum='dad' if optical else 'ms1', initial_layout=panel._mode)

    def open_spectrum_export(self, optical=False):
        dialog = self.spectrum_export_dialog(optical)
        if dialog is None:
            QMessageBox.information(self, 'Export spectra', 'Inspect a spectrum first.')
            return
        dialog.exec()
        dialog.deleteLater()

    def activate_trace(self, trace_id: int, retarget: bool = True) -> None:
        if self.session.find_trace(trace_id) is None:
            return
        previous = self.walk_trace_id
        self.active_trace_id = trace_id
        self.plots.set_active(trace_id)
        self._mark_tree_active()
        self._sync_actions()
        if retarget and self.walk_mode and previous != trace_id:
            self._retarget_walk(trace_id)

    def inspect_rt(self, trace_id: int, rt: float) -> None:
        """A double-click includes the source once and returns to Walk mode."""
        found = self.session.find_trace(trace_id)
        if found is None or not found[1].rt.size:
            return
        self.walk_action.setChecked(True)
        self.spectrum_dock.show()
        loaded, trace = found
        finite = np.flatnonzero(np.isfinite(trace.rt))
        if not finite.size or not np.isfinite(rt):
            return
        index = int(finite[np.argmin(np.abs(trace.rt[finite] - rt))])
        self._show_walk_index(loaded, trace, index, add=True)

    def _navigate_rt(self, trace_id: int, rt: float) -> None:
        if self._selected_rt is None:
            if self.walk_mode:
                self.inspect_rt(trace_id, rt)
            return
        found = self.session.find_trace(trace_id)
        if found is None or not np.isfinite(rt):
            return
        loaded, trace = found
        finite = np.flatnonzero(np.isfinite(trace.rt))
        if finite.size:
            index = int(finite[np.argmin(np.abs(trace.rt[finite] - rt))])
            self._show_walk_index(loaded, trace, index)

    def walk_by(self, step: int) -> None:
        if not self.walk_mode:
            return
        found = None if self.walk_trace_id is None else self.session.find_trace(self.walk_trace_id)
        if found is None or self.walk_i is None:
            if self.active_trace_id is not None:
                self._retarget_walk(self.active_trace_id)
            return
        loaded, trace = found
        if trace.rt.size:
            index = min(max(self.walk_i + step, 0), int(trace.rt.size) - 1)
            self._show_walk_index(loaded, trace, index)

    def _retarget_walk(self, trace_id: int) -> None:
        found = self.session.find_trace(trace_id)
        if found is None:
            return
        loaded, trace = found
        if trace.rt.size:
            rt = self._selected_rt if self._selected_rt is not None else float(trace.rt[0])
            self._navigate_rt(trace_id, rt)

    def _show_walk_index(self, loaded, trace, index: int, *, add=False) -> None:
        row = int(trace.row[index])
        scans = loaded.scans.dad if trace.optical else loaded.scans
        rt = float(scans.rt_min[row])
        if not np.isfinite(rt):
            return
        first = self._selected_rt is None
        self.walk_trace_id, self.walk_i = trace.trace_id, index
        self.active_trace_id = trace.trace_id
        self._selected_rt = rt
        if trace.optical:
            self._reference_scan = None
            self._reference_dad_scan = (loaded.file_id, row)
            if first or not self.spectrum_file_ids:
                self._select_initial_ms_polarity(loaded)
        else:
            self._selected_polarity = int(loaded.scans.polarity[row])
            self._reference_scan = (loaded.file_id, row)
            self._reference_dad_scan = None
        if (add or first) and len(loaded.scans) and loaded.file_id not in self.spectrum_file_ids:
            self.spectrum_file_ids.add(loaded.file_id)
            self._invalidate_comparison()
        if (add or first) and loaded.file_id in self._dad_enabled and loaded.file_id not in self.dad_file_ids:
            self.dad_file_ids.add(loaded.file_id)
            self._invalidate_comparison()
        self.plots.set_walk_clicks(True)
        self.plots.set_walk_rt(rt)
        self.plots.set_active(trace.trace_id)
        self._queue_comparison()
        self._mark_tree_active()
        self._sync_spectrum_eyes()
        self._sync_actions()

    def toggle_spectrum(self, file_id: int) -> None:
        self.set_spectrum_visibility({file_id}, detector="ms1", visible=None)

    def set_spectrum_visibility(self, file_ids, *, detector, visible):
        """Change each source once, then invalidate and request one comparison."""
        included = self.spectrum_file_ids if detector == "ms1" else self.dad_file_ids
        chosen = set(file_ids)
        changed = False
        for file_id, loaded in self.session.files.items():
            if file_id not in chosen or (detector == "dad" and file_id not in self._dad_enabled):
                continue
            show = file_id not in included if visible is None else visible
            if show == (file_id in included):
                continue
            if not show:
                included.remove(file_id)
                changed = True
                continue
            if detector == "ms1":
                finite = np.flatnonzero(np.isfinite(loaded.scans.rt_min))
                if not finite.size:
                    self.statusBar().showMessage("This file has no MS1 scans with a finite retention time.")
                    continue
                self.spectrum_dock.show()
                if self._selected_rt is None:
                    row = int(finite[0])
                    self._selected_rt = float(loaded.scans.rt_min[row])
                    self._selected_polarity = int(loaded.scans.polarity[row])
                    self._reference_scan = (file_id, row)
                    if loaded.traces:
                        self.walk_trace_id = loaded.traces[0].trace_id
                        self.walk_i = int(np.argmin(np.abs(loaded.traces[0].rt - self._selected_rt))) if loaded.traces[0].rt.size else None
                        self.active_trace_id = self.walk_trace_id
                    self.walk_action.setChecked(True)
                    self.plots.set_walk_clicks(True)
                    self.plots.set_walk_rt(self._selected_rt)
            else:
                self.dad_dock.show()
            included.add(file_id)
            changed = True
        if changed:
            self._invalidate_comparison()
            self._queue_comparison()
        self._sync_spectrum_eyes()
        self.plots.set_active(self.active_trace_id)
        self._mark_tree_active()
        self._sync_actions()

    def _sync_spectrum_eyes(self) -> None:
        for file_id, button in self._all_spectrum_eyes:
            button.set_visible(file_id in self.spectrum_file_ids)
        for file_id, button in self._all_dad_eyes:
            button.set_visible(file_id in self.dad_file_ids)

    def toggle_dad_spectrum(self, file_id):
        self.set_spectrum_visibility({file_id}, detector="dad", visible=None)

    def _invalidate_comparison(self) -> None:
        """Membership/lifecycle changes must never resurrect a closed or hidden file."""
        self._inflight_comparison = None
        self._queued_comparison = None
        self._requested_comparison_key = None
        retained = [frame for frame in self.spectrum._frames if frame.file_id in self.spectrum_file_ids]
        if retained:
            self.spectrum.show_frames(retained, self.spectrum.caption.text())
        else:
            self.spectrum.clear_scan("Choose an MS1 eye or double-click a chromatogram.")
        retained_dad = [frame for frame in self.dad_spectrum._frames if frame.file_id in self.dad_file_ids]
        self.dad_spectrum.show_frames(retained_dad, self.dad_spectrum.caption.text())

    def _queue_comparison(self) -> None:
        if self._selected_rt is None or not (self.spectrum_file_ids or self.dad_file_ids):
            return
        specs = []
        reference = self._reference_scan
        for loaded in self.session.files.values():
            if loaded.file_id not in self.spectrum_file_ids:
                continue
            match = match_scan(loaded.scans, self._selected_rt, self._selected_polarity)
            row = reference[1] if reference is not None and reference[0] == loaded.file_id else match.row
            specs.append((loaded.file_id, row, match.message if row is None else ""))
        for loaded in self.session.files.values():
            if loaded.file_id not in self.dad_file_ids:
                continue
            scans = loaded.scans.dad
            row, message = None, ''
            if scans is None or not len(scans):
                message = 'No DAD scans'
            elif self._reference_dad_scan is not None and self._reference_dad_scan[0] == loaded.file_id:
                row = self._reference_dad_scan[1]
            elif self._selected_rt < scans.rt_min.min() or self._selected_rt > scans.rt_min.max():
                message = 'Outside DAD run range'
            else:
                row = int(np.argmin(np.abs(scans.rt_min - self._selected_rt)))
            specs.append(((loaded.file_id, 'dad'), row, message))
        key = (self._selected_rt, self._selected_polarity, tuple(specs))
        if key == self._requested_comparison_key:
            return
        self._requested_comparison_key = key
        self._queued_comparison = (key, specs)
        self.spectrum.caption.setText(f"RT {self._selected_rt:.4f} min · loading…")
        self._dispatch_comparison()

    def _dispatch_comparison(self) -> None:
        if self.runtime.closing or self._inflight_comparison is not None or self._queued_comparison is None:
            return
        key, specs = self._queued_comparison
        self._queued_comparison = None
        self._token += 1
        self._inflight_comparison = (self._token, key, specs)
        requests = []
        for identity, row, _message in specs:
            if row is not None:
                optical = isinstance(identity, tuple)
                file_id = identity[0] if optical else identity
                loaded = self.session.files[file_id]
                scans = loaded.scans.dad if optical else loaded.scans
                sparse = loaded.scans.sparse_profile[row] if not optical and loaded.scans.sparse_profile else None
                requests.append((identity, loaded.path, int(scans.spectrum_index[row]), sparse))
        self._spectrum_worker.request_batch(self._token, requests)

    def _on_comparison(self, token: int, payload) -> None:
        current = self._inflight_comparison
        if self.runtime.closing or current is None or token != current[0]:
            return
        _token, key, specs = current
        self._inflight_comparison = None
        frames, dad_frames = [], []
        for identity, row, message in specs:
            optical = isinstance(identity, tuple)
            file_id = identity[0] if optical else identity
            loaded = self.session.files.get(file_id)
            if loaded is None or file_id not in (self.dad_file_ids if optical else self.spectrum_file_ids):
                continue
            mz = intensity = np.array([], dtype=np.float64)
            connections = None
            profile_curve = None
            centroid = False
            rt = None
            polarity = key[1]
            caption = loaded.name
            if row is not None:
                scans = loaded.scans.dad if optical else loaded.scans
                centroid = not optical and int(scans.centroid[row]) == 1
                rt = float(scans.rt_min[row])
                polarity = 0 if optical else int(scans.polarity[row])
                mode = 'DAD' if optical else 'MS1 centroid' if centroid else 'MS1 profile'
                caption = (f"{loaded.name} · {mode} · RT {rt:.4f} min · "
                           f"spectrum {int(scans.spectrum_index[row])} · {scans.scan_id[row]}")
                result = payload.get(identity)
                if isinstance(result, Exception) or result is None:
                    message = "Could not read scan"
                    caption += f" · {result}"
                else:
                    mz, intensity, connections, profile_curve = result
                    if not mz.size:
                        message = "Empty scan"
            (dad_frames if optical else frames).append(SpectrumFrame(file_id, loaded.name, loaded.color, caption, mz, intensity,
                                        centroid, connections, message, rt, polarity, profile_curve, optical))
        order = {fid: i for i, fid in enumerate(self.session.files)}
        frames.sort(key=lambda frame: order[frame.file_id])
        dad_frames.sort(key=lambda frame: order[frame.file_id])
        summary = frames[0].caption if len(frames) == 1 else f"Counts · RT {key[0]:.4f} min · {len(frames)} spectra"
        if self._queued_comparison is not None:
            summary += " · loading…"
        self.spectrum.show_frames(frames, summary)
        self.dad_spectrum.show_frames(dad_frames, f'Absorbance · RT {key[0]:.4f} min · {len(dad_frames)} spectra')
        # Keep one decode batch in flight and one latest request: sustained drags
        # still display complete intermediate frames, then land on the final RT.
        self._dispatch_comparison()

    def _set_walk(self, enabled: bool) -> None:
        self.walk_mode = enabled
        self.plots.set_walk_clicks(enabled or self._selected_rt is not None)
        self.left_shortcut.setEnabled(enabled)
        self.right_shortcut.setEnabled(enabled)
        if enabled:
            self.plots.set_mouse_mode("walk")

    def _set_overlay(self, enabled: bool) -> None:
        self.plots.set_mode("overlay" if enabled else "stacked")

    def remove_trace(self, trace_id: int) -> None:
        self.remove_traces({trace_id})

    def open_manage_chromatograms(self):
        if self._busy or not self.session.ordered_traces():
            return
        dialog = ManageChromatogramsDialog(self.session, self, spectrum_file_ids=self.spectrum_file_ids,
                                          dad_file_ids=self.dad_file_ids, dad_enabled=self._dad_enabled)
        def apply(action, trace_ids):
            if action == "remove":
                self.remove_traces(trace_ids)
            elif action in {"show_plots", "hide_plots"}:
                self.set_trace_visibility(trace_ids, visible=action == "show_plots")
            else:
                file_ids = {loaded.file_id for loaded, trace in self.session.ordered_traces() if trace.trace_id in trace_ids}
                operation, detector = action.split("_")
                self.set_spectrum_visibility(file_ids, detector=detector,
                                             visible=None if operation == "toggle" else operation == "show")
            dialog.refresh(self.spectrum_file_ids, self.dad_file_ids, self._dad_enabled)
        dialog.action_requested.connect(apply)
        dialog.exec()
        dialog.deleteLater()

    def remove_traces(self, trace_ids: set[int]) -> None:
        self._drop_filter_traces(trace_ids)
        self.session.remove_traces(trace_ids)
        if self.active_trace_id in trace_ids:
            self.active_trace_id = None
        if self.walk_trace_id in trace_ids:
            self.walk_trace_id = None
            self.walk_i = None
        self._refresh_all()

    def restore_tic(self, file_id: int) -> None:
        loaded = self.session.files.get(file_id)
        if loaded is None or self.session.has_tic(file_id):
            return
        tic = make_tic_trace(self.session.new_trace_id(), loaded.color, loaded.scans)
        tic.file_id = file_id
        loaded.traces.insert(0, tic)
        self._filter_refresh_pending = self._mz_range is not None
        self.active_trace_id = tic.trace_id
        self._refresh_all()

    def close_active_file(self) -> None:
        file_id = self._active_file_id()
        if file_id is not None:
            self.close_file(file_id)

    def close_selected_files(self) -> None:
        file_ids = self._selected_file_ids()
        if not file_ids:
            active = self._active_file_id()
            file_ids = [active] if active is not None else []
        self.close_files(file_ids)

    def close_file(self, file_id: int) -> None:
        self.close_files([file_id])

    def close_files(self, file_ids) -> None:
        chosen = set(file_ids) & self.session.files.keys()
        if not chosen:
            return
        self.runtime.cancel()
        trace_ids = set()
        for file_id in chosen:
            loaded = self.session.files[file_id]
            trace_ids.update(trace.trace_id for trace in loaded.traces)
            self.runtime.opened.discard(loaded.path)
            self._array_cache.drop_path(loaded.path)
            self._spectrum_worker.close_path(loaded.path)
            self.session.remove_file(file_id)
        self._drop_filter_traces(trace_ids)
        if self._reference_scan is not None and self._reference_scan[0] in chosen:
            self._reference_scan = None
        self.spectrum_file_ids.difference_update(chosen)
        self.dad_file_ids.difference_update(chosen)
        self._dad_enabled.difference_update(chosen)
        if self._reference_dad_scan is not None and self._reference_dad_scan[0] in chosen:
            self._reference_dad_scan = None
        if not self._dad_enabled:
            self.dad_dock.hide()
        self._invalidate_comparison()
        if self.active_trace_id in trace_ids:
            self.active_trace_id = None
        if self.walk_trace_id in trace_ids:
            self.walk_trace_id = None
            self.walk_i = None
        if not self.session.files:
            self._selected_rt = None
            self.plots.set_walk_clicks(self.walk_mode)
        self._queue_comparison()
        self._refresh_all()

    def _cache_message(self, prefix: str) -> str:
        hits = self._array_cache.hits - self._cache_hits_before
        misses = self._array_cache.misses - self._cache_misses_before
        megabytes = self._array_cache.nbytes / (1024 * 1024)
        return f"{prefix} Spectrum cache: {hits} reused, {misses} decoded, {megabytes:.0f} MB retained."

    def _drop_pending_spectrum(self) -> None:
        self._inflight_comparison = self._queued_comparison = None
        self._requested_comparison_key = None
        self.spectrum.clear_scan("Double-click a chromatogram to inspect an MS1 scan.")
        self.dad_spectrum.clear_scan('Extract a wavelength chromatogram to inspect DAD spectra.')

    def _active_file_id(self) -> int | None:
        current = self.tree.currentItem()
        data = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        if data and data[0] == "file" and data[1] in self.session.files:
            return data[1]
        if self.active_trace_id is not None:
            found = self.session.find_trace(self.active_trace_id)
            if found is not None:
                return found[0].file_id
        if self.session.files:
            return next(iter(self.session.files))
        return None

    def _refresh_all(self) -> None:
        self._refresh_plots()
        self._refresh_navigator()
        self._sync_actions()
        if self._filter_refresh_pending:
            QTimer.singleShot(0, self._ensure_mz_filter)

    def _refresh_plots(self) -> None:
        rows = []
        units = {}
        for loaded, original in self.session.ordered_traces(visible_only=True):
            trace = self._display_trace(original)
            if trace is not None:
                rows.append((trace.trace_id, f"{loaded.name}    {trace.label}", trace.color, trace.rt, trace.y))
                units[trace.trace_id] = trace.unit
        self.plots.set_traces(rows, units=units)
        self.plots.set_emphasized(self.session.emphasized)
        if self.active_trace_id is None or self.session.find_trace(self.active_trace_id) is None:
            visible = self.session.ordered_traces(visible_only=True)
            self.active_trace_id = visible[0][1].trace_id if visible else None
        self.plots.set_active(self.active_trace_id)
        self.plots.set_walk_rt(self._selected_rt)

    def set_grouping(self, mode) -> None:
        if mode not in {"file", "type"} or mode == self.session.grouping:
            return
        self.session.grouping = mode
        with QSignalBlocker(self.group_by):
            self.group_by.setCurrentIndex(self.group_by.findData(mode))
        self._refresh_all()

    def _navigator_items(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            yield top
            for j in range(top.childCount()):
                yield top.child(j)

    def _add_spectrum_eyes(self, item, loaded):
        fid = loaded.file_id
        eye = SpectrumEye(loaded.name, fid in self.spectrum_file_ids,
                          lambda _checked, fid=fid: self.toggle_spectrum(fid), self.tree)
        self.tree.setItemWidget(item, 1, eye)
        self._spectrum_eyes.setdefault(fid, eye)
        self._all_spectrum_eyes.append((fid, eye))
        if fid in self._dad_enabled:
            eye = SpectrumEye(loaded.name, fid in self.dad_file_ids,
                              lambda _checked, fid=fid: self.toggle_dad_spectrum(fid), self.tree, detector='DAD')
            self.tree.setItemWidget(item, 2, eye)
            self._dad_eyes.setdefault(fid, eye)
            self._all_dad_eyes.append((fid, eye))

    def _refresh_navigator(self) -> None:
        selected = {tuple(item.data(0, Qt.ItemDataRole.UserRole)) for item in self.tree.selectedItems()}
        current = self.tree.currentItem()
        current_key = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        if self.tree.grouping != self.session.grouping:
            # Preserve the selected traces when parents change their meaning.
            selected = {( "trace", tid) for tid in self._selected_trace_ids()} | {
                ("file", fid) for fid in self._selected_file_ids() if not self.session.files[fid].traces}
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            self._navigator_expanded[(self.tree.grouping, tuple(top.data(0, Qt.ItemDataRole.UserRole)))] = top.isExpanded()
        scroll = self.tree.verticalScrollBar().value()
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.grouping = self.session.grouping
        self.tree.setDragEnabled(self.session.grouping == "file")
        self.tree.setToolTip("Shift-click for a range; Command/Ctrl-click for individual rows. Right-click for group actions. " +
                            ("Drag file names to reorder." if self.session.grouping == "file" else
                             "Matching extraction settings share a group. Switch to file grouping to drag files."))
        self._spectrum_eyes.clear()
        self._dad_eyes.clear()
        self._all_spectrum_eyes.clear()
        self._all_dad_eyes.clear()
        self.tree.setColumnHidden(2, not bool(self._dad_enabled))
        items = {}

        def add_item(parent, key, label, tooltip, color=None):
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            item.setToolTip(0, tooltip)
            item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsDropEnabled)
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            if color:
                item.setIcon(0, color_icon(color))
            items[key] = item
            return item

        def source_name(loaded):
            name = loaded.name
            if self.session.grouping == "type":
                for suffix in (".sp.mzml", ".mzml"):
                    if name.lower().endswith(suffix):
                        name = name[:-len(suffix)]
                        break
            return ("SP · " if any(loaded.scans.sparse_profile) else "") + name

        def source_tooltip(loaded):
            return ("Sparse Profile. " + SP_HELP + "\n" if any(loaded.scans.sparse_profile) else "") + loaded.path

        if self.session.grouping == "file":
            for loaded in self.session.files.values():
                top = add_item(None, ("file", loaded.file_id), source_name(loaded), source_tooltip(loaded), loaded.color)
                self._add_spectrum_eyes(top, loaded)
                for trace in loaded.traces:
                    add_item(top, ("trace", trace.trace_id), trace.label, trace.detail, trace.color)
        else:
            for key, rows in self.session.trace_groups():
                trace = rows[0][1]
                top = add_item(None, ("group", key), trace.group_label, trace.detail)
                for loaded, trace in rows:
                    child = add_item(top, ("trace", trace.trace_id), source_name(loaded),
                                     source_tooltip(loaded) + "\n" + trace.detail, trace.color)
                    self._add_spectrum_eyes(child, loaded)
            empty = [loaded for loaded in self.session.files.values() if not loaded.traces]
            if empty:
                top = add_item(None, ("group", "empty"), "Files without chromatograms",
                               "These files remain open for extraction and spectrum inspection.")
                top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                for loaded in empty:
                    child = add_item(top, ("file", loaded.file_id), source_name(loaded), source_tooltip(loaded), loaded.color)
                    child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    self._add_spectrum_eyes(child, loaded)
        self._sync_tree_checks()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            key = (self.tree.grouping, tuple(top.data(0, Qt.ItemDataRole.UserRole)))
            top.setExpanded(self._navigator_expanded.get(key, True))
        if current_key and tuple(current_key) in items:
            self.tree.setCurrentItem(items[tuple(current_key)], 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        for key in selected:
            if key in items:
                items[key].setSelected(True)
        self.tree.verticalScrollBar().setValue(scroll)
        self._mark_tree_active()
        self.tree.blockSignals(False)

    def _set_parent_check(self, top: QTreeWidgetItem) -> None:
        states = [top.child(index).checkState(0) for index in range(top.childCount())]
        if states and all(state == Qt.CheckState.Unchecked for state in states):
            top.setCheckState(0, Qt.CheckState.Unchecked)
        elif states and any(state == Qt.CheckState.Unchecked for state in states):
            top.setCheckState(0, Qt.CheckState.PartiallyChecked)
        else:
            top.setCheckState(0, Qt.CheckState.Checked)

    def _mark_tree_active(self) -> None:
        from PySide6.QtGui import QFont

        blocked = self.tree.blockSignals(True)
        for top_index in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(top_index)
            for child_index in range(top.childCount()):
                child = top.child(child_index)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                font = QFont(child.font(0))
                selected = bool(data) and data[0] == "trace" and data[1] == self.active_trace_id
                font.setBold(selected)
                if child.font(0).bold() != selected:
                    child.setFont(0, font)
        self.tree.blockSignals(blocked)

    def _item_trace_ids(self, item) -> set[int]:
        kind, ident = item.data(0, Qt.ItemDataRole.UserRole)
        if kind == "trace":
            return {ident} if self.session.find_trace(ident) else set()
        if kind == "file":
            loaded = self.session.files.get(ident)
            return {t.trace_id for t in loaded.traces} if loaded else set()
        return set().union(*(self._item_trace_ids(item.child(i)) for i in range(item.childCount())))

    def _item_file_ids(self, item) -> set[int]:
        kind, ident = item.data(0, Qt.ItemDataRole.UserRole)
        if kind == "file":
            return {ident} if ident in self.session.files else set()
        if kind == "trace":
            found = self.session.find_trace(ident)
            return {found[0].file_id} if found else set()
        return set().union(*(self._item_file_ids(item.child(i)) for i in range(item.childCount())))

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        for tid in self._item_trace_ids(item):
            self.session.set_visible(tid, item.checkState(0) == Qt.CheckState.Checked)
        self._sync_tree_checks()
        self._refresh_plots()

    def _on_selection(self, item, _previous) -> None:
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "trace":
            self.activate_trace(data[1], retarget=True)

    def _selected_trace_ids(self) -> set[int]:
        return set().union(*(self._item_trace_ids(item) for item in self.tree.selectedItems()))

    def _selected_file_ids(self) -> list[int]:
        selected = set().union(*(self._item_file_ids(item) for item in self.tree.selectedItems()))
        return [fid for fid in self.session.files if fid in selected]

    def _sync_tree_checks(self) -> None:
        traces = {trace.trace_id: trace for _, trace in self.session.ordered_traces()}
        with QSignalBlocker(self.tree):
            for item in self._navigator_items():
                kind, ident = item.data(0, Qt.ItemDataRole.UserRole)
                if kind == "trace" and ident in traces:
                    item.setCheckState(0, Qt.CheckState.Checked if traces[ident].visible else Qt.CheckState.Unchecked)
            for i in range(self.tree.topLevelItemCount()):
                top = self.tree.topLevelItem(i)
                if top.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    self._set_parent_check(top)

    def set_trace_visibility(self, trace_ids: set[int], *, only=False, visible=False) -> None:
        for _, trace in self.session.ordered_traces():
            if only:
                trace.visible = trace.trace_id in trace_ids
            elif trace.trace_id in trace_ids:
                trace.visible = visible
        self._sync_tree_checks()
        self._refresh_plots()

    def _context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if not item.isSelected():
            self.tree.setCurrentItem(item, 0, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        menu = self._selection_menu(item)
        menu.exec(self.tree.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def _selection_menu(self, item) -> QMenu:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        trace_ids = self._selected_trace_ids()
        file_ids = self._selected_file_ids()
        menu = QMenu(self)
        extract = menu.addAction("Extract EIC…", lambda: self.open_extract_dialog(file_ids=file_ids))
        extract.setEnabled(bool(file_ids) and not self._busy)
        bpc = menu.addAction("Extract BPC…", lambda: self.open_bpc_dialog(file_ids=file_ids))
        bpc.setEnabled(bool(file_ids) and not self._busy)
        for title, total in (('Extract TWC…', True), ('Extract EWC…', False)):
            action = menu.addAction(title, lambda total=total: self.open_wavelength_dialog(total=total, file_ids=file_ids))
            action.setEnabled(bool(file_ids) and not self._busy)
        for title, callback in (
            ("Show only selected", lambda: self.set_trace_visibility(trace_ids, only=True)),
            ("Hide selected", lambda: self.set_trace_visibility(trace_ids)),
            ("Change selected color…", lambda: self.choose_trace_color(trace_ids)),
            ("Remove selected chromatograms", lambda: self.remove_traces(trace_ids)),
        ):
            action = menu.addAction(title, callback)
            action.setEnabled(bool(trace_ids))
        menu.addSeparator()
        if len(file_ids) == 1:
            file_id = file_ids[0]
            menu.addAction("Choose source color…", lambda: self.choose_file_color(file_id))
            if not self.session.has_tic(file_id):
                menu.addAction("Restore TIC MS1", lambda: self.restore_tic(file_id))
        close = menu.addAction("Close file" if len(file_ids) == 1 else "Close selected files",
                               lambda: self.close_files(file_ids))
        close.setEnabled(bool(file_ids) and not self._busy)
        return menu

    def choose_trace_color(self, trace_ids: set[int]) -> None:
        traces = [trace for _, trace in self.session.ordered_traces() if trace.trace_id in trace_ids]
        if not traces:
            return
        color = QColorDialog.getColor(QColor(traces[0].color), self, "Selected chromatogram color")
        if color.isValid():
            self.set_trace_colors(trace_ids, color.name())

    def set_trace_colors(self, trace_ids: set[int], color: str) -> None:
        self.session.set_trace_colors(trace_ids, color)
        self.plots.update_colors({tid: color for tid in trace_ids})
        self._sync_tree_colors()

    def _sync_tree_colors(self) -> None:
        with QSignalBlocker(self.tree):
            for item in self._navigator_items():
                kind, ident = item.data(0, Qt.ItemDataRole.UserRole)
                if kind == "file":
                    item.setIcon(0, color_icon(self.session.files[ident].color))
                elif kind == "trace":
                    found = self.session.find_trace(ident)
                    if found:
                        item.setIcon(0, color_icon(found[1].color))

    def move_files(self, file_ids: list[int], before: int) -> None:
        if self.session.move_files(file_ids, before):
            self._files_reordered()

    def sort_files(self) -> None:
        if self.session.sort_files():
            self._files_reordered()

    def _files_reordered(self) -> None:
        # Reuse decoded frames. Pending callbacks adopt the latest display order.
        self.spectrum.reorder_files(self.session.files)
        self.dad_spectrum.reorder_files(self.session.files)
        self._refresh_all()

    def choose_file_color(self, file_id: int) -> None:
        loaded = self.session.files.get(file_id)
        if loaded is None:
            return
        color = QColorDialog.getColor(QColor(loaded.color), self, "Chromatogram color — " + loaded.name)
        if color.isValid():
            self.set_file_color(file_id, color.name())

    def set_file_color(self, file_id: int, color: str) -> None:
        self.session.set_file_color(file_id, color)
        self.plots.update_colors({trace.trace_id: trace.color for _, trace in self.session.ordered_traces()})
        self.spectrum.update_colors({fid: loaded.color for fid, loaded in self.session.files.items()})
        self.dad_spectrum.update_colors({fid: loaded.color for fid, loaded in self.session.files.items()})
        self._sync_tree_colors()

    def _toggle_emphasis(self, trace_id: int) -> None:
        self.session.toggle_emphasis(trace_id)
        self.plots.set_emphasized(self.session.emphasized)

    def _start_worker(self, worker, on_success) -> None:
        self.runtime.start(worker, on_success)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.statusBar().showMessage(message)
        if total <= 0:
            self._progress.setRange(0, 0)
            return
        self._progress.setRange(0, 1000)
        self._progress.setValue(int(1000 * done / total))

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "KmzViewer", message)

    def _on_cancelled(self) -> None:
        self.statusBar().showMessage("Cancelled.")

    def _cancel_job(self) -> None:
        self.runtime.cancel()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        self.cancel_action.setVisible(busy)
        self._progress.setVisible(busy)
        if not busy:
            if self._filter_running:
                self._filter_running = False
                previous = range_label(self._mz_range) if self._mz_range else "unfiltered"
                self.mz_filter.status.setText(f"Filter was not applied. Current view: {previous}. Apply to retry or Clear to reset.")
                if not self._filter_complete():
                    self.mz_filter.status.setText(self.mz_filter.status.text() + " Newly added ion traces are hidden until then.")
            self._progress.setRange(0, 1000)
            self._progress.setValue(0)
            if self._filter_refresh_pending:
                QTimer.singleShot(0, self._ensure_mz_filter)
        if message:
            self.statusBar().showMessage(message)
        self._sync_actions()

    def _filter_complete(self):
        return self._mz_range is None or all(trace.optical or trace.trace_id in self._filter_values
                                             for _, trace in self.session.ordered_traces())

    def _sync_actions(self) -> None:
        has_files = bool(self.session.files)
        self.extract_action.setEnabled(has_files and not self._busy)
        self.bpc_action.setEnabled(has_files and not self._busy)
        self.manage_chromatograms_action.setEnabled(bool(self.session.ordered_traces()) and not self._busy)
        self.manage_chromatograms_button.setEnabled(self.manage_chromatograms_action.isEnabled())
        self.twc_action.setEnabled(has_files and not self._busy)
        self.ewc_action.setEnabled(has_files and not self._busy)
        self.dad_dock.toggleViewAction().setEnabled(bool(self._dad_enabled))
        self.export_action.setEnabled(has_files and not self._busy and self._filter_complete())
        self.export_current_action.setEnabled(self.active_trace_id is not None and not self._busy and self._filter_complete())
        has_ions = any(not trace.optical for _, trace in self.session.ordered_traces())
        self.mz_filter.apply_button.setEnabled(has_ions and not self._busy)
        self.mz_filter.start.setEnabled(not self._busy)
        self.mz_filter.stop.setEnabled(not self._busy)
        self.mz_filter.clear_button.setEnabled(self._mz_range is not None or self._filter_running
                                              or bool(self.mz_filter.start.text() or self.mz_filter.stop.text()))
        self.close_action.setEnabled(has_files and not self._busy)
        self.open_action.setEnabled(not self.runtime.closing)
        self.ms2_files_action.setEnabled(has_files and not self.runtime.closing)
        self.ms2_blank_action.setEnabled(not self.runtime.closing)

    def closeEvent(self, event) -> None:  # noqa: N802
        for window in list(self._gnps_windows):
            window.close()
        if self._gnps_windows:
            event.ignore()
            QTimer.singleShot(30, self.close)
            return
        self.runtime.close_event(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_balanced_initial_layout", False):
            self._balanced_initial_layout = True
            QTimer.singleShot(0, self._balance_plot_widths)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and (self.isMaximized() or self.isFullScreen()):
            QTimer.singleShot(0, self._balance_plot_widths)

    def _balance_plot_widths(self) -> None:
        if self.spectrum_dock.isVisible() and self.dockWidgetArea(self.spectrum_dock) == Qt.DockWidgetArea.RightDockWidgetArea:
            width = (self.plots.width() + self.spectrum_dock.width()) // 2
            self.resizeDocks([self.spectrum_dock], [width], Qt.Orientation.Horizontal)
