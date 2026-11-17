"""Heights-first desktop workflow, backed by automatically saved analyses."""
from pathlib import Path
from types import SimpleNamespace
import math
import os
import sys

from PySide6.QtCore import QEvent, QModelIndex, QSettings, QThread, QTimer, Qt, QUrl, Slot
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
    QDockWidget, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QProgressBar, QTabBar, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from kmzcore.plot_paint import finish_pending
from kmzcore.runtime import resource_path
from kmzcore.theme import apply_theme
from kmzcore.version import about_text
from kmzcomet.csv_export import write_csv
from kmzcomet.excel import write_workbook
from kmzcomet.history import history_directory
from kmzcomet.sources import InputSource
from kmzcomet.chromatograms import ChromatogramCache
from kmzcomet.pca import feature_pca

from .dialogs import DetailsDialog, SettingsDialog, benchmark_settings
from .table import EICFeatureTable, EICMatrixModel, feature_filter
from .workers import Worker, analysis_worker, load_worker, save_worker


class MainWindow(QMainWindow):
    def __init__(self, *, history=None):
        super().__init__()
        from PySide6.QtWidgets import QApplication
        apply_theme(QApplication.instance())
        self.setWindowTitle('KmzCoMet'); self.resize(1320,820)
        self.history = Path(history) if history is not None else history_directory()
        self.analysis_settings = benchmark_settings()
        self.inputs = []
        self.result = None; self.saved_path = None
        self._eic_cache = None; self._plot_serial = 0
        self._pca_data = None; self._pca_serial = 0
        self._on_heights = True; self._heights_inputs_visible = True
        self._thread = None; self._worker = None; self._closing = False
        self._operation = ''; self._pending_paths = []
        self.runtime = SimpleNamespace(closed=False)
        override = os.environ.get('KMZ_LOCAL_STATE_ROOT')
        if override:
            base = Path(override); base.mkdir(parents=True,exist_ok=True)
            self.preferences = QSettings(str(base/'comet.ini'),QSettings.Format.IniFormat)
        else:
            self.preferences = QSettings('KmzSuite','org.kmzsuite.comet')
        self._build_ui(); self.setAcceptDrops(True)
        self._sync()

    def _action(self, label, callback, shortcut=None):
        action = QAction(label,self); action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _build_ui(self):
        self.add_action = self._action('Add WIFF Files…',self.choose_inputs,QKeySequence.StandardKey.Open)
        self.run_action = self._action('Run Analysis',self.start_analysis,'Ctrl+R')
        self.cancel_action = self._action('Cancel',self.cancel)
        self.past_action = self._action('Past Analyses…',self.choose_history,'Ctrl+Shift+O')
        self.csv_action = self._action('Export Heights as CSV…',lambda:self.export('csv'))
        self.excel_action = self._action('Export Excel…',lambda:self.export('xlsx'))
        self.details_action = self._action('Analysis Details…',self.show_details)
        self.settings_action = self._action('Settings…',self.edit_settings)
        self.retry_action = self._action('Retry Saving Analysis',self.retry_save)
        self.plot_action = self._action('Plot EICs',self.plot_eics,'Ctrl+E')
        self.pca_action = self._action('Plot PCA',self.plot_pca)
        self.pca_action.setToolTip('Plot normalized profiles of all features across all samples, independently of table filters or EIC selections.')
        toolbar = QToolBar('Analysis'); toolbar.setMovable(False); self.addToolBar(toolbar)
        toolbar.addActions([self.add_action,self.run_action,self.cancel_action,self.settings_action])
        toolbar.addSeparator(); toolbar.addAction(self.past_action)
        toolbar.addSeparator(); toolbar.addActions([self.plot_action,self.pca_action])
        toolbar.addSeparator(); toolbar.addActions([self.csv_action,self.excel_action,self.details_action,self.retry_action])
        menu = self.menuBar().addMenu('File')
        menu.addActions([self.add_action,self.past_action]);menu.addSeparator()
        menu.addActions([self.csv_action,self.excel_action]);menu.addSeparator()
        close = self._action('Close Tab / Window',self.close_active_tab,'Ctrl+W');menu.addAction(close)
        analysis = self.menuBar().addMenu('Analysis');analysis.addActions([self.run_action,self.cancel_action,self.settings_action,self.details_action,self.retry_action])
        analysis.addActions([self.plot_action,self.pca_action])
        help_menu = self.menuBar().addMenu('Help')
        manual = self._action('KmzCoMet Manual',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(resource_path('docs/KmzCoMet.html')))))
        help_menu.addAction(manual)
        notices = self._action('Dependency Notices',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(resource_path('resources/THIRD_PARTY_NOTICES.txt')))))
        help_menu.addAction(notices)
        about = self._action('About KmzCoMet',lambda:QMessageBox.about(self,'KmzCoMet',about_text('KmzCoMet','Comparative MS1 ion features.')))
        about.setMenuRole(QAction.MenuRole.AboutRole);help_menu.addAction(about)

        self.queue = QListWidget(); self.queue.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.queue.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.queue.model().rowsMoved.connect(self._queue_reordered)
        self.remove_button = QPushButton('Remove');self.remove_button.clicked.connect(self.remove_inputs)
        self.clear_button = QPushButton('Clear');self.clear_button.clicked.connect(self.clear_inputs)
        holder = QWidget(); layout = QVBoxLayout(holder)
        layout.addWidget(self.queue)
        buttons = QHBoxLayout();buttons.addWidget(self.remove_button);buttons.addWidget(self.clear_button);layout.addLayout(buttons)
        self.input_note = QLabel('Drop WIFF files here or use Add WIFF Files. Drag entries to reorder; the first is the RT reference.')
        self.input_note.setWordWrap(True);layout.addWidget(self.input_note)
        self.dock = QDockWidget('Input Acquisitions',self);self.dock.setObjectName('inputs');self.dock.setWidget(holder)
        self.dock.setMinimumWidth(240);self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,self.dock)
        analysis.addAction(self.dock.toggleViewAction())

        center = QWidget();layout = QVBoxLayout(center);layout.setContentsMargins(10,10,10,8)
        self.summary = QLabel('Add WIFF acquisitions to start, or reopen a saved analysis.');self.summary.setWordWrap(True);layout.addWidget(self.summary)
        filters = QHBoxLayout()
        self.search = QLineEdit();self.search.setPlaceholderText('Find feature ID, m/z or RT');self.search.textChanged.connect(self.apply_filters)
        filters.addWidget(self.search,2)
        self.bounds = []
        for label in ('m/z min','m/z max','RT min','RT max'):
            box = QLineEdit();box.setPlaceholderText(label);box.setMaximumWidth(115);box.textChanged.connect(self.apply_filters)
            filters.addWidget(box);self.bounds.append(box)
        reset = QPushButton('Clear Filters');reset.clicked.connect(self.clear_filters);filters.addWidget(reset);layout.addLayout(filters)
        self.filter_note = QLabel('');layout.addWidget(self.filter_note)
        selections = QHBoxLayout()
        self.selection_note = QLabel('0 EICs selected');selections.addWidget(self.selection_note);selections.addStretch()
        self.clear_selection = QPushButton('Clear EIC Selection');selections.addWidget(self.clear_selection);layout.addLayout(selections)
        self.model = EICMatrixModel();self.proxy = feature_filter(self);self.proxy.setSourceModel(self.model)
        self.model.selection_changed.connect(self._sync_selection)
        self.clear_selection.clicked.connect(self.model.clear_picks)
        self.table = EICFeatureTable();self.table.setModel(self.proxy);self.table.horizontalHeader().setDefaultSectionSize(165)
        layout.addWidget(self.table,1)
        self.heights_page = center
        self.tabs = QTabWidget();self.tabs.setDocumentMode(True);self.tabs.setTabsClosable(True);self.tabs.setMovable(True)
        self.tabs.addTab(center,'Heights')
        for side in (QTabBar.ButtonPosition.LeftSide,QTabBar.ButtonPosition.RightSide):self.tabs.tabBar().setTabButton(0,side,None)
        self.tabs.tabCloseRequested.connect(self.close_plot_tab)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)
        self.count = QLabel('0 features');self.statusBar().addPermanentWidget(self.count)
        self.progress = QProgressBar();self.progress.setMaximumWidth(140);self.progress.setRange(0,0);self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

    def event(self, event):
        if event.type() == QEvent.Type.UpdateRequest:
            tab = self.tabs.currentWidget()
            # EICTab loads with the first EIC tab; until then no tab can be one.
            chromatograms = sys.modules.get(__package__+'.chromatograms')
            eic = chromatograms is not None and isinstance(tab, chromatograms.EICTab)
            finish_pending(tab.plots.view if eic else None)
        return super().event(event)

    @property
    def busy(self):
        return self._thread is not None

    def _sync(self):
        available = not self.busy and not self._closing
        for action in (self.add_action,self.past_action,self.settings_action):action.setEnabled(available)
        self.run_action.setEnabled(available and bool(self.inputs))
        self.pca_action.setEnabled(available and self.result is not None and bool(self.result.features))
        self.cancel_action.setEnabled(self.busy and not self._closing)
        for action in (self.csv_action,self.excel_action,self.details_action):action.setEnabled(available and self.result is not None)
        self.retry_action.setVisible(self.result is not None and self.saved_path is None)
        self.retry_action.setEnabled(available and self.result is not None and self.saved_path is None)
        for widget in (self.queue,self.remove_button,self.clear_button):widget.setEnabled(available)
        self.progress.setVisible(self.busy)
        self._sync_selection()

    def _sync_selection(self):
        count = sum(len(samples) for samples in self.model.picks.values())
        hidden = sum(len(samples) for row,samples in self.model.picks.items()
                     if not self.proxy.mapFromSource(self.model.index(row,0)).isValid())
        self.selection_note.setText(f'{count:,} EICs selected' + (f' · {hidden:,} hidden by filters' if hidden else ''))
        self.clear_selection.setEnabled(bool(count))
        self.plot_action.setEnabled(bool(count) and not self.busy and not self._closing)

    def _tab_changed(self, index):
        heights = self.tabs.widget(index) is self.heights_page
        if not heights and self._on_heights:
            self._heights_inputs_visible = not self.dock.isHidden();self.dock.hide()
        elif heights and not self._on_heights:
            self.dock.setVisible(self._heights_inputs_visible)
        if heights != self._on_heights:
            # Apply the input dock change now: plots then paint once, at their
            # final size, instead of again after a deferred relayout.
            self.layout().activate()
        self._on_heights = heights
        if not heights and index >= 0:
            tab = self.tabs.widget(index)
            # EICTab loads pyqtgraph with the first EIC tab; until then no tab can be one.
            chromatograms = sys.modules.get(__package__+'.chromatograms')
            eic = chromatograms is not None and isinstance(tab,chromatograms.EICTab)
            (tab.plots.view if eic else tab).setFocus()

    def close_plot_tab(self, index):
        widget = self.tabs.widget(index)
        if widget is not None and widget is not self.heights_page:
            self.tabs.removeTab(index);widget.deleteLater()

    def close_active_tab(self):
        if self.tabs.currentWidget() is self.heights_page:self.close()
        else:self.close_plot_tab(self.tabs.currentIndex())

    def plot_eics(self):
        self._plot_cells(self.model.selected_cells)

    def plot_feature_eics(self, row):
        if self.result is None or not 0 <= row < len(self.result.features):return
        self._plot_cells(tuple((row,sample) for sample in range(len(self.result.samples))))

    def _plot_cells(self, cells):
        if self.busy or self.result is None or self._closing:return
        if not cells:return
        cache = self._eic_cache
        def operation(worker):
            return cache.extract(cells,progress=worker.progress.emit,checkpoint=worker.checkpoint)
        self._start(Worker(operation),self._eics_ready,'eics')

    def plot_pca(self):
        if self.busy or self.result is None or self._closing:return
        if self._pca_data is not None:
            self._pca_ready(self._pca_data);return
        result = self.result
        def operation(worker):
            return feature_pca(result,progress=worker.progress.emit,checkpoint=worker.checkpoint)
        self._start(Worker(operation),self._pca_ready,'pca')

    def _pca_ready(self, data):
        if self._closing:return
        from .pca import PCATab
        self._pca_data = data;self._pca_serial += 1
        tab = PCATab(data);tab.feature_requested.connect(self.plot_feature_eics)
        index = self.tabs.addTab(tab,f'PCA {self._pca_serial} · {len(data.points)}')
        self.tabs.setTabToolTip(index,f'All {data.feature_count} feature rows · analysis {self.result.created_utc[:19]} UTC')
        self.tabs.setCurrentIndex(index)
        self.statusBar().showMessage(f'Feature PCA · {len(data.points):,} plotted · {len(data.missing_rows):,} incomplete · {len(data.zero_rows):,} all-zero rows omitted')

    def _eics_ready(self, batch):
        if self._closing:return
        try:
            from .chromatograms import EICTab
        except (ImportError, OSError) as error:
            # The renderer loads with the first EIC tab, after the worker reported success.
            self._failed(str(error));return
        self._plot_serial += 1
        tab = EICTab(batch,sample_order=[s.id for s in self.result.samples])
        index = self.tabs.addTab(tab,f'EICs {self._plot_serial} · {len(batch.entries)}')
        self.tabs.setTabToolTip(index,f'{len(batch.entries)} selected EICs · analysis {self.result.created_utc[:19]} UTC')
        self.tabs.setCurrentIndex(index)
        unavailable = sum(entry.data is None for entry in batch.entries)
        self.statusBar().showMessage(f'{len(batch.entries)} EICs · {batch.reused} cached · {batch.files_read} source reads' +
                                     (f' · {unavailable} unavailable (see legend)' if unavailable else ''))

    @Slot(object)
    def _exported(self, path):
        # Bound slot: delivered on the GUI thread (a lambda would run on the worker).
        self.statusBar().showMessage('Exported '+str(path))

    @Slot()
    def _cancelled(self):
        self.statusBar().showMessage('Cancelled.')

    def _start(self, worker, success, operation):
        if self.busy:
            raise RuntimeError('Wait for the current operation to finish')
        self._worker = worker;self._operation = operation;self._thread = QThread(self)
        worker.moveToThread(self._thread)
        # A functor on the QThread runs the worker, which therefore receives no connection from
        # a Python-wrapper sender: destroying it on its thread never asks PySide for the GIL
        # under Qt's connection lock, and it never takes that lock itself while holding the GIL.
        self._thread.started.connect(lambda:worker.run(),Qt.ConnectionType.DirectConnection)
        worker.progress.connect(self.statusBar().showMessage)
        worker.succeeded.connect(success)
        worker.failed.connect(self._failed)
        worker.cancelled.connect(self._cancelled)  # A bound slot runs on the GUI thread; a lambda would not.
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._thread.quit,Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._reap)
        self._sync();self._thread.start()

    def _reap(self):
        if self._thread is None:return
        if not self._thread.wait(0):
            QTimer.singleShot(10,self._reap);return
        self._thread.deleteLater();self._thread=None;self._worker=None;self._sync()
        if self._closing:self.close()
        elif self._pending_paths:
            paths,self._pending_paths=self._pending_paths,[];self.open_paths(paths)

    def _failed(self, message):
        self.statusBar().showMessage('Operation failed: '+message)
        if not self._closing:QMessageBox.warning(self,'KmzCoMet',message)

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel();self.statusBar().showMessage('Cancelling… waiting for the current operation to finish.')

    def choose_inputs(self):
        paths,_ = QFileDialog.getOpenFileNames(self,'Add WIFF Acquisitions',self.preferences.value('lastDirectory',str(Path.home())),
                                               'SCIEX acquisitions (*.wiff *.wiff2)')
        self.open_paths(paths)

    def open_paths(self, paths):
        if not paths:return
        if self.busy:
            self._pending_paths.extend(paths);return
        try:
            converted=[]
            for value in paths:
                if isinstance(value,QUrl):
                    if not value.isLocalFile():raise ValueError('Only local files can be opened')
                    value=value.toLocalFile()
                converted.append(Path(value).expanduser().resolve(strict=True))
            if any(p.suffix.lower()=='.kmzcomet' for p in converted):
                if len(converted)!=1:raise ValueError('Open one saved analysis at a time')
                if not self._can_replace_result():return
                self._start(load_worker(converted[0]),self._completed,'open');return
            if any(p.suffix.lower() not in {'.wiff','.wiff2'} for p in converted):
                raise ValueError('Choose WIFF acquisitions or a saved .kmzcomet analysis')
            def inspect(worker):
                from kmzconvert_core.samples import list_samples
                entries=[]
                for path in converted:
                    worker.checkpoint()
                    worker.progress.emit('Reading input metadata: '+path.name)
                    entries.append((path,[s.id for s in list_samples(path)]))
                worker.checkpoint();return entries
            self._start(Worker(inspect),self._add_inspected,'inputs')
            self.preferences.setValue('lastDirectory',str(converted[0].parent))
        except (OSError,ValueError) as error:self._failed(str(error))

    def _add_inspected(self, entries):
        if self._closing:return
        try:
            known=set()
            for source in self.inputs:
                try:known.add(source.identity())
                except OSError:pass  # Reopened results need no available raw files.
            for path,ids in entries:
                if len(ids)>1:
                    dialog=QDialog(self);dialog.setWindowTitle('Choose stored samples — '+path.name);layout=QVBoxLayout(dialog)
                    boxes=[]
                    for ident in ids:
                        box=QCheckBox(f'Stored sample {ident}');box.setChecked(True);boxes.append(box);layout.addWidget(box)
                    buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
                    buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
                    if dialog.exec()!=QDialog.DialogCode.Accepted:continue
                    ids=[ident for ident,box in zip(ids,boxes) if box.isChecked()]
                for ident in ids:
                    source=InputSource(path,ident).resolved();key=source.identity()
                    if key not in known:self.inputs.append(source);known.add(key)
            self._refresh_queue();self.statusBar().showMessage(f'{len(self.inputs)} acquisition runs selected.')
        except (OSError,ValueError) as error:self._failed(str(error))

    def _refresh_queue(self):
        self.queue.clear()
        for source in self.inputs:
            item=QListWidgetItem(f'{source.path.name}\nStored sample {source.sample_id}')
            item.setToolTip(str(source.path));item.setData(Qt.ItemDataRole.UserRole,source);self.queue.addItem(item)
        self._sync()

    def _queue_reordered(self,*_):
        self.inputs=[self.queue.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.queue.count())]

    def remove_inputs(self):
        rows={self.queue.row(i) for i in self.queue.selectedItems()}
        self.inputs=[s for i,s in enumerate(self.inputs) if i not in rows];self._refresh_queue()

    def clear_inputs(self):
        self.inputs=[];self._refresh_queue()

    def start_analysis(self):
        if not self.inputs or self.busy:return
        if not self._can_replace_result():return
        self.statusBar().showMessage('Starting analysis…')
        self._start(analysis_worker(tuple(self.inputs),self.analysis_settings,self.history),self._completed,'analysis')

    def _can_replace_result(self):
        if self.result is None or self.saved_path is not None:return True
        answer=QMessageBox.question(self,'Unsaved analysis','This result has not been saved. Replace it without saving?',
                                   QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
        return answer==QMessageBox.StandardButton.Yes

    def _completed(self, completed):
        if self._closing:return
        changed = completed.result is not self.result
        self.result=completed.result;self.saved_path=completed.path
        self.analysis_settings=self.result.settings
        if changed:
            for index in range(self.tabs.count()-1,-1,-1):self.close_plot_tab(index)
            self.tabs.setCurrentWidget(self.heights_page)
            self._eic_cache=ChromatogramCache(self.result)
            self._pca_data=None
            self.model.set_result(self.result);self.clear_filters()
            for i,width in enumerate((48,115,115,100,95)):self.table.setColumnWidth(i,width)
            self.table.horizontalHeader().setMinimumHeight(48)
            self.table._refresh_frozen()
            self.table.sortByColumn(2,Qt.SortOrder.AscendingOrder)
            self.table.clearSelection();self.table.setCurrentIndex(QModelIndex())
        self.inputs=[InputSource(s.path,s.native_sample_id) for s in self.result.samples]
        self._refresh_queue()
        saved=f'Saved · {self.result.created_utc[:19].replace("T"," ")} UTC' if self.saved_path else 'Not saved — use Retry Saving Analysis'
        self.summary.setText(f'{len(self.result.features):,} features · {len(self.result.samples)} samples · {saved}')
        self.setWindowTitle('KmzCoMet — '+self.result.created_utc[:19].replace('T',' '))
        self.statusBar().showMessage('Analysis ready.' if self.saved_path else 'Results are available but automatic saving failed.')
        if completed.save_error:QMessageBox.warning(self,'Analysis not saved',completed.save_error+'\nResults remain available. Retry saving before closing.')
        self._sync()

    def retry_save(self):
        if self.result is not None and not self.busy:
            self._start(save_worker(self.result,self.history),self._completed,'save')

    def choose_history(self):
        try:
            self.history.mkdir(parents=True,exist_ok=True)
            path,_=QFileDialog.getOpenFileName(self,'Past Analyses',str(self.history),'KmzCoMet analyses (*.kmzcomet)')
            if path:self.open_paths([path])
        except OSError as error:self._failed(str(error))

    def edit_settings(self):
        dialog=SettingsDialog(self.analysis_settings,self)
        if dialog.exec()==QDialog.DialogCode.Accepted:self.analysis_settings=dialog.settings

    def show_details(self):
        if self.result is not None:DetailsDialog(self.result,self).exec()

    def apply_filters(self):
        try:
            bounds=tuple(float(box.text()) if box.text().strip() else None for box in self.bounds)
            if any(v is not None and not math.isfinite(v) for v in bounds):raise ValueError()
            for lo,hi in (bounds[:2],bounds[2:]):
                if lo is not None and hi is not None and lo>hi:raise ValueError()
            self.proxy.set_filters(self.search.text(),bounds);self.filter_note.setText('')
        except ValueError:self.filter_note.setText('Enter finite numeric bounds with minimum ≤ maximum.');return
        self.count.setText(f'{self.proxy.rowCount():,} / {self.model.rowCount():,} features')
        self._sync_selection()

    def clear_filters(self):
        self.search.clear()
        for box in self.bounds:box.clear()
        self.apply_filters()

    def export(self, kind):
        if self.result is None or self.busy:return
        title='Export Complete Heights Matrix' if kind=='csv' else 'Export Complete Analysis'
        name='KmzCoMet-'+self.result.created_utc[:10]+'.'+kind
        path,_=QFileDialog.getSaveFileName(self,title,str(Path(self.preferences.value('lastDirectory',str(Path.home())))/name),
                                        'CSV (*.csv)' if kind=='csv' else 'Excel (*.xlsx)',options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:return
        destination=Path(path)
        if not destination.suffix:destination=destination.with_suffix('.'+kind)
        result=self.result
        def operation(worker):
            worker.progress.emit('Exporting complete analysis…')
            return (write_csv if kind=='csv' else write_workbook)(result,destination,checkpoint=worker.checkpoint)
        self._start(Worker(operation),self._exported,'export')
        self.preferences.setValue('lastDirectory',str(destination.parent))

    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls() and all(u.isLocalFile() for u in event.mimeData().urls()):event.acceptProposedAction()

    def dropEvent(self,event):
        self.open_paths(event.mimeData().urls());event.acceptProposedAction()

    def closeEvent(self,event):
        if self.busy:
            self._closing=True;self.cancel();self.setEnabled(False);event.ignore();return
        if self.result is not None and self.saved_path is None:
            answer=QMessageBox.question(self,'Unsaved analysis','This result has not been saved. Close without saving?',
                                       QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
            if answer!=QMessageBox.StandardButton.Yes:
                self._closing=False;self.setEnabled(True);event.ignore();return
        self.runtime.closed=True;event.accept()
