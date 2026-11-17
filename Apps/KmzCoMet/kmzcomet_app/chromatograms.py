"""A compact, independent chromatogram comparison tab; no spectrum inspection."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QColorDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMenu, QScrollArea, QSplitter, QToolBar, QVBoxLayout, QWidget)

from kmzcore.trace_style import chromatogram_palette
from kmzviewer.gui.plots import ChromatogramPanel


class EICTab(QWidget):
    def __init__(self, batch, *, sample_order, parent=None):
        super().__init__(parent)
        self.batch = batch
        self.entries = dict(enumerate(batch.entries, start=1))
        palette = chromatogram_palette()
        self.colors = {tid: palette[sample_order.index(entry.sample_id) % len(palette)] for tid, entry in self.entries.items()}
        self.emphasized = set()
        self.plots = ChromatogramPanel(inspection_enabled=False, legend=False)
        self.plots.emphasis_requested.connect(self.toggle_emphasis)
        self.plots.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        toolbar = QToolBar('EIC controls'); toolbar.setMovable(False)
        layout.addWidget(toolbar)

        self.stacked_action = QAction('Stacked', self); self.stacked_action.setCheckable(True); self.stacked_action.setChecked(True)
        self.overlay_action = QAction('Overlay', self); self.overlay_action.setCheckable(True)
        self.layout_group = QActionGroup(self)
        for action in (self.stacked_action, self.overlay_action):
            self.layout_group.addAction(action); toolbar.addAction(action)
        self.stacked_action.triggered.connect(lambda: self.set_mode('stacked'))
        self.overlay_action.triggered.connect(lambda: self.set_mode('overlay'))
        toolbar.addSeparator()
        self.pan_action = QAction('Pan', self); self.pan_action.setCheckable(True); self.pan_action.setChecked(True)
        self.zoom_action = QAction('Zoom box', self); self.zoom_action.setCheckable(True)
        self.mouse_group = QActionGroup(self)
        for action in (self.pan_action, self.zoom_action):
            self.mouse_group.addAction(action); toolbar.addAction(action)
        self.pan_action.triggered.connect(lambda: self.plots.set_mouse_mode('pan'))
        self.zoom_action.triggered.connect(lambda: self.plots.set_mouse_mode('zoom'))
        self.full_action = toolbar.addAction('Full view', self.plots.reset_view)
        self.full_action.setToolTip('Restore the full acquisition and Y ranges (F, Command+F or Command+0).')
        toolbar.addSeparator()
        self.autoscale_action = QAction('Autoscale Y', self); self.autoscale_action.setCheckable(True); self.autoscale_action.setChecked(True)
        self.autoscale_action.toggled.connect(self.plots.set_autoscale_y); toolbar.addAction(self.autoscale_action)
        self.match_action = QAction('Match Y', self); self.match_action.setCheckable(True)
        self.match_action.toggled.connect(self.plots.set_match_y); toolbar.addAction(self.match_action)
        self.legend_action = QAction('Legend', self); self.legend_action.setCheckable(True); self.legend_action.setChecked(True)
        toolbar.addAction(self.legend_action)
        self.shortcuts = []
        for key in ('F', 'Ctrl+F', 'Ctrl+0'):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self.plots.reset_view); self.shortcuts.append(shortcut)

        failed = sum(entry.data is None for entry in batch.entries)
        note = 'Summed EICs · original retention times · full acquisition. Peak heights can differ from the Heights table.'
        if failed:
            note += f'  {failed} unavailable — hover the legend entries for details.'
        self.note = QLabel(note); self.note.setWordWrap(True); layout.addWidget(self.note)
        self.splitter = QSplitter()
        layout.addWidget(self.splitter, 1)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(self.plots); self.splitter.addWidget(self.scroll)
        self.legend_holder = QWidget(); legend_layout = QVBoxLayout(self.legend_holder); legend_layout.setContentsMargins(4, 0, 0, 0)
        legend_layout.addWidget(QLabel('Legend · check to show'))
        self.legend = QListWidget(); self.legend.setAccessibleName('EIC legend')
        self.legend.setIconSize(QSize(24, 12)); self.legend.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.legend.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.legend.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.legend.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.legend.customContextMenuRequested.connect(self.legend_menu)
        self.legend.setToolTip('Check to show/hide. Drag to reorder. Double-click to change color. Right-click for emphasis.')
        legend_layout.addWidget(self.legend)
        self.splitter.addWidget(self.legend_holder); self.splitter.setSizes([950, 300]); self.splitter.setStretchFactor(0, 1)
        self.legend_action.toggled.connect(self.legend_holder.setVisible)
        self.readout = QLabel('Hover an EIC for native retention time and summed intensity.')
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.readout.setWordWrap(True); layout.addWidget(self.readout)
        self.plots.readout.connect(self.readout.setText)
        for tid, entry in self.entries.items():
            stored = f' · stored {entry.native_sample_id}' if entry.native_sample_id is not None else ''
            sign = {1: '+', -1: '−', 0: '?'}[entry.polarity]
            label = f'{entry.filename} · {entry.sample_id}{stored}\n{entry.feature_id} · m/z {entry.mz:.6f} {sign}'
            if entry.error:
                label += ' · unavailable'
            elif not entry.data.rt.size:
                label += ' · no scans of this polarity'
            elif entry.data.missing_coverage == entry.data.rt.size:
                label += ' · no acquired mass coverage'
            item = QListWidgetItem(self._icon(self.colors[tid]), label)
            item.setData(Qt.ItemDataRole.UserRole, tid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if entry.data is not None else Qt.CheckState.Unchecked)
            radius = entry.mz * entry.ppm * 1e-6
            details = f'{entry.title}\nSum within {entry.mz-radius:.8f}–{entry.mz+radius:.8f} m/z (±{entry.ppm:g} ppm).\nNative intensity unit: {entry.unit}.'
            if entry.data is not None:
                details += f'\n{entry.data.rt.size:,} scans; {entry.data.missing_coverage:,} outside acquired mass coverage; {entry.data.unknown_coverage:,} with unspecified mass coverage.'
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                details += '\nUnavailable: ' + entry.error
            item.setToolTip(details); self.legend.addItem(item)
        self.legend.itemChanged.connect(self.refresh)
        self.legend.model().rowsMoved.connect(self.refresh)
        self.legend.itemDoubleClicked.connect(self.change_color)
        self.refresh(); self.plots.reset_view()

    @staticmethod
    def _icon(color):
        pixmap = QPixmap(24, 12); pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap); painter.setPen(QPen(QColor(color), 3)); painter.drawLine(1, 6, 23, 6); painter.end()
        return QIcon(pixmap)

    def refresh(self, *_):
        rows = []
        for index in range(self.legend.count()):
            item = self.legend.item(index); tid = item.data(Qt.ItemDataRole.UserRole)
            entry = self.entries[tid]
            if item.checkState() == Qt.CheckState.Checked and entry.data is not None:
                rows.append((tid, entry.title, self.colors[tid], entry.data.rt, entry.data.y))
        self.plots.set_traces(rows, units={row[0]: self.entries[row[0]].unit for row in rows})
        self.plots.set_emphasized(self.emphasized)
        self._size_stack(len(rows))
        if not rows:
            self.readout.setText('No EICs visible. Check a legend entry to show it; unavailable entries explain the source error in their tooltip.')

    def _size_stack(self, count):
        self.plots.setMinimumHeight(max(220, count * 105) if self.stacked_action.isChecked() else 220)

    def set_mode(self, mode):
        self.stacked_action.setChecked(mode == 'stacked'); self.overlay_action.setChecked(mode == 'overlay')
        self.plots.set_mode(mode); self._size_stack(len(self.plots._rows))

    def toggle_emphasis(self, tid):
        if tid in self.emphasized:
            self.emphasized.remove(tid)
        else:
            self.emphasized.add(tid)
        self.plots.set_emphasized(self.emphasized)

    def change_color(self, item, *_):
        tid = item.data(Qt.ItemDataRole.UserRole)
        color = QColorDialog.getColor(QColor(self.colors[tid]), self, 'EIC color')
        if color.isValid():
            self.colors[tid] = color.name()
            self.legend.blockSignals(True); item.setIcon(self._icon(color.name())); self.legend.blockSignals(False)
            self.plots.update_colors({tid: color.name()})

    def legend_menu(self, position):
        item = self.legend.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        menu.addAction('Change color…', lambda: self.change_color(item))
        emphasize = menu.addAction('Toggle emphasis', lambda: self.toggle_emphasis(item.data(Qt.ItemDataRole.UserRole)))
        emphasize.setEnabled(self.overlay_action.isChecked() and item.checkState() == Qt.CheckState.Checked)
        menu.exec(self.legend.viewport().mapToGlobal(position))
