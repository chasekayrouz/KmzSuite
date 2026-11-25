"""Local, interactive feature-profile PCA with native Qt controls."""
import sys

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QToolTip, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d

from kmzcore.theme import BORDER, INK, MUTED, PAPER


ZOOM_STEP = 1.15  # per wheel notch, or per 60 px of trackpad or Magic Mouse travel


class PCACanvas(FigureCanvasQTAgg):
    point_clicked = Signal(QPointF, bool)
    focus_requested = Signal(QPointF)
    panned = Signal(QPointF, QPointF)
    pinched = Signal(float, QPointF)

    def __init__(self, figure):
        super().__init__(figure)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._press = self._pan = None

    def mousePressEvent(self, event):
        self.setFocus()
        left = event.button() == Qt.MouseButton.LeftButton
        command = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._press = (event.position(), command) if left else None
        # Shift-drag pans: a trackpad or Magic Mouse has no middle button for Matplotlib's own pan.
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self._pan = event.position() if left and shift and not command else None
        self.unsetCursor()
        if self._pan is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif not (self._press and command):
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            self.panned.emit(self._pan, event.position()); self._pan = event.position()
        elif not (self._press and self._press[1]):
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        press, self._press = self._press, None
        if self._pan is not None:
            self._pan = None; self.unsetCursor()
        elif not (press and press[1]):
            super().mouseReleaseEvent(event)
        if press and event.button() == Qt.MouseButton.LeftButton:
            if (event.position() - press[0]).manhattanLength() <= 4:
                self.point_clicked.emit(event.position(), press[1])

    def mouseDoubleClickEvent(self, event):
        # The first click has already selected the point; Matplotlib would start a rotation here.
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
        elif not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.focus_requested.emit(event.position())

    def event(self, event):
        # Matplotlib's Qt canvas ignores trackpad pinches. Qt delivers them with window-relative positions.
        if event.type() == QEvent.Type.NativeGesture and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.pinched.emit(event.value(), self.mapFromGlobal(event.globalPosition())); return True
        return super().event(event)


class PCATab(QWidget):
    feature_requested = Signal(int)

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self._selected = None
        self._chooser = None
        self._aim = None
        self._scatter = self._highlight = self._colorbar = None
        self._drawn = False
        layout = QVBoxLayout(self); layout.setContentsMargins(10, 8, 10, 8)
        controls = QHBoxLayout()
        self.axis_boxes = []
        for dimension in range(3):
            controls.addWidget(QLabel(('X', 'Y', 'Z')[dimension]))
            box = QComboBox(); box.setAccessibleName(('X', 'Y', 'Z')[dimension] + ' principal component')
            for index in range(max(3, data.rank)):
                ratio = data.variance_ratio[index] if index < data.rank else 0
                box.addItem(f'PC{index+1} · {ratio:.1%}' if index < data.rank else f'PC{index+1} · no variation', index)
            box.setCurrentIndex(dimension)
            box.setEnabled(data.rank > 3)
            controls.addWidget(box); self.axis_boxes.append(box)
            box.currentIndexChanged.connect(lambda _, axis=dimension: self._axis_changed(axis))
        controls.addStretch()
        controls.addWidget(QLabel('Color'))
        self.color_mode = QComboBox(); self.color_mode.addItems(['Summed height · log', 'Summed height · linear'])
        self.color_mode.setAccessibleName('PCA color scale')
        self.color_mode.currentIndexChanged.connect(self._recolor); controls.addWidget(self.color_mode)
        self.reset_button = QPushButton('Full view'); self.reset_button.clicked.connect(self.reset_view)
        controls.addWidget(self.reset_button); layout.addLayout(controls)

        self.summary = QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.summary.setToolTip('Each feature row is divided by its own summed height. Sample columns are then mean-centered. '
            'All samples have equal weight. No sample normalization, grouping, log transform of profiles, or missing-value filling. '
            'The Heights table and saved analysis are unchanged.')
        self.omissions = QLabel(); self.omissions.setWordWrap(True); layout.addWidget(self.omissions)
        self.omissions.setText(f'Omitted rows: {len(data.missing_rows):,} incomplete · {len(data.zero_rows):,} all-zero.')
        self.omissions.setVisible(bool(data.missing_rows or data.zero_rows))
        self.omissions.setToolTip('Blank = no acquired measurement; zero = measured zero. Incomplete rows are omitted, never filled. '
            'An all-zero row has no relative distribution to normalize. See Analysis Details → Measurements for coverage flags.')

        self.figure = Figure(figsize=(8, 5), facecolor=PAPER)
        self.canvas = PCACanvas(self.figure); self.setFocusProxy(self.canvas)
        self.axes = self.figure.add_axes([.04, .06, .77, .89], projection='3d', facecolor=PAPER)
        self.color_axes = self.figure.add_axes([.86, .20, .022, .59])
        self.axes.set_proj_type('ortho')
        self.axes.zaxis.label.set_rotation(90)
        self.axes.tick_params(colors=MUTED, labelsize=9)
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.set_pane_color((.96, .97, .98, 1))
            axis.line.set_color(BORDER)
            axis.set_major_locator(MaxNLocator(4))
            axis.set_rotate_label(False)
        layout.addWidget(self.canvas, 1)
        self.readout = QLabel('Click a point to inspect its feature ID, m/z, RT and original summed height.')
        self.readout.setWordWrap(True); self.readout.setMinimumHeight(38)
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.readout)
        command = 'Command' if sys.platform == 'darwin' else 'Ctrl'
        hint = QLabel('Drag to rotate · scroll or pinch over points to zoom in on them · double-click to center · '
                      f'Shift-drag to pan · {command}-click a point for all-sample EICs.')
        hint.setWordWrap(True); layout.addWidget(hint)
        self.canvas.point_clicked.connect(self._click)
        self.canvas.focus_requested.connect(self._focus)
        self.canvas.panned.connect(self._pan)
        self.canvas.pinched.connect(lambda value, position: self._zoom(1 / max(.2, 1 + value), position))
        self.canvas.mpl_connect('motion_notify_event', self._hover)
        self.canvas.mpl_connect('scroll_event', self._scroll)
        self.canvas.mpl_connect('draw_event', lambda _: setattr(self, '_drawn', True))
        self.shortcuts = []
        for keys, action in ((('F', 'Ctrl+F', 'Ctrl+0'), self.reset_view),
                             (('Ctrl+=', 'Ctrl++'), lambda: self._zoom(1 / 1.5)), (('Ctrl+-',), lambda: self._zoom(1.5))):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(action); self.shortcuts.append(shortcut)
        self._axes_selection = [0, 1, 2]
        self._plot()
        self.reset_view()

    def _axis_changed(self, changed):
        # Swap an already-used component instead of plotting the same axis twice.
        chosen = self.axis_boxes[changed].currentIndex()
        for axis, previous in enumerate(self._axes_selection):
            if axis != changed and previous == chosen:
                box = self.axis_boxes[axis]; box.blockSignals(True)
                box.setCurrentIndex(self._axes_selection[changed]); box.blockSignals(False)
        self._axes_selection = [box.currentIndex() for box in self.axis_boxes]
        self._plot(); self._fit(); self.canvas.draw_idle()

    def _coordinates(self):
        return np.column_stack([self.data.scores[:, index] if index < self.data.rank else
                                np.zeros(len(self.data.points)) for index in self._axes_selection])

    def _norm(self):
        values = [point.total for point in self.data.points]
        low, high = min(values), max(values)
        if low == high:
            # Keep a finite range for a one-value legend, including tiny units.
            low, high = low * .5, high * 1.5
        return (LogNorm if self.color_mode.currentIndex() == 0 else Normalize)(low, high)

    def _plot(self):
        self._drawn = False
        if self._scatter is not None:
            self._scatter.remove(); self._scatter = None
        if self._highlight is not None:
            self._highlight.remove(); self._highlight = None
        self.xyz = self._coordinates()
        if self.data.points:
            self._scatter = self.axes.scatter(*self.xyz.T, c=[p.total for p in self.data.points],
                cmap='viridis', norm=self._norm(), s=30, edgecolors=PAPER, linewidths=.35, depthshade=False,
                axlim_clip=True)
            self._recolor()
        else:
            self.color_axes.set_visible(False)
            self.axes.set_axis_off()
            self.color_mode.setEnabled(False); self.reset_button.setEnabled(False)
            self.axes.text2D(.5, .5, 'No complete, nonzero feature rows to plot.',
                transform=self.axes.transAxes, ha='center', color=INK)
        for axis, index in zip((self.axes.xaxis, self.axes.yaxis, self.axes.zaxis), self._axes_selection):
            ratio = self.data.variance_ratio[index] if index < self.data.rank else 0
            label = f'PC{index+1} ({ratio:.1%})' if index < self.data.rank else f'PC{index+1} (no variation)'
            axis.set_label_text(label, color=INK, fontsize=10)
            axis.labelpad = 22 if axis is self.axes.zaxis else 12
        captured = sum(self.data.variance_ratio[i] for i in self._axes_selection if i < self.data.rank)
        note = f'{len(self.data.points):,} / {self.data.feature_count:,} features · {self.data.sample_count:,} samples · row-sum normalized'
        if self.data.rank:
            note += f' · shown components explain {captured:.1%} of profile variation.'
        else:
            note += ' · no variation between normalized profiles.' if self.data.points else ' · no eligible profiles.'
        if 0 < self.data.rank < 3:
            note += f' Only {self.data.rank} independent component' + ('s.' if self.data.rank > 1 else '.')
        self.summary.setText(note)
        self._mark_selection()

    def _recolor(self, *_):
        if self._scatter is None:
            return
        self._scatter.set_norm(self._norm())
        if self._colorbar is None:
            self._colorbar = self.figure.colorbar(self._scatter, cax=self.color_axes)
        else:
            self._colorbar.update_normal(self._scatter)
        scale = 'log scale' if self.color_mode.currentIndex() == 0 else 'linear scale'
        self._colorbar.set_label(f'Original summed height ({scale})', color=INK, fontsize=9)
        self._colorbar.ax.tick_params(colors=MUTED, labelsize=9)
        self._colorbar.outline.set_edgecolor(BORDER)
        self.canvas.draw_idle()

    def _fit(self):
        if len(self.xyz):
            low, high = self.xyz.min(axis=0), self.xyz.max(axis=0)
            extent = float((high-low).max())
            self._home = (low + high) / 2, extent * .6 if extent > 0 else .5, (high - low) / 2
        else:
            self._home = np.zeros(3), .5, np.zeros(3)
        self._aim = None
        self._set_view(*self._home[:2])
        self.axes.set_box_aspect((1, 1, 1))

    def reset_view(self):
        self.axes.view_init(elev=24, azim=-55, roll=0)
        self._fit()

    def _bounded(self, half):
        return float(np.clip(half, self._home[1] * 1e-6, self._home[1]))

    def _set_view(self, center, half, settle=False):
        """Show the cube of this half-width around center; rotation pivots on its center."""
        home, full, span = self._home
        half = self._bounded(half)
        # The pivot stays among the features. Zooming out settles on the full view over its last stretch.
        reach = span * (min(1, 3 * (1 - half / full)) if settle else 1)
        center = np.clip(center, home - reach, home + reach)
        # Identical units on all axes: do not stretch tiny PCs to look important.
        for setter, middle in zip((self.axes.set_xlim, self.axes.set_ylim, self.axes.set_zlim), center):
            setter(middle-half, middle+half)
        self.canvas.draw_idle()

    def _project(self, xyz):
        x, y, depth = proj3d.proj_transform(*np.asarray(xyz, dtype=float).T, self.axes.get_proj())
        pixels = self.axes.transData.transform(np.column_stack((x, y))) / self.canvas.device_pixel_ratio
        pixels[:, 1] = self.canvas.height() - pixels[:, 1]
        return np.column_stack((pixels, depth))

    def projected_points(self):
        """Rendered positions in Qt logical pixels; shared by picking and UI tests."""
        return self._project(self.xyz) if len(self.xyz) else np.empty((0, 3))

    def _frame(self):
        """View cube center and half-width, its rendered center, and the map from data offsets to (x, y, depth)."""
        limits = np.array([self.axes.get_xlim(), self.axes.get_ylim(), self.axes.get_zlim()])
        center, half = limits.mean(axis=1), float(np.ptp(limits, axis=1).max()) / 2
        # The orthographic projection is affine, so three offsets give the whole map.
        origin, *ends = self._project(np.vstack([center, center + half * np.eye(3)]))
        return center, half, origin, (np.array(ends) - origin).T / half

    def _position(self, event):
        ratio = self.canvas.device_pixel_ratio
        return QPointF(event.x/ratio, self.canvas.height() - event.y/ratio)

    def _over_plot(self, position):
        ratio = self.canvas.device_pixel_ratio
        return self.axes.bbox.contains(position.x()*ratio, (self.canvas.height()-position.y())*ratio)

    def _target(self, position):
        """Where the pointer aims, at the median depth of the features under or else around it, and whether any are."""
        center, _, origin, basis = self._frame()
        # The median follows the group under the pointer, not a stray feature in front of it.
        near = self._hits(position) or self._hits(position, 40)
        depth = np.median(self.projected_points()[near, 2]) if near else origin[2]
        aim = center + np.linalg.solve(basis, [position.x()-origin[0], position.y()-origin[1], depth-origin[2]])
        return aim, bool(near)

    def _zoom(self, factor, position=None):
        """Scale the view cube. Zooming in homes in on the features under the pointer; otherwise the center holds."""
        if not len(self.xyz) or (position is not None and not self._over_plot(position)):
            return
        center, half, *_ = self._frame()
        scale = self._bounded(half * factor) / half
        if position is not None and factor < 1:
            # The target leaves the pointer as it nears the middle, so keep it until the pointer moves.
            if self._aim is None or (position - self._aim[0]).manhattanLength() > 3:
                aim, found = self._target(position)
                self._aim = position, aim if found else None
            if self._aim[1] is not None:
                # Its offset shrinks faster than the cube, clear of the clipping faces and onto the rotation pivot.
                center = self._aim[1] + (center - self._aim[1]) * scale**2
        self._set_view(center, half * scale, settle=factor > 1)

    def _scroll(self, event):
        wheel = event.guiEvent
        if wheel is None:
            notches = event.step
        elif wheel.phase() == Qt.ScrollPhase.ScrollMomentum:
            return  # zooming stops with the fingers
        else:
            # On macOS Matplotlib's step is in pixels, 20 for one wheel notch. Qt's angle counts notches,
            # with 60 px of trackpad or Magic Mouse travel as one.
            notches = wheel.angleDelta().y() / 120
        self._zoom(ZOOM_STEP ** -float(np.clip(notches, -8, 8)), self._position(event))

    def _pan(self, before, after):
        if not len(self.xyz):
            return
        center, half, _, basis = self._frame()
        self._set_view(center - np.linalg.solve(basis, [after.x()-before.x(), after.y()-before.y(), 0]), half)

    def _focus(self, position):
        """Center the feature under the pointer: rotation pivots on it and zooming in closes on it."""
        if not len(self.xyz) or not self._over_plot(position):
            return
        hits = self._hits(position)
        target = self.xyz[hits[0]] if hits else self._target(position)[0]
        self._aim = position, target
        self._set_view(target, self._frame()[1])

    def _hits(self, position, radius=8):
        if not self._drawn or not len(self.xyz):
            return []
        projected = self.projected_points()
        distance = np.linalg.norm(projected[:, :2] - [position.x(), position.y()], axis=1)
        visible = np.ones(len(self.xyz), dtype=bool)
        for axis, bounds in enumerate((self.axes.get_xlim(), self.axes.get_ylim(), self.axes.get_zlim())):
            visible &= (self.xyz[:, axis] >= min(bounds)) & (self.xyz[:, axis] <= max(bounds))
        indices = np.flatnonzero(visible & (distance <= radius))
        return sorted(indices.tolist(), key=lambda i: (distance[i], projected[i, 2], -self.data.points[i].total))

    def _description(self, index):
        point = self.data.points[index]
        polarity = {1: '+', -1: '−', 0: '?'}[point.polarity]
        return (f'{point.feature_id} · m/z {point.mz:.6f} {polarity} · RT {point.rt:.4f} min · '
                f'summed height {point.total:.8g} ({self.data.units})')

    def _hover(self, event):
        if event.inaxes is not self.axes or event.buttons:
            QToolTip.hideText(); return
        position = self._position(event)
        hits = self._hits(position)
        if hits:
            text = self._description(hits[0])
            if len(hits) > 1:
                text += f'\n{len(hits)} nearby / overlapping points; Command-click to choose.'
            QToolTip.showText(self.canvas.mapToGlobal(position.toPoint()), text, self.canvas)
        else:
            QToolTip.hideText()

    def _click(self, position, command):
        hits = self._hits(position)
        if not hits:
            return
        self._selected = hits[0]; self.readout.setText(self._description(hits[0]))
        self._mark_selection(); self.canvas.draw_idle()
        if command:
            if len(hits) == 1:
                self.feature_requested.emit(self.data.points[hits[0]].row)
            else:
                self._choose_point(hits)

    def _mark_selection(self):
        if self._highlight is not None:
            self._highlight.remove(); self._highlight = None
        if self._selected is not None:
            self._highlight = self.axes.scatter(*self.xyz[self._selected], s=110, facecolors='none',
                edgecolors=INK, linewidths=1.3, depthshade=False, axlim_clip=True)

    def _choose_point(self, hits):
        if self._chooser is not None:
            self._chooser.close()
        dialog = QDialog(self); dialog.setWindowTitle('Choose feature for EICs'); dialog.resize(740, 380)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f'{len(hits):,} points overlap here. Choose the feature to inspect.'))
        search = QLineEdit(); search.setPlaceholderText('Filter by feature ID, m/z or RT'); layout.addWidget(search)
        listing = QListWidget(); listing.setAccessibleName('Overlapping PCA features'); layout.addWidget(listing)
        for index in hits:
            item = QListWidgetItem(self._description(index)); item.setData(Qt.ItemDataRole.UserRole, index)
            listing.addItem(item)
        listing.setCurrentRow(0)
        def filter_items(text):
            first = None
            for row in range(listing.count()):
                item = listing.item(row); item.setHidden(text.casefold() not in item.text().casefold())
                if first is None and not item.isHidden():first = item
            if first is not None:listing.setCurrentItem(first)
            buttons.button(QDialogButtonBox.StandardButton.Open).setEnabled(first is not None)
        search.textChanged.connect(filter_items)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Open).setText('Plot EICs')
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        listing.itemDoubleClicked.connect(lambda _: dialog.accept())
        def chosen():
            item = listing.currentItem()
            if item is not None and not item.isHidden():
                self._selected = item.data(Qt.ItemDataRole.UserRole)
                self.readout.setText(self._description(self._selected)); self._mark_selection(); self.canvas.draw_idle()
                self.feature_requested.emit(self.data.points[self._selected].row)
        dialog.accepted.connect(chosen)
        dialog.finished.connect(lambda _: setattr(self, '_chooser', None))
        self._chooser = dialog; dialog.open()
