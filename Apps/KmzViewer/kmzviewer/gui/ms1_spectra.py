"""Compact, synchronized native MS1 plots with persistent graphics and linked m/z."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGraphicsScene, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from kmzcore.labels import format_mz_label, label_aware_y, select_peak_labels
from kmzcore.qualified import qualified
from kmzcore.ranges import padded_y
from kmzcore.theme import INK, PAPER, SPECTRUM
from kmzviewer.comparison import SpectrumFrame
from kmzviewer.display_scale import peak_reference, percent, visible_signal
from kmzviewer.gui.axes import GridAxis


class ElidedLabel(QLabel):
    """Keep full metadata accessible without letting long names widen the dock."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text):  # noqa: N802
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setPen(self.palette().windowText().color())
        painter.drawText(self.contentsRect(), Qt.AlignmentFlag.AlignVCenter,
                         self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideMiddle, self.width()))


class CountsAxis(GridAxis):
    def tickStrings(self, values, scale, spacing):  # noqa: N802
        return [f"{value:.2g}" for value in values]


class SpectrumName(pg.TextItem):
    def __init__(self, view, font):
        super().__init__(color=INK, fill=PAPER)
        self.setFont(font)
        self.textItem.document().setDocumentMargin(0)
        self.setParentItem(view)
        self.setZValue(30)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.textItem.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def update_frame(self, frame, row=0):
        view = self.parentItem()
        self.setColor(frame.color)
        metrics = QFontMetricsF(self.textItem.font())
        self.setText(metrics.elidedText(frame.title, Qt.TextElideMode.ElideMiddle, max(0, int(view.width()) - 10)))
        self.setToolTip(frame.caption + ("\n" + frame.message if frame.message else ""))
        self.setPos(5, 2 + row * (metrics.height() + 2))
        self.setVisible(view.width() > 40 and view.height() > metrics.height() + 5)


class SpectrumPanel(QWidget):
    """One pane per file by default; overlay and matched-counts scales are optional.

    Arrays and profile connection masks are never modified. Plot/curve/label objects
    survive scan changes, and a frame is installed in one GUI-thread operation.
    """
    readout = Signal(str)
    msms_requested = Signal(int, float)
    gnps_requested = Signal(int, float, object, int)

    def __init__(self, parent=None, *, optical=False):
        super().__init__(parent)
        self.optical = optical
        self.x_label = 'Wavelength (nm)' if optical else 'm/z'
        self.y_label = 'Absorbance (mAU)' if optical else 'Counts'
        self._frames: list[SpectrumFrame] = []
        self._mode = "stacked"
        self._user_x = None
        self._guard = False
        self._layout_key = None
        self._plots_by_file = {}
        self._curves = {}
        self._references = {}
        self._curve_state = {}
        self._dots = {}
        self._names = {}
        self._label_pools = {}
        self._label_items = []
        self._mz = self._intensity = None  # Single-scan inspection API.
        self._label_font = QFont()
        self._label_font.setPointSize(9)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._update_labels)

        self.caption = ElidedLabel("Double-click a chromatogram to inspect an MS1 scan.")
        if optical:
            self.caption.setText('Extract a wavelength chromatogram to inspect DAD spectra.')
        self.position = ElidedLabel()
        self.position.hide()
        self.mode = QComboBox()
        self.mode.addItems(["Stacked", "Overlay"])
        self.mode.setAccessibleName("DAD spectrum layout" if optical else "MS1 spectrum layout")
        self.mode.currentTextChanged.connect(lambda text: self.set_mode(text.lower()))
        self.match_y = QCheckBox("Match Y")
        self.match_y.setToolTip(f"Share absolute {self.y_label} scales. When off, Overlay scales each spectrum's visible peak to 100%.")
        self.match_y.toggled.connect(self._update_labels)
        self.reset_button = QPushButton("Full m/z")
        self.reset_button.setToolTip("Restore the full m/z range across all visible spectra.")
        self.reset_button.clicked.connect(self.reset_range)
        if optical:
            self.reset_button.setText('Full wavelength')
            self.reset_button.setToolTip('Restore the full wavelength range. Shortcut: F in DAD spectra.')
        header = QHBoxLayout()
        header.setSpacing(6)
        header.addWidget(self.mode)
        header.addWidget(self.match_y)
        header.addWidget(self.caption, 1)
        header.addWidget(self.reset_button)
        self.view = pg.GraphicsLayoutWidget()
        # Labels and curves move with every frame; as in ChromatogramPanel, an
        # unindexed scene keeps no spatial index across retired label/plot trees.
        self.view.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.view.ci.setSpacing(0)
        self.view.ci.setContentsMargins(0, 0, 0, 0)
        self.view.scene().sigMouseMoved.connect(self._moved)
        self.view.scene().sigMouseClicked.connect(self._clicked)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 2)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self.view, 1)
        self._rebuild()

    @property
    def plot(self):
        """Primary PlotItem, also used by single-spectrum tooling."""
        return self._plots()[0]

    def _plots(self):
        return list(dict.fromkeys(self._plots_by_file.values())) or [self._empty_plot]

    def set_mode(self, mode):
        if mode not in {"stacked", "overlay"} or mode == self._mode:
            return
        self._mode = mode
        self.mode.setCurrentText(mode.title())
        self._rebuild()
        self._render()

    @property
    def relative_overlay(self):
        return self._mode == "overlay" and not self.match_y.isChecked()

    def _display_y(self, file_id, values):
        return percent(values, self._references.get(file_id, 1.0)) if self.relative_overlay else values

    def clear_scan(self, message):
        self._frames = []
        self._mz = self._intensity = None
        self.caption.setText(message)
        self.position.setText("")
        self._rebuild()

    def show_scan(self, mz, intensity, centroid, caption, *, connections=None):
        """Retained single-scan entry point for standalone plots and qualification."""
        frame = SpectrumFrame(0, caption, SPECTRUM, caption, np.asarray(mz), np.asarray(intensity),
                              centroid, connections)
        self.show_frames([frame], caption)

    def show_frames(self, frames, caption):
        self._frames = list(frames)
        self.caption.setText(caption)
        self.position.setText("")
        self._mz = frames[0].mz if frames else None
        self._intensity = frames[0].intensity if frames else None
        self._rebuild()
        self._render()

    def update_colors(self, colors):
        for frame in self._frames:
            frame.color = colors.get(frame.file_id, frame.color)
        self._render()

    def reorder_files(self, file_ids):
        """Move complete panes without rebuilding curves or decoding arrays."""
        frames = {frame.file_id: frame for frame in self._frames}
        ordered = [frames[fid] for fid in file_ids if fid in frames]
        if [f.file_id for f in ordered] == [f.file_id for f in self._frames]:
            return
        self._guard = True
        try:
            self._frames = ordered
            self._mz = ordered[0].mz if ordered else None
            self._intensity = ordered[0].intensity if ordered else None
            self._plots_by_file = {f.file_id: self._plots_by_file[f.file_id] for f in ordered}
            if self._mode == "stacked":
                plots = self._plots()
                for plot in plots:
                    plot.setXLink(None)
                    self.view.ci.removeItem(plot)
                for i, plot in enumerate(plots):
                    self.view.ci.addItem(plot, row=i, col=0)
                    plot.setLabel("bottom", self.x_label if i == len(plots) - 1 else "")
                    plot.showAxis("bottom", i == len(plots) - 1)
                    plot.getAxis("bottom").setStyle(tickFont=self._label_font)
                    if i:
                        plot.setXLink(plots[0])
            self._layout_key = (self._mode, tuple(f.file_id for f in ordered))
        finally:
            self._guard = False
        self._update_labels()

    def _rebuild(self):
        key = (self._mode, tuple(frame.file_id for frame in self._frames))
        if key == self._layout_key:
            return
        if (self._layout_key and key[0] == self._layout_key[0]
                and set(key[1]) == set(self._layout_key[1]) and key[1]):
            # A late read can carry a new display order while membership is stable.
            frames = self._frames
            by_id = {frame.file_id: frame for frame in frames}
            self._frames = [by_id[fid] for fid in self._layout_key[1]]
            self.reorder_files(key[1])
            return
        self._guard = True
        try:
            self.view.ci.clear()
            self._plots_by_file.clear()
            self._curves.clear()
            self._references.clear()
            self._curve_state.clear()
            self._dots.clear()
            self._names.clear()
            self._label_pools.clear()
            self._label_items.clear()
            count = len(self._frames) if self._mode == "stacked" else min(len(self._frames), 1)
            plots = []
            for index in range(max(1, count)):
                axis = CountsAxis("left")
                axis.setWidth(57)
                axis.setStyle(tickFont=self._label_font, tickTextOffset=3, autoExpandTextSpace=False)
                axis.enableAutoSIPrefix(False)
                plot = pg.PlotItem(axisItems={"left": axis, "bottom": GridAxis("bottom")}, enableMenu=False)
                plot.layout.setContentsMargins(0, 0, 0, 0)
                plot.layout.setVerticalSpacing(0)
                if self.optical:
                    plot.setLabel('left', 'mAU')
                plot.setMenuEnabled(False)
                plot.getViewBox().setMenuEnabled(False)
                plot.setMouseEnabled(x=True, y=False)
                plot.enableAutoRange(enable=False)
                plot.showGrid(x=True, y=True, alpha=.10)
                plot.hideButtons()
                if index == max(1, count) - 1:
                    plot.setLabel("bottom", self.x_label)
                    plot.getAxis("bottom").setStyle(tickFont=self._label_font)
                else:
                    plot.hideAxis("bottom")
                self.view.ci.addItem(plot, row=index, col=0)
                if plots:
                    plot.setXLink(plots[0])
                plots.append(plot)
                plot.sigXRangeChanged.connect(self._on_x)
                plot.getViewBox().sigResized.connect(self._schedule_labels)
            self._empty_plot = plots[0]
            for index, frame in enumerate(self._frames):
                plot = plots[index] if self._mode == "stacked" else plots[0]
                self._plots_by_file[frame.file_id] = plot
                self._curves[frame.file_id] = plot.plot([], [], pen=pg.mkPen(frame.color, width=1))
                dots = pg.ScatterPlotItem(size=3, pen=None, brush=frame.color)
                plot.addItem(dots)
                self._dots[frame.file_id] = dots
                self._names[frame.file_id] = SpectrumName(plot.getViewBox(), self._label_font)
                self._label_pools[frame.file_id] = []
            self._layout_key = key
        finally:
            self._guard = False

    def _render(self):
        if not self._frames:
            return
        self._guard = True
        try:
            bounds = self._user_x or self._full_x()
            if bounds is not None:
                for plot in self._plots():
                    plot.setXRange(*bounds, padding=0)
        finally:
            self._guard = False
        self._update_labels()

    def _draw_frame(self, frame):
        fid = frame.file_id
        state = (frame, self._references.get(fid, 1.0), self.relative_overlay, frame.color)
        old = self._curve_state.get(fid)
        if old is not None and old[0] is frame and old[1:] == state[1:]:
            return
        curve = self._curves[fid]
        pen = pg.mkPen(frame.color, width=1)
        if frame.centroid:
            x, y = _sticks(frame.mz, self._display_y(fid, frame.intensity))
            curve.setData(x, y, pen=pen, connect="pairs")
        else:
            x, y = frame.profile_curve
            curve.setData(x, self._display_y(fid, y), pen=pen,
                          connect="auto" if frame.connections is None else "all")
        dots = self._dots[fid]
        if frame.connections is not None and frame.mz.size and not frame.centroid:
            isolated = ~np.r_[False, frame.connections] & ~np.r_[frame.connections, False]
            dots.setData(frame.mz[isolated], self._display_y(fid, frame.intensity[isolated]), brush=frame.color)
        else:
            dots.setData([], [])
        self._curve_state[fid] = state

    def _full_x(self):
        ends = []
        for frame in self._frames:
            finite = frame.mz[np.isfinite(frame.mz)]
            if finite.size:
                ends.extend((float(finite.min()), float(finite.max())))
        if not ends:
            return None
        lo, hi = min(ends), max(ends)
        if lo == hi:
            lo, hi = lo - .5, hi + .5
        pad = (hi - lo) * .02
        return lo - pad, hi + pad

    def reset_range(self):
        self._user_x = None
        self._render()

    def _on_x(self, view, _range):
        if self._guard or not self._frames:
            return
        self._user_x = tuple(map(float, view.viewRange()[0]))
        self._guard = True
        try:
            for plot in self._plots():
                if plot.getViewBox() is not view:
                    plot.setXRange(*self._user_x, padding=0)
        finally:
            self._guard = False
        self._update_labels()

    def _schedule_labels(self):
        if not self._guard:
            self._resize_timer.start(0)

    def _update_labels(self, *_):
        if self._guard or not self._frames:
            return
        self._guard = True
        try:
            plots = self._plots()
            x0, x1 = map(float, plots[0].viewRange()[0])
            self._label_items = []
            for plot in plots:
                plot.setLabel("left", "Relative intensity (%)" if self.relative_overlay else ("mAU" if self.optical else ""))
                has_data = any(frame.mz.size for frame in self._frames
                               if self._plots_by_file[frame.file_id] is plot)
                axis = plot.getAxis("left")
                if axis.style["showValues"] != has_data:
                    axis.setStyle(showValues=has_data, tickLength=-5 if has_data else 0)
                    plot.showGrid(x=True, y=has_data, alpha=.10)
            bounds = {}
            candidates = {}
            for index, frame in enumerate(self._frames):
                plot = self._plots_by_file[frame.file_id]
                view = plot.getViewBox()
                self._names[frame.file_id].update_frame(frame, index if self._mode == "overlay" else 0)
                x, y = (frame.mz, frame.intensity) if frame.centroid else frame.profile_curve
                visible = visible_signal(x, y, x0, x1, sticks=frame.centroid)
                self._references[frame.file_id] = peak_reference(visible) if self.relative_overlay else 1.0
                self._draw_frame(frame)
                # Include the physical zero baseline for centroid sticks and counts comparisons.
                displayed = self._display_y(frame.file_id, visible)
                if self.relative_overlay and (not displayed.size or not np.any(displayed)):
                    displayed = np.array([0., 100.])
                bounds[frame.file_id] = padded_y(np.r_[0., displayed])
                candidates[frame.file_id] = select_peak_labels(frame.mz, frame.intensity, x0, x1,
                    centroid=frame.centroid, view_width_px=view.width())
            if self._mode == "overlay":
                # One collision budget across every source, with the tallest visible peaks first.
                combined = sorted(((float(self._display_y(fid, peak)), mass, fid, peak) for fid, points in candidates.items()
                                   for mass, peak in points), reverse=True)
                candidates = {frame.file_id: [] for frame in self._frames}
                used = []
                for _display_peak, mass, fid, peak in combined:
                    px = (mass - x0) / max(x1 - x0, 1e-12) * plots[0].getViewBox().width()
                    if all(abs(px - other) >= 72 for other in used):
                        candidates[fid].append((mass, peak))
                        used.append(px)
            all_bounds = []
            for frame in self._frames:
                fid = frame.file_id
                plot = self._plots_by_file[fid]
                view = plot.getViewBox()
                pool = self._label_pools[fid]
                points = candidates[fid]
                while len(pool) < len(points):
                    item = pg.TextItem(color=INK, anchor=(.5, 1.15))
                    item.setFont(self._label_font)
                    item.setZValue(20)
                    item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    item.textItem.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    plot.addItem(item, ignoreBounds=True)
                    pool.append(item)
                anchors = []
                label_height = 0
                for index, item in enumerate(pool):
                    if index >= len(points):
                        item.hide()
                        continue
                    mass, peak = points[index]
                    item.msms_target = (fid, float(mass))
                    item.setToolTip(f"{frame.name}\n{mass:g} nm; {peak:g} mAU" if self.optical else
                                    f"{frame.name}\nRight-click to search MS/MS at m/z {mass!r} ±20 ppm.")
                    item.setText(f'{mass:g}' if self.optical else format_mz_label(mass), color=frame.color if self._mode == "overlay" else INK)
                    rect = item.textItem.boundingRect()
                    title_height = QFontMetricsF(self._label_font).height() + 5
                    band = title_height * (len(self._frames) if self._mode == "overlay" else 1)
                    if view.height() < rect.height() * 1.15 + band + 12 or view.width() < rect.width() + 6:
                        item.hide()
                        continue
                    px = (mass - x0) / max(x1 - x0, 1e-12) * view.width()
                    half = rect.width() / 2
                    shift = max(3 - (px - half), 0) - max(px + half - (view.width() - 3), 0)
                    item.setAnchor((.5 - shift / rect.width(), 1.15))
                    display_peak = float(self._display_y(fid, peak))
                    item.setPos(mass, display_peak)
                    item.show()
                    self._label_items.append(item)
                    anchors.append((mass, display_peak))
                    label_height = max(label_height, rect.height() * 1.15 + band + 3)
                bounds[fid] = label_aware_y(*bounds[fid], anchors, view.height(), label_height)
                all_bounds.append(bounds[fid])
            shared = (min(b[0] for b in all_bounds), max(b[1] for b in all_bounds))
            for frame in self._frames:
                limits = shared if self._mode == "overlay" or self.match_y.isChecked() else bounds[frame.file_id]
                self._plots_by_file[frame.file_id].setYRange(*limits, padding=0)
        finally:
            self._guard = False
        # Apply view/label transforms now, as the next paint would; pyqtgraph's
        # paint-time update re-dirties the scene and repaints dense spectra twice.
        # Hidden panels are not painted and keep pyqtgraph's deferred update, as
        # do Qt or pyqtgraph versions other than those qualified.
        if self.view.isVisible() and qualified("Qt", "PySide6", "pyqtgraph"):
            self.view.scene().prepareForPaint()

    def _label_context_menu(self, scene_pos):
        if self.optical:
            return None
        for item in reversed(self._label_items):
            if not item.isVisible() or not item.textItem.contains(item.textItem.mapFromScene(scene_pos)):
                continue
            file_id, mass = item.msms_target
            view = self._plots_by_file[file_id].getViewBox()
            if not view.sceneBoundingRect().contains(scene_pos):
                continue
            menu = QMenu(self)
            menu.addAction("Open MS/MS in Kmz2Viewer",
                           lambda fid=file_id, mz=mass: self.msms_requested.emit(fid, mz))
            frame = next(frame for frame in self._frames if frame.file_id == file_id)
            menu.addAction("Spectral analysis…",
                           lambda fid=file_id, mz=mass, rt=frame.rt, polarity=frame.polarity:
                           self.gnps_requested.emit(fid, mz, rt, polarity))
            return menu
        return None

    def _clicked(self, event):
        if event.button() != Qt.MouseButton.RightButton:
            return
        menu = self._label_context_menu(event.scenePos())
        if menu is not None:
            event.accept()
            menu.aboutToHide.connect(menu.deleteLater)
            menu.popup(event.screenPos().toPoint())

    def _moved(self, scene_pos):
        best = None
        for frame in reversed(self._frames):
            view = self._plots_by_file[frame.file_id].getViewBox()
            if not view.sceneBoundingRect().contains(scene_pos) or not frame.mz.size:
                continue
            mass = float(view.mapSceneToView(scene_pos).x())
            distance = np.abs(frame.mz - mass)
            if not np.any(np.isfinite(distance)):
                continue
            index = int(np.nanargmin(distance))
            peak = view.mapViewToScene(pg.QtCore.QPointF(float(frame.mz[index]), float(self._display_y(frame.file_id, frame.intensity[index]))))
            separation = abs(peak.y() - scene_pos.y())
            if best is None or separation < best[0]:
                best = (separation, frame, index)
        if best is not None:
            _, frame, index = best
            text = (f"{frame.name} · {frame.mz[index]:g} nm · {frame.intensity[index]:.6g} mAU" if self.optical else
                    f"{frame.name} · m/z {frame.mz[index]:.4f} · {frame.intensity[index]:.6g} counts")
            self.position.setText(text)
            self.readout.emit(text)


def _sticks(mz, intensity):
    x = np.repeat(mz, 2)
    y = np.zeros(len(intensity) * 2, dtype=np.float64)
    y[1::2] = intensity
    return x, y
