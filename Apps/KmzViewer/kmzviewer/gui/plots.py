"""Interactive chromatogram and spectrum plots."""

from __future__ import annotations

import os
import weakref
from contextlib import contextmanager

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from PySide6.QtWidgets import QApplication, QGraphicsScene, QVBoxLayout, QWidget

from kmzcore.qualified import qualified
from kmzcore.ranges import fill_floor, padded_y
from kmzcore.theme import ACCENT, INK, PAPER, configure_plots
from kmzcore.trace_style import fill_rgba, line_style

from kmzviewer.gui.axes import GridAxis
from kmzviewer.gui.curves import ChromatogramItem
from kmzviewer.gui.ms1_spectra import SpectrumPanel
from kmzviewer.display_scale import peak_reference, percent, visible_signal

configure_plots()


@contextmanager
def _one_layout_pass(layout):
    """Lay out a batch of stacked panes once.

    GraphicsLayout.addItem re-activates the whole grid for every pane, resizing
    all earlier panes again (quadratic resize/range/axis-label cascades).
    """
    layout.activate = lambda: None
    try:
        yield
    finally:
        del layout.activate
        layout.activate()


class TraceSample(ItemSample):
    """Overlay sample: emphasis only, replacing ItemSample's visibility toggle."""
    clicked = Signal(int, bool)

    def __init__(self, item):
        super().__init__(item)
        self.trace_id: int | None = None

    def paint(self, painter, *args):
        painter.setPen(self.item.opts["pen"])
        painter.drawLine(1, 10, 19, 10)

    def mouseClickEvent(self, event):
        pos = event.pos()
        if self.trace_id is None or event.button() != Qt.MouseButton.LeftButton or not (0 <= pos.x() <= 20 and 5 <= pos.y() <= 15):
            return
        self.clicked.emit(self.trace_id, event.double())
        event.accept()


class PlotLabel(pg.TextItem):
    """Pixel-positioned interior title, outside data items and plot layout."""
    def __init__(self, title, view):
        paper = QColor(PAPER)
        paper.setAlpha(224)
        super().__init__(color=INK, fill=paper)
        self.title = title
        font = QFont(QApplication.font())
        font.setPointSize(9)
        self.setFont(font)
        self.textItem.document().setDocumentMargin(0)
        self.setZValue(30)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.textItem.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setToolTip(title)
        self.setParentItem(view)
        self.setPos(5, 3)
        view.sigResized.connect(self._relayout)
        self._relayout()

    def _relayout(self):
        view = self.parentItem()
        if view is None:
            return
        metrics = QFontMetricsF(self.textItem.font())
        width = max(0, int(view.width()) - 10)
        self.setText(metrics.elidedText(self.title, Qt.TextElideMode.ElideMiddle, width))
        self.setVisible(width >= 20 and view.height() >= metrics.height() + 6)


class ChromatogramViewBox(pg.ViewBox):
    """Let Walk own left drags without interfering with axes or other mouse modes."""
    def __init__(self, panel):
        # The panel disables every context menu; never build one to discard.
        super().__init__(enableMenu=False)
        self._panel = weakref.ref(panel)
        self._scrub_trace = None

    def mouseDragEvent(self, event, axis=None):  # noqa: N802
        panel = self._panel()
        if panel is not None and axis is None and event.button() == Qt.MouseButton.LeftButton and panel._mouse_mode == "walk":
            event.accept()
            panel._cancel_emphasis()
            if event.isStart():
                hit = panel._hit(event.buttonDownScenePos())
                self._scrub_trace = hit[0] if hit is not None else None
            if self._scrub_trace is not None:
                # Keep the source picked at drag start, even when overlay curves cross.
                x = float(self.mapSceneToView(event.scenePos()).x())
                panel.scan_requested.emit(self._scrub_trace, x)
            if event.isFinish():
                self._scrub_trace = None
            return
        super().mouseDragEvent(event, axis)


class ChromatogramPanel(QWidget):
    """Stacked or overlaid chromatograms with a shared retention-time axis."""

    trace_activated = Signal(int)
    scan_requested = Signal(int, float)
    spectrum_added = Signal(int, float)
    readout = Signal(str)
    emphasis_requested = Signal(int)

    def __init__(self, parent=None, *, inspection_enabled=True, legend=True, normalize_overlay=False) -> None:
        super().__init__(parent)
        self.inspection_enabled = inspection_enabled
        self.legend_visible = legend
        self.normalize_overlay = normalize_overlay
        self._references = {}
        self.walk_clicks = False
        self.autoscale_y = True
        self.match_y = False
        self._mouse_mode = "pan"
        self._active_id: int | None = None
        self._guard = False
        self._mode = "stacked"
        self._units = {}
        self._right_view = None
        self._rows: list = []
        self._series: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._labels: dict[int, str] = {}
        self._plot_for_trace: dict[int, pg.PlotItem] = {}
        self._walk_lines: list[pg.InfiniteLine] = []
        self._walk_rt = None
        self._hover_lines: list[pg.InfiniteLine] = []
        self._overlay_plot: pg.PlotItem | None = None
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._samples: dict[int, ItemSample] = {}
        self._inplot_labels: dict[int, PlotLabel] = {}
        self._colors: dict[int, str] = {}
        self._emphasized: set[int] = set()
        self._fitted = None
        self._pending_emphasis: int | None = None
        self._emphasis_timer = QTimer(self)
        self._emphasis_timer.setSingleShot(True)
        self._emphasis_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._emphasis_timer.timeout.connect(self._commit_emphasis)

        self.view = pg.GraphicsLayoutWidget()
        # Layout switches replace whole item trees. Use an unindexed dynamic
        # scene so native paint/hit traversal does not maintain a spatial index
        # across retirement and collection of earlier overlay trees.
        self.view.scene().setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.view.ci.setSpacing(0)
        self.view.ci.setContentsMargins(0, 0, 0, 0)
        self.view.scene().sigMouseClicked.connect(self._clicked)
        self.view.scene().sigMouseMoved.connect(self._moved)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

    def set_walk_clicks(self, enabled: bool) -> None:
        self.walk_clicks = enabled and self.inspection_enabled

    def set_mode(self, mode: str) -> None:
        if mode not in {"stacked", "overlay"} or mode == self._mode:
            return
        self._mode = mode
        self._rebuild(keep_x=True)

    def set_traces(self, rows: list[tuple[int, str, str, np.ndarray, np.ndarray]], *, units=None) -> None:
        """rows are (trace_id, title, color, rt, y) for visible traces, in display order."""
        previous_rows = {row[0]: row for row in self._rows}
        new_units = units or {}
        units_changed = new_units != self._units
        # Stack panes whose title, colour, arrays and unit are unchanged survive
        # a membership change; only added traces build panes. Automatic Y only:
        # a rebuild fits manual-Y panes to a new pane's default X window.
        retain = (self._mode == "stacked" and self.autoscale_y and bool(rows) and bool(previous_rows)
                  and all(row[0] not in previous_rows
                          or (row[:3] == previous_rows[row[0]][:3] and row[3] is previous_rows[row[0]][3]
                              and row[4] is previous_rows[row[0]][4]
                              and new_units.get(row[0], 'counts') == self._units.get(row[0], 'counts'))
                          for row in rows))
        self._units = dict(new_units)
        if (not units_changed and len(rows) == len(previous_rows)
                and all(row[0] in previous_rows and row[:3] == previous_rows[row[0]][:3]
                        and row[3] is previous_rows[row[0]][3] and row[4] is previous_rows[row[0]][4]
                        for row in rows)):
            if [row[0] for row in rows] != [row[0] for row in self._rows]:
                self._reorder_traces(rows)
            return
        if (not units_changed and [row[0] for row in rows] == [row[0] for row in self._rows]
                and all(row[3] is previous_rows[row[0]][3] for row in rows)):
            # A display filter replaces values/titles, not plot membership or
            # native RTs. Reuse panes, curves, cursors and the current X/manual Y.
            self._rows = rows
            self._references.clear()
            for tid, title, color, rt, values in rows:
                self._remember(tid, title, rt, values, self._plot_for_trace[tid])
                curve = self._curves.get(tid)
                if curve is not None:
                    curve.setData(rt, values, fillLevel=fill_floor(rt, values))
                    curve.opts['name'] = title
                label = self._inplot_labels.get(tid)
                if label is not None:
                    label.title = title
                    label.setToolTip(title)
                    label._relayout()
                if self._overlay_plot is not None:
                    for sample, text in self._overlay_plot.legend.items:
                        if sample.trace_id == tid:
                            text.setText(title)
            self.update_colors({row[0]: row[2] for row in rows})
            if self.relative_overlay:
                self._update_relative_curves()
            if self.autoscale_y:
                self._fit_all_y()
            self._settle_transforms()
            return
        previous = self.current_x_range() if self._plot_for_trace else None
        manual_y = ({tid: self._trace_view(tid).viewRange()[1] for tid in self._plot_for_trace}
                    if not self.autoscale_y else {})
        self._rows = rows
        if retain:
            self._restack()
        else:
            self._rebuild(keep_x=False)
        if previous is not None and self._plot_for_trace:
            self.set_x_range(*previous)
        elif self._plot_for_trace:
            self.reset_view()
        for tid, limits in manual_y.items():
            if tid in self._plot_for_trace:
                self._trace_view(tid).setYRange(*limits, padding=0)

    def _reorder_traces(self, rows) -> None:
        """Reorder existing panes/legend entries without recreating their graphics."""
        self._cancel_emphasis()
        self._guard = True
        try:
            hover_by_plot = dict(zip(self._plots(), self._hover_lines))
            walk_by_plot = dict(zip(self._plots(), self._walk_lines))
            self._rows = rows
            self._plot_for_trace = {row[0]: self._plot_for_trace[row[0]] for row in rows}
            self._curves = {row[0]: self._curves[row[0]] for row in rows if row[0] in self._curves}
            if self._mode == "stacked":
                plots = self._plots()
                self._hover_lines = [hover_by_plot[plot] for plot in plots]
                self._walk_lines = [walk_by_plot[plot] for plot in plots]
                for plot in plots:
                    plot.setXLink(None)
                    self.view.ci.removeItem(plot)
                with _one_layout_pass(self.view.ci.layout):
                    for i, plot in enumerate(plots):
                        self.view.ci.addItem(plot, row=i, col=0)
                        plot.setLabel("bottom", "Retention time (min)" if i == len(plots) - 1 else "")
                for plot in plots[1:]:
                    plot.setXLink(plots[0])
            else:
                legend = self._overlay_plot.legend
                entries = {sample.trace_id: (sample, label) for sample, label in legend.items}
                for i in range(legend.layout.count() - 1, -1, -1):
                    legend.layout.removeAt(i)
                legend.items = [entries[row[0]] for row in rows if row[0] in entries]
                for i, (sample, label) in enumerate(legend.items):
                    legend.layout.addItem(sample, i, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
                    legend.layout.addItem(label, i, 1)
                    sample.item.setZValue(i / (len(legend.items) + 1))
                legend.updateSize()
        finally:
            self._guard = False

    def set_active(self, trace_id: int | None) -> None:
        self._active_id = trace_id
        if self._mode == "overlay":
            if self._overlay_plot is not None:
                pen = pg.mkPen(ACCENT, width=1.5) if trace_id in self._series else None
                self._overlay_plot.getViewBox().setBorder(pen)
            return
        for candidate, plot in self._plot_for_trace.items():
            pen = pg.mkPen(ACCENT, width=1.5) if candidate == trace_id else None
            plot.getViewBox().setBorder(pen)

    def set_autoscale_y(self, enabled: bool) -> None:
        self.autoscale_y = enabled
        self._apply_y_policy()

    def set_match_y(self, enabled: bool) -> None:
        was_relative = self.relative_overlay
        self.match_y = enabled
        if was_relative != self.relative_overlay:
            self._rebuild(keep_x=True)
            self._fit_all_y()
            return
        self._apply_y_policy()

    @property
    def relative_overlay(self):
        return self.normalize_overlay and self._mode == "overlay" and not self.match_y

    def _display_y(self, trace_id, values):
        return percent(values, self._references.get(trace_id, 1.0)) if self.relative_overlay else values

    def set_mouse_mode(self, mode: str) -> None:
        if mode == 'walk' and not self.inspection_enabled:
            return
        self._mouse_mode = mode
        view_mode = pg.ViewBox.RectMode if mode == "zoom" else pg.ViewBox.PanMode
        for plot in self._plots():
            plot.getViewBox().setMouseMode(view_mode)
        if self._right_view is not None:
            self._right_view.setMouseMode(view_mode)

    def current_x_range(self) -> tuple[float, float] | None:
        plots = self._plots()
        if not plots:
            return None
        x0, x1 = plots[0].viewRange()[0]
        return float(x0), float(x1)

    def data_x_range(self) -> tuple[float, float] | None:
        finite = []
        for rt, _y in self._series.values():
            values = rt[np.isfinite(rt)] if rt.size else rt
            if values.size:
                finite.append(float(values.min()))
                finite.append(float(values.max()))
        if not finite:
            return None
        return min(finite), max(finite)

    def set_x_range(self, xmin: float, xmax: float) -> None:
        plots = self._plots()
        if not plots or xmax <= xmin:
            return
        already = self._guard
        self._guard = True
        try:
            plots[0].setXRange(xmin, xmax, padding=0)
            if self.autoscale_y:
                self._fit_all_y((xmin, xmax))
            elif self.relative_overlay:
                self._update_relative_curves((xmin, xmax))
        finally:
            self._guard = already
        self._settle_transforms()

    def reset_view(self) -> None:
        bounds = self.data_x_range()
        if bounds is None:
            return
        xmin, xmax = bounds
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        pad = (xmax - xmin) * 0.01
        self.set_x_range(xmin - pad, xmax + pad)
        if not self.autoscale_y:
            self._fit_all_y()

    def autoscale_visible_y(self) -> None:
        self._fit_all_y()

    def set_walk_rt(self, rt: float | None) -> None:
        self._walk_rt = rt
        visible = rt is not None
        for line in self._walk_lines:
            line.setVisible(visible)
            if visible:
                line.setPos(rt)

    def _rebuild(self, keep_x: bool) -> None:
        already = self._guard
        self._guard = True
        try:
            self._rebuild_plots(keep_x)
        finally:
            self._guard = already
        self._settle_transforms()

    def _settle_transforms(self) -> None:
        """Apply pending view transforms now, exactly as the next paint would.

        pyqtgraph otherwise applies them when painting starts, which marks the
        scene dirty again and repeats the complete paint of every dense trace.
        Hidden panels are not painted and keep pyqtgraph's deferred update, as
        do Qt or pyqtgraph versions other than those qualified.
        """
        if self.view.isVisible() and qualified("Qt", "PySide6", "pyqtgraph"):
            self.view.scene().prepareForPaint()

    def _rebuild_plots(self, keep_x: bool) -> None:
        self._cancel_emphasis()
        rows = getattr(self, "_rows", [])
        previous = self.current_x_range() if keep_x else None
        if self._right_view is not None:
            self._right_view.sigXRangeChanged.disconnect(self._x_changed)
            self._overlay_plot.getViewBox().sigResized.disconnect(self._sync_right_geometry)
            self.view.scene().removeItem(self._right_view)
            self._right_view.deleteLater()
            self._right_view = None
        self.view.ci.clear()
        self._series.clear()
        self._labels.clear()
        self._plot_for_trace.clear()
        self._walk_lines.clear()
        self._hover_lines.clear()
        self._overlay_plot = None
        self._curves.clear()
        self._samples.clear()
        self._inplot_labels.clear()
        self._colors.clear()
        self._references.clear()
        self._fitted = None
        if not rows:
            return
        if self._mode == "overlay":
            plot = self._new_plot(bottom=True)
            self._overlay_plot = plot
            self.view.ci.addItem(plot, row=0, col=0)
            if not self.relative_overlay and any(self._units.get(row[0]) == 'mAU' for row in rows):
                self._right_view = ChromatogramViewBox(self)
                self._right_view.setMenuEnabled(False)
                self._right_view.enableAutoRange(enable=False)
                self.view.scene().addItem(self._right_view)
                plot.showAxis('right')
                plot.setLabel('right', 'Absorbance (mAU)')
                plot.getAxis('right').enableAutoSIPrefix(False)
                plot.getAxis('right').linkToView(self._right_view)
                self._right_view.setXLink(plot)
                self._right_view.sigXRangeChanged.connect(self._x_changed)
                plot.getViewBox().sigResized.connect(self._sync_right_geometry)
                self._sync_right_geometry()
                plot.showAxis('left', any(self._units.get(row[0], 'counts') == 'counts' for row in rows))
            legend = plot.addLegend(offset=(8, 8), sampleType=TraceSample)
            legend.setLabelTextColor(INK)
            legend.setVisible(self.legend_visible)
            ion_units = [self._units.get(row[0], 'counts') for row in rows if self._units.get(row[0]) != 'mAU']
            if self.relative_overlay:
                plot.setLabel('left', 'Relative intensity (%)')
                plot.getAxis('left').enableAutoSIPrefix(False)
            elif ion_units:
                plot.setLabel('left', self._unit_label(ion_units[0]))
            # The legend may consume native double clicks before the scene
            # delivers a synthetic double-click event to its sample.
            legend.sigDoubleClicked.connect(self._cancel_emphasis)
            for trace_id, title, color, rt, values in rows:
                self._remember(trace_id, title, rt, values, plot)
                if rt.size:
                    self._add_curve(plot, trace_id, color, rt, values, name=title)
                    sample = legend.items[-1][0]
                    sample.trace_id = trace_id
                    sample.setToolTip("Click the sample or plot line to toggle emphasis." +
                                      (" Double-click the plot to inspect a scan." if self.inspection_enabled else ''))
                    sample.clicked.connect(self._emphasis_click)
                    self._samples[trace_id] = sample
            self._add_cursors(plot)
        else:
            plots = []
            with _one_layout_pass(self.view.ci.layout):
                for index, row in enumerate(rows):
                    plots.append(self._stacked_pane(index, row, index == len(rows) - 1))
            # Link once each pane has its final width: X follows the first pane.
            for plot in plots[1:]:
                plot.setXLink(plots[0])
        self.set_mouse_mode(self._mouse_mode)
        for tid, curve in self._curves.items():
            rt = self._series[tid][0]
            # Clip only after attachment to the final ViewBox. Repeated RTs
            # must remain un-clipped: the library can omit tied edge peaks.
            curve.setClipToView(bool(np.isfinite(rt).all() and np.all(rt[1:] > rt[:-1])))
        self._apply_y_policy()
        self.set_active(self._active_id)
        self.set_walk_rt(self._walk_rt)
        if previous is not None:
            self.set_x_range(*previous)
        if not self.autoscale_y:
            # A new layout/unit system needs valid initial bounds. Subsequent
            # zooms retain manual Y; set_traces restores any retained pane bounds.
            self._fit_all_y()

    def _stacked_pane(self, index, row, bottom) -> pg.PlotItem:
        trace_id, title, color, rt, values = row
        plot = self._new_plot(bottom=bottom)
        if self._units.get(trace_id, 'counts') != 'counts':
            plot.setLabel('left', self._unit_label(self._units[trace_id]))
            plot.getAxis('left').enableAutoSIPrefix(False)
        self.view.ci.addItem(plot, row=index, col=0)
        self._remember(trace_id, title, rt, values, plot)
        if rt.size:
            self._add_curve(plot, trace_id, color, rt, values)
        plot.setTitle(None)
        self._inplot_labels[trace_id] = PlotLabel(title, plot.getViewBox())
        self._add_cursors(plot)
        return plot

    def _restack(self) -> None:
        """Change stack membership, retaining unchanged panes as built.

        Equivalent to _rebuild_plots(keep_x=False) with automatic Y for stacked
        rows whose surviving panes keep title, colour, arrays and unit. Retired
        panes leave the scene as complete trees; retained cursors return to
        their initial state.
        """
        self._cancel_emphasis()
        already = self._guard
        self._guard = True
        try:
            rows = self._rows
            old = self._plot_for_trace
            cursors = dict(zip(self._plots(), zip(self._walk_lines, self._hover_lines)))
            kept = {row[0] for row in rows}
            for trace_id, plot in old.items():
                plot.setXLink(None)
                self.view.ci.removeItem(plot)
                if trace_id not in kept:
                    for owned in (self._series, self._labels, self._curves, self._inplot_labels, self._colors):
                        owned.pop(trace_id, None)
            self._plot_for_trace = {}
            self._walk_lines = []
            self._hover_lines = []
            self._references.clear()
            self._fitted = None
            plots = []
            with _one_layout_pass(self.view.ci.layout):
                for index, row in enumerate(rows):
                    bottom = index == len(rows) - 1
                    plot = old.get(row[0])
                    if plot is None:
                        plot = self._stacked_pane(index, row, bottom)
                    else:
                        self.view.ci.addItem(plot, row=index, col=0)
                        plot.setLabel("bottom", "Retention time (min)" if bottom else "")
                        self._plot_for_trace[row[0]] = plot
                        for line in cursors[plot]:
                            line.setVisible(False)
                            line.setPos(pg.Point(0, 0))
                        self._walk_lines.append(cursors[plot][0])
                        self._hover_lines.append(cursors[plot][1])
                    plots.append(plot)
            for plot in plots[1:]:
                plot.setXLink(plots[0])
            self._curves = {row[0]: self._curves[row[0]] for row in rows if row[0] in self._curves}
            self.set_mouse_mode(self._mouse_mode)
            for tid, curve in self._curves.items():
                rt = self._series[tid][0]
                curve.setClipToView(bool(np.isfinite(rt).all() and np.all(rt[1:] > rt[:-1])))
            self._apply_y_policy()
            self.set_active(self._active_id)
            self.set_walk_rt(self._walk_rt)
        finally:
            self._guard = already
        self._settle_transforms()

    def _sync_right_geometry(self):
        if self._right_view is not None:
            primary = self._overlay_plot.getViewBox()
            self._right_view.setGeometry(primary.mapRectToScene(primary.rect()))
            self._right_view.linkedViewChanged(self._overlay_plot.getViewBox(), self._right_view.XAxis)

    @staticmethod
    def _unit_label(unit):
        return {'counts': 'Counts', 'mAU': 'Absorbance (mAU)', 'unknown': 'Native intensity (unit unspecified)'}.get(unit, unit)

    def _trace_view(self, trace_id):
        if self._right_view is not None and self._units.get(trace_id) == 'mAU':
            return self._right_view
        return self._plot_for_trace[trace_id].getViewBox()

    def _add_curve(self, plot, trace_id, color, rt, values, **kwargs):
        self._colors[trace_id] = color
        stroke, width = line_style(color, self._mode == "overlay" and trace_id in self._emphasized)
        style = dict(pen=pg.mkPen(stroke, width=width), fillLevel=fill_floor(rt, values),
                     fillBrush=pg.mkBrush(*fill_rgba(color)), **kwargs)
        if self._right_view is not None and self._units.get(trace_id) == 'mAU':
            # One graphics owner: do not leave an optical curve registered in
            # PlotItem.items while a separate ViewBox owns its native lifetime.
            curve = ChromatogramItem(rt, values, **style)
            self._right_view.addItem(curve)
            if plot.legend is not None:
                plot.legend.addItem(curve, kwargs['name'])
        else:
            curve = ChromatogramItem(rt, values, **style)
            plot.addItem(curve)
        # Used only for scene-space hit testing; leave normal drag/click dispatch intact.
        curve.curve.setClickable(False, width=10)
        self._curves[trace_id] = curve
        return curve

    def update_colors(self, colors: dict[int, str]) -> None:
        self._rows = [(tid, title, colors.get(tid, color), rt, y) for tid, title, color, rt, y in self._rows]
        for trace_id, color in colors.items():
            curve = self._curves.get(trace_id)
            if curve is not None:
                self._colors[trace_id] = color
                curve.setFillBrush(pg.mkBrush(*fill_rgba(color)))
        self.set_emphasized(self._emphasized)

    def set_emphasized(self, trace_ids: set[int]) -> None:
        self._emphasized = set(trace_ids)
        for trace_id, curve in self._curves.items():
            color, width = line_style(self._colors[trace_id], self._mode == "overlay" and trace_id in self._emphasized)
            pen = pg.mkPen(color, width=width)
            if curve.opts['pen'] == pen:
                # setPen would re-set identical data and rebuild every cached path.
                continue
            curve.setPen(pen)
            sample = self._samples.get(trace_id)
            if sample is not None:
                sample.update()

    def _cancel_emphasis(self) -> None:
        self._emphasis_timer.stop()
        self._pending_emphasis = None

    def _commit_emphasis(self) -> None:
        trace_id = self._pending_emphasis
        self._cancel_emphasis()
        if trace_id in self._curves and self._mode == "overlay":
            self.emphasis_requested.emit(trace_id)

    def _emphasis_click(self, trace_id: int, double: bool) -> None:
        if double:
            self._cancel_emphasis()
            return
        if self._mode != "overlay":
            return
        if self._pending_emphasis is not None and self._pending_emphasis != trace_id:
            self._commit_emphasis()
        self._pending_emphasis = trace_id
        self._emphasis_timer.start(QApplication.doubleClickInterval())

    def _new_plot(self, bottom: bool) -> pg.PlotItem:
        plot = pg.PlotItem(viewBox=ChromatogramViewBox(self),
                           axisItems={side: GridAxis(side) for side in ("left", "bottom", "right")})
        plot.layout.setContentsMargins(0, 0, 0, 0)
        plot.layout.setVerticalSpacing(0)
        plot.setMenuEnabled(False)
        plot.getViewBox().setMenuEnabled(False)
        plot.showGrid(x=True, y=True, alpha=0.16)
        plot.setLabel("left", "Counts")
        plot.showAxis("right", False)
        plot.showAxis("top", False)
        if bottom:
            plot.setLabel("bottom", "Retention time (min)")
        else:
            plot.setLabel("bottom", "")
        plot.getAxis("left").setStyle(tickTextOffset=4)
        # Equal view widths keep linked RT bounds identical across counts/mAU
        # panes when Y tick text changes width during autoscaling.
        plot.getAxis("left").setWidth(76)
        plot.getAxis("bottom").setStyle(tickTextOffset=1)
        plot.sigXRangeChanged.connect(self._x_changed)
        plot.enableAutoRange(enable=False)
        return plot

    def _remember(self, trace_id, title, rt, values, plot) -> None:
        self._fitted = None
        self._series[trace_id] = (np.asarray(rt, dtype=np.float64), np.asarray(values, dtype=np.float64))
        self._labels[trace_id] = title
        self._plot_for_trace[trace_id] = plot

    def _add_cursors(self, plot: pg.PlotItem) -> None:
        walk = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#b42318", width=1.6))
        hover = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#8a9099", width=1))
        walk.setZValue(12)
        hover.setZValue(11)
        walk.setVisible(False)
        hover.setVisible(False)
        plot.addItem(walk)
        plot.addItem(hover)
        self._walk_lines.append(walk)
        self._hover_lines.append(hover)

    def _plots(self) -> list[pg.PlotItem]:
        seen: list[pg.PlotItem] = []
        for plot in self._plot_for_trace.values():
            if plot not in seen:
                seen.append(plot)
        return seen

    def _apply_y_policy(self) -> None:
        for plot in self._plots():
            plot.getViewBox().setMouseEnabled(x=True, y=not self.autoscale_y)
        if self._right_view is not None:
            self._right_view.setMouseEnabled(x=True, y=not self.autoscale_y)
        self._apply_y_links()
        if self.autoscale_y:
            self._fit_all_y()

    def _apply_y_links(self) -> None:
        plots = self._plots()
        for plot in plots:
            plot.setYLink(None)
        if self.autoscale_y or not self.match_y or len(plots) < 2:
            return
        for unit in {self._units.get(tid, 'counts') for tid in self._plot_for_trace}:
            group = [plot for tid, plot in self._plot_for_trace.items() if self._units.get(tid, 'counts') == unit]
            master = self._plot_for_trace.get(self._active_id)
            if master not in group:
                master = group[0] if group else None
            for plot in group:
                if plot is not master:
                    plot.setYLink(master)

    def _as_plot_item(self, plot) -> pg.PlotItem | None:
        """Range signals emit the ViewBox, while stored traces are keyed by PlotItem."""
        if plot is self._right_view:
            return self._overlay_plot
        if plot in self._plot_for_trace.values():
            return plot
        for item in self._plots():
            if item.getViewBox() is plot:
                return item
        return None

    def _x_changed(self, plot, _range) -> None:
        if self._guard:
            return
        item = self._as_plot_item(plot)
        if item is None:
            return
        self._guard = True
        try:
            # Qt already propagates linked X changes. Writing X again here
            # feeds geometry-adjusted ranges back through the right ViewBox
            # while the scene is painting. Fit all Y using the received window.
            if self.autoscale_y:
                # Every linked stacked pane reports the received window in turn.
                # Refitting unchanged inputs would set the same Y ranges again.
                if self.relative_overlay or not self._same_fit(_range):
                    self._fit_all_y(_range)
                    self._fitted = None if self.relative_overlay else self._fit_inputs(_range)
            elif self.relative_overlay:
                self._update_relative_curves(_range)
        finally:
            self._guard = False
        self._settle_transforms()

    def _fit_inputs(self, window):
        """What a non-relative fit reads (objects by identity) and the Y ranges it left."""
        views = dict.fromkeys(self._trace_view(tid) for tid in self._plot_for_trace)
        key = (tuple(map(float, window)), self._mode, self.match_y, tuple(self._units.items()),
               tuple(self._plot_for_trace), tuple(tuple(map(float, view.viewRange()[1])) for view in views))
        objects = [obj for tid, plot in self._plot_for_trace.items() for obj in (plot, self._series[tid])]
        return key, objects + list(views)

    def _same_fit(self, window) -> bool:
        if self._fitted is None:
            return False
        key, objects = self._fit_inputs(window)
        return (key == self._fitted[0] and len(objects) == len(self._fitted[1])
                and all(a is b for a, b in zip(objects, self._fitted[1])))

    def _update_relative_curves(self, window=None):
        if not self.relative_overlay or not self._overlay_plot:
            return
        x0, x1 = window if window is not None else self._overlay_plot.viewRange()[0]
        for tid, (x, y) in self._series.items():
            reference = peak_reference(visible_signal(x, y, x0, x1))
            if self._references.get(tid) == reference:
                continue
            self._references[tid] = reference
            if tid in self._curves:
                self._curves[tid].setData(x, percent(y, reference),
                                          fillLevel=float(percent(fill_floor(x, y), reference)))

    def _fit_all_y(self, window=None) -> None:
        already = self._guard
        self._guard = True
        try:
            self._update_relative_curves(window)
            if self.match_y and len(self._plots()) > 1:
                self._fit_matched_y(window)
            else:
                for plot in self._plots():
                    self._fit_plot(plot, window)
        finally:
            self._guard = already
        if not already:
            self._settle_transforms()

    def _fit_plot(self, plot: pg.PlotItem, window=None) -> None:
        x0, x1 = window if window is not None else plot.viewRange()[0]
        pieces = {}
        for trace_id, owner in self._plot_for_trace.items():
            if owner is not plot:
                continue
            rt, values = self._series[trace_id]
            visible = visible_signal(rt, values, float(x0), float(x1))
            pieces.setdefault(self._trace_view(trace_id), []).append(self._display_y(trace_id, visible))
        for view, parts in pieces.items():
            values = np.concatenate(parts)
            if self.relative_overlay:
                values = np.r_[0., values] if values.size and np.any(values) else np.array([0., 100.])
            view.setYRange(*padded_y(values), padding=0)

    def _fit_matched_y(self, window=None) -> None:
        plots = self._plots()
        if not plots:
            return
        x0, x1 = window if window is not None else plots[0].viewRange()[0]
        for unit in {self._units.get(tid, 'counts') for tid in self._series}:
            ids = [tid for tid in self._series if self._units.get(tid, 'counts') == unit]
            pieces = [visible_signal(*self._series[tid], float(x0), float(x1)) for tid in ids]
            if pieces:
                bounds = padded_y(np.concatenate(pieces))
                for tid in ids:
                    self._trace_view(tid).setYRange(*bounds, padding=0)

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit(event.scenePos())
        if hit is None:
            return
        trace_id, x_value = hit
        if event.double():
            self._cancel_emphasis()
        elif self._mode == "overlay":
            stroke_id = self._stroke_hit(event.scenePos())
            if stroke_id is not None:
                self._emphasis_click(stroke_id, False)
        self.trace_activated.emit(trace_id)
        if self.inspection_enabled and (event.double() or self.walk_clicks):
            self.scan_requested.emit(trace_id, x_value)
        if self.inspection_enabled and event.double():
            self.spectrum_added.emit(trace_id, x_value)
        event.accept()

    def _stroke_hit(self, scene_pos) -> int | None:
        """Strict Overlay emphasis hit; scan inspection keeps its existing _hit."""
        plot = self._overlay_plot
        if plot is None:
            return None
        # A zero baseline can coincide with the axes; exclude the frame itself.
        interior = plot.getViewBox().sceneBoundingRect().adjusted(1, 1, -1, -1)
        if not interior.contains(scene_pos):
            return None
        # Legend text/blank space must never select a stroke drawn behind it.
        if plot.legend.isVisible() and plot.legend.sceneBoundingRect().contains(scene_pos):
            return None
        # Where strokes coincide, use the topmost curve in drawing order.
        for trace_id, curve in reversed(self._curves.items()):
            if curve.isVisible() and curve.curve.mouseShape().contains(curve.curve.mapFromScene(scene_pos)):
                return trace_id
        return None

    def _moved(self, scene_pos) -> None:
        for line in self._hover_lines:
            line.setVisible(False)
        hit = self._hit(scene_pos)
        if hit is None:
            return
        trace_id, x_value = hit
        rt, values = self._series[trace_id]
        if rt.size == 0:
            return
        index = int(np.argmin(np.abs(rt - x_value)))
        plot = self._plot_for_trace[trace_id]
        # Hover lines follow plot order, which matches _plots().
        plots = self._plots()
        if plot in plots:
            line = self._hover_lines[plots.index(plot)]
            line.setPos(float(rt[index]))
            line.setVisible(True)
        value_text = (f"{values[index]:.6g} {self._units.get(trace_id, 'counts')}" if np.isfinite(values[index])
                      else "No acquired value")
        self.readout.emit(f"RT {rt[index]:.4f} min    {value_text}    {self._labels[trace_id]}")

    def leaveEvent(self, event) -> None:
        for line in self._hover_lines:
            line.setVisible(False)
        super().leaveEvent(event)

    def _hit(self, scene_pos) -> tuple[int, float] | None:
        if self._mode == "overlay":
            plot = self._overlay_plot
            if plot is None or not plot.getViewBox().sceneBoundingRect().contains(scene_pos):
                return None
            point = plot.getViewBox().mapSceneToView(scene_pos)
            return self._nearest_overlay_trace(scene_pos), float(point.x())
        for trace_id, plot in self._plot_for_trace.items():
            if plot.getViewBox().sceneBoundingRect().contains(scene_pos):
                point = plot.getViewBox().mapSceneToView(scene_pos)
                return trace_id, float(point.x())
        return None

    def _nearest_overlay_trace(self, scene_pos) -> int:
        plot = self._overlay_plot
        assert plot is not None
        best_id = next(iter(self._series))
        best_distance = float("inf")
        for trace_id, (rt, values) in self._series.items():
            view = self._trace_view(trace_id)
            if rt.size == 0:
                continue
            x_value = float(view.mapSceneToView(scene_pos).x())
            index = int(np.argmin(np.abs(rt - x_value)))
            mapped = view.mapViewToScene(pg.QtCore.QPointF(float(rt[index]), float(self._display_y(trace_id, values[index]))))
            distance = (mapped - scene_pos).manhattanLength()
            if distance < best_distance:
                best_distance = distance
                best_id = trace_id
        return best_id
