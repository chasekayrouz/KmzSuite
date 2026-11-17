"""Stack membership changes retain panes; the result equals a complete rebuild."""
import gc

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from kmzviewer.gui.plots import ChromatogramPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


X = np.linspace(0., 10., 400)
ROWS = {tid: (tid, f"Trace {tid}", ("#0072b2", "#d55e00", "#009e73")[tid % 3], X,
              np.abs(np.sin(X * (tid + 1))) * 10. ** (tid % 4) - (tid == 3) * 5.)
        for tid in range(1, 8)}
ROWS[6] = (6, "Repeated RT", "#cc3377", np.array([0., 1., 1., 3., 4.]), np.array([0., 9., 7., 2., 0.]))
ROWS[7] = (7, "Empty", "#009e73", np.array([]), np.array([]))
UNITS = {tid: ("mAU" if tid == 2 else "counts") for tid in ROWS}


def hexes(values):
    return [float(v).hex() for v in values]


def state(panel):
    plots = panel._plots()
    out = []
    for tid, plot in panel._plot_for_trace.items():
        view = plot.getViewBox()
        axes = {side: (plot.getAxis(side).labelText, plot.getAxis(side).isVisible(),
                       plot.getAxis(side).label.isVisible(), plot.getAxis(side).autoSIPrefix)
                for side in ("left", "bottom", "right", "top")}
        curve = panel._curves.get(tid)
        data = None if curve is None else [hexes(a) for a in (*curve.getData(), *curve.curve.getData())]
        label = panel._inplot_labels[tid]
        out.append((tid, plots.index(plot), [hexes(r) for r in view.viewRange()], axes,
                    view.linkedView(0) is (None if plot is plots[0] else plots[0].getViewBox()),
                    view.linkedView(1) and plots.index(next(p for p in plots if p.getViewBox() is view.linkedView(1))),
                    view.state["mouseEnabled"], view.state["mouseMode"], view.border,
                    label.textItem.toPlainText(), label.isVisible(), data,
                    None if curve is None else (curve.opts["clipToView"], curve.opts["pen"], curve.opts["fillLevel"])))
    lines = [(line.value(), line.isVisible(), line.pos()) for line in panel._walk_lines + panel._hover_lines]
    return out, lines, [p.geometry() for p in plots], len(panel.view.ci.items)


def drive(panel, autoscale, match, first, second, qapp):
    panel.set_traces([ROWS[t] for t in first], units=UNITS)
    panel.set_x_range(1.5, 8.25)
    panel.set_match_y(match)
    panel.set_autoscale_y(autoscale)
    panel.set_active(first[1])
    panel.set_walk_rt(4.0)
    if not autoscale:
        panel._trace_view(first[0]).setYRange(-3, 7, padding=0)
    panel._hover_lines[0].setPos(2.5)
    panel._hover_lines[0].setVisible(True)
    qapp.processEvents()
    panel.previous_panes = dict(panel._plot_for_trace)
    panel.set_traces([ROWS[t] for t in second], units=UNITS)
    panel.set_walk_rt(None if match else 5.5)
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
    for _ in range(3):
        qapp.processEvents()
        panel.grab()
    return state(panel)


@pytest.mark.parametrize("first, second", [((1, 2, 3, 4), (2, 5, 3, 1, 6, 7)), ((1, 3, 6), (3,)),
                                           ((4, 5, 6, 7), (1, 4, 2, 5, 3, 6, 7))])
@pytest.mark.parametrize("autoscale, match", [(True, False), (False, False), (False, True)])
def test_restack_equals_rebuild(qapp, first, second, autoscale, match):
    panels = []
    for retain in (True, False):
        panel = ChromatogramPanel()
        panel.resize(900, 1100)
        panel.show()
        if not retain:
            panel._restack = lambda panel=panel: panel._rebuild(keep_x=False)
        panels.append(panel)
    try:
        retained = drive(panels[0], autoscale, match, first, second, qapp)
        before = panels[0].previous_panes
        rebuilt = drive(panels[1], autoscale, match, first, second, qapp)
        assert retained == rebuilt
        # Manual Y rebuilds (its new panes fit a fresh pane's default X window).
        assert all((panels[0]._plot_for_trace[t] is before[t]) == autoscale for t in set(first) & set(second))
        assert len(panels[0].view.scene().items()) == len(panels[1].view.scene().items())
    finally:
        for panel in panels:
            panel.close()
            panel.deleteLater()
        qapp.processEvents()


def test_retired_panes_leave_scene_and_survivors_keep_paths(qapp):
    panel = ChromatogramPanel()
    panel.resize(900, 900)
    panel.show()
    try:
        panel.set_traces([ROWS[t] for t in (1, 2, 3)], units=UNITS)
        panel.grab()
        retired, kept = panel._plot_for_trace[1], panel._plot_for_trace[3]
        path = panel._curves[3].curve.getPath()
        panel.set_traces([ROWS[t] for t in (3, 4)], units=UNITS)
        assert retired.scene() is None and kept.scene() is panel.view.scene()
        # Unchanged pens/brushes no longer reset data and every cached path.
        assert panel._curves[3].curve.path is path
        assert panel.view.ci.getItem(0, 0) is kept
        assert panel._plots()[1].getAxis("bottom").labelText == "Retention time (min)"
        assert kept.getAxis("bottom").labelText == ""
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_emphasis_updates_only_changed_pens(qapp):
    panel = ChromatogramPanel(normalize_overlay=True)
    panel.resize(900, 600)
    panel.show()
    try:
        panel.set_traces([ROWS[1], ROWS[3]], units=UNITS)
        panel.set_mode("overlay")
        panel.grab()
        curves = {tid: item.curve for tid, item in panel._curves.items()}
        paths = {tid: curve.getPath() for tid, curve in curves.items()}
        panel.set_emphasized(set())
        panel.update_colors({1: ROWS[1][2], 3: ROWS[3][2]})
        assert all(curve.path is paths[tid] for tid, curve in curves.items())
        panel.set_emphasized({1})
        assert curves[1].path is None and curves[3].path is paths[3]
        assert panel._curves[1].opts["pen"].widthF() == pytest.approx(3.4)
        panel.update_colors({3: "#cc3377"})
        assert panel._curves[3].opts["pen"].color().name() == "#cc3377"
        assert panel._curves[3].opts["fillBrush"].color().name() == "#cc3377"
        assert pg.mkPen(panel._curves[1].opts["pen"]) == panel._curves[1].opts["pen"]
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()
