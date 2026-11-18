"""Session styling and real Qt plot-event/label geometry regressions."""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, QTimer, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QColorDialog

from kmzcore.labels import label_aware_y
from kmzcore.theme import apply_theme, INK, PAPER
from kmzcore.trace_style import chromatogram_palette, fill_rgba, line_style, LINE_WIDTH
from kmzviewer.gui.app import MainWindow
from kmzviewer.gui.workers import plan_jobs
from kmzviewer.gui.plots import ChromatogramPanel, SpectrumPanel, TraceSample
from kmz2viewer.gui.spectrum import SpectrumView
from kmz2viewer.model import Ms2Scan
from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def test_palette_and_derived_styles():
    colors = chromatogram_palette()
    assert len(colors) == len(set(colors)) == 24
    for color in colors:
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb]
        luminance = sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))
        assert 1.05 / (luminance + .05) >= 3
        assert fill_rgba(color)[:3] == tuple(round(c * 255) for c in rgb)
        assert fill_rgba(color)[3] < 20
        bold, width = line_style(color, True)
        assert width == LINE_WIDTH * 2
        assert all(a < b or b == 0 for a, b in zip(fill_rgba(bold)[:3], fill_rgba(color)[:3]))


def test_incremental_palette_matches_full_maximin_search():
    from colorsys import hsv_to_rgb
    from kmzcore import trace_style
    colors = list(chromatogram_palette()[:4])
    points = {}
    for hue in range(0, 360, 5):
        for saturation in (.60, .80, 1.0):
            for value in (.65, .82, 1.0):
                color = '#' + ''.join(f'{round(c * 255):02x}' for c in hsv_to_rgb(hue / 360, saturation, value))
                luminance = sum(w * trace_style._linear(c) for w, c in zip((.2126, .7152, .0722), trace_style._rgb(color)))
                if 1.05 / (luminance + .05) >= 3 and color not in colors:
                    points[color] = None
    points = {color: trace_style._lab(color) for color in sorted(points)}
    while len(colors) < 24:
        selected = [trace_style._lab(color) for color in colors]
        chosen = max(points, key=lambda color: min(sum((a - b) ** 2 for a, b in zip(points[color], other))
                                                   for other in selected))
        colors.append(chosen); del points[chosen]
    assert chromatogram_palette() == tuple(colors)


def test_measured_headroom_shrinks_with_height():
    small = label_aware_y(0, 105, [(100, 100)], 100, 24)
    large = label_aware_y(0, 105, [(100, 100)], 600, 24)
    assert (small[1] - 100) / small[1] * 100 == pytest.approx(24)
    assert large == (0, 105)
    assert label_aware_y(-1, 1, [], 100, 24) == (-1, 1)


@pytest.fixture
def viewer(qapp, tmp_path):
    paths = [tmp_path / 'Source A.mzML', tmp_path / 'Source B.mzML']
    for index, path in enumerate(paths):
        write_mzml(path, [
            {'mz': [100., 200.], 'intensity': [10 + index, 30.], 'rt': rt, 'tic': height * (1 + .35 * index),
             'centroid': True, 'polarity': 1}
            for rt, height in [(1., 10.), (2., 100.), (3., 25.), (4., 60.), (5., 10.)]
        ])
    window = MainWindow()
    window.resize(1100, 780)
    window.show()
    window.resizeDocks([window.spectrum_dock], [230], Qt.Orientation.Vertical)
    qapp.processEvents()
    assert window.load_paths_sync([str(path) for path in paths]) == []
    window.match_action.setChecked(True)  # These gesture fixtures use native counts coordinates.
    QTest.qWait(30)
    for _ in range(3):
        qapp.processEvents()
    yield window, paths
    _close(window, qapp)


def _close(window, qapp):
    window.close()
    deadline = time.monotonic() + 10
    while not window.runtime.closed and time.monotonic() < deadline:
        QTest.qWait(10)
    assert window.runtime.closed
    qapp.processEvents()


def test_source_colors_manual_dialog_late_results_restore_and_no_writes(viewer, qapp, monkeypatch):
    window, paths = viewer
    window.runtime.settings.setFallbacksEnabled(False)
    settings_before = window.runtime.settings.allKeys()
    original = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
    files = list(window.session.files.values())
    defaults = [loaded.color for loaded in files]
    assert defaults[0] != defaults[1]
    window.extract_sync('100, 200', 10, 'all', False, [files[0].file_id])
    window.plots.set_mode('overlay')
    assert len(files[0].traces) == 3
    assert {trace.color for trace in files[0].traces} == {defaults[0]}
    picked = []
    def select_color():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QColorDialog):
            dialog.setCurrentColor(QColor('#17a3bd'))
            picked.append(True)
            dialog.accept()
    QTimer.singleShot(0, select_color)
    window.choose_file_color(files[0].file_id)
    assert picked  # Exercise the real Qt picker in the offscreen platform.
    assert {trace.color for trace in files[0].traces} == {'#17a3bd'}
    for trace in files[0].traces:
        curve = window.plots._curves[trace.trace_id]
        assert curve.opts['pen'].color().name() == '#17a3bd'
        assert curve.opts['fillBrush'].color().getRgb() == fill_rgba('#17a3bd')
        assert window.plots._samples[trace.trace_id].item is curve
    assert files[1].color == defaults[1]
    monkeypatch.setattr(QColorDialog, 'getColor', lambda *a, **kw: QColor())
    window.choose_file_color(files[0].file_id)
    assert files[0].color == '#17a3bd'
    # Worker results with a stale snapshot must adopt the color at GUI delivery.
    jobs = plan_jobs([files[0]], '100', 10, 'all', False, window.session.new_trace_id)
    assert jobs[0]['colors'] == ['#17a3bd']
    window.set_file_color(files[0].file_id, '#e45b22')
    window.extract_sync('200', 10, 'all', False, [files[0].file_id])
    import dataclasses
    late = dataclasses.replace(files[0].traces[-1], trace_id=jobs[0]['trace_ids'][0], color='#17a3bd')
    window._on_extracted([(files[0].file_id, [late])])
    assert late.color == '#e45b22'
    tic_id = files[0].traces[0].trace_id
    window.remove_trace(tic_id)
    window.restore_tic(files[0].file_id)
    assert {trace.color for trace in files[0].traces} == {'#e45b22'}
    window.plots.set_mode('overlay')
    qapp.processEvents()
    for trace in files[0].traces:
        assert window.plots._samples[trace.trace_id].item.opts['pen'].color().name() == '#e45b22'
    assert [hashlib.sha256(path.read_bytes()).digest() for path in paths] == original
    assert {path.name for path in paths[0].parent.iterdir()} == {path.name for path in paths}
    assert window.runtime.settings.allKeys() == settings_before
    fresh = MainWindow()
    try:
        fresh.load_paths_sync([str(path) for path in paths])
        assert [loaded.color for loaded in fresh.session.files.values()] == defaults
        assert fresh.session.emphasized == set()
    finally:
        _close(fresh, qapp)
    code = '''
import json, sys, time
from PySide6.QtWidgets import QApplication
from kmzviewer.gui.app import MainWindow
app = QApplication([])
window = MainWindow()
assert window.load_paths_sync(sys.argv[1:]) == []
print(json.dumps([file.color for file in window.session.files.values()]))
assert not window.session.emphasized
window.close()
while not window.runtime.closed:
    app.processEvents()
    time.sleep(.01)
'''
    clean = subprocess.run([sys.executable, '-c', code, *map(str, paths)], capture_output=True,
                           text=True, check=True, timeout=20, env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})
    assert json.loads(clean.stdout) == defaults


def _double_click(panel, scene_pos):
    viewport = panel.view.viewport()
    pos = panel.view.mapFromScene(scene_pos)
    QTest.mouseMove(viewport, pos)
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=pos)
    QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=pos)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=pos)
    QApplication.processEvents()


def _settle_clicks(panel):
    """Outlast the double-click window and any pending single-click emphasis.

    The emphasis timer can fire well after the interval on a loaded runner, so a
    fixed wait alone could assert before (or miss) its commit.
    """
    QTest.qWait(QApplication.doubleClickInterval() + 20)
    deadline = time.monotonic() + 5
    while panel._emphasis_timer.isActive() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not panel._emphasis_timer.isActive()


def _single_click(panel, scene_pos):
    viewport = panel.view.viewport()
    pos = panel.view.mapFromScene(scene_pos)
    QTest.mouseMove(viewport, pos)
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=pos)
    _settle_clicks(panel)


def _drag(viewport, start, end):
    # Approach the start with an unthrottled hover (the legend may have moved).
    QTest.mouseMove(viewport, (start + end) / 2)
    QTest.qWait(25)
    QTest.mouseMove(viewport, start)
    QTest.qWait(25)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    # QTest.mouseMove does not carry a held-button mask on the offscreen plugin.
    for pos in ((start + end) / 2, end):
        QTest.qWait(25)  # Respect pyqtgraph's mouse-move rate limit.
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(viewport.mapToGlobal(pos)),
                            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(viewport, event)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    QApplication.processEvents()


def _visibility(window):
    return (
        [(trace.trace_id, trace.visible) for _, trace in window.session.ordered_traces()],
        [curve.isVisible() for curve in window.plots._curves.values()],
        [(window.tree.topLevelItem(i).checkState(0),
          [window.tree.topLevelItem(i).child(j).checkState(0)
           for j in range(window.tree.topLevelItem(i).childCount())])
         for i in range(window.tree.topLevelItemCount())],
    )


def _pan_and_zoom(panel, plot):
    panel.set_mouse_mode('pan')
    vb = plot.getViewBox()
    viewport = panel.view.viewport()
    start = panel.view.mapFromScene(vb.mapViewToScene(QPointF(3.5, 120.)))
    before_x = tuple(plot.viewRange()[0])
    end = panel.view.mapFromScene(vb.mapViewToScene(QPointF(4., 120.)))
    _drag(viewport, start, end)
    assert tuple(plot.viewRange()[0]) != before_x
    panel.set_mouse_mode('zoom')
    start = panel.view.mapFromScene(vb.mapViewToScene(QPointF(2.5, 40.)))
    end = panel.view.mapFromScene(vb.mapViewToScene(QPointF(3.5, 100.)))
    before_span = np.ptp(plot.viewRange()[0])
    _drag(viewport, start, end)
    assert np.ptp(plot.viewRange()[0]) < before_span


def test_overlay_line_legend_emphasis_and_visibility(viewer, qapp):
    window, _ = viewer
    panel = window.plots
    panel.set_mode('overlay')
    qapp.processEvents()
    ids = list(panel._curves)
    curve = panel._curves[ids[0]]
    plot = panel._plot_for_trace[ids[0]]
    fill = curve.opts['fillBrush'].color().getRgb()
    ranges = np.asarray(plot.viewRange()).copy()
    scans, activated = [], []
    panel.scan_requested.connect(lambda *args: scans.append(args))
    panel.trace_activated.connect(activated.append)
    visibility = _visibility(window)
    point = plot.getViewBox().mapViewToScene(QPointF(2.5, 62.5))
    _single_click(panel, point)
    assert window.session.emphasized == {ids[0]}
    assert curve.opts['pen'].widthF() == LINE_WIDTH * 2
    assert curve.opts['pen'].color().name() == line_style(panel._colors[ids[0]], True)[0]
    assert panel._samples[ids[0]].item.opts['pen'] == curve.opts['pen']
    assert curve.opts['fillBrush'].color().getRgb() == fill
    assert np.asarray(plot.viewRange()) == pytest.approx(ranges)
    assert activated and not scans
    _single_click(panel, point)
    assert window.session.emphasized == set()
    _single_click(panel, panel._samples[ids[0]].sceneBoundingRect().center())
    assert window.session.emphasized == {ids[0]}
    _single_click(panel, panel._samples[ids[0]].sceneBoundingRect().center())
    assert window.session.emphasized == set()
    assert curve.opts['pen'].widthF() == LINE_WIDTH
    assert _visibility(window) == visibility
    assert all(isinstance(sample, TraceSample) for sample in panel._samples.values())
    for index, trace_id in enumerate(ids):
        point_i = plot.getViewBox().mapViewToScene(QPointF(2.5, 62.5 * (1 + .35 * index)))
        _single_click(panel, point_i)
        assert trace_id in window.session.emphasized
    assert window.session.emphasized == set(ids)
    _single_click(panel, panel._samples[ids[0]].sceneBoundingRect().center())
    assert window.session.emphasized == {ids[1]}
    unchanged = set(window.session.emphasized)
    legend_label = panel._samples[ids[0]].parentItem().items[0][1]
    for scene_pos in (
        plot.getViewBox().mapViewToScene(QPointF(3.5, 80.)),
        plot.getAxis('bottom').mapRectToScene(plot.getAxis('bottom').rect()).center(),
        plot.getAxis('left').mapRectToScene(plot.getAxis('left').rect()).center(),
        legend_label.sceneBoundingRect().center(),
        panel._samples[ids[0]].mapToScene(QPointF(10, 2)),
        panel._samples[ids[0]].mapToScene(QPointF(22, 10)),
    ):
        _single_click(panel, scene_pos)
        assert window.session.emphasized == unchanged
    # A legend text click must not hit a curve underneath the legend.
    legend = plot.legend
    old_pos = legend.pos()
    legend.setPos(old_pos + point - legend_label.sceneBoundingRect().center())
    qapp.processEvents()
    _single_click(panel, legend_label.sceneBoundingRect().center())
    assert window.session.emphasized == unchanged
    legend.setPos(old_pos)
    qapp.processEvents()
    # Match the actual viewport pixel, especially at a nearest-RT midpoint.
    expected_scan = panel._hit(panel.view.mapToScene(panel.view.mapFromScene(point)))
    count = len(scans)
    _double_click(panel, point)
    _settle_clicks(panel)
    assert window.session.emphasized == {ids[1]}
    assert len(scans) > count and scans[-1][0] == expected_scan[0]
    assert scans[-1][1] == pytest.approx(expected_scan[1], abs=.02)
    _double_click(panel, panel._samples[ids[0]].sceneBoundingRect().center())
    _settle_clicks(panel)
    assert window.session.emphasized == unchanged
    assert _visibility(window) == visibility
    # Base-color edits keep emphasis and fill independent, and preserve manual Y.
    panel.set_autoscale_y(False)
    plot.setYRange(-20, 140, padding=0)
    manual = np.asarray(plot.viewRange()).copy()
    source = next(iter(window.session.files.values()))
    window.set_file_color(source.file_id, '#c7459e')
    assert np.asarray(plot.viewRange()) == pytest.approx(manual)
    assert curve.opts['fillBrush'].color().getRgb() == fill_rgba('#c7459e')
    assert curve.opts['pen'].color().name() == '#c7459e'
    assert panel._samples[ids[1]].item.opts['pen'].widthF() == LINE_WIDTH * 2
    # The existing double-click-to-scan gesture remains in empty plot space.
    _double_click(panel, plot.getViewBox().mapViewToScene(QPointF(3.5, 120.)))
    assert scans
    _pan_and_zoom(panel, plot)
    _settle_clicks(panel)
    assert window.session.emphasized == unchanged
    assert _visibility(window) == visibility
    # Stack mode renders normally while keeping the Overlay session state.
    panel.set_mode('stacked')
    assert not panel._samples
    assert window.session.emphasized == unchanged
    assert all(item.opts['pen'].widthF() == LINE_WIDTH for item in panel._curves.values())
    panel.set_mode('overlay')
    assert panel._curves[ids[1]].opts['pen'].widthF() == LINE_WIDTH * 2
    assert panel._curves[ids[0]].opts['pen'].widthF() == LINE_WIDTH


def test_vertical_stack_activation_walk_hover_pan_zoom_without_emphasis(viewer, qapp):
    window, _ = viewer
    panel = window.plots
    ids = list(panel._curves)
    window._toggle_emphasis(ids[0])  # Overlay state may survive switching modes.
    assert not panel._samples
    assert set(panel._inplot_labels) == set(ids)
    assert all(plot.legend is None for plot in panel._plots())
    assert all(curve.opts['pen'].widthF() == LINE_WIDTH for curve in panel._curves.values())
    plot = panel._plot_for_trace[ids[0]]
    point = plot.getViewBox().mapViewToScene(QPointF(2.5, 62.5))
    scans, activated, readouts = [], [], []
    panel.scan_requested.connect(lambda *args: scans.append(args))
    panel.trace_activated.connect(activated.append)
    panel.readout.connect(readouts.append)
    _single_click(panel, point)
    assert window.session.emphasized == {ids[0]}
    assert activated[-1] == ids[0] and not scans
    _double_click(panel, point)
    assert scans[-1][0] == ids[0]
    panel.set_walk_clicks(True)
    count = len(scans)
    _single_click(panel, point)
    assert len(scans) > count and scans[-1][0] == ids[0]
    assert window.session.emphasized == {ids[0]}
    panel.set_walk_clicks(False)
    QTest.mouseMove(panel.view.viewport(), panel.view.mapFromScene(
        plot.getViewBox().mapViewToScene(QPointF(3.5, 70.))))
    QTest.qWait(30)
    assert readouts and 'RT ' in readouts[-1]
    assert any(line.isVisible() for line in panel._hover_lines)
    _single_click(panel, panel._inplot_labels[ids[0]].sceneBoundingRect().center())
    assert window.session.emphasized == {ids[0]}
    # Real mouse drags still pan and rectangle-zoom the existing ViewBox.
    panel.set_autoscale_y(False)
    plot.setYRange(-20, 140, padding=0)
    _pan_and_zoom(panel, plot)
    assert window.session.emphasized == {ids[0]}
    assert all(curve.opts['pen'].widthF() == LINE_WIDTH for curve in panel._curves.values())


def test_overlay_axis_frame_does_not_hit_zero_baseline(qapp):
    panel = ChromatogramPanel()
    panel.resize(800, 500)
    panel.show()
    try:
        panel.set_traces([(1, 'Synthetic zero baseline', '#0072b2',
                           np.array([0., 1., 2., 3., 4.]), np.array([0., 0., 0., 10., 0.]))])
        panel.set_mode('overlay')
        qapp.processEvents()
        plot = panel._plots()[0]
        plot.setYRange(0, 11, padding=0)
        qapp.processEvents()
        requests = []
        panel.emphasis_requested.connect(requests.append)
        frame = plot.getViewBox().mapViewToScene(QPointF(1., 0.))
        _single_click(panel, frame)
        assert requests == []
        # The zero trace is still reachable within the normal stroke tolerance.
        _single_click(panel, frame - QPointF(0, 3))
        assert requests == [1]
    finally:
        panel.close()
        qapp.processEvents()


@pytest.mark.parametrize('count', [7, 8, 10, 12])
def test_vertical_stack_interior_labels_keep_data_height_and_bounds(qapp, count):
    panel = ChromatogramPanel()
    rt = np.linspace(0, 10, 101)
    titles = [f'Source {i // 3}.mzML    EIC {100 + i} ±10 ppm' for i in range(count)]
    rows = [(i, titles[i], '#0072b2', rt, np.exp(-((rt - 5) / .7) ** 2) * (i + 1))
            for i in range(count)]
    panel.resize(1100, 780)
    panel.show()
    try:
        panel.set_traces(rows)
        for _ in range(4):
            qapp.processEvents()
        assert not panel._samples
        bounds = np.asarray([plot.viewRange() for plot in panel._plots()])
        for trace_id, label in panel._inplot_labels.items():
            plot = panel._plot_for_trace[trace_id]
            view = plot.getViewBox()
            assert label.parentItem() is view and label not in view.addedItems
            assert not plot.titleLabel.isVisible() and plot.legend is None
            assert view.geometry().top() <= 2  # No dedicated title/legend row.
            assert view.height() > 35
            assert label.isVisible() and label.toPlainText() == titles[trace_id]
            rect = label.sceneBoundingRect()
            assert rect.left() > view.sceneBoundingRect().left() + 3
            assert rect.top() > view.sceneBoundingRect().top() + 1
            assert rect.right() < view.sceneBoundingRect().right() - 3
            label.setVisible(False)
        panel.autoscale_visible_y()
        assert np.asarray([plot.viewRange() for plot in panel._plots()]) == pytest.approx(bounds)
        # Long titles elide in the middle, retaining the full title as a tooltip.
        long_title = 'A very long source filename ' * 12 + ' EIC 273.1559 ±10 ppm'
        panel.set_traces([(i, long_title, color, x, y) for i, _, color, x, y in rows])
        panel.resize(460, 420)
        for _ in range(4):
            qapp.processEvents()
        for label in panel._inplot_labels.values():
            assert label.toolTip() == long_title
            assert label.toPlainText() != long_title
            assert label.boundingRect().width() <= label.parentItem().width() - 9
        panel.resize(460, 260)
        for _ in range(4):
            qapp.processEvents()
        for label in panel._inplot_labels.values():
            if label.isVisible():
                assert label.sceneBoundingRect().bottom() < label.parentItem().sceneBoundingRect().bottom()
        assert not panel._samples
    finally:
        panel.close()
        qapp.processEvents()


def _spectrum(case):
    if case == 'single':
        return np.array([200.]), np.array([1e9])
    if case == 'dense':
        mz = np.linspace(100, 900, 4000)
        return mz, (np.sin(np.arange(4000) * .17) + 1) * 500
    cases = {
        'dominant': ([100., 200., 300., 400., 900.], [0., 1., 1e9, 3., 0.]),
        'left': ([100., 200., 500., 900.], [1e9, 2., 3., 0.]),
        'right': ([100., 200., 500., 900.], [0., 2., 3., 1e9]),
        'equal': ([100., 300., 500., 900.], [1e9, .999e9, .998e9, .997e9]),
        'sparse': ([100., 200., 900.], [5., 1e9, 8.]),
    }
    return tuple(np.array(values) for values in cases[case])


@pytest.mark.parametrize('viewer_type', ['ms1', 'ms2'])
@pytest.mark.parametrize('centroid', [True, False])
@pytest.mark.parametrize('case', ['single', 'dominant', 'left', 'right', 'equal', 'sparse', 'dense'])
def test_peak_labels_fit_after_resize_zoom_and_reset(qapp, viewer_type, centroid, case):
    mz, y = _spectrum(case)
    widget = SpectrumPanel() if viewer_type == 'ms1' else SpectrumView()
    widget.show()
    try:
        if viewer_type == 'ms1':
            widget.show_scan(mz, y, centroid, 'Synthetic scan')
        else:
            scan = Ms2Scan('synthetic.mzML', 'Synthetic', 0, 'scan', 1., None, None, None, None,
                           centroid, mz, y, float(mz[y.argmax()]), float(y.max()), float(y.sum()), len(mz))
            widget.show_scan(scan, None, keep_window=False)
        for size in [(1000, 560), (760, 300), (640, 230), (1000, 560)]:
            widget.resize(*size)
            for _ in range(3):
                qapp.processEvents()
            for zoomed in (False, True):
                if zoomed:
                    peak_mz = float(mz[y.argmax()])
                    span = max(float(np.ptp(mz)) * .2, 1.)
                    x0 = max(float(mz.min()), peak_mz - span / 2)
                    x1 = min(float(mz.max()), x0 + span) if len(mz) > 1 else x0 + span
                    if x1 <= peak_mz:
                        x1 = peak_mz + span / 2
                    widget.plot.setXRange(x0, x1, padding=0)
                else:
                    widget.reset_range()
                qapp.processEvents()
                if zoomed:
                    assert tuple(widget.plot.viewRange()[0]) == pytest.approx((x0, x1))
                assert widget._label_items
                view_rect = widget.plot.getViewBox().sceneBoundingRect()
                for label in widget._label_items:
                    rect = label.textItem.mapRectToScene(label.textItem.boundingRect())
                    assert rect.top() >= view_rect.top() + 1
                    assert rect.left() >= view_rect.left() + 1
                    assert rect.right() <= view_rect.right() - 1
                    assert rect.bottom() <= view_rect.bottom()
        assert np.array_equal(mz, widget._mz if viewer_type == 'ms1' else widget._scan.mz)
    finally:
        widget.close()
        qapp.processEvents()


def test_theme_active_inactive_disabled_palettes(qapp):
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        assert qapp.palette().color(group, QPalette.ColorRole.Text).name() == INK
        assert qapp.palette().color(group, QPalette.ColorRole.Base).name() == PAPER
    assert qapp.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text) != qapp.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base)
