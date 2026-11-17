"""Independent numeric contracts for visible-window percent display."""
import numpy as np
import pytest

from kmzviewer.display_scale import peak_reference, percent, visible_signal


def test_window_edges_use_rendered_line_but_centroid_does_not_interpolate():
    x = np.array([0., 1., 2., 3.])
    y = np.array([0., 10., 2., 1000.])
    visible = visible_signal(x, y, 1.25, 1.75)
    assert sorted(visible) == pytest.approx([4., 8.])
    assert percent(visible, peak_reference(visible)).max() == 100.
    assert visible_signal(x, y, 1.25, 1.75, sticks=True).size == 0
    np.testing.assert_array_equal(y, [0., 10., 2., 1000.])


def test_window_handles_nan_gaps_repeated_and_reversed_coordinates():
    assert visible_signal(np.array([0., 1., 2.]), np.array([1., np.nan, 3.]), .5, 1.5).size == 0
    assert visible_signal(np.array([0., np.nan, 2.]), np.array([1., 2., 3.]), .5, 1.5).size == 0
    assert sorted(visible_signal(np.array([2., 0.]), np.array([10., 2.]), .5, 1.5)) == [4., 8.]
    assert sorted(visible_signal(np.array([1., 1., 2.]), np.array([2., 4., 8.]), 1., 1.)) == [2., 4.]


@pytest.mark.parametrize("values, expected", [
    ([0., 2., 10.], [0., 20., 100.]),
    ([-20., 5., 10.], [-200., 50., 100.]),
    ([-10., -2., 0.], [-100., -20., 0.]),
    ([0., 0.], [0., 0.]),
    ([], []),
    ([1e-300, 2e-300], [50., 100.]),
])
def test_percent_retains_sign_zeros_and_small_positive_values(values, expected):
    native = np.array(values)
    snapshot = native.copy()
    np.testing.assert_allclose(percent(native, peak_reference(native)), expected)
    np.testing.assert_array_equal(native, snapshot)


def test_curve_clipping_preserves_repeated_rt_boundary_peaks():
    from PySide6.QtWidgets import QApplication
    from kmzviewer.gui.plots import ChromatogramPanel
    app = QApplication.instance() or QApplication([])
    panel = ChromatogramPanel(normalize_overlay=True)
    x, y = np.array([0., 1., 1., 1., 2.]), np.array([0., 0., 100., 0., 0.])
    try:
        panel.show()
        panel.set_traces([(1, 'Repeated times', '#0072b2', x, y),
                          (2, 'Increasing times', '#d55e00', np.arange(5.), y)])
        panel.set_x_range(0., 1.)
        app.processEvents()
        assert not panel._curves[1].opts['clipToView']
        assert panel._curves[2].opts['clipToView']
        assert panel._curves[1].curve.yData.max() == 100.
        np.testing.assert_array_equal(panel._curves[1].yData, y)
    finally:
        panel.close()
        panel.deleteLater()


def test_mixed_overlay_rebuild_keeps_range_callbacks_out_of_partial_scene():
    from PySide6.QtWidgets import QApplication
    from kmzviewer.gui.plots import ChromatogramPanel
    app = QApplication.instance() or QApplication([])
    panel = ChromatogramPanel(normalize_overlay=True)
    x = np.arange(5.)
    try:
        panel.show()
        panel.set_traces([(1, 'TIC', '#0072b2', x, x*100), (2, 'EWC', '#d55e00', x, x-2)],
                         units={1:'counts', 2:'mAU'})
        for _ in range(12):
            panel.set_mode('overlay')
            panel.set_match_y(True)
            panel._right_view.setXRange(.5, 2.5, padding=0)
            app.processEvents()
            assert panel.current_x_range() == pytest.approx((.5, 2.5))
            panel.set_match_y(False)
            app.processEvents()
            panel.set_mode('stacked')
            app.processEvents()
        assert not panel._guard
    finally:
        panel.close()
        panel.deleteLater()


def test_hover_cursor_follows_trace_when_grouping_reorders_existing_panes():
    from PySide6.QtWidgets import QApplication
    from kmzviewer.gui.plots import ChromatogramPanel
    app = QApplication.instance() or QApplication([])
    panel = ChromatogramPanel()
    x = np.arange(5.)
    rows = [(i, f'Trace {i}', '#0072b2', x, x*i) for i in range(1, 5)]
    try:
        panel.resize(800, 800)
        panel.show()
        panel.set_traces(rows)
        app.processEvents()
        cursor = panel._hover_lines[2]
        panel.set_traces([rows[i] for i in (0, 2, 1, 3)])
        app.processEvents()
        panel._moved(panel._plot_for_trace[3].getViewBox().sceneBoundingRect().center())
        assert [line for line in panel._hover_lines if line.isVisible()] == [cursor]
    finally:
        panel.close()
        panel.deleteLater()


def test_mixed_overlay_retirement_survives_collection_paint_and_native_hit_tests():
    import gc
    from PySide6.QtCore import QEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from kmzviewer.gui.plots import ChromatogramPanel
    app = QApplication.instance() or QApplication([])
    panel = ChromatogramPanel(normalize_overlay=True)
    x = np.arange(5.)
    try:
        panel.resize(800, 500)
        panel.show()
        panel.set_traces([(1, 'TIC', '#0072b2', x, x*100), (2, 'EWC', '#d55e00', x, x-2)],
                         units={1: 'counts', 2: 'mAU'})
        for _ in range(12):
            for mode, match in [('overlay', True), ('stacked', True), ('overlay', False)]:
                panel.set_mode(mode)
                panel.set_match_y(match)
                panel.set_x_range(.5, 2.5)
                app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                gc.collect()
                app.processEvents()
                panel.view.viewport().repaint()
                point = panel._plots()[0].getViewBox().sceneBoundingRect().center()
                assert panel.view.scene().items(point)
                QTest.mouseMove(panel.view.viewport(), panel.view.mapFromScene(point))
                assert panel.current_x_range() == pytest.approx((.5, 2.5))
                assert not panel.view.viewport().grab().isNull()
        np.testing.assert_array_equal(panel._series[2][1], x-2)
    finally:
        panel.close()
        panel.deleteLater()


def test_manual_y_overlay_transition_fits_visible_percent_not_offscreen_peaks():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from kmzviewer.gui.plots import ChromatogramPanel
    app = QApplication.instance() or QApplication([])
    panel = ChromatogramPanel(normalize_overlay=True)
    x, y = np.arange(5.), np.array([1000., 10., 20., 30., 1000.])
    try:
        panel.show()
        panel.set_traces([(1, 'Local peaks', '#0072b2', x, y)])
        panel.set_x_range(1., 3.)
        panel.set_autoscale_y(False)
        panel.set_mode('overlay')
        QTest.qWait(25)
        assert panel.current_x_range() == pytest.approx((1., 3.))
        assert panel._plots()[0].viewRange()[1] == pytest.approx((-5., 105.))
        assert not any(panel._plots()[0].getViewBox().autoRangeEnabled())
        panel._plots()[0].setYRange(-10, 120, padding=0)
        panel.set_x_range(.5, 2.)
        app.processEvents()
        assert panel._plots()[0].viewRange()[1] == pytest.approx((-10., 120.))
    finally:
        panel.close()
        panel.deleteLater()
