"""Sparse display geometry stays separate from measurements and the omitted grid."""
import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from kmzviewer.comparison import SpectrumFrame
from kmzviewer.gui.ms1_spectra import SpectrumPanel
from kmzviewer.profile_rendering import zero_baseline_curve


@pytest.mark.parametrize('values, connections, expected_x, expected_y', [
    ([], [], [], []),
    ([9], [], [0, 0, 0], [0, 9, 0]),
    ([0], [], [0], [0]),
    ([5, 9, 4, 6], [True, False, True], [0, 0, 1, 1, 2, 2, 3, 3], [0, 5, 9, 0, 0, 4, 6, 0]),
    ([0, 9, 0, 0, 6, 0], [True, True, False, True, True], list(range(6)), [0, 9, 0, 0, 6, 0]),
    ([0, 9, 4, 0], [False, False, False], [0, 1, 1, 1, 2, 2, 2, 3], [0, 0, 9, 0, 0, 4, 0, 0]),
    ([5, 9, 4], [True, True], [0, 0, 1, 2, 2], [0, 5, 9, 4, 0]),
])
def test_zero_closure_and_boundary_cases(values, connections, expected_x, expected_y):
    mz = 100. + np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.asarray(connections, dtype=bool)
    for array in (mz, y, mask):
        array.setflags(write=False)
    x, drawn = zero_baseline_curve(mz, y, mask)
    np.testing.assert_array_equal(x, 100. + np.asarray(expected_x))
    np.testing.assert_array_equal(drawn, expected_y)
    np.testing.assert_array_equal(y, values)
    assert len(x) <= len(mz) + 2 * np.count_nonzero(~mask) + 2


def test_ordinary_arrays_and_existing_baseline_pass_through_without_copies():
    mz, y = np.array([100., 101., 105., 106.]), np.array([0., 5., 4., 0.])
    for connections in (None, np.ones(3, bool)):
        x, drawn = zero_baseline_curve(mz, y, connections)
        assert x is mz and drawn is y
    ordinary = SpectrumFrame(0, 'ordinary.sp.mzML', '#0077bb', '', mz, y)
    assert ordinary.profile_curve[0] is mz  # The suffix does not enable closure.


def test_gap_span_does_not_control_vertex_count_and_native_segments_are_exact():
    rng = np.random.default_rng(71)
    for count in (2, 15, 4000):
        mz = np.cumsum(rng.uniform(.001, .003, count)) + 100.
        y = rng.uniform(1., 1000., count)
        connections = rng.random(count - 1) > .2
        mz[1:] += np.cumsum(np.where(connections, 0., 1e6))
        x, drawn = zero_baseline_curve(mz, y, connections)
        positions = np.searchsorted(x, mz, side='left')
        positions += drawn[positions] == 0
        np.testing.assert_array_equal(x[positions], mz)
        np.testing.assert_array_equal(drawn[positions], y)
        np.testing.assert_array_equal(np.diff(positions)[connections], 1)
        for left in np.flatnonzero(~connections):
            np.testing.assert_array_equal(drawn[positions[left] + 1:positions[left + 1]], [0., 0.])
        assert len(x) == count + 2 * (np.count_nonzero(~connections) + 1)


def test_panel_continuity_reuse_native_readout_and_mixed_modes():
    app = QApplication.instance() or QApplication([])
    mz, y = np.array([100., 101., 110., 111.]), np.array([5., 200., 150., 4.])
    sparse = SpectrumFrame(1, 'Sparse', '#0077bb', '', mz, y,
                           connections=np.array([True, False, True]))
    ordinary = SpectrumFrame(2, 'Ordinary', '#bb7700', '', mz, y)
    centroid = SpectrumFrame(3, 'Centroid', '#77bb00', '', mz, y, centroid=True)
    panel = SpectrumPanel()
    panel.resize(950, 800)
    panel.show()
    try:
        panel.show_frames([sparse, ordinary, centroid], 'Synthetic comparison')
        original_curve = sparse.profile_curve
        for mode in ('stacked', 'overlay', 'stacked'):
            panel.match_y.setChecked(mode == 'overlay')  # Verify native geometry in both layouts.
            panel.set_mode(mode)
            panel.update_colors({1: '#0077bb'})
            panel.plot.setXRange(100.5, 110.5, padding=0)
            app.processEvents()
            curve = panel._curves[1]
            assert curve.opts['connect'] == 'all'
            path = curve.curve.getPath()
            assert path.elementCount() == len(curve.xData)
            assert sum(path.elementAt(i).isMoveTo() for i in range(path.elementCount())) == 1
            assert sparse.profile_curve is original_curve
            np.testing.assert_array_equal(panel._curves[2].yData, y)
            assert panel._curves[2].opts['connect'] == 'auto'
            assert panel._curves[3].opts['connect'] == 'pairs'
            np.testing.assert_array_equal(sparse.mz, mz)
            np.testing.assert_array_equal(sparse.intensity, y)
        # Hovering a synthetic baseline corner still reports the native point.
        point = panel._plots_by_file[1].getViewBox().mapViewToScene(QPointF(110., 0.))
        panel._moved(point)
        assert '150 counts' in panel.position.text()
        panel.reset_range()
        assert sparse.profile_curve is original_curve
    finally:
        panel.close()
        app.processEvents()
