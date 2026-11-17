"""Independent stress checks for signed, relative and linked viewer plots."""
import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kmzcore.ranges import padded_y
from kmzviewer.comparison import SpectrumFrame
from kmzviewer.display_scale import visible_signal
from kmzviewer.gui.ms1_spectra import SpectrumPanel
from kmzviewer.gui.plots import ChromatogramPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chrom(qapp):
    panel = ChromatogramPanel(normalize_overlay=True)
    panel.resize(950, 650)
    panel.show()
    yield panel
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def rows():
    x = np.array([0., 1., 2., 3., 4.])
    return [(1, "Ion", "#0072b2", x, np.array([1000., 10., 20., 30., 1000.])),
            (2, "Optical", "#d55e00", x, np.array([-50., -2., 3., 2., 50.]))]


def test_dad_zoom_from_last_stack_and_right_overlay_axis(chrom):
    panel = chrom
    data = rows()
    panel.set_traces(data, units={1: "counts", 2: "mAU"})
    QTest.qWait(20)
    panel._plots()[-1].setXRange(1., 3., padding=0)
    for tid, _title, _color, _x, y in data:
        assert panel._trace_view(tid).viewRange()[0] == pytest.approx([1., 3.])
        assert panel._trace_view(tid).viewRange()[1] == pytest.approx(padded_y(y[1:4]))
    panel.set_match_y(True)
    panel.set_mode("overlay")
    QTest.qWait(20)
    assert panel._right_view is not None
    panel._right_view.setXRange(1.25, 2.75, padding=0)
    assert panel.current_x_range() == pytest.approx([1.25, 2.75])
    assert panel._right_view.viewRange()[1] == pytest.approx((-0.9375, 3.1875))
    assert panel._trace_view(1).viewRange()[1] == pytest.approx((11.75, 28.25))


def test_signed_relative_chromatograms_preserve_native_hover_and_roundtrip(chrom):
    panel = chrom
    x = np.array([0., 1., 2., 3., 4.])
    native = [np.array([-20., -2., 10., 0., 1000.]), np.array([-50., -10., -4., -30., -100.])]
    original = [y.copy() for y in native]
    panel.set_traces([(i + 1, f"Optical {i + 1}", color, x, y)
                      for i, (color, y) in enumerate(zip(("#0072b2", "#d55e00"), native))],
                     units={1: "mAU", 2: "mAU"})
    panel.set_mode("overlay")
    panel.set_x_range(1., 3.)
    QTest.qWait(20)
    np.testing.assert_allclose(panel._curves[1].yData, native[0] * 10.)
    np.testing.assert_allclose(panel._curves[2].yData, native[1] / 30. * 100.)
    assert panel._trace_view(2).viewRange()[1][0] <= -100
    reads = []
    panel.readout.connect(reads.append)
    position = panel._trace_view(2).mapViewToScene(QPointF(3., -100.))
    panel._moved(position)
    assert reads and "-30 mAU" in reads[-1] and "Optical 2" in reads[-1]
    panel.set_match_y(True)
    for tid, y in enumerate(native, 1):
        np.testing.assert_array_equal(panel._curves[tid].yData, y)
    panel.set_match_y(False)
    panel.set_mode("stacked")
    for tid, (y, expected) in enumerate(zip(native, original), 1):
        np.testing.assert_array_equal(y, expected)
        np.testing.assert_array_equal(panel._curves[tid].yData, expected)
        assert panel._plot_for_trace[tid].getAxis("bottom").isVisible()


def test_enter_relative_overlay_with_manual_native_y_fits_new_units(chrom):
    panel = chrom
    panel.set_traces(rows(), units={1: "counts", 2: "mAU"})
    panel.set_x_range(1., 3.)
    panel.set_autoscale_y(False)
    panel._plots()[0].setYRange(0, 50, padding=0)
    panel.set_mode("overlay")
    # A fresh percent axis must not retain the new PlotItem's default [0, 1].
    assert panel._plots()[0].viewRange()[1][1] >= 100.
    assert panel.current_x_range() == pytest.approx([1., 3.])


def test_profile_boundaries_nan_gaps_and_centroid_windows_are_distinct():
    x = np.array([0., 1., 2., 3., 4.])
    y = np.array([0., 10., np.nan, -10., 0.])
    np.testing.assert_array_equal(visible_signal(x, y, .2, .4), [2., 4.])
    assert not visible_signal(x, y, 1.2, 2.8).size
    assert not visible_signal(x, y, .2, .4, sticks=True).size
    np.testing.assert_allclose(visible_signal(x[::-1], y[::-1], 3.2, 3.4), [-8., -6.])


@pytest.mark.parametrize("kind", ["profile", "centroid", "sparse", "dad"])
def test_spectrum_normalization_labels_native_hover_and_cache(qapp, monkeypatch, kind):
    panel = SpectrumPanel(optical=kind == "dad")
    panel.resize(950, 650)
    x = np.array([100., 110.123456789, 120., 130., 140.])
    a = np.array([0., 10., 0., 2., 1000.])
    b = np.array([0., 200., 0., 20., 2000.])
    frames = [SpectrumFrame(i + 1, f"Run {i + 1}", color, f"Run {i + 1}", x.copy(), y.copy(),
                            centroid=kind == "centroid", optical=kind == "dad",
                            connections=np.array([True, False, False, True]) if kind == "sparse" else None)
              for i, (color, y) in enumerate(zip(("#0072b2", "#d55e00"), (a, b)))]
    originals = [f.intensity.copy() for f in frames]
    try:
        panel.show_frames(frames, "Compare")
        panel.set_mode("overlay")
        panel.show()
        panel.plot.setXRange(100., 130., padding=0)
        QTest.qWait(30)
        assert panel.match_y.isEnabled()
        for frame in frames:
            curve = panel._curves[frame.file_id]
            assert np.max(visible_signal(curve.xData, curve.yData, 100., 130.)) == pytest.approx(100.)
        reads = []
        panel.readout.connect(reads.append)
        panel._moved(panel.plot.getViewBox().mapViewToScene(QPointF(x[1], 100.)))
        assert reads and ("200 mAU" if kind == "dad" else "200 counts") in reads[-1]
        assert panel._label_items
        if kind != "dad":
            item = next(item for item in panel._label_items if item.msms_target[1] == x[1])
            target = item.msms_target
            pos = item.textItem.mapToScene(item.textItem.boundingRect().center())
            menu = panel._label_context_menu(pos)
            assert menu is not None
            requests = []
            panel.msms_requested.connect(lambda fid, mass: requests.append((fid, mass)))
            menu.actions()[0].trigger()
            menu.deleteLater()
            assert requests == [target]
        calls = []
        for curve in panel._curves.values():
            method = curve.setData
            monkeypatch.setattr(curve, "setData", lambda *args, _method=method, **kwargs:
                                (calls.append(1), _method(*args, **kwargs))[1])
        for _ in range(5):
            panel._update_labels()
        assert not calls, "Unchanged spectra should not re-upload arrays during label-only work"
        panel.match_y.setChecked(True)
        for frame, original in zip(frames, originals):
            np.testing.assert_array_equal(frame.intensity, original)
            curve = panel._curves[frame.file_id]
            assert np.nanmax(curve.yData) == np.max(original)
        panel.match_y.setChecked(False)
        panel.plot.setXRange(110.5, 119.5, padding=0)
        if kind in {"centroid", "sparse"}:
            assert panel.plot.viewRange()[1] == pytest.approx([-5., 105.])
        else:
            assert 100. <= panel.plot.viewRange()[1][1] < 120.
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_zero_nan_empty_and_negative_spectra_do_not_invent_positive_peaks(qapp):
    panel = SpectrumPanel(optical=True)
    x = np.array([250., 260., 270., 280.])
    values = [np.zeros(4), np.full(4, np.nan), np.array([-2., -10., -4., -1.]), np.array([])]
    frames = [SpectrumFrame(i + 1, f"Edge {i + 1}", "#0072b2", "", x if y.size else np.array([]), y,
                            optical=True) for i, y in enumerate(values)]
    try:
        panel.resize(900, 650)
        panel.show_frames(frames, "Edges")
        panel.set_mode("overlay")
        panel.show()
        panel.plot.setXRange(250, 280, padding=0)
        QTest.qWait(20)
        np.testing.assert_array_equal(panel._curves[1].yData, np.zeros(4))
        assert np.isnan(panel._curves[2].yData).all()
        assert np.min(panel._curves[3].yData) == -100.
        assert np.max(panel._curves[3].yData) == -10.
        assert panel._curves[4].xData is None or not panel._curves[4].xData.size
        assert not panel._label_items
        assert np.isfinite(panel.plot.viewRange()).all()
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_spectrum_label_hit_geometry_ready_after_frame_presentation(qapp):
    panel = SpectrumPanel()
    panel.resize(700, 600)
    panel.show()
    QTest.qWait(20)
    try:
        frame = SpectrumFrame(1, "Immediate", "#0072b2", "", np.array([100., 200.]), np.array([5., 2.]), True)
        panel.show_frames([frame], "Immediate")
        # Hit coordinates must come from the presented frame, as in real input.
        panel.view.viewport().repaint()
        item = panel._label_items[0]
        pos = item.textItem.mapToScene(item.textItem.boundingRect().center())
        menu = panel._label_context_menu(pos)
        assert menu is not None
        menu.deleteLater()
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_repeated_zoom_membership_and_layout_changes_keep_percent_geometry(qapp):
    panel = SpectrumPanel(optical=True)
    panel.resize(900, 650)
    panel.show()
    x = np.linspace(200, 600, 1201)
    frames = [SpectrumFrame(i + 1, f"Run {i + 1}", "#0072b2", "", x,
                            (i + 1) * (np.exp(-((x - 270 - i * 7) / 9)**2)
                                       + .2 * np.exp(-((x - 450) / 12)**2)), optical=True)
              for i in range(6)]
    originals = [frame.intensity.copy() for frame in frames]
    try:
        for step in range(72):
            chosen = frames[:2 + step % 5]
            if step % 2:
                chosen = list(reversed(chosen))
            panel.show_frames(chosen, "Repeated updates")
            panel.set_mode("overlay" if step % 3 else "stacked")
            panel.match_y.setChecked(step % 4 == 0)
            lo = 240. + (step % 12) * 9
            hi = lo + 140.
            panel._plots()[-1].setXRange(lo, hi, padding=0)
            qapp.processEvents()
            for plot in panel._plots():
                assert plot.viewRange()[0] == pytest.approx([lo, hi])
                assert np.isfinite(plot.viewRange()).all()
            for frame in chosen:
                y = frame.intensity
                reference = max(np.interp(lo, x, y), np.interp(hi, x, y),
                                np.max(y[(x >= lo) & (x <= hi)]))
                curve = panel._curves[frame.file_id]
                expected = y / reference * 100 if panel.relative_overlay else y
                np.testing.assert_allclose(curve.yData, expected)
        for frame, original in zip(frames, originals):
            np.testing.assert_array_equal(frame.intensity, original)
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()
