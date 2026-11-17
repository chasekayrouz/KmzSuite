"""Real Qt gestures and lifecycle checks for synchronized MS1 comparisons."""
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kmzviewer.comparison import match_scan
from kmzviewer.gui.app import MainWindow
from kmzviewer.model import ScanTable
from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.005)  # Yield the GIL to the Python spectrum-decoder thread.
    assert predicate()


def settled(window):
    wait_for(lambda: window._inflight_comparison is None and window._queued_comparison is None)
    QTest.qWait(20)


def close(window):
    window.close()
    wait_for(lambda: window.runtime.closed)


def make_run(path, *, offset=0, scale=1, polarity=1, count=24, profile=False):
    mz = np.linspace(100, 900, 8000) if profile else np.array([100., 173.12, 273.1559, 301.14, 445.22, 701.41, 900.])
    base = np.array([0., 40., 100., 65., 30., 70., 0.])
    if profile:
        base = sum(height * np.exp(-((mz - mass) / .3) ** 2) for mass, height in [(173.12,40),(273.1559,100),(301.14,65),(445.22,30),(701.41,70)])
    write_mzml(path, [dict(mz=mz.tolist(), intensity=(base * scale * (1 + i / 10)).tolist(),
                          rt=1 + i * .1 + offset, tic=100 * scale * (1 + np.sin(i * .4)),
                          polarity=polarity if polarity else (1 if i % 2 == 0 else -1),
                          centroid=not profile, id=f'scan-{i}') for i in range(count)])


@pytest.fixture
def viewer(qapp, tmp_path):
    paths = [tmp_path / f'Source {i + 1}.mzML' for i in range(3)]
    for i, p in enumerate(paths):
        make_run(p, offset=i * .015, scale=i + 1)
    w = MainWindow()
    w.resize(1512, 930)
    w.show()
    assert not w.load_paths_sync([str(p) for p in paths])
    wait_for(lambda: abs(w.plots.width() - w.spectrum.width()) < 10)
    yield w, list(w.session.files.values())
    close(w)


def point(window, trace, rt):
    plot = window.plots._plot_for_trace[trace.trace_id]
    view = plot.getViewBox()
    y0, y1 = view.viewRange()[1]
    return window.plots.view.mapFromScene(view.mapViewToScene(QPointF(rt, (y0+y1)/2)))


def click(window, trace, rt, double=False):
    viewport = window.plots.view.viewport()
    pos = point(window, trace, rt)
    QTest.mouseMove(viewport, pos)
    QTest.qWait(20)
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=pos)
    if double:
        QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=pos)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=pos)
    QTest.qWait(25)


def drag(window, trace, start_rt, end_rt):
    viewport = window.plots.view.viewport()
    start, end = point(window, trace, start_rt), point(window, trace, end_rt)
    QTest.mouseMove(viewport, start)
    QTest.qWait(25)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    for fraction in np.linspace(.1, 1, 12):
        pos = QPointF(start) + (QPointF(end)-QPointF(start)) * fraction
        event = QMouseEvent(QEvent.Type.MouseMove, pos, QPointF(viewport.mapToGlobal(pos.toPoint())),
                            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(viewport, event)
        QTest.qWait(20)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    QTest.qWait(20)


def test_matching_is_rt_polarity_aware_and_never_interpolates():
    scans = ScanTable(np.arange(5), np.array([1., 2., 2., 3., 2.5]), np.ones(5),
                      np.array([1, -1, 1, 1, 0]), np.ones(5), ['a','b','c','d','e'])
    assert match_scan(scans, 2.1, 1).row == 2
    assert match_scan(scans, 2.1, -1).row == 1
    assert match_scan(scans, 2.1, 0).row == 4
    assert match_scan(scans, 1.5, 1).row == 0  # Deterministic acquisition-order tie.
    assert match_scan(scans, .99, 1).message == 'Outside run range'
    assert match_scan(scans, 3.01, 1).row is None
    scans.polarity[:] = 1
    assert match_scan(scans, 2., -1).message == 'No negative MS1 scans'
    scans.rt_min[:] = np.nan
    assert match_scan(scans, 2., 1).message == 'No MS1 scans'


def test_real_clicks_membership_eyes_and_independent_visibility(viewer):
    w, files = viewer
    a,b,c = [f.traces[0] for f in files]
    assert w.dockWidgetArea(w.spectrum_dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert abs(w.plots.width()-w.spectrum.width()) < 10
    click(w, a, 1.5)
    assert not w.spectrum_file_ids
    click(w, a, 1.5, double=True)
    settled(w)
    assert w.spectrum_file_ids == {files[0].file_id}
    assert w.walk_action.isChecked() and not w.pan_action.isChecked()
    click(w, b, 2.0)
    settled(w)
    assert w.spectrum_file_ids == {files[0].file_id}
    assert w._selected_rt == pytest.approx(2.015)
    assert w.spectrum._frames[0].rt == pytest.approx(2.)
    click(w, b, 2., double=True)
    settled(w)
    assert w.spectrum_file_ids == {files[0].file_id, files[1].file_id}
    assert [f.rt for f in w.spectrum._frames] == pytest.approx([2., 2.015])
    w.extract_sync('273.1559', 10, 'all', False, [files[1].file_id])
    QTest.qWait(20)
    click(w, files[1].traces[1], 2.2, double=True)
    settled(w)
    assert len(w.spectrum_file_ids) == 2
    before = [(t.trace_id,t.visible) for _,t in w.session.ordered_traces()]
    QTest.mouseClick(w._spectrum_eyes[files[0].file_id], Qt.MouseButton.LeftButton)
    settled(w)
    assert [f.file_id for f in w.spectrum._frames] == [files[1].file_id]
    QTest.mouseClick(w._spectrum_eyes[files[2].file_id], Qt.MouseButton.LeftButton)
    settled(w)
    assert [(t.trace_id,t.visible) for _,t in w.session.ordered_traces()] == before
    assert set(f.file_id for f in w.spectrum._frames) == {files[1].file_id,files[2].file_id}
    w.tree.topLevelItem(1).child(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert files[1].file_id in w.spectrum_file_ids
    for fid in list(w.spectrum_file_ids): w.toggle_spectrum(fid)
    settled(w)
    assert not w.spectrum._frames
    click(w, c, 2.5)
    assert not w.spectrum_file_ids
    w.toggle_spectrum(files[0].file_id)
    settled(w)
    assert len(w.spectrum._frames) == 1
    assert w.spectrum._frames[0].rt == pytest.approx(2.5)


def test_walk_scrubs_without_pan_and_zoom_keeps_common_mz(viewer):
    w, files = viewer
    a,b,_ = [f.traces[0] for f in files]
    click(w, a, 1.4, double=True)
    click(w, b, 1.4, double=True)
    settled(w)
    x_before = w.plots.current_x_range()
    curves = dict(w.spectrum._curves)
    w.spectrum._plots()[-1].setXRange(260., 310., padding=0)
    QTest.qWait(30)
    drag(w, a, 1.4, 2.6)
    settled(w)
    assert w._selected_rt == pytest.approx(2.6)
    assert w.plots.current_x_range() == pytest.approx(x_before)
    assert w.spectrum._curves == curves
    assert all(p.viewRange()[0] == pytest.approx([260.,310.]) for p in w.spectrum._plots())
    assert [f.rt for f in w.spectrum._frames] == pytest.approx([2.6,2.615])
    w.walk_by(1)
    settled(w)
    assert w._selected_rt == pytest.approx(2.7)
    w.pan_action.trigger()
    assert not w.walk_mode
    drag(w, a, 1.5, 1.8)
    assert w.plots.current_x_range() != pytest.approx(x_before)
    click(w, b, 2.)
    settled(w)
    assert len(w.spectrum_file_ids) == 2 and not w.walk_mode
    w.zoom_action.trigger()
    span = np.ptp(w.plots.current_x_range())
    drag(w, a, 1.5, 2.)
    assert np.ptp(w.plots.current_x_range()) < span
    w.spectrum.set_mode('overlay')
    assert len(w.spectrum._plots()) == 1
    assert w.spectrum.plot.viewRange()[0] == pytest.approx([260.,310.])
    assert w.plots._mode == 'stacked'
    w.spectrum.set_mode('stacked')
    assert all(p.viewRange()[0] == pytest.approx([260.,310.]) for p in w.spectrum._plots())
    assert w.spectrum._plots()[1].viewRange()[1][1] > w.spectrum._plots()[0].viewRange()[1][1]
    w.spectrum.match_y.setChecked(True)
    assert w.spectrum._plots()[0].viewRange()[1] == pytest.approx(w.spectrum._plots()[1].viewRange()[1])
    w.spectrum.reset_range()
    assert w.spectrum.plot.viewRange()[0][0] < 100.


def test_empty_polarity_outside_range_and_per_file_read_failure(qapp, tmp_path, monkeypatch):
    paths = [tmp_path/f'Polarity {i}.mzML' for i in range(3)]
    make_run(paths[0], polarity=0)
    make_run(paths[1], polarity=1)
    make_run(paths[2], polarity=0, offset=2)
    w=MainWindow(); w.show()
    try:
        assert not w.load_paths_sync([str(p) for p in paths])
        files=list(w.session.files.values())
        w.inspect_rt(files[0].traces[0].trace_id,1.5)
        for f in files[1:]: w.toggle_spectrum(f.file_id)
        settled(w)
        assert w._selected_polarity == -1
        assert w.spectrum._frames[1].message == 'No negative MS1 scans'
        assert w.spectrum._frames[2].message == 'Outside run range'
        assert all(f.mz.size==0 for f in w.spectrum._frames[1:])
        assert all(w.spectrum._plots_by_file[f.file_id].getAxis('left').style['tickLength']==0
                   for f in w.spectrum._frames[1:])
        read = w._spectrum_worker._read
        def failing(path,index):
            if path==str(paths[1]): raise ValueError('Synthetic read failure')
            return read(path,index)
        monkeypatch.setattr(w._spectrum_worker,'_read',failing)
        w._navigate_rt(files[0].traces[0].trace_id,2.)
        settled(w)
        assert w.spectrum._frames[0].mz.size
        assert w.spectrum._frames[1].message == 'Could not read scan'
        assert not w.spectrum._frames[1].mz.size
        assert 'Synthetic read failure' in w.spectrum._frames[1].caption
        assert w.spectrum._frames[2].message == 'Outside run range'
    finally: close(w)


def test_latest_position_wins_and_hidden_closed_files_never_return(viewer, monkeypatch):
    w, files=viewer
    w.inspect_rt(files[0].traces[0].trace_id,1.5)
    for f in files[1:]: w.toggle_spectrum(f.file_id)
    settled(w)
    read=w._spectrum_worker._read
    entered, release=threading.Event(), threading.Event()
    def slow(path,index):
        entered.set()
        assert release.wait(5)
        return read(path,index)
    monkeypatch.setattr(w._spectrum_worker,'_read',slow)
    w._navigate_rt(files[0].traces[0].trace_id,1.8)
    wait_for(entered.is_set)
    try:
        for rt in np.linspace(1.8,3.,100): w._navigate_rt(files[0].traces[0].trace_id,float(rt))
        assert w._spectrum_worker._queue.qsize() <= 1
        w.toggle_spectrum(files[1].file_id)
        w.close_file(files[2].file_id)
    finally: release.set()
    settled(w)
    assert len(w.spectrum._frames)==1
    assert w.spectrum._frames[0].file_id==files[0].file_id
    assert w.spectrum._frames[0].rt==pytest.approx(3.)
    assert files[2].path not in w._spectrum_worker._readers
    w.remove_trace(files[0].traces[0].trace_id)
    assert len(w.spectrum._frames)==1  # File visibility survives chromatogram removal.
    w.close_file(files[0].file_id)
    settled(w)
    assert not w.spectrum._frames


def test_eye_can_start_inspection_and_duplicate_rt_keeps_exact_reference(qapp, tmp_path):
    paths = [tmp_path / 'Repeated RT.mzML', tmp_path / 'Comparison.mzML']
    write_mzml(paths[0], [dict(mz=[100.], intensity=[value], rt=1., tic=value, polarity=1,
                             centroid=True, id=f'repeat-{i}') for i, value in enumerate([10., 20.])])
    make_run(paths[1])
    w = MainWindow(); w.show()
    try:
        assert not w.load_paths_sync([str(p) for p in paths])
        a, b = list(w.session.files.values())
        QTest.mouseClick(w._spectrum_eyes[b.file_id], Qt.MouseButton.LeftButton)
        settled(w)
        assert w.active_trace_id == b.traces[0].trace_id
        assert w.plots._active_id == w.active_trace_id
        w.inspect_rt(a.traces[0].trace_id, 1.)
        w.walk_by(1)
        settled(w)
        assert w.spectrum._frames[0].intensity.tolist() == [20.]
        w.toggle_spectrum(b.file_id); settled(w)
        w.toggle_spectrum(b.file_id); settled(w)
        assert w.spectrum._frames[0].intensity.tolist() == [20.]
        assert 'repeat-1' in w.spectrum._frames[0].caption
    finally:
        close(w)


@pytest.mark.parametrize('size', [(1512,930),(1440,860),(1280,800)])
def test_six_panes_have_legible_labels_and_keep_native_arrays(qapp,tmp_path,size):
    paths=[tmp_path/f'Source {i+1} with a deliberately very long descriptive filename.mzML' for i in range(6)]
    for i,p in enumerate(paths): make_run(p,scale=i+1,profile=bool(i%2),count=4)
    w=MainWindow(); w.resize(*size); w.show()
    try:
        assert not w.load_paths_sync([str(p) for p in paths])
        for f in w.session.files.values(): w.inspect_rt(f.traces[0].trace_id,1.2)
        settled(w)
        assert len(w.spectrum._plots())==6
        assert min(p.getViewBox().height() for p in w.spectrum._plots()) >= 85
        arrays=[(f.mz.copy(), f.intensity.copy()) for f in w.spectrum._frames]
        for mode in ('stacked','overlay','stacked'):
            w.spectrum.set_mode(mode); QTest.qWait(30)
            if mode=='stacked':
                for fid,pool in w.spectrum._label_pools.items():
                    visible=[l for l in pool if l.isVisible()]
                    assert visible
                    view=w.spectrum._plots_by_file[fid].getViewBox()
                    rect=view.sceneBoundingRect()
                    title=w.spectrum._names[fid].sceneBoundingRect()
                    for label in visible:
                        bounds=label.textItem.mapRectToScene(label.textItem.boundingRect())
                        assert bounds.top() >= title.bottom()
                        assert bounds.left() >= rect.left()+1
                        assert bounds.right() <= rect.right()-1
                        assert bounds.bottom() <= rect.bottom()
            for frame,(mz,y) in zip(w.spectrum._frames,arrays):
                np.testing.assert_array_equal(frame.mz,mz)
                np.testing.assert_array_equal(frame.intensity,y)
    finally: close(w)
