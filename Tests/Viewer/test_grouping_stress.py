"""Independent grouping/bulk-close stress cases using disposable synthetic runs."""
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from kmzviewer.gui.app import MainWindow
from kmzviewer.model import DadScanTable
from Tests.Viewer.test_ms1_comparison import close, make_run, settled, wait_for
from Tests.Viewer.test_navigator import click_item


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def runs(qapp, tmp_path):
    paths = [tmp_path / name for name in ("Run 20.mzML", "Run 3.mzML", "Run 1.mzML")]
    for path in paths:
        make_run(path, count=8)
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    assert not window.load_paths_sync([str(path) for path in paths])
    window.extract_sync("173.12,273.1559", 10, "all", False, list(window.session.files))
    QTest.qWait(15)
    yield window, list(window.session.files.values())
    close(window)


def groups(window):
    return [window.tree.topLevelItem(i) for i in range(window.tree.topLevelItemCount())]


def key(item):
    return tuple(item.data(0, Qt.ItemDataRole.UserRole))


def find_row(window, row_key):
    return next(item for item in window._navigator_items() if key(item) == row_key)


def select(window, *items):
    window.tree.clearSelection()
    for item in items:
        item.setSelected(True)


def test_grouping_roundtrip_retains_selection_visibility_arrays_and_spectra_order(runs):
    window, files = runs
    window.inspect_rt(files[0].traces[0].trace_id, 1.2)
    for file in files[1:]:
        window.toggle_spectrum(file.file_id)
    settled(window)
    before_frames = list(window.spectrum._frames)
    native = {trace.trace_id: (trace.rt, trace.y, trace.row) for _, trace in window.session.ordered_traces()}
    window.tree.topLevelItem(1).setExpanded(False)
    select(window, window.tree.topLevelItem(0), window.tree.topLevelItem(2).child(1))
    selected = window._selected_trace_ids()
    hidden = files[1].traces[2]
    window.set_trace_visibility({hidden.trace_id})
    window.plots.set_x_range(1.1, 1.4)
    window.set_grouping("type")
    assert window._selected_trace_ids() == selected
    assert [f.file_id for f in window.spectrum._frames] == [f.file_id for f in files]
    assert all(a is b for a, b in zip(window.spectrum._frames, before_frames))
    assert [row[0] for row in window.plots._rows] == [
        file.traces[index].trace_id for index in range(3) for file in files
        if file.traces[index].visible]
    assert [top.childCount() for top in groups(window)] == [3, 3, 3]
    groups(window)[1].setExpanded(False)
    window.set_grouping("file")
    assert not window.tree.topLevelItem(1).isExpanded()
    assert window._selected_trace_ids() == selected
    window.set_grouping("type")
    assert not groups(window)[1].isExpanded()
    assert window.plots.current_x_range() == pytest.approx((1.1, 1.4))
    for _, trace in window.session.ordered_traces():
        assert all(a is b for a, b in zip((trace.rt, trace.y, trace.row), native[trace.trace_id]))
    assert not hidden.visible


def test_group_checkbox_color_remove_and_duplicate_sources_are_batched(runs, monkeypatch):
    window, files = runs
    duplicate = replace(files[0].traces[1], trace_id=window.session.new_trace_id())
    window.session.add_traces(files[0].file_id, [duplicate])
    window.set_grouping("type")
    top = groups(window)[1]
    assert top.childCount() == 4
    calls = []
    refresh = window._refresh_plots
    monkeypatch.setattr(window, "_refresh_plots", lambda: (calls.append(1), refresh()))
    top.setCheckState(0, Qt.CheckState.Unchecked)
    assert len(calls) == 1
    assert all(not window.session.find_trace(key(top.child(i))[1])[1].visible for i in range(4))
    top.child(0).setCheckState(0, Qt.CheckState.Checked)
    assert top.checkState(0) == Qt.CheckState.PartiallyChecked
    select(window, top, top.child(0), groups(window)[2].child(0))
    assert window._selected_file_ids() == [f.file_id for f in files]
    chosen = window._selected_trace_ids()
    window.set_trace_colors(chosen, "#123456")
    assert len(calls) == 2
    assert all(trace.color == "#123456" for _, trace in window.session.ordered_traces() if trace.trace_id in chosen)
    window.remove_traces(chosen)
    assert len(calls) == 3
    assert len(window.session.files) == 3
    assert not chosen.intersection(t.trace_id for _, t in window.session.ordered_traces())
    assert all(window._spectrum_eyes[f.file_id] for f in files)


def test_type_rows_support_real_range_additive_selection_and_context_close(runs):
    window, files = runs
    window.set_grouping("type")
    first, second, third = groups(window)
    click_item(window, first.child(0))
    click_item(window, first.child(2), Qt.KeyboardModifier.ShiftModifier)
    expected = {file.traces[0].trace_id for file in files}
    assert window._selected_trace_ids() == expected
    click_item(window, second.child(1), Qt.KeyboardModifier.ControlModifier)
    assert window._selected_trace_ids() == expected | {files[1].traces[1].trace_id}
    click_item(window, second.child(0))
    click_item(window, third.child(2), Qt.KeyboardModifier.ControlModifier)
    assert window._selected_file_ids() == [files[0].file_id, files[2].file_id]
    before = window._selected_trace_ids()
    click_item(window, second.child(0), button=Qt.MouseButton.RightButton)
    assert window._selected_trace_ids() == before
    menu = window._selection_menu(second.child(0))
    next(action for action in menu.actions() if action.text() == "Close selected files").trigger()
    menu.deleteLater()
    assert list(window.session.files) == [files[1].file_id]


def test_grouped_eye_copies_track_one_file_for_both_detectors(runs, monkeypatch):
    window, files = runs
    # Add detector metadata only to this generated run; the fake optical reads
    # keep this test focused on eye ownership and independent inclusion states.
    file = files[0]
    file.scans.dad = DadScanTable(np.array([100, 101]), np.array([1., 1.7]), ["dad-1", "dad-2"])
    window._dad_enabled.add(file.file_id)
    read = window._spectrum_worker._read
    monkeypatch.setattr(window._spectrum_worker, "_read", lambda path, index:
                        (np.array([250., 254., 260.]), np.array([-1., 10., 0.])) if index >= 100 else read(path, index))
    window.set_grouping("type")
    ms_eyes = [button for fid, button in window._all_spectrum_eyes if fid == file.file_id]
    dad_eyes = [button for fid, button in window._all_dad_eyes if fid == file.file_id]
    assert len(ms_eyes) == len(dad_eyes) == 3
    QTest.mouseClick(ms_eyes[1], Qt.MouseButton.LeftButton)
    settled(window)
    assert all(button.isChecked() for button in ms_eyes)
    assert all(not button.isChecked() for button in dad_eyes)
    QTest.mouseClick(dad_eyes[2], Qt.MouseButton.LeftButton)
    settled(window)
    assert window.dad_file_ids == {file.file_id}
    assert all(button.isChecked() for button in dad_eyes)
    QTest.mouseClick(ms_eyes[0], Qt.MouseButton.LeftButton)
    settled(window)
    assert all(not button.isChecked() for button in ms_eyes)
    assert all(button.isChecked() for button in dad_eyes)
    assert not window.spectrum_file_ids and window.dad_file_ids == {file.file_id}


def test_group_sort_and_rejected_drag_keep_file_and_spectrum_order(runs):
    window, files = runs
    window.inspect_rt(files[0].traces[0].trace_id, 1.2)
    for file in files[1:]:
        window.toggle_spectrum(file.file_id)
    settled(window)
    window.set_grouping("type")
    before_keys = [key(top) for top in groups(window)]
    mime = window.tree.mime_for_files([files[0].file_id])
    enter = QDragEnterEvent(QPoint(80, 80), Qt.DropAction.MoveAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(window.tree.viewport(), enter)
    assert not enter.isAccepted()
    drop = QDropEvent(QPointF(80, 80), Qt.DropAction.MoveAction, mime,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(window.tree.viewport(), drop)
    assert not drop.isAccepted()
    window.sort_files()
    order = [files[2].file_id, files[1].file_id, files[0].file_id]
    assert list(window.session.files) == order
    assert [key(top) for top in groups(window)] == before_keys
    for top in groups(window):
        assert [window.session.find_trace(key(top.child(i))[1])[0].file_id for i in range(top.childCount())] == order
    assert [frame.file_id for frame in window.spectrum._frames] == order


def test_empty_source_remains_selectable_extractable_and_closable_in_type_mode(runs):
    window, files = runs
    window.remove_traces({trace.trace_id for trace in files[0].traces})
    window.set_grouping("type")
    empty = groups(window)[-1]
    assert empty.text(0) == "Files without chromatograms"
    select(window, empty)
    assert window._selected_trace_ids() == set()
    assert window._selected_file_ids() == [files[0].file_id]
    menu = window._selection_menu(empty)
    actions = {action.text(): action for action in menu.actions()}
    assert actions["Extract EIC…"].isEnabled()
    assert not actions["Remove selected chromatograms"].isEnabled()
    actions["Restore TIC MS1"].trigger()
    menu.deleteLater()
    assert len(files[0].traces) == 1
    restored = find_row(window, ("trace", files[0].traces[0].trace_id))
    select(window, restored)
    window.close_action.trigger()
    assert files[0].file_id not in window.session.files


def test_group_close_deduplicates_and_late_frames_cannot_resurrect_sources(runs, monkeypatch):
    window, files = runs
    # Retain a different extraction in file 3 so selecting a type group closes
    # exactly files 1+2, even when that group and its child are both selected.
    window.remove_trace(files[2].traces[1].trace_id)
    window.set_grouping("type")
    target = groups(window)[1]
    assert target.childCount() == 2
    gate, release = Event(), Event()
    read = window._spectrum_worker._read
    def held(path, index):
        if path == files[0].path and not release.is_set():
            gate.set()
            assert release.wait(10)
        return read(path, index)
    monkeypatch.setattr(window._spectrum_worker, "_read", held)
    dropped = []
    drop = window._spectrum_worker.close_path
    monkeypatch.setattr(window._spectrum_worker, "close_path", lambda path: (dropped.append(path), drop(path)))
    window.spectrum_file_ids = {f.file_id for f in files}
    try:
        window.inspect_rt(files[0].traces[0].trace_id, 1.2)
        wait_for(gate.is_set)
        select(window, target, target.child(0))
        window.close_selected_files()
        assert list(window.session.files) == [files[2].file_id]
        assert sorted(dropped) == sorted(f.path for f in files[:2])
        release.set()
        settled(window)
        assert [frame.file_id for frame in window.spectrum._frames] == [files[2].file_id]
        wait_for(lambda: all(f.path not in window._spectrum_worker._readers for f in files[:2]))
        assert window.spectrum_file_ids == {files[2].file_id}
    finally:
        release.set()


def test_same_label_but_distinct_extraction_settings_never_merge(runs):
    window, files = runs
    # Rounded labels can be identical even when native requested values differ.
    for settings in (("173.120000000001", 10, "all"), ("173.12", 20, "all"), ("173.12", 10, "positive")):
        window.extract_sync(*settings, False, [files[0].file_id, files[1].file_id])
    window.set_grouping("type")
    assert len(groups(window)) == 6  # TIC, 2 original EICs, 3 distinct settings.
    assert [top.childCount() for top in groups(window)] == [3, 3, 3, 2, 2, 2]
    for top in groups(window):
        traces = [window.session.find_trace(key(top.child(i))[1])[1] for i in range(top.childCount())]
        assert len({trace.extraction_key for trace in traces}) == 1
