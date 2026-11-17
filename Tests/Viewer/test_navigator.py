"""Selection, bulk actions, file drag/drop and focus-scoped reset regressions."""
import dataclasses

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QColorDialog, QDockWidget

from kmzviewer.gui.app import MainWindow
from kmzviewer.gui.dialogs import ExtractDialog
from Tests.Viewer.test_ms1_comparison import close, make_run, settled, wait_for


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(qapp, tmp_path):
    paths = [tmp_path / name for name in ("Run 10.mzML", "run 2.mzML", "Run 1.mzML")]
    for path in paths:
        make_run(path, count=6)
    w = MainWindow()
    w.show()
    w.activateWindow()
    assert not w.load_paths_sync([str(p) for p in paths])
    w.extract_sync("173.12,273.1559", 10, "all", False, list(w.session.files))
    w.plots.set_mode("overlay")
    QTest.qWait(30)
    yield w, list(w.session.files.values())
    close(w)


def click_item(w, item, modifiers=Qt.KeyboardModifier.NoModifier, button=Qt.MouseButton.LeftButton):
    w.tree.scrollToItem(item)
    rect = w.tree.visualItemRect(item)
    # Text, clear of expanders and checkboxes.
    pos = QPoint(rect.left() + 70, rect.center().y())
    QTest.mouseClick(w.tree.viewport(), button, modifiers, pos)
    return pos


def selected_keys(w):
    return {tuple(item.data(0, Qt.ItemDataRole.UserRole)) for item in w.tree.selectedItems()}


def action(menu, title):
    return next(a for a in menu.actions() if a.text() == title)


def test_shift_and_command_selection_survives_active_updates(viewer):
    w, files = viewer
    top = w.tree.topLevelItem(0)
    click_item(w, top.child(0))
    click_item(w, top.child(2), Qt.KeyboardModifier.ShiftModifier)
    expected = {("trace", t.trace_id) for t in files[0].traces}
    assert selected_keys(w) == expected
    assert w.active_trace_id == files[0].traces[2].trace_id
    click_item(w, w.tree.topLevelItem(2).child(1), Qt.KeyboardModifier.ControlModifier)
    expected.add(("trace", files[2].traces[1].trace_id))
    assert selected_keys(w) == expected
    w.inspect_rt(files[1].traces[0].trace_id, 1.2)
    settled(w)
    assert selected_keys(w) == expected
    w.tree.topLevelItem(1).setExpanded(False)
    w._refresh_navigator()
    assert selected_keys(w) == expected
    assert not w.tree.topLevelItem(1).isExpanded()


def test_group_menu_visibility_color_and_remove_are_batched(viewer, monkeypatch):
    w, files = viewer
    click_item(w, w.tree.topLevelItem(0).child(1))
    click_item(w, w.tree.topLevelItem(0).child(2), Qt.KeyboardModifier.ShiftModifier)
    click_item(w, w.tree.topLevelItem(2).child(1), Qt.KeyboardModifier.ControlModifier)
    selected = w._selected_trace_ids()
    keys = selected_keys(w)
    calls = []
    refresh = w._refresh_plots
    monkeypatch.setattr(w, "_refresh_plots", lambda: (calls.append(1), refresh()))
    w.inspect_rt(files[0].traces[0].trace_id, 1.2)
    settled(w)
    menu = w._selection_menu(w.tree.topLevelItem(0).child(1))
    action(menu, "Show only selected").trigger()
    assert len(calls) == 1
    assert {t.trace_id for _, t in w.session.ordered_traces(True)} == selected
    assert w.tree.topLevelItem(0).checkState(0) == Qt.CheckState.PartiallyChecked
    assert selected_keys(w) == keys
    action(menu, "Hide selected").trigger()
    assert len(calls) == 2 and not w.session.ordered_traces(True)
    action(menu, "Show only selected").trigger()
    curves = dict(w.plots._curves)
    x = w.plots.current_x_range()
    monkeypatch.setattr(QColorDialog, "getColor", lambda *a: QColor("#b12345"))
    action(menu, "Change selected color…").trigger()
    assert len(calls) == 3 and curves == w.plots._curves
    assert w.plots.current_x_range() == x
    for loaded, trace in w.session.ordered_traces():
        assert trace.color == ("#b12345" if trace.trace_id in selected else loaded.color)
    assert w.spectrum._frames[0].color == files[0].color
    w.session.emphasized.update(selected)
    w.walk_trace_id = next(iter(selected))
    action(menu, "Remove selected chromatograms").trigger()
    assert len(calls) == 4
    assert all(t.trace_id not in selected for _, t in w.session.ordered_traces())
    assert not w.session.emphasized and w.walk_trace_id is None
    assert len(w.session.files) == 3 and len(w.spectrum._frames) == 1
    menu.deleteLater()


def test_parent_checkbox_refreshes_once_and_mixed_selection_deduplicates(viewer, monkeypatch):
    w, files = viewer
    calls = []
    refresh = w._refresh_plots
    monkeypatch.setattr(w, "_refresh_plots", lambda: (calls.append(1), refresh()))
    top = w.tree.topLevelItem(0)
    top.setCheckState(0, Qt.CheckState.Unchecked)
    assert len(calls) == 1
    assert not any(t.visible for t in files[0].traces)
    assert all(top.child(i).checkState(0) == Qt.CheckState.Unchecked for i in range(top.childCount()))
    top.setCheckState(0, Qt.CheckState.Checked)
    assert len(calls) == 2 and all(t.visible for t in files[0].traces)
    click_item(w, top)
    click_item(w, top.child(1), Qt.KeyboardModifier.ControlModifier)
    assert w._selected_trace_ids() == {t.trace_id for t in files[0].traces}
    assert w._selected_file_ids() == [files[0].file_id]


def test_extract_dialog_preselects_source_files_once_and_retains_selection(viewer):
    w, files = viewer
    click_item(w, w.tree.topLevelItem(0).child(1))
    click_item(w, w.tree.topLevelItem(0).child(2), Qt.KeyboardModifier.ShiftModifier)
    click_item(w, w.tree.topLevelItem(2).child(1), Qt.KeyboardModifier.ControlModifier)
    keys = selected_keys(w)
    captured = []
    def accept():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, ExtractDialog)
        captured.append(dialog.checked_file_ids())
        assert dialog.file_list.count() == 3
        dialog.mz_edit.setText("301.14")
        dialog.accept()
    QTimer.singleShot(0, accept)
    w.open_extract_dialog()
    wait_for(lambda: not w._busy)
    assert captured == [[files[0].file_id, files[2].file_id]]
    assert [len(f.traces) for f in files] == [4, 3, 4]
    assert selected_keys(w) == keys
    default = ExtractDialog([(f.file_id, f.name) for f in files])
    assert default.checked_file_ids() == list(w.session.files)
    default.close()


def drop_files(tree, ids, pos):
    mime = tree.mime_for_files(ids)
    enter = QDragEnterEvent(pos, Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(tree.viewport(), enter)
    move = QDragMoveEvent(pos, Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(tree.viewport(), move)
    drop = QDropEvent(QPointF(pos), Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(tree.viewport(), drop)
    assert enter.isAccepted() and move.isAccepted() and drop.isAccepted()


def test_group_drop_sort_and_background_frames_follow_file_order(viewer):
    w, files = viewer
    w.inspect_rt(files[0].traces[0].trace_id, 1.2)
    for f in files[1:]:
        w.toggle_spectrum(f.file_id)
    settled(w)
    for top in [w.tree.topLevelItem(i) for i in range(3)]:
        top.setExpanded(False)
    click_item(w, w.tree.topLevelItem(0))
    click_item(w, w.tree.topLevelItem(1), Qt.KeyboardModifier.ShiftModifier)
    keys = selected_keys(w)
    w.plots.set_x_range(1.1, 1.3)
    w.autoscale_action.setChecked(False)
    w.plots._plots()[0].setYRange(5, 50, padding=0)
    w.spectrum.plot.setXRange(170, 310, padding=0)
    before_arrays = {f.file_id: f.mz for f in w.spectrum._frames}
    spectrum_curves = dict(w.spectrum._curves)
    chrom_curves = dict(w.plots._curves)
    samples = dict(w.plots._samples)
    bottom = w.tree.visualItemRect(w.tree.topLevelItem(2)).bottom() + 10
    drop_files(w.tree, [files[0].file_id, files[1].file_id], QPoint(80, bottom))
    expected = [files[2].file_id, files[0].file_id, files[1].file_id]
    assert list(w.session.files) == w.tree.file_ids() == expected
    assert [f.file_id for f in w.spectrum._frames] == expected
    assert w.spectrum._curves == spectrum_curves
    assert [p.getAxis("bottom").isVisible() for p in w.spectrum._plots()] == [False, False, True]
    assert w.plots._curves == chrom_curves and w.plots._samples == samples
    assert all(f.mz is before_arrays[f.file_id] for f in w.spectrum._frames)
    assert selected_keys(w) == keys
    assert w.plots.current_x_range() == pytest.approx((1.1, 1.3))
    assert w.plots._plots()[0].viewRange()[1] == pytest.approx((5, 50))
    assert w.spectrum.plot.viewRange()[0] == pytest.approx((170, 310))
    # Start another read, then sort before its queued callback can arrive.
    w._navigate_rt(files[0].traces[0].trace_id, 1.4)
    QTest.mouseClick(w.sort_button, Qt.MouseButton.LeftButton)
    settled(w)
    expected = [files[2].file_id, files[1].file_id, files[0].file_id]
    assert list(w.session.files) == w.tree.file_ids() == expected
    assert [f.file_id for f in w.spectrum._frames] == expected
    assert selected_keys(w) == keys
    assert [row[0] for row in w.plots._rows] == [t.trace_id for _, t in w.session.ordered_traces(True)]
    assert [sample.trace_id for sample, _ in w.plots._overlay_plot.legend.items] == list(w.plots._curves)
    assert not w.session.move_files(expected[:2], 1)  # Drop within contiguous group.
    assert not w.session.move_files([999], 0)
    assert not w.session.move_files([expected[0]], 99)


def test_stacked_reorder_reuses_panes_and_retains_manual_ranges(viewer):
    w, files = viewer
    w.plots.set_mode("stacked")
    w.autoscale_action.setChecked(False)
    w.plots.set_x_range(1.1, 1.3)
    panes = dict(w.plots._plot_for_trace)
    for tid, plot in panes.items():
        plot.setYRange(tid, tid + 10, padding=0)
    w.sort_files()
    assert w.plots._plot_for_trace == panes
    for tid, plot in w.plots._plot_for_trace.items():
        assert plot.viewRange()[0] == pytest.approx((1.1, 1.3))
        assert plot.viewRange()[1] == pytest.approx((tid, tid + 10))
    assert w.plots.view.ci.getItem(0, 0) is panes[files[2].traces[0].trace_id]


def test_drop_over_child_moves_whole_file_without_reparenting(viewer):
    w, files = viewer
    pos = w.tree.visualItemRect(w.tree.topLevelItem(1).child(0)).center()
    drop_files(w.tree, [files[0].file_id], pos)
    assert list(w.session.files) == [files[1].file_id, files[0].file_id, files[2].file_id]
    assert [w.tree.topLevelItem(i).childCount() for i in range(3)] == [3, 3, 3]
    assert all(t.file_id == f.file_id for f in files for t in f.traces)


def test_external_mzml_drop_still_opens_files(viewer, monkeypatch):
    from PySide6.QtCore import QMimeData, QUrl
    w, files = viewer
    opened = []
    monkeypatch.setattr(w.runtime, "request", lambda paths: opened.extend(paths))
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(files[0].path)])
    pos = QPoint(90, 50)
    enter = QDragEnterEvent(pos, Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w.tree.viewport(), enter)
    drop = QDropEvent(QPointF(pos), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w.tree.viewport(), drop)
    assert enter.isAccepted() and drop.isAccepted()
    assert opened == [files[0].path]
    assert list(w.session.files) == [f.file_id for f in files]


def test_color_overrides_survive_new_extraction_but_source_command_resets_them(viewer):
    w, files = viewer
    chosen = files[0].traces[1]
    w.set_trace_colors({chosen.trace_id}, "#b12345")
    late = dataclasses.replace(chosen, trace_id=w.session.new_trace_id(), color="#777777")
    w._on_extracted([(files[0].file_id, [late])])
    assert chosen.color == "#b12345" and late.color == files[0].color
    w.set_file_color(files[0].file_id, "#123456")
    assert {t.color for t in files[0].traces} == {"#123456"}


def test_right_click_keeps_group_and_empty_sources_remain_extractable(viewer):
    from PySide6.QtWidgets import QMenu
    w, files = viewer
    top = w.tree.topLevelItem(0)
    click_item(w, top.child(0))
    pos = click_item(w, top.child(2), Qt.KeyboardModifier.ShiftModifier)
    expected = selected_keys(w)
    captured = []
    def dismiss():
        popup = QApplication.activePopupWidget()
        if isinstance(popup, QMenu):
            captured.append(selected_keys(w))
            popup.close()
    timer = QTimer(w)
    timer.timeout.connect(dismiss)
    timer.start(10)
    click_item(w, top.child(1), button=Qt.MouseButton.RightButton)
    w._context_menu(pos)
    QTest.qWait(30)
    timer.stop()
    assert captured and all(keys == expected for keys in captured)
    w.remove_traces(w._selected_trace_ids())
    assert not files[0].traces and files[0].file_id in w.session.files
    click_item(w, w.tree.topLevelItem(0))
    assert w._selected_file_ids() == [files[0].file_id]
    menu = w._selection_menu(w.tree.topLevelItem(0))
    assert action(menu, "Extract EIC…").isEnabled()
    assert not action(menu, "Remove selected chromatograms").isEnabled()
    action(menu, "Restore TIC MS1").trigger()
    assert len(files[0].traces) == 1 and files[0].traces[0].kind == "tic"
    menu.deleteLater()


def test_drag_start_uses_only_selected_parents_in_display_order(viewer, monkeypatch):
    import kmzviewer.gui.navigator as navigator
    w, files = viewer
    click_item(w, w.tree.topLevelItem(0))
    click_item(w, w.tree.topLevelItem(2), Qt.KeyboardModifier.ShiftModifier)
    captured = []
    class Drag:
        def __init__(self, _parent): pass
        def setMimeData(self, mime): captured.append(bytes(mime.data(w.tree.MIME)).decode())
        def exec(self, _action): pass
    monkeypatch.setattr(navigator, "QDrag", Drag)
    w.tree.startDrag(Qt.DropAction.MoveAction)
    assert captured == [w.tree._token + ":" + ",".join(str(f.file_id) for f in files)]
    click_item(w, w.tree.topLevelItem(0).child(0))
    w.tree.startDrag(Qt.DropAction.MoveAction)
    assert len(captured) == 1


def press_f(widget):
    widget.window().activateWindow()
    widget.setFocus()
    QTest.qWait(10)
    QTest.keyClick(widget, Qt.Key.Key_F)
    QTest.qWait(10)


def test_f_resets_only_focused_area_including_manual_y_and_floating_dock(viewer):
    w, files = viewer
    w.inspect_rt(files[0].traces[0].trace_id, 1.2)
    settled(w)
    w.plots.set_x_range(1.1, 1.3)
    w.autoscale_action.setChecked(False)
    w.plots._plots()[0].setYRange(0, 1, padding=0)
    w.spectrum.plot.setXRange(170, 310, padding=0)
    press_f(w.plots.view)
    assert w.plots.current_x_range()[0] < 1 and w.plots.current_x_range()[1] > 1.5
    assert w.plots._plots()[0].viewRange()[1][1] > 100
    assert not w.autoscale_action.isChecked()
    assert w.spectrum.plot.viewRange()[0] == pytest.approx((170, 310))
    w.plots.set_x_range(1.1, 1.3)
    press_f(w.spectrum.view)
    assert w.spectrum.plot.viewRange()[0][0] < 100
    assert w.spectrum.plot.viewRange()[0][1] > 900
    assert w.plots.current_x_range() == pytest.approx((1.1, 1.3))
    w.spectrum_dock.setFloating(True)
    w.spectrum.plot.setXRange(170, 310, padding=0)
    press_f(w.spectrum.view)
    assert w.spectrum.plot.viewRange()[0][1] > 900
    w.spectrum_dock.setFloating(False)


def test_ms2_f_and_text_entry(qapp, tmp_path):
    from kmz2viewer.gui.app import MainWindow as MS2Window
    from kmz2viewer.model import Ms2Scan
    scan = Ms2Scan(path="synthetic.mzML", file_name="synthetic.mzML", spectrum_index=0,
                   scan_id="scan-1", rt_min=1., precursor_mz=300., charge=1,
                   precursor_intensity=None, collision_energy=None, centroid=True,
                   mz=np.array([100., 150., 200.]), intensity=np.array([10., 30., 5.]),
                   base_peak_mz=150., base_peak_intensity=30., tic=45., n_peaks=3)
    w = MS2Window()
    w.show()
    try:
        w.spectrum.show_scan(scan, None, keep_window=False)
        w.spectrum.plot.setXRange(145, 155, padding=0)
        press_f(w.spectrum.plot)
        assert w.spectrum.plot.viewRange()[0][0] < 100
        assert w.spectrum.plot.viewRange()[0][1] > 300
        w.spectrum.plot.setXRange(145, 155, padding=0)
        w.precursor.setFocus()
        QTest.keyClick(w.precursor, Qt.Key.Key_F)
        assert w.precursor.text() == "f"
        assert w.spectrum.plot.viewRange()[0] == pytest.approx((145, 155))
        w.precursor.clear()
        dock = w.spectrum.parentWidget()
        assert isinstance(dock, QDockWidget)
        dock.setFloating(True)
        press_f(w.spectrum.plot)
        assert w.spectrum.plot.viewRange()[0][1] > 300
    finally:
        close(w)
