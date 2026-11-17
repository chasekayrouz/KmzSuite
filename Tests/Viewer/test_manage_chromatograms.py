"""Grouped session actions keep trace visibility and source spectrum eyes independent."""
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from kmzcore.runtime.synthetic import write_fixture
from kmzviewer.dad import WavelengthExtraction
from kmzviewer.extract import make_tic_trace
from kmzviewer.gui.app import MainWindow
from kmzviewer.gui.manage_chromatograms import ManageChromatogramsDialog
from kmzviewer.model import ScanTable, Session, Trace
from Tests.Viewer.test_ms1_comparison import close, settled, wait_for


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session():
    s = Session()
    scans = ScanTable(np.array([0]), np.array([1.]), np.array([4.]), np.array([1]), np.array([False]), ['scan'])
    for name in ('Alpha', 'Beta'):
        loaded = s.add_file(f'{name}.mzML', scans, make_tic_trace(s.new_trace_id(), s.next_color(), scans))
        specs = [('bpc', 'BPC MS1', ('bpc', 'all')), ('bpc', 'BPC MS1', ('bpc', 'positive')),
                 ('eic', 'EIC 100', ('eic', 'all', 'masses', (100.,), 10.)),
                 ('eic', 'EIC 100', ('eic', 'all', 'masses', (100.00000000001,), 10.)),
                 ('twc', 'TWC', ('twc',)), ('ewc', 'EWC 254', ('ewc',254.,4.,None)),
                 ('ewc', 'EWC 254', ('ewc',254.,4.,(258.,2.)))]
        traces = [Trace(s.new_trace_id(), loaded.file_id, kind, label, scans.rt_min, np.array([1.]),
                        np.array([0]), 'Native extraction', loaded.color, visible=(name == 'Alpha'), extraction_key=key)
                  for kind, label, key in specs]
        s.add_traces(loaded.file_id, traces)
    return s


def family(dialog, kind):
    return next(dialog.tree.topLevelItem(i) for i in range(dialog.tree.topLevelItemCount())
                if dialog.tree.topLevelItem(i).text(0).startswith(kind+' —'))


def test_group_and_family_selection_with_individual_exceptions(qapp, session):
    before = [t for _, t in session.ordered_traces()]
    d = ManageChromatogramsDialog(session)
    try:
        assert d.tree.topLevelItemCount() == 5
        assert not d.selected_trace_ids() and not d.remove_button.isEnabled()
        bpc = family(d, 'BPC')
        assert bpc.childCount() == 2  # Polarity groups remain distinct.
        bpc.setCheckState(0, Qt.CheckState.Checked)
        expected = {t.trace_id for _, t in session.ordered_traces() if t.kind == 'bpc'}
        assert d.selected_trace_ids() == expected
        assert d.remove_button.text() == 'Remove 4 chromatograms'
        leaf = bpc.child(0).child(1)
        assert leaf.text(2) == 'Hidden'
        leaf.setCheckState(0, Qt.CheckState.Unchecked)
        assert d.selected_trace_ids() == expected - {leaf.data(0, Qt.ItemDataRole.UserRole)}
        assert bpc.checkState(0) == Qt.CheckState.PartiallyChecked
        assert bpc.child(0).checkState(0) == Qt.CheckState.PartiallyChecked
        assert family(d, 'EIC').childCount() == 2
        assert family(d, 'EIC').child(0).toolTip(0) != family(d, 'EIC').child(1).toolTip(0)
        assert family(d, 'EWC').childCount() == 2
        assert 'reference: none' in family(d, 'EWC').child(0).toolTip(0)
        d.reject()
        assert [t for _, t in session.ordered_traces()] == before
    finally:
        d.close()


def test_search_limits_bulk_actions_and_keeps_outside_selections_explicit(qapp, session):
    d = ManageChromatogramsDialog(session)
    try:
        family(d, 'TIC').setCheckState(0, Qt.CheckState.Checked)
        original = d.selected_trace_ids()
        d.search.setText('bpc Beta')
        assert len(d._matches) == 2
        family(d, 'BPC').setCheckState(0, Qt.CheckState.Checked)
        wanted = {t.trace_id for f,t in session.ordered_traces() if f.name == 'Beta.mzML' and t.kind == 'bpc'}
        assert d.selected_trace_ids() == original | wanted
        assert '2 selected outside this search' in d.count.text()
        assert family(d, 'BPC').text(1) == '2 of 4'
        assert all(d._leaves[tid].text(2) == 'Hidden' for tid in wanted)
        d.search.clear()
        assert family(d, 'BPC').checkState(0) == Qt.CheckState.PartiallyChecked
        d.search.setText('nothing matches this')
        assert not d.select_shown.isEnabled()
        assert d.selected_trace_ids() == original | wanted
        d.clear_selection.click()
        assert not d.selected_trace_ids() and not d.remove_button.isEnabled()
        d.search.clear()
        d.select_shown.click()
        assert d.selected_trace_ids() == {t.trace_id for _,t in session.ordered_traces()}
    finally:
        d.close()


def test_keyboard_search_return_does_not_remove_and_space_checks_group(qapp, session):
    d = ManageChromatogramsDialog(session)
    d.show()
    try:
        d.search.setFocus()
        QTest.keyClicks(d.search, 'TWC')
        QTest.keyClick(d.search, Qt.Key.Key_Return)
        assert d.isVisible() and d.result() != QDialog.DialogCode.Accepted
        item = family(d, 'TWC')
        d.tree.setCurrentItem(item)
        d.tree.setFocus()
        QTest.keyClick(d.tree, Qt.Key.Key_Space)
        assert len(d.selected_trace_ids()) == 2
        QTest.keyClick(d, Qt.Key.Key_Escape)
        assert d.result() == QDialog.DialogCode.Rejected
    finally:
        d.close()
    confirm = ManageChromatogramsDialog(session)
    requests = []
    confirm.action_requested.connect(lambda action, ids: requests.append((action, ids)))
    confirm.show()
    try:
        confirm.select_shown.click()
        confirm.remove_button.setFocus()
        qapp.processEvents()
        QTest.keyClick(confirm.remove_button, Qt.Key.Key_Return)
        assert requests == [("remove", {t.trace_id for _, t in session.ordered_traces()})]
        assert confirm.isVisible()  # Applying is separate from closing.
    finally:
        confirm.close()


def test_empty_dialog_and_duplicate_trace_instances(qapp, session):
    empty = ManageChromatogramsDialog(Session())
    assert all(not button.isEnabled() for button in empty.action_buttons.values())
    empty.close()
    loaded = next(iter(session.files.values()))
    source = loaded.traces[1]
    duplicate = replace(source, trace_id=session.new_trace_id())
    session.add_traces(loaded.file_id, [duplicate])
    d = ManageChromatogramsDialog(session)
    try:
        leaf = d._leaves[duplicate.trace_id]
        assert '(copy 2)' in leaf.text(0)
        leaf.setCheckState(0, Qt.CheckState.Checked)
        assert d.selected_trace_ids() == {duplicate.trace_id}
    finally:
        d.close()


def test_unavailable_spectra_and_refresh_preserve_search_and_selection(qapp, session):
    alpha, beta = session.files.values()
    alpha.scans = replace(alpha.scans, rt_min=np.array([np.nan]))
    d = ManageChromatogramsDialog(session, dad_enabled={beta.file_id})
    try:
        d.search.setText('Alpha bpc')
        d.select_shown.click()
        chosen = d.selected_trace_ids()
        assert len(chosen) == 2
        assert d.action_buttons['show_plots'].isEnabled()
        assert not d.action_buttons['show_ms1'].isEnabled()
        assert not d.action_buttons['show_dad'].isEnabled()
        assert all(d._leaves[tid].text(3) == 'Unavailable' for tid in chosen)
        d.search.setText('Beta bpc')
        d.select_shown.click()
        assert '2 selected outside this search' in d.count.text()
        assert '1 MS1 files; 1 DAD files' in d.count.text()
        assert all(button.isEnabled() for button in d.action_buttons.values())
        selected = d.selected_trace_ids()
        group = family(d, 'BPC').child(0)
        group.setExpanded(False)
        group_key = group.data(0, Qt.ItemDataRole.UserRole)
        removed = next(t for t in alpha.traces if t.trace_id in chosen)
        session.remove_trace(removed.trace_id)
        d.refresh({beta.file_id}, {beta.file_id}, {beta.file_id})
        assert d.search.text() == 'Beta bpc'
        assert d.selected_trace_ids() == selected - {removed.trace_id}
        group = next(item for item in d._branches if item.data(0, Qt.ItemDataRole.UserRole) == group_key)
        assert not group.isExpanded()
        assert all(d._leaves[t.trace_id].text(3) == 'Visible' for t in beta.traces)
        assert all(d._leaves[t.trace_id].text(4) == 'Visible' for t in beta.traces)
    finally:
        d.close()


def test_repeated_plot_and_spectrum_actions_are_independent_and_batch_sources(qapp, tmp_path, monkeypatch):
    paths = [tmp_path / f'{name}.mzML' for name in ('Alpha', 'Beta', 'Gamma')]
    for i, path in enumerate(paths):
        write_fixture(path, include_dad=i < 2)
    w = MainWindow()
    w.show()
    try:
        w.load_paths_sync([str(path) for path in paths])
        files = list(w.session.files.values())
        alpha, beta, gamma = files
        w.extract_bpc_sync('all', list(w.session.files))
        w.extract_wavelength_sync([WavelengthExtraction()], [alpha.file_id, beta.file_id])
        w.set_grouping('type')
        w.set_spectrum_visibility({alpha.file_id}, detector='ms1', visible=True)
        w.set_spectrum_visibility({alpha.file_id}, detector='dad', visible=False)
        settled(w)
        assert w.spectrum_file_ids == {alpha.file_id}
        assert w.dad_file_ids == {beta.file_id}
        originals = {t.trace_id: (t.rt, t.y, t.row) for _, t in w.session.ordered_traces()}
        invalidations = []
        invalidate = w._invalidate_comparison
        monkeypatch.setattr(w, '_invalidate_comparison', lambda: (invalidations.append(1), invalidate()))
        errors = []

        def choose():
            d = QApplication.activeModalWidget()
            try:
                assert isinstance(d, ManageChromatogramsDialog)
                assert w.manage_chromatograms_button.text() == 'Remove/Show/Hide…'
                d.select_shown.click()  # Several selected traces share each source eye.
                selected = d.selected_trace_ids()
                d.search.setText('Alpha')
                assert 'selected outside this search' in d.count.text()
                included = (set(w.spectrum_file_ids), set(w.dad_file_ids))
                for action, visible in (('hide_plots', False), ('show_plots', True)):
                    d.action_buttons[action].click()
                    assert all(t.visible == visible for _, t in w.session.ordered_traces())
                    assert (w.spectrum_file_ids, w.dad_file_ids) == included
                    assert d.isVisible() and d.selected_trace_ids() == selected
                    assert d.search.text() == 'Alpha'
                    assert all(leaf.text(2) == ('Visible' if visible else 'Hidden') for leaf in d._leaves.values())

                expected_ms1, expected_dad = set(w.spectrum_file_ids), set(w.dad_file_ids)
                for detector, targets in (('ms1', set(w.session.files)), ('dad', {alpha.file_id, beta.file_id})):
                    for operation in ('toggle', 'show', 'show', 'hide', 'toggle'):
                        before = len(invalidations)
                        previous = set(expected_ms1 if detector == 'ms1' else expected_dad)
                        expected = (previous ^ targets if operation == 'toggle' else targets if operation == 'show' else set())
                        if detector == 'ms1':
                            expected_ms1 = expected
                        else:
                            expected_dad = expected
                        d.action_buttons[f'{operation}_{detector}'].click()
                        assert len(invalidations) - before == int(previous != expected)
                        assert w.spectrum_file_ids == expected_ms1
                        assert w.dad_file_ids == expected_dad
                        assert d.isVisible() and d.selected_trace_ids() == selected
                        for fid, eye in w._all_spectrum_eyes:
                            assert eye.isChecked() == (fid in expected_ms1)
                        for fid, eye in w._all_dad_eyes:
                            assert eye.isChecked() == (fid in expected_dad)
                        for tid, leaf in d._leaves.items():
                            fid = d._file_ids[tid]
                            assert leaf.text(3) == ('Visible' if fid in expected_ms1 else 'Hidden')
                            assert leaf.text(4) == ('Unavailable' if fid == gamma.file_id else
                                                     'Visible' if fid in expected_dad else 'Hidden')
                assert all(t.visible for _, t in w.session.ordered_traces())
                for _, trace in w.session.ordered_traces():
                    assert all(a is b for a, b in zip((trace.rt, trace.y, trace.row), originals[trace.trace_id]))
            except BaseException as error:
                errors.append(error)
            finally:
                d.reject()

        QTimer.singleShot(0, choose)
        w.manage_chromatograms_button.click()
        if errors:
            raise errors[0]
        settled(w)
        assert {frame.file_id for frame in w.spectrum._frames} == set(w.session.files)
        assert {frame.file_id for frame in w.dad_spectrum._frames} == {alpha.file_id, beta.file_id}
    finally:
        close(w)


def test_dialog_removes_once_preserving_sources_spectra_and_remaining_filter_cache(qapp, tmp_path, monkeypatch):
    path = tmp_path/'generic.mzML'
    write_fixture(path, include_dad=True)
    w = MainWindow()
    w.show()
    try:
        w.load_paths_sync([str(path)])
        loaded = next(iter(w.session.files.values()))
        w.extract_bpc_sync('all', [loaded.file_id])
        w.extract_sync('100', 10., 'all', False, [loaded.file_id])
        w.extract_wavelength_sync([WavelengthExtraction()], [loaded.file_id])
        w.inspect_rt(loaded.traces[0].trace_id, 1.)
        settled(w)
        w.apply_mz_filter(100,200)
        wait_for(lambda: not w._busy)
        bpc = next(t for t in loaded.traces if t.kind == 'bpc')
        twc = next(t for t in loaded.traces if t.kind == 'twc')
        w.session.set_visible(bpc.trace_id, False)
        original_arrays = {t.trace_id:t.y for t in loaded.traces}
        w.activate_trace(bpc.trace_id, retarget=False)
        w.walk_trace_id, w.walk_i = bpc.trace_id, 0
        w.session.emphasized.add(bpc.trace_id)
        included = (set(w.spectrum_file_ids),set(w.dad_file_ids))
        refreshes=[]
        refresh=w._refresh_all
        monkeypatch.setattr(w,'_refresh_all',lambda: (refreshes.append(1),refresh()))
        def choose():
            d=QApplication.activeModalWidget()
            assert isinstance(d,ManageChromatogramsDialog)
            family(d,'BPC').setCheckState(0,Qt.CheckState.Checked)
            family(d,'TWC').setCheckState(0,Qt.CheckState.Checked)
            assert d.selected_trace_ids() == {bpc.trace_id,twc.trace_id}
            d.remove_button.click()
            assert d.isVisible() and not d.selected_trace_ids()
            assert not ({bpc.trace_id, twc.trace_id} & d._leaves.keys())
            d.reject()
        QTimer.singleShot(0,choose)
        w.manage_chromatograms_button.click()
        assert refreshes == [1]
        assert loaded.file_id in w.session.files
        assert (w.spectrum_file_ids,w.dad_file_ids) == included
        assert [t.kind for t in loaded.traces] == ['tic','eic']
        assert bpc.trace_id not in w._filter_values and bpc.trace_id not in w.session.emphasized
        assert w.walk_trace_id is None
        for t in loaded.traces:
            assert t.y is original_arrays[t.trace_id]
            assert t.trace_id in w._filter_values
        QTimer.singleShot(0,lambda: QApplication.activeModalWidget().reject())
        w.open_manage_chromatograms()
        assert refreshes == [1]  # Cancel never changes the session.
        def remove_all():
            d=QApplication.activeModalWidget();d.select_shown.click();d.remove_button.click()
            assert d.isVisible() and not d._leaves
            assert all(not b.isEnabled() for b in d.action_buttons.values())
            d.reject()
        QTimer.singleShot(0,remove_all)
        w.manage_chromatograms_action.trigger()
        assert len(w.session.files) == 1 and not loaded.traces
        assert not w.manage_chromatograms_button.isEnabled()
        assert not w.manage_chromatograms_action.isEnabled()
        w.restore_tic(loaded.file_id)
        wait_for(lambda: not w._busy and not w._filter_refresh_pending)
        assert w.manage_chromatograms_button.isEnabled()
    finally:
        close(w)
