"""File ownership, fresh-session launch, and precursor-only search regressions."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from kmzcore.runtime import handoff
from kmzcore.runtime.launch import startup_arguments
from kmzviewer.comparison import SpectrumFrame
from kmzviewer.gui.app import MainWindow
from kmzviewer.gui.dialogs import Ms2FilesDialog
from Tests.Viewer.mzml_factory import write_mzml
from Tests.Viewer.test_ms1_comparison import close, wait_for


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def test_development_command_roundtrips_full_precision_and_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff.sys, 'frozen', False, raising=False)
    paths = [tmp_path / 'source Ω one.mzML', tmp_path / '-source two.mzML']
    mass = 300.1234567890123
    command = handoff.ms2_command(paths, precursor_mz=mass)
    assert command[:3] == [handoff.sys.executable, '-m', 'kmz2viewer']
    parsed = startup_arguments('ms2', command[3:])
    assert parsed.paths == [str(p) for p in paths]
    assert parsed.precursor_mz == mass
    assert startup_arguments('ms2', handoff.ms2_command()[3:]).paths == []
    assert startup_arguments('ms2', ['-psn_0_123', str(paths[0])]).precursor_mz is None


def test_diagnostic_popup_waits_for_ready_children_and_parent_activation(qapp, tmp_path, monkeypatch):
    from kmzcore.runtime.handoff_smoke import run_handoff

    state = {'application': Qt.ApplicationState.ApplicationInactive, 'window': False}
    requests, popups, launches = [], [], []
    monkeypatch.setattr(qapp, 'applicationState', lambda: state['application'])
    monkeypatch.setattr('kmzviewer.gui.app.open_ms2',
                        lambda paths, **kw: launches.append((paths, kw)))
    monkeypatch.setattr(QMenu, 'popup', lambda *args: popups.append(True))
    window = MainWindow()
    window.show()
    monkeypatch.setattr(window, 'isActiveWindow', lambda: state['window'])
    monkeypatch.setattr(window, 'raise_', lambda: requests.append('raise'))
    monkeypatch.setattr(window, 'activateWindow', lambda: requests.append('activate'))
    try:
        run_handoff(qapp, window, 'ms1', tmp_path, 'parent')
        wait_for(lambda: (tmp_path / 'files/expected.json').exists()
                 and window._inflight_comparison is None and bool(window.spectrum._label_items))
        timer = window._handoff_timer
        timer.stop()
        # Drive the actual diagnostic callback at each synchronization boundary.
        timer.timeout.emit()
        assert len(launches) == 2 and not popups and not requests
        (tmp_path / 'blank/ready.json').write_text('{"ok":true,"pid":101}')
        timer.timeout.emit()
        assert not popups and not requests
        (tmp_path / 'files/ready.json').write_text('{"ok":true,"pid":102}')
        timer.timeout.emit()
        assert requests == ['raise', 'activate'] and not popups
        state['application'] = Qt.ApplicationState.ApplicationActive
        timer.timeout.emit()
        assert not popups  # An active sibling window is not the source window.
        state['window'] = True
        timer.timeout.emit()
        assert len(popups) == 1 and len(launches) == 3
        assert (tmp_path / 'peak/expected.json').exists()
        assert not (tmp_path / 'blank/release').exists()  # Sessions stay concurrent.
    finally:
        window._handoff_timer.stop()
        close(window)


@pytest.mark.parametrize('mass', ['nan', 'inf', '-1', '0', 'invalid'])
def test_invalid_startup_mass_is_rejected(mass):
    with pytest.raises(SystemExit):
        startup_arguments('ms2', [f'--precursor-mz={mass}'])


def test_frozen_launch_uses_exact_sibling_and_new_instance(tmp_path, monkeypatch):
    folder = tmp_path / 'Installed suite'
    executable = folder / 'KmzViewer.app/Contents/MacOS/KmzViewer'
    target = folder / 'Kmz2Viewer.app/Contents/MacOS/Kmz2Viewer'
    target.parent.mkdir(parents=True)
    target.touch()
    monkeypatch.setattr(handoff.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(handoff.sys, 'executable', str(executable))
    calls = []
    monkeypatch.setattr(handoff.subprocess, 'run', lambda command, **kw:
                        calls.append((command, kw)) or SimpleNamespace(returncode=0))
    handoff.open_ms2([tmp_path / 'A.mzML'], precursor_mz=300.123456789)
    assert calls[0][0][:5] == ['/usr/bin/open', '-n', '-a', str(target.parents[2]), '--args']
    assert startup_arguments('ms2', calls[0][0][5:]).precursor_mz == 300.123456789
    target.unlink()
    with pytest.raises(FileNotFoundError, match='beside'):
        handoff.open_ms2()
    assert len(calls) == 1  # Missing sibling must never fall back to another installation.


def test_picker_empty_selection_and_same_names(qapp):
    files = [SimpleNamespace(file_id=1, path='/folder-a/run.mzML'),
             SimpleNamespace(file_id=2, path='/folder-b/run.mzML')]
    dialog = Ms2FilesDialog(files, selected_file_ids=[2])
    try:
        assert dialog.checked_file_ids() == [2]
        assert dialog.file_list.item(0).text() != dialog.file_list.item(1).text()
        dialog.file_list.item(1).setCheckState(Qt.CheckState.Unchecked)
        assert not dialog.open_button.isEnabled()
        dialog.accept()
        assert dialog.result() == 0
        dialog.file_list.item(0).setCheckState(Qt.CheckState.Checked)
        assert dialog.open_button.isEnabled()
        dialog.accept()
        assert dialog.result() == 1
    finally:
        dialog.close()


def test_toolbar_blank_multi_file_cancel_and_failure(qapp, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('kmzviewer.gui.app.open_ms2', lambda paths, **kw: calls.append((paths, kw)))
    window = MainWindow()
    window.show()
    try:
        assert window.ms2_button.text() == 'Open Kmz2Viewer for MS/MS'
        assert not window.ms2_files_action.isEnabled()
        window.ms2_blank_action.trigger()
        assert calls == [([], {'precursor_mz': None})]
        paths = [str(tmp_path / f'Run {i}.mzML') for i in range(3)]
        for path in paths:
            write_mzml(Path(path), [dict(mz=[100], intensity=[5], rt=1)])
        assert window.load_paths_sync(paths) == []
        assert window.ms2_files_action.isEnabled()
        def choose():
            dialog = QApplication.activeModalWidget()
            assert isinstance(dialog, Ms2FilesDialog)
            for i in range(3):
                dialog.file_list.item(i).setCheckState(Qt.CheckState.Checked if i in (0, 2)
                                                     else Qt.CheckState.Unchecked)
            dialog.accept()
        QTimer.singleShot(0, choose)
        window.ms2_files_action.trigger()
        assert calls[-1] == ([paths[0], paths[2]], {'precursor_mz': None})
        QTimer.singleShot(0, lambda: QApplication.activeModalWidget().reject())
        window.ms2_files_action.trigger()
        assert len(calls) == 2
        errors = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args: errors.append(args[-1]))
        def fail(*args, **kwargs):
            raise OSError('Missing sibling')
        monkeypatch.setattr('kmzviewer.gui.app.open_ms2', fail)
        window.ms2_blank_action.trigger()
        assert errors == ['Missing sibling']
        assert len(window.session.files) == 3
    finally:
        close(window)


@pytest.mark.parametrize('mode', ['stacked', 'overlay'])
def test_real_label_right_click_uses_owner_and_unrounded_mass(qapp, tmp_path, monkeypatch, mode):
    calls = []
    monkeypatch.setattr('kmzviewer.gui.app.open_ms2', lambda paths, **kw: calls.append((paths, kw)))
    window = MainWindow()
    window.resize(1512, 930)
    window.show()
    try:
        paths = [tmp_path / f'Run {i}.mzML' for i in range(2)]
        masses = [100.123456789, 300.987654321]
        frames = []
        for i, path in enumerate(paths):
            write_mzml(path, [dict(mz=[masses[i]], intensity=[50], rt=1)])
            assert window.load_paths_sync([str(path)]) == []
            source = list(window.session.files.values())[-1]
            frames.append(SpectrumFrame(source.file_id, source.name, source.color, source.name,
                                        np.array([masses[i]]), np.array([50.]), True))
        window.spectrum.show_frames(frames, 'Comparison')
        window.spectrum.set_mode(mode)
        QTest.qWait(40)
        # Navigator ownership deliberately differs from the target label.
        window.tree.setCurrentItem(window.tree.topLevelItem(0))
        label = next(item for item in window.spectrum._label_items if item.msms_target[0] == frames[1].file_id)
        scene_pos = label.textItem.mapToScene(label.textItem.boundingRect().center())
        pos = window.spectrum.view.mapFromScene(scene_pos)
        QTest.mouseClick(window.spectrum.view.viewport(), Qt.MouseButton.RightButton, pos=pos)
        wait_for(lambda: isinstance(QApplication.activePopupWidget(), QMenu))
        menu = QApplication.activePopupWidget()
        assert menu.actions()[0].text() == 'Open MS/MS in Kmz2Viewer'
        # A pending frame may update while the popup is open; the clicked target is fixed.
        frames[1].mz = np.array([400.])
        window.spectrum.show_frames(frames, 'Next scan')
        menu.actions()[0].trigger()
        menu.close()
        assert calls == [([str(paths[1])], {'precursor_mz': masses[1]})]
        assert window.spectrum._label_context_menu(QPointF(-100, -100)) is None
        window.spectrum.clear_scan('No spectra')
        assert window.spectrum._label_context_menu(scene_pos) is None
    finally:
        close(window)


def test_precursor_only_initialization_and_async_results(qapp, tmp_path):
    from kmz2viewer.gui.app import MainWindow as Ms2Window, _read_filter
    from kmz2viewer.filter import ScanFilter
    mass = 300.123456789
    path = tmp_path / 'All RTs Ω.mzML'
    write_mzml(path, [dict(id=f'scan-{i}', ms_level=2, rt=rt,
                          precursor_mz=mass * (1 + ppm / 1e6),
                          mz=[80], intensity=[10], polarity=polarity)
                      for i, (ppm, rt, polarity) in enumerate([
                          (-19.9, 1, 1), (19.9, 90, -1), (-20.1, 1, 1), (20.1, 90, 1)])])
    window = Ms2Window(precursor_mz=mass)
    blank = Ms2Window()
    try:
        spec, note = _read_filter(window)
        assert note == ''
        assert spec == ScanFilter(precursor_mz=mass)
        window.open_paths([str(path)])
        wait_for(lambda: str(path) in window.runtime.opened and window.runtime.job is None)
        assert {scan.scan_id for scan in window._matches} == {'scan-0', 'scan-1'}
        assert _read_filter(blank)[0] == ScanFilter()
        assert blank.library.scans == []
        window.clear_filters()
        assert len(window._matches) == 4
    finally:
        close(window)
        close(blank)
