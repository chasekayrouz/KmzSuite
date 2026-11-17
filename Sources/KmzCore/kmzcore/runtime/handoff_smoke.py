"""Opt-in cross-app diagnostics with synthetic data and independent live sessions."""
from pathlib import Path
import json
import os
import sys
import time
import traceback

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

from kmzcore.runtime import handoff
from kmzcore.runtime.synthetic import write_fixture


def _ready_for_peak_menu(app, window, output):
    # LaunchServices can activate a newly launched child while Cocoa is showing
    # the parent's popup. Wait for both sessions, then regain the foreground on
    # a separate event-loop turn, as an interactive right-click would require.
    if not all((output / name / 'ready.json').exists() for name in ('blank', 'files')):
        return False
    if app.applicationState() != Qt.ApplicationState.ApplicationActive or not window.isActiveWindow():
        window.raise_()
        window.activateWindow()
        return False
    return True


def run_handoff(app, window, key, destination, role):
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Diagnostic-only: use bundled vocabulary to avoid unrelated network startup.
    from psims.controlled_vocabulary.controlled_vocabulary import obo_cache
    obo_cache.use_remote = False
    app.setQuitOnLastWindowClosed(False)
    timer = QTimer(window)
    timer.setInterval(30)
    started = time.monotonic()
    state = {'step': 0}
    cases = ('blank', 'files', 'peak')
    if role == 'parent':
        assert key == 'ms1'
        paths = [str(output / name) for name in ('Source Ω one.mzML', 'Source two.mzML')]
        for path in paths:
            write_fixture(path)
        window.open_paths(paths)
    else:
        assert key == 'ms2'
        expected = json.loads((output / 'expected.json').read_text())

    def launch_child(name, paths, mass, callback):
        folder = output / name
        folder.mkdir()
        (folder / 'expected.json').write_text(json.dumps({'paths': paths, 'mass': mass}))
        original = handoff.ms2_command
        def diagnostic_command(*args, **kwargs):
            command = original(*args, **kwargs)
            offset = command.index('--')
            command[offset:offset] = ['--smoke-output', str(folder), '--smoke-handoff', 'child']
            return command
        handoff.ms2_command = diagnostic_command
        try:
            callback()
        finally:
            handoff.ms2_command = original

    def finish(result):
        timer.stop()
        (output / 'result.json').write_text(json.dumps(result | {
            'frozen': bool(getattr(sys, 'frozen', False)), 'pid': os.getpid(),
            'qt_platform': app.platformName(), 'seconds': round(time.monotonic() - started, 3)}, indent=2))
        window.close()
        cleanup = QTimer(window)
        cleanup.timeout.connect(lambda: app.exit(0 if result['ok'] else 1) if window.runtime.closed else None)
        cleanup.start(20)
        window._handoff_cleanup = cleanup

    def verify_child():
        from kmz2viewer.gui.app import _read_filter
        from kmz2viewer.filter import ScanFilter, apply_filter
        assert window.runtime.opened == set(expected['paths'])
        assert _read_filter(window) == (ScanFilter(precursor_mz=expected['mass']), '')
        assert window._matches == apply_filter(window.library.scans, ScanFilter(precursor_mz=expected['mass']))
        assert {scan.path for scan in window.library.scans} == set(expected['paths'])

    def tick():
        try:
            if time.monotonic() - started > 90:
                raise TimeoutError(f'Handoff timed out: {role} {state}')
            if role == 'child':
                if window.runtime.job is not None or window.runtime.pending:
                    return
                verify_child()
                if state['step'] == 0:
                    window.grab().save(str(output / 'session.png'))
                    (output / 'ready.json').write_text(json.dumps({'pid': os.getpid(), 'ok': True}))
                    state['step'] = 1
                if (output / 'release').exists():
                    finish({'ok': True, 'checks': ['exact source files', 'precursor-only filter',
                                                  'independent live session', 'filter still unchanged']})
                return
            if state['step'] == 0:
                if window.runtime.opened != set(paths) or window.runtime.job is not None:
                    return
                state['step'] = 1
                launch_child('blank', [], None, window.ms2_blank_action.trigger)
                def accept_files():
                    dialog = QApplication.activeModalWidget()
                    assert dialog.checked_file_ids() == list(window.session.files)
                    dialog.accept()
                QTimer.singleShot(0, accept_files)
                launch_child('files', paths, None, window.ms2_files_action.trigger)
                loaded = list(window.session.files.values())[1]
                window.inspect_rt(loaded.traces[0].trace_id, 1.)
            elif state['step'] == 1:
                if window._inflight_comparison is not None or not window.spectrum._label_items:
                    return
                if not _ready_for_peak_menu(app, window, output):
                    return
                # A real right-click follows presentation. Read scene coordinates
                # only after Qt has applied the latest frame's view transform.
                window.spectrum.view.viewport().repaint()
                label = window.spectrum._label_items[0]
                fid, mass = label.msms_target
                pos = label.textItem.mapToScene(label.textItem.boundingRect().center())
                menu = window.spectrum._label_context_menu(pos)
                assert menu is not None
                menu.popup(window.spectrum.view.mapToGlobal(window.spectrum.view.mapFromScene(pos)))
                app.processEvents()
                window.grab().save(str(output / 'source.png'))
                launch_child('peak', [window.session.files[fid].path], mass, menu.actions()[0].trigger)
                menu.close()
                menu.deleteLater()
                state['step'] = 2
            elif state['step'] == 2:
                for name in cases:
                    result = output / name / 'result.json'
                    if result.exists():
                        assert json.loads(result.read_text())['ok'], result.read_text()
                if not all((output / name / 'ready.json').exists() for name in cases):
                    return
                pids = {json.loads((output / name / 'ready.json').read_text())['pid'] for name in cases}
                assert len(pids) == 3 and os.getpid() not in pids
                for name in cases:
                    (output / name / 'release').touch()
                state['step'] = 3
            elif all((output / name / 'result.json').exists() for name in cases):
                assert all(json.loads((output / name / 'result.json').read_text())['ok'] for name in cases)
                assert window.runtime.opened == set(paths)
                finish({'ok': True, 'checks': ['blank action', 'multi-file picker', 'label source and mass',
                                              'three concurrent independent Kmz2Viewer processes',
                                              'source session unchanged']})
        except Exception:
            for name in cases if role == 'parent' else ():
                if (output / name).is_dir():
                    (output / name / 'release').touch()
            finish({'ok': False, 'traceback': traceback.format_exc()})
    timer.timeout.connect(tick)
    timer.start()
    window._handoff_timer = timer
