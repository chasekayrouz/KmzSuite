"""Gated lifecycle diagnostics for development bundles; never enabled normally."""
from pathlib import Path
from threading import Event
import json
import sys
import time
import traceback

from PySide6.QtCore import QTimer
from kmzcore.runtime.synthetic import write_fixture


def run_lifecycle(app, window, key, destination, operation, action):
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'Synthetic lifecycle Ω.mzML'
    write_fixture(path)
    entered, release = Event(), Event()
    if operation == 'load':
        if key == 'ms1':
            import kmzviewer.gui.workers as workers
            name = 'load_ms1'
        else:
            import kmz2viewer.workers as workers
            name = 'index_ms2'
        original = getattr(workers, name)
        def held(*args, **kwargs):
            result = original(*args, **kwargs)
            entered.set()
            if not release.wait(20):
                raise TimeoutError('Diagnostic gate was not released')
            return result
        setattr(workers, name, held)
        window.open_paths([str(path)])
    else:
        assert key == 'ms1'
        assert window.load_paths_sync([str(path)]) == []
        from kmzviewer.spectra import SpectrumReader
        original = SpectrumReader.read
        def held(self, index):
            result = original(self, index)
            entered.set()
            if not release.wait(20):
                raise TimeoutError('Diagnostic gate was not released')
            return result
        SpectrumReader.read = held
        if operation == 'extract':
            window.extract_request('100', 10, 'all', False, list(window.session.files))
        else:
            window._spectrum_worker.request(1, str(path), 0)
    app.setQuitOnLastWindowClosed(False)
    start = time.monotonic()
    state = {'step': 0}
    timer = QTimer(window)
    timer.setInterval(10)

    def tick():
        try:
            if time.monotonic() - start > 40:
                raise TimeoutError(f'Lifecycle diagnostic timed out: {state}')
            if state['step'] == 0:
                if not entered.is_set():
                    return
                if action == 'quit':
                    app.quit()
                elif action == 'close':
                    window.close()
                elif action == 'clear':
                    window.clear_files()
                else:
                    window.runtime.cancel()
                assert not window.runtime.closed
                if operation != 'spectrum':
                    assert window.runtime.job is not None
                    assert window.runtime.job.worker._cancel.is_set()
                state['step'] = 1
            elif state['step'] == 1:
                # A later timer callback proves the GUI event loop ran while decode was held.
                assert not window.runtime.closed
                release.set()
                state['step'] = 2
            elif state['step'] == 2:
                if window.runtime.job is not None:
                    return
                if operation == 'load':
                    assert not (window.session.files if key == 'ms1' else window.library.scans)
                if operation == 'extract':
                    assert len(next(iter(window.session.files.values())).traces) == 1
                window.close()
                if not window.runtime.closed:
                    return
                if key == 'ms1':
                    assert not window._spectrum_thread.is_alive()
                    assert not window._spectrum_worker._readers
                timer.stop()
                (output / 'result.json').write_text(json.dumps({'ok': True, 'viewer': key,
                    'operation': operation, 'action': action, 'qt_platform': app.platformName(),
                    'frozen': bool(getattr(sys, 'frozen', False)),
                    'checks': ['decode gate held', 'cancellation observed', 'GUI heartbeat', 'late result discarded', 'actual thread completion', 'reader cleanup']}, indent=2))
                app.exit(0)
        except Exception:
            release.set()
            timer.stop()
            (output / 'result.json').write_text(json.dumps({'ok': False, 'traceback': traceback.format_exc()}, indent=2))
            window.close()
            cleanup = QTimer(window)
            cleanup.timeout.connect(lambda: app.exit(1) if window.runtime.closed else None)
            cleanup.start(20)
            window._diagnostic_cleanup = cleanup
    timer.timeout.connect(tick)
    timer.start()
    window._lifecycle_timer = timer
