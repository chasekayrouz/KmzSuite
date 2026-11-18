"""Lifecycle regressions use explicit gates, never timing-dependent sleep races."""
from pathlib import Path
from threading import Event

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
from PySide6.QtGui import QFileOpenEvent
from PySide6.QtWidgets import QApplication

from kmzcore.cancel import Cancelled
from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setQuitOnLastWindowClosed(False)
    return instance


def until(predicate, timeout=5000):
    if predicate():
        return
    loop = QEventLoop()
    tick = QTimer()
    tick.setInterval(1)
    tick.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    tick.start()
    deadline.start(timeout)
    loop.exec()
    tick.stop()
    assert predicate(), 'Event-loop deadline expired'


def fixture_file(path):
    write_mzml(path, [dict(id='source=7', ms_level=1, rt=1, mz=[100], intensity=[5], tic=5),
                      dict(id='source=9', ms_level=2, rt=2, mz=[100], intensity=[7], precursor_mz=200, charge=1)])
    return str(path)


@pytest.mark.parametrize('key', ['ms1', 'ms2'])
@pytest.mark.parametrize('action', ['cancel', 'close', 'clear'])
def test_live_load_lifetime_and_late_result(app, tmp_path, monkeypatch, key, action):
    if key == 'ms1':
        from kmzviewer.gui.app import MainWindow
        import kmzviewer.gui.workers as workers
        function = 'load_ms1'
    else:
        from kmz2viewer.gui.app import MainWindow
        import kmz2viewer.workers as workers
        function = 'index_ms2'
    original = getattr(workers, function)
    entered, release = Event(), Event()

    def held(*args, **kwargs):
        # Mimics one non-interruptible decode, including success racing cancellation.
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(workers, function, held)
    window = MainWindow()
    window.show()
    path = fixture_file(tmp_path / 'space Ω.mzML')
    try:
        window.open_paths([path])
        until(entered.is_set)
        job = window.runtime.job
        assert job.thread.isRunning()
        if action == 'close':
            window.close()
            assert not window.runtime.closed
            assert window.isVisible()
        elif action == 'clear' and key == 'ms2':
            window.clear_files()
        else:
            window.runtime.cancel()
        assert job.worker._cancel.is_set()
        assert window.runtime.job is job
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        until(lambda: bool(heartbeat))
        release.set()
        until(lambda: window.runtime.job is None)
        assert not (window.session.files if key == 'ms1' else window.library.scans)
    finally:
        release.set()
        window.close()
        until(lambda: window.runtime.closed)


@pytest.mark.parametrize('key', ['ms1', 'ms2'])
def test_documents_queue_deduplicate_and_retry(app, tmp_path, key):
    if key == 'ms1':
        from kmzviewer.gui.app import MainWindow
    else:
        from kmz2viewer.gui.app import MainWindow
    window = MainWindow()
    a = fixture_file(tmp_path / 'A Ω.mzML')
    b = fixture_file(tmp_path / 'B space.mzML')
    alias = tmp_path / 'alias.mzML'
    alias.symlink_to(a)
    hardlink = tmp_path / 'hardlink.mzML'
    hardlink.hardlink_to(a)
    try:
        window._set_busy(True)
        window.open_paths([a, QUrl.fromLocalFile(a), str(alias), str(hardlink), b])
        app.processEvents()
        assert window.runtime.pending == [a, b]
        assert window.runtime.job is None
        window._set_busy(False)
        window.runtime.drain()
        window.open_paths([a, b])
        assert window.runtime.pending == []
        until(lambda: window.runtime.job is None)
        assert window.runtime.opened == {a, b}
        window.open_paths([a, b])
        app.processEvents()
        assert window.runtime.job is None
    finally:
        window.close()
        until(lambda: window.runtime.closed)


@pytest.mark.parametrize('activation', ['window-before-application', 'other-window'])
def test_full_view_shortcut_check_follows_the_shortcut_map(app, tmp_path, monkeypatch, activation):
    """macOS makes a window key before reporting the application active; F already works then."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget
    from kmzcore.runtime.smoke import check_full_view_shortcut
    from kmzcore.runtime.synthetic import write_fixture
    from kmzviewer.gui.app import MainWindow
    path = tmp_path / 'one.mzML'
    write_fixture(path)
    window, other = MainWindow(), QWidget()
    try:
        window.show()
        assert not window.load_paths_sync([str(path)])
        until(lambda: bool(window.plots._plots()))
        target = other if activation == 'other-window' else window
        target.show()
        target.activateWindow()
        until(lambda: QApplication.activeWindow() is target)
        if activation == 'window-before-application':
            monkeypatch.setattr(app, 'applicationState', lambda: Qt.ApplicationState.ApplicationInactive)
        expected = 'F active full view' if target is window else 'F inactive window ignored'
        assert check_full_view_shortcut(app, window, window.plots.view, window.plots._plots()[0], (1.1, 1.2)) == expected
    finally:
        other.close()
        window.close()
        until(lambda: window.runtime.closed)


def test_ms2_cancellation_inside_stream_and_fallback(tmp_path, monkeypatch):
    import kmz2viewer.index as index
    from pyteomics import mzml
    path = fixture_file(tmp_path / 'fallback.mzML')
    cancel = Event()
    original = index._decode_arrays
    def decode(block):
        result = original(block)
        cancel.set()
        return result
    monkeypatch.setattr(index, '_decode_arrays', decode)
    with pytest.raises(Cancelled):
        index.index_ms2(path, is_cancelled=cancel.is_set)
    cancel.clear()
    monkeypatch.setattr(index, '_decode_arrays', lambda block: None)
    closed = []
    class Reader:
        def __init__(self, path, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): closed.append(True)
        def get_by_index(self, index):
            cancel.set()
            return {'m/z array': [100], 'intensity array': [7]}
    monkeypatch.setattr(mzml, 'MzML', Reader)
    with pytest.raises(Cancelled):
        index.index_ms2(path, is_cancelled=cancel.is_set)
    assert closed == [True]


@pytest.mark.parametrize('operation', ['extract', 'spectrum'])
def test_close_during_ms1_decode_keeps_reader_alive(app, tmp_path, monkeypatch, operation):
    from kmzviewer.gui.app import MainWindow
    from kmzviewer.spectra import SpectrumReader
    original = SpectrumReader.read
    entered, release, closed = Event(), Event(), Event()
    original_close = SpectrumReader.close
    def read(self, index):
        import numpy as np
        result = (np.array([100.]), np.array([5.]))
        entered.set()
        assert release.wait(5)
        return result
    def close(self):
        original_close(self)
        closed.set()
    monkeypatch.setattr(SpectrumReader, 'read', read)
    monkeypatch.setattr(SpectrumReader, 'close', close)
    window = MainWindow()
    path = fixture_file(tmp_path / 'decode.mzML')
    try:
        assert window.load_paths_sync([path]) == []
        if operation == 'extract':
            window.extract_request('100', 10, 'all', False, list(window.session.files))
        else:
            window._spectrum_worker.request(1, path, 0)
        until(entered.is_set)
        window.close()
        assert not window.runtime.closed
        assert not closed.is_set()
        app.processEvents()
        release.set()
        until(lambda: window.runtime.closed)
        assert closed.is_set()
        assert not window._spectrum_thread.is_alive()
        assert window.runtime.job is None
    finally:
        release.set()
        window.close()
        until(lambda: window.runtime.closed)


def test_failed_worker_is_reaped_and_queue_continues(app, tmp_path, monkeypatch):
    from kmz2viewer.gui.app import MainWindow
    from PySide6.QtWidgets import QMessageBox
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    bad = tmp_path / 'bad.mzML'
    bad.write_text('invalid')
    window = MainWindow()
    try:
        window.open_paths([str(bad)])
        window.runtime.drain()
        good = fixture_file(tmp_path / 'good.mzML')
        window.open_paths([good])
        until(lambda: good in window.runtime.opened and window.runtime.job is None)
        assert errors and len(window.library.scans) == 1
    finally:
        window.close()
        until(lambda: window.runtime.closed)


@pytest.mark.parametrize('key', ['ms1', 'ms2'])
def test_worker_teardown_never_waits_for_the_gil_under_a_connection_lock(tmp_path, key):
    from Tests.worker_teardown import run_owner
    result = run_owner(key, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('key', ['ms1', 'ms2'])
def test_finishing_worker_never_takes_a_connection_lock_while_the_gui_destroys_objects(tmp_path, key):
    from Tests.worker_teardown import MIRROR, run_owner
    result = run_owner(key, tmp_path, MIRROR)
    assert result.returncode == 0, result.stdout + result.stderr


def test_allocating_worker_never_collects_gui_objects_held_by_a_cycle(app):
    import gc
    import threading
    import weakref
    from PySide6.QtCore import QObject
    item = QObject()
    item.cycle = item  # Only the cyclic collector can free it.
    ref = weakref.ref(item)
    del item
    worker = threading.Thread(target=lambda: [[] for _ in range(100_000)])  # Far past the GC threshold.
    worker.start()
    worker.join()
    assert ref() is not None, 'A worker thread destroyed a GUI-thread Qt object'
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize('key,operation', [('ms1', 'load'), ('ms2', 'load'), ('ms1', 'extract'), ('ms1', 'spectrum')])
def test_application_quit_waits_for_running_work_in_separate_process(tmp_path, key, operation):
    """A real QApplication must receive Quit; the suite's plain app cannot substitute."""
    import subprocess
    import sys
    script = r'''
import sys
from pathlib import Path
from threading import Event
from PySide6.QtCore import QEvent, QTimer
from kmzcore.runtime import IDENTITIES
from kmzcore.runtime.qt import DocumentApplication
from Tests.Viewer.test_runtime import fixture_file
key, operation, folder = sys.argv[1:]
app = DocumentApplication([], IDENTITIES[key])
import gc
assert app.collector.timer.isActive() and not gc.isenabled()  # Cycles are collected on this thread only.
entered, release = Event(), Event()
if key == 'ms1':
    from kmzviewer.gui.app import MainWindow
    import kmzviewer.gui.workers as workers
else:
    from kmz2viewer.gui.app import MainWindow
    import kmz2viewer.workers as workers
if operation == 'load':
    name = 'load_ms1' if key == 'ms1' else 'index_ms2'
    original = getattr(workers, name)
    def held(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result
    setattr(workers, name, held)
else:
    import numpy as np
    from kmzviewer.spectra import SpectrumReader
    def read(self, index):
        entered.set()
        assert release.wait(10)
        return np.array([100.]), np.array([5.])
    SpectrumReader.read = read
window = MainWindow()
window.show()
app.attach(window)
path = fixture_file(Path(folder) / 'quit.mzML')
if operation == 'load':
    window.open_paths([path])
else:
    window.load_paths_sync([path])
    if operation == 'extract':
        window.extract_request('100', 10, 'all', False, list(window.session.files))
    else:
        window._spectrum_worker.request(1, path, 0)
observed = []
def tick():
    if entered.is_set() and not observed:
        app.quit()
        observed.append(window.runtime.closing and not window.runtime.closed)
        release.set()
timer = QTimer()
timer.timeout.connect(tick)
timer.start(1)
QTimer.singleShot(12000, app.quit)
app.exec()
assert observed == [True]
assert window.runtime.closed and window.runtime.job is None
if key == 'ms1': assert not window._spectrum_thread.is_alive()
'''
    result = subprocess.run([sys.executable, '-c', script, key, operation, str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('module,key', [('kmzviewer', 'ms1'), ('kmz2viewer', 'ms2')])
def test_module_launch_arguments_with_spaces_unicode_and_invalid_file(tmp_path, module, key):
    import subprocess
    import sys
    path = fixture_file(tmp_path / 'CLI Ω sample.mzML')
    script = r'''
import runpy, sys
from pathlib import Path
from PySide6.QtCore import QTimer
from kmzcore.runtime.qt import DocumentApplication
module, path = sys.argv[1:]
original = DocumentApplication.exec
observed = []
def execute(self):
    timer = QTimer(self)
    def tick():
        if self.window.runtime.job is None and str(Path(path).resolve()) in self.window.runtime.opened:
            if not observed:
                assert len(self.window.runtime.messages) == 1
                for box in list(self.window.runtime.messages):
                    box.accept()
                observed.append(True)
            self.window.close()
    timer.timeout.connect(tick)
    timer.start(10)
    QTimer.singleShot(8000, lambda: self.exit(2))
    result = original()
    assert observed and self.window.runtime.closed
    return result
DocumentApplication.exec = execute
sys.argv = [module, path, path, path + '.missing']
runpy.run_module(module, run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script, module, path], capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, result.stdout + result.stderr
