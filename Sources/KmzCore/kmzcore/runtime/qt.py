"""Qt document delivery and cooperative lifetime ownership for the two windows."""
import gc
from pathlib import Path
import os

from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import QApplication, QMessageBox

from kmzcore.runtime import IDENTITIES, resource_path
from kmzcore.version import __version__, about_text


def local_path(value) -> str:
    if isinstance(value, QUrl):
        if not value.isLocalFile():
            raise ValueError('Only local mzML files can be opened.')
        value = value.toLocalFile()
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != '.mzml':
        raise ValueError(f'Not a local mzML file: {path.name}')
    return str(path)


def path_identity(path):
    """Also deduplicate hard links and macOS case/Unicode aliases by file identity."""
    try:
        info = os.stat(path)
        return info.st_dev, info.st_ino
    except OSError:
        return os.path.realpath(path)


class MainThreadCollector(QObject):
    """Collect reference cycles only on the GUI thread (pyqtgraph's GarbageCollector policy).

    Python's collector otherwise runs on whichever thread allocates, so a worker could
    destroy GUI-thread Qt objects kept alive by a cycle, such as a QTimer that later
    fires on freed memory. Automatic collection is disabled; the interpreter's
    generation thresholds are checked here on every tick instead. CPython's
    long-lived heuristic is not available here, so under sustained allocation a full
    collection can run about every two minutes.
    """

    def __init__(self, parent, interval_ms=1000):
        super().__init__(parent)
        self.threshold = gc.get_threshold()
        gc.disable()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check)
        self.timer.start(interval_ms)

    @Slot()
    def check(self):
        counts, limits = gc.get_count(), self.threshold
        if counts[0] > limits[0]:
            generation = 0
            if counts[1] > limits[1]:
                generation = 2 if counts[2] > limits[2] else 1
            gc.collect(generation)


class DocumentApplication(QApplication):
    def __init__(self, argv, identity):
        self.window = None
        self.pending = []
        super().__init__(argv)
        self.collector = MainThreadCollector(self)
        from kmzcore.theme import apply_theme
        apply_theme(self)
        self.setOrganizationName('KmzSuite')
        self.setApplicationName(identity.bundle_id)
        self.setApplicationDisplayName(identity.name)
        self.setApplicationVersion(__version__)

    def attach(self, window, paths=()):
        self.window = window
        window.open_paths([*self.pending, *paths])
        self.pending.clear()

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            value = event.url() if not event.url().isEmpty() else event.file()
            if self.window is None:
                self.pending.append(value)
            else:
                self.window.open_paths([value])
            return True
        if event.type() == QEvent.Type.Quit and self.window is not None:
            if not self.window.runtime.closed:
                self.window.close()
                return True
        return super().event(event)


class Job(QObject):
    """Own worker and thread until wait(0) proves all execution has ended.

    Cancellation is a direct thread-safe Event.set(), never a queued worker slot.
    Worker results are delivered to these GUI-thread slots before reaping.
    """
    ended = Signal()

    def __init__(self, runtime, worker, success, *, stack_size=0):
        super().__init__(runtime)
        self.runtime = runtime
        self.worker = worker
        self.success = success
        self.invalidated = False
        self.reaped = False
        self.thread = QThread(self)
        if stack_size:
            self.thread.setStackSize(stack_size)
        worker.moveToThread(self.thread)
        # A functor on the QThread runs the worker, which therefore receives no connection from
        # a Python-wrapper sender: destroying it on its thread never asks PySide for the GIL
        # under Qt's connection lock, and it never takes that lock itself while holding the GIL.
        self.thread.started.connect(lambda: worker.run(), Qt.ConnectionType.DirectConnection)
        worker.progress.connect(self.progress)
        worker.succeeded.connect(self.succeeded)
        worker.failed.connect(self.failed)
        worker.cancelled.connect(self.cancelled)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self.reap)
        self.timer = QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.reap)

    def cancel(self):
        self.invalidated = True
        # The C++ worker may already be deleted; cancel only touches its Python Event.
        self.worker.cancel()

    def accepts(self):
        return not self.invalidated and not self.runtime.closing

    @Slot(int, int, str)
    def progress(self, done, total, message):
        if self.accepts():
            self.runtime.window._on_progress(done, total, message)

    @Slot(object)
    def succeeded(self, result):
        if self.accepts():
            self.success(result)

    @Slot(str)
    def failed(self, message):
        if self.accepts():
            self.runtime.window._on_failed(message)

    @Slot()
    def cancelled(self):
        if not self.runtime.closing:
            self.runtime.window.statusBar().showMessage('Cancelled.')

    @Slot()
    def reap(self):
        if not self.thread.wait(0):
            self.timer.start()
            return
        self.timer.stop()
        if self.reaped:
            return
        self.reaped = True
        # Drop Python-wrapper sender connections while holding the GIL, so deleting
        # this Job never asks PySide for the GIL under Qt's connection lock.
        self.timer.timeout.disconnect(self.reap)
        self.thread.finished.disconnect(self.reap)
        self.ended.emit()
        self.deleteLater()


class WindowRuntime(QObject):
    def __init__(self, window, key):
        super().__init__(window)
        self.window = window
        self.identity = IDENTITIES[key]
        self.closing = False
        self.closed = False
        self.job = None
        self.pending = []
        self.inflight = set()
        self.opened = set()
        if os.environ.get('KMZ_LOCAL_STATE_ROOT'):
            QSettings.setDefaultFormat(QSettings.Format.IniFormat)
            QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, os.environ['KMZ_LOCAL_STATE_ROOT'])
        self.settings = QSettings('KmzSuite', self.identity.bundle_id)
        self.shutdown_timer = QTimer(self)
        self.shutdown_timer.setInterval(20)
        self.shutdown_timer.timeout.connect(window.close)
        self.drain_timer = QTimer(self)
        self.drain_timer.setSingleShot(True)
        self.drain_timer.timeout.connect(self.drain)
        self.messages = []  # Keep non-modal error boxes alive.
        window.setWindowTitle(self.identity.name)
        self._menus()

    def _menus(self):
        menu = self.window.menuBar().addMenu('Help')
        manual = QAction('User Manual', self)
        manual.setShortcut(QKeySequence.StandardKey.HelpContents)
        manual.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(resource_path(self.identity.manual)))))
        menu.addAction(manual)
        notices = QAction('Dependency Notices', self)
        notices.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(resource_path('resources/THIRD_PARTY_NOTICES.txt')))))
        menu.addAction(notices)
        about = QAction('About ' + self.identity.name, self)
        about.setMenuRole(QAction.MenuRole.AboutRole)
        about.triggered.connect(lambda: QMessageBox.about(self.window, self.identity.name,
                                                          about_text(self.identity.name)))
        menu.addAction(about)

    @property
    def directory(self):
        return self.settings.value('lastDirectory', str(Path.home()))

    def request(self, values):
        if self.closing:
            return
        errors = []
        known = self.opened | self.inflight | set(self.pending)
        # Include files loaded through the synchronous test/development helpers.
        if hasattr(self.window, 'session'):
            known.update(os.path.realpath(f.path) for f in self.window.session.files.values())
        else:
            known.update(os.path.realpath(s.path) for s in self.window.library.scans)
        known_keys = {path_identity(path) for path in known}
        for value in values:
            try:
                path = local_path(value)
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(str(exc))
                continue
            identity = path_identity(path)
            if identity not in known_keys:
                self.pending.append(path)
                known_keys.add(identity)
                self.settings.setValue('lastDirectory', str(Path(path).parent))
        if errors:
            box = QMessageBox(QMessageBox.Icon.Warning, 'Could not open files', '\n'.join(errors), parent=self.window)
            self.messages.append(box)
            box.finished.connect(lambda _result, box=box: self.messages.remove(box))
            box.open()
        self.drain_timer.start(0)

    @Slot()
    def drain(self):
        if self.closing or self.job is not None or self.window._busy or not self.pending:
            return
        paths, self.pending = self.pending, []
        self.inflight = set(paths)
        self.window._open_paths_now(paths)

    def start(self, worker, success, *, stack_size=0):
        if self.closing or self.job is not None:
            raise RuntimeError('A background job is already running or the window is closing.')
        self.job = Job(self, worker, success, stack_size=stack_size)
        self.window._worker = worker
        self.window._thread = self.job.thread
        self.job.ended.connect(self.finished)
        self.window._set_busy(True, 'Working…')
        self.job.thread.start()

    @Slot()
    def finished(self):
        self.job = None
        self.inflight.clear()
        self.window._worker = None
        self.window._thread = None
        self.window._set_busy(False)
        self.drain_timer.start(0)

    def cancel(self, clear_queue=True):
        if clear_queue:
            self.pending.clear()
        if self.job is not None:
            self.job.cancel()
            self.window.statusBar().showMessage('Cancelling… waiting for the current decode to return.')

    def close_event(self, event):
        if not self.closing:
            self.closing = True
            self.cancel()
            if hasattr(self.window, '_spectrum_worker'):
                self.window._drop_pending_spectrum()
                self.window._spectrum_worker.stop()
            self.window.statusBar().showMessage('Closing… waiting for background readers to finish.')
            self.window.setEnabled(False)
        spectrum = getattr(self.window, '_spectrum_thread', None)
        if self.job is not None or (spectrum is not None and spectrum.is_alive()):
            event.ignore()
            self.shutdown_timer.start()
            return
        if spectrum is not None:
            spectrum.join(timeout=0)
        self.shutdown_timer.stop()
        self.closed = True
        event.accept()
