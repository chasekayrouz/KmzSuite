"""Reproduce the worker-teardown lock-order deadlock deterministically in a fresh process.

A worker destroyed on its QThread removes its incoming connections under Qt's
address-hashed signal-slot mutex. If one comes from a PySide-created sender such as
the QThread, PySide's disconnectNotify then waits for the GIL. The GUI thread holds
the GIL and keeps locking that same mutex until the worker thread has finished; the
worker starts only once that loop runs. An owner that leaves the connection in place
therefore deadlocks on the first job.

MIRROR reproduces the reverse order: the GUI event loop (GIL released) destroys an object
whose incoming connection comes from a fresh PySide sender, so it holds that object's
mutex and waits for the GIL, while the worker, holding the GIL, finishes. The object
shares the mutex of the worker, its QThread or a receiver of the worker's signals, and
the worker leaves run() only once that destruction has started. An owner whose finishing
worker takes a connection lock (e.g. disconnects QThread.started) deadlocks.
"""
import os
import subprocess
import sys

SCRIPT = r'''
import faulthandler, sys
from threading import Event
import shiboken6
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

owner, folder = sys.argv[1:]
faulthandler.dump_traceback_later(20, exit=True)
app = QApplication([])
gate = Event()
if owner == 'comet':
    from kmzcomet_app.app import MainWindow
    from kmzcomet_app.workers import Worker
    window = MainWindow(history=folder)
    make = lambda: Worker(lambda worker: gate.wait())
    start = lambda worker: window._start(worker, lambda result: None, 'test')
    thread = lambda: window._thread
    idle = lambda: window._thread is None
else:
    if owner == 'ms1':
        from kmzviewer.gui.app import MainWindow
        from kmzviewer.gui.workers import IndexWorker as Loader
    else:
        from kmz2viewer.gui.app import MainWindow
        from kmz2viewer.workers import LoadWorker as Loader
    class Gated(Loader):
        def run(self):
            gate.wait()
            super().run()
    window = MainWindow()
    make = lambda: Gated([])
    start = lambda worker: window.runtime.start(worker, lambda result: None)
    thread = lambda: window.runtime.job.thread
    idle = lambda: window.runtime.job is None
spare = [QTimer() for _ in range(2000)]
sink = QObject()
lock = lambda item: (shiboken6.getCppPointer(item)[0] & 0xffffffff) % 131  # Qt: uint(pointer) % pool size
for _ in range(20):
    worker = make()
    same = [item for item in spare if lock(item) == lock(worker)][:4]
    start(worker)
    running = thread()
    gate.set()
    while not running.isFinished():
        for item in same:
            QObject.disconnect(item.timeout.connect(sink.deleteLater))
    gate.clear()
    while not idle():
        app.processEvents()
window.close()
'''


MIRROR = r'''
import faulthandler, sys
from threading import Event
import shiboken6
from PySide6.QtCore import QEventLoop, QObject, QTimer, Slot
from PySide6.QtWidgets import QApplication

owner, folder = sys.argv[1:]
faulthandler.dump_traceback_later(20, exit=True)
app = QApplication([])
gate = Event()
flag = QTimer()  # Started by the destruction through a native connection (no GIL).
def spin():
    gate.wait()
    while not flag.isActive():  # Hold the GIL until the GUI thread is destroying the object.
        pass
class Sink(QObject):
    @Slot(object)
    def accept(self, result):
        pass
sink = Sink()
if owner == 'comet':
    from kmzcomet_app.app import MainWindow
    from kmzcomet_app.workers import Worker
    window = MainWindow(history=folder)
    make = lambda: Worker(lambda worker: spin())
    start = lambda worker: window._start(worker, sink.accept, 'test')
    job = lambda: (window._thread, [window, window.statusBar(), sink])
    idle = lambda: window._thread is None
else:
    if owner == 'ms1':
        from kmzviewer.gui.app import MainWindow
        from kmzviewer.gui.workers import IndexWorker as Loader
    else:
        from kmz2viewer.gui.app import MainWindow
        from kmz2viewer.workers import LoadWorker as Loader
    class Spinning(Loader):
        def run(self):
            spin()
            super().run()
    window = MainWindow()
    make = lambda: Spinning([])
    start = lambda worker: window.runtime.start(worker, lambda result: None)
    job = lambda: (window.runtime.job.thread, [window.runtime.job])
    idle = lambda: window.runtime.job is None
lock = lambda item: (shiboken6.getCppPointer(item)[0] & 0xffffffff) % 131  # Qt: uint(pointer) % pool size
spare = [QObject() for _ in range(3000)]
for index in range(20):
    worker = make()
    start(worker)
    thread, receivers = job()
    targets = [worker, thread, *receivers]
    target = targets[index % len(targets)]
    doomed = next(item for item in spare if lock(item) == lock(target))
    spare.remove(doomed)
    sender = QTimer()  # Fresh wrapper: its first disconnectNotify asks for the GIL.
    sender.timeout.connect(doomed.deleteLater)
    doomed.destroyed.connect(flag.start)
    flag.stop()
    doomed.deleteLater()
    del doomed
    gate.set()
    loop, tick = QEventLoop(), QTimer()  # Deferred deletion runs in a nested loop, as in until().
    tick.timeout.connect(lambda: loop.quit() if idle() else None)
    tick.start(1)
    loop.exec()
    tick.stop()
    gate.clear()
window.close()
'''


def run_owner(owner, folder, script=SCRIPT):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-c', script, owner, str(folder)], env=env,
                          capture_output=True, text=True, timeout=40)
