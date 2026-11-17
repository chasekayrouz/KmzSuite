"""Applications collect reference cycles only on the GUI thread (kmzcore.runtime.qt)."""
import gc
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from kmzcore.runtime import qt
from kmzcore.runtime.qt import MainThreadCollector


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_worker_allocations_are_collected_later_on_the_gui_thread(qapp):
    collections = []

    def record(phase, info):
        if phase == "stop":
            collections.append(threading.current_thread() is threading.main_thread())
    enabled = gc.isenabled()
    gc.collect()
    gc.enable()  # As in an application before the collector starts.
    gc.callbacks.append(record)
    collector = MainThreadCollector(None, interval_ms=10)
    try:
        assert not gc.isenabled()
        kept = []
        worker = threading.Thread(target=lambda: kept.extend([] for _ in range(20 * gc.get_threshold()[0])))
        worker.start()
        worker.join()
        assert collections == []  # Automatic collection would have run on the worker.
        deadline = time.monotonic() + 5
        while not collections and time.monotonic() < deadline:
            qapp.processEvents()
        assert collections and all(collections)
    finally:
        collector.timer.stop()
        collector.deleteLater()
        gc.callbacks.remove(record)
        gc.enable() if enabled else gc.disable()


@pytest.mark.parametrize("counts, generation", [
    ((2000, 50, 50), None), ((2001, 10, 50), 0), ((2001, 11, 10), 1), ((2001, 11, 11), 2)])
def test_collector_applies_the_interpreter_thresholds(qapp, monkeypatch, counts, generation):
    collected = []
    collector = MainThreadCollector(None, interval_ms=60_000)
    try:
        collector.threshold = (2000, 10, 10)
        monkeypatch.setattr(qt.gc, "get_count", lambda: counts)
        monkeypatch.setattr(qt.gc, "collect", collected.append)
        collector.check()
        assert collected == ([] if generation is None else [generation])
    finally:
        collector.timer.stop()
        collector.deleteLater()
