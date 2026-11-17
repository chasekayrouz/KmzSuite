"""Interruption and cooperative cancellation never leave converter workers running (synthetic inputs).

Real SIGINTs are raised at the boundaries an asynchronous KeyboardInterrupt could
otherwise split: right after a pool worker starts (before the executor registers
it) and while the executor joins its workers.
"""
from concurrent.futures import ThreadPoolExecutor
import signal
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from kmzconvert_core import native, writer
from kmzconvert_core.mass_correction import Reference, estimate_mass_correction
from Tests.Convert.test_parallel_writer import acquisition, workers  # noqa: F401 (fixture)


def interrupt_after_first_worker_start(monkeypatch, started):
    """SIGINT immediately after the first converter worker thread starts."""
    original = threading.Thread.start
    def start(thread):
        original(thread)
        if thread.name.startswith('kmzconvert_') and not started.is_set():
            started.set()
            signal.raise_signal(signal.SIGINT)
    monkeypatch.setattr(threading.Thread, 'start', start)


def test_sigint_at_worker_registration_joins_every_worker(monkeypatch):
    started, release = threading.Event(), threading.Event()
    interrupt_after_first_worker_start(monkeypatch, started)
    threading.Timer(.2, release.set).start()
    def task(i):
        release.wait(5)
        return i
    with pytest.raises(KeyboardInterrupt):
        list(native.ordered(task, range(64), 4))
    assert started.is_set() and not workers()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


@pytest.mark.parametrize('profile', ['full', 'sparse'])
def test_sigint_at_worker_registration_during_conversion(acquisition, tmp_path, monkeypatch, profile):
    started, release, seen = threading.Event(), threading.Event(), []
    interrupt_after_first_worker_start(monkeypatch, started)
    threading.Timer(.2, release.set).start()
    original = native.NativeWiffReader.block
    def block(reader, scan):
        release.wait(5)
        seen.append(reader._mapping.closed)  # A worker outliving the reader would see it closed.
        return original(reader, scan)
    monkeypatch.setattr(native.NativeWiffReader, 'block', block)
    with native.NativeWiffReader(acquisition) as reader:
        with pytest.raises(KeyboardInterrupt):
            writer.write_mzml(reader, tmp_path / 'cancelled.mzML', profile=profile)
        assert not workers()
    assert started.is_set() and seen and not any(seen)
    assert not list(tmp_path.glob('*.mzML')) and not list(tmp_path.glob('.kmzconvert-*.tmp'))


def test_sigint_while_joining_workers_is_raised_after_the_join(monkeypatch):
    joining, started, alive = threading.Event(), threading.Event(), []
    original = ThreadPoolExecutor.shutdown
    def shutdown(pool, *args, **kwargs):
        joining.set()
        original(pool, *args, **kwargs)
        alive.extend(workers())
    monkeypatch.setattr(ThreadPoolExecutor, 'shutdown', shutdown)
    def task(i):
        if i == 0:
            return i
        if i == 1:
            started.set()
        joining.wait(5)
        if i == 1:
            signal.raise_signal(signal.SIGINT)  # Delivered to the main thread, now joining.
        return i
    results = native.ordered(task, range(8), 4)
    assert next(results) == 0
    # close() cancels queued tasks; under load task 1 must already be running.
    assert started.wait(5)
    with pytest.raises(KeyboardInterrupt):
        results.close()
    assert joining.is_set() and not alive and not workers()


def test_interruption_point_respects_custom_and_ignored_handlers(monkeypatch):
    received = []
    previous = signal.signal(signal.SIGINT, lambda *args: received.append(args))
    try:
        values = native.ordered(lambda i: (signal.raise_signal(signal.SIGINT) if i == 3 else None) or i, range(20), 2)
        assert list(values) == list(range(20))  # A custom handler is called once, after the workers stop.
        assert len(received) == 1 and not workers()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        assert list(native.ordered(lambda i: (signal.raise_signal(signal.SIGINT) if i == 3 else None) or i, range(9), 2)) == list(range(9))
    finally:
        signal.signal(signal.SIGINT, previous)


class Cancelled(Exception):
    pass


def test_mass_reference_decoding_checks_cancellation_before_each_sequential_decode():
    class Reader:
        scans = [SimpleNamespace(experiment=1, rt_seconds=60., native_id=str(i)) for i in range(10)]
        experiments = [SimpleNamespace(ms_level=1)]
        calls = 0
        def arrays(self, scan):
            self.calls += 1
            return np.array([99.9, 100., 100.1]), np.array([0., 10., 0.])
    reader = Reader()
    def check():
        if reader.calls:
            raise Cancelled
    with pytest.raises(Cancelled):
        estimate_mass_correction(reader, [Reference('r', 100., 1.)], cancel_check=check)
    assert reader.calls == 1


def test_mass_reference_decoding_checks_cancellation_per_threaded_scan(acquisition, monkeypatch):
    decoded, checks = [], []
    original = native.reader_arrays
    def counted(*args, **kwargs):
        decoded.append(None)
        return original(*args, **kwargs)
    monkeypatch.setattr(native, 'reader_arrays', counted)
    def check():
        checks.append(len(decoded))
        if len(decoded) >= 3:
            raise Cancelled
    with native.NativeWiffReader(acquisition) as reader:
        ms1 = sum(reader.experiments[s.experiment-1].ms_level == 1 for s in reader.scans)
        with pytest.raises(Cancelled):
            estimate_mass_correction(reader, [Reference('r', 100.2, .1)], cancel_check=check)
        assert not workers()
    # Bounded read-ahead (two tasks per worker) between the checks, never the whole window.
    assert len(decoded) <= 2 * native.workers(reader) + 3 < ms1
