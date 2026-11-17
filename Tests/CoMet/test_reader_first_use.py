"""UnitMzML is defined on the first mzML read, once even for simultaneous readers."""
import builtins
from concurrent.futures import ThreadPoolExecutor
import threading

from kmzcomet import reader


def test_simultaneous_first_reads_define_one_reader_class(monkeypatch):
    monkeypatch.delitem(vars(reader), "UnitMzML", raising=False)
    original = builtins.__import__
    importers = []
    first, more, release = threading.Event(), threading.Event(), threading.Event()

    def held(name, *args, **kwargs):
        if name == "pyteomics":
            importers.append(threading.get_ident())
            (more if first.is_set() else first).set()
            assert release.wait(timeout=5)
        return original(name, *args, **kwargs)

    barrier = threading.Barrier(8)

    def first_use(_):
        barrier.wait(timeout=5)
        return reader._unit_mzml()

    monkeypatch.setattr(builtins, "__import__", held)
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(first_use, i) for i in range(8)]
        try:
            assert first.wait(timeout=5)
            # Without the lock, the other seven callers import within this window.
            more.wait(timeout=.2)
        finally:
            release.set()
        classes = [future.result(timeout=5) for future in futures]
    assert len(importers) == 1
    assert all(cls is classes[0] for cls in classes)
    assert reader.UnitMzML is classes[0]
