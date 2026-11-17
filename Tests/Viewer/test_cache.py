"""Decoded spectrum arrays are reused until the memory budget evicts a file."""

import errno
import os
import queue
import threading
import weakref

import numpy as np
import pytest

from kmzviewer import spectra
from kmzviewer.spectra import ArrayCache, read_arrays


def test_second_read_does_not_decode_again():
    calls = {"n": 0}

    def read():
        calls["n"] += 1
        return np.array([273.1559, 300.0]), np.array([12.0, 1.0])

    cache = ArrayCache()
    first = read_arrays(cache, "sample.mzML", 4, read)
    second = read_arrays(cache, "sample.mzML", 4, read)
    assert calls["n"] == 1
    assert cache.misses == 1
    assert cache.hits == 1
    assert np.array_equal(first[1], second[1])
    cache.drop_path("sample.mzML")
    assert cache.stored_scans("sample.mzML") == 0
    assert cache.nbytes == 0


def test_budget_keeps_the_current_file_prefix_and_drops_another_file():
    cache = ArrayCache(budget_bytes=200)
    block = (np.zeros(10), np.zeros(10))
    assert cache.put("a.mzML", 0, block)
    assert cache.put("a.mzML", 1, block) is False
    assert cache.stored_scans("a.mzML") == 1
    smaller = (np.zeros(4), np.zeros(4))
    assert cache.put("b.mzML", 0, smaller)
    assert cache.stored_scans("a.mzML") == 0
    assert cache.stored_scans("b.mzML") == 1


def _owner(array):
    """The object that finally owns an array's memory: pages or bytes."""
    while isinstance(array, np.ndarray):
        array = array.base
    return array


def _bits(array):
    return np.asarray(array, dtype=np.float64).view(np.uint64)


def _large_file(path, sizes=(5000, 3, 4100, 0)):
    from Tests.Viewer.mzml_factory import write_mzml

    rng = np.random.default_rng(7)
    write_mzml(path, [{"mz": np.sort(rng.uniform(100, 900, n)), "intensity": rng.normal(size=n) * 1e6,
                       "rt": .1 * i, "tic": 1.} for i, n in enumerate(sizes)])
    return str(path)


def _immutable(array):
    """No view, base or new array over this memory can be made writable."""
    assert not array.flags.writeable and array.flags.aligned and array.flags.c_contiguous
    for alias in (array, array[::2], array.base, np.asarray(array), array.view(np.uint64)):
        with pytest.raises(ValueError):
            alias.setflags(write=True)
        with pytest.raises(ValueError):
            alias[...] = 0
    return True


def test_large_spectra_become_exact_float64_in_private_pages():
    rng = np.random.default_rng(31)
    mz = np.sort(rng.uniform(100, 900, 6000))
    specials = np.array([np.uint32(0x7fc00123).view(np.float32), np.nan, -0.0, np.inf, -np.inf,
                         np.float32(1e-45), np.float32(3.4e38)], dtype=np.float32)
    for dtype in ("<f4", ">f4", "<f8", ">f8", "<i4", "<i8", "<u2", "<f2"):
        values = (rng.normal(size=mz.size) * 1e4).astype(dtype)
        if values.dtype.kind == "f" and values.dtype.itemsize >= 4:
            values[:specials.size] = specials.astype(dtype)
        masses = mz.astype(">f8" if dtype.startswith(">") else "<f8")
        first, second = spectra._float64_pair(masses, values)
        frozen = spectra._freeze(masses, values)
        assert type(_owner(first)) is spectra._Pages and second.base is first.base
        assert first.flags.writeable and first.flags.c_contiguous and second.flags.c_contiguous
        assert type(_owner(frozen[0])) is spectra._Pages and frozen[1].base is frozen[0].base
        for got in ((first, second), frozen):
            np.testing.assert_array_equal(got[0].view(np.uint64), _bits(mz))
            np.testing.assert_array_equal(got[1].view(np.uint64), _bits(np.asarray(values, dtype=np.float64)))
    small = np.arange(10.)
    assert spectra._float64_pair(small, small) == (small, small)
    frozen = spectra._freeze(small, small.astype(">f4"))
    assert type(_owner(frozen[0])) is bytes and _immutable(frozen[0]) and _immutable(frozen[1])
    np.testing.assert_array_equal(frozen[1], small)


def test_put_copies_arrays_their_caller_could_still_change(tmp_path):
    # Review finding: a writable view taken before put() later changed the cache.
    with spectra.SpectrumReader(_large_file(tmp_path / "alias.mzML", (5000,))) as reader:
        read = reader.read(0)
    assert read[0].flags.writeable and type(_owner(read[0])) is spectra._Pages
    cache = ArrayCache()
    inputs = (read, (np.arange(4.), np.ones(4)), spectra._float64_pair(np.arange(5000.), np.ones(5000)),
              spectra._freeze(np.arange(5000.), np.ones(5000)), (np.arange(5000), np.ones(5000, np.float32)))
    for index, arrays in enumerate(inputs):
        expected = [array.copy() for array in arrays]
        views = [array.view() for array in arrays]
        assert cache.put("a", index, arrays)
        stored = cache.get("a", index)
        assert not any(np.shares_memory(mine, theirs) for mine in stored for theirs in arrays)
        for view in views:
            if view.flags.writeable:
                view[:] = 42
        for got, want in zip(stored, expected):
            np.testing.assert_array_equal(got, want)
            assert _immutable(got)


def test_cached_arrays_cannot_be_made_writable(tmp_path):
    # Review finding: setflags(write=True) succeeded on cached arrays.
    path = _large_file(tmp_path / "frozen.mzML")
    cache = ArrayCache()
    cache.put("put", 0, (np.arange(5000.), np.ones(5000)))
    cache.put("put", 1, (np.arange(4.), np.ones(4)))
    with spectra.SpectrumReader(path) as reader:
        direct = [read_arrays(cache, path, i, lambda i=i: reader.read(i)) for i in (0, 1)]
    with spectra.SpectrumReader(path) as reader:
        spectra.read_ahead(reader, cache, path + "#ahead", range(3))
        ahead = [read_arrays(cache, path + "#ahead", i, lambda i=i: reader.read(i)) for i in range(3)]
    hits = [cache.get("put", 0), cache.get("put", 1), *direct, *ahead]
    assert [type(_owner(pair[0])) for pair in hits] == [spectra._Pages, bytes] * 2 + [spectra._Pages, bytes, spectra._Pages]
    assert all(_immutable(array) for pair in hits for array in pair)


@pytest.mark.skipif(spectra._mprotect is None, reason="libc mprotect is unavailable")
def test_frozen_pages_are_read_only_in_hardware():
    class Raw:
        def __init__(self, array):
            self.__array_interface__ = dict(array.__array_interface__, data=(array.ctypes.data, False))

    mz, intensity = spectra._freeze(np.linspace(100, 900, 5000), np.ones(5000))
    pages = _owner(mz)
    assert type(pages) is spectra._Pages
    with pytest.raises(TypeError):
        pages[:8] = bytes(8)
    # Even through a raw writable view of the address, the kernel cannot write them.
    view = np.asarray(Raw(intensity))
    assert view.flags.writeable
    read, write = os.pipe()
    try:
        os.write(write, bytes(8))
        with pytest.raises(OSError) as error:
            os.readv(read, [view])
    finally:
        os.close(read)
        os.close(write)
    assert error.value.errno == errno.EFAULT and intensity[0] == 1.


def test_read_arrays_stores_the_reader_arrays_without_copying(tmp_path):
    path = _large_file(tmp_path / "adopt.mzML")
    cache = ArrayCache()
    for planned in (False, True):
        key = f"{path}#{planned}"
        returned = []

        def load(index, reader):
            returned.append(reader.read(index))
            return returned[-1]

        with spectra.SpectrumReader(path) as reader:
            if planned:
                spectra.read_ahead(reader, cache, key, range(4))
            got = [read_arrays(cache, key, i, lambda i=i: load(i, reader)) for i in range(4)]
        # The loader's own references stay harmless: the arrays are immutable.
        assert all(type(b) is spectra._Frozen and a[0] is b[0] and a[1] is b[1] for a, b in zip(got, returned))
        assert [cache.get(key, i) is got[i] for i in range(3)] == [True, True, True]
        assert not cache.contains(key, 3) and got[3][0].size == 0 and not got[3][0].flags.writeable
    assert cache.nbytes == 2 * 16 * (5000 + 3 + 4100)


def test_plain_reads_stay_writable_around_read_arrays(tmp_path):
    path = _large_file(tmp_path / "plain.mzML")
    with spectra.SpectrumReader(path) as reader:
        expected = [[array.copy() for array in reader.read(i)] for i in range(3)]
    cache = ArrayCache()
    with spectra.SpectrumReader(path) as reader, spectra.SpectrumReader(path) as other:
        elsewhere = []

        def load():
            # Only this thread's reads inside the loader are handed over frozen.
            thread = threading.Thread(target=lambda: elsewhere.append(other.read(0)))
            thread.start()
            thread.join(10)
            return reader.read(0)

        assert not any(array.flags.writeable for array in read_arrays(cache, path, 0, load))
        assert type(elsewhere[0]) is tuple and elsewhere[0][0].flags.writeable
        with pytest.raises(RuntimeError):
            read_arrays(cache, path, 1, lambda: (_ for _ in ()).throw(RuntimeError()))
        assert reader.read(0)[1].flags.writeable
        # Spectra planned for the cache but read plainly become writable copies;
        # spectra planned plainly but read for the cache become frozen copies.
        spectra.read_ahead(reader, cache, path, [1, 2])
        thawed = reader.read(1)
        reader.read_ahead([2])
        frozen = read_arrays(cache, path, 2, lambda: reader.read(2))
    assert type(thawed) is tuple and all(array.flags.writeable for array in thawed)
    assert not frozen[0].flags.writeable and cache.get(path, 2) is frozen
    for want, got in zip(expected[1:], (thawed, frozen)):
        for a, b in zip(want, got):
            np.testing.assert_array_equal(b.view(np.uint64), _bits(a))


def test_dropped_and_evicted_files_release_their_pages(tmp_path):
    path = _large_file(tmp_path / "evict.mzML", (5000,) * 4)
    cache = ArrayCache(budget_bytes=4 * 80000)
    pages = []
    for name in ("a.mzML", "b.mzML"):
        with spectra.SpectrumReader(path) as reader:
            spectra.read_ahead(reader, cache, name, range(4))
            for index in range(4):
                pages.append(weakref.ref(_owner(read_arrays(cache, name, index, lambda: reader.read(index))[0])))
    # Four b spectra evicted file a whole; dropping b releases the rest.
    assert [ref() is None for ref in pages] == [True] * 4 + [False] * 4
    cache.drop_path("b.mzML")
    assert all(ref() is None for ref in pages) and cache.nbytes == 0


def test_reader_arrays_match_pyteomics_with_and_without_read_ahead(tmp_path):
    from pyteomics import mzml

    from kmzviewer.spectra import SpectrumReader, _local_vocabulary, read_ahead

    path = _large_file(tmp_path / "large.mzML")
    with mzml.MzML(path, cv=_local_vocabulary()) as reference:
        expected = [(spectrum["m/z array"], spectrum["intensity array"]) for spectrum in reference]
    cache = ArrayCache()
    with SpectrumReader(path) as reader:
        read_ahead(reader, cache, path, range(4))
        planned = [read_arrays(cache, path, i, lambda i=i: reader.read(i)) for i in range(4)]
    with SpectrumReader(path) as reader:
        plain = [reader.read(i) for i in range(4)]
    with SpectrumReader(path) as reader:
        reader.read_ahead(range(4))
        ahead = [reader.read(i) for i in range(4)]
    for want, *results in zip(expected, planned, plain, ahead):
        for got in results:
            for a, b in zip(want, got):
                np.testing.assert_array_equal(b.view(np.uint64), _bits(a))
    assert type(_owner(planned[0][0])) is spectra._Pages and type(_owner(planned[1][0])) is bytes
    assert cache.get(path, 2) is planned[2] and not planned[2][1].flags.writeable
    for arrays in (plain[0], ahead[0], ahead[2]):
        assert type(_owner(arrays[0])) is spectra._Pages and arrays[0].flags.writeable


def test_spectrum_worker_keeps_no_arrays_once_a_batch_is_delivered():
    from PySide6.QtCore import Qt

    from kmzviewer.gui.workers import SpectrumWorker

    worker = SpectrumWorker(ArrayCache(budget_bytes=0))
    made = []

    def read(path, index):
        pair = np.arange(10.), np.ones(10)
        made.append(weakref.ref(pair[0]))
        return pair

    worker._read = read
    delivered = queue.Queue()
    worker.batch_result.connect(lambda token, payload: delivered.put(payload), Qt.ConnectionType.DirectConnection)
    worker.result.connect(lambda token, payload: delivered.put(payload), Qt.ConnectionType.DirectConnection)
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        worker.request_batch(1, [(1, "a.mzML", 0, None), (2, "b.mzML", 0, None)])
        assert set(delivered.get(timeout=10)) == {1, 2}
        worker.request(2, "a.mzML", 1)  # runs only after the batch has returned
        assert set(delivered.get(timeout=10)) == {"mz", "intensity"}
        assert made[0]() is None and made[1]() is None
    finally:
        worker.stop()
        thread.join(10)
