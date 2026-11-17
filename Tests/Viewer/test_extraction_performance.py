"""Batched extraction parity, cache ownership, and offline reader startup."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from kmzcore.cancel import Cancelled
from kmzviewer.extract import MassList, MzRange, _sum_windows, extract_eics
from kmzviewer.model import ScanTable
from kmzviewer.spectra import ArrayCache, SpectrumReader
from Tests.Viewer.mzml_factory import write_mzml


def slow_interval(mz, intensity, lower, upper):
    """Independent pre-batching oracle, including stable sort/reduction order."""
    if not mz.size or not intensity.size or upper < lower:
        return 0.0
    finite = np.isfinite(mz)
    mz, intensity = mz[finite], intensity[finite]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    values = intensity[(mz >= lower) & (mz <= upper)]
    values = values[np.isfinite(values)]
    return float(np.sum(values, dtype=np.float64)) if values.size else 0.0


@pytest.mark.parametrize("order", ["sorted", "descending", "shuffled"])
def test_batched_windows_exactly_match_native_slice_oracle(order):
    rng = np.random.default_rng(2908)
    for length in (0, 1, 10, 2048):
        mz = np.sort(rng.integers(100, 500, length).astype(float))
        intensity = rng.normal(size=length) * rng.choice([1e-15, 1., 1e15], length)
        if length > 10:
            intensity[::41] = np.nan
            intensity[1::67] = np.inf
            mz[3::103] = np.nan
            mz[4::107] = np.inf
        permutation = np.arange(length)
        if order == "descending":
            permutation = permutation[::-1]
        elif order == "shuffled":
            rng.shuffle(permutation)
        mz, intensity = mz[permutation], intensity[permutation]
        bounds = np.array([(100., 499.), (150., 150.), (230., 150.), (499., 600.),
                           (-np.inf, np.inf), (275., 276.), (275., 276.), (50., 70.)] +
                          [(a, a + 20) for a in rng.uniform(80, 520, 40)])
        before = (mz.copy(), intensity.copy())
        expected = np.array([slow_interval(mz, intensity, *b) for b in bounds])
        actual = _sum_windows(mz, intensity, bounds)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        np.testing.assert_array_equal(mz, before[0])
        np.testing.assert_array_equal(intensity, before[1])


def test_tiny_window_after_huge_peak_is_not_lost_to_prefix_subtraction():
    mz = np.array([100., 200., 200., 300.])
    values = np.array([1e30, 1e-10, 2e-10, 9e30])
    np.testing.assert_array_equal(_sum_windows(mz, values, np.array([[200., 200.], [100., 200.]])),
                                  [3e-10, 1e30])


def table():
    return ScanTable(np.array([0, 2, 3]), np.array([.1, .2, .3]), np.array([1., 2., 3.]),
                     np.array([1, -1, 0]), np.array([False, True, False]), ["a", "b", "c"])


@pytest.mark.parametrize("polarity,rows", [("all", [0, 1, 2]), ("positive", [0]), ("negative", [1])])
@pytest.mark.parametrize("merge", [False, True])
def test_extract_keeps_overlap_multiplicity_native_order_and_polarity(monkeypatch, polarity, rows, merge):
    import kmzviewer.extract as module
    mz = np.array([101., 100., 100., 102., np.nan])
    y = np.array([1., 1e16, -1e16, 3., 999.])
    reads = []
    class Reader:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, index):
            reads.append(index)
            return mz, y * (index + 1)
    monkeypatch.setattr(module, "SpectrumReader", Reader)
    scans = table()
    target = MassList((100., 100., 101.), 20000.)
    count = 1 if merge else 3
    traces = extract_eics("synthetic.mzML", scans, target, polarity, merge,
                          list(range(count)), ["#0072b2"] * count)
    windows = [module.ppm_window(m, target.ppm) for m in target.masses]
    expected = [[slow_interval(mz, y * (scans.spectrum_index[row] + 1), *w) for row in rows]
                for w in windows]
    if merge:
        expected = [[sum((column[i] for column in expected), 0.0) for i in range(len(rows))]]
    np.testing.assert_array_equal([t.y for t in traces], expected)
    assert reads == scans.spectrum_index[rows].tolist()
    for trace in traces:
        np.testing.assert_array_equal(trace.row, rows)
        np.testing.assert_array_equal(trace.rt, scans.rt_min[rows])


def test_extract_cancels_after_decode_without_publishing(monkeypatch):
    import kmzviewer.extract as module
    cancelled = False
    class Reader:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, index):
            nonlocal cancelled
            cancelled = True
            return np.array([100.]), np.array([1.])
    monkeypatch.setattr(module, "SpectrumReader", Reader)
    with pytest.raises(Cancelled):
        extract_eics("synthetic.mzML", table(), MassList((100.,), 10.), "all", False,
                     [1], ["#0072b2"], is_cancelled=lambda: cancelled)


def test_extraction_keys_canonicalize_equivalent_numeric_inputs(tmp_path):
    path = tmp_path / "canonical.mzML"
    write_mzml(path, [{"mz": [100.], "intensity": [5.], "rt": .1, "tic": 5.}])
    from kmzviewer.mzml_index import index_ms1
    scans = index_ms1(path)
    for integer, floating in ((MassList((100,), 10), MassList((100.,), 10.)),
                              (MzRange(100, 101), MzRange(100., 101.))):
        a = extract_eics(str(path), scans, integer, "all", False, [1], ["#0072b2"])[0]
        b = extract_eics(str(path), scans, floating, "all", False, [2], ["#0072b2"])[0]
        assert a.group_key == b.group_key


def test_cache_acceptance_owns_arrays_and_rejects_without_copying(monkeypatch):
    cache = ArrayCache(budget_bytes=32)
    mz, y = np.array([100., 101.]), np.array([4., 5.])
    assert cache.put("a", 0, (mz, y))
    mz[:] = 0; y[:] = 0
    stored = cache.get("a", 0)
    np.testing.assert_array_equal(stored[0], [100., 101.])
    np.testing.assert_array_equal(stored[1], [4., 5.])
    class CopyForbidden(np.ndarray):
        def copy(self, *args, **kwargs):
            pytest.fail("A rejected scan was copied")
    rejected = np.arange(100., dtype=float).view(CopyForbidden)
    monkeypatch.setattr("kmzviewer.spectra.np.asarray", lambda a, dtype=None: a)
    assert not cache.put("a", 1, (rejected, rejected))
    assert cache.nbytes == 32 and cache.stored_scans("a") == 1


def test_parallel_reader_startup_uses_one_pinned_vocabulary_without_network(tmp_path, monkeypatch):
    import kmzviewer.spectra as spectra
    from kmzcore import psi_ms
    from psims.controlled_vocabulary import controlled_vocabulary as cv
    path = tmp_path / "offline.mzML"
    write_mzml(path, [{"mz": [100., 200.], "intensity": [4., 8.], "rt": .1, "tic": 12.}])
    monkeypatch.setattr(psi_ms, "_vocabulary", None)
    monkeypatch.setattr(cv, "urlopen", lambda *_a, **_k: pytest.fail("Reader attempted network I/O"))
    monkeypatch.setattr(spectra, "open_direct", lambda _path: None)  # pyteomics startup path
    load = psi_ms.load
    calls = []
    def counted():
        calls.append(None)
        return load()
    monkeypatch.setattr(psi_ms, "load", counted)
    def read(_):
        with SpectrumReader(path) as reader:
            arrays = reader.read(0)
            return arrays, reader._reader.cv
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, range(8)))
    assert len(calls) == 1
    assert len({id(vocab) for _arrays, vocab in results}) == 1
    for arrays, _vocab in results:
        np.testing.assert_array_equal(arrays[0], [100., 200.])
        np.testing.assert_array_equal(arrays[1], [4., 8.])
