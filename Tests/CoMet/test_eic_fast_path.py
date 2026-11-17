"""Independent framed-reader oracle for the CoMet-only unframed EIC path."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from Tests.Convert.test_samples import multi
from kmzcore.cancel import Cancelled
from kmzcomet import sources
from kmzcomet.chromatograms import ChromatogramCache
from kmzcomet.engine import analyze
from kmzcomet.model import Feature, Scan
from kmzviewer.extract import ppm_window, sum_interval


@pytest.mark.parametrize('ident', [2, 7])
def test_unframed_native_preserves_measured_arrays_and_boundary_sums(multi, ident):
    path, _ = multi
    sample = SimpleNamespace(path=path, native_sample_id=ident)
    framed = list(sources.native_scans(sources.InputSource(path, ident)))
    fast = list(sources.eic_scans(sample))
    assert len(fast) == len(framed) > 0
    for a, b in zip(framed, fast):
        assert (a.rt, a.polarity, a.windows, a.mode) == (b.rt, b.polarity, b.windows, b.mode)
        mask = a.intensity != 0
        np.testing.assert_array_equal(a.mz[mask], b.mz[b.intensity != 0])
        np.testing.assert_array_equal(a.intensity[mask], b.intensity[b.intensity != 0])
        assert np.sum(a.intensity) == np.sum(b.intensity)
        assert np.max(a.intensity, initial=0) == np.max(b.intensity, initial=0)
        # Singleton inclusive boundaries and every suffix/prefix expose off-by-one errors.
        for mass in a.mz[mask]:
            for lo, hi in [(mass, mass), (0, mass), (mass, float('inf')), ppm_window(mass, 30)]:
                assert sum_interval(a.mz, a.intensity, lo, hi) == sum_interval(b.mz, b.intensity, lo, hi)


def test_native_new_row_cache_cancel_and_integrity(multi):
    path, _ = multi
    result = analyze([sources.InputSource(path, 2), sources.InputSource(path, 7)])
    scan = next(sources.native_scans(sources.InputSource(path, 2)))
    masses = scan.mz[scan.intensity != 0]
    result.features = [Feature(f'F{i}', float(m), .1, 1, [], 0., .2) for i, m in enumerate(masses[:2])]
    cache = ChromatogramCache(result)
    first = cache.extract([(0, 0), (0, 1)])
    assert first.files_read == 2
    assert all(e.data is not None for e in first.entries)
    assert first.entries[0].data.y[0] != first.entries[1].data.y[0]
    assert cache.extract([(0, 0)]).reused == 1
    new = cache.extract([(0, 0), (1, 0)])
    assert new.files_read == 1 and new.reused == 1
    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        cache.extract([(1, 1)], checkpoint=cancel)
    # Same-length replacement must fail the content hash even for a cached row.
    path.write_bytes(b'X' * path.stat().st_size)
    rejected = cache.extract([(0, 0)])
    assert rejected.entries[0].data is None
    assert 'differs' in rejected.entries[0].error


def test_vector_windows_coverage_zeros_and_polarity(monkeypatch):
    import kmzcomet.chromatograms as module
    low, high = ppm_window(100., 10.)
    scans = [Scan(i + 1., polarity, 'profile', np.array([low, 100., high]),
                  np.array([2., 0., 7.]), windows, 'number of detector counts')
             for i, (polarity, windows) in enumerate([(1, ((low, high),)), (1, ()),
                                                      (1, ((100., high),)), (-1, ())])]
    monkeypatch.setattr(module, 'sample_scans', lambda *_: iter(scans))
    key = (0, 100., 1, 10.)
    data = ChromatogramCache._read(SimpleNamespace(id='S', scans=4), (key,), lambda _: None, lambda: None)[key]
    np.testing.assert_array_equal(data.rt, [1., 2., 3.])
    np.testing.assert_array_equal(data.y, [9., 9., np.nan])
    assert data.missing_coverage == data.unknown_coverage == 1


@pytest.mark.parametrize('field', ['mz', 'intensity'])
def test_unframed_validation_rejects_nan(field):
    scan = Scan(1., 1, 'profile', np.array([100.]), np.array([1.]), (), 'counts')
    getattr(scan, field)[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        sources.validate_scan(scan, {})


@pytest.mark.parametrize('tic,expected', [(2**53-1, False), (2**53, True), (2**53+2, True)])
def test_extreme_integer_totals_keep_reference_framing(monkeypatch, tic, expected):
    # Metadata at the exact-integer boundary must choose the reference layout,
    # even though exercising that total with real 32-bit points would be huge.
    import kmzconvert_core.native as module
    calls = []
    def framed(reader, record):
        calls.append(True)
        return np.array([100.]), np.array([1.])
    class Reader:
        def __init__(self, *args, **kwargs):
            self.scans = [SimpleNamespace(experiment=1, rt_seconds=60., tic=tic)]
            self.experiments = [SimpleNamespace(ms_level=1, polarity='positive', lower_mz=50., upper_mz=200.)]
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def arrays(self, record, *, framing, decoder):
            calls.append(framing)
            return np.array([100.]), np.array([1.])
    monkeypatch.setattr(module, 'NativeWiffReader', Reader)
    monkeypatch.setattr(module, 'reader_arrays', framed)
    source = sources.InputSource(Path('synthetic.wiff'), 1)
    list(sources.native_scans(source, framing=False))
    list(sources.native_scans(source))
    assert calls == [expected, True]
