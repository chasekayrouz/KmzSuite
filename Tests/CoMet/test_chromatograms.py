"""Full-run EICs are inspection data, independent of the Heights calculation."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kmzcore.cancel import Cancelled
from kmzcomet import chromatograms
from kmzcomet.chromatograms import ChromatogramCache
from kmzcomet.engine import analyze, file_hash
from kmzcomet.model import Feature, Result, Sample, Scan, Settings
from kmzviewer.extract import ppm_window, sum_interval
from scripts.benchmarks.comet_fixtures import Truth, write_run


def fixture_result(tmp_path, *, samples=1):
    sources = []
    for i in range(samples):
        path = tmp_path / f'sample-{i}.mzML'; path.write_bytes(b'synthetic input identity')
        sources.append(Sample(path, f'S{i+1:03d}', file_hash(path), path.stat().st_size,
                              scans=4, units={'number of detector counts'}, shifts={1: .4}))
    features = [Feature('F1', 100., 1., 1, [], .9, 1.1),
                Feature('F2', 200., 1.5, -1, [], 1.4, 1.6),
                Feature('F3', 100., 2., 1, [], 1.9, 2.1)]
    return Result(Settings(mz_ppm=10), sources, features, [[] for _ in sources], 0.)


def scan(rt, polarity=1, *, windows=((50., 250.),), unit='number of detector counts'):
    low, high = ppm_window(100., 10.)
    return Scan(rt, polarity, 'profile', np.array([low, 100., high, 200.]),
                np.array([2., 7., 3., 20.]), windows, unit)


def test_full_run_native_rt_sum_endpoints_and_polarity(tmp_path, monkeypatch):
    result = fixture_result(tmp_path)
    scans = [scan(0.), scan(.5, -1), scan(1.), scan(2.)]
    monkeypatch.setattr(chromatograms, 'sample_scans', lambda *args: iter(scans))
    batch = ChromatogramCache(result).extract([(0, 0), (1, 0)])
    positive, negative = (entry.data for entry in batch.entries)
    np.testing.assert_array_equal(positive.rt, [0., 1., 2.])
    np.testing.assert_array_equal(positive.y, [12., 12., 12.])  # Sum, not the 7-count maximum.
    np.testing.assert_array_equal(negative.rt, [.5])
    np.testing.assert_array_equal(negative.y, [20.])
    assert batch.files_read == 1 and batch.reused == 0
    assert not positive.rt.flags.writeable and not positive.y.flags.writeable


def test_unacquired_is_nan_empty_is_zero_unknown_coverage_is_explicit(tmp_path, monkeypatch):
    result = fixture_result(tmp_path)
    scans = [scan(0., windows=((150., 250.),)), scan(1., windows=()),
             replace(scan(2.), mz=np.array([150.]), intensity=np.array([9.]))]
    monkeypatch.setattr(chromatograms, 'sample_scans', lambda *args: iter(scans))
    entries = ChromatogramCache(result).extract([(0, 0), (1, 0)]).entries
    np.testing.assert_array_equal(entries[0].data.y, [np.nan, 12., 0.])
    assert entries[0].data.missing_coverage == entries[0].data.unknown_coverage == 1
    assert entries[1].data.rt.size == entries[1].data.y.size == 0
    result.samples[0].units = {'unknown'}
    assert ChromatogramCache(result).extract([(0, 0)]).entries[0].unit == 'unknown'


def test_one_pass_per_sample_deduplication_and_session_reuse(tmp_path, monkeypatch):
    result = fixture_result(tmp_path, samples=2)
    calls = []
    def read(sample, checkpoint):
        calls.append(sample.id)
        return iter([scan(0.), scan(1.)])
    monkeypatch.setattr(chromatograms, 'sample_scans', read)
    cache = ChromatogramCache(result)
    batch = cache.extract([(0, 0), (2, 0), (0, 1), (0, 0)])
    assert calls == ['S001', 'S002'] and len(batch.entries) == 3
    assert batch.entries[0].data is batch.entries[1].data
    again = cache.extract([(2, 0), (0, 1)])
    assert again.files_read == 0 and again.reused == 2 and len(calls) == 2
    assert again.entries[0].data is batch.entries[0].data


def test_cache_eviction_does_not_change_open_tab_data(tmp_path, monkeypatch):
    result = fixture_result(tmp_path, samples=2)
    monkeypatch.setattr(chromatograms, 'sample_scans', lambda *args: iter([scan(0.), scan(1.)]))
    cache = ChromatogramCache(result, max_bytes=32)
    first = cache.extract([(0, 0)]).entries[0].data
    cache.extract([(0, 1)])
    assert cache.nbytes <= 32
    reread = cache.extract([(0, 0)])
    assert reread.files_read == 1 and reread.entries[0].data is not first
    np.testing.assert_array_equal(first.y, [12., 12.])


@pytest.mark.parametrize('change', ['content', 'missing'])
def test_cached_sources_are_rechecked_and_partial_failures_are_labeled(tmp_path, monkeypatch, change):
    result = fixture_result(tmp_path, samples=2)
    monkeypatch.setattr(chromatograms, 'sample_scans', lambda *args: iter([scan(0.)]))
    cache = ChromatogramCache(result); cache.extract([(0, 0), (0, 1)])
    path = result.samples[0].path
    if change == 'content':
        path.write_bytes(b'changed'.ljust(path.stat().st_size, b'!'))
    else:
        path.unlink()
    batch = cache.extract([(0, 0), (0, 1)])
    assert batch.entries[0].data is None and batch.entries[0].error
    assert batch.entries[1].data is not None and batch.reused == 1


def test_source_change_during_read_discards_traces(tmp_path, monkeypatch):
    result = fixture_result(tmp_path)
    def read(sample, checkpoint):
        yield scan(0.)
        sample.path.write_bytes(b'changed during read')
        yield scan(1.)
    monkeypatch.setattr(chromatograms, 'sample_scans', read)
    cache = ChromatogramCache(result)
    batch = cache.extract([(0, 0)])
    assert batch.entries[0].data is None and 'changed' in batch.entries[0].error
    assert cache.nbytes == 0


def test_cancellation_during_read_publishes_no_partial_batch(tmp_path, monkeypatch):
    result = fixture_result(tmp_path)
    stopped = False
    def checkpoint():
        if stopped:
            raise Cancelled()
    def read(*args):
        nonlocal stopped
        yield scan(0.)
        stopped = True
        yield scan(1.)
    monkeypatch.setattr(chromatograms, 'sample_scans', read)
    cache = ChromatogramCache(result)
    with pytest.raises(Cancelled):
        cache.extract([(0, 0)], checkpoint=checkpoint)
    assert cache.nbytes == 0


def test_real_mzml_trace_matches_independent_viewer_sum(tmp_path):
    from kmzcomet.reader import iter_scans
    path = write_run(tmp_path/'profile.mzML', [Truth('a', 200., 1., (9000.,))],
                     sample=0, profile=True, noise=0., rt_stop=2.)
    result = analyze([path], Settings(alignment='none'))
    before = file_hash(path)
    entry = ChromatogramCache(result).extract([(0, 0)]).entries[0]
    low, high = ppm_window(entry.mz, entry.ppm)
    expected = [(s.rt, sum_interval(s.mz, s.intensity, low, high)) for s in iter_scans(path) if s.polarity == entry.polarity]
    np.testing.assert_array_equal(entry.data.rt, [row[0] for row in expected])
    np.testing.assert_array_equal(entry.data.y, [row[1] for row in expected])
    assert file_hash(path) == before
    assert entry.data.y.max() > result.measurements[0][0].height
