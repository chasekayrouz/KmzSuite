"""The optimized analysis equals main@0a039d5's implementation exactly (synthetic inputs).

``Tests.CoMet.reference_analysis`` keeps the original discovery, measurement and
Excel writer verbatim. Comparisons use ``repr`` so float bits (including -0.0)
and container order must match, not only values.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import datetime
from pathlib import Path
import signal
import socket
import threading
import time
import types
import zipfile

import numpy as np
import pytest

from kmzcore.cancel import Cancelled
from kmzcomet import detection, engine, excel, sources
from kmzcomet.model import Candidate, Feature, Sample, Scan, Settings
from kmzconvert_core import native
from kmzconvert_core.binary import FormatError
from kmzconvert_core.reader import WiffReader
from Tests.CoMet import reference_analysis as reference
from Tests.Convert.test_parallel_writer import acquisition  # noqa: F401  (fixture)

SETTINGS = [Settings(), Settings(mz_ppm=30, discovery_window=.35, rt_window=.15, min_contrast=2, min_height=100),
            Settings(mz_ppm=5, discovery_height=200., max_scan_gap=.02, min_points=4, rt_tolerance=.05)]


def workers():
    return [t for t in threading.enumerate() if t.name.startswith('kmzconvert')]


def profile_values(rng, size):
    """Counts with framing zeros, plateaus, shoulders, ties and isolated spikes."""
    y = rng.choice([0., 0., 0., 3., 40., 400., 900., 1000., 1000., 1200., 3000., 25000.], size)
    y[rng.random(size) < .05] = 1000.
    for start in rng.integers(0, max(1, size-8), size // 20):
        y[start:start+5] = np.array([200., 1500., 1400., 1600., 100.]) * rng.choice([1., 2., .5])
    return y


def scan_series(rng, *, mode='profile', count=160, ions=40, polarities=(1,), windows=((50., 2000.),), gaps=True):
    """Chromatographic ions (some within a few ppm of each other) plus noise and spikes."""
    masses = np.sort(np.concatenate([rng.uniform(100, 900, ions // 2),
                                     rng.uniform(100, 900, ions // 4) * (1 + rng.choice([2e-6, 8e-6, 25e-6], ions // 4))]))
    centers, widths = rng.uniform(0, 3, len(masses)), rng.uniform(.02, .2, len(masses))
    heights = rng.choice([500., 3000., 20000., 2e5], len(masses))
    scans, rt = [], 0.
    for i in range(count):
        rt += rng.choice([.01, .01, .01, .03, .2]) if gaps else .01
        for polarity in polarities:
            signal = heights * np.exp(-.5*((rt-centers)/widths)**2) * rng.uniform(.8, 1.2, len(masses))
            if mode == 'profile':
                offsets = np.arange(-3, 4) * 4e-6
                mz = (masses[:, None] * (1 + offsets)).ravel()
                y = (signal[:, None] * np.array([0., .2, .7, 1., .65, .3, 0.])).ravel()
                y = np.round(y + rng.choice([0., 0., 5., 50.], len(y)))
            else:
                mz, y = masses * (1 + rng.normal(0, 2e-6, len(masses))), np.round(signal)
            order = np.argsort(mz, kind='stable')
            mz, y = mz[order], y[order]
            keep = np.concatenate(([True], np.diff(mz) > 0))
            scan_windows = windows if i % 7 else ((150., 600.), (650., 2000.))
            scans.append(Scan(round(rt, 6) + polarity * 1e-4, polarity, mode, mz[keep], y[keep], scan_windows,
                              'number of detector counts'))
    return scans


def test_mass_peaks_equal_reference_on_random_profiles():
    rng = np.random.default_rng(3)
    for trial in range(400):
        size = int(rng.integers(0, 300))
        y = profile_values(rng, size)
        for mode in ('profile', 'centroid'):
            scan = Scan(1., 1, mode, np.arange(size) + 100., y, (), 'count')
            for threshold in (1., 900., 1000., 1500.):
                expected, actual = reference.mass_peaks(scan, threshold), detection.mass_peaks(scan, threshold)
                assert actual.dtype == expected.dtype and actual.tolist() == expected.tolist()


@pytest.mark.parametrize('settings', SETTINGS)
@pytest.mark.parametrize('mode,polarities', [('profile', (1,)), ('profile', (-1, 0, 1)), ('centroid', (1, -1))])
def test_detector_equals_reference(settings, mode, polarities):
    rng = np.random.default_rng(len(polarities) + 7 * (mode == 'centroid'))
    scans = scan_series(rng, mode=mode, polarities=polarities)
    old, new = reference.Detector(2, settings), detection.Detector(2, settings)
    for scan in scans:
        old.add(scan)
        new.add(scan)
    assert repr(new.complete()) == repr(old.complete())
    assert repr(new.candidates) == repr(old.candidates)
    assert new.serial == old.serial


def track_state(detector):
    """Active tracks per polarity as (mass, serial, times, masses, heights), either layout."""
    state = {}
    for polarity, active in detector.active.items():
        if isinstance(active, tuple):
            entries = zip(active[0], active[1], active[2])
        else:
            entries = active
        state[polarity] = [(m, s, t.times, t.masses, t.heights) for m, s, t in entries]
    return repr(state)


DYADIC = Settings(mz_ppm=30, rt_tolerance=.0625, rt_window=.125, discovery_window=.125,
                  max_scan_gap=.0625, discovery_height=1000., min_contrast=2)


def dyadic_scans(rng, count=240):
    """Exactly representable times, masses and spacings: ties in track distance,
    RT window edges and scan gaps equal to their limits occur exactly."""
    d = 2.**-12
    scans = []
    for k in range(count):
        if rng.random() < .15:
            continue  # A skipped scan makes some gaps exactly max_scan_gap.
        rt = k * 2.**-5
        masses, heights = [], []
        for j in range(12):
            base = 100. + j
            level = 20000. * np.exp(-.5*((rt - (j % 4 + 1) * .9)/.12)**2) + rng.choice([0., 1500., 1500.])
            pattern = rng.integers(4)
            points = [base, base + 2*d] if pattern == 0 else [base + d] if pattern == 1 else [base, base + d, base + 2*d]
            for mass in points:
                masses.append(mass)
                heights.append(float(np.round(level * rng.choice([1., 1., .5]))))
        order = np.argsort(masses, kind='stable')
        scans.append(Scan(rt, 1, 'centroid', np.array(masses)[order], np.array(heights)[order], ((50., 2000.),), 'counts'))
    return scans


@pytest.mark.parametrize('seed', range(4))
def test_detector_exact_ties_and_limits_equal_reference(seed):
    scans = dyadic_scans(np.random.default_rng(seed))
    old, new = reference.Detector(0, DYADIC), detection.Detector(0, DYADIC)
    for scan in scans:
        old.add(scan)
        new.add(scan)
        assert track_state(new) == track_state(old)
    assert repr(new.complete()) == repr(old.complete())
    assert repr(new.candidates) == repr(old.candidates)


@pytest.mark.parametrize('seed', range(3))
def test_quantify_exact_window_edges_equal_reference(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    scans = dyadic_scans(rng)
    features = []
    for i in range(80):
        mz = float(rng.choice(scans[40].mz)) if i % 2 else 100. + int(rng.integers(12)) + 2.**-13
        left = int(rng.integers(0, 230)) * 2.**-5
        right = left + int(rng.integers(1, 12)) * 2.**-5
        features.append(Feature(f'F{i:06d}', mz, (left + right) / 2, 1, [], left, right, bool(i % 9 == 0)))
    edges = [(f.mz*(1-DYADIC.mz_ppm*1e-6), f.mz*(1+DYADIC.mz_ppm*1e-6)) for f in features[:6]]
    for k, scan in enumerate(scans):
        if k % 3 == 0:
            scan.windows = (edges[k % 6],)  # Coverage bounds exactly at a feature's mass window.
        elif k % 3 == 1:
            low, high = edges[k % 6]
            scan.windows = ((low, np.nextafter(high, 0)),)
    sample = Sample(Path('unused'), 'S001', '', 0, scans=len(scans), shifts={1: .0625})
    monkeypatch.setattr(reference, 'sample_scans', lambda *_: iter(scans))
    monkeypatch.setattr(engine, 'sample_scans', lambda *_: iter(scans))
    expected = reference.quantify(sample, 0, features, DYADIC, lambda _: None)
    assert repr(engine.quantify(sample, 0, features, DYADIC, lambda _: None)) == repr(expected)
    assert any('partial_mass_coverage' in m.flags for m in expected)


@pytest.mark.parametrize('ppm', [1e-9, 10., 5e4])
def test_duplicate_collapse_equals_reference(ppm):
    rng = np.random.default_rng(int(ppm) + 1)
    settings = Settings(mz_ppm=ppm, rt_tolerance=.05, rt_window=.1)
    base = rng.uniform(100, 400, 30)
    candidates = [Candidate(0, float(m * (1 + d)), float(t), float(h), int(p))
                  for m in base for d, t, h, p in zip(rng.choice([0., ppm*1e-6/2, ppm*1e-6, 2*ppm*1e-6], 6),
                                                      rng.choice([1., 1.03, 1.05, 1.06], 6),
                                                      rng.choice([5000., 5000., 7000.], 6), rng.choice([1, -1], 6))]
    old, new = reference.Detector(0, settings), detection.Detector(0, settings)
    old.candidates, new.candidates = list(candidates), list(candidates)
    assert repr(new.complete()) == repr(old.complete())


def test_trace_metrics_equal_reference_rows():
    rng = np.random.default_rng(11)
    for length in list(range(1, 12)) + [17, 40]:
        for settings in SETTINGS:
            times = np.cumsum(rng.choice([.01, .01, .02, .2], (25, length)), axis=1)
            values = rng.choice([0., 1., 5., 5., 100., 1000., 3000.], (25, length)) * rng.choice([1., 1.37], (25, 1))
            batch = detection.trace_metrics(times, values, settings)
            for t, y, metrics in zip(times, values, batch):
                expected = reference.signal_metrics(t, y, settings)
                assert repr(metrics) == repr(expected)
                assert repr(detection.signal_metrics(t, y, settings)) == repr(expected)


def test_window_maxima_take_numpy_first_argmax():
    rng = np.random.default_rng(5)
    for _ in range(300):
        size = int(rng.integers(0, 60))
        y = rng.choice([0., -0., 2., 2., 7.], size)
        mz = np.arange(size) * .5 + 100.
        starts = rng.integers(0, size + 1, 20)
        stops = np.minimum(size, starts + rng.integers(0, 6, 20))
        values, masses = engine.window_maxima(mz, y, starts, stops)
        for a, b, value, mass in zip(starts, stops, values, masses):
            if a < b:
                apex = a + int(np.argmax(y[a:b]))
                assert repr((float(y[apex]), float(mz[apex]))) == repr((float(value), float(mass)))
            else:
                assert value == 0 and np.isnan(mass)


def test_array_validation_equals_reference_predicate():
    def expected(m, y):
        return not (m.ndim != 1 or m.shape != y.shape or not np.isfinite(m).all() or not np.isfinite(y).all()
                    or np.any(m <= 0) or np.any(np.diff(m) <= 0) or np.any(y < 0))
    rng = np.random.default_rng(13)
    special = [np.nan, np.inf, -np.inf, 0., -0., -1., 1e-300, 5., 5., 1e308]
    for _ in range(4000):
        size = int(rng.integers(0, 6))
        m = np.sort(rng.uniform(1, 10, size)) if rng.random() < .5 else rng.choice(special, size)
        y = rng.choice(special, size) if rng.random() < .5 else rng.uniform(0, 10, size)
        if rng.random() < .3 and size:
            m[rng.integers(size)] = rng.choice(special)
        assert sources.ordered_finite(m, y) == expected(m, y)
    assert not sources.ordered_finite(np.ones((2, 2)), np.ones((2, 2)))
    assert not sources.ordered_finite(np.arange(1., 4.), np.ones(2))


def features_for(rng, scans, count=120):
    masses = np.concatenate([[float(m) for m in rng.choice(scans[len(scans)//2].mz, count // 2)],
                             rng.uniform(100, 900, count - count // 2)])
    features = []
    for i, mz in enumerate(masses):
        rt = float(rng.uniform(-.2, 3.2))
        left, right = rt - float(rng.choice([.15, .05, .3])), rt + float(rng.choice([.15, .05, .3]))
        polarity = int(rng.choice([1, 1, -1, 0]))
        members = [Candidate(int(j), float(mz), rt, 1000., polarity) for j in rng.choice(3, int(rng.integers(0, 3)), replace=False)]
        features.append(Feature(f'F{i:06d}', float(mz), rt, polarity, members, left, right, bool(rng.random() < .1)))
    return features


@pytest.mark.parametrize('settings', SETTINGS)
@pytest.mark.parametrize('windows', [((50., 2000.),), (), ((50., 300.), (310., 2000.))])
def test_quantify_equals_reference(monkeypatch, settings, windows):
    rng = np.random.default_rng(len(windows))
    scans = scan_series(rng, polarities=(1, -1, 0), windows=windows, count=120)
    features = features_for(rng, scans)
    sample = Sample(Path('unused'), 'S002', '', 0, scans=len(scans), shifts={1: .03, -1: -.02, 0: 0.})
    monkeypatch.setattr(reference, 'sample_scans', lambda *_: iter(scans))
    monkeypatch.setattr(engine, 'sample_scans', lambda *_: iter(scans))
    messages = ([], [])
    expected = reference.quantify(sample, 1, features, settings, messages[0].append)
    actual = engine.quantify(sample, 1, features, settings, messages[1].append)
    assert repr(actual) == repr(expected)
    assert messages[0] == messages[1]
    assert engine.quantify(sample, 1, [], settings, lambda _: None) == []


def test_summarize_single_trace_equals_reference():
    rng = np.random.default_rng(17)
    sample = Sample(Path('unused'), 'S001', '', 0, shifts={1: .01})
    for length in range(0, 30):
        for settings in SETTINGS:
            times = np.cumsum(rng.choice([.01, .01, .1], length)) + .8
            trace = [(float(t), float(v), float(m), bool(k)) for t, v, m, k in zip(
                times, rng.choice([0., 3., 300., 3000.], length), rng.uniform(199.99, 200.01, length), rng.random(length) < .9)]
            feature = Feature('F1', 200., 1., 1, [Candidate(0, 200., 1., 1., 1)], .85, 1.15, bool(length % 2))
            for missing in (0, 2):
                assert (repr(engine.summarize(feature, sample, 0, trace, settings, missing_mass_scans=missing)) ==
                        repr(reference.summarize(feature, sample, 0, trace, settings, missing_mass_scans=missing)))


def frozen_openpyxl(monkeypatch):
    fixed = datetime.datetime(2026, 1, 1)

    class Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.replace(tzinfo=tz) if tz else fixed
    clock = types.SimpleNamespace(datetime=Frozen, timezone=datetime.timezone)
    import openpyxl.packaging.core as core
    import openpyxl.writer.excel as writer
    monkeypatch.setattr(core, 'datetime', clock)
    monkeypatch.setattr(writer, 'datetime', clock)


def test_workbook_equals_reference(tmp_path, monkeypatch):
    from scripts.benchmarks.comet_fixtures import cohort
    paths, _, _ = cohort(tmp_path / 'inputs', profile=True)
    result = engine.analyze(paths)
    result.measurements[1][0].flags = 'odd "flag" <&>'
    frozen_openpyxl(monkeypatch)
    excel.write_workbook(result, tmp_path / 'new.xlsx')
    reference.write_workbook(result, tmp_path / 'old.xlsx')
    with zipfile.ZipFile(tmp_path / 'new.xlsx') as new, zipfile.ZipFile(tmp_path / 'old.xlsx') as old:
        assert new.namelist() == old.namelist()
        for name in old.namelist():
            assert new.read(name) == old.read(name), name


def test_mzml_vocabulary_is_the_pinned_release_without_network(monkeypatch):
    """The reader shares the suite's pinned PSI-MS release (the current release main
    downloaded) and loads it without network access."""
    from kmzcore import psi_ms
    from kmzcomet import reader
    attempts = []

    def offline(*args, **kwargs):
        attempts.append(args)
        raise OSError('network access')
    monkeypatch.setattr(socket, 'create_connection', offline)
    monkeypatch.setattr(psi_ms, '_vocabulary', None)
    loaded = reader.vocabulary()
    assert loaded is psi_ms.vocabulary() and loaded.version == psi_ms.VERSION
    assert not attempts


NEWER_PARAM = ('<cvParam cvRef="MS" accession="MS:1004014" '
               'name="ion mobility isolation window target" value="1.2"/>')


def test_mzml_reader_accepts_metadata_from_the_pinned_vocabulary(tmp_path, monkeypatch):
    """Pyteomics looks up every valued MS cvParam. The pinned release accepts a term
    that psims' older bundled copy (4.1.258) rejects, as main did when online."""
    from psims.controlled_vocabulary import controlled_vocabulary as cv
    from pyteomics.xml import CVParamParser
    from kmzcomet import reader
    from Tests.Viewer.mzml_factory import write_mzml
    path = tmp_path / 'metadata.mzML'
    write_mzml(path, [dict(rt=.01, mz=np.arange(100., 110.), intensity=np.arange(10.))])
    path.write_text(path.read_text().replace('<scanList', NEWER_PARAM + '<scanList', 1))

    def outcome(iter_scans):
        # Pyteomics caches each accession's value type for the whole process.
        monkeypatch.setattr(CVParamParser, '_cv_type_cache', {})
        return scan_outcome(iter_scans(path))
    expected = outcome(reference.iter_scans)
    assert expected[1] is None and len(expected[0]) == 1
    assert outcome(reader.iter_scans) == expected
    with cv._use_vendored_psims_obo() as stream:
        bundled_only = cv.ControlledVocabulary.from_obo(stream)
    assert 'MS:1004014' not in bundled_only.terms
    monkeypatch.setattr(reader, 'vocabulary', lambda: bundled_only)
    assert outcome(reader.iter_scans)[1][0] is KeyError


def reference_scans(path, framing):
    """main@0a039d5's adapter: the plain interpreted WiffReader, sequentially."""
    with WiffReader(path) as reader:
        for record in reader.scans:
            if reader.experiments[record.experiment-1].ms_level == 1:
                mz, y = reader.arrays(record, framing=framing or record.tic >= 2**53)
                yield record.rt_seconds/60, mz, y


@pytest.mark.parametrize('framing', [True, False])
def test_threaded_native_scans_equal_interpreted_reader(acquisition, monkeypatch, framing):
    expected = [(rt, mz.tobytes(), y.tobytes()) for rt, mz, y in reference_scans(acquisition, framing)]
    assert len(expected) == 240
    for decoders, extension in [(2, True), (0, True), (2, False)]:
        with monkeypatch.context() as patched:
            patched.setattr(sources, 'DECODERS', decoders)
            if not extension:
                patched.setattr(native, '_native', None)
            scans = list(sources.native_scans(sources.InputSource(acquisition), framing=framing))
        assert [(s.rt, s.mz.tobytes(), s.intensity.tobytes()) for s in scans] == expected
        assert all(s.mz.dtype == s.intensity.dtype == np.float64 for s in scans)
    assert not workers()


def test_cancellation_and_worker_errors_join_decoders_before_closing(acquisition, monkeypatch):
    readers = []

    class Tracked(native.NativeWiffReader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            readers.append(self)
    monkeypatch.setattr(native, 'NativeWiffReader', Tracked)
    calls = 0

    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 101:
            raise Cancelled()
    seen = []
    with pytest.raises(Cancelled):
        for scan in sources.native_scans(sources.InputSource(acquisition), checkpoint):
            seen.append(scan.rt)
    assert 0 < len(seen) < 50 and not workers() and readers[-1]._mapping.closed

    original = native.reader_arrays

    def broken(reader, record):
        if record.slot == 2 * 60:
            raise FormatError('synthetic damage')
        return original(reader, record)
    monkeypatch.setattr(native, 'reader_arrays', broken)
    seen.clear()
    with pytest.raises(FormatError, match='synthetic damage'):
        for scan in sources.native_scans(sources.InputSource(acquisition)):
            seen.append(scan)
    assert len(seen) == 60 and not workers() and readers[-1]._mapping.closed


def test_native_analysis_equals_interpreted_codec(acquisition, monkeypatch):
    accelerated = engine.analyze([acquisition])
    with monkeypatch.context() as patched:
        patched.setattr(native, '_native', None)
        interpreted = engine.analyze([acquisition])
    for result in (accelerated, interpreted):
        result.elapsed_seconds, result.analysis_id, result.created_utc = 0., '', ''
    assert repr(asdict(accelerated)) == repr(asdict(interpreted))
    assert accelerated.features and not workers()


def mzml_threads():
    return [t for t in threading.enumerate() if t.name.startswith('kmzcomet-mzml')]


def scan_outcome(scans):
    """Every scan's values and array bytes, then the exception type and message."""
    seen = []
    try:
        for scan in scans:
            seen.append((repr((scan.rt, scan.polarity, scan.mode, scan.windows, scan.intensity_unit)),
                         scan.mz.dtype.str, scan.mz.tobytes(), scan.intensity.dtype.str, scan.intensity.tobytes()))
    except Exception as error:
        return seen, (type(error), str(error))
    return seen, None


@pytest.mark.parametrize('defer', [0, None])
@pytest.mark.parametrize('profile', [True, False])
def test_deferred_mzml_scans_equal_reference(tmp_path, monkeypatch, profile, defer):
    from kmzcomet import reader
    if defer is not None:
        monkeypatch.setattr(reader, 'DEFER_CHARS', defer)  # Every array on a thread.
    from scripts.benchmarks.comet_fixtures import Truth, cohort, write_run
    paths, _, _ = cohort(tmp_path, profile=profile)
    paths.append(write_run(tmp_path / 'seconds.mzML', [Truth('a', 200., 1., (9000.,), polarity=-1),
                                                        Truth('b', 300., 1.2, (5000.,))],
                           rt_unit='second', scan_windows=False, rt_stop=2., profile=profile))
    for path in paths:
        expected = scan_outcome(reference.iter_scans(path))
        assert expected[1] is None and expected[0]
        assert scan_outcome(reader.iter_scans(path)) == expected
    assert not mzml_threads()


def damaged_mzml(path, tmp_path, name, *, binary=(), unit=None, truncate=None, length=None):
    """Copy ``path`` with replaced binary texts (array ordinal, text), one RT unit
    changed or array length made unconvertible (spectrum ordinal), and/or the
    document cut inside or after a spectrum."""
    import re
    data = path.read_text()
    if length is not None:
        starts = [m.start(1) for m in re.finditer(r'defaultArrayLength="([0-9]+)"', data)]
        data = data[:starts[length]] + 'twenty' + data[starts[length] + 2:]
    spans = [m.span(1) for m in re.finditer(r'<binary>([^<]*)</binary>', data)]
    for ordinal, text in sorted(binary, reverse=True):
        a, b = spans[ordinal]
        data = data[:a] + text + data[b:]
    if unit is not None:
        units = [m.start() for m in re.finditer('unitName="minute"', data)]
        data = data[:units[unit]] + 'unitName="hour"' + data[units[unit] + len('unitName="minute"'):]
    if truncate is not None:
        ends = [m.end() for m in re.finditer('</spectrum>', data)]
        data = data[:ends[truncate[0]] + truncate[1]]
    target = tmp_path / name
    target.write_text(data)
    return target


@pytest.mark.parametrize('damage', [
    dict(binary=[(11, 'bm90IHpsaWIgZGF0YQ==')]),                      # MS1 intensity: not zlib
    dict(binary=[(10, 'abc')]),                                          # MS1 m/z: bad base64 padding
    dict(binary=[(10, 'qé')]),                                      # non-ASCII base64 text
    dict(binary=[(7, 'bm90IHpsaWIgZGF0YQ==')]),                         # MS2 spectrum arrays still decode
    dict(binary=[(13, '')]),                                             # empty array text
    dict(truncate=(8, 0)),                                               # truncated document
    dict(truncate=(9, -400)),                                            # cut inside a spectrum
    dict(binary=[(12, 'bm90IHpsaWIgZGF0YQ==')], truncate=(7, 0)),       # decode error before parse error
    dict(binary=[(13, 'bm90IHpsaWIgZGF0YQ==')], unit=6),                # decode error before metadata error
    dict(binary=[(15, 'bm90IHpsaWIgZGF0YQ==')], unit=6),                # metadata error before a later decode
    dict(binary=[(10, 'abc'), (11, 'bm90IHpsaWIgZGF0YQ==')]),           # first array's error first
    dict(length=5),                                                      # pyteomics conversion error
    dict(binary=[(11, 'bm90IHpsaWIgZGF0YQ==')], length=5),              # decode before the same spectrum's error
])
def test_deferred_mzml_errors_surface_in_reference_order(tmp_path, monkeypatch, damage):
    from kmzcomet import reader
    monkeypatch.setattr(reader, 'DEFER_CHARS', 0)  # Small synthetic arrays still decode on threads.
    from Tests.Viewer.mzml_factory import write_mzml
    rng = np.random.default_rng(23)
    spectra = []
    for i in range(14):
        ms2 = i % 4 == 3
        spectra.append(dict(rt=.01 * (i + 1), mz=np.sort(rng.uniform(100, 900, 20)), intensity=rng.uniform(0, 9000, 20),
                            ms_level=2 if ms2 else 1, precursor_mz=500. if ms2 else None, polarity=1 - 2 * (i % 5 == 4)))
    source = tmp_path / 'source.mzML'
    write_mzml(source, spectra)
    path = damaged_mzml(source, tmp_path, 'damaged.mzML', **damage)
    expected = scan_outcome(reference.iter_scans(path))
    assert expected[1] is not None
    assert scan_outcome(reader.iter_scans(path)) == expected
    assert not mzml_threads()



def test_unqualified_pyteomics_decodes_while_parsing(tmp_path, monkeypatch, unqualified):
    from kmzcomet import reader
    from Tests.Viewer.mzml_factory import write_mzml
    monkeypatch.setattr(reader, 'DEFER_CHARS', 0)
    path = tmp_path / 'synthetic.mzML'
    write_mzml(path, [dict(rt=.01 * (i + 1), mz=np.arange(100., 110.) + 10 * i, intensity=np.arange(10.) + 10 * i,
                           ms_level=2 if i % 4 == 3 else 1, precursor_mz=150. if i % 4 == 3 else None)
                      for i in range(8)])
    expected = scan_outcome(reference.iter_scans(path))
    assert expected[1] is None and expected[0]
    for qualified in (True, False):
        if not qualified:
            unqualified('pyteomics')
        threads = []

        def scans():
            for scan in reader.iter_scans(path):
                threads.extend(mzml_threads())
                yield scan
        assert scan_outcome(scans()) == expected
        # Otherwise there is no decoder pool: pyteomics decodes each array as it parses.
        assert bool(threads) is qualified

def test_deferred_mzml_close_joins_decoders(tmp_path, monkeypatch):
    from kmzcomet import reader
    monkeypatch.setattr(reader, 'DEFER_CHARS', 0)
    from scripts.benchmarks.comet_fixtures import cohort
    paths, _, _ = cohort(tmp_path, profile=True)
    scans = reader.iter_scans(paths[0])
    for _ in range(5):
        next(scans)
    assert mzml_threads()
    scans.close()
    assert not mzml_threads()


def interrupted_mzml(tmp_path, monkeypatch, hold):
    """Eight small synthetic spectra; every array decodes on a decoder thread, which then
    calls ``hold`` with the array's spectrum ordinal (its values start at 10 * ordinal)."""
    from kmzcomet import reader
    from Tests.Viewer.mzml_factory import write_mzml
    path = tmp_path / 'synthetic.mzML'
    write_mzml(path, [dict(rt=.01 * (i + 1), mz=np.arange(100., 110.) + 10 * i, intensity=np.arange(10.) + 10 * i)
                      for i in range(8)])
    original = reader.UnitMzML.decode_data_array

    def decode(*args, **kwargs):
        array = original(*args, **kwargs)
        hold(int(array[0]) // 10 % 10)
        return array
    monkeypatch.setattr(reader, 'DEFER_CHARS', 0)
    monkeypatch.setattr(reader.UnitMzML, 'decode_data_array', decode)
    reader.vocabulary()  # Loaded (and possibly downloaded) before any decoder starts.
    return reader, path


def joined_by(monkeypatch, joining, alive):
    """Set ``joining`` when a decoder pool shuts down; once its shutdown completes (not
    when an interrupt cuts it short), record the decoders still alive."""
    original = ThreadPoolExecutor.shutdown

    def shutdown(pool, *args, **kwargs):
        joining.set()
        original(pool, *args, **kwargs)
        alive.append(mzml_threads())
    monkeypatch.setattr(ThreadPoolExecutor, 'shutdown', shutdown)


def test_sigint_at_mzml_decoder_registration_joins_every_decoder(tmp_path, monkeypatch):
    """A real SIGINT right after the first decoder thread starts (before the executor
    registers it) is raised only once every decoder is joined, as in the converter."""
    started, joining, alive = threading.Event(), threading.Event(), []
    reader, path = interrupted_mzml(tmp_path, monkeypatch, lambda spectrum: joining.wait(5))
    joined_by(monkeypatch, joining, alive)
    original = threading.Thread.start

    def start(thread):
        original(thread)
        if thread.name.startswith('kmzcomet-mzml') and not started.is_set():
            started.set()
            signal.raise_signal(signal.SIGINT)
    monkeypatch.setattr(threading.Thread, 'start', start)
    with pytest.raises(KeyboardInterrupt):
        list(reader.iter_scans(path))
    assert started.is_set() and alive == [[]] and not mzml_threads()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_sigint_while_joining_mzml_decoders_is_raised_after_the_join(tmp_path, monkeypatch):
    joining, busy, alive, held, lock = threading.Event(), threading.Event(), [], [], threading.Lock()
    joined_by(monkeypatch, joining, alive)

    def hold(spectrum):
        if not spectrum:  # The first spectrum decodes; later ones wait for the join.
            return
        with lock:
            held.append(spectrum)
            count = len(held)
        if count == reader.DECODERS:
            busy.set()
        joining.wait(5)
        if count == 1:
            signal.raise_signal(signal.SIGINT)  # Delivered to the main thread, now joining.
        else:
            time.sleep(.3)  # Still running unless the join outlasts the SIGINT.
    reader, path = interrupted_mzml(tmp_path, monkeypatch, hold)
    scans = reader.iter_scans(path)
    assert next(scans).rt == .01
    assert busy.wait(5)  # Every decoder holds a later spectrum's array.
    with pytest.raises(KeyboardInterrupt):
        scans.close()
    assert alive == [[]] and not mzml_threads()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
