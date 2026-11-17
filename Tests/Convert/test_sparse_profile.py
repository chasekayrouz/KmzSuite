"""Versioned production contract, negative provenance gates, and GUI integration."""
from copy import deepcopy
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pyteomics import mzml as py_mzml

from Tests.Convert.test_writer_cli import synthetic
from kmzconvert.cli import main
from kmzconvert_core import __version__
from kmzconvert_core.reference import iter_records, NS
from kmzconvert_core.writer import write_mzml, spectrum
from kmzcore.sparse_profile import (
    FAMILY, FORMAT, METHOD_PARAMS, PARAMS, SAMPLING_PARAM, V1, V2, V3, Provenance, candidate_regions, canonical,
    event_unit, gap_connections, output_path, peak_indices, read_provenance, rendering_connections, rescue_selection,
    retained_indices, source_sampling_check,
)
from kmzviewer.mzml_index import index_ms1
from scripts.benchmarks.ms1_algorithms import (event_peak_indices, event_rescue_indices, guard_indices,
                                              gap_connections as frozen_gaps)


def kept(y):
    return peak_indices(y)[0]
from scripts.validation.check_mzml import check


@pytest.mark.parametrize('values, expected', [
    ([0, 100, 101, 200, 100, 0], [1, 2, 3, 4]),
    ([101, 0, 0, 101], [0, 1, 2, 3]),
    ([0, 101, 0, 101, 0], [0, 1, 2, 3, 4]),
    ([0, 100, 99], []), ([], []), ([101], [0]), ([100], []),
    ([101, 101, 0, 0, 0, 101], [0, 1, 2, 4, 5]),
])
def test_fixed_guard_edges_and_threshold(values, expected):
    y = np.array(values, dtype=float)
    before = y.copy()
    np.testing.assert_array_equal(retained_indices(y), expected)
    np.testing.assert_array_equal(retained_indices(y), guard_indices(y, guards=1))
    np.testing.assert_array_equal(y, before)


@pytest.mark.parametrize('values, expected', [
    # Unit 10: 30-point region (>20) triggers above 200; two-point and low runs drop.
    ([0, 10, 10, 30, 250, 300, 30, 10, 0, 10, 25, 500, 10, 0, 1], [2, 3, 4, 5, 6, 7]),
    ([10, 10, 20, 200, 20, 10], []),            # exactly 2u is outside; exactly 20u does not trigger
    ([10, 10, 21, 200.5, 21, 10], [1, 2, 3, 4, 5]),
    ([300, 300, 300, 10, 10, 10, 10], [0, 1, 2, 3]),  # array edge needs no outer guard
    ([10, 10, 10, 30, 30, 30], []),
    ([10, 10, 10, 10, 500, 500, 500, 10, 500, 500, 500, 10], list(range(3, 12))),  # shared guard
    ([], []), ([0, 0], []), ([5.], []),
])
def test_v2_event_peak_regions(values, expected):
    y = np.array(values, dtype=float)
    before = y.copy()
    np.testing.assert_array_equal(kept(y), expected)
    np.testing.assert_array_equal(kept(y), event_peak_indices(y)[0])
    np.testing.assert_array_equal(y, before)


def test_v2_unit_is_most_frequent_positive_value_and_scale_invariant():
    assert event_unit(np.array([0, 0, 0, 52, 52, 78, 105])) == 52
    assert event_unit(np.array([30, 30, 10, 10, 500])) == 10  # ties use the smaller value
    assert event_unit(np.zeros(4)) == 0
    rng = np.random.default_rng(7)
    for _ in range(60):
        unit = float(rng.choice([5, 26, 252]))
        y = unit * rng.choice([0, 1, 1, 1, 2, 3, 5, 25, 40], 400) * (rng.random(400) > .3)
        indices, observed, selected, regions = peak_indices(y)
        reference, reference_unit = event_peak_indices(y)
        np.testing.assert_array_equal(indices, reference)
        assert observed == reference_unit
        # A spectrum-wide rescaling (gain or attenuation change) keeps the same points.
        np.testing.assert_array_equal(peak_indices(y * 9.5)[0], indices)
        assert selected >= 3 * regions and len(indices) - selected <= 2 * regions


def _v3(spectra):
    """Production v3 indices and attestations for one MS1 sequence."""
    chosen = rescue_selection([candidate_regions(m, y) for m, y in spectra])
    return [source_sampling_check(m, y, sampling_family=FAMILY, selection=c) for (m, y), c in zip(spectra, chosen)]


AXIS = (10 + np.arange(40) * .0001) ** 2


def _rescue_run():
    """Unit 10. Peak A seeds in scan 2 and is followed to 12u and 9u neighbors but not
    to a 7u edge; peak B never reaches 20u; peak C seeds alone in scan 3."""
    def scan(a=None, b=None, c=None):
        y = np.full(40, 10.)
        for start, apex in ((5, a), (20, b), (30, c)):
            if apex:
                y[start:start + 3] = [apex / 2, apex, apex / 2]
        return y
    return [scan(a=120, b=150), scan(a=300, b=150), scan(a=90, b=150, c=500), scan(a=70, b=150)]


def test_v3_rescue_follows_seeded_peaks_across_spectra():
    run = [(AXIS, y) for y in _rescue_run()]
    results = _v3(run)
    a, c = list(range(4, 9)), list(range(29, 34))
    for (indices, _), expected in zip(results, [a, a, a + c, []]):
        np.testing.assert_array_equal(indices, expected)
    assert [check['rescued_regions'] for _, check in results] == [1, 0, 1, 0]
    assert [check['selected_regions'] for _, check in results] == [1, 1, 2, 0]
    for (indices, check), (reference, unit) in zip(results, event_rescue_indices(run)):
        np.testing.assert_array_equal(indices, reference)
        assert check['event_unit'] == unit == 10
    # A spectrum alone keeps only its seeds, as v2 would.
    for m, y in run:
        np.testing.assert_array_equal(source_sampling_check(m, y, sampling_family=FAMILY)[0], peak_indices(y)[0])
    # Links need overlapping m/z spans in consecutive spectra: shifting scan 1 by
    # more than a peak width, or inserting a spectrum without the peak, breaks them.
    shifted = [(AXIS * 1.001, run[0][1])] + run[1:]
    np.testing.assert_array_equal(_v3(shifted)[0][0], [])
    gap = [run[0], (AXIS, np.full(40, 10.)), *run[1:]]
    np.testing.assert_array_equal(_v3(gap)[0][0], [])
    with pytest.raises(ValueError, match='selection'):
        source_sampling_check(AXIS, run[0][1], sampling_family=FAMILY, selection=np.ones(5, bool))
    with pytest.raises(ValueError, match='v3'):
        source_sampling_check(AXIS, run[0][1], sampling_family=FAMILY, sparse_format=V2, selection=np.ones(1, bool))


def test_v3_matches_loop_oracle_and_is_gain_invariant():
    rng = np.random.default_rng(11)
    for _ in range(25):
        spectra = []
        for j in range(8):
            unit = float(rng.choice([5, 26, 252]))
            y = unit * rng.choice([0, 1, 1, 1, 2, 3], 300).astype(float)
            for start in range(10, 290, 23):
                if rng.random() < .6:
                    y[start:start + 4] = unit * rng.choice([3, 6, 9, 12, 18, 25, 40]) * np.array([.5, 1, .9, .4])
            spectra.append(((10 + (np.arange(300) + rng.integers(-2, 3)) * .0001) ** 2, y))
        results = _v3(spectra)
        for (indices, check), (reference, unit) in zip(results, event_rescue_indices(spectra)):
            np.testing.assert_array_equal(indices, reference)
            assert check['event_unit'] == unit and check['rescued_regions'] <= check['selected_regions']
        # Each spectrum's own gain may change; selection follows its event unit.
        rescaled = _v3([(m, y * g) for (m, y), g in zip(spectra, rng.choice([.5, 3, 9.5], len(spectra)))])
        for (a, _), (b, _) in zip(results, rescaled):
            np.testing.assert_array_equal(a, b)


def test_v3_writer_selects_across_ms1_scans_and_skips_ms2(synthetic, tmp_path):
    from dataclasses import replace
    synthetic.sparse_profile_sampling_family = FAMILY
    ms1, ms2 = synthetic.scans
    synthetic.scans = [replace(ms1, cycle=1), replace(ms2, cycle=1)] + [replace(ms1, cycle=c, rt_seconds=c) for c in (2, 3, 4)]
    run = _rescue_run()
    synthetic.arrays = lambda scan: ((AXIS.copy(), run[scan.cycle - 1].copy()) if scan.experiment == 1
                                     else (np.array([100., 100.001]), np.array([5., 7.])))
    synthetic.tic_arrays = lambda: (np.arange(5.), np.ones(5), np.array([1, 2, 1, 1, 1]))
    seen = []
    path = write_mzml(synthetic, tmp_path/'run.mzML', lambda done, total: seen.append(done), profile='sparse')
    assert seen[0] == 0
    records = [r for r in iter_records(path) if r['ms_level'] == 1]
    table = index_ms1(path)
    for row, (record, (reference, _)) in enumerate(zip(records, event_rescue_indices([(AXIS, y) for y in run]), strict=True)):
        np.testing.assert_array_equal(record['arrays']['intensity'], run[row][reference])
        check = table.sparse_profile[row]
        assert check is not None and check['rescued_regions'] == [1, 0, 1, 0][row]
        assert rendering_connections(check, record['arrays']['mz'], record['arrays']['intensity']) is not None
    assert read_provenance(path).formats == {V3: V3}


@pytest.mark.parametrize('name, expected', [
    ('sample', 'sample.sp.mzML'), ('sample.mzML', 'sample.sp.mzML'),
    ('sample.sp.mzML', 'sample.sp.mzML'), ('sample.MZML', 'sample.sp.mzML'),
    ('sample.SP.MZML', 'sample.sp.mzML'), ('sample.v2.mzML', 'sample.v2.sp.mzML'),
    ('sample.mzML.sp.mzML', 'sample.sp.mzML'),
])
def test_sparse_filename(name, expected):
    assert output_path(Path('folder')/name, 'sparse') == Path('folder')/expected
    assert output_path(name) == Path(name)


@pytest.fixture
def output(synthetic, tmp_path):
    synthetic.sparse_profile_sampling_family = FAMILY
    y = np.array([0., 10., 10., 30., 250., 300., 30., 10., 0., 10., 25., 500., 10., 0., 1.])
    m = (10 + np.arange(len(y)) * .0001) ** 2
    synthetic.arrays = lambda _: (m.copy(), y.copy())
    full = write_mzml(synthetic, tmp_path/'full.mzML')
    sparse = write_mzml(synthetic, tmp_path/'sparse.mzML', profile='sparse')
    return full, sparse, m, y, synthetic


@pytest.fixture
def legacy(output, tmp_path):
    *_, m, y, synthetic = output
    path = write_mzml(synthetic, tmp_path/'legacy.mzML', profile='sparse', sparse_format=V1)
    return path, m, y


@pytest.fixture
def legacy_v2(output, tmp_path):
    *_, m, y, synthetic = output
    return write_mzml(synthetic, tmp_path/'legacy-v2.mzML', profile='sparse', sparse_format=V2), m, y


def test_full_default_and_sparse_exact_ms1_only(output, tmp_path):
    full, sparse, m, y, reader = output
    assert sparse.name == 'sparse.sp.mzML'
    explicit = write_mzml(reader, tmp_path/'explicit.mzML', profile='full')
    assert explicit.read_bytes() == full.read_bytes()
    roots = [ET.parse(p).getroot() for p in (full, sparse)]
    a, b = [list(iter_records(p)) for p in (full, sparse)]
    keep = kept(y)
    np.testing.assert_array_equal(keep, [2, 3, 4, 5, 6, 7])
    for kind, values in [('mz', m), ('intensity', y)]:
        np.testing.assert_array_equal(a[0]['arrays'][kind], values)
        np.testing.assert_array_equal(b[0]['arrays'][kind], values[keep])
        np.testing.assert_array_equal(a[1]['arrays'][kind], b[1]['arrays'][kind])
    for path in ('fileDescription', 'softwareList', 'instrumentConfigurationList', 'run/chromatogramList'):
        query = './/' + '/'.join(NS + part for part in path.split('/'))
        assert ET.tostring(roots[0].find(query)) == ET.tostring(roots[1].find(query))
    # MS2 is structurally identical, including all metadata and binary arrays.
    assert ET.tostring(roots[0].findall('.//'+NS+'spectrum')[1]) == ET.tostring(roots[1].findall('.//'+NS+'spectrum')[1])
    assert not read_provenance(full).methods
    assert roots[1].find('.//'+NS+'software[@id="kmzconvert_core"]').get('version') == __version__
    with py_mzml.MzML(str(sparse)) as ordinary_reader:
        assert len(ordinary_reader) == 2
        np.testing.assert_array_equal(ordinary_reader[0]['m/z array'], m[keep])
    assert roots[1].findall('.//'+NS+'cvParam[@accession="MS:1000128"]')
    assert not roots[1].findall('.//'+NS+'cvParam[@accession="MS:1000127"]')


def test_sparse_schema_and_cv_when_cached(output):
    schema, cv = Path('.cache/standards/mzML1.1.0_idx.xsd'), Path('.cache/standards/psi-ms.obo')
    if not schema.exists() or not cv.exists():
        pytest.skip('Optional official schemas/CV not cached')
    result = check(output[1], schema, cv)
    assert result['psi_ms_names_valid'] and result['index']['checksum_valid']


@pytest.mark.parametrize('values', [[], [10.], [0., 10., 0.], [10., 300., 300., 300.],
                                    [300., 300., 300., 10., 10., 10., 10., 300., 300., 300.]])
def test_sparse_empty_and_boundary_arrays_roundtrip(synthetic, tmp_path, values):
    synthetic.sparse_profile_sampling_family = FAMILY
    y = np.asarray(values)
    m = (10 + np.arange(len(y))*.0001)**2
    synthetic.arrays = lambda _: (m, y)
    output = write_mzml(synthetic, tmp_path/'edge.mzML', profile='sparse')
    record = next(iter_records(output))
    keep = event_peak_indices(y)[0]
    np.testing.assert_array_equal(record['arrays']['mz'], m[keep])
    np.testing.assert_array_equal(record['arrays']['intensity'], y[keep])
    table = index_ms1(output)
    assert table.sparse_profile[0] is not None
    assert rendering_connections(table.sparse_profile[0], m[keep], y[keep]) is not None


def _elements(sparse, sparse_format=FORMAT):
    root = ET.parse(sparse).getroot()
    s = root.find('.//'+NS+'spectrum')
    method = root.find('.//'+NS+'dataProcessing[@id="'+sparse_format+'"]/'+NS+'processingMethod[@order="1"]')
    return root, s, method


def test_recognition_roundtrip_and_filename_independence(output, tmp_path):
    full, sparse, m, y, _ = output
    for path, expected in [(sparse, True), (full, False)]:
        table = index_ms1(path, chunk_size=97)
        assert bool(table.sparse_profile[0]) is expected
    renamed = tmp_path/'renamed.mzML'
    renamed.write_bytes(sparse.read_bytes())
    assert index_ms1(renamed).sparse_profile[0]
    misleading = tmp_path/'ordinary.sp.mzML'
    misleading.write_bytes(full.read_bytes())
    assert index_ms1(misleading).sparse_profile == [None]
    root, s, method = _elements(sparse)
    assert {e.get('name'): e.get('value') for e in method.findall(NS+'userParam')} == PARAMS == METHOD_PARAMS[V3]
    assert [c.get('accession') for c in method.findall(NS+'cvParam')] == ['MS:1001486']
    check = Provenance(root, 'conversion').sampling(s)
    assert check == dict(sampling_family=FAMILY, source_points=len(y), event_unit=10.0, retained_points=6,
                         selected_points=4, selected_regions=1, rescued_regions=0, added_guard_points=2,
                         source_signal_breaks=0)
    mask = rendering_connections(check, m[kept(y)], y[kept(y)])
    np.testing.assert_array_equal(mask, frozen_gaps(m[kept(y)], factor=4, radius=8))
    del s.attrib['dataProcessingRef']
    assert Provenance(root, V3).sampling(s) == check


def test_v2_files_remain_recognized_and_rendered(legacy_v2):
    path, m, y = legacy_v2
    root, s, method = _elements(path, V2)
    assert {e.get('name'): e.get('value') for e in method.findall(NS+'userParam')} == METHOD_PARAMS[V2]
    check = index_ms1(path).sparse_profile[0]
    assert check is not None and 'rescued_regions' not in check
    keep = event_peak_indices(y)[0]
    np.testing.assert_array_equal(next(iter_records(path))['arrays']['intensity'], y[keep])
    np.testing.assert_array_equal(rendering_connections(check, m[keep], y[keep]), frozen_gaps(m[keep], factor=4, radius=8))
    # A v2 region below 20u is not admissible even where v3 would accept a rescue.
    assert rendering_connections(check, m[keep], y[keep] / 2) is None
    # A v2 attestation under v3 parameters fails closed.
    for e in method.findall(NS+'userParam'):
        method.remove(e)
    for name, value in METHOD_PARAMS[V3].items():
        ET.SubElement(method, NS+'userParam', name=name, value=value)
    assert Provenance(root, 'conversion').sampling(s) is None


def test_v1_files_remain_recognized_and_rendered(legacy):
    path, m, y = legacy
    root, s, method = _elements(path, V1)
    assert {e.get('name'): e.get('value') for e in method.findall(NS+'userParam')} == METHOD_PARAMS[V1]
    record = next(iter_records(path))
    np.testing.assert_array_equal(record['arrays']['intensity'], y[retained_indices(y)])
    table = index_ms1(path)
    check = table.sparse_profile[0]
    assert check is not None and 'above_threshold_points' in check
    keep = guard_indices(y, guards=1)
    np.testing.assert_array_equal(rendering_connections(check, m[keep], y[keep]), frozen_gaps(m[keep], factor=4, radius=8))
    # Formats are not interchangeable: a v1 attestation under v2 parameters fails closed.
    for e in method.findall(NS+'userParam'):
        method.remove(e)
    for name, value in METHOD_PARAMS[V2].items():
        ET.SubElement(method, NS+'userParam', name=name, value=value)
    assert Provenance(root, 'conversion').sampling(s) is None
    assert Provenance(root).methods == set()


@pytest.mark.parametrize('field', list(PARAMS))
def test_malformed_or_future_method_fails_closed(output, field):
    _, path, *_ = output
    root, s, method = _elements(path)
    method.find(NS+'userParam[@name="'+field+'"]').set('value', '4' if field.endswith('version') else 'wrong')
    assert Provenance(root, 'conversion').sampling(s) is None


@pytest.mark.parametrize('failure', ['threshold', 'units', 'missing-threshold', 'counts'])
def test_v1_provenance_fails_closed(legacy, failure):
    path, *_ = legacy
    root, s, method = _elements(path, V1)
    threshold = method.find(NS+'cvParam[@accession="MS:1000787"]')
    if failure == 'threshold': threshold.set('value', '99')
    elif failure == 'units': threshold.set('unitCvRef', 'UO')
    elif failure == 'missing-threshold': method.remove(threshold)
    else:
        import json
        marker = s.find(NS+'userParam[@name="'+SAMPLING_PARAM+'"]')
        attestation = json.loads(marker.get('value'))
        attestation['retained_points'] += 1
        marker.set('value', canonical(attestation))
    assert Provenance(root, 'conversion').sampling(s) is None


@pytest.mark.parametrize('failure', ['duplicate', 'software', 'ref', 'profile', 'polarity', 'ms2', 'threshold', 'family', 'counts', 'breaks',
                                     'json', 'missing', 'missing-id', 'extra-action', 'unit-negative', 'unit-text', 'unit-bool',
                                     'regions', 'guards', 'rescued', 'rescued-text', 'v1-keys', 'v2-keys'])
def test_provenance_and_sampling_fail_closed(output, failure):
    _, path, *_ = output
    root, s, method = _elements(path)
    marker = s.find(NS+'userParam[@name="'+SAMPLING_PARAM+'"]')
    import json
    attestation = json.loads(marker.get('value'))
    if failure == 'duplicate': method.append(deepcopy(method.find(NS+'userParam')))
    elif failure == 'software': method.set('softwareRef', 'acquisition')
    elif failure == 'ref': s.set('dataProcessingRef', 'absent')
    elif failure == 'profile': s.find(NS+'cvParam[@accession="MS:1000128"]').set('accession', 'MS:1000127')
    elif failure == 'polarity': s.find(NS+'cvParam[@accession="MS:1000130"]').set('accession', 'MS:1000129')
    elif failure == 'ms2': s.find(NS+'cvParam[@accession="MS:1000511"]').set('value', '2')
    elif failure == 'threshold': ET.SubElement(method, NS+'cvParam', accession='MS:1000787', cvRef='MS', name='inclusive low intensity threshold', value='100')
    elif failure == 'family': attestation['sampling_family'] = 'arbitrary-qtof'
    elif failure == 'counts': attestation['retained_points'] += 1
    elif failure == 'breaks': attestation['source_signal_breaks'] = 1
    elif failure == 'json': marker.set('value', '{"source_points":1,"source_points":2}')
    elif failure == 'missing': s.remove(marker)
    elif failure == 'missing-id': root.find('.//'+NS+'software[@id="kmzconvert_core"]').attrib.pop('id')
    elif failure == 'extra-action': ET.SubElement(method, NS+'cvParam', accession='MS:1000035', cvRef='MS', name='peak picking')
    elif failure == 'unit-negative': attestation['event_unit'] = -10.
    elif failure == 'unit-text': attestation['event_unit'] = '10'
    elif failure == 'unit-bool': attestation['event_unit'] = True
    elif failure == 'regions': attestation['selected_regions'] = 2
    elif failure == 'guards': attestation.update(added_guard_points=3, selected_points=3)
    elif failure == 'rescued': attestation['rescued_regions'] = 2
    elif failure == 'rescued-text': attestation['rescued_regions'] = '0'
    elif failure == 'v2-keys': del attestation['rescued_regions']
    elif failure == 'v1-keys':
        attestation = {k: attestation[k] for k in ('sampling_family', 'source_points', 'retained_points', 'added_guard_points', 'source_signal_breaks')}
        attestation['above_threshold_points'] = 4
    if failure in ('family', 'counts', 'breaks') or failure.startswith(('unit-', 'regions', 'guards', 'rescued', 'v1-', 'v2-')):
        marker.set('value', canonical(attestation))
    assert Provenance(root, 'conversion').sampling(s) is None


def test_array_sampling_and_source_family_fail_closed(output):
    _, path, m, y, _ = output
    root, s, _ = _elements(path)
    check = Provenance(root).sampling(s)
    keep = kept(y)
    assert rendering_connections(check, m[keep], y[keep]) is not None
    assert rendering_connections(check, m[keep][::-1], y[keep]) is None
    assert rendering_connections(check, m[keep], np.zeros(len(keep))) is None
    assert rendering_connections(check, m[keep], y[keep] / 2) is None    # apex 15u, region split below 2u
    weak = y[keep].copy(); weak[2:4] = 120, 150                          # same region, apex 15u
    assert rendering_connections(check, m[keep], weak) is None           # not attested as rescued
    assert rendering_connections(dict(check, rescued_regions=1), m[keep], weak) is not None
    weak[2:4] = 60, 70
    assert rendering_connections(dict(check, rescued_regions=1), m[keep], weak) is None  # 7u: below the floor
    low = y[keep].copy(); low[2] = 10
    assert rendering_connections(check, m[keep], low) is None           # region split below the rule
    assert rendering_connections(check, m[keep][:-1], y[keep][:-1]) is None
    with pytest.raises(ValueError, match='family'):
        source_sampling_check(m, y, sampling_family='arbitrary-tof')
    # A real signal discontinuity in a full source prevents export attestation.
    bad = np.r_[np.arange(10, 10.01, .001), np.arange(11, 11.01, .001)] ** 2
    with pytest.raises(ValueError, match='source-neighbor'):
        source_sampling_check(bad, np.full(len(bad), 200.), sampling_family=FAMILY)


def test_frozen_renderer_equivalence():
    rng = np.random.default_rng(13)
    for _ in range(40):
        m = (10 + np.cumsum(rng.choice([.0001, .0002, .002, .1], 180))) ** 2
        np.testing.assert_array_equal(gap_connections(m), frozen_gaps(m, factor=4, radius=8))


def test_cli_sparse_normalizes_and_emits_only_choice(output, tmp_path, monkeypatch):
    _, _, _, _, reader = output
    reader.calibration_resolutions = []; reader.nominal_window_overhangs = {}; reader.close = lambda: None
    monkeypatch.setattr('kmzconvert.cli.WiffReader', lambda _: reader)
    monkeypatch.chdir(tmp_path)
    assert main(['example.wiff', '--profile', 'sparse']) == 0
    assert (tmp_path/'example.sp.mzML').exists() and not (tmp_path/'example.mzML').exists()
    assert main(['example.wiff', '--profile', 'sparse', '-o', 'chosen.mzML']) == 0
    assert (tmp_path/'chosen.sp.mzML').exists() and not (tmp_path/'chosen.mzML').exists()


def test_production_viewer_badge_and_zero_baseline_curve(output):
    from PySide6.QtWidgets import QApplication
    from kmzviewer.gui.app import MainWindow
    import pyqtgraph as pg
    qapp = QApplication.instance() or QApplication([])
    full, sparse, *_ = output
    window = MainWindow()
    try:
        assert window.load_paths_sync([str(full), str(sparse)]) == []
        assert window.tree.topLevelItem(0).text(0) == full.name
        assert window.tree.topLevelItem(1).text(0).startswith('SP · ')
        assert 'Sparse Profile' in window.tree.topLevelItem(1).toolTip(0)
        loaded = list(window.session.files.values())[1]
        window._show_walk_index(loaded, loaded.traces[0], 0)
        deadline = time.monotonic() + 5
        while 'loading' in window.spectrum.caption.text() and time.monotonic() < deadline:
            qapp.processEvents(); time.sleep(.01)
        items = window.spectrum.plot.listDataItems()
        assert items[0].opts['connect'] == 'all'
        assert items[0].yData[0] == items[0].yData[-1] == 0
        loaded = list(window.session.files.values())[0]
        window.inspect_rt(loaded.traces[0].trace_id, float(loaded.scans.rt_min[0]))
        deadline = time.monotonic() + 5
        while 'loading' in window.spectrum.caption.text() and time.monotonic() < deadline:
            qapp.processEvents(); time.sleep(.01)
        assert window.spectrum.plot.listDataItems()[0].opts['connect'] == 'auto'
        assert len(window.spectrum._frames) == 2
        assert window.spectrum._curves[list(window.session.files)[1]].opts['connect'] == 'all'
        # Singleton retains its observed dot and receives a display-only stem.
        window.spectrum.show_scan(np.array([100.]), np.array([101.]), False, 'SP', connections=np.array([], bool))
        assert any(isinstance(i, pg.ScatterPlotItem) and len(i.data) for i in window.spectrum.plot.getViewBox().addedItems)
        np.testing.assert_array_equal(window.spectrum._curves[0].yData, [0., 101., 0.])
    finally:
        window.close(); qapp.processEvents()


def test_worker_bridges_real_sparse_output_without_modifying_file_or_cache(synthetic, tmp_path):
    import hashlib
    from PySide6.QtWidgets import QApplication
    from kmzviewer.gui.app import MainWindow

    synthetic.sparse_profile_sampling_family = FAMILY
    mz = (10 + np.arange(180) * .0001) ** 2
    intensity = np.full(180, 40.)
    intensity[20:32] = intensity[130:142] = np.r_[np.arange(1, 7), np.arange(6, 0, -1)] * 400.
    synthetic.arrays = lambda _: (mz.copy(), intensity.copy())
    path = write_mzml(synthetic, tmp_path / 'two-peaks.mzML', profile='sparse')
    before = hashlib.sha256(path.read_bytes()).digest()
    keep = kept(intensity)
    assert len(keep) == 28
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert not window.load_paths_sync([str(path)])
        loaded = next(iter(window.session.files.values()))
        window.inspect_rt(loaded.traces[0].trace_id, float(loaded.scans.rt_min[0]))
        deadline = time.monotonic() + 5
        while window._inflight_comparison is not None and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.005)
        assert window._inflight_comparison is None
        frame = window.spectrum._frames[0]
        assert np.count_nonzero(~frame.connections) == 1
        np.testing.assert_array_equal(frame.mz, mz[keep])
        np.testing.assert_array_equal(frame.intensity, intensity[keep])
        cached = window._spectrum_worker._cache.get(str(path), int(loaded.scans.spectrum_index[0]))
        np.testing.assert_array_equal(cached[0], mz[keep])
        np.testing.assert_array_equal(cached[1], intensity[keep])
        curve = window.spectrum._curves[loaded.file_id]
        left = np.flatnonzero(~frame.connections)[0]
        inside = (curve.xData >= frame.mz[left]) & (curve.xData <= frame.mz[left + 1])
        np.testing.assert_array_equal(curve.yData[inside], [40., 0., 0., 40.])
        assert len(curve.xData) == len(keep) + 4
        assert curve.opts['connect'] == 'all'
        assert hashlib.sha256(path.read_bytes()).digest() == before
    finally:
        window.close(); app.processEvents()
