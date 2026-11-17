"""Frozen D held-out qualification utilities; never imported by production apps."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from kmzconvert_core import writer
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import NS, iter_records, read_arrays
from scripts.benchmarks.ms1_algorithms import (
    gap_connections, guard_indices, shape_metrics, validate_arrays,
)
from scripts.benchmarks.ms1_mzml import InspectionReader, write_raw_inspection
from scripts.benchmarks.ms1_provenance import (
    CONTRACT, FAMILY, METHOD_PARAM, SAMPLING_PARAM, canonical,
    inspection_connections, source_sampling_check,
)


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def inventory(fixtures):
    files = sorted(Path(fixtures).iterdir())
    wiffs = [p for p in files if p.name.lower().endswith('.wiff')]
    scans = [p for p in files if p.name.lower().endswith('.wiff.scan')]
    if len(wiffs) != 5 or len(scans) != 5 or len(files) != 10:
        raise ValueError('Qualification requires exactly five pairs and ten files')
    if {p.name + '.scan' for p in wiffs} != {p.name for p in scans}:
        raise ValueError('Orphaned WIFF/SCAN member')
    rows = []
    for i, p in enumerate(wiffs, 1):
        for q, kind in ((p, 'wiff'), (p.with_name(p.name + '.scan'), 'wiff.scan')):
            with q.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            rows.append(dict(acquisition=f'heldout-{i:02d}', filename=q.name, type=kind,
                             bytes=q.stat().st_size, sha256=digest, pair=p.name))
    return wiffs, rows


def write_d(reader, destination):
    """Native isolated D export with the proposed method and source attestations.

    Refuse an attestation for incompatible source sampling, but still export the
    frozen source subset so a scientific counterexample can be measured. Patches
    are process-local, sequential and restored even if conversion fails.
    """
    original_metadata, original_spectrum = writer.metadata, writer.spectrum
    def metadata(adapted):
        elements = original_metadata(adapted)
        processing = next(e for e in elements if e.tag == 'dataProcessingList')
        writer.user(processing.find('.//processingMethod'), METHOD_PARAM, canonical(CONTRACT))
        return elements
    def spectrum(adapted, scan, index):
        result = original_spectrum(adapted, scan, index)
        if reader.experiments[scan.experiment - 1].ms_level == 1:
            mz, y = reader.arrays(scan)
            try:
                check = source_sampling_check(mz, y, sampling_family=FAMILY)
            except ValueError as error:
                marker = writer.user(result, 'Experimental source sampling rejection', str(error))
            else:
                marker = writer.user(result, SAMPLING_PARAM, canonical(check))
            # mzML's ParamGroup precedes scanList/precursorList/binaryDataArrayList.
            # Only this experimental annotation moves; stored arrays stay exact.
            result.remove(marker)
            first_child = next((i for i, child in enumerate(result)
                                if child.tag not in ('cvParam', 'userParam', 'referenceableParamGroupRef')), len(result))
            result.insert(first_child, marker)
        return result
    with patch.object(writer, 'metadata', metadata), patch.object(writer, 'spectrum', spectrum):
        return writer.write_mzml(InspectionReader(reader, 1), destination)


def exact_subset(mz, y, indices, candidate_mz, candidate_y):
    """Measure decoded-output deviations and enforce source identity, not tolerance."""
    if len(indices) != len(candidate_mz) or len(indices) != len(candidate_y):
        raise AssertionError('Retained count mismatch')
    if len(indices) and np.any(np.diff(indices) <= 0):
        raise AssertionError('Duplicate or unordered source indices')
    dm, dy = abs(candidate_mz - mz[indices]), abs(candidate_y - y[indices])
    if np.any(dm) or np.any(dy):
        raise AssertionError('Source measurement preservation failed')
    return dict(mz_max_absolute_deviation=float(dm.max(initial=0)),
                mz_median_absolute_deviation=float(np.median(dm)) if len(dm) else 0.,
                intensity_max_absolute_deviation=float(dy.max(initial=0)),
                intensity_median_absolute_deviation=float(np.median(dy)) if len(dy) else 0.)


def evaluate(mz, y, candidate_mz, candidate_y, guards):
    mz, y = validate_arrays(mz, y)
    start = time.perf_counter()
    idx = guard_indices(y, threshold=100, guards=guards)
    fidelity = exact_subset(mz, y, idx, candidate_mz, candidate_y)
    assert np.count_nonzero(candidate_y > 100) == np.count_nonzero(y > 100)
    connections = gap_connections(candidate_mz, factor=4, radius=8) if guards else np.ones(max(0, len(idx)-1), dtype=bool)
    prepare_seconds = time.perf_counter() - start
    result, peaks = shape_metrics(mz, y, idx, connections)
    result.update(fidelity)
    above = int(np.count_nonzero(y > 100))
    regions = int(np.count_nonzero((y > 100) & ~np.r_[False, (y > 100)[:-1]])) if len(y) else 0
    added = len(idx) - above
    assert 0 <= added <= 2 * regions
    deleted = np.diff(idx) > 1
    bridged = deleted & connections
    endpoint_area = np.diff(candidate_mz) * (candidate_y[:-1] + candidate_y[1:]) / 2
    result.update(reference_area=float(np.trapezoid(y, mz)), retained_regions=regions,
                  deleted_regions=int(deleted.sum()), guards_fraction_final=added / len(idx) if len(idx) else 0.,
                  guard_expansion_percent=100 * added / above if above else 0.,
                  plot_prepare_seconds=prepare_seconds,
                  residual_bridge_endpoint_area=float(endpoint_area[bridged].sum()),
                  residual_bridge_max_endpoint_area=float(endpoint_area[bridged].max(initial=0)),
                  exactly_100_points=int(np.count_nonzero(y == 100)),
                  no_above_threshold=int(above == 0), one_above_threshold=int(above == 1),
                  one_final_point=int(len(idx) == 1),
                  boundary_retained_regions=int(bool(len(y) and y[0] > 100)) + int(bool(len(y) and y[-1] > 100)),
                  many_tiny_regions=int(regions >= 10 and above / regions <= 2),
                  fwhm_reference_incomplete=sum(p['reference_fwhm'] is None for p in peaks),
                  fwhm_strong_apices=sum(p['apex_intensity'] >= 1000 for p in peaks),
                  fwhm_strong_incomplete=sum(p['apex_intensity'] >= 1000 and p['apparent_fwhm'] is None for p in peaks),
                  fwhm_max_absolute_relative_error=max((abs(p['fwhm_relative_error']) for p in peaks if p['fwhm_relative_error'] is not None), default=0.))
    return result, peaks


def summarize(rows, acquisition, condition):
    selected = [r for r in rows if r['condition'] == condition and (acquisition == 'pooled' or r['acquisition'] == acquisition)]
    def total(key):
        value = sum(r[key] for r in selected)
        return value.item() if isinstance(value, np.generic) else value
    def quantile(key, q):
        values = [r[key] for r in selected if r[key] is not None]
        return float(np.quantile(values, q)) if values else None
    result = dict(acquisition=acquisition, condition=condition, ms1_spectra=len(selected))
    sums = ('profile_points', 'retained_points', 'above_threshold_points', 'above_threshold_retained',
            'added_guard_points', 'deleted_regions', 'deleted_regions_bridged', 'render_discontinuities',
            'breaks_between_source_neighbors_with_signal', 'major_apices', 'major_apices_retained',
            'fwhm_incomplete', 'fwhm_reference_incomplete', 'fwhm_strong_apices', 'fwhm_strong_incomplete',
            'no_above_threshold', 'one_above_threshold', 'one_final_point', 'boundary_retained_regions',
            'many_tiny_regions', 'source_signal_breaks', 'gate_accepted', 'gate_declined', 'ink_iad', 'reference_area',
            'residual_bridge_endpoint_area', 'plot_prepare_seconds')
    result.update({k: total(k) for k in sums})
    result.update(guard_expansion_percent=100 * total('added_guard_points') / max(1, total('above_threshold_points')),
                  guards_fraction_final=total('added_guard_points') / max(1, total('retained_points')),
                  area_weighted_IAD_ratio=total('ink_iad') / max(1e-300, total('reference_area')),
                  bridge_rate=total('deleted_regions_bridged') / max(1, total('deleted_regions')))
    for key in ('ink_iad_normalized', 'ink_nrmse_apex', 'drawn_mz_fraction', 'omitted_reference_area_fraction',
                'threshold_edge_mae_counts', 'peak_region_nrmse_apex', 'guard_expansion_percent',
                'plot_prepare_seconds', 'median_absolute_fwhm_relative_error'):
        for label, q in (('median', .5), ('p90', .9), ('p99', .99), ('max', 1)):
            result[f'{key}_{label}'] = quantile(key, q)
    for key in ('mz_max_absolute_deviation', 'mz_median_absolute_deviation',
                'intensity_max_absolute_deviation', 'intensity_median_absolute_deviation',
                'bridge_max_mz', 'bridge_max_endpoint_intensity', 'residual_bridge_max_endpoint_area',
                'fwhm_max_absolute_relative_error'):
        result[key] = max(r[key] for r in selected)
    return result


def gate_file(path):
    """Resolve actual exported processing metadata for every spectrum, streaming."""
    rows = []
    root = None
    for event, element in ET.iterparse(path, events=('start', 'end')):
        if root is None:
            root = element
        if event != 'end' or element.tag != NS + 'spectrum':
            continue
        arrays = read_arrays(element)
        mask = inspection_connections(root, element, arrays['mz'], arrays['intensity'])
        level = int(element.find(NS + 'cvParam[@accession="MS:1000511"]').get('value'))
        rows.append({'index': int(element.get('index')), 'ms_level': level, 'accepted': mask is not None})
        if mask is not None:
            np.testing.assert_array_equal(mask, gap_connections(arrays['mz'], factor=4, radius=8))
        element.clear()
    return rows


def select_representatives(rows):
    """All-scan deterministic selection; ties use acquisition, then spectrum index."""
    d = [r for r in rows if r['condition'] == 'D']
    chosen = {}
    def add(row, reason):
        key = (row['acquisition'], row['index'])
        chosen.setdefault(key, []).append(reason)
    def pick(group, key, reverse=False):
        return sorted(group, key=lambda r: ((-r[key] if reverse else r[key]), r['acquisition'], r['index']))[0]
    for acquisition in sorted({r['acquisition'] for r in d}):
        group = [r for r in d if r['acquisition'] == acquisition]
        ordered = sorted(group, key=lambda r: (r['ink_iad_normalized'], r['index']))
        add(ordered[len(ordered)//2], 'ordinary median D IAD')
        add(pick(group, 'guard_expansion_percent', True), 'highest guard expansion')
        add(pick(group, 'ink_iad_normalized', True), 'worst D normalized IAD')
        add(pick(group, 'above_threshold_points'), 'sparsest threshold retention')
        add(pick(group, 'weak_peak_count', True), 'most threshold-adjacent apices')
        add(pick(group, 'cluster_score', True), 'densest apex cluster')
        add(pick(group, 'broad_fwhm', True), 'broadest major structure')
        add(pick(group, 'base_peak', True), 'strongest apex')
        add(pick(group, 'native_spacing_ratio', True), 'most irregular positive-source spacing')
        bridges = [r for r in group if r['deleted_regions_bridged']]
        if bridges:
            add(pick(bridges, 'residual_bridge_max_endpoint_area', True), 'largest residual bridge endpoint area')
        for key, reason in (('no_above_threshold', 'no retained points'), ('one_above_threshold', 'single above-threshold point'),
                            ('boundary_retained_regions', 'retained region at spectrum boundary'),
                            ('many_tiny_regions', 'many tiny retained regions'),
                            ('breaks_between_source_neighbors_with_signal', 'candidate source-neighbor signal break'),
                            ('source_signal_breaks', 'full-source sampling rejection')):
            group2 = [r for r in group if r[key]]
            if group2:
                add(pick(group2, key, True), reason)
    return chosen
