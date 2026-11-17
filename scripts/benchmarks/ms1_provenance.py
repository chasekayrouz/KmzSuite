"""Proposed inspection provenance gate; experimental tests only, no app imports.

The exporter must establish the source family from a qualified reader, not infer
it from a filename or from a generic mzML profile flag. This contract does not
provide scientific qualification or authenticate third-party metadata.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

from scripts.benchmarks.ms1_algorithms import gap_connections, guard_indices, validate_arrays

NS = '{http://psi.hupo.org/ms/mzml}'
METHOD_PARAM = 'KmzConvert MS1 inspection method'
SAMPLING_PARAM = 'KmzConvert MS1 source sampling check'
FAMILY = 'sciex-positive-tof-framed-profile-v1'
CONTRACT = {
    'schema': 'org.kmzsuite.ms1-inspection', 'version': 1, 'strategy': 'D-v1',
    'lossy': True, 'ms_level': 1, 'threshold': 100, 'comparator': '>',
    'source_boundary_guards': 1, 'centroiding': False, 'measurements': 'exact source subset',
    'sampling_family': FAMILY, 'renderer': {'coordinate': 'sqrt(m/z)', 'factor': 4,
    'radius': 8, 'side_quantile': 0.25, 'minimum_side_intervals': 3,
    'minimum_consistent_intervals': 2, 'consistency_fraction': 0.2,
    'unsupported_spacing': 'keep segment', 'singletons': 'observed dots'},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate provenance field')
        result[key] = value
    return result


def _read_marker(element, name):
    markers = [e for e in element.findall(NS + 'userParam') if e.get('name') == name]
    if len(markers) != 1:
        raise ValueError('Missing or ambiguous inspection marker')
    return json.loads(markers[0].get('value', ''), object_pairs_hook=_unique_object)


def _add_marker(element, name, value):
    if any(e.get('name') == name for e in element.findall(NS + 'userParam')):
        raise ValueError('Refuse to replace an existing inspection marker')
    marker = ET.SubElement(element, NS + 'userParam', name=name, value=canonical(value))
    if element.tag == NS + 'spectrum':
        # Spectrum ParamGroup must precede scanList and binaryDataArrayList.
        element.remove(marker)
        group_tags = {NS + tag for tag in ('cvParam', 'userParam', 'referenceableParamGroupRef')}
        at = next((i for i, child in enumerate(element) if child.tag not in group_tags), len(element))
        element.insert(at, marker)


def source_sampling_check(mz, intensity, *, sampling_family):
    """Check full source arrays using the frozen detector, without retuning it.

    A native neighboring signal interval that the detector would break rejects
    this attestation. Native zero-to-zero jumps from existing framing are allowed.
    Sparse/empty arrays still require an explicitly known source family.
    """
    if sampling_family != FAMILY:
        raise ValueError('Unqualified source sampling family')
    mz, intensity = validate_arrays(mz, intensity)
    connections = gap_connections(mz, factor=4, radius=8)
    signal = (intensity[:-1] > 0) | (intensity[1:] > 0)
    if np.any(~connections & signal):
        raise ValueError('Frozen gap rule breaks a source-neighbor signal interval')
    kept = guard_indices(intensity, threshold=100, guards=1)
    above = int(np.count_nonzero(intensity > 100))
    return {'sampling_family': FAMILY, 'source_points': len(mz),
            'retained_points': len(kept), 'above_threshold_points': above,
            'added_guard_points': len(kept) - above, 'source_signal_breaks': 0}


def annotate_experimental(method, spectrum, mz, intensity, *, sampling_family):
    """Add method/source attestations to temporary XML only; do not filter arrays.

    The caller binds the method to KmzConvert software and the spectrum's actual
    dataProcessingRef/defaultDataProcessingRef. Matching shared method annotations
    are reused; existing spectrum annotations are not replaced.
    """
    check = source_sampling_check(mz, intensity, sampling_family=sampling_family)
    # Validate both targets before mutation.
    method_marked = any(e.get('name') == METHOD_PARAM for e in method.findall(NS + 'userParam'))
    if method_marked and canonical(_read_marker(method, METHOD_PARAM)) != canonical(CONTRACT):
        raise ValueError('Incompatible existing method marker')
    if any(e.get('name') == SAMPLING_PARAM for e in spectrum.findall(NS + 'userParam')):
        raise ValueError('Existing source sampling marker')
    if not method_marked:
        _add_marker(method, METHOD_PARAM, CONTRACT)
    _add_marker(spectrum, SAMPLING_PARAM, check)
    return check


def inspection_connections(root, spectrum, mz, intensity):
    """Mask only for an explicitly marked, supported MS1; otherwise return None.

    None declines the experimental rendering path. This is fail-closed recognition,
    not a production fallback or a new viewer setting. Arrays are never mutated.
    """
    try:
        mz, intensity = validate_arrays(mz, intensity)
        if root.tag == NS + 'indexedmzML':
            root = root.find(NS + 'mzML')
        flags = {p.get('accession') for p in spectrum.findall(NS + 'cvParam')}
        levels = spectrum.findall(NS + 'cvParam[@accession="MS:1000511"]')
        if len(levels) != 1 or levels[0].get('value') != '1':
            return None
        if not {'MS:1000128', 'MS:1000130'} <= flags or flags & {'MS:1000127', 'MS:1000129'}:
            return None
        listing = root.find(NS + 'run/' + NS + 'spectrumList')
        ref = spectrum.get('dataProcessingRef') or listing.get('defaultDataProcessingRef')
        processing = [p for p in root.findall(NS + 'dataProcessingList/' + NS + 'dataProcessing') if p.get('id') == ref]
        if not ref or len(processing) != 1:
            return None
        methods = [p for p in processing[0].findall(NS + 'processingMethod')
                   if any(e.get('name') == METHOD_PARAM for e in p.findall(NS + 'userParam'))]
        if len(methods) != 1 or canonical(_read_marker(methods[0], METHOD_PARAM)) != canonical(CONTRACT):
            return None
        software = [p for p in root.findall(NS + 'softwareList/' + NS + 'software')
                    if p.get('id') == methods[0].get('softwareRef')]
        if len(software) != 1 or not software[0].get('version', '').strip():
            return None
        if not any(p.get('accession') == 'MS:1000799' and p.get('value') == 'KmzConvert'
                   for p in software[0].findall(NS + 'cvParam')):
            return None
        check = _read_marker(spectrum, SAMPLING_PARAM)
        counts = {'source_points', 'retained_points', 'above_threshold_points', 'added_guard_points', 'source_signal_breaks'}
        if not isinstance(check, dict) or set(check) != counts | {'sampling_family'}:
            return None
        if check['sampling_family'] != FAMILY or any(type(check[k]) is not int or check[k] < 0 for k in counts):
            return None
        above = int(np.count_nonzero(intensity > 100))
        if not (check['source_points'] >= check['retained_points'] == len(mz)
                and check['above_threshold_points'] == above
                and check['added_guard_points'] == len(mz) - above <= 2 * above
                and check['source_signal_breaks'] == 0):
            return None
        return gap_connections(mz, factor=4, radius=8)
    except (ValueError, TypeError, AttributeError):
        return None
