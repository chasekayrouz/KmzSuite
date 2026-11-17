"""Mechanical XML gating checks; synthetic arrays are not qualification evidence."""
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts.benchmarks.ms1_algorithms import gap_connections, guard_indices
from scripts.benchmarks.ms1_provenance import (
    CONTRACT, FAMILY, METHOD_PARAM, NS, SAMPLING_PARAM,
    annotate_experimental, inspection_connections, source_sampling_check,
)
from Tests.Viewer.mzml_factory import write_mzml


def marked_document(y=None):
    root = ET.fromstring(f'''<mzML xmlns="{NS[1:-1]}">
      <softwareList count="1"><software id="convert" version="0.1.0">
        <cvParam accession="MS:1000799" value="KmzConvert"/>
      </software></softwareList>
      <dataProcessingList count="1"><dataProcessing id="inspection">
        <processingMethod order="0" softwareRef="convert"/>
      </dataProcessing></dataProcessingList>
      <run><spectrumList count="1" defaultDataProcessingRef="inspection">
        <spectrum id="scan=1"><cvParam accession="MS:1000511" value="1"/>
          <cvParam accession="MS:1000128"/><cvParam accession="MS:1000130"/>
        </spectrum>
      </spectrumList></run></mzML>''')
    y = np.asarray(y if y is not None else [0, 0, 101, 250, 110, 0, 0, 0, 0, 0, 0, 101, 0, 0], dtype=float)
    mz = (10 + np.arange(len(y)) * .0001) ** 2
    method = root.find('.//' + NS + 'processingMethod')
    spectrum = root.find('.//' + NS + 'spectrum')
    annotate_experimental(method, spectrum, mz, y, sampling_family=FAMILY)
    idx = guard_indices(y, guards=1)
    return root, spectrum, mz[idx], y[idx], mz, y


def marker(node, name):
    return next(p for p in node.findall(NS + 'userParam') if p.get('name') == name)


def test_xml_roundtrip_links_frozen_method_without_mutating_arrays():
    root, spectrum, mz, y, _, _ = marked_document()
    original_mz, original_y = mz.copy(), y.copy()
    # Serialize/parse namespaced XML, as an actual metadata reader must do.
    root = ET.fromstring(ET.tostring(root))
    spectrum = root.find('.//' + NS + 'spectrum')
    expected = gap_connections(mz, factor=4, radius=8)
    np.testing.assert_array_equal(inspection_connections(root, spectrum, mz, y), expected)
    np.testing.assert_array_equal(inspection_connections(root, spectrum, mz, y), expected)
    np.testing.assert_array_equal(mz, original_mz)
    np.testing.assert_array_equal(y, original_y)
    assert json.loads(marker(root.find('.//' + NS + 'processingMethod'), METHOD_PARAM).get('value')) == CONTRACT


@pytest.mark.parametrize('change', ['unmarked', 'old_benchmark_marker', 'centroid', 'negative', 'ms2',
                                  'software', 'missing_version', 'unrelated_processing', 'unknown_version',
                                  'changed_factor', 'missing_source_check', 'bad_counts', 'bad_family',
                                  'duplicate_method', 'duplicate_source', 'malformed_json', 'duplicate_json_keys'])
def test_gate_declines_unsupported_ambiguous_or_unmarked_inputs(change):
    root, spectrum, mz, y, _, _ = marked_document()
    method = root.find('.//' + NS + 'processingMethod')
    m = marker(method, METHOD_PARAM)
    a = marker(spectrum, SAMPLING_PARAM)
    if change == 'unmarked': method.remove(m)
    elif change == 'old_benchmark_marker': m.set('name', 'KmzSuite spectrum benchmark')
    elif change in ('centroid', 'negative'):
        ET.SubElement(spectrum, NS + 'cvParam', accession='MS:1000127' if change == 'centroid' else 'MS:1000129')
    elif change == 'ms2': spectrum.find(NS + 'cvParam[@accession="MS:1000511"]').set('value', '2')
    elif change == 'software': root.find('.//' + NS + 'software/' + NS + 'cvParam').set('value', 'Other tool')
    elif change == 'missing_version': root.find('.//' + NS + 'software').set('version', '')
    elif change == 'unrelated_processing': spectrum.set('dataProcessingRef', 'other')
    elif change in ('unknown_version', 'changed_factor'):
        payload = json.loads(m.get('value'))
        if change == 'unknown_version': payload['version'] = 2
        else: payload['renderer']['factor'] = 2
        m.set('value', json.dumps(payload))
    elif change == 'missing_source_check': spectrum.remove(a)
    elif change in ('bad_counts', 'bad_family'):
        payload = json.loads(a.get('value'))
        if change == 'bad_counts': payload['retained_points'] += 1
        else: payload['sampling_family'] = 'arbitrary profile'
        a.set('value', json.dumps(payload))
    elif change == 'duplicate_method': method.append(deepcopy(m))
    elif change == 'duplicate_source': spectrum.append(deepcopy(a))
    elif change == 'malformed_json': m.set('value', '{')
    else: m.set('value', '{"version":1,"version":2}')
    assert inspection_connections(root, spectrum, mz, y) is None


def test_explicit_processing_reference_overrides_list_default_and_indexed_wrapper():
    root, spectrum, mz, y, _, _ = marked_document()
    root.find('.//' + NS + 'spectrumList').set('defaultDataProcessingRef', 'unmarked')
    assert inspection_connections(root, spectrum, mz, y) is None
    spectrum.set('dataProcessingRef', 'inspection')
    wrapper = ET.Element(NS + 'indexedmzML'); wrapper.append(root)
    np.testing.assert_array_equal(inspection_connections(wrapper, spectrum, mz, y), gap_connections(mz))


@pytest.mark.parametrize('values', [[], [101], [0, 101, 0], [0, 0, 100, 0]])
def test_empty_singleton_sparse_and_no_above_threshold_do_not_crash(values):
    root, spectrum, mz, y, _, _ = marked_document(values)
    c = inspection_connections(root, spectrum, mz, y)
    assert c is not None and c.shape == (max(len(mz) - 1, 0),)


def test_source_attestation_rejects_unknown_family_and_native_signal_break():
    mz = (10 + np.r_[np.arange(15), np.arange(100, 115)] * .0001) ** 2
    y = np.full(len(mz), 101.)
    with pytest.raises(ValueError, match='source-neighbor signal'):
        source_sampling_check(mz, y, sampling_family=FAMILY)
    with pytest.raises(ValueError, match='Unqualified'):
        source_sampling_check(mz, y, sampling_family='arbitrary TOF')
    y[14:16] = 0
    assert source_sampling_check(mz, y, sampling_family=FAMILY)['source_signal_breaks'] == 0


def test_shared_method_is_reused_and_existing_spectrum_marker_not_replaced():
    root, spectrum, _, _, mz, y = marked_document()
    method = root.find('.//' + NS + 'processingMethod')
    other = ET.Element(NS + 'spectrum')
    annotate_experimental(method, other, mz, y, sampling_family=FAMILY)
    assert len([p for p in method.findall(NS + 'userParam') if p.get('name') == METHOD_PARAM]) == 1
    before = ET.tostring(spectrum)
    with pytest.raises(ValueError, match='Existing source'):
        annotate_experimental(method, spectrum, mz, y, sampling_family=FAMILY)
    assert ET.tostring(spectrum) == before


def test_viewer_parser_preserves_method_and_source_attestation(tmp_path):
    from pyteomics import mzml as parser
    root, spectrum, m, y, _, _ = marked_document()
    path = tmp_path / 'mechanical-gate.mzML'
    write_mzml(path, [{'id': 'scan=1', 'rt': 1., 'mz': m, 'intensity': y}])
    output = ET.parse(path).getroot()
    output.insert(0, deepcopy(root.find(NS + 'softwareList')))
    output.insert(1, deepcopy(root.find(NS + 'dataProcessingList')))
    output.find('.//' + NS + 'spectrumList').set('defaultDataProcessingRef', 'inspection')
    target = output.find('.//' + NS + 'spectrum')
    target.append(deepcopy(marker(spectrum, SAMPLING_PARAM)))
    ET.ElementTree(output).write(path, encoding='utf-8', xml_declaration=True)
    with parser.MzML(str(path)) as reader:
        decoded = reader.get_by_id('scan=1')
        np.testing.assert_array_equal(decoded['m/z array'], m)
        np.testing.assert_array_equal(decoded['intensity array'], y)
        assert json.loads(decoded[SAMPLING_PARAM])['retained_points'] == len(m)
        processing = list(reader.iterfind('dataProcessing'))
        assert json.loads(processing[0]['processingMethod'][0][METHOD_PARAM]) == CONTRACT


def test_source_marker_precedes_structural_spectrum_children_without_changing_them():
    root, _, _, _, mz, y = marked_document()
    method = root.find('.//' + NS + 'processingMethod')
    target = ET.fromstring(f'<spectrum xmlns="{NS[1:-1]}"><cvParam accession="MS:1000511" value="1"/>'
                          '<scanList count="1"><scan/></scanList><binaryDataArrayList count="0"/></spectrum>')
    before = [ET.tostring(child) for child in target]
    annotate_experimental(method, target, mz, y, sampling_family=FAMILY)
    assert [child.tag[len(NS):] for child in target] == ['cvParam', 'userParam', 'scanList', 'binaryDataArrayList']
    assert [ET.tostring(child) for child in target if child.tag != NS + 'userParam'] == before
