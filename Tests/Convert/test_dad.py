import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from kmzconvert_core.dad import decode_dad, read_dad
from kmzconvert_core.binary import FormatError
from kmzconvert_core.writer import write_mzml
from kmzconvert_core.reference import iter_records
from Tests.dad_fixtures import sidecars, acquisition
from scripts.validation.check_mzml import check, check_index


def test_native_decoding_preserves_signed_values_times_and_wavelengths():
    scan, index, streams, expected = sidecars()
    actual = decode_dad(scan, index, streams)
    for field in ('wavelength', 'rt_min', 'absorbance_mau', 'total_mau'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.absorbance_mau.min() < 0


@pytest.mark.parametrize('damage', ['scan-length', 'index-length', 'scan-magic', 'index-magic', 'index-offset',
                                  'time', 'nan', 'total', 'wavelength', 'duplicate-array', 'duplicate-time', 'channels'])
def test_guarded_layout_rejects_corruption(damage):
    scan, index, streams, _ = sidecars()
    scan, index = bytearray(scan), bytearray(index)
    header_key = next(k for k in streams if k.endswith('DADSampleHeader'))
    header = bytearray(streams[header_key])
    if damage == 'scan-length': scan.pop()
    elif damage == 'index-length': index.pop()
    elif damage == 'scan-magic': scan[0] = 0
    elif damage == 'index-magic': index[0] = 0
    elif damage == 'index-offset': struct.pack_into('<Q', index, 56, 8)
    elif damage == 'time': struct.pack_into('<d', index, 52+18, 9.)
    elif damage == 'nan': struct.pack_into('<d', scan, 44, float('nan'))
    elif damage == 'total': struct.pack_into('<d', index, 52+26, 900.)
    elif damage == 'wavelength': struct.pack_into('<f', header, 54, 999.)
    elif damage == 'channels': struct.pack_into('<I', header, 42, 2)
    else:
        key = next(k for k in streams if k.endswith('DADRealTimeData' if damage == 'duplicate-array' else 'DADIdx'))
        value = bytearray(streams[key]); struct.pack_into('<f', value, 32, 111.); streams[key] = value
    streams[header_key] = header
    with pytest.raises(FormatError):
        decode_dad(scan, index, streams)


def test_absent_incomplete_and_broken_sidecars_are_nonfatal(tmp_path):
    entry = tmp_path/'sample.wiff'
    scan, index, streams, _ = sidecars()
    data, state, note = read_dad(entry, streams)
    assert data is None and state == 'absent' and 'warning' not in note.lower()
    entry.with_suffix('.dad.scan').write_bytes(scan)
    data, state, note = read_dad(entry, streams)
    assert data is None and state == 'unreadable' and 'sample.dad.sidx' in note
    entry.with_suffix('.dad.sidx').write_bytes(b'broken')
    assert read_dad(entry, streams)[0] is None
    entry.with_suffix('.dad.sidx').write_bytes(index)
    data, state, note = read_dad(entry, streams)
    assert state == 'included' and len(data.sources) == 2


@pytest.mark.parametrize('profile', ['full', 'sparse'])
def test_mzml_optical_units_offsets_and_ms_arrays(tmp_path, profile):
    reader = acquisition(tmp_path)
    reader.sample_id = 2
    reader.dad.sample_id = 2
    for scan in reader.scans:
        scan.sample_id = 2
    # Sparse qualification uses the measured source family, with a fine grid.
    reader.sparse_profile_sampling_family = 'sciex-positive-tof-framed-profile-v1'
    reader.arrays = lambda _: (np.linspace(100., 100.04, 101), np.r_[np.zeros(20), np.ones(61)*500, np.zeros(20)])
    path = write_mzml(reader, tmp_path/f'{profile}.mzML', profile=profile)
    records = list(iter_records(path))
    root = ET.parse(path).getroot()
    ns = {'m': 'http://psi.hupo.org/ms/mzml'}
    optical_instrument = root.find('.//m:instrumentConfiguration[@id="IC-DAD"]', ns)
    assert optical_instrument.find('m:cvParam', ns).get('accession') == 'MS:1000031'
    assert 'model not decoded' in optical_instrument.find('m:userParam', ns).get('value')
    optical_refs = root.findall('.//m:scan[@instrumentConfigurationRef="IC-DAD"]', ns)
    assert len(optical_refs) == len(reader.dad)
    optical = [r for r in records if 'MS:1000806' in r['cv']]
    assert len(optical) == len(reader.dad)
    for row, record in enumerate(optical):
        np.testing.assert_array_equal(record['arrays']['wavelength'], reader.dad.wavelength)
        np.testing.assert_array_equal(record['arrays']['intensity'], reader.dad.absorbance_mau[row]/1000.)
        assert record['cv']['MS:1000515']['unitAccession'] == 'UO:0000269'
        assert 'MS:1000511' not in record['cv'] and record['polarity'] is None
        assert record['rt_seconds'] == reader.dad.rt_min[row]*60
    assert check_index(path)['offsets_verified']['spectrum'] == len(reader.scans)+len(reader.dad)
    specs = Path(__file__).resolve().parents[2]/'.cache/standards'
    if (specs/'mzML1.1.0_idx.xsd').exists():
        result = check(path, specs/'mzML1.1.0_idx.xsd', specs/'psi-ms.obo')
        assert result['xsd_valid'] and result['psi_ms_names_valid']
