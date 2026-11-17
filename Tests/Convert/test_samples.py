"""Independent synthetic multi-run containers; no private WIFF measurements."""
from io import BytesIO
import json
import struct

import numpy as np
import pytest

from kmzconvert.cli import main
from kmzconvert_core.binary import FormatError
from kmzconvert_core.reader import INDEX, WiffReader
from kmzconvert_core.samples import STREAM_HEADER, list_samples, sample_index, scan_segments, sample_output_path
from kmzconvert_core.reference import iter_records
from scripts.validation.check_mzml import check_index
from Tests.dad_fixtures import sidecars


def framed_segments(payloads):
    data = bytearray(struct.pack('<IQII', 2200, 0, 0, 0))
    offsets = []
    for ident, payload in payloads:
        offsets.append((ident, len(data), 0))
        data += struct.pack('<IIIQI', 0x11111111, 2200, ident, len(payload), 0) + payload
    struct.pack_into('<Q', data, 4, len(data))
    return data, offsets


@pytest.fixture
def multi(tmp_path, monkeypatch):
    streams = {'FileRec_Str': 'SCIEX OS 2.2.0.0'.encode('utf-16le')}
    period = bytearray(78); struct.pack_into('<I', period, 74, 2)
    method = {'Period0/PeriodStream': period}
    for exp, kind in enumerate((8, 9)):
        prefix = f'Period0/Experiment{exp}/'
        header = bytearray(98); struct.pack_into('<I', header, 94, kind)
        method[prefix+'ExperimentHeader'] = header
        method[prefix+'ExperimentHeaderEx'] = bytes(66) + struct.pack('<ddd', 50., 1600., 200.)+bytes(4)
        name = 'IS'.encode('utf-16le')
        method[prefix+'IonSourceParamsTable/Parameter0/ParameterData'] = bytes(34)+struct.pack('<H', len(name))+name+struct.pack('<ff', 5500., 5500.)
        name = 'CE'.encode('utf-16le')
        method[prefix+'MassRangeEx/MassRangeEx'] = bytes(32)+struct.pack('<II', 1, 1)+bytes(22)+struct.pack('<H', len(name))+name+struct.pack('<ffI', 30., 30., 0)
    for ident in (2, 7):
        streams.update({f'MethodSubtree/Method{ident}/DeviceMethod0/'+k: bytes(v) for k, v in method.items()})
    payloads = []
    for ident in (2, 7):
        a, b, seed = .000025, 0., 4_000_000
        def vi(value):
            data = bytearray()
            while value >= 128:
                data.append((value & 127) | 128); value >>= 7
            return bytes(data) + bytes([value])
        cal = b'\x09'+struct.pack('<d', a)+b'\x11'+struct.pack('<d', b)
        checkpoint = b'\x08'+vi(seed)+b'\x10\x08'
        metadata = b'\x0a'+vi(len(cal))+cal+b'\x12'+vi(len(checkpoint))+checkpoint
        metadata = vi(len(metadata))+metadata
        intensity = 110+ident
        block = b'\xff'*4+struct.pack('<I', seed)+b'\x00'+bytes([intensity])*10+b'\xff'
        payloads.append((ident, metadata+block))
        prefix = f'SampleSubtree/Sample{ident}/'
        streams[prefix+'Idx'] = STREAM_HEADER + INDEX.pack(len(metadata), len(block), ident*1000., 6, intensity*10., float(intensity), 0., 0., 0) + INDEX.pack(len(metadata)+len(block), 0, ident*1000., 6, 0., 0., 0., 0., 0)
        streams[prefix+'TOFCalibrationData'] = bytes(52)+struct.pack('<I', 2)+struct.pack('<ddI', a, b, 0)*2
        streams[prefix+'TDCInfo'] = bytes(32)+struct.pack('<d', .1)
        streams[prefix+'DDERealTimeData'] = bytes(64)
        streams[prefix+'Log'] = 'Mass Spectrometer: X500 QTOF'.encode('utf-16le')
    scan, offsets = framed_segments(payloads)
    streams['SampleSubtree/SampleIdxTable'] = STREAM_HEADER+b''.join(struct.pack('<IQI', *r) for r in offsets)
    path = tmp_path/'multi.wiff'; path.write_bytes(b'synthetic OLE stand-in')
    path.with_suffix('.wiff.scan').write_bytes(scan)
    class Ole:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def listdir(self): return [k.split('/') for k in streams]
        def openstream(self, key): return BytesIO(streams['/'.join(key) if isinstance(key, list) else key])
    monkeypatch.setattr('olefile.OleFileIO', Ole)
    return path, streams


def test_discovery_selection_boundaries_and_source_ids(multi):
    path, _ = multi
    assert [s.id for s in list_samples(path)] == [2, 7]
    with pytest.raises(FormatError, match='contains 2 samples'):
        WiffReader(path)
    with pytest.raises(FormatError, match='Unknown WIFF sample'):
        WiffReader(path, sample_id=1)
    for ident in (2, 7):
        with WiffReader(path, sample_id=ident) as r:
            scan, = r.scans
            assert scan.native_id.startswith(f'sample={ident} ')
            assert scan.rt_seconds == ident
            assert r.arrays(scan)[1].sum() == (110+ident)*10
            assert r.tic_arrays()[0][0] == ident/60


@pytest.mark.parametrize('damage', ['offset', 'duplicate', 'cross-boundary', 'method', 'calibration', 'tic'])
def test_rejects_ambiguous_or_damaged_sample_data(multi, damage):
    path, streams = multi
    if damage in ('offset', 'duplicate'):
        key = 'SampleSubtree/SampleIdxTable'
        data = bytearray(streams[key])
        if damage == 'offset': struct.pack_into('<Q', data, 52, 20)
        else: struct.pack_into('<I', data, 48, 2)
        streams[key] = data
    elif damage == 'method':
        streams['MethodSubtree/Method7/DeviceMethod0/Period0/Experiment0/ExperimentHeaderEx'] = bytes(66)+struct.pack('<ddd', 100., 800., 200.)+bytes(4)
    elif damage == 'calibration':
        key = 'SampleSubtree/Sample2/TOFCalibrationData'
        data = bytearray(streams[key]); struct.pack_into('<d', data, 56, .000026); streams[key] = data
    else:
        key = 'SampleSubtree/Sample2/Idx'
        data = bytearray(streams[key])
        if damage == 'cross-boundary': struct.pack_into('<I', data, 36, path.with_suffix('.wiff.scan').stat().st_size)
        else: struct.pack_into('<d', data, 50, 999.)
        streams[key] = data
    with pytest.raises(FormatError):
        with WiffReader(path, sample_id=2) as r:
            r.arrays(r.scans[0])


@pytest.mark.parametrize('profile', ['full', 'sparse'])
def test_cli_requires_selection_and_writes_only_selected_run(multi, tmp_path, capsys, profile):
    path, _ = multi
    assert main([str(path), '--list-samples']) == 0
    assert json.loads(capsys.readouterr().out) == {'samples': [{'id': 2}, {'id': 7}]}
    base = tmp_path/'custom.mzML'
    assert main([str(path), '-o', str(base)]) == 1
    assert not base.exists()
    assert main([str(path), '--sample', '7', '--profile', profile, '-o', str(base)]) == 0
    output = tmp_path/('custom_7.sp.mzML' if profile == 'sparse' else 'custom_7.mzML')
    record, = iter_records(output)
    assert record['id'].startswith('sample=7 ')
    assert float(record['cv']['MS:1000285']['value']) == 1170
    # Sparse Profile omits this flat run: it is its own event unit, not a peak.
    assert float(record['arrays']['intensity'].sum()) == (1170 if profile == 'full' else 0)
    assert check_index(output)['checksum_valid']
    original = output.read_bytes()
    assert main([str(path), '--sample', '7', '--profile', profile, '-o', str(base)]) == 1
    assert output.read_bytes() == original
    assert not list(tmp_path.glob('*_2*.mzML')) and not list(tmp_path.glob('.kmzconvert-*.tmp'))


def test_segment_framing_rejects_truncated_duplicate_and_unaccounted_bytes():
    good, _ = framed_segments([(2, b'abc'), (7, b'def')])
    assert set(scan_segments(good)) == {2, 7}
    for corrupt in (good[:-1], good+b'x', framed_segments([(2, b'a'), (2, b'b')])[0]):
        with pytest.raises(FormatError): scan_segments(corrupt)


def test_dad_selection_preserves_sample_specific_arrays_and_native_id(multi, tmp_path):
    path, streams = multi
    scans, indices, expected = [], [], {}
    for ident, count in ((2, 3), (7, 5)):
        scan, index, optical, data = sidecars(count=count)
        streams.update({k.replace('/Sample1/', f'/Sample{ident}/'): v for k, v in optical.items()})
        scans.append((ident, scan[44:])); indices.append((ident, index[52:])); expected[ident] = data
    path.with_suffix('.dad.scan').write_bytes(framed_segments(scans)[0])
    index = bytearray(struct.pack('<8sIQII', b'@dadsidx', 1002, 0, 1, 0))
    for ident, payload in indices:
        index += struct.pack('<IIQII', 1002, ident, len(index)+24+len(payload), 0, 0)+payload
    struct.pack_into('<Q', index, 12, len(index))
    path.with_suffix('.dad.sidx').write_bytes(index)
    for ident in (2, 7):
        with WiffReader(path, sample_id=ident) as r:
            assert r.dad_status == 'included'
            np.testing.assert_array_equal(r.dad.absorbance_mau, expected[ident].absorbance_mau)
        assert main([str(path), '--sample', str(ident), '-o', str(tmp_path/'optical.mzML')]) == 0
        records = list(iter_records(tmp_path/f'optical_{ident}.mzML'))
        assert len(records) == 1+len(expected[ident])
        assert all(r['id'].startswith(f'sample={ident} ') for r in records)
    # Swapping the DAD segment IDs must not silently attach another sample.
    raw = bytearray(path.with_suffix('.dad.scan').read_bytes())
    segments = scan_segments(raw)
    struct.pack_into('<I', raw, segments[2][0]+8, 7)
    struct.pack_into('<I', raw, segments[7][0]+8, 2)
    path.with_suffix('.dad.scan').write_bytes(raw)
    with WiffReader(path, sample_id=2) as r:
        assert r.dad is None and r.dad_status == 'unreadable'


@pytest.mark.parametrize(('name', 'expected'), [('run.mzML', 'run_7.mzML'),
    ('run.sp.mzML', 'run_7.sp.mzML'), ('run.mc.sp.mzML', 'run.mc_7.sp.mzML'),
    ('run_7.mzML', 'run_7_7.mzML')])
def test_sample_filename_uses_complete_extension(name, expected):
    assert sample_output_path(name, 7).name == expected
