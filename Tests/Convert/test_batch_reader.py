"""Guard the local acquisition set and vendor-backed startup calibration rule."""
from pathlib import Path
import mmap
import os
import shutil
import struct
import sys

import numpy as np
import pytest
import olefile

from kmzconvert_core.binary import FormatError
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import iter_records

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/validation'))
from probe_calibration import inspect

FIXTURES = Path(os.environ.get('KMZCONVERT_FIXTURES', Path(__file__).resolve().parents[2]/'TestData/Local'))
FILES = sorted(p for p in FIXTURES.glob('*.wiff') if p.stem != 'test')


@pytest.mark.parametrize('path', FILES, ids=[f'acquisition-{i+1:02}' for i in range(len(FILES))])
def test_additional_layouts_and_calibration_guard(path, monkeypatch):
    evidence = inspect(path)
    original = Path.read_bytes
    def no_full_scan_copy(p):
        if str(p).endswith('.wiff.scan'):
            raise AssertionError('Whole scan-file read is forbidden')
        return original(p)
    monkeypatch.setattr(Path, 'read_bytes', no_full_scan_copy)
    with WiffReader(path) as reader:
        mapping = reader.data
        assert isinstance(mapping, mmap.mmap)
        assert len(reader.scans) == evidence['spectra']
        assert len(reader.scans) == sum(bool(r[1]) for r in reader.index)
        assert [r['slot'] for r in reader.calibration_resolutions] == [r['slot'] for r in evidence['mismatches']]
        for conflict in evidence['mismatches']:
            scan = next(s for s in reader.scans if s.slot == conflict['slot'])
            assert [scan.cal_a, scan.cal_b] == conflict['ole']
        for i in sorted({0, len(reader.scans)//2, len(reader.scans)-1}):
            scan = reader.scans[i]
            mz, intensity = reader.arrays(scan)
            assert np.all(np.diff(mz) > 0) and np.isfinite(mz).all()
            assert intensity.sum() == scan.tic and intensity.max() == scan.base_peak_intensity
    assert mapping.closed
    with pytest.raises(FormatError, match='closed'):
        reader.arrays(reader.scans[0])


REFERENCES = sorted(FIXTURES.glob('*_from_*.mzML'))


@pytest.mark.parametrize('reference', REFERENCES, ids=[f'vendor-reference-{i+1:02}' for i in range(len(REFERENCES))])
def test_startup_and_neighboring_calibrations_match_both_vendor_paths(reference):
    path = reference.with_name(reference.name.split('_from_')[0]+'.wiff')
    with WiffReader(path) as reader:
        first_cycle = reader.scans[0].cycle
        checked = 0
        for scan, ref in zip(reader.scans, iter_records(reference), strict=True):
            if scan.cycle > first_cycle + 3:
                break
            assert scan.native_id == ref['id']
            mz, intensity = reader.arrays(scan, framing=False)
            measured = ref['arrays']['intensity'] > 0
            np.testing.assert_array_equal(intensity, ref['arrays']['intensity'][measured])
            np.testing.assert_allclose(mz, ref['arrays']['mz'][measured], rtol=0, atol=1e-10)
            checked += 1
        assert checked > len(reader.calibration_resolutions) > 0


def test_uncorroborated_startup_in_temporary_source_is_rejected(tmp_path, capsys):
    from kmzconvert.cli import main
    path = next((p for p in FILES if inspect(p)['mismatches']), None)
    if path is None:
        pytest.skip('No local startup-conflict acquisition available')
    with WiffReader(path) as reader:
        following = reader.calibration_resolutions[0]['corroborating_slot']
    copied = tmp_path/'renamed.wiff'
    for suffix in ('.wiff', '.wiff.scan'):
        shutil.copyfile(path.with_suffix(suffix), copied.with_suffix(suffix))
    # Break only the corroborating table record in a temporary CFB copy.
    # The actual scan preamble remains intact; both coefficients remain finite.
    with olefile.OleFileIO(copied, write_mode=True) as ole:
        stream = 'SampleSubtree/Sample1/TOFCalibrationData'
        data = bytearray(ole.openstream(stream).read())
        pos = 56+following*20
        a = struct.unpack_from('<d', data, pos)[0]
        struct.pack_into('<d', data, pos, np.nextafter(a, np.inf))
        ole.write_stream(stream, bytes(data))
    output = tmp_path/'rejected.mzML'
    assert main([str(copied), '-o', str(output)]) == 1
    assert 'Calibration disagreement' in capsys.readouterr().err
    assert not output.exists() and not list(tmp_path.glob('.kmzconvert-*.tmp'))
