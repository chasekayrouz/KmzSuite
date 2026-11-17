#!/usr/bin/env python3
"""Offline end-to-end conversion, native-array and independent viewer checks."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/validation'))
from check_mzml import check_index
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import iter_records, value
from kmzviewer.mzml_index import index_ms1
from kmzviewer.extract import extract_eics, parse_targets
from kmz2viewer.index import index_ms2


def reader_signature(reader):
    """Digest all preserved scan metadata and framed/unframed arrays, not paths."""
    digest = hashlib.sha256()
    for scan in reader.scans:
        digest.update(json.dumps(asdict(scan), sort_keys=True, separators=(',', ':')).encode())
        for framing in (False, True):
            for array in reader.arrays(scan, framing=framing):
                digest.update(np.asarray(array, dtype='<f8').tobytes())
    for array in reader.tic_arrays():
        digest.update(np.asarray(array, dtype='<f8').tobytes())
    return {'spectra': len(reader.scans), 'levels': dict(Counter(reader.experiments[s.experiment-1].ms_level for s in reader.scans)),
            'startup_resolutions': len(reader.calibration_resolutions), 'scientific_sha256': digest.hexdigest()}


def verify_output(entry, output, profile='full', *, include_signature=True, sample_id=None):
    from scripts.benchmarks.ms1_algorithms import event_rescue_plan, gap_connections
    from kmzcore.sparse_profile import read_provenance, rendering_connections
    provenance = read_provenance(output)
    with WiffReader(entry, sample_id=sample_id) as reader:
        levels = Counter()
        records = iter(iter_records(output))
        # Sparse Profile v3 selects across each MS1 experiment's spectra.
        plan = event_rescue_plan((i, scan.experiment, *reader.arrays(scan)) for i, scan in enumerate(reader.scans)
                                 if reader.experiments[scan.experiment-1].ms_level == 1) if profile == 'sparse' else {}
        for i, scan in enumerate(reader.scans):
            record = next(records)
            mz, intensity = reader.arrays(scan)
            if i in plan:
                mz, intensity = mz[plan[i]], intensity[plan[i]]
            np.testing.assert_array_equal(record['arrays']['mz'], mz)
            np.testing.assert_array_equal(record['arrays']['intensity'], intensity)
            assert record['id'] == scan.native_id
            assert record['ms_level'] == reader.experiments[scan.experiment-1].ms_level
            assert record['representation'] == 'profile'
            np.testing.assert_allclose(record['rt_seconds'], scan.rt_seconds, rtol=0, atol=1e-9)
            assert float(value(record['cv'], 'MS:1000285')) == scan.tic
            levels[record['ms_level']] += 1
        dad_count = len(reader.dad) if reader.dad is not None else 0
        for row, record in zip(range(dad_count), records, strict=True):
            assert record['id'] == f'sample={reader.sample_id} period=1 cycle={row+1} experiment=0'
            assert record['ms_level'] == 0 and 'MS:1000806' in record['cv']
            assert record['cv']['MS:1000515']['unitAccession'] == 'UO:0000269'
            np.testing.assert_array_equal(record['arrays']['wavelength'], reader.dad.wavelength)
            np.testing.assert_array_equal(record['arrays']['intensity'], reader.dad.absorbance_mau[row] / 1000.)
            assert record['rt_seconds'] == reader.dad.rt_min[row] * 60
        tic, = iter_records(output, 'chromatogram')
        for kind, raw in zip(('time', 'intensity', 'ms level'), reader.tic_arrays(), strict=True):
            np.testing.assert_array_equal(tic['arrays'][kind], raw)
        ms1 = index_ms1(output)
        ms2 = index_ms2(output)
        assert len(ms1) == levels[1] and len(ms2) == levels[2]
        if dad_count:
            from kmzviewer.dad import extract_wavelengths, WavelengthBand, WavelengthExtraction
            assert len(ms1.dad) == dad_count
            settings = [WavelengthExtraction(), WavelengthExtraction(WavelengthBand(254, 4)),
                        WavelengthExtraction(WavelengthBand(280, 8), WavelengthBand(360, 20))]
            optical = extract_wavelengths(str(output), ms1.dad, settings, [10,11,12], '#123456')
            source = reader.dad
            direct = [source.absorbance_mau.sum(axis=1)]
            for setting in settings[1:]:
                a = setting.signal
                y = source.absorbance_mau[:, (source.wavelength >= a.lower) & (source.wavelength <= a.upper)].mean(axis=1)
                if setting.reference:
                    a = setting.reference
                    y -= source.absorbance_mau[:, (source.wavelength >= a.lower) & (source.wavelength <= a.upper)].mean(axis=1)
                direct.append(y)
            for trace, expected in zip(optical, direct):
                np.testing.assert_allclose(trace.y, expected, rtol=2e-14, atol=1e-9)
                np.testing.assert_array_equal(trace.rt, source.rt_min)
        if profile == 'sparse':
            assert provenance.methods and all(c is not None for c in ms1.sparse_profile)
            for row, record in enumerate(r for r in iter_records(output) if r['ms_level'] == 1):
                m, y = record['arrays']['mz'], record['arrays']['intensity']
                actual = rendering_connections(ms1.sparse_profile[row], m, y)
                assert actual is not None
                np.testing.assert_array_equal(actual, gap_connections(m, factor=4, radius=8))
        for scan in ms2:
            raw = reader.scans[scan.spectrum_index]
            mz, intensity = reader.arrays(raw)
            np.testing.assert_array_equal(scan.mz, mz)
            np.testing.assert_array_equal(scan.intensity, intensity)
        # The viewer EIC data path must agree with direct native inclusive sums.
        traces = extract_eics(str(output), ms1, parse_targets('250-300', 10), 'all', False, [1], ['#1f4e79'])
        direct = []
        for index in ms1.spectrum_index:
            mz, intensity = reader.arrays(reader.scans[int(index)])
            if profile == 'sparse':
                mz, intensity = mz[plan[int(index)]], intensity[plan[int(index)]]
            direct.append(float(intensity[(mz >= 250) & (mz <= 300)].sum()))
        np.testing.assert_array_equal(traces[0].y, direct)
        signature = reader_signature(reader) if include_signature else {'spectra': len(reader.scans)}
    indices = check_index(output)
    assert indices['checksum_valid'] and indices['offsets_verified']['spectrum'] == sum(levels.values()) + dad_count
    return signature | {'index': indices, 'ms1': len(ms1), 'ms2': len(ms2), 'dad': dad_count,
                        'arrays_tic_eic_exact': True, 'dad_arrays_exact': bool(dad_count), 'twc_ewc_verified': bool(dad_count)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'TestData/Local')
    parser.add_argument('--cli', type=Path, default=ROOT / 'dist/kmzconvert')
    parser.add_argument('--reference-report', type=Path, help='Optional pre-migration scientific signatures')
    parser.add_argument('--report', type=Path, default=ROOT / '.local/validation/integration.json')
    args = parser.parse_args()
    entries = sorted(args.fixtures.glob('acquisition-*.wiff'))
    if not entries:
        parser.error('No local acquisition pairs available; integration was not run.')
    expected = json.loads(args.reference_report.read_text()) if args.reference_report else None
    scratch = ROOT / '.local/validation'
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='integration-', dir=scratch) as directory:
        for entry in entries:
            output = Path(directory) / (entry.stem + '.mzML')
            completed = subprocess.run([str(args.cli.resolve()), str(entry.resolve()), '-o', str(output)], capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr
            result = verify_output(entry, output)
            if expected is not None:
                reference = expected[entry.name]
                assert result['scientific_sha256'] == reference['scientific_sha256'], entry.name
                result['source_baseline_exact'] = True
            results.append({'acquisition': entry.stem} | result)
            output.unlink()
            print(entry.stem, result['spectra'], 'spectra passed', flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'ok': True, 'acquisitions': results, 'total_spectra': sum(r['spectra'] for r in results)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
