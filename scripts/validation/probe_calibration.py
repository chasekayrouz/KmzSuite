"""Compare redundant calibration records without relaxing the reader's checks.

Writes local evidence only; never modifies acquisition files.
"""
import argparse
import json
import mmap
from pathlib import Path
import struct

import numpy as np
import olefile

from kmzconvert_core.binary import scan_metadata, calibrate, decode_tokens
from kmzconvert_core.reader import INDEX


def inspect(path, audit_conflicts=False):
    with olefile.OleFileIO(path) as ole:
        prefix = "SampleSubtree/Sample1/"
        idx = ole.openstream(prefix + "Idx").read()
        cal = ole.openstream(prefix + "TOFCalibrationData").read()
        tdc = struct.unpack_from('<d', ole.openstream(prefix+'TDCInfo').read(), 32)[0]
    result = {"input": str(path), "spectra": 0, "mismatches": []}
    with path.with_suffix('.wiff.scan').open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        cursor = 44
        for slot, record in enumerate(INDEX.iter_unpack(idx[32:])):
            offset, size = record[:2]
            if not size:
                continue
            a, b, _ = struct.unpack_from('<ddI', cal, 56 + slot*20)
            ma, mb, _ = scan_metadata(data, cursor, offset+44)
            result['spectra'] += 1
            if (a, b) != (ma, mb):
                x, y = decode_tokens(data[offset+44:offset+44+size])
                mz = calibrate(x, a, b, tdc)
                alternate = calibrate(x, ma, mb, tdc)
                result['mismatches'].append({
                    'slot': slot, 'ole': [a, b], 'scan': [ma, mb],
                    'absolute_difference': [abs(a-ma), abs(b-mb)],
                    'ulp_difference': [float(abs(a-ma)/abs(np.spacing(a))), float(abs(b-mb)/abs(np.spacing(b)))],
                    'max_mz_difference': float(np.max(abs(mz-alternate))),
                    'max_mz_ppm_difference': float(np.max(abs(mz-alternate)/mz*1e6)),
                    'intensity_tic_bpi_agree': bool(y.sum() == record[4] and y.max() == record[5]),
                })
            cursor = offset + 44 + size
        if audit_conflicts and result['mismatches']:
            count = points = 0
            for record in INDEX.iter_unpack(idx[32:]):
                offset, size = record[:2]
                if not size:
                    continue
                x, y = decode_tokens(data[offset+44:offset+44+size])
                assert np.all(np.diff(x) > 0) and np.all(y > 0)
                assert y.sum() == record[4] and y.max() == record[5], 'Raw TIC/BPI disagreement'
                count += 1
                points += len(x)
            result['raw_intensity_audit'] = {'spectra': count, 'signal_points': points,
                                             'all_tic_bpi_and_coordinate_order_checks_pass': True,
                                             'note': 'This does not select or validate either conflicting mass calibration.'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--audit-conflicts', action='store_true')
    args = parser.parse_args()
    results = [inspect(path, args.audit_conflicts) for path in sorted(args.fixtures.glob('*.wiff'))]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(Path(row['input']).stem, row['spectra'], 'mismatches:', len(row['mismatches']),
              'max ULP:', max((max(r['ulp_difference']) for r in row['mismatches']), default=0))
