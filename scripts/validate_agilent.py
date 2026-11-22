#!/usr/bin/env python3
"""Qualify Agilent MassHunter .d conversion against local reference mzML/mzXML files.

Each ``--case DIR.d=REFERENCE.mzML`` (or ``.mzXML``) is converted with the production
CLI into an ignored output directory and checked for: mzML 1.1 schema, PSI-MS names,
index and checksum; exact round trip of the written arrays; spectrum identity, order,
retention time, MS level, polarity, precursor m/z, charge and collision energy
against the reference. Profile conversions also check the flight-time calibration,
by decoding the bundle's centroid block (MSPeak.bin) with the reader's calibration
and comparing it with the reference peaks, and that every reference peak lies on
written profile signal. Centroid-only conversions compare the written centroids
with the reference peaks directly. ``--refuse DIR.d`` bundles must be refused
cleanly. Originals are never modified; only aggregate results with safe IDs are
reported.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from pyteomics import mzml, mzxml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmzconvert.cli import main as convert  # noqa: E402
from kmzconvert_core.agilent import CENTROID, AgilentReader, _child  # noqa: E402
from scripts.validation.check_mzml import check  # noqa: E402


def tree_digest(path):
    digest = hashlib.sha256()
    for item in sorted(p for p in Path(path).rglob('*') if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode() + b'\0')
        with item.open('rb') as stream:
            digest.update(hashlib.file_digest(stream, 'sha256').digest())
    return digest.hexdigest()


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = convert([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


def centroids(reader, index):
    """Reader-calibrated m/z and intensity of one scan's stored centroid block, or None."""
    record = reader.records[index]
    blocks = [record[name] for name in record.dtype.names if name.startswith('block')]
    found = [b for b in blocks if b['SpectrumFormatID'] == CENTROID]
    path = _child(reader.acq, 'MSPeak.bin')
    if not found or path is None:
        return None
    block = found[0]
    n, offset = int(block['PointCount']), int(block['SpectrumOffset'])
    if not hasattr(reader, '_peak'):
        reader._peak = path.read_bytes()
    data = reader._peak
    t = np.frombuffer(data, '<f8', n, offset)
    y = np.frombuffer(data, '<f4', n, offset + 8 * n)
    return reader.calibration(reader.scans[index]).mz(t), y.astype(np.float64)


def rm_bits(array):
    return np.asarray(array).dtype.itemsize * 8


def at_precision(ours, reference):
    """Our array rounded to the reference array's floating-point width, as float64."""
    width = np.dtype(f'<f{np.asarray(reference).dtype.itemsize}')
    return np.asarray(ours).astype(width).astype(np.float64)


def reference_spectra(path):
    """Reference spectra as the mzML-style dictionaries used below (mzXML is translated)."""
    if Path(path).suffix.lower() != '.mzxml':
        return [s for s in mzml.MzML(str(path)) if s['id'].startswith('scanId=')]
    out = []
    for s in mzxml.MzXML(str(path)):
        spectrum = {'id': f"scanId={s['num']}", 'ms level': s['msLevel'],
                    'scanList': {'scan': [{'scan start time': s['retentionTime']}]},
                    'm/z array': s['m/z array'], 'intensity array': s['intensity array']}
        polarity = {'+': 'positive scan', '-': 'negative scan'}.get(s.get('polarity'))
        if polarity is not None:  # an unknown polarity leaves neither, so it cannot match
            spectrum[polarity] = ''
        if s['msLevel'] > 1:
            ion = {'selected ion m/z': s['precursorMz'][0]['precursorMz']}
            if 'precursorCharge' in s['precursorMz'][0]:
                ion['charge state'] = s['precursorMz'][0]['precursorCharge']
            activation = {'collision energy': s['collisionEnergy']} if 'collisionEnergy' in s else {}
            spectrum['precursorList'] = {'precursor': [{'selectedIonList': {'selectedIon': [ion]},
                                                        'activation': activation}]}
        out.append(spectrum)
    return out


def qualify(ident, bundle, reference, output, cv):
    before = tree_digest(bundle)
    started = time.monotonic()
    destination = output / f'{ident}.mzML'
    code, _, err = run_cli([bundle, '-o', destination])
    assert code == 0, err
    seconds = time.monotonic() - started
    validation = check(destination, ROOT / '.cache/standards/mzML1.1.0_idx.xsd', cv)
    reader = AgilentReader(bundle)
    written = list(mzml.MzML(str(destination)))
    refs = reference_spectra(reference)
    # mzML references carry full-precision minutes; mzXML prints seconds to the millisecond.
    rt_tolerance = 0.0005 / 60 if Path(reference).suffix.lower() == '.mzxml' else 1e-9
    assert [s['id'] for s in written] == [s.native_id for s in reader.scans], 'written IDs'
    # A reference may cover a retention-time window: its IDs must be an ordered subset of ours.
    position = {s['id']: i for i, s in enumerate(written)}
    matched = [position.get(s['id']) for s in refs]
    assert None not in matched and matched == sorted(matched), 'reference spectrum identity/order'
    roundtrip = 0
    for scan, ours in zip(reader.scans, written, strict=True):
        mz, y = reader.arrays(scan)
        roundtrip += np.array_equal(mz, ours['m/z array']) and np.array_equal(y, ours['intensity array'])
    rt = level = polarity = precursor = charge = energy = 0
    ppm, intensity_equal, on_signal, peaks, apex_ppm, precisions, empty_matched, empty_with_tic = [], 0, 0, 0, [], set(), 0, 0
    for i, ref in zip(matched, refs, strict=True):
        scan, ours = reader.scans[i], written[i]
        mz, y = reader.arrays(scan)
        ours_scan, ref_scan = ours['scanList']['scan'][0], ref['scanList']['scan'][0]
        rt += abs(ours_scan['scan start time'] - ref_scan['scan start time']) <= rt_tolerance
        level += ours['ms level'] == ref['ms level']
        polarity += ('negative scan' in ours) == ('negative scan' in ref) and ('positive scan' in ours) == ('positive scan' in ref)
        if ref['ms level'] > 1:
            ion = lambda s: s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
            act = lambda s: s['precursorList']['precursor'][0]['activation']
            # References print 15 significant digits.
            precursor += abs(ion(ours)['selected ion m/z'] - ion(ref)['selected ion m/z']) <= 1e-12 * ion(ref)['selected ion m/z']
            charge += ion(ours).get('charge state') == ion(ref).get('charge state')
            energy += act(ours).get('collision energy') == act(ref).get('collision energy')
        else:
            precursor += 'precursorList' not in ours
            charge += 1
            energy += 1
        if reader.centroided:
            # The written spectra are the stored centroids: compare them with the reference peaks.
            # Compare at each reference array's stored precision (any byte order): some vendor
            # exports write 32-bit m/z.
            cm, cy = (at_precision(ours[k], ref[k]) for k in ('m/z array', 'intensity array'))
            rm, ry = (np.asarray(ref[k], dtype=np.float64) for k in ('m/z array', 'intensity array'))
            precisions.add(f"{rm_bits(ref['m/z array'])}-bit m/z, {rm_bits(ref['intensity array'])}-bit intensity")
            if len(cm) != len(rm):
                ppm.append(float('inf'))
            elif len(rm) == 0:
                empty_matched += 1  # no masses to compare
                empty_with_tic += ours['total ion current'] > 0
            else:
                ppm.append(float(np.max(np.abs(cm - rm) / rm * 1e6)))
                intensity_equal += np.array_equal(cy, ry)
            continue
        peak = centroids(reader, i)
        if peak is not None:
            cm, cy = at_precision(peak[0], ref['m/z array']), at_precision(peak[1], ref['intensity array'])
            rm, ry = (np.asarray(ref[k], dtype=np.float64) for k in ('m/z array', 'intensity array'))
            precisions.add(f"{rm_bits(ref['m/z array'])}-bit m/z, {rm_bits(ref['intensity array'])}-bit intensity")
            if len(cm) == len(rm) and len(rm):
                ppm.append(float(np.max(np.abs(cm - rm) / rm * 1e6)))
                intensity_equal += np.array_equal(cy, ry)
        # Each reference peak must lie on stored signal: a nonzero point within two native bins.
        t, stored = reader.decoded(scan)
        axis = reader.calibration(scan).mz(t)
        rm = ref['m/z array']
        at = np.searchsorted(axis, rm)
        nonzero = np.concatenate(([0], np.cumsum(stored > 0)))
        lo, hi = np.clip(at - 2, 0, len(axis)), np.clip(at + 2, 0, len(axis))
        on_signal += int(((nonzero[hi] - nonzero[lo]) > 0).sum())
        peaks += len(rm)
        # Calibration check without stored centroids: 3-point apex centroid of the
        # strongest reference peaks (vendor peak picking is not reproduced exactly).
        for m in rm[np.argsort(ref['intensity array'])[-20:]]:
            i = int(np.searchsorted(axis, m))
            lo, hi = max(i - 6, 1), min(i + 6, len(axis) - 1)
            if hi <= lo or stored[lo:hi].max() <= 0:
                continue
            j = lo + int(np.argmax(stored[lo:hi]))
            w = stored[j - 1:j + 2].astype(float)
            apex_ppm.append(float(((axis[j - 1:j + 2] * w).sum() / w.sum() - m) / m * 1e6))
    n, m = len(reader.scans), len(refs)
    after = tree_digest(bundle)
    encoding = reader.profile_encoding() or ('centroid-only' if reader.centroided else None)
    reader.close()
    return {
        'case': ident, 'instrument': reader.instrument.get('Model'), 'software': reader.acquisition_software,
        'centroid_only': reader.centroided,
        'profile_encoding': encoding, 'calibration': reader.calibration_source,
        'spectra': n, 'ms_levels': {int(k): int(v) for k, v in zip(*np.unique([s.ms_level for s in reader.scans], return_counts=True))},
        'seconds': round(seconds, 2), 'output_bytes': destination.stat().st_size,
        'xsd_valid': validation['xsd_valid'], 'psi_ms_names_valid': validation['psi_ms_names_valid'],
        'index': validation['index'],
        'arrays_roundtrip_exact': roundtrip == n, 'reference_spectra_matched_in_order': f'{m}/{n}', 'matched_spectra': m,
        'rt_within_reference_precision': rt == m, 'rt_tolerance_min': rt_tolerance, 'ms_level_exact': level == m, 'polarity_exact': polarity == m,
        'precursor_mz_within_1e-12': precursor == m, 'charge_exact': charge == m, 'collision_energy_exact': energy == m,
        'centroid_calibration_scans': len(ppm),
        'centroid_calibration_max_ppm': max(ppm) if ppm else None,
        'reference_precision': sorted(precisions),
        'centroid_empty_spectra_matched': empty_matched,
        'centroid_empty_spectra_with_stored_tic': int(empty_with_tic),
        'centroid_intensity_exact_scans': intensity_equal,
        'reference_peaks_on_stored_signal': on_signal == peaks,
        'apex_vs_reference_peaks': len(apex_ppm),
        'apex_vs_reference_median_ppm': float(np.median(apex_ppm)) if apex_ppm else None,
        'apex_vs_reference_p95_abs_ppm': float(np.percentile(np.abs(apex_ppm), 95)) if apex_ppm else None,
        'reference_peaks_on_profile_signal': f'{on_signal}/{peaks}',
        'source_unchanged': before == after,
    }


def refuse(ident, bundle):
    code, _, err = run_cli([bundle, '--inspect'])
    message = err.strip().splitlines()[-1] if err.strip() else ''
    return {'case': ident, 'refused': code == 1 and 'Traceback' not in err, 'message': message.removeprefix('kmzconvert: ')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', default=[], metavar='DIR.d=REFERENCE.mzML')
    parser.add_argument('--refuse', action='append', default=[], type=Path, metavar='DIR.d')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cv', type=Path, default=ROOT / '.cache/standards/psi-ms.obo')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for i, case in enumerate(args.case, 1):
        bundle, reference = case.split('=', 1)
        results.append(qualify(f'agilent-{i:02}', Path(bundle), Path(reference), args.output, args.cv))
    refusals = [refuse(f'refuse-{i:02}', path) for i, path in enumerate(args.refuse, 1)]
    def passed(r):
        exact = ('xsd_valid', 'psi_ms_names_valid', 'arrays_roundtrip_exact', 'rt_within_reference_precision', 'ms_level_exact',
                 'polarity_exact', 'precursor_mz_within_1e-12', 'charge_exact', 'collision_energy_exact',
                 'reference_peaks_on_stored_signal', 'source_unchanged')
        centroids_ok = r['centroid_calibration_scans'] == 0 or (
            r['centroid_calibration_max_ppm'] <= 1e-3 and r['centroid_intensity_exact_scans'] == r['centroid_calibration_scans'])
        # Centroid-only output must have been compared on at least one non-empty spectrum.
        centroids_ok = centroids_ok and (not r['centroid_only'] or r['centroid_calibration_scans'] > 0)
        # Without stored centroids, the apex statistic is the calibration check (it is biased
        # by vendor peak picking, so stored centroids decide when present).
        apex_ok = r['centroid_calibration_scans'] > 0 or (
            r['apex_vs_reference_peaks'] > 0 and abs(r['apex_vs_reference_median_ppm']) <= 1.0)
        return (all(r[k] for k in exact) and r['index'].get('checksum_valid') and centroids_ok and apex_ok
                and r['matched_spectra'] > 0)
    summary = {'conversions': results, 'refusals': refusals,
               'ok': bool(results) and all(passed(r) for r in results) and all(r['refused'] for r in refusals)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
