#!/usr/bin/env python3
"""Production Sparse Profile v3 integration parity against the independent loop oracle; no tuning."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from kmzconvert_core.native import NativeWiffReader
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import NS, read_arrays
from kmzconvert_core.writer import write_mzml
from kmzcore.sparse_profile import SAMPLING_PARAM, read_provenance, rendering_connections
from kmzviewer.mzml_index import index_ms1
from kmzviewer.spectra import SpectrumReader
from scripts.benchmarks.ms1_algorithms import event_rescue_plan, gap_connections
from scripts.validation.check_mzml import check


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def elements(path, tag):
    for _, e in ET.iterparse(path):
        if e.tag == NS + tag:
            yield e
            e.clear()
        elif e.tag in (NS+'spectrum', NS+'chromatogram'):
            e.clear()


def convert(path, output, ident):
    """Production full and sparse outputs (the converter's native reader) and their timing,
    then the interpreted reference full output, which the production full must equal."""
    row, outputs = {}, {}
    with NativeWiffReader(path) as reader:
        for profile in ('full', 'sparse'):
            start = time.perf_counter()
            outputs[profile] = write_mzml(reader, output/(ident+('.mzML' if profile=='full' else '.sp.mzML')), profile=profile)
            row[profile+'_seconds'] = time.perf_counter()-start
            row[profile+'_bytes'] = outputs[profile].stat().st_size
    with WiffReader(path) as reader:
        reference = write_mzml(reader, output/(ident+'-interpreted.mzML'))
    row['full_profile_interpreted_byte_equal'] = digest(outputs['full']) == digest(reference)
    assert row['full_profile_interpreted_byte_equal'], 'Production full output differs from the interpreted reference'
    return row, reference, outputs['full'], outputs['sparse']


def compare(full, sparse):
    """Check sparse against ``full``, which must come from the interpreted reference path."""
    provenance = read_provenance(sparse)
    table = index_ms1(sparse)
    ms1 = points = retained = spectra = 0
    sampled = []
    def sources():
        for ordinal, a in enumerate(elements(full, 'spectrum')):
            if a.find(NS+'cvParam[@accession="MS:1000511"]').get('value') == '1':
                experiment = a.find('.//'+NS+'cvParam[@accession="MS:1000616"]').get('value')
                yield ordinal, experiment, *(read_arrays(a)[k] for k in ('mz', 'intensity'))
    plan = event_rescue_plan(sources())
    for ordinal, (a, b) in enumerate(zip(elements(full, 'spectrum'), elements(sparse, 'spectrum'), strict=True)):
        spectra += 1
        aa, bb = read_arrays(a), read_arrays(b)
        a.tail = b.tail = None  # iterparse may expose trailing layout at different read boundaries.
        level = a.find(NS+'cvParam[@accession="MS:1000511"]').get('value')
        if level == '1':
            indices = plan[ordinal]
            for kind in ('mz', 'intensity'):
                assert np.array_equal(bb[kind], aa[kind][indices]), (ordinal, kind)
            expected = gap_connections(bb['mz'], factor=4, radius=8)
            check = provenance.sampling(b)
            assert check == table.sparse_profile[ms1] and check is not None
            actual = rendering_connections(check, bb['mz'], bb['intensity'])
            assert actual is not None and np.array_equal(actual, expected), ordinal
            ms1 += 1; points += len(aa['mz']); retained += len(indices)
            if len(sampled) < 3 or ordinal % 503 == 0:
                sampled.append((ordinal, bb['mz'].copy(), bb['intensity'].copy()))
            # Remove only the representation's declared changes to compare metadata.
            b = deepcopy(b)
            b.attrib.pop('dataProcessingRef')
            b.set('defaultArrayLength', a.get('defaultArrayLength'))
            b.remove(b.find(NS+'userParam[@name="'+SAMPLING_PARAM+'"]'))
            a = deepcopy(a)
            a.remove(a.find(NS+'binaryDataArrayList'))
            b.remove(b.find(NS+'binaryDataArrayList'))
        assert ET.tostring(a) == ET.tostring(b), ('metadata or MS2', ordinal)
    for a, b in zip(elements(full, 'chromatogram'), elements(sparse, 'chromatogram'), strict=True):
        a.tail = b.tail = None
        assert ET.tostring(a) == ET.tostring(b), 'chromatogram'
    # Ordinary indexed pyteomics reader, also used by the production viewer.
    with SpectrumReader(sparse) as reader:
        for ordinal, m, y in sampled:
            actual_m, actual_y = reader.read(ordinal)
            assert np.array_equal(actual_m, m) and np.array_equal(actual_y, y)
    return dict(spectra=spectra, ms1_spectra=ms1, source_ms1_points=points,
                sparse_ms1_points=retained, indexed_readbacks=len(sampled),
                arrays_exact=True, metadata_ms2_chromatograms_exact=True,
                rendering_masks_exact=True, all_ms1_gates_pass=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=ROOT/'FromChase')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualified-reference', type=Path)
    parser.add_argument('--cv', type=Path, default=ROOT/'.cache/standards/psi-ms.obo')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'.local'):
        parser.error('Integration output must be inside ignored .local')
    paths = sorted(args.fixtures.glob('*.wiff'))
    if len(paths) != 5 or any(not p.with_suffix('.wiff.scan').is_file() for p in paths):
        parser.error('Expected five complete native fixture pairs')
    output.mkdir(parents=True, exist_ok=False)
    sources = [p for path in paths for p in (path, path.with_suffix('.wiff.scan'))]
    before = {p.name: digest(p) for p in sources}
    (output/'source-hashes-before.json').write_text(json.dumps(before, indent=2)+'\n')
    rows = []; validations = []
    try:
        for number, path in enumerate(paths, 1):
            ident = f'heldout-{number:02d}'
            row = dict(acquisition=ident)
            timing, reference, full, sparse = convert(path, output, ident)
            row.update(timing)
            for profile, destination in (('full', full), ('sparse', sparse)):
                validation = check(destination, ROOT/'.cache/standards/mzML1.1.0_idx.xsd', args.cv)
                assert validation['psi_ms_names_valid']
                validations.append(dict(acquisition=ident, profile=profile, result=validation))
            if args.qualified_reference:
                original = args.qualified_reference/(ident+'-reference.mzML')
                row['full_profile_baseline_byte_equal'] = digest(full) == digest(original)
                assert row['full_profile_baseline_byte_equal']
            row.update(compare(reference, sparse))
            reference.unlink()
            row['size_reduction_percent'] = 100*(1-row['sparse_bytes']/row['full_bytes'])
            rows.append(row)
            (output/'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(f"{ident}: {row['ms1_spectra']} MS1 exact; {row['size_reduction_percent']:.2f}% smaller; full {row['full_seconds']:.2f}s / sparse {row['sparse_seconds']:.2f}s", flush=True)
        with (output/'performance.csv').open('w') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (output/'validation.json').write_text(json.dumps(validations, indent=2)+'\n')
    finally:
        after = {p.name: digest(p) for p in sources}
        (output/'source-hashes-after.json').write_text(json.dumps(after, indent=2)+'\n')
        assert before == after, 'Private source files changed'


if __name__ == '__main__':
    main()
