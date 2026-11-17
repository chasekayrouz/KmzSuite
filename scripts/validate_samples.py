#!/usr/bin/env python3
"""Local, per-sample Full/Sparse qualification with independent mzML/viewer readback."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time

from psims.controlled_vocabulary.controlled_vocabulary import obo_cache
from kmzconvert_core.native import NativeWiffReader
from kmzconvert_core.samples import list_samples, sample_output_path
from kmzconvert_core.writer import write_mzml


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New ignored local result directory')
    parser.add_argument('--cli', type=Path, help='Optional built/installed CLI to qualify instead of direct writer')
    parser.add_argument('--profile', choices=('full', 'sparse', 'both'), default='both')
    args = parser.parse_args()
    entries = sorted(args.fixtures.glob('*.wiff'))
    if not entries:
        parser.error('No WIFF files found')
    args.output.mkdir(parents=True, exist_ok=False)
    obo_cache.use_remote = False
    verify = runpy.run_path(str(Path(__file__).with_name('integration-parity.py')))['verify_output']
    check = runpy.run_path(str(Path(__file__).parent/'validation/check_mzml.py'))['check']
    standards = Path(__file__).resolve().parents[1]/'.cache/standards'
    originals = {p: digest(p) for p in args.fixtures.iterdir() if p.is_file()}
    results = []
    for ordinal, entry in enumerate(entries, 1):
        samples = list_samples(entry)
        for sample in samples:
            for profile in (('full', 'sparse') if args.profile == 'both' else (args.profile,)):
                started = time.monotonic()
                ident = f'input-{ordinal:02}'
                base = args.output/(ident+('.sp.mzML' if profile == 'sparse' else '.mzML'))
                output = sample_output_path(base, sample.id)
                if args.cli:
                    completed = subprocess.run([str(args.cli.resolve()), str(entry.resolve()), '--sample', str(sample.id),
                        '--profile', profile, '-o', str(base.resolve())], capture_output=True, text=True)
                    (args.output/f'{ident}-{sample.id}-{profile}.log').write_text(completed.stdout+completed.stderr)
                    assert completed.returncode == 0, f'{ident} sample {sample.id} {profile}: conversion failed; see local log'
                else:
                    # The converter's production reader; verify_output decodes through the interpreted reference.
                    with NativeWiffReader(entry, sample_id=sample.id) as reader:
                        write_mzml(reader, output, profile=profile)
                converted = time.monotonic()
                result = verify(entry, output, profile, include_signature=False, sample_id=sample.id)
                schema = check(output, standards/'mzML1.1.0_idx.xsd', standards/'psi-ms.obo')
                assert schema['xsd_valid'] and schema['psi_ms_names_valid']
                results.append(dict(id=ident, sample_id=sample.id, profile=profile,
                    output=output.name, bytes=output.stat().st_size,
                    xsd_valid=True, psi_ms_names_valid=True,
                    conversion_seconds=round(converted-started, 3),
                    verification_seconds=round(time.monotonic()-converted, 3), **result))
                (args.output/'summary.json').write_text(json.dumps({'ok': False, 'pending': True, 'runs': results}, indent=2)+'\n')
                print(ident, sample.id, profile, 'PASS', flush=True)
    assert all(digest(p) == before for p, before in originals.items()), 'An original input changed'
    (args.output/'summary.json').write_text(json.dumps({'ok': True, 'originals_unchanged': True,
        'source_files': len(originals), 'independent_vendor_reference': False, 'runs': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
