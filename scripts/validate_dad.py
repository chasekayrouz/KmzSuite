#!/usr/bin/env python3
"""Qualify local DAD acquisitions without moving or modifying the originals.

Only aggregate results use safe IDs. Generated mzMLs contain source metadata and
belong in an ignored local output directory, never a public review bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

from psims.controlled_vocabulary.controlled_vocabulary import obo_cache
from kmzconvert_core.native import NativeWiffReader
from kmzconvert_core.writer import write_mzml


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    entries = sorted(args.fixtures.glob('*.wiff'))
    if not entries:
        parser.error('No WIFF acquisitions found.')
    args.output.mkdir(parents=True, exist_ok=False)
    obo_cache.use_remote = False
    verify = runpy.run_path(str(Path(__file__).with_name('integration-parity.py')))['verify_output']
    inputs = {p: digest(p) for p in args.fixtures.iterdir() if p.is_file()}
    results = []
    for i, entry in enumerate(entries, 1):
        for mode in ('full', 'sparse'):
            started = time.monotonic()
            ident = f'dad-{i:02}'
            # The converter's production reader; verify_output decodes through the interpreted reference.
            with NativeWiffReader(entry) as reader:
                assert reader.dad is not None, reader.dad_message
                path = write_mzml(reader, args.output/f'{ident}-{mode}.mzML', profile=mode)
                converted = time.monotonic()
            result = verify(entry, path, mode, include_signature=False)
            results.append(dict(id=ident, profile=mode, bytes=path.stat().st_size,
                                conversion_seconds=round(converted-started, 3),
                                verification_seconds=round(time.monotonic()-converted, 3), **result))
            (args.output/'summary.json').write_text(json.dumps(dict(ok=False, pending=True, files=results), indent=2))
            print(ident, mode, 'passed', flush=True)
    assert all(digest(p) == before for p, before in inputs.items()), 'An original input changed'
    (args.output/'summary.json').write_text(json.dumps(dict(ok=True, originals_unchanged=True, input_files=len(inputs),
                                                         independent_vendor_reference=False, files=results), indent=2))


if __name__ == '__main__':
    main()
