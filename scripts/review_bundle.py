#!/usr/bin/env python3
"""Archive reviewed source and explicitly selected result evidence, excluding raw data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
BUDGET_BYTES = 1024 * 1024
RESERVED = {'REVIEW_START_HERE.md', 'MANIFEST.json', 'GIT_STATUS.txt', 'SOURCE_DIFF.patch'}
EVIDENCE_SUFFIXES = {'.md', '.html', '.csv', '.tsv', '.json', '.log', '.txt', '.svg', '.png', '.patch'}
EXCLUDED_PARTS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'build', 'dist', 'review-bundle', 'FromChase'}


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def relative_path(value, parser):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        parser.error('Bundle paths must be relative and contain no parent traversal.')
    return path


def checked_path(relative, parser):
    path = ROOT / relative
    if any((ROOT / Path(*relative.parts[:i])).is_symlink()
           for i in range(1, len(relative.parts) + 1)) or not path.resolve().is_relative_to(ROOT):
        parser.error(f'Bundle input must be a regular file inside the repository: {relative}')
    if any(part in EXCLUDED_PARTS or part.lower().startswith(('.env', 'credentials', 'secrets'))
           for part in relative.parts):
        parser.error(f'Excluded data/state/credential path: {relative}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, help='Short task label for the archive filename')
    parser.add_argument('--include', action='append', default=[], help='Reviewed new source file, relative to the repository; repeat as needed')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--source-subset', action='store_true', help='Include only explicit --include files (the default; retained for existing callers)')
    selection.add_argument('--tracked-source', action='store_true', help='Explicitly request the tracked source tree; size/privacy limits still apply')
    parser.add_argument('--evidence-root', help='Repository-relative directory containing reviewed results (may be ignored)')
    parser.add_argument('--evidence', action='append', default=[], help='Explicit result file relative to --evidence-root; archived under results/; repeat as needed')
    parser.add_argument('--allow-large', action='store_true', help='Deliberately override the compressed 1 MiB budget; requires --large-reason')
    parser.add_argument('--large-reason', help='Why essential review evidence cannot fit below 1 MiB; recorded in the guide and manifest')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.label):
        parser.error('Use letters, digits, dots, underscores and hyphens in the label.')
    if bool(args.evidence_root) != bool(args.evidence):
        parser.error('--evidence-root and at least one --evidence must be supplied together.')
    reason = ' '.join((args.large_reason or '').split())
    if args.allow_large and not reason:
        parser.error('--allow-large requires a non-empty --large-reason.')
    if args.large_reason is not None and not args.allow_large:
        parser.error('--large-reason requires --allow-large.')
    names = set(git('ls-files', '-z').decode().rstrip('\0').split('\0')) if args.tracked_source else set()
    names.update(args.include)
    files = {}
    kinds = {}
    origins = {}
    for name in sorted(names):
        if not name:
            continue
        relative = relative_path(name, parser)
        name = relative.as_posix()
        if name in RESERVED or relative.parts[0] == 'results':
            parser.error(f'Reserved generated archive entry: {name}')
        path = checked_path(relative, parser)
        if not path.exists():
            if name in args.include:
                parser.error(f'Requested source file is missing: {name}')
            continue  # A tracked file intentionally deleted in the working tree.
        ignored = subprocess.run(['git', 'check-ignore', '--no-index', '--quiet', '--', name], cwd=ROOT)
        if ignored.returncode not in (0, 1):
            raise SystemExit('Could not check archive exclusions.')
        if ignored.returncode == 0:
            parser.error(f'Ignored file needs explicit --evidence selection: {name}')
        if not path.is_file():
            parser.error(f'Only regular source files can enter a review archive: {name}')
        if (path.name.lower().startswith(('.env', 'credentials', 'secrets'))
                or path.suffix.lower() in {'.wiff', '.wiff2', '.scan', '.pem', '.key', '.p12'}
                or any(part.lower().endswith('.d') for part in Path(name).parts[:-1])
                or path.suffix.lower() == '.mzml' and not name.startswith('TestData/Regression/')):
            parser.error(f'Private/data/credential file cannot enter a review archive: {name}')
        files[name] = path.read_bytes()
        kinds[name] = 'source'
        origins[name] = name
    source_names = sorted(files)
    if args.evidence_root:
        evidence_root = relative_path(args.evidence_root, parser)
        for requested in sorted(set(args.evidence)):
            relative = relative_path(requested, parser)
            path = checked_path(evidence_root / relative, parser)
            if not path.is_file():
                parser.error(f'Requested evidence file is missing or not regular: {requested}')
            if any(part.lower().endswith('.d') for part in relative.parts[:-1]):
                parser.error(f'Files inside acquisition folders cannot be evidence: {requested}')
            if path.suffix.lower() not in EVIDENCE_SUFFIXES:
                parser.error(f'Unsupported evidence type (raw datasets/arrays are excluded): {requested}')
            if path.stat().st_size > 64 * 1024 * 1024:
                parser.error(f'Evidence exceeds 64 MiB; select a compact representative output: {requested}')
            name = 'results/' + relative.as_posix()
            files[name] = path.read_bytes()
            kinds[name] = 'result evidence'
            origins[name] = (evidence_root / relative).as_posix()
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    status = git('status', '--porcelain=v1', '--untracked-files=all')
    dirty = bool(status)
    guide = f'''# KmzSuite review: {args.label}

This archive snapshots reviewed working-tree files, including any explicit new
source files and explicitly selected result evidence. Baseline HEAD: `{commit}` on `{branch}`.
Working tree at capture: {"contains uncommitted changes" if dirty else "clean"}.

Start with results/REVIEW_NOTES.md and results/REPORT.md when present, then the
included source/validation documentation. TODO.md and docs/TODO.html contain
outstanding work. MANIFEST.json lists every payload file, size and SHA-256 digest.
GIT_STATUS.txt records the working tree. SOURCE_DIFF.patch shows tracked changes
against HEAD for included source files; untracked implementations are included as
source files and identified in Git status, rather than omitted from review.

This is a {"tracked-source" if args.tracked_source else "selected-source"} bundle with
{len(args.evidence)} explicitly reviewed result files. Ignored results are included
only through that selection; no ignored directory is copied wholesale. Raw/private
datasets, credentials, environments, caches, build products and previous archives
are excluded. Safe input identifiers/metadata may describe excluded datasets.
No application build or validation is implied by archiving. This bundle stays local.

Compressed-size policy: less than 1 MiB by default.
Large-bundle override: {reason if args.allow_large else "none; the default budget is enforced"}.
'''
    files['GIT_STATUS.txt'] = status
    files['SOURCE_DIFF.patch'] = git('diff', 'HEAD', '--', *source_names) if source_names else b''
    files['REVIEW_START_HERE.md'] = guide.encode('utf-8')
    for name in ('GIT_STATUS.txt', 'SOURCE_DIFF.patch', 'REVIEW_START_HERE.md'):
        kinds[name] = 'review metadata'
    manifest = {'baseline_commit': commit, 'branch': branch, 'working_tree_dirty': dirty,
                'size_policy': {'default_budget_bytes': BUDGET_BYTES, 'comparison': '<',
                                'allow_large': args.allow_large, 'large_reason': reason or None},
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                'file_details': {name: {'bytes': len(data), 'kind': kinds[name],
                                       'source': origins.get(name)} for name, data in files.items()}}
    destination = ROOT / 'review-bundle'
    if destination.is_symlink():
        parser.error('review-bundle must be a real repository-local directory.')
    destination.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = destination / f'KmzSuite-{args.label}-{stamp}-{commit[:7]}.zip'
    created = False
    try:
        with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as bundle:
            created = True
            for name, data in files.items():
                bundle.writestr(name, data)
            bundle.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        size = archive.stat().st_size
        if size >= BUDGET_BYTES and not args.allow_large:
            with ZipFile(archive) as bundle:
                ranked = sorted(bundle.infolist(), key=lambda item: item.compress_size, reverse=True)
            print(f'Review bundle exceeds the less-than-1-MiB budget: {size / BUDGET_BYTES:.2f} MiB compressed.', file=sys.stderr)
            print('Largest archive contributors (compressed; ZIP overhead is additional):', file=sys.stderr)
            for item in ranked[:10]:
                print(f'  {item.filename:<55} {item.compress_size / 1024:.1f} KiB', file=sys.stderr)
            print('Use a smaller evidence selection or explicitly authorize a larger bundle with '
                  '--allow-large --large-reason "Essential review evidence requires...". '
                  'Oversized archive removed.', file=sys.stderr)
            raise SystemExit(2)
    except BaseException:
        if created:
            archive.unlink(missing_ok=True)
        raise
    print(archive.relative_to(ROOT))


if __name__ == '__main__':
    main()
