#!/usr/bin/env python3
"""Review or reclaim declared local artifacts; protect installed apps and inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from installed_apps import installed_directory, smoke
from macos_apps import APPS, bundle_hashes, running_apps, validate_bundle

ROOT = Path(__file__).resolve().parents[1]
RAW_SUFFIXES = {'.wiff', '.wiff2', '.scan', '.mzml'}


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Artifact paths must be repository-relative')
    path = root / relative
    for count in range(1, len(relative.parts) + 1):
        if (root / Path(*relative.parts[:count])).is_symlink():
            raise ValueError('Artifact path traverses a symlink: ' + str(relative))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact is outside the checkout')
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def candidates(root, mode, retention):
    """Only known disposable trees and explicitly inventoried generated files."""
    result = []
    directories = ['build', 'dist', '.pytest_cache']
    local = root / '.local'
    if local.is_dir() and not local.is_symlink():
        directories += [str(p.relative_to(root)) for p in local.iterdir()
                        if p.is_dir() and re.fullmatch(r'install-\d{8}T\d+Z', p.name)]
    if mode == 'deep':
        directories += ['.venv', '.build', '.cache/uv', '.cache/pyinstaller', '.cache/swiftpm', '.cache/pip']
    protected = [safe_path(root, value['path']) for value in retention.get('golden', [])]
    for name in directories:
        path = safe_path(root, name)
        if not path.exists():
            continue
        if not path.is_dir():
            raise ValueError('Expected artifact directory: ' + name)
        if any(p == path or p.is_relative_to(path) for p in protected):
            raise ValueError('Candidate directory contains golden data: ' + name)
        if running_apps(path):
            raise RuntimeError('Quit checkout-local apps before cleaning: ' + name)
        if any(p.suffix.lower() in RAW_SUFFIXES for p in path.rglob('*') if not p.is_dir()):
            raise ValueError('Candidate contains acquisition files; preserve and inspect: ' + name)
        result.append(path)
    for item in retention.get('transient', []):
        path = safe_path(root, item['path'])
        if not path.is_relative_to(root / '.local') or path.suffix.lower() not in {'.mzml', '.npz'}:
            raise ValueError('Only declared local generated mzML/NPZ artifacts may be pruned')
        if path in protected:
            raise ValueError('Transient candidate is also marked golden')
        if path.exists():
            if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError('Declared transient changed; inspect before deletion: ' + str(path.relative_to(root)))
            result.append(path)
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().rstrip('\0').split('\0'))
    for path in result:
        relative = path.relative_to(root).as_posix()
        if any(t == relative or t.startswith(relative + '/') for t in tracked):
            raise ValueError('Candidate contains tracked source: ' + relative)
    return result


def check_install(directory):
    directory = Path(directory).resolve(strict=True)
    if directory.is_relative_to(ROOT):
        raise ValueError('Installed applications must be independent of the checkout')
    if not any(directory.is_relative_to(base) for base in (Path.home() / 'Applications', Path('/Applications'))):
        raise ValueError('Installed applications must be in a durable Applications location')
    receipt = json.loads((directory / '.kmz-install.json').read_text())
    required = {'.venv', 'build', '.cache', '.local', 'Sources', 'Apps', 'Tools'}
    if not receipt.get('ok') or not required.issubset(receipt.get('checkout_resources_unavailable', [])):
        raise ValueError('A successful checkout-isolation receipt is required before cleanup')
    for name in APPS:
        validate_bundle(directory / name)
    if bundle_hashes(directory) != receipt['bundle_sha256']:
        raise ValueError('Installed apps changed since isolation validation')
    return directory


def allocated(path):
    seen, total = set(), 0
    for entry in [path, *path.rglob('*')] if path.is_dir() else [path]:
        stat = entry.lstat()
        key = stat.st_dev, stat.st_ino
        if key not in seen:
            seen.add(key)
            total += stat.st_blocks * 512
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('standard', 'deep'), default='standard')
    parser.add_argument('--apply', action='store_true', help='Apply the printed plan; default is dry-run')
    parser.add_argument('--dry-run', action='store_true', help='Explicitly request the default dry-run')
    parser.add_argument('--installed', type=Path, default=installed_directory())
    parser.add_argument('--register-transient', type=Path, action='append', default=[])
    parser.add_argument('--golden', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error('--apply and --dry-run are mutually exclusive')
    manifest = ROOT / '.local/retention.json'
    retention = json.loads(manifest.read_text()) if manifest.exists() else {'schema': 1, 'golden': [], 'transient': []}
    if args.register_transient or args.golden:
        if args.apply:
            parser.error('Registration and deletion must be separate invocations')
        for kind, values in [('transient', args.register_transient), ('golden', args.golden)]:
            entries = {v['path']: v for v in retention[kind]}
            for relative in values:
                path = safe_path(ROOT, relative)
                if not path.is_relative_to(ROOT / '.local') or path.suffix.lower() not in {'.mzml', '.npz'} or not path.is_file():
                    parser.error('Register regular generated files inside .local only')
                name = path.relative_to(ROOT).as_posix()
                entries[name] = {'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
                if kind == 'golden':
                    retention['transient'] = [v for v in retention['transient'] if v['path'] != name]
                elif any(v['path'] == name for v in retention['golden']):
                    parser.error('Golden data cannot be registered as transient')
            retention[kind] = list(entries.values())
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps(retention, indent=2) + '\n')
        print('Saved explicit artifact retention declarations; nothing deleted.')
        return
    installed = check_install(args.installed)
    plan = candidates(ROOT, args.mode, retention)
    details = [{'path': p.relative_to(ROOT).as_posix(), 'allocated_bytes': allocated(p)} for p in plan]
    print(json.dumps({'mode': args.mode, 'apply': args.apply, 'candidates': details,
                      'candidate_GB': sum(p['allocated_bytes'] for p in details) / 1e9}, indent=2), flush=True)
    if not args.apply:
        return
    # Bind the final live smoke to the same unchanged installed bundle identity.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    smoke(installed, ROOT / '.local/validation/cleanup' / stamp)
    if candidates(ROOT, args.mode, retention) != plan:
        raise RuntimeError('Cleanup plan changed during validation')
    # Check every golden hash before doing any deletion.
    for item in retention.get('golden', []):
        path = safe_path(ROOT, item['path'])
        if not path.is_file() or digest(path) != item['sha256']:
            raise ValueError('Golden data identity changed; cleanup stopped')
    log = ROOT / '.local/cleanup-receipts' / (stamp + '.json')
    log.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'mode': args.mode, 'installed_versions': json.loads((installed / '.kmz-install.json').read_text())['versions'],
               'planned': details, 'deleted': [], 'golden_retained': len(retention.get('golden', []))}
    log.write_text(json.dumps(receipt, indent=2) + '\n')
    for path, detail in zip(plan, details):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        receipt['deleted'].append(detail)
        log.write_text(json.dumps(receipt, indent=2) + '\n')
    print('Cleanup receipt:', log.relative_to(ROOT))


if __name__ == '__main__':
    main()
