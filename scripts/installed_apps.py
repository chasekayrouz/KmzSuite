"""Durable installation identity and dependency-free LaunchServices smoke checks."""
import errno
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time
from uuid import uuid4
from zipfile import ZipFile

from macos_apps import APPS, bundle_hashes, require_stopped, validate_bundle

REGISTRAR = '/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister'


def installed_directory():
    return Path.home() / 'Applications' / 'KmzSuite'


def clean_environment():
    return {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(Path.home()),
            'USER': os.environ.get('USER', ''), 'LANG': 'en_US.UTF-8'}


def register(directory):
    for name in APPS:
        subprocess.run([REGISTRAR, '-f', str(Path(directory) / name)], check=True)


def archive_rollback(directory, archive, verification):
    """Create one Finder-restorable, verified archive, without indexed old apps."""
    directory, archive, verification = Path(directory), Path(archive), Path(verification)
    if archive.exists() or verification.exists():
        raise FileExistsError('Rollback staging paths must be new')
    require_stopped(directory)
    # A qualified four-app installation is a valid rollback during the first
    # five-app upgrade. Candidate validation still always requires every app.
    names = [name for name in APPS if (directory/name).is_dir()]
    if set(names) not in (set(APPS), set(APPS)-{'KmzCoMet.app'}):
        raise ValueError('Previous installation is incomplete')
    before = bundle_hashes(directory, names)
    subprocess.run(['/usr/bin/ditto', '-c', '-k', '--sequesterRsrc', '--keepParent',
                    str(directory), str(archive)], check=True)
    with ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Rollback ZIP CRC check failed')
    verification.mkdir(parents=True)
    try:
        subprocess.run(['/usr/bin/ditto', '-x', '-k', str(archive), str(verification)], check=True)
        restored = verification / directory.name
        for name in names:
            validate_bundle(restored / name)
        if bundle_hashes(restored, names) != before or bundle_hashes(directory, names) != before:
            raise RuntimeError('Rollback archive round-trip changed bundle contents')
    finally:
        # Finder can add metadata after rmtree has enumerated this temporary
        # round-trip copy. Retry only an identified metadata-only residue.
        for attempt in range(3):
            try:
                shutil.rmtree(verification)
                break
            except OSError as error:
                if error.errno != errno.ENOTEMPTY or attempt == 2:
                    raise
                remaining = list(verification.rglob('*'))
                if any(p.is_symlink() or (not p.is_dir() and p.name != '.DS_Store') for p in remaining):
                    raise
    return before


def wait_result(path, directory, timeout=110, *, additional_results=()):
    from macos_apps import running_apps

    started = time.monotonic()
    deadline = started + timeout
    results = []
    expected = (path, *additional_results)
    phase = 'results'
    try:
        # A launcher can finish while its children are still exercising the viewers.
        # Give every diagnostic the same execution budget, then check shutdown.
        for result_path in expected:
            while not result_path.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'No installed-app smoke result: {result_path}')
                time.sleep(.1)
            result = json.loads(result_path.read_text())
            if not result.get('ok'):
                raise RuntimeError(f'Installed-app smoke failed: {result_path}: {result}')
            results.append(result)
        phase = 'shutdown'
        require_exit = time.monotonic() + 15
        while running_apps(directory):
            if time.monotonic() >= require_exit:
                raise TimeoutError('Installed-app smoke did not shut down')
            time.sleep(.1)
    except Exception as error:
        # Local evidence only. A successful in-app result is not proof of process
        # exit, and a missing result is not proof that the app is still running.
        # Never kill a process or let evidence capture turn failure into success.
        try:
            failure = {'ok': False, 'phase': phase, 'error': str(error),
                       'elapsed_seconds': round(time.monotonic() - started, 3),
                       'results': [{'path': str(p), 'exists': p.exists()} for p in expected],
                       'running_apps': [{'pid': pid, 'executable': executable}
                                        for pid, executable in running_apps(directory)]}
            (path.parent / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        except Exception as evidence_error:
            print(f'Could not save smoke failure evidence: {evidence_error}', file=sys.stderr)
        raise
    return results[0]


def smoke(directory, output, entry=None):
    """Launch real GUI bundles via the same LaunchServices layer as Spotlight."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require_stopped(directory)
    output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in APPS:
        validate_bundle(directory / name)
        info = plistlib.loads((directory / name / 'Contents/Info.plist').read_bytes())
        versions[name] = info['CFBundleShortVersionString']
    if len(set(versions.values())) != 1:
        raise ValueError('Installed application versions disagree')
    before = bundle_hashes(directory)
    results = {}
    siblings = ('KmzViewer', 'Kmz2Viewer', 'KmzConvert', 'KmzCoMet')
    for name in (*siblings, 'KmzSuite'):
        target = output / name
        target.mkdir()
        args = ['--smoke-output', str(target)]
        if name == 'KmzSuite':
            args += ['--smoke-launch-all']
        elif name == 'KmzConvert' and entry:
            args += ['--smoke-input', str(Path(entry).resolve()), '--smoke-inspect']
        elif name == 'KmzCoMet' and entry:
            args += ['--smoke-input', str(Path(entry).resolve())]
        subprocess.run(['/usr/bin/open', '-n', '-a', str(directory / (name + '.app')),
                        '--stdout', str(target / 'stdout.log'), '--stderr', str(target / 'stderr.log'),
                        '--args', *args], cwd=output, env=clean_environment(), check=True)
        children = tuple(target / sibling / 'result.json' for sibling in siblings) if name == 'KmzSuite' else ()
        result = wait_result(target / 'result.json', directory, additional_results=children)
        if name in ('KmzViewer', 'Kmz2Viewer', 'KmzCoMet'):
            if not result.get('frozen'):
                raise RuntimeError('Viewer did not use its frozen runtime')
        if name == 'KmzSuite':
            if set(result.get('launched', [])) != set(siblings):
                raise RuntimeError('Launcher could not open all installed siblings')
            for sibling in result['launched']:
                child = json.loads((target / sibling / 'result.json').read_text())
                if not child.get('ok'):
                    raise RuntimeError('Installed sibling smoke failed')
        results[name] = {'ok': True, 'checks': result.get('checks', []),
                         'frozen': result.get('frozen'), 'inspected_input': bool(entry) if name == 'KmzConvert' else None}
    target = output / 'viewer-handoff'
    target.mkdir()
    subprocess.run(['/usr/bin/open', '-n', '-a', str(directory / 'KmzViewer.app'),
                    '--stdout', str(target / 'stdout.log'), '--stderr', str(target / 'stderr.log'), '--args',
                    '--smoke-output', str(target), '--smoke-handoff', 'parent'],
                   cwd=output, env=clean_environment(), check=True)
    result = wait_result(target / 'result.json', directory)
    results['viewer-handoff'] = {'ok': True, 'checks': result['checks'], 'frozen': result['frozen']}
    if bundle_hashes(directory) != before:
        raise RuntimeError('Installed bundles changed during validation')
    receipt = {'schema': 1, 'ok': True, 'mechanism': 'macOS LaunchServices',
               'versions': versions, 'bundle_sha256': before, 'checks': results}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def isolated_smoke(directory, root, output):
    """Temporarily hide all checkout runtime resources; restore even on failure."""
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.is_relative_to(root):
        raise ValueError('Isolation evidence must be outside the checkout')
    directory = Path(directory).resolve()
    if directory.is_relative_to(root):
        raise ValueError('Isolation requires an independently installed/staged distribution')
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / 'fixtures'
    fixtures.mkdir()
    entry = root / 'TestData/Local/acquisition-01.wiff'
    copied_entry = None
    if entry.is_file():
        for source in (entry, entry.with_suffix('.wiff.scan')):
            shutil.copy2(source, fixtures / source.name)
        copied_entry = fixtures / entry.name
    hold = root.parent / ('.KmzSuite-development-hold-' + uuid4().hex)
    hold.mkdir()
    moved = []
    names = ('.venv', 'build', 'dist', '.build', '.cache', '.local', 'Apps', 'Sources',
             'Tools', 'resources', 'docs', 'packaging', 'scripts')
    try:
        for name in names:
            source = root / name
            if source.exists() or source.is_symlink():
                source.rename(hold / name)
                moved.append(name)
        receipt = smoke(directory, output / 'smoke', copied_entry)
        receipt['checkout_resources_unavailable'] = list(names)
        receipt['checkout_resources_temporarily_moved'] = moved
    finally:
        for name in reversed(moved):
            if (root / name).exists() or (root / name).is_symlink():
                raise RuntimeError(f'Restore blocked; originals remain safely in {hold}: {name}')
            (hold / name).rename(root / name)
        if any(hold.iterdir()):
            # Finder may create .DS_Store while folders are temporarily relocated.
            # Preserve any such residue rather than failing after a full restore.
            hold.rename(output / 'restore-metadata')
        else:
            hold.rmdir()
    (output / 'isolation-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
