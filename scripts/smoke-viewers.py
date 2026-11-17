#!/usr/bin/env python3
"""Test packaged viewer apps with repo-local evidence, including LaunchServices."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmzcore.runtime.synthetic import write_fixture
from macos_apps import APPS, bundle_hashes, copy_bundle, require_stopped, validate_bundle, running_apps


def wait_file(path, seconds=110):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(.2)
    return json.loads(path.read_text())


def audit(app):
    frameworks = app / 'Contents/Frameworks'
    for item in app.rglob('*'):
        if item.is_symlink() and not item.resolve().is_relative_to(app.resolve()):
            raise AssertionError(f'External bundle symlink: {item}')
    candidates = [p for p in frameworks.rglob('*') if p.is_file() and not p.is_symlink() and (p.suffix in ('.so', '.dylib') or '.framework/Versions/' in str(p) and p.suffix == '')]
    checked = 0
    for path in candidates:
        raw = path.open('rb').read(4)
        if raw not in (b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xfe\xed\xfa\xcf'):
            continue
        info = subprocess.check_output(['otool', '-L', str(path)], text=True)
        for line in info.splitlines()[1:]:
            dependency = line.strip().split(' (', 1)[0]
            assert dependency.startswith(('@', '/System/Library/', '/usr/lib/')), (path, dependency)
        checked += 1
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('--in-place', action='store_true', help='Test the exact supplied apps, without copying')
    parser.add_argument('--report', type=Path, help='Write a new promotion receipt after all checks pass')
    args = parser.parse_args()
    build = args.build.resolve()
    if args.report and args.report.exists():
        parser.error('Refusing to overwrite an existing validation receipt')
    for name in APPS:
        validate_bundle(build / name)
    before = bundle_hashes(build)
    if args.in_place:
        require_stopped(build)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    outside = ROOT / '.local/validation/viewers' / stamp
    outside.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'QT_', 'DYLD_', 'VIRTUAL_ENV', 'CONDA', 'MPLCONFIG', 'PYQTGRAPH', '_PYI'))}
    env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    results = []
    for key, name in [('ms1', 'KmzViewer'), ('ms2', 'Kmz2Viewer')]:
        source = build / (name + '.app')
        app = source if args.in_place else outside / source.name
        if not args.in_place:
            copy_bundle(source, app)
        count = audit(app)
        output = outside / (key + '-direct')
        output.mkdir()
        with (output / 'process.log').open('w') as log:
            process = subprocess.Popen([str(app / 'Contents/MacOS' / name), '--smoke-output', str(output)], cwd=outside, env=env, stdout=log, stderr=log)
            code = process.wait(timeout=120)
        assert code == 0, ((output / 'result.json').read_text() if (output / 'result.json').exists() else (output / 'process.log').read_text())
        result = wait_file(output / 'result.json')
        assert code == 0 and result['ok'], result
        results.append(result | {'output': str(output), 'native_binaries_audited': count})
        print(key, 'direct:', result['seconds'], 'seconds', flush=True)
        output = outside / (key + '-launchservices')
        output.mkdir()
        a, b = output / 'Synthetic Ω one.mzML', output / 'Synthetic two.mzML'
        write_fixture(a)
        write_fixture(b)
        subprocess.run(['/usr/bin/open', '-n', '-a', str(app), str(a), '--args', '--smoke-output', str(output), '--smoke-external'], env=env, cwd=outside, check=True)
        wait_file(output / 'ready.json')
        subprocess.run(['/usr/bin/open', '-a', str(app), str(b), str(a)], env=env, cwd=outside, check=True)
        result = wait_file(output / 'result.json')
        assert result['ok'], result
        results.append(result | {'output': str(output)})
        deadline = time.monotonic() + 10
        while running_apps(app.parent):
            if time.monotonic() > deadline:
                raise TimeoutError('LaunchServices app did not exit')
            time.sleep(.1)
        print(key, 'LaunchServices:', result['seconds'], 'seconds', flush=True)
        scenarios = [('load', 'cancel'), ('load', 'close'), ('load', 'quit')]
        scenarios += [('load', 'clear')] if key == 'ms2' else [
            ('extract', 'cancel'), ('extract', 'close'), ('extract', 'quit'),
            ('spectrum', 'close'), ('spectrum', 'quit')]
        for operation, action in scenarios:
            output = outside / f'{key}-{operation}-{action}'
            output.mkdir()
            with (output / 'process.log').open('w') as log:
                process = subprocess.Popen([str(app / 'Contents/MacOS' / name),
                    '--smoke-output', str(output), '--smoke-lifecycle', operation, action],
                    cwd=outside, env=env, stdout=log, stderr=log)
                code = process.wait(timeout=60)
            result = wait_file(output / 'result.json')
            assert code == 0 and result['ok'], result
            results.append(result | {'output': str(output)})
            print(key, operation, action, 'passed', flush=True)
    output = outside / 'viewer-handoff'
    source = build if args.in_place else outside
    subprocess.run(['/usr/bin/open', '-n', '-a', str(source / 'KmzViewer.app'), '--args',
                    '--smoke-output', str(output), '--smoke-handoff', 'parent'],
                   env=env, cwd=outside, check=True)
    result = wait_file(output / 'result.json')
    assert result['ok'], result
    results.append(result | {'output': str(output)})
    print('Cross-viewer handoff: passed', flush=True)
    deadline = time.monotonic() + 10
    while running_apps(source):
        if time.monotonic() > deadline:
            raise TimeoutError('Handoff apps did not exit')
        time.sleep(.1)
    if bundle_hashes(build) != before:
        raise RuntimeError('Build changed during validation; no receipt issued')
    (outside / 'validation.json').write_text(json.dumps(results, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as handle:
            json.dump(dict(schema=1, ok=True, build=str(build), in_place=args.in_place,
                           bundle_sha256=before, results=results), handle, indent=2)
            handle.write('\n')
        print('Verified receipt:', args.report, flush=True)
    print(outside, flush=True)


if __name__ == '__main__':
    main()
