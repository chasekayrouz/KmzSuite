"""Real native windows, sibling launching, and packaged GUI/CLI conversion."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from kmzcore.version import __version__

ROOT = Path(__file__).resolve().parents[1]
DIST = Path.home() / 'Applications/KmzSuite'


def read_result(directory, process=None, timeout=90):
    deadline = time.monotonic() + timeout
    path = directory / 'result.json'
    while not path.exists():
        if process and process.poll() is not None:
            raise RuntimeError(f'App exited before reporting: {directory}')
        if time.monotonic() > deadline:
            raise TimeoutError(directory)
        time.sleep(.1)
    if process:
        assert process.wait(timeout=10) == 0
    return json.loads(path.read_text())


def native(directory, app, *arguments):
    directory.mkdir(parents=True, exist_ok=True)
    executable = DIST / (app + '.app') / 'Contents/MacOS' / app
    with (directory / 'process.log').open('w') as handle:
        process = subprocess.Popen([str(executable), '--smoke-output', str(directory), *map(str, arguments)],
                                   cwd=directory, stdout=handle, stderr=handle)
        try:
            return read_result(directory, process)
        finally:
            if process.poll() is None:
                process.send_signal(2)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)


def main():
    global DIST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution', type=Path, default=DIST)
    args = parser.parse_args()
    DIST = args.distribution.resolve(strict=True)
    root = ROOT / '.local/validation/native' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True)
    result = native(root / 'launcher', 'KmzSuite', '--smoke-launch-all')
    assert result['ok'] and set(result['launched']) == {'KmzViewer', 'Kmz2Viewer', 'KmzConvert', 'KmzCoMet'}, result
    for name in result['launched']:
        assert read_result(root / 'launcher' / name)['ok']
    summaries = [{'case': 'launcher-and-independent-siblings', 'ok': True}]
    entry = ROOT / 'TestData/Local/acquisition-01.wiff'
    if entry.exists():
        for case, arguments, success in [
            ('convert', ['--smoke-input', entry], True),
            ('sparse', ['--smoke-input', entry, '--smoke-profile', 'sparse'], True),
            ('inspect', ['--smoke-input', entry, '--smoke-inspect'], True),
            ('error', ['--smoke-input', root / 'missing.wiff'], False),
            ('cancel', ['--smoke-input', entry, '--smoke-cancel'], False),
        ]:
            report = native(root / case, 'KmzConvert', *arguments)
            assert report['ok'] == success, report
            if case == 'convert':
                assert (root / case / 'converted.mzML').is_file()
                assert 'Converting ' in report['log'] and 'Wrote ' in report['log']
                from importlib.util import spec_from_file_location, module_from_spec
                spec = spec_from_file_location('integration_parity', ROOT / 'scripts/integration-parity.py')
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                verified = module.verify_output(entry, root / case / 'converted.mzML')
                assert verified['arrays_tic_eic_exact']
                (root / case / 'converted.mzML').unlink()
            elif case == 'sparse':
                output = root / case / 'converted.sp.mzML'
                assert output.is_file() and not (root / case / 'converted.mzML').exists()
                verified = module.verify_output(entry, output, profile='sparse')
                assert verified['arrays_tic_eic_exact']
                output.unlink()
            elif case == 'inspect':
                assert '"spectra"' in report['log'] and not (root / case / 'converted.mzML').exists()
            elif case == 'error':
                assert 'Missing entry point' in report['log']
            elif case == 'cancel':
                assert not (root / case / 'converted.mzML').exists()
                assert not list((root / case).glob('.kmzconvert-*.tmp'))
                assert 'cancelled' in report['status'].lower()
            summaries.append({'case': 'gui-' + case, 'ok': True, 'exit': report['exit']})
            # Retain useful small evidence without acquisition metadata or paths.
            (root / case / 'result.json').write_text(json.dumps({'ok': report['ok'], 'exit': report['exit'], 'status': report['status']}, indent=2) + '\n')
        second = ROOT / 'TestData/Local/acquisition-02.wiff'
        if second.exists():
            for case, profile, cancel in [('batch-full', 'full', False), ('batch-sparse', 'sparse', False),
                                          ('batch-cancel', 'full', True)]:
                directory = root / case
                directory.mkdir()
                suffix = '.sp.mzML' if profile == 'sparse' else '.mzML'
                first_output = directory / (entry.stem + suffix)
                second_output = directory / (second.stem + suffix)
                if profile == 'sparse':
                    first_output.write_bytes(b'existing-output')
                arguments = ['--smoke-input', entry, '--smoke-input', root / 'missing.wiff',
                             '--smoke-input', second, '--smoke-profile', profile]
                if cancel:
                    arguments.append('--smoke-cancel')
                report = native(directory, 'KmzConvert', *arguments)
                # Sample discovery runs before conversion; a missing queued
                # input has already failed when the first conversion is cancelled.
                expected = (['Cancelled', 'Failed', 'Not run'] if cancel else
                            ['Skipped' if profile == 'sparse' else 'Converted', 'Failed', 'Converted'])
                assert report['outcomes'] == expected, report
                assert report['exit'] == (130 if cancel else 1), report
                if cancel:
                    assert not first_output.exists() and not second_output.exists()
                    assert not list(directory.glob('.kmzconvert-*.tmp'))
                else:
                    assert 'Missing entry point' in report['log'] and 'Batch complete:' in report['status']
                    assert module.verify_output(second, second_output, profile=profile)['arrays_tic_eic_exact']
                    if profile == 'sparse':
                        assert first_output.read_bytes() == b'existing-output'
                    else:
                        assert module.verify_output(entry, first_output)['arrays_tic_eic_exact']
                    first_output.unlink()
                    second_output.unlink()
                summaries.append({'case': 'gui-' + case, 'ok': True, 'outcomes': report['outcomes']})
                (directory / 'result.json').write_text(json.dumps({key: report[key] for key in
                                                                ('ok', 'exit', 'status', 'outcomes', 'completed')}, indent=2) + '\n')
        else:
            summaries.append({'case': 'gui-batch-integration', 'skipped': 'No acquisition-02.wiff'})
        cli = DIST / 'kmzconvert'
        # Both frozen converter runtimes carry the shared native codec.
        for runtime in (DIST / 'Converter', DIST / 'KmzConvert.app/Contents/Resources/Converter'):
            assert list(runtime.rglob('kmzconvert_core/_native*.so')), runtime
        version = subprocess.run([str(cli), '--version'], cwd=root, capture_output=True, text=True)
        assert version.returncode == 0 and version.stdout.strip() == __version__
        inspected = subprocess.run([str(cli), str(entry), '--inspect'], cwd=root, capture_output=True, text=True)
        assert inspected.returncode == 0 and json.loads(inspected.stdout)['spectra'] > 0
        output = root / 'cli.mzML'
        converted = subprocess.run([str(cli), str(entry), '-o', str(output)], cwd=root, capture_output=True, text=True)
        assert converted.returncode == 0, converted.stderr
        original = output.stat().st_size
        rejected = subprocess.run([str(cli), str(entry), '-o', str(output)], cwd=root, capture_output=True, text=True)
        assert rejected.returncode == 1 and 'Refusing to overwrite' in rejected.stderr
        assert output.stat().st_size == original
        module.verify_output(entry, output)
        output.unlink()
        summaries.append({'case': 'frozen-cli-inspect-convert-no-clobber', 'ok': True})
        sparse_output = root / 'cli-sparse.sp.mzML'
        converted = subprocess.run([str(cli), str(entry), '--profile', 'sparse', '-o', str(root / 'cli-sparse.mzML')], cwd=root, capture_output=True, text=True)
        assert converted.returncode == 0, converted.stderr
        assert sparse_output.exists() and not (root / 'cli-sparse.mzML').exists()
        module.verify_output(entry, sparse_output, profile='sparse')
        rejected = subprocess.run([str(cli), str(entry), '--profile', 'sparse', '-o', str(sparse_output)], cwd=root, capture_output=True, text=True)
        assert rejected.returncode == 1 and 'Refusing to overwrite' in rejected.stderr
        sparse_output.unlink()
        summaries.append({'case': 'frozen-cli-sparse-normalization-no-clobber', 'ok': True})
    else:
        summaries.append({'case': 'local-converter-integration', 'skipped': 'No acquisition-01.wiff'})
    (root / 'validation.json').write_text(json.dumps({'ok': True, 'checks': summaries}, indent=2) + '\n')
    print(root)
    print('Native launcher, sibling apps and available converter checks passed.')


if __name__ == '__main__':
    main()
