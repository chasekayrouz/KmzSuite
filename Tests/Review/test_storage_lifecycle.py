"""Destructive storage operations must protect software, originals and goldens."""
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cleanup = load('cleanup')
installer = load('install_local')
installed = load('installed_apps')


def test_fingerprint_ignores_only_the_mutable_app_management_label(tmp_path, monkeypatch):
    import macos_apps
    (tmp_path / 'program').write_bytes(b'executable')
    attributes = []
    monkeypatch.setattr(macos_apps, 'extended_attributes', lambda _: attributes)
    original = macos_apps.fingerprint(tmp_path)
    for value in (b'first launch', b'new host access state'):
        attributes[:] = [(b'com.apple.macl', value)]
        assert macos_apps.fingerprint(tmp_path) == original
    attributes.clear()
    assert macos_apps.fingerprint(tmp_path) == original
    for key in (b'com.apple.macl.other', b'com.apple.quarantine', b'com.apple.ResourceFork'):
        attributes[:] = [(key, b'one')]
        first = macos_apps.fingerprint(tmp_path)
        attributes[:] = [(key, b'two')]
        assert first != original and macos_apps.fingerprint(tmp_path) != first


@pytest.mark.parametrize('change', ['bytes', 'mode', 'link', 'entry'])
def test_fingerprint_still_detects_payload_changes(tmp_path, monkeypatch, change):
    import macos_apps
    monkeypatch.setattr(macos_apps, 'extended_attributes', lambda _: [])
    program = tmp_path / 'program'
    program.write_bytes(b'executable')
    program.chmod(0o755)
    link = tmp_path / 'alias'
    link.symlink_to('program')
    original = macos_apps.fingerprint(tmp_path)
    if change == 'bytes':
        program.write_bytes(b'replacement')
    elif change == 'mode':
        program.chmod(0o644)
    elif change == 'link':
        link.unlink()
        link.symlink_to('elsewhere')
    else:
        (tmp_path / 'extra').mkdir()
    assert macos_apps.fingerprint(tmp_path) != original


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS ditto and xattrs')
def test_bundle_copy_preserves_modes_links_and_extended_attributes(tmp_path):
    import subprocess
    from macos_apps import copy_bundle, fingerprint
    source, target = tmp_path / 'source', tmp_path / 'copy'
    source.mkdir()
    (source / 'program').write_bytes(b'fixture')
    (source / 'program').chmod(0o755)
    (source / 'alias').symlink_to('program')
    subprocess.run(['/usr/bin/xattr', '-w', 'org.kmzsuite.test', 'fixture', str(source / 'program')], check=True)
    copy_bundle(source, target)
    assert fingerprint(target) == fingerprint(source)


@pytest.mark.parametrize('residue', ['metadata', 'unexpected', 'symlink'])
def test_rollback_cleanup_retries_only_finder_metadata(tmp_path, monkeypatch, residue):
    import errno
    original = tmp_path/'KmzSuite';original.mkdir()
    for name in installed.APPS:(original/name).mkdir()
    archive,verification=tmp_path/'rollback.zip',tmp_path/'verification'
    protected=tmp_path/'keep.txt';protected.write_text('original')
    monkeypatch.setattr(installed,'require_stopped',lambda p:None)
    monkeypatch.setattr(installed,'validate_bundle',lambda p:None)
    monkeypatch.setattr(installed,'bundle_hashes',lambda *a:{'unchanged':'identity'})
    def run(command,**kwargs):
        if '-c' in command:
            with ZipFile(archive,'w') as zipped:zipped.writestr('KmzSuite/check.txt','verified')
        else:
            (verification/original.name).mkdir()
    monkeypatch.setattr(installed.subprocess,'run',run)
    remove=installed.shutil.rmtree;calls=[]
    def raced(path):
        calls.append(path)
        if len(calls)==1:
            remove(path);path.mkdir()
            if residue=='symlink':(path/'.DS_Store').symlink_to(protected)
            else:(path/('.DS_Store' if residue=='metadata' else 'unexpected.txt')).write_bytes(b'late file')
            raise OSError(errno.ENOTEMPTY,'Directory not empty',str(path))
        remove(path)
    monkeypatch.setattr(installed.shutil,'rmtree',raced)
    if residue=='metadata':
        assert installed.archive_rollback(original,archive,verification)=={'unchanged':'identity'}
        assert len(calls)==2 and not verification.exists()
    else:
        with pytest.raises(OSError):installed.archive_rollback(original,archive,verification)
        assert len(calls)==1 and verification.exists()
    assert protected.read_text()=='original' and original.is_dir()


@pytest.mark.parametrize('scenario', ['success', 'child-failure', 'shutdown-timeout'])
def test_launcher_waits_for_child_diagnostics_before_shutdown(tmp_path, monkeypatch, scenario):
    from types import SimpleNamespace
    import macos_apps
    parent, child = tmp_path / 'parent.json', tmp_path / 'child.json'
    parent.write_text('{"ok":true,"launched":["viewer"]}')
    elapsed = [0.]

    def advance(seconds):
        elapsed[0] += seconds
        if elapsed[0] >= 20 and not child.exists():
            child.write_text(json.dumps({'ok': scenario != 'child-failure'}))

    monkeypatch.setattr(installed, 'time', SimpleNamespace(monotonic=lambda: elapsed[0], sleep=advance))
    monkeypatch.setattr(macos_apps, 'running_apps',
                        lambda _: [(123, '/test/Viewer.app/Contents/MacOS/Viewer')]
                        if elapsed[0] < 22 or scenario == 'shutdown-timeout' else [])
    if scenario == 'success':
        assert installed.wait_result(parent, tmp_path, additional_results=(child,))['launched'] == ['viewer']
        assert 22 <= elapsed[0] < 23
    else:
        error = RuntimeError if scenario == 'child-failure' else TimeoutError
        with pytest.raises(error, match='smoke failed|did not shut down'):
            installed.wait_result(parent, tmp_path, additional_results=(child,))
        failure = json.loads((tmp_path / 'failure.json').read_text())
        assert failure['phase'] == ('results' if scenario == 'child-failure' else 'shutdown')
        assert all(result['exists'] for result in failure['results'])
        assert failure['running_apps'][0]['pid'] == 123


@pytest.mark.parametrize('scenario', ['quits', 'still-running'])
def test_native_smoke_waits_for_launched_siblings_to_quit(tmp_path, monkeypatch, scenario):
    """release.py requires every app stopped right after validation; a sibling may still be quitting."""
    from types import SimpleNamespace
    import installed_apps
    import macos_apps
    native = load('smoke-native')
    siblings = ['KmzViewer', 'Kmz2Viewer', 'KmzConvert', 'KmzCoMet']

    def launcher(directory, app, *arguments):
        for name in siblings:
            (directory / name).mkdir(parents=True)
            (directory / name / 'result.json').write_text('{"ok": true}')
        (directory / 'result.json').write_text(json.dumps({'ok': True, 'launched': siblings}))
        return {'ok': True, 'launched': siblings}

    elapsed = [0.]
    monkeypatch.setattr(native, 'native', launcher)
    monkeypatch.setattr(native, 'ROOT', tmp_path)  # No local acquisitions: the launcher stage only.
    monkeypatch.setattr(installed_apps, 'time', SimpleNamespace(
        monotonic=lambda: elapsed[0], sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)))
    monkeypatch.setattr(macos_apps, 'running_apps', lambda _: [(123, '/test/KmzCoMet.app/Contents/MacOS/KmzCoMet')]
                        if elapsed[0] < 2 or scenario == 'still-running' else [])
    monkeypatch.setattr(sys, 'argv', ['smoke-native.py', '--distribution', str(tmp_path)])
    runs = tmp_path / '.local/validation/native'
    if scenario == 'quits':
        native.main()
        assert 2 <= elapsed[0] < 3
        assert json.loads(next(runs.glob('*/validation.json')).read_text())['ok']
    else:
        with pytest.raises(TimeoutError, match='did not shut down'):
            native.main()
        failure = json.loads(next(runs.glob('*/launcher/failure.json')).read_text())
        assert failure['phase'] == 'shutdown' and failure['running_apps'][0]['pid'] == 123
        assert not list(runs.glob('*/validation.json'))


@pytest.mark.parametrize('scenario', ['missing-result', 'failed-result', 'shutdown-timeout'])
def test_live_smoke_failure_preserves_cleanup_candidates(repository, monkeypatch, scenario):
    from types import SimpleNamespace
    import macos_apps

    build = repository / 'build'
    build.mkdir()
    sentinel = build / 'staging.txt'
    sentinel.write_text('retained build staging')
    elapsed = [0.]
    monkeypatch.setattr(installed, 'time', SimpleNamespace(
        monotonic=lambda: elapsed[0], sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)))
    live = [(123, '/test/Viewer.app/Contents/MacOS/Viewer')] if scenario == 'shutdown-timeout' else []
    monkeypatch.setattr(macos_apps, 'running_apps', lambda _: live)
    monkeypatch.setattr(cleanup, 'ROOT', repository)
    monkeypatch.setattr(cleanup, 'check_install', lambda _: repository / 'installed')
    monkeypatch.setattr(sys, 'argv', ['cleanup.py', '--mode', 'standard', '--apply'])

    def failed_smoke(directory, output):
        output.mkdir(parents=True)
        if scenario != 'missing-result':
            (output / 'result.json').write_text(json.dumps({'ok': scenario == 'shutdown-timeout'}))
        installed.wait_result(output / 'result.json', directory, timeout=1)

    monkeypatch.setattr(cleanup, 'smoke', failed_smoke)
    with pytest.raises((RuntimeError, TimeoutError), match='smoke'):
        cleanup.main()
    assert sentinel.read_text() == 'retained build staging'
    assert not (repository / '.local/cleanup-receipts').exists()
    failure = json.loads(next((repository / '.local/validation/cleanup').glob('*/failure.json')).read_text())
    assert not failure['ok']
    assert failure['results'][0]['exists'] == (scenario != 'missing-result')
    assert bool(failure['running_apps']) == (scenario == 'shutdown-timeout')
    assert failure['phase'] == ('shutdown' if scenario == 'shutdown-timeout' else 'results')


def test_smoke_evidence_failure_keeps_original_error(tmp_path, monkeypatch, capsys):
    import macos_apps

    result = tmp_path / 'result.json'
    result.write_text('{"ok": false}')
    def cannot_enumerate(_):
        raise OSError('process enumeration unavailable')
    monkeypatch.setattr(macos_apps, 'running_apps', cannot_enumerate)
    with pytest.raises(RuntimeError, match='Installed-app smoke failed'):
        installed.wait_result(result, tmp_path)
    assert 'Could not save smoke failure evidence' in capsys.readouterr().err


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(cleanup.subprocess, 'check_output', lambda *a, **kw: b'')
    monkeypatch.setattr(cleanup, 'running_apps', lambda p: [])
    return root


@pytest.mark.parametrize('suffix', ['.wiff', '.mzML'])
def test_original_acquisition_in_disposable_tree_stops_cleanup(repository, suffix):
    build = repository / 'build'
    build.mkdir()
    (build / ('original' + suffix)).write_bytes(b'immutable')
    with pytest.raises(ValueError, match='acquisition'):
        cleanup.candidates(repository, 'deep', {})
    assert (build / ('original' + suffix)).read_bytes() == b'immutable'


def test_golden_and_changed_transient_stop_cleanup(repository):
    build = repository / 'build'
    build.mkdir()
    reference = build / 'reference.mzML'
    reference.write_bytes(b'golden')
    with pytest.raises(ValueError, match='golden'):
        cleanup.candidates(repository, 'standard', {'golden': [{'path': 'build/reference.mzML'}]})
    local = repository / '.local'
    local.mkdir()
    reference.rename(local / 'reference.mzML')
    output = local / 'generated.mzML'
    output.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        cleanup.candidates(repository, 'standard', {'transient': [{'path': '.local/generated.mzML', 'bytes': 7, 'sha256': 'wrong'}]})


def test_symlink_candidate_and_parent_traversal_are_rejected(repository, tmp_path):
    (repository / 'build').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='symlink'):
        cleanup.candidates(repository, 'deep', {})
    with pytest.raises(ValueError, match='relative'):
        cleanup.safe_path(repository, '../outside')


def test_standard_retains_environment_standards_and_unregistered_data(repository):
    for name in ['.venv', '.build', '.cache/standards', 'TestData/Local', 'FromChase', '.local/unknown']:
        path = repository / name
        path.mkdir(parents=True)
        (path / 'keep.txt').write_text('keep')
    (repository / 'build').mkdir()
    (repository / 'dist').mkdir()
    paths = cleanup.candidates(repository, 'standard', {})
    assert {p.name for p in paths} == {'build', 'dist'}
    deep = cleanup.candidates(repository, 'deep', {})
    assert {p.name for p in deep} == {'build', 'dist', '.venv', '.build'}


def test_tracked_files_and_running_checkout_apps_stop_cleanup(repository, monkeypatch):
    (repository / 'build').mkdir()
    monkeypatch.setattr(cleanup.subprocess, 'check_output', lambda *a, **kw: b'build/important.py\0')
    with pytest.raises(ValueError, match='tracked'):
        cleanup.candidates(repository, 'standard', {})
    monkeypatch.setattr(cleanup, 'running_apps', lambda p: [(123, 'running')])
    with pytest.raises(RuntimeError, match='Quit'):
        cleanup.candidates(repository, 'deep', {})


@pytest.mark.parametrize('interpreter', [sys.version_info, (3, 9)])
@pytest.mark.parametrize('residue', ['metadata', 'persistent', 'unexpected', 'symlink', 'directory'])
def test_cleanup_retries_only_finder_metadata_during_removal(repository, tmp_path, monkeypatch, residue, interpreter):
    import os
    from types import SimpleNamespace
    # Cleanup also runs on the system Python, whose rmtree has only onerror.
    monkeypatch.setattr(cleanup, 'sys', SimpleNamespace(version_info=interpreter))
    stage = repository / 'build/stage'
    stage.mkdir(parents=True)
    (stage / 'staging.txt').write_text('disposable')
    (repository / 'installed').mkdir()
    (repository / 'installed/.kmz-install.json').write_text('{"versions": {}}')
    protected = tmp_path / 'keep.txt'
    protected.write_text('original')
    monkeypatch.setattr(cleanup, 'ROOT', repository)
    monkeypatch.setattr(cleanup, 'check_install', lambda _: repository / 'installed')
    monkeypatch.setattr(cleanup, 'smoke', lambda directory, output: None)
    monkeypatch.setattr(cleanup, 'time', SimpleNamespace(sleep=lambda seconds: None))
    monkeypatch.setattr(sys, 'argv', ['cleanup.py', '--mode', 'standard', '--apply'])
    rmdir, writes = os.rmdir, []

    def finder_writes_late(path, *, dir_fd=None):
        # Finder drops its state into the emptied folder just before each rmdir.
        if Path(path).name == 'stage' and (residue == 'persistent' or not writes):
            writes.append(path)
            late = stage / ('unexpected.txt' if residue == 'unexpected' else '.DS_Store')
            if residue == 'symlink':
                late.symlink_to(protected)
            elif residue == 'directory':
                late.mkdir()
            else:
                late.write_bytes(b'Finder state')
        rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, 'rmdir', finder_writes_late)
    if residue == 'metadata':
        cleanup.main()
        assert not (repository / 'build').exists() and len(writes) == 1
    else:
        with pytest.raises(OSError, match='not empty'):
            cleanup.main()
        late = {p.name for p in stage.iterdir()}
        assert late == ({'unexpected.txt'} if residue == 'unexpected' else {'.DS_Store'})
        assert len(writes) == (6 if residue == 'persistent' else 1)
    receipt = json.loads(next((repository / '.local/cleanup-receipts').glob('*.json')).read_text())
    assert [p['path'] for p in receipt['deleted']] == (['build'] if residue == 'metadata' else [])
    assert protected.read_text() == 'original'


def test_isolation_failure_restores_every_checkout_folder(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    for name in ['.venv', 'build', '.local', 'Sources', 'scripts']:
        (root / name).mkdir()
        (root / name / 'sentinel').write_text(name)
    distribution = tmp_path / 'durable'
    distribution.mkdir()

    def fail(*args):
        assert all(not (root / name).exists() for name in ['.venv', 'build', '.local', 'Sources', 'scripts'])
        raise RuntimeError('simulated app failure')

    monkeypatch.setattr(installed, 'smoke', fail)
    with pytest.raises(RuntimeError, match='simulated'):
        installed.isolated_smoke(distribution, root, tmp_path / 'evidence')
    for name in ['.venv', 'build', '.local', 'Sources', 'scripts']:
        assert (root / name / 'sentinel').read_text() == name


def test_failed_candidate_validation_leaves_installed_apps_untouched(tmp_path, monkeypatch):
    applications = tmp_path / 'Applications'
    current = applications / 'KmzSuite'
    current.mkdir(parents=True)
    (current / '.kmz-install.json').write_text('{}')
    (current / 'build.json').write_text('previous software')
    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setattr(installer, 'ROOT', tmp_path / 'checkout')
    monkeypatch.setattr(installer.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(installer, 'require_stopped', lambda p: None)
    monkeypatch.setattr(installer, 'validate_bundle', lambda p: None)
    monkeypatch.setattr(installer, 'check_distribution', lambda p: None)
    monkeypatch.setattr(installer, 'register', lambda p: None)

    def fail(*args):
        raise RuntimeError('rejected candidate')

    monkeypatch.setattr(installer, 'isolated_smoke', fail)
    with pytest.raises(RuntimeError, match='rejected candidate'):
        installer.install(source, current, tmp_path / 'receipt')
    assert (current / 'build.json').read_text() == 'previous software'
    assert not list(applications.glob('.KmzSuite-install-*'))


def test_registration_failure_restores_previous_install(tmp_path, monkeypatch):
    applications = tmp_path / 'Applications'
    current = applications / 'KmzSuite'
    current.mkdir(parents=True)
    (current / '.kmz-install.json').write_text('{}')
    (current / 'build.json').write_text('previous software')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'new').write_text('replacement')
    monkeypatch.setattr(installer, 'ROOT', tmp_path / 'checkout')
    monkeypatch.setattr(installer.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(installer, 'require_stopped', lambda p: None)
    monkeypatch.setattr(installer, 'validate_bundle', lambda p: None)
    monkeypatch.setattr(installer, 'check_distribution', lambda p: None)
    monkeypatch.setattr(installer, 'archive_rollback', lambda *a: a[1].write_bytes(b'rollback'))

    def validated(stage, root, output):
        output.mkdir()
        return {'ok': True}

    calls = []
    def register(path):
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError('registration failed')

    monkeypatch.setattr(installer, 'isolated_smoke', validated)
    monkeypatch.setattr(installer, 'register', register)
    with pytest.raises(RuntimeError, match='registration failed'):
        installer.install(source, current, tmp_path / 'receipt')
    assert (current / 'build.json').read_text() == 'previous software'
    assert not (current / 'new').exists()
    assert len(calls) == 2


def test_rollback_exchange_failure_restores_previous_install(tmp_path, monkeypatch):
    applications = tmp_path / 'Applications'
    current = applications / 'KmzSuite'
    current.mkdir(parents=True)
    (current / '.kmz-install.json').write_text('{}')
    (current / 'build.json').write_text('previous software')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'new').write_text('replacement')
    monkeypatch.setattr(installer, 'ROOT', tmp_path / 'checkout')
    monkeypatch.setattr(installer.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(installer, 'require_stopped', lambda p: None)
    monkeypatch.setattr(installer, 'validate_bundle', lambda p: None)
    monkeypatch.setattr(installer, 'check_distribution', lambda p: None)
    monkeypatch.setattr(installer, 'register', lambda p: None)
    monkeypatch.setattr(installer, 'archive_rollback', lambda *a: a[1].write_bytes(b'rollback'))

    def validated(stage, root, output):
        output.mkdir()
        return {'ok': True}

    def fail_rollback(stage, target):
        raise OSError('rollback publication failed')

    monkeypatch.setattr(installer, 'isolated_smoke', validated)
    monkeypatch.setattr(installer.os, 'replace', fail_rollback)
    with pytest.raises(OSError, match='rollback publication'):
        installer.install(source, current, tmp_path / 'receipt')
    assert (current / 'build.json').read_text() == 'previous software'
    assert not (current / 'new').exists()


def test_missing_isolation_and_changed_installation_stop_cleanup(tmp_path, monkeypatch):
    directory = tmp_path / 'Applications/KmzSuite'
    directory.mkdir(parents=True)
    monkeypatch.setattr(cleanup, 'ROOT', tmp_path / 'repo')
    monkeypatch.setattr(cleanup.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(cleanup, 'validate_bundle', lambda p: None)
    receipt = {'ok': True, 'bundle_sha256': {'app': 'before'}}
    (directory / '.kmz-install.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='isolation'):
        cleanup.check_install(directory)
    receipt['checkout_resources_unavailable'] = ['.venv', 'build', '.cache', '.local', 'Sources', 'Apps', 'Tools']
    (directory / '.kmz-install.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(cleanup, 'bundle_hashes', lambda p: {'app': 'after'})
    with pytest.raises(ValueError, match='changed'):
        cleanup.check_install(directory)


def test_additional_user_files_stop_installation_before_mutation(tmp_path, monkeypatch):
    current = tmp_path / 'Applications/KmzSuite'
    current.mkdir(parents=True)
    (current / '.kmz-install.json').write_text('{}')
    (current / 'personal.mzML').write_bytes(b'original')
    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setattr(installer, 'ROOT', tmp_path / 'checkout')
    monkeypatch.setattr(installer.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(installer, 'require_stopped', lambda p: None)
    with pytest.raises(RuntimeError, match='user files'):
        installer.install(source, current, tmp_path / 'receipt')
    assert (current / 'personal.mzML').read_bytes() == b'original'


def test_isolation_preserves_finder_residue_and_records_absent_resources(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'Sources').mkdir()
    distribution = tmp_path / 'installed'
    distribution.mkdir()

    def passed(*args):
        hold = next(tmp_path.glob('.KmzSuite-development-hold-*'))
        (hold / '.DS_Store').write_bytes(b'Finder metadata')
        return {'ok': True}

    monkeypatch.setattr(installed, 'smoke', passed)
    receipt = installed.isolated_smoke(distribution, root, tmp_path / 'evidence')
    assert (root / 'Sources').exists()
    assert '.venv' in receipt['checkout_resources_unavailable']
    assert receipt['checkout_resources_temporarily_moved'] == ['Sources']
    assert (tmp_path / 'evidence/restore-metadata/.DS_Store').read_bytes() == b'Finder metadata'


def test_successful_install_replaces_one_rollback_and_leaves_no_old_app_tree(tmp_path, monkeypatch):
    current = tmp_path / 'Applications/KmzSuite'
    current.mkdir(parents=True)
    (current / '.kmz-install.json').write_text('{"ok":true}')
    (current / 'build.json').write_text('previous software')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'build.json').write_text('replacement software')
    rollback = current.parent / '.KmzSuite-rollback.zip'
    with ZipFile(rollback, 'w') as z:
        z.writestr('KmzSuite/.kmz-install.json', '{"ok":true}')
        z.writestr('KmzSuite/build.json', 'obsolete rollback')
    monkeypatch.setattr(installer, 'ROOT', tmp_path / 'checkout')
    monkeypatch.setattr(installer.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(installer, 'require_stopped', lambda p: None)
    monkeypatch.setattr(installer, 'validate_bundle', lambda p: None)
    monkeypatch.setattr(installer, 'check_distribution', lambda p: None)
    monkeypatch.setattr(installer, 'register', lambda p: None)

    def archive(directory, output, verification):
        with ZipFile(output, 'w') as z:
            z.writestr('KmzSuite/.kmz-install.json', '{"ok":true}')
            z.writestr('KmzSuite/build.json', (directory / 'build.json').read_text())

    def validated(stage, root, output):
        output.mkdir()
        return {'ok': True}

    monkeypatch.setattr(installer, 'archive_rollback', archive)
    monkeypatch.setattr(installer, 'isolated_smoke', validated)
    installer.install(source, current, tmp_path / 'receipt')
    assert (current / 'build.json').read_text() == 'replacement software'
    with ZipFile(rollback) as z:
        assert z.read('KmzSuite/build.json') == b'previous software'
    assert sorted(p.name for p in current.parent.iterdir()) == ['.KmzSuite-rollback.zip', 'KmzSuite']
