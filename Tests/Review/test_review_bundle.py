"""Exercise evidence inclusion and raw/state exclusions in a temporary repository."""
import hashlib
import json
import random
import subprocess
import sys
from zipfile import ZipFile

import pytest

from scripts import review_bundle


@pytest.fixture
def repository(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('.local/\nreview-bundle/\n')
    (tmp_path / 'README.md').write_text('Baseline source\n')
    (tmp_path / 'unrelated.txt').write_text('Unchanged unrelated source\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Bundle test', '-c',
                    'user.email=bundle-test@example.invalid', 'commit', '-qm', 'Fixture'],
                   cwd=tmp_path, check=True)
    (tmp_path / '.local/results').mkdir(parents=True)
    monkeypatch.setattr(review_bundle, 'ROOT', tmp_path)
    return tmp_path


def run_bundle(monkeypatch, *arguments):
    monkeypatch.setattr(sys, 'argv', ['review_bundle.py', '--label', 'test', *arguments])
    review_bundle.main()


def test_selected_ignored_results_are_complete_and_hashed(repository, monkeypatch):
    (repository / 'README.md').write_text('Reviewed source change\n')
    (repository / 'new.py').write_text('# Explicit untracked implementation\n')
    (repository / '.local/results/REPORT.md').write_text('Completed result\n')
    (repository / '.local/results/summary.csv').write_text('condition,error\nA,1\nD,0\n')
    # A neighboring raw input must not be swept into the archive.
    (repository / '.local/results/private.wiff').write_bytes(b'Excluded raw fixture')
    run_bundle(monkeypatch, '--source-subset', '--include', 'README.md', '--include', 'new.py',
               '--evidence-root', '.local/results', '--evidence', 'REPORT.md',
               '--evidence', 'summary.csv')
    with ZipFile(next((repository / 'review-bundle').glob('*.zip'))) as archive:
        names = set(archive.namelist())
        assert names == {'README.md', 'new.py', 'results/REPORT.md', 'results/summary.csv',
                         'MANIFEST.json', 'REVIEW_START_HERE.md', 'GIT_STATUS.txt', 'SOURCE_DIFF.patch'}
        assert archive.testzip() is None
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert manifest['branch'] == 'main'
        assert manifest['working_tree_dirty']
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert len(archive.read(name)) == manifest['file_details'][name]['bytes']
        assert b'new.py' in archive.read('GIT_STATUS.txt')
        assert b'Reviewed source change' in archive.read('SOURCE_DIFF.patch')
        assert manifest['file_details']['results/summary.csv']['kind'] == 'result evidence'


@pytest.mark.parametrize('name', ['raw.wiff', 'raw.wiff.scan', 'raw.WIFF', 'raw.WIFF.SCAN', 'raw.mzML', 'arrays.npz', 'secrets.json',
                                 'cache/__pycache__/output.json', '../outside.json', 'run.D/AcqData/Devices.json'])
def test_evidence_rejects_raw_state_and_traversal(repository, monkeypatch, name):
    path = repository / '.local/results' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('Never package\n')
    with pytest.raises(SystemExit) as error:
        run_bundle(monkeypatch, '--source-subset', '--evidence-root', '.local/results',
                   '--evidence', name)
    assert error.value.code == 2
    assert not (repository / 'review-bundle').exists()


def test_evidence_rejects_symlink_parent(repository, monkeypatch):
    (repository / '.local/actual').mkdir()
    (repository / '.local/actual/REPORT.md').write_text('Result\n')
    (repository / '.local/results/link').symlink_to('../actual', target_is_directory=True)
    with pytest.raises(SystemExit):
        run_bundle(monkeypatch, '--source-subset', '--evidence-root', '.local/results',
                   '--evidence', 'link/REPORT.md')


def test_ignored_source_requires_explicit_evidence(repository, monkeypatch):
    (repository / '.local/results/REPORT.md').write_text('Result\n')
    with pytest.raises(SystemExit):
        run_bundle(monkeypatch, '--include', '.local/results/REPORT.md')


@pytest.mark.parametrize('selection', ['source', 'evidence'])
def test_from_chase_directory_is_never_bundled(repository, monkeypatch, selection):
    (repository / 'FromChase').mkdir()
    (repository / 'FromChase/private.json').write_text('Private scientific metadata\n')
    arguments = ('--include', 'FromChase/private.json') if selection == 'source' else (
        '--evidence-root', 'FromChase', '--evidence', 'private.json')
    with pytest.raises(SystemExit):
        run_bundle(monkeypatch, *arguments)
    assert not (repository / 'review-bundle').exists()


def test_default_compact_source_selection(repository, monkeypatch):
    run_bundle(monkeypatch, '--include', 'README.md')
    with ZipFile(next((repository / 'review-bundle').glob('*.zip'))) as archive:
        assert 'README.md' in archive.namelist()
        assert 'unrelated.txt' not in archive.namelist()
        assert not any(name.startswith('results/') for name in archive.namelist())


def test_explicit_tracked_selection(repository, monkeypatch):
    run_bundle(monkeypatch, '--tracked-source')
    with ZipFile(next((repository / 'review-bundle').glob('*.zip'))) as archive:
        assert 'unrelated.txt' in archive.namelist()


def test_evidence_requires_root_and_selection(repository, monkeypatch):
    with pytest.raises(SystemExit):
        run_bundle(monkeypatch, '--evidence-root', '.local/results')


def incompressible_evidence(repository):
    # Deterministic generated bytes just over the budget; not a real dataset.
    (repository / '.local/results/noise.txt').write_bytes(
        random.Random(123).randbytes(review_bundle.BUDGET_BYTES + 4096))
    return ('--evidence-root', '.local/results', '--evidence', 'noise.txt')


def test_actual_compressed_size_failure_removes_zip_and_ranks_contributors(repository, monkeypatch, capsys):
    args = incompressible_evidence(repository)
    with pytest.raises(SystemExit) as error:
        run_bundle(monkeypatch, *args)
    assert error.value.code == 2
    assert not list((repository / 'review-bundle').glob('*.zip'))
    diagnostic = capsys.readouterr().err
    assert 'MiB compressed' in diagnostic
    assert 'results/noise.txt' in diagnostic
    assert diagnostic.index('results/noise.txt') < diagnostic.index('MANIFEST.json')
    assert '--allow-large --large-reason' in diagnostic
    assert 'Oversized archive removed' in diagnostic


def test_large_uncompressed_but_compact_compressed_evidence_succeeds(repository, monkeypatch):
    (repository / '.local/results/repeated.txt').write_bytes(b'x' * (2 * review_bundle.BUDGET_BYTES))
    run_bundle(monkeypatch, '--evidence-root', '.local/results', '--evidence', 'repeated.txt')
    path = next((repository / 'review-bundle').glob('*.zip'))
    assert path.stat().st_size < review_bundle.BUDGET_BYTES
    with ZipFile(path) as archive:
        item = archive.getinfo('results/repeated.txt')
        assert item.file_size > review_bundle.BUDGET_BYTES
        assert item.compress_size < review_bundle.BUDGET_BYTES


@pytest.mark.parametrize('arguments', [('--allow-large',), ('--allow-large', '--large-reason', '  '),
                                     ('--large-reason', 'Reason without override')])
def test_large_override_requires_both_flag_and_nonempty_reason(repository, monkeypatch, arguments):
    with pytest.raises(SystemExit) as error:
        run_bundle(monkeypatch, *arguments)
    assert error.value.code == 2
    assert not (repository / 'review-bundle').exists()


def test_large_override_records_reason_and_actual_large_archive(repository, monkeypatch):
    reason = 'Controlled test of the compressed-size override'
    run_bundle(monkeypatch, *incompressible_evidence(repository), '--allow-large', '--large-reason', reason)
    path = next((repository / 'review-bundle').glob('*.zip'))
    assert path.stat().st_size > review_bundle.BUDGET_BYTES
    with ZipFile(path) as archive:
        assert archive.getinfo('results/noise.txt').compress_size > review_bundle.BUDGET_BYTES
        assert archive.testzip() is None
        policy = json.loads(archive.read('MANIFEST.json'))['size_policy']
        assert policy['allow_large'] and policy['large_reason'] == reason
        assert reason.encode() in archive.read('REVIEW_START_HERE.md')


def test_override_does_not_bypass_raw_exclusion(repository, monkeypatch):
    (repository / '.local/results/raw.wiff.scan').write_bytes(b'Excluded raw fixture')
    with pytest.raises(SystemExit):
        run_bundle(monkeypatch, '--evidence-root', '.local/results', '--evidence', 'raw.wiff.scan',
                   '--allow-large', '--large-reason', 'Privacy exclusions still apply')
