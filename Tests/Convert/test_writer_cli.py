"""Synthetic format and failure-path tests, independent of private acquisitions."""
from pathlib import Path
from types import SimpleNamespace
import errno
import os
import sys
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kmzconvert.cli import main
from kmzconvert_core.reader import Experiment, Scan
from kmzconvert_core.reference import iter_records, NS
from kmzconvert_core.writer import write_mzml
from kmzconvert_core.validate import compare

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/validation'))
from check_mzml import check_index, check


@pytest.fixture
def unsupported_link(monkeypatch):
    def unsupported(*args):
        raise OSError(errno.ENOTSUP, 'Operation not supported')
    monkeypatch.setattr('kmzconvert_core.writer.os.link', unsupported)


def test_atomic_publication_uses_link_without_copy(synthetic, tmp_path, monkeypatch):
    real_link = os.link
    calls = []
    def link(source, destination):
        assert check_index(source)['checksum_valid']
        calls.append((source, destination))
        real_link(source, destination)
    def unexpected_copy(*args, **kwargs):
        pytest.fail('Atomic path must not copy')
    monkeypatch.setattr('kmzconvert_core.writer.os.link', link)
    monkeypatch.setattr('kmzconvert_core.writer.shutil.copyfileobj', unexpected_copy)
    output = write_mzml(synthetic, tmp_path/'atomic.mzML')
    assert len(calls) == 1 and calls[0][1] == output
    assert check_index(output)['checksum_valid']
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


@pytest.mark.parametrize('code', sorted({errno.ENOTSUP, errno.EOPNOTSUPP}))
def test_unsupported_link_copies_identical_completed_file(synthetic, tmp_path, monkeypatch, code):
    def unsupported(*args):
        raise OSError(code, 'Operation not supported')
    monkeypatch.setattr('kmzconvert_core.writer.os.link', unsupported)
    expected = write_mzml(synthetic, tmp_path/'expected.mzML').read_bytes()
    actual = write_mzml(synthetic, tmp_path/'copied.mzML')
    assert actual.read_bytes() == expected
    assert check_index(actual)['checksum_valid']
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


@pytest.mark.parametrize('symlink', [False, True])
def test_fallback_exclusive_create_rejects_racing_destination(synthetic, tmp_path, monkeypatch, symlink):
    output = tmp_path/'raced.mzML'
    untouched = tmp_path/'untouched'
    untouched.write_bytes(b'concurrent data')
    def race(source, destination):
        if symlink:
            destination.symlink_to(untouched)
        else:
            destination.write_bytes(b'concurrent data')
        raise OSError(errno.ENOTSUP, 'Operation not supported')
    monkeypatch.setattr('kmzconvert_core.writer.os.link', race)
    with pytest.raises(FileExistsError):
        write_mzml(synthetic, output)
    assert output.is_symlink() == symlink
    assert output.read_bytes() == untouched.read_bytes() == b'concurrent data'
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


def test_existing_destination_is_refused_before_conversion(synthetic, tmp_path, unsupported_link):
    output = tmp_path/'existing.mzML'
    output.write_bytes(b'user data')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        write_mzml(synthetic, output)
    assert output.read_bytes() == b'user data'
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


@pytest.mark.parametrize('failure', ['copy', 'short', 'corrupt', 'fsync', 'readback'])
def test_fallback_failures_remove_partial_output_and_retain_completed_temp(
        synthetic, tmp_path, monkeypatch, unsupported_link, failure):
    output = tmp_path/'failed.mzML'
    if failure in {'copy', 'short', 'corrupt'}:
        def bad_copy(source, target, **kwargs):
            data = source.read()
            if failure == 'copy':
                target.write(data[:20])
                raise OSError(errno.ENOSPC, 'No space left on device')
            target.write(data[:20] if failure == 'short' else b'X'*len(data))
        monkeypatch.setattr('kmzconvert_core.writer.shutil.copyfileobj', bad_copy)
    elif failure == 'fsync':
        real_fsync = os.fsync
        calls = []
        def bad_fsync(fd):
            calls.append(fd)
            if len(calls) == 2:
                raise OSError(errno.EIO, 'I/O failure')
            real_fsync(fd)
        monkeypatch.setattr('kmzconvert_core.writer.os.fsync', bad_fsync)
    else:
        real_open = Path.open
        def bad_open(path, mode='r', *args, **kwargs):
            if path == output and mode == 'rb':
                raise OSError(errno.EIO, 'Readback failure')
            return real_open(path, mode, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', bad_open)
    with pytest.raises(OSError) as caught:
        write_mzml(synthetic, output)
    assert not output.exists()
    retained, = tmp_path.glob('.kmzconvert-*.tmp')
    assert check_index(retained)['checksum_valid']
    assert any(str(retained) in note for note in caught.value.__notes__)


def test_unsupported_final_fsync_is_allowed(synthetic, tmp_path, monkeypatch, unsupported_link):
    real_fsync = os.fsync
    calls = []
    def fsync(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(errno.ENOTSUP, 'Operation not supported')
        real_fsync(fd)
    monkeypatch.setattr('kmzconvert_core.writer.os.fsync', fsync)
    output = write_mzml(synthetic, tmp_path/'synced.mzML')
    assert len(calls) == 2 and check_index(output)['checksum_valid']
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


@pytest.mark.parametrize('code', [errno.EACCES, errno.EIO, errno.ENOENT, errno.EXDEV, errno.EINVAL])
def test_unrelated_publication_errors_surface_without_copy(synthetic, tmp_path, monkeypatch, code):
    error = OSError(code, 'Simulated publication error')
    def fail(*args):
        raise error
    def unexpected_copy(*args, **kwargs):
        pytest.fail('Unrelated errors must not trigger copying')
    monkeypatch.setattr('kmzconvert_core.writer.os.link', fail)
    monkeypatch.setattr('kmzconvert_core.writer.shutil.copyfileobj', unexpected_copy)
    with pytest.raises(OSError) as caught:
        write_mzml(synthetic, tmp_path/'failed.mzML')
    assert caught.value is error and caught.value.errno == code
    retained, = tmp_path.glob('.kmzconvert-*.tmp')
    assert check_index(retained)['checksum_valid']
    assert any(str(retained) in note for note in error.__notes__)


def test_conversion_failure_removes_unfinished_temp(synthetic, tmp_path):
    def fail(scan):
        raise ValueError('Decoding failed')
    synthetic.arrays = fail
    with pytest.raises(ValueError, match='Decoding failed'):
        write_mzml(synthetic, tmp_path/'failed.mzML')
    assert not (tmp_path/'failed.mzML').exists()
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


def test_cli_identifies_retained_completed_temp(synthetic, tmp_path, monkeypatch, capsys):
    synthetic.nominal_window_overhangs = {}
    synthetic.calibration_resolutions = []
    synthetic.close = lambda: None
    monkeypatch.setattr('kmzconvert.cli.WiffReader', lambda _: synthetic)
    def fail(*args):
        raise PermissionError(errno.EACCES, 'Permission denied')
    monkeypatch.setattr('kmzconvert_core.writer.os.link', fail)
    assert main(['source.wiff', '-o', str(tmp_path/'failed.mzML')]) == 1
    retained, = tmp_path.glob('.kmzconvert-*.tmp')
    stderr = capsys.readouterr().err
    assert 'Permission denied' in stderr and str(retained) in stderr
    assert 'Completed temporary output retained' in stderr and 'Traceback' not in stderr


def test_failed_partial_cleanup_is_reported(synthetic, tmp_path, monkeypatch, unsupported_link):
    output = tmp_path/'failed.mzML'
    real_unlink = Path.unlink
    def fail_copy(source, target, **kwargs):
        target.write(b'partial')
        raise OSError(errno.ENOSPC, 'No space left on device')
    def fail_unlink(path, *args, **kwargs):
        if path == output:
            raise PermissionError(errno.EACCES, 'Cannot remove partial output')
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr('kmzconvert_core.writer.shutil.copyfileobj', fail_copy)
    monkeypatch.setattr(Path, 'unlink', fail_unlink)
    with pytest.raises(OSError) as caught:
        write_mzml(synthetic, output)
    assert caught.value.errno == errno.ENOSPC
    assert any('Incomplete output could not be removed' in n and str(output) in n
               for n in caught.value.__notes__)
    retained, = tmp_path.glob('.kmzconvert-*.tmp')
    assert check_index(retained)['checksum_valid']


@pytest.fixture
def synthetic(tmp_path):
    paths = [tmp_path/('synthetic-\u00e9'+suffix) for suffix in ('.wiff2', '.wiff', '.wiff.scan')]
    for p in paths:
        p.write_bytes(b'synthetic, not SCIEX data')
    scans = [Scan(0, 1, 1, 1., 10., 10., 0, 0, .1, .2, None, None),
             Scan(1, 1, 2, 1.1, 10., 10., 0, 0, .1, .2, 100., 1000.)]
    return SimpleNamespace(entry=paths[0], wiff_path=paths[1], scan_path=paths[2],
                           acquisition_software='SCIEX OS 2.2.0.0', instrument={'Mass Spectrometer': 'X500 QTOF'},
                           scans=scans, experiments=[Experiment(1, 'positive', 100., 1600., 200., 5., {}),
                                                    Experiment(2, 'positive', 50., 1000., 50., 30., {})],
                           arrays=lambda _: (np.array([100., 100.001, 100.002]), np.array([0., 10., 0.])),
                           tic_arrays=lambda: (np.array([1./60, 1.1/60]), np.array([10., 10.]), np.array([1, 2])))


def test_index_offsets_checksum_unicode_and_decoded_arrays(synthetic, tmp_path):
    output = write_mzml(synthetic, tmp_path/'output.mzML')
    assert check_index(output) == {'indexed': True, 'offsets_verified': {'spectrum': 2, 'chromatogram': 1}, 'checksum_valid': True}
    records = list(iter_records(output))
    assert len(records) == 2
    for r in records:
        np.testing.assert_array_equal(r['arrays']['intensity'], [0., 10., 0.])
    root = ET.parse(output).getroot()
    assert root.tag == NS+'indexedmzML'
    sources = root.findall('.//'+NS+'sourceFile')
    assert len(sources) == 3 and sources[0].find(NS+'cvParam[@accession="MS:1003448"]') is not None
    assert records[1]['cv']['MS:1000044']['value'] == ''
    assert 'MS:1000422' not in records[1]['cv'] and 'MS:1000041' not in records[1]['cv']
    assert 'unitAccession' not in records[0]['cv']['MS:1000285']


def test_wiff_entry_does_not_declare_nonexistent_wiff2(synthetic, tmp_path):
    synthetic.entry = synthetic.wiff_path
    output = write_mzml(synthetic, tmp_path/'two.mzML')
    sources = ET.parse(output).findall('.//'+NS+'sourceFile')
    assert [s.get('id') for s in sources] == ['WIFF', 'WIFFSCAN']


def test_official_schema_and_cv_when_cached(synthetic, tmp_path):
    specs = Path(__file__).resolve().parents[2]/'.cache/standards'
    if not (specs/'mzML1.1.0_idx.xsd').exists() or not (specs/'psi-ms.obo').exists():
        pytest.skip('Optional official schemas/CV not cached in .cache/standards/')
    output = write_mzml(synthetic, tmp_path/'schema.mzML')
    result = check(output, specs/'mzML1.1.0_idx.xsd', specs/'psi-ms.obo')
    assert result['xsd_valid'] and result['psi_ms_names_valid']


def test_native_openms_index_and_semantics_when_installed(synthetic, tmp_path):
    python = Path(__file__).resolve().parents[2]/'.cache/semantic-env/bin/python'
    if not python.exists():
        pytest.skip('Optional local pyOpenMS validator not installed')
    output = write_mzml(synthetic, tmp_path/'openms.mzML')
    program = '''
import pyopenms, sys
valid, errors, warnings = pyopenms.MzMLFile().isSemanticallyValid(sys.argv[1])
assert valid and not errors and not warnings, (errors, warnings)
reader = pyopenms.OnDiscMSExperiment()
assert reader.openFile(sys.argv[1])
assert reader.getNrSpectra() == 2 and reader.getNrChromatograms() == 1
assert reader.getSpectrum(0).getMSLevel() == 1
assert reader.getSpectrum(1).getMSLevel() == 2
assert reader.getSpectrum(0).getNativeID().endswith('experiment=1')
assert reader.getChromatogram(0).size() == 2
'''
    result = subprocess.run([str(python), '-c', program, str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr


def test_index_checker_detects_wrong_offsets_and_checksum(synthetic, tmp_path):
    output = write_mzml(synthetic, tmp_path/'output.mzML')
    original = output.read_bytes()
    output.write_bytes(original.replace(b'idRef="TIC"', b'idRef="BAD"'))
    with pytest.raises(AssertionError, match='ID/offset'):
        check_index(output)
    output.write_bytes(original.replace(b'id="KmzConvert"', b'id="BadConvert"'))
    with pytest.raises(AssertionError, match='checksum'):
        check_index(output)


def test_oracle_cli_rejects_chromatogram_only_regression(synthetic, tmp_path):
    reference = write_mzml(synthetic, tmp_path/'reference.mzML')
    synthetic.tic_arrays = lambda: (np.array([1./60, 1.1/60]), np.array([11., 10.]), np.array([1, 2]))
    actual = write_mzml(synthetic, tmp_path/'changed.mzML')
    result = compare(reference, actual)
    assert result['measured_spectra_pass'] and not result['chromatograms_pass']
    checked = subprocess.run([sys.executable, '-m', 'kmzconvert_core.validate', str(reference), str(actual), str(tmp_path/'report.json')], capture_output=True, text=True)
    assert checked.returncode == 1


@pytest.mark.parametrize('mass_error, intensity', [(0., 10.), (1e-8, 10.), (0., 11.)])
def test_signal_comparison_keeps_framing_and_measurement_failures_separate(synthetic, tmp_path, mass_error, intensity):
    reference = write_mzml(synthetic, tmp_path/'reference.mzML')
    synthetic.arrays = lambda _: (np.array([100.001+mass_error]), np.array([intensity]))
    actual = write_mzml(synthetic, tmp_path/'without-endpoint-zeros.mzML')
    result = compare(reference, actual)
    assert result['length_mismatches'] == 2 and not result['measured_spectra_pass']
    assert not result['strict_all_mz_points_pass']
    assert result['measured_signal']['passed'] == (mass_error == 0 and intensity == 10)
    assert result['measured_signal']['reference_points'] == result['measured_signal']['actual_points'] == 2


def test_signal_result_does_not_hide_metadata_disagreement(synthetic, tmp_path):
    synthetic.experiments[1].collision_energy = None
    reference = write_mzml(synthetic, tmp_path/'reference.mzML')
    synthetic.experiments[1].collision_energy = 30.
    actual = write_mzml(synthetic, tmp_path/'actual.mzML')
    result = compare(reference, actual)
    assert result['measured_signal']['passed']
    assert not result['measured_spectra_pass'] and not result['complete_metadata_equivalence']
    assert result['metadata_mismatches']['collision_energy'] == 1


def test_reader_closes_mapping_on_constructor_failure(tmp_path, monkeypatch):
    import mmap
    from kmzconvert_core.reader import WiffReader
    from kmzconvert_core.binary import FormatError
    maps, constructor = [], mmap.mmap
    def tracked(*args, **kwargs):
        mapping = constructor(*args, **kwargs)
        maps.append(mapping)
        return mapping
    monkeypatch.setattr('kmzconvert_core.reader.mmap.mmap', tracked)
    path = tmp_path/'invalid.wiff'
    path.write_bytes(b'synthetic')
    path.with_suffix('.wiff.scan').write_bytes(b'\0'*44)
    with pytest.raises(FormatError, match='version'):
        WiffReader(path)
    assert len(maps) == 1 and maps[0].closed


@pytest.mark.parametrize('args, message', [
    (['missing.wiff'], 'Missing entry point'),
    (['wrong.txt'], 'Input must be'),
    (['missing.wiff2'], 'Missing entry point'),
])
def test_cli_missing_or_wrong_inputs(args, message, capsys):
    assert main(args) == 1
    stderr = capsys.readouterr().err
    assert message in stderr and 'Traceback' not in stderr


def test_missing_sidecar_and_truncated_input(tmp_path, capsys):
    wiff = tmp_path/'input.wiff'
    wiff.write_bytes(b'synthetic')
    assert main([str(wiff)]) == 1
    assert 'sidecar missing' in capsys.readouterr().err
    wiff.with_suffix('.wiff.scan').write_bytes(b'short')
    assert main([str(wiff)]) == 1
    assert 'Truncated scan file' in capsys.readouterr().err


def test_output_paths_and_racing_destination(synthetic, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='extension'):
        write_mzml(synthetic, tmp_path/'wrong.txt')
    with pytest.raises(FileNotFoundError, match='directory'):
        write_mzml(synthetic, tmp_path/'missing'/'output.mzML')
    dangling = tmp_path/'symlink.mzML'
    dangling.symlink_to(tmp_path/'absent.mzML')
    with pytest.raises(FileExistsError, match='symlink'):
        write_mzml(synthetic, dangling)
    assert not (tmp_path/'absent.mzML').exists()
    output = tmp_path/'raced.mzML'
    def concurrent_file(done, total):
        if not output.exists():
            output.write_bytes(b'concurrent user data')
    with pytest.raises(FileExistsError):
        write_mzml(synthetic, output, concurrent_file)
    assert output.read_bytes() == b'concurrent user data'
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


def test_default_cli_output_is_in_working_directory(synthetic, tmp_path, monkeypatch):
    synthetic.nominal_window_overhangs = {}
    synthetic.calibration_resolutions = []
    synthetic.close = lambda: None
    monkeypatch.setattr('kmzconvert.cli.WiffReader', lambda _: synthetic)
    monkeypatch.chdir(tmp_path)
    assert main(['somewhere/example.wiff2']) == 0
    assert (tmp_path/'example.mzML').is_file()
    assert main(['somewhere/example.wiff2']) == 1


def test_cli_reports_startup_calibration_selection(synthetic, tmp_path, monkeypatch, capsys):
    synthetic.nominal_window_overhangs = {}
    synthetic.calibration_resolutions = [{'native_id': synthetic.scans[0].native_id}]
    synthetic.close = lambda: None
    monkeypatch.setattr('kmzconvert.cli.WiffReader', lambda _: synthetic)
    assert main(['source.wiff', '-o', str(tmp_path/'reported.mzML')]) == 0
    assert 'WIFF slot table for 1 startup spectra' in capsys.readouterr().err
