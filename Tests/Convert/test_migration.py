from pathlib import Path
from types import SimpleNamespace
import shutil

import pytest
from kmzconvert.cli import main
from kmzconvert_core.writer import write_mzml
from kmzcore.runtime import IDENTITIES, resource_path


def test_committed_fixture_and_fresh_runtime_resources():
    root = Path(__file__).resolve().parents[2]
    assert (root / 'TestData/Regression/centroid.mzML').is_file()
    assert {item.name for item in IDENTITIES.values()} == {'KmzViewer', 'Kmz2Viewer', 'KmzCoMet'}
    assert len({item.bundle_id for item in IDENTITIES.values()}) == 3
    for item in IDENTITIES.values():
        assert resource_path(item.manual).is_file()


def test_cli_interrupt_closes_reader_and_reports_without_traceback(monkeypatch, capsys, tmp_path):
    closed = []
    reader = SimpleNamespace(calibration_resolutions=[], close=lambda: closed.append(True))
    monkeypatch.setattr('kmzconvert.cli.WiffReader', lambda _: reader)
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr('kmzconvert.cli.write_mzml', interrupt)
    assert main(['sample.wiff', '-o', str(tmp_path / 'output.mzML')]) == 130
    assert closed == [True]
    stderr = capsys.readouterr().err
    assert 'cancelled' in stderr and 'Traceback' not in stderr


def test_writer_interrupt_during_progress_removes_unfinished_output(tmp_path):
    # Uses the existing synthetic writer fixture, including a real metadata path.
    from Tests.Convert.test_writer_cli import synthetic
    reader = synthetic.__wrapped__(tmp_path)
    def interrupt(done, total):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        write_mzml(reader, tmp_path / 'cancelled.mzML', interrupt)
    assert not (tmp_path / 'cancelled.mzML').exists()
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))


@pytest.mark.integration
def test_wiff2_discovery_checks_header_then_uses_only_matching_sidecars(tmp_path):
    from kmzconvert_core.reader import WiffReader
    from kmzconvert_core.binary import FormatError
    source = Path(__file__).resolve().parents[2] / 'TestData/Local/acquisition-01.wiff'
    if not source.is_file():
        pytest.skip('Optional local acquisition pair unavailable')
    for suffix in ('.wiff', '.wiff.scan'):
        shutil.copyfile(source.with_suffix(suffix), tmp_path / ('sample' + suffix))
    entry = tmp_path / 'sample.wiff2'
    # Discovery-header fixture only; no synthetic claim about opaque database data.
    entry.write_bytes(bytes(16) + bytes.fromhex('100001010c402020'))
    with WiffReader(entry) as discovered, WiffReader(tmp_path / 'sample.wiff') as direct:
        assert discovered.scans == direct.scans
        assert len(discovered.scans) > 0
    entry.write_bytes(bytes(24))
    with pytest.raises(FormatError, match='Unrecognized WIFF2 page header'):
        WiffReader(entry)
