"""Reject obsolete conversion metadata before either viewer displays arrays."""
import pytest

from kmzcore.cancel import Cancelled
from kmzcore.mzml import LEGACY_INPUT_MESSAGE, LegacyInputError, require_current_input
from kmzcore.sparse_profile import read_provenance
from kmzviewer.mzml_index import index_ms1
from kmz2viewer.index import index_ms2
from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def fixture(path, method="", *, namespace=False, note=""):
    write_mzml(path, [dict(ms_level=n, mz=[100., 200.], intensity=[10., 20.],
                         rt=1., tic=30., centroid=True) for n in (1, 2)])
    text = path.read_text().replace('<run id="run">',
        '<fileDescription><fileContent>' + note + '</fileContent></fileDescription>'
        '<dataProcessingList count="1"><dataProcessing id="conversion">'
        '<processingMethod order="0" softwareRef="converter">' + method +
        '</processingMethod></dataProcessing></dataProcessingList><run id="run">')
    if namespace:
        # The shared gate supports namespaced processing tags independently of
        # the existing spectrum scanners' supported unprefixed array syntax.
        text = text.replace('<dataProcessingList ', '<dataProcessingList xmlns:p="http://psi.hupo.org/ms/mzml" ')
        for tag in ('dataProcessing', 'processingMethod', 'userParam'):
            text = text.replace('<' + tag + ' ', '<p:' + tag + ' ').replace('</' + tag + '>', '</p:' + tag + '>')
    path.write_text(text)
    return path


@pytest.mark.parametrize('field', ['RetiredConvert decoding limitations',
                                  'RetiredSuite sparse profile version',
                                  'RetiredSuite mass correction version'])
@pytest.mark.parametrize('namespace', [False, True])
def test_both_indexers_reject_obsolete_contract(tmp_path, field, namespace):
    path = fixture(tmp_path/'renamed.mzML', f'<userParam name="{field}" value="1"/>', namespace=namespace)
    before = path.read_bytes()
    for read in (index_ms1, index_ms2, read_provenance):
        with pytest.raises(LegacyInputError) as error:
            read(path)
        assert str(error.value) == LEGACY_INPUT_MESSAGE
    assert path.read_bytes() == before


@pytest.mark.parametrize('method', ['', '<userParam name="Vendor software" value="1"/>',
    '<userParam name="KmzConvert decoding limitations" value="documented"/>',
    '<userParam name="KmzSuite sparse profile version" value="1"/>',
    '<userParam name="KmzSuite mass correction version" value="1"/>'])
def test_current_and_ordinary_mzml_remain_readable(tmp_path, method):
    path = fixture(tmp_path/'RetiredSuite.mzML', method,
                   note='<userParam name="note" value="RetiredSuite sparse profile version"/>')
    assert len(index_ms1(path)) == 1
    assert len(index_ms2(path)) == 1


def test_gate_stops_before_binary_data_and_is_cancellable(tmp_path):
    path = tmp_path/'headers.mzML'
    path.write_bytes(b'<mzML><run><spectrumList count="1">' + b'not XML or arrays')
    require_current_input(path)
    with pytest.raises(Cancelled):
        require_current_input(path, is_cancelled=lambda: True)


def test_large_metadata_does_not_bypass_gate(tmp_path):
    path = fixture(tmp_path/'large.mzML', '<userParam name="RetiredConvert decoding limitations" value="1"/>')
    path.write_text(path.read_text().replace('<dataProcessingList', '<!--' + 'x'*(5*1024*1024) + '--><dataProcessingList'))
    with pytest.raises(LegacyInputError):
        read_provenance(path)


def test_both_viewers_present_the_reconversion_message(tmp_path, qapp):
    from kmzviewer.gui.app import MainWindow as Ms1Window
    from kmz2viewer.gui.app import MainWindow as Ms2Window
    path = fixture(tmp_path/'legacy.mzML', '<userParam name="RetiredConvert decoding limitations" value="1"/>')
    for window_type in (Ms1Window, Ms2Window):
        window = window_type()
        try:
            assert window.load_paths_sync([str(path)]) == [LEGACY_INPUT_MESSAGE]
        finally:
            window.close()
            qapp.processEvents()
