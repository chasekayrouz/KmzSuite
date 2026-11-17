"""Focused experimental tests; synthetic arrays are tests, never benchmark evidence."""
from pathlib import Path
import sys
import numpy as np
import pytest

from scripts.benchmarks.ms1_algorithms import guard_indices, gap_connections, plotted_trace, shape_metrics, half_width
from scripts.benchmarks.ms1_mzml import filter_profile_mzml, write_raw_inspection, InspectionReader
from kmzconvert_core.reference import iter_records
from kmzconvert_core.reader import WiffReader
from kmzconvert_core import writer
from Tests.Viewer.mzml_factory import write_mzml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/validation"))
from check_mzml import check_index


@pytest.mark.parametrize("values,expected", [
    ([0, 100, 101, 100, 0], [1, 2, 3]),
    ([0, 2, 101, 120, 140, 2, 0], [1, 2, 3, 4, 5]),
    ([101, 2, 0, 0, 2, 101], [0, 1, 4, 5]),
    ([0, 101, 0, 101, 0], [0, 1, 2, 3, 4]),
    ([0, 1, 100], []), ([], []), ([101], [0]), ([101, 120], [0, 1]),
])
def test_exact_boundary_union(values, expected):
    y = np.array(values, dtype=float)
    original = y.copy()
    indices = guard_indices(y)
    assert indices.tolist() == expected
    np.testing.assert_array_equal(y, original)
    assert set(np.flatnonzero(y > 100)).issubset(indices)


def test_two_guards_not_recursive_and_no_synthetic_values():
    m = np.linspace(10, 11, 9)
    y = np.array([0, 0, 0, 0, 101, 0, 0, 0, 0], dtype=float)
    indices = guard_indices(y, guards=2)
    assert indices.tolist() == [2, 3, 4, 5, 6]
    assert list(zip(m[indices], y[indices])) == list(zip(m, y))[2:7]
    assert guard_indices(y, guards=0).tolist() == [4]


def test_dense_broad_and_gradually_variable_spacing_connect():
    for coordinates in (np.arange(40.), np.cumsum(np.linspace(1., 3., 40))):
        m = (10 + coordinates * .00005) ** 2
        original = m.copy()
        assert gap_connections(m).all()
        np.testing.assert_array_equal(m, original)


def test_deleted_region_breaks_deterministically():
    coords = np.r_[np.arange(20), np.arange(40, 60)]
    m = (10 + coords * .00005) ** 2
    expected = np.ones(39, dtype=bool)
    expected[19] = False
    np.testing.assert_array_equal(gap_connections(m), expected)
    np.testing.assert_array_equal(gap_connections(m), gap_connections(m))


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 20])
def test_uniformly_sparse_safe_fallback(count):
    m = np.arange(1., count + 1.) ** 2
    assert gap_connections(m).shape == (max(count - 1, 0),)
    assert gap_connections(m).all()


def test_render_blanks_are_not_measurements():
    m = np.array([100., 101., 105., 106.])
    y = np.array([110., 120., 130., 140.])
    grid = np.array([100., 100.5, 101., 102., 104., 105., 105.5, 106.])
    drawn = plotted_trace(grid, m, y, np.array([True, False, True]))
    assert np.isnan(drawn[3:5]).all()
    np.testing.assert_array_equal(drawn[[0, 2, 5, 7]], y)
    np.testing.assert_array_equal(y, [110., 120., 130., 140.])


def test_reference_metrics_zero_and_missing_fwhm_not_imputed():
    m = np.array([100., 100.001, 100.002, 100.003, 100.004])
    y = np.array([0., 50., 200., 50., 0.])
    metrics, _ = shape_metrics(m, y, np.arange(5), np.ones(4, dtype=bool))
    assert metrics["ink_iad"] == 0
    assert metrics["ink_nrmse_apex"] == 0
    assert metrics["drawn_mz_fraction"] == 1
    assert half_width(m[2:3], y[2:3], 0, np.array([], dtype=bool)) is None


def test_postconversion_metadata_ms2_and_index_preserved(tmp_path):
    source = tmp_path / "source.mzML"
    write_mzml(source, [
        {"mz": [100, 101, 102, 103, 104], "intensity": [0, 1, 101, 1, 0], "rt": 1, "tic": 103, "ms_level": 1},
        {"mz": [50, 51], "intensity": [5, 200], "rt": 1.1, "tic": 205, "ms_level": 2},
    ])
    source_bytes = source.read_bytes()
    destination = tmp_path / "guard.mzML"
    filter_profile_mzml(source, destination, 1)
    records = list(iter_records(destination))
    assert records[0]["arrays"]["mz"].tolist() == [101, 102, 103]
    assert records[0]["arrays"]["intensity"].tolist() == [1, 101, 1]
    assert records[1]["arrays"]["mz"].tolist() == [50, 51]
    assert records[1]["arrays"]["intensity"].tolist() == [5, 200]
    assert records[0]["rt_seconds"] == 60
    assert source.read_bytes() == source_bytes
    assert check_index(destination)["checksum_valid"]
    # Independent viewer decoder exercises indexed output, not our XML decoder.
    from kmzviewer.spectra import SpectrumReader
    with SpectrumReader(destination) as reader:
        m,y=reader.read(0)
        np.testing.assert_array_equal(m,[101,102,103])
        np.testing.assert_array_equal(y,[1,101,1])
    with pytest.raises(FileExistsError):
        filter_profile_mzml(source, destination, 0)


def test_real_raw_adapter_and_processing_patch_restored(tmp_path):
    path = Path(__file__).resolve().parents[2] / "TestData/Local/acquisition-01.wiff"
    if not path.exists():
        pytest.skip("Optional real complete WIFF fixture absent")
    original = writer.metadata
    with WiffReader(path) as reader:
        adapter = InspectionReader(reader, 1)
        scan = reader.scans[0]
        m, y = reader.arrays(scan)
        i = guard_indices(y)
        np.testing.assert_array_equal(adapter.arrays(scan)[0], m[i])
        np.testing.assert_array_equal(adapter.arrays(scan)[1], y[i])
        # Reduce only the test output's scan list; original input remains read-only.
        reader.scans = reader.scans[:2]
        out = tmp_path / "raw-guard.mzML"
        write_raw_inspection(reader, out, 1)
        assert "BENCHMARK ONLY" in out.read_text()
        assert check_index(out)["checksum_valid"]
    assert writer.metadata is original


@pytest.mark.parametrize("m,y", [([100, 99], [1, 2]), ([100], [1, 2]), ([100], [float("nan")])])
def test_invalid_input_rejected(m, y):
    from scripts.benchmarks.ms1_algorithms import validate_arrays
    with pytest.raises(ValueError):
        validate_arrays(m, y)


def test_postconversion_preserves_float32_precision_and_chromatograms(tmp_path):
    import base64
    import zlib
    import xml.etree.ElementTree as ET
    from kmzconvert_core.reference import NS
    source=tmp_path/"float32.mzML"
    write_mzml(source,[{"mz":[100.,101.,102.],"intensity":[.25,150.5,.75],"rt":1,"tic":151.5}],chromatogram=True)
    tree=ET.parse(source)
    for array in tree.findall('.//'+NS+'spectrum/'+NS+'binaryDataArrayList/'+NS+'binaryDataArray'):
        param=array.find(NS+'cvParam[@accession="MS:1000523"]')
        param.set('accession','MS:1000521'); param.set('name','32-bit float')
        binary=array.find(NS+'binary')
        values=np.frombuffer(zlib.decompress(base64.b64decode(binary.text)),dtype='<f8').astype('<f4')
        binary.text=base64.b64encode(zlib.compress(values.tobytes())).decode()
        array.set('encodedLength',str(len(binary.text)))
    tree.write(source,encoding='utf-8',xml_declaration=True)
    destination=tmp_path/'guard.mzML'
    filter_profile_mzml(source,destination,1)
    a,b=next(iter_records(source)),next(iter_records(destination))
    for kind in ('mz','intensity'):
        assert b['arrays'][kind].dtype==np.dtype('<f4')
        np.testing.assert_array_equal(a['arrays'][kind],b['arrays'][kind])
    from kmzviewer.spectra import SpectrumReader
    with SpectrumReader(destination) as reader:
        m,y=reader.read(0)
        np.testing.assert_array_equal(m,a['arrays']['mz'])
        np.testing.assert_array_equal(y,a['arrays']['intensity'])
    a,b=next(iter_records(source,'chromatogram')),next(iter_records(destination,'chromatogram'))
    for kind in a['arrays']: np.testing.assert_array_equal(a['arrays'][kind],b['arrays'][kind])
    assert check_index(destination)['checksum_valid']


def test_unqualified_auxiliary_array_removes_partial_output(tmp_path):
    import xml.etree.ElementTree as ET
    from kmzconvert_core.reference import NS
    source=tmp_path/'source.mzML'
    write_mzml(source,[{"mz":[100,101],"intensity":[101,1],"rt":1,"tic":102}])
    tree=ET.parse(source)
    array=tree.find('.//'+NS+'binaryDataArrayList/'+NS+'binaryDataArray')
    param=array.find(NS+'cvParam[@accession="MS:1000514"]')
    param.set('accession','MS:1000786')
    tree.write(source,encoding='utf-8',xml_declaration=True)
    output=tmp_path/'failed.mzML'
    with pytest.raises((KeyError,ValueError)): filter_profile_mzml(source,output,1)
    assert not output.exists()


def test_apices_and_relative_heights_preserved():
    m=np.array([100,100.001,100.002,101,101.001,101.002],dtype=float)
    y=np.array([0,200,0,0,400,0],dtype=float)
    idx=guard_indices(y)
    _,peaks=shape_metrics(m,y,idx,gap_connections(m[idx]))
    assert all(p['apex_mz_deviation']==0 and p['apex_intensity_deviation']==0 for p in peaks)
    assert all(p['relative_apex_height_source']==p['relative_apex_height_retained'] for p in peaks)
