from pathlib import Path

import numpy as np
import pytest
from kmzviewer.mzml_index import index_ms1
from kmzviewer.spectra import SpectrumReader
from kmz2viewer.index import index_ms2
from kmz2viewer.differences import analyze_spectrum
from kmz2viewer.formula import FormulaIndex, FormulaSettings

FIXTURES = Path(__file__).resolve().parents[2] / 'TestData/Regression'


def test_committed_profile_keeps_time_units_native_arrays_and_charge_rule():
    path = FIXTURES / 'profile.mzML'
    table = index_ms1(path)
    assert len(table) == 1 and table.rt_min.tolist() == [1.0]
    assert table.tic.tolist() == [5.0] and table.centroid.tolist() == [0]
    with SpectrumReader(path) as reader:
        mz, intensity = reader.read(int(table.spectrum_index[0]))
        np.testing.assert_array_equal(mz, [99, 100, 101])
        np.testing.assert_array_equal(intensity, [0, 5, 0])
    scan, = index_ms2(path)
    assert scan.rt_min == pytest.approx(61 / 60)
    np.testing.assert_array_equal(scan.intensity, [0, 8, 0])
    analysis = analyze_spectrum(scan.mz, scan.intensity, centroid=scan.centroid,
                                precursor_mz=scan.precursor_mz, charge=scan.charge,
                                formulas=FormulaIndex(), settings=FormulaSettings())
    assert analysis.precursor_virtual
    assert all(not pair.formula_allowed for pair in analysis.pairs if pair.involves_precursor)


def test_committed_centroid_keeps_scan_tic_separate_from_native_sum():
    path = FIXTURES / 'centroid.mzML'
    table = index_ms1(path)
    assert table.tic.tolist() == [40.0, 50.0]
    with SpectrumReader(path) as reader:
        _, intensity = reader.read(0)
        assert intensity.sum() == 7.0
    scan, = index_ms2(path)
    assert scan.centroid and scan.precursor_mz == 300 and scan.charge == 1
