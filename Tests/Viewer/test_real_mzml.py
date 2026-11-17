"""Checks against the mzML files shipped in this workspace."""

from pathlib import Path

import time

import numpy as np
import pytest

from kmzviewer.export_svg import ExportOptions, ExportTrace, export_chromatograms, svg_is_vector
from kmzviewer.extract import extract_eics, make_tic_trace, parse_targets, ppm_window, sum_interval
from kmzviewer.mzml_index import index_ms1
from pyteomics import mzml

from kmzviewer.spectra import ArrayCache, SpectrumReader

ROOT = Path(__file__).resolve().parents[2]


def local_fixture(name):
    """Use a root fixture when present, otherwise look in local user data."""
    root_path = ROOT / name
    return root_path if root_path.is_file() else ROOT / "TestData/Local" / name


TEST1 = local_fixture("viewer-01.mzML")
TEST2 = local_fixture("viewer-02.mzML")


pytestmark = pytest.mark.skipif(not TEST1.exists() or not TEST2.exists(), reason="sample mzML files are not in the workspace")


def test_sample_files_index_ms1_only():
    first = index_ms1(TEST1)
    second = index_ms1(TEST2)
    assert len(first) == 1837
    assert len(second) == 1852
    assert first.rt_min[0] == pytest.approx(0.779916666667)
    assert first.tic[0] == pytest.approx(1100326.0)
    assert int(first.spectrum_index[-1]) == 1980
    assert 1 not in set(int(value) for value in first.spectrum_index)
    assert set(int(value) for value in first.polarity) == {1}
    assert set(int(value) for value in first.centroid) == {0}
    assert np.all(np.isfinite(first.tic))
    assert second.rt_min[0] == pytest.approx(0.779833333333, rel=1e-4)
    assert second.tic[0] == pytest.approx(3.5789e6, rel=1e-4)


def test_sample_eic_matches_manual_sum_and_exports_vector_svg(tmp_path):
    table = index_ms1(TEST1)
    lower, upper = ppm_window(273.1559, 10)
    with SpectrumReader(TEST1) as reader:
        mz, intensity = reader.read(int(table.spectrum_index[0]))
        manual = 0.0
        for mass, value in zip(mz, intensity):
            if lower <= mass <= upper and np.isfinite(value):
                manual += float(value)
        assert sum_interval(mz, intensity, lower, upper) == manual == 31.0
    with mzml.MzML(str(TEST1)) as handle:
        assert handle.get_by_index(1)["ms level"] == 2

    cache = ArrayCache()
    started = time.perf_counter()
    traces = extract_eics(
        str(TEST1), table, parse_targets("273.1559", 10), "all", False, [7], ["#1f4e79"], cache=cache
    )
    first_pass = time.perf_counter() - started
    assert traces[0].y[0] == 31.0
    assert float(traces[0].y.max()) == 631.0
    assert traces[0].rt.shape == table.rt_min.shape
    assert cache.misses == len(table)
    assert cache.hits == 0
    started = time.perf_counter()
    again = extract_eics(
        str(TEST1), table, parse_targets("273.1559", 10), "all", False, [8], ["#1f4e79"], cache=cache
    )
    second_pass = time.perf_counter() - started
    assert np.array_equal(traces[0].y, again[0].y)
    assert cache.hits == len(table)
    assert cache.misses == len(table)
    assert second_pass < first_pass
    assert second_pass < 2.0

    tic = make_tic_trace(1, "#1f4e79", table)
    destination = tmp_path / "tic.svg"
    export_chromatograms(
        destination,
        [ExportTrace("viewer-01.mzML", tic.label, tic.color, tic.rt, tic.y)],
        ExportOptions(
            layout="stacked",
            width_in=7.0,
            height_in=3.0,
            linewidth=1.0,
            fontsize=9,
            xmin=float(tic.rt[0]),
            xmax=float(tic.rt[-1]),
            auto_y=True,
            ymin=None,
            ymax=None,
            filename_labels=True,
            trace_labels=True,
            transparent=False,
        ),
    )
    assert svg_is_vector(destination)
    assert "<image" not in destination.read_text(encoding="utf-8").lower()
