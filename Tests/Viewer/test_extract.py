"""Numerical TIC/EIC tests with hand-calculated intensities."""

import numpy as np
import pytest

from kmzviewer.extract import (
    describe_targets,
    extract_eics,
    fill_missing_tic,
    format_number,
    make_tic_trace,
    parse_targets,
    ppm_window,
    rows_for_polarity,
    sum_interval,
    trace_plan,
)
from kmzviewer.model import ScanTable
from kmzviewer.mzml_index import index_ms1
from Tests.Viewer.mzml_factory import write_mzml


def test_ppm_window_matches_spec_formula():
    mz = 273.1559
    lower, upper = ppm_window(mz, 10)
    assert lower == mz * (1 - 10e-6)
    assert upper == mz * (1 + 10e-6)


def test_sum_interval_is_inclusive_and_ignores_outside_points():
    mz = np.array([99.999, 100.0, 100.5, 101.0, 101.001])
    intensity = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert sum_interval(mz, intensity, 100.0, 101.0) == 14.0
    assert sum_interval(mz, intensity, 100.0, 100.0) == 2.0
    assert sum_interval(mz, intensity, 250.0, 300.0) == 0.0


def test_sum_interval_sorts_and_skips_non_finite():
    mz = np.array([101.0, 100.0, 100.5, np.nan])
    intensity = np.array([8.0, np.inf, 4.0, 5.0])
    # 100.5 -> 4 and 101 -> 8. The non-finite intensity at 100 is dropped.
    # nan m/z sorts to the end and is outside the window.
    assert sum_interval(mz, intensity, 100.0, 101.0) == 12.0


def test_repeated_extraction_is_identical():
    mz = np.array([273.1559 * (1 - 9e-6), 273.1559, 273.1559 * (1 + 11e-6)])
    intensity = np.array([3.5, 4.25, 9.0])
    lower, upper = ppm_window(273.1559, 10)
    first = sum_interval(mz, intensity, lower, upper)
    second = sum_interval(mz, intensity, lower, upper)
    assert first == second == 7.75


def test_parse_single_list_and_range():
    single = parse_targets("273.1559", 10)
    assert single.masses == (273.1559,)
    many = parse_targets("273.1559, 301.1412, 438.1532", 10)
    assert many.masses == (273.1559, 301.1412, 438.1532)
    parsed_range = parse_targets("250-300", 10)
    assert (parsed_range.lower, parsed_range.upper) == (250.0, 300.0)
    with pytest.raises(ValueError, match="250-300"):
        parse_targets("273.1559, 250-300", 10)


def test_range_plan_does_not_use_ppm_and_merge_sums_windows():
    parsed_range = parse_targets("250-300", 25)
    plan = trace_plan(parsed_range, "all", merge=True)
    assert plan[0][2] == [(250.0, 300.0)]
    assert "not applied" in plan[0][1]
    masses = parse_targets("100, 200", 10)
    separate = trace_plan(masses, "positive", merge=False)
    merged = trace_plan(masses, "positive", merge=True)
    assert len(separate) == 2
    assert len(merged) == 1
    assert merged[0][2] == [ppm_window(100, 10), ppm_window(200, 10)]


def test_describe_range_mentions_ppm():
    text = describe_targets(parse_targets("250-300", 10), "all", False, 2)
    assert "ppm is not applied" in text


def test_polarity_rows_keep_unspecified_scans_only_for_all():
    scans = ScanTable(
        spectrum_index=np.array([0, 1, 2]),
        rt_min=np.array([0.1, 0.2, 0.3]),
        tic=np.array([1.0, 1.0, 1.0]),
        polarity=np.array([1, -1, 0], dtype=np.int8),
        centroid=np.array([0, 0, 0], dtype=np.int8),
        scan_id=["a", "b", "c"],
    )
    assert rows_for_polarity(scans, "all").tolist() == [0, 1, 2]
    assert rows_for_polarity(scans, "positive").tolist() == [0]
    assert rows_for_polarity(scans, "negative").tolist() == [1]


def test_file_eic_matches_manual_sums(tmp_path):
    path = tmp_path / "sample.mzML"
    write_mzml(
        path,
        [
            {
                "id": "pos",
                "mz": [99.0, 100.0, 100.5, 101.0, 102.0],
                "intensity": [1.0, 2.0, 4.0, 8.0, 16.0],
                "rt": 1.0,
                "tic": 100.0,
                "polarity": 1,
            },
            {
                "id": "ms2",
                "ms_level": 2,
                "mz": [100.0],
                "intensity": [50.0],
                "rt": 1.1,
                "tic": 50.0,
                "polarity": 1,
            },
            {
                "id": "neg",
                "mz": [100.0, 100.2],
                "intensity": [5.0, 7.0],
                "rt": 90.0,
                "rt_unit": "second",
                "tic": 12.0,
                "polarity": -1,
            },
        ],
        chromatogram=True,
    )
    scans = index_ms1(path, chunk_size=64)
    assert scans.scan_id == ["pos", "neg"]
    assert scans.spectrum_index.tolist() == [0, 2]
    assert scans.rt_min.tolist() == pytest.approx([1.0, 1.5])
    tic = make_tic_trace(1, "#1f4e79", scans)
    assert tic.y.tolist() == [100.0, 12.0]

    target = parse_targets("100.5", 0)
    # 0 ppm keeps only the exact 100.5 point on the positive scan: intensity 4.
    traces = extract_eics(str(path), scans, target, "positive", False, [2], ["#c05621"])
    assert traces[0].y.tolist() == [4.0]
    assert traces[0].rt.tolist() == pytest.approx([1.0])

    # Explicit range sums 100, 100.5 and 101 on scan 1 (2+4+8=14) and both
    # points on the negative scan (5+7=12). MS2 is absent from the index.
    ranged = parse_targets("100-101", 50)
    both = extract_eics(str(path), scans, ranged, "all", False, [3], ["#2f6f4e"])
    assert both[0].y.tolist() == [14.0, 12.0]

    merged = parse_targets("100, 102", 0)
    merged_trace = extract_eics(str(path), scans, merged, "positive", True, [4], ["#333333"])
    # Positive scan: exact 100 -> 2 and exact 102 -> 16. Negative scan is excluded.
    assert merged_trace[0].y.tolist() == [18.0]


def test_missing_tic_falls_back_to_intensity_sum(tmp_path):
    path = tmp_path / "notic.mzML"
    write_mzml(
        path,
        [
            {
                "mz": [10.0, 11.0],
                "intensity": [3.0, 4.0],
                "rt": 0.2,
                "tic": None,
                "polarity": 1,
            }
        ],
    )
    scans = index_ms1(path, chunk_size=48)
    assert np.isnan(scans.tic[0])
    fill_missing_tic(str(path), scans)
    assert scans.tic[0] == 7.0


def test_overlapping_merge_counts_a_shared_point_twice():
    mz = np.array([100.0])
    intensity = np.array([5.0])
    # Two identical windows. The spec sums each requested window.
    total = sum_interval(mz, intensity, 100.0, 100.0) + sum_interval(mz, intensity, 100.0, 100.0)
    assert total == 10.0


def test_format_number_keeps_given_precision():
    assert format_number(273.1559) == "273.1559"
    assert format_number(10) == "10"
