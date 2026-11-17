"""MS2 indexing and AND filters, including missing optional metadata."""

import numpy as np

from kmz2viewer.filter import ScanFilter, apply_filter
from kmz2viewer.index import index_ms2
from Tests.Viewer.mzml_factory import write_mzml


def _write(path):
    write_mzml(
        path,
        [
            {
                "id": "ms1",
                "ms_level": 1,
                "mz": [100.0, 200.0],
                "intensity": [10.0, 20.0],
                "rt": 1.0,
                "tic": 30.0,
                "centroid": True,
            },
            {
                "id": "early",
                "ms_level": 2,
                "mz": [100.0, 132.076, 180.0],
                "intensity": [5.0, 50.0, 100.0],
                "rt": 4.0,
                "tic": 155.0,
                "centroid": True,
                "precursor_mz": 250.0,
                "charge": 1,
                "precursor_intensity": 1000.0,
                "collision_energy": 25.0,
            },
            {
                "id": "late",
                "ms_level": 2,
                "mz": [90.0, 132.076],
                "intensity": [1.0, 4.0],
                "rt": 6.0,
                "tic": None,
                "centroid": True,
                "precursor_mz": 400.0,
            },
            {
                "id": "profile",
                "ms_level": 2,
                "mz": [131.9, 132.076, 132.2],
                "intensity": [1.0, 20.0, 1.0],
                "rt": 90.0,
                "rt_unit": "second",
                "tic": 22.0,
                "centroid": False,
                "precursor_mz": 250.001,
            },
        ],
    )


def test_index_keeps_ms2_only_and_optional_metadata(tmp_path):
    path = tmp_path / "run.mzML"
    _write(path)
    scans = index_ms2(path, chunk_size=48)
    assert [scan.scan_id for scan in scans] == ["early", "late", "profile"]
    early = scans[0]
    assert early.precursor_mz == 250.0
    assert early.charge == 1
    assert early.precursor_intensity == 1000.0
    assert early.collision_energy == 25.0
    assert early.base_peak_mz == 180.0
    assert early.n_peaks == 3
    assert early.mz.tolist() == [100.0, 132.076, 180.0]
    late = scans[1]
    assert late.charge is None
    assert late.collision_energy is None
    assert late.precursor_intensity is None
    assert late.tic == 5.0
    assert scans[2].rt_min == 1.5
    assert scans[2].centroid is False
    assert scans[2].n_peaks == 1


def test_filters_combine_with_and(tmp_path):
    one = tmp_path / "one.mzML"
    two = tmp_path / "two.mzML"
    _write(one)
    write_mzml(
        two,
        [
            {
                "id": "other",
                "ms_level": 2,
                "mz": [132.076],
                "intensity": [80.0],
                "rt": 5.2,
                "tic": 80.0,
                "centroid": True,
                "precursor_mz": 250.002,
                "charge": 1,
            }
        ],
    )
    scans = index_ms2(one, chunk_size=40) + index_ms2(two)
    assert len(apply_filter(scans, ScanFilter())) == 4
    precursor = apply_filter(scans, ScanFilter(precursor_mz=250.0, precursor_tolerance=20, precursor_unit="ppm"))
    assert {scan.scan_id for scan in precursor} == {"early", "profile", "other"}
    fragment = apply_filter(scans, ScanFilter(fragment_mz=132.076, fragment_tolerance=20, fragment_unit="ppm"))
    assert {scan.scan_id for scan in fragment} == {"early", "late", "profile", "other"}
    strong = apply_filter(
        scans,
        ScanFilter(fragment_mz=132.076, fragment_tolerance=20, fragment_unit="ppm", fragment_min_percent=30),
    )
    # early: 50/100 = 50%. late: 4/4 = 100%. profile: 20/20 = 100%. other: 100%.
    assert {scan.scan_id for scan in strong} == {"early", "late", "profile", "other"}
    weak_hidden = apply_filter(
        scans,
        ScanFilter(fragment_mz=100.0, fragment_tolerance=0.01, fragment_unit="Da", fragment_min_percent=20),
    )
    assert weak_hidden == []
    window = apply_filter(scans, ScanFilter(rt_min=5.0, rt_max=5.5))
    assert [scan.scan_id for scan in window] == ["other"]
    assert [scan.scan_id for scan in apply_filter(scans, ScanFilter(rt_min=5.0))] == ["late", "other"]
    assert [scan.scan_id for scan in apply_filter(scans, ScanFilter(rt_max=4.5))] == ["early", "profile"]
    combined = apply_filter(
        scans,
        ScanFilter(
            precursor_mz=250.0,
            precursor_tolerance=20,
            precursor_unit="ppm",
            fragment_mz=132.076,
            fragment_tolerance=20,
            fragment_unit="ppm",
            rt_min=5.0,
            rt_max=5.5,
        ),
    )
    assert [scan.scan_id for scan in combined] == ["other"]
    assert np.isclose(scans[2].mz[1], 132.076)
