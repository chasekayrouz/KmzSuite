"""AND filters over indexed MS2 scans. Blank fields are ignored."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kmz2viewer.model import Ms2Scan
from kmz2viewer.peaks import max_in_window


@dataclass
class ScanFilter:
    precursor_mz: float | None = None
    precursor_tolerance: float = 20.0
    precursor_unit: str = "ppm"
    fragment_mz: float | None = None
    fragment_tolerance: float = 20.0
    fragment_unit: str = "ppm"
    fragment_min_percent: float = 0.0
    rt_min: float | None = None
    rt_max: float | None = None


def mass_window(center: float, tolerance: float, unit: str) -> tuple[float, float]:
    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")
    if unit == "ppm":
        delta = tolerance * 1e-6
        return center * (1.0 - delta), center * (1.0 + delta)
    if unit == "Da":
        return center - tolerance, center + tolerance
    raise ValueError(f"Unknown tolerance unit: {unit}")


def scan_matches(scan: Ms2Scan, spec: ScanFilter) -> bool:
    if spec.precursor_mz is not None:
        if scan.precursor_mz is None or not np.isfinite(scan.precursor_mz):
            return False
        lower, upper = mass_window(spec.precursor_mz, spec.precursor_tolerance, spec.precursor_unit)
        if not (lower <= scan.precursor_mz <= upper):
            return False
    if spec.rt_min is not None:
        if not np.isfinite(scan.rt_min) or scan.rt_min < spec.rt_min:
            return False
    if spec.rt_max is not None:
        if not np.isfinite(scan.rt_min) or scan.rt_min > spec.rt_max:
            return False
    if spec.fragment_mz is not None:
        lower, upper = mass_window(spec.fragment_mz, spec.fragment_tolerance, spec.fragment_unit)
        observed = max_in_window(scan.mz, scan.intensity, lower, upper)
        if observed <= 0:
            return False
        if spec.fragment_min_percent > 0:
            if scan.base_peak_intensity <= 0:
                return False
            if observed < scan.base_peak_intensity * (spec.fragment_min_percent / 100.0):
                return False
    return True


def apply_filter(scans: list[Ms2Scan], spec: ScanFilter) -> list[Ms2Scan]:
    return [scan for scan in scans if scan_matches(scan, spec)]
