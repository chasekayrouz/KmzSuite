"""In-memory MS2 scans. Arrays stay with the scan until its file is closed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Ms2Scan:
    path: str
    file_name: str
    spectrum_index: int
    scan_id: str
    rt_min: float
    precursor_mz: float | None
    charge: int | None
    precursor_intensity: float | None
    collision_energy: float | None
    centroid: bool
    mz: np.ndarray
    intensity: np.ndarray
    base_peak_mz: float | None
    base_peak_intensity: float
    tic: float
    n_peaks: int
    polarity: int = 0
    representation: str = "unknown"
    precursor_source: str = "unknown"

    @property
    def key(self) -> tuple[str, int]:
        return (self.path, self.spectrum_index)


@dataclass
class Library:
    scans: list[Ms2Scan] = field(default_factory=list)

    def add(self, scans: list[Ms2Scan]) -> None:
        present = {scan.key for scan in self.scans}
        for scan in scans:
            if scan.key not in present:
                self.scans.append(scan)
                present.add(scan.key)

    def clear(self) -> None:
        self.scans.clear()

    def drop_path(self, path: str) -> None:
        self.scans = [scan for scan in self.scans if scan.path != path]

    @property
    def files(self) -> list[str]:
        seen: list[str] = []
        for scan in self.scans:
            if scan.file_name not in seen:
                seen.append(scan.file_name)
        return seen
