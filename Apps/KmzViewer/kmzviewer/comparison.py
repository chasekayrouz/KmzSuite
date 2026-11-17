"""RT/polarity selection for native MS1 comparisons; no interpolation or alignment."""
from dataclasses import dataclass

import numpy as np

from kmzviewer.model import ScanTable
from kmzviewer.profile_rendering import zero_baseline_curve


@dataclass(frozen=True)
class ScanMatch:
    row: int | None
    message: str = ""


def match_scan(scans: ScanTable, rt: float, polarity: int) -> ScanMatch:
    finite = np.isfinite(scans.rt_min)
    times = scans.rt_min[finite]
    if not times.size:
        return ScanMatch(None, "No MS1 scans")
    if rt < times.min() or rt > times.max():
        return ScanMatch(None, "Outside run range")
    # Unknown polarity does not imply positive or negative: match unknown only.
    rows = np.flatnonzero(finite & (scans.polarity == polarity))
    if not rows.size:
        name = {1: "positive", -1: "negative", 0: "unspecified-polarity"}[polarity]
        return ScanMatch(None, f"No {name} MS1 scans")
    # argmin preserves acquisition order at ties, including repeated/nonmonotonic RTs.
    return ScanMatch(int(rows[np.argmin(np.abs(scans.rt_min[rows] - rt))]))


@dataclass
class SpectrumFrame:
    file_id: int
    name: str
    color: str
    caption: str
    mz: np.ndarray
    intensity: np.ndarray
    centroid: bool = False
    connections: np.ndarray | None = None
    message: str = ""
    rt: float | None = None
    polarity: int = 0
    profile_curve: tuple[np.ndarray, np.ndarray] | None = None
    optical: bool = False

    def __post_init__(self):
        # The worker normally prepares this once. Standalone panels use the same
        # path; layout/color changes reuse it, while labels/readouts stay native.
        if self.profile_curve is None and not self.centroid:
            self.profile_curve = zero_baseline_curve(self.mz, self.intensity, self.connections)

    @property
    def title(self) -> str:
        name = self.name.removesuffix(".mzML").removesuffix(".mzml")
        if self.message:
            return f"{name} · {self.message}"
        if self.optical:
            return f"{name} · {self.rt:.4f} min · DAD" if self.rt is not None else name
        polarity = {1: "+", -1: "−", 0: "?"}[self.polarity]
        return f"{name} · {self.rt:.4f} min · {polarity}" if self.rt is not None else name
