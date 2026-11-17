"""Explicit units and evidence; no calibrated presence probability."""
from dataclasses import dataclass, field
from pathlib import Path
import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from . import METHOD, __version__


@dataclass(frozen=True)
class Settings:
    mz_ppm: float = 10.0
    rt_tolerance: float = 0.10  # minutes, clustering and alignment residual
    rt_window: float = 0.15  # minutes, extraction half-window
    discovery_window: float = 0.15  # rise/fall search; never smaller than rt_window
    discovery_height: float = 1000.0
    min_height: float = 0.0  # optional evidence floor, never a measurement cutoff
    min_points: int = 3
    min_snr: float = 5.0
    min_contrast: float = 3.0
    max_scan_gap: float = 0.05  # minutes, never integrate over a larger gap
    max_rt_shift: float = 0.50
    min_anchors: int = 5
    alignment: str = "shift"

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "alignment":
                continue
            if name == "min_height" and not isinstance(value, bool) and math.isfinite(value) and value == 0:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (not isinstance(self.min_points, int) or self.min_points < 3 or
                not isinstance(self.min_anchors, int) or self.min_anchors < 3):
            raise ValueError("min_points and min_anchors must be integers >= 3")
        if self.rt_tolerance > self.rt_window or self.mz_ppm >= 1e5:
            raise ValueError("Require rt_tolerance <= rt_window and mz_ppm < 100000")
        if self.min_contrast <= 1 or self.alignment not in {"shift", "none"}:
            raise ValueError("Require min_contrast > 1 and alignment shift or none")


@dataclass
class Scan:
    rt: float
    polarity: int
    mode: str
    mz: np.ndarray
    intensity: np.ndarray
    windows: tuple[tuple[float, float], ...]
    intensity_unit: str


@dataclass(frozen=True)
class Candidate:
    sample: int
    mz: float
    rt: float
    height: float
    polarity: int


@dataclass
class Sample:
    path: Path
    id: str
    sha256: str
    size_bytes: int
    scans: int = 0
    modes: set[str] = field(default_factory=set)
    units: set[str] = field(default_factory=set)
    rt_min: float = math.inf
    rt_max: float = -math.inf
    candidates: list[Candidate] = field(default_factory=list)
    shifts: dict[int, float] = field(default_factory=dict)
    anchors: dict[int, int] = field(default_factory=dict)
    alignment_mad: dict[int, float | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    native_sample_id: int | None = None
    source_files: list[dict] = field(default_factory=list)


@dataclass
class Feature:
    id: str
    mz: float
    rt: float  # aligned minutes, reference time axis
    polarity: int
    members: list[Candidate]
    left: float = 0.0  # aligned extraction bounds
    right: float = 0.0
    ambiguous: bool = False


@dataclass
class Measurement:
    feature_id: str
    sample_id: str
    discovered: bool
    status: str
    height: float | None = None
    height_above_background: float | None = None
    window_area: float | None = None  # native intensity * minute
    apex_rt: float | None = None  # native time axis
    apex_mz: float | None = None
    background: float | None = None
    noise: float | None = None
    snr: float | None = None
    contrast: float | None = None
    points: int = 0
    support_points: int = 0
    flags: str = ""


@dataclass
class Result:
    settings: Settings
    samples: list[Sample]
    features: list[Feature]
    measurements: list[list[Measurement]]  # sample major
    elapsed_seconds: float
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    method: str = METHOD
    suite_version: str = __version__
