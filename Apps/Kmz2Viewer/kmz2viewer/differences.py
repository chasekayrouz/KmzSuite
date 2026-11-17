"""Pairwise exact-mass differences among the major peaks of one MS2 scan.

These are predicted formulas for mass differences, not identified fragments.
Precursor m/z is an anchor. When no observed peak sits on it, the anchor is
virtual and is not a spectrum peak. For a known charge above 1, precursor-to-fragment
differences are left unassigned because Δm/z is not an unambiguous neutral loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kmz2viewer.formula import FormulaHit, FormulaIndex, FormulaSettings
from kmz2viewer.peaks import top_peaks

PRECURSOR_MATCH_PPM = 10.0


@dataclass
class Anchor:
    mz: float
    intensity: float
    is_precursor: bool
    virtual: bool


@dataclass
class PairAssignment:
    left_mz: float
    right_mz: float
    delta: float
    left_intensity: float
    right_intensity: float
    involves_precursor: bool
    formula_allowed: bool
    hits: list[FormulaHit] = field(default_factory=list)

    @property
    def best(self) -> FormulaHit | None:
        return self.hits[0] if self.hits else None


@dataclass
class SpectrumAnalysis:
    anchors: list[Anchor]
    pairs: list[PairAssignment]
    precursor_added: bool
    precursor_virtual: bool
    seconds: float = 0.0

    @property
    def assigned_pairs(self) -> list[PairAssignment]:
        return [pair for pair in self.pairs if pair.best is not None]


def _close_to_precursor(mz: float, precursor: float) -> bool:
    if precursor == 0:
        return abs(mz - precursor) <= 1e-4
    return abs(mz - precursor) <= abs(precursor) * PRECURSOR_MATCH_PPM * 1e-6


def select_anchors(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    centroid: bool,
    precursor_mz: float | None,
    limit: int = 10,
) -> tuple[list[Anchor], bool, bool]:
    """Return anchors, whether a precursor anchor was added beyond the top peaks, and whether it is virtual."""
    peak_mz, peak_y = top_peaks(mz, intensity, centroid, limit)
    anchors = [
        Anchor(mz=float(mass), intensity=float(height), is_precursor=False, virtual=False)
        for mass, height in zip(peak_mz, peak_y)
    ]
    if precursor_mz is None or not np.isfinite(precursor_mz):
        return anchors, False, False
    observed_mz, observed_y = top_peaks(mz, intensity, centroid, limit=10**9)
    nearest = None
    nearest_distance = None
    for mass, height in zip(observed_mz, observed_y):
        if not _close_to_precursor(float(mass), float(precursor_mz)):
            continue
        distance = abs(float(mass) - float(precursor_mz))
        if nearest_distance is None or distance < nearest_distance:
            nearest = (float(mass), float(height))
            nearest_distance = distance
    if nearest is not None:
        mass, height = nearest
        for anchor in anchors:
            if anchor.mz == mass:
                anchor.is_precursor = True
                return anchors, False, False
        anchors.append(Anchor(mz=mass, intensity=height, is_precursor=True, virtual=False))
        return anchors, True, False
    anchors.append(Anchor(mz=float(precursor_mz), intensity=0.0, is_precursor=True, virtual=True))
    return anchors, True, True


def analyze_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    centroid: bool,
    precursor_mz: float | None,
    charge: int | None,
    formulas: FormulaIndex,
    settings: FormulaSettings,
) -> SpectrumAnalysis:
    import time

    started = time.perf_counter()
    anchors, added, virtual = select_anchors(
        mz, intensity, centroid=centroid, precursor_mz=precursor_mz
    )
    charge_blocks_precursor = charge is not None and charge > 1
    pairs: list[PairAssignment] = []
    for left in range(len(anchors)):
        for right in range(left + 1, len(anchors)):
            a = anchors[left]
            b = anchors[right]
            if b.mz >= a.mz:
                low, high = a, b
            else:
                low, high = b, a
            delta = high.mz - low.mz
            if delta <= 0 or not np.isfinite(delta):
                continue
            involves = low.is_precursor or high.is_precursor
            allowed = not (involves and charge_blocks_precursor)
            hits = formulas.lookup(delta, settings) if allowed else []
            pairs.append(
                PairAssignment(
                    left_mz=low.mz,
                    right_mz=high.mz,
                    delta=delta,
                    left_intensity=low.intensity,
                    right_intensity=high.intensity,
                    involves_precursor=involves,
                    formula_allowed=allowed,
                    hits=hits,
                )
            )
    return SpectrumAnalysis(
        anchors=anchors,
        pairs=pairs,
        precursor_added=added,
        precursor_virtual=virtual,
        seconds=time.perf_counter() - started,
    )
