"""Peak summaries for MS2 scans. Profile local maxima are display anchors only."""

from __future__ import annotations

import numpy as np


def local_maxima_mask(intensity: np.ndarray) -> np.ndarray:
    count = int(intensity.size)
    keep = np.zeros(count, dtype=bool)
    if count == 0:
        return keep
    if count == 1:
        keep[0] = np.isfinite(intensity[0]) and intensity[0] > 0
        return keep
    finite = np.isfinite(intensity)
    keep[0] = bool(finite[0] and finite[1] and intensity[0] > intensity[1] and intensity[0] > 0)
    keep[-1] = bool(finite[-1] and finite[-2] and intensity[-1] > intensity[-2] and intensity[-1] > 0)
    if count > 2:
        center = intensity[1:-1]
        keep[1:-1] = (
            finite[1:-1]
            & finite[:-2]
            & finite[2:]
            & (center >= intensity[:-2])
            & (center > intensity[2:])
            & (center > 0)
        )
    return keep


def peak_table(mz: np.ndarray, intensity: np.ndarray, centroid: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return m/z and intensity of fragment peaks. Profile data contributes local maxima."""
    if mz.size == 0 or intensity.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if centroid:
        keep = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    else:
        keep = local_maxima_mask(intensity) & np.isfinite(mz)
    return mz[keep], intensity[keep]


def base_peak(mz: np.ndarray, intensity: np.ndarray) -> tuple[float | None, float]:
    if intensity.size == 0:
        return None, 0.0
    finite = np.isfinite(intensity) & np.isfinite(mz)
    if not np.any(finite):
        return None, 0.0
    index = int(np.argmax(np.where(finite, intensity, -np.inf)))
    return float(mz[index]), float(intensity[index])


def total_intensity(intensity: np.ndarray) -> float:
    if intensity.size == 0:
        return 0.0
    finite = intensity[np.isfinite(intensity)]
    if finite.size == 0:
        return 0.0
    return float(np.sum(finite, dtype=np.float64))


def max_in_window(mz: np.ndarray, intensity: np.ndarray, lower: float, upper: float) -> float:
    if mz.size == 0 or upper < lower:
        return 0.0
    order = np.argsort(mz, kind="mergesort") if mz.size > 1 and np.any(np.diff(mz) < 0) else None
    if order is not None:
        mz = mz[order]
        intensity = intensity[order]
    left = int(np.searchsorted(mz, lower, side="left"))
    right = int(np.searchsorted(mz, upper, side="right"))
    if left >= right:
        return 0.0
    chunk = intensity[left:right]
    finite = chunk[np.isfinite(chunk)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def top_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    centroid: bool,
    limit: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    peak_mz, peak_y = peak_table(mz, intensity, centroid)
    if peak_mz.size == 0 or limit <= 0:
        return peak_mz[:0], peak_y[:0]
    order = np.argsort(-peak_y, kind="mergesort")
    chosen = order[: min(limit, order.size)]
    # Stable by intensity, then restore m/z order for deterministic pairs.
    chosen = chosen[np.argsort(peak_mz[chosen], kind="mergesort")]
    return peak_mz[chosen], peak_y[chosen]
