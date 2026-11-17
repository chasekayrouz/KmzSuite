"""Choose readable m/z labels for the visible part of a spectrum.

Profile traces contribute local maxima only. Those maxima are label anchors.
The stored profile arrays are not changed.
"""

from __future__ import annotations

import numpy as np

RELATIVE_HEIGHT = 0.05
MIN_SEPARATION_PX = 72.0


def label_aware_y(ymin, ymax, anchors, view_height_px, label_top_px):
    """Minimum upper limit leaving a measured pixel band above labeled peaks.

    Solve (ymax - peak) / (ymax - ymin) >= label_top_px / view_height_px.
    Start from fresh numeric bounds, so resizing/zooming can reduce headroom too.
    """
    if not anchors or view_height_px <= label_top_px:
        return ymin, ymax
    fraction = label_top_px / view_height_px
    top = max((height - fraction * ymin) / (1 - fraction) for _mass, height in anchors)
    return ymin, max(ymax, top)


def format_mz_label(mz: float) -> str:
    return f"{mz:.4f}"


def select_peak_labels(
    mz: np.ndarray,
    intensity: np.ndarray,
    xmin: float,
    xmax: float,
    *,
    centroid: bool,
    view_width_px: float,
    relative_height: float = RELATIVE_HEIGHT,
    min_separation_px: float = MIN_SEPARATION_PX,
) -> list[tuple[float, float]]:
    """Return (m/z, intensity) anchors, tallest first among those that fit."""
    if mz.size == 0 or intensity.size == 0 or xmax <= xmin or view_width_px < 30:
        return []
    visible = (mz >= xmin) & (mz <= xmax) & np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    if not np.any(visible):
        return []
    mz_view = mz[visible]
    intensity_view = intensity[visible]
    visible_max = float(np.max(intensity_view))
    if visible_max <= 0:
        return []
    if centroid:
        candidates_mz = mz_view
        candidates_y = intensity_view
    else:
        keep = _local_maxima(intensity_view)
        if not np.any(keep):
            return []
        candidates_mz = mz_view[keep]
        candidates_y = intensity_view[keep]
    strong = candidates_y >= visible_max * relative_height
    if not np.any(strong):
        return []
    candidates_mz = candidates_mz[strong]
    candidates_y = candidates_y[strong]
    order = np.argsort(-candidates_y, kind="stable")
    span = float(xmax - xmin)
    candidates_mz = candidates_mz[order]
    candidates_y = candidates_y[order]
    # Greedy, tallest first: a candidate closer than the separation to any
    # accepted label is skipped. Same float64 pixel arithmetic per element.
    pixels = (np.asarray(candidates_mz, dtype=np.float64) - float(xmin)) / span * float(view_width_px)
    free = np.ones(pixels.size, dtype=bool)
    accepted: list[tuple[float, float]] = []
    index = 0
    while True:
        remaining = np.flatnonzero(free[index:])
        if not remaining.size:
            return accepted
        index += int(remaining[0])
        accepted.append((float(candidates_mz[index]), float(candidates_y[index])))
        free &= ~(np.abs(pixels - pixels[index]) < min_separation_px)
        index += 1


def _local_maxima(intensity: np.ndarray) -> np.ndarray:
    count = int(intensity.size)
    keep = np.zeros(count, dtype=bool)
    if count == 1:
        keep[0] = True
        return keep
    keep[0] = intensity[0] > intensity[1]
    keep[-1] = intensity[-1] > intensity[-2]
    if count > 2:
        center = intensity[1:-1]
        keep[1:-1] = (center >= intensity[:-2]) & (center > intensity[2:])
    return keep
