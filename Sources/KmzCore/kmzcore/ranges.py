"""Axis limits shared by the on-screen plots and the SVG export."""

from __future__ import annotations

import numpy as np


def finite_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


def fill_floor(x: np.ndarray, y: np.ndarray) -> float:
    """Shade toward zero or the lowest finite point, without moving the data.

    Use the complete trace so zooming/cropping cannot change the fill boundary.
    Nonfinite coordinates are gaps, not measurements.
    """
    x, y = np.asarray(x), np.asarray(y)
    finite = y[np.isfinite(x) & np.isfinite(y)]
    return min(0.0, float(finite.min())) if finite.size else 0.0


def padded_y(values: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    finite = finite_values(values)
    if finite.size == 0:
        return -1.0, 1.0
    ymin = float(finite.min())
    ymax = float(finite.max())
    span = ymax - ymin
    if span == 0.0:
        pad = 1.0 if ymax == 0.0 else abs(ymax) * fraction
    else:
        pad = span * fraction
    return ymin - pad, ymax + pad


def values_in_window(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    if x.size == 0:
        return y
    mask = (x >= xmin) & (x <= xmax) & np.isfinite(y)
    if not np.any(mask):
        return np.asarray([], dtype=np.float64)
    return y[mask]
