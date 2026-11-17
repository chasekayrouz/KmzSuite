"""Visible rendered signal and reversible, signed percent display scaling."""
from __future__ import annotations

import numpy as np


def visible_signal(x, y, xmin, xmax, *, sticks=False):
    """Include line intersections at the window edges, never bridge NaN gaps.

    Centroid sticks contribute only their acquired peaks. Profile callers pass
    their rendered geometry (including explicit sparse zero-baseline vertices).
    Intersections affect display bounds only, not measurements or extraction.
    """
    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    parts = [y[finite & (x >= xmin) & (x <= xmax)]]
    if not sticks and x.size > 1:
        valid = finite[:-1] & finite[1:]
        for edge in (xmin, xmax):
            crossing = valid & (((x[:-1] < edge) & (x[1:] > edge)) |
                                ((x[1:] < edge) & (x[:-1] > edge)))
            i = np.flatnonzero(crossing)
            if i.size:
                weight = (edge - x[i]) / (x[i + 1] - x[i])
                parts.append(y[i] * (1 - weight) + y[i + 1] * weight)
    return np.concatenate(parts)


def peak_reference(visible):
    """Positive peak is 100%; negative-only data use their trough magnitude.

    Zero/empty windows use a neutral divisor and remain zero/empty. No baseline
    subtraction, sign inversion, clipping or source-array mutation occurs.
    """
    finite = visible[np.isfinite(visible)]
    if not finite.size:
        return 1.0
    maximum = float(finite.max())
    return maximum if maximum > 0 else (abs(float(finite.min())) or 1.0)


def percent(values, reference):
    return np.asarray(values) / reference * 100.0
