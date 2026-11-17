"""Benchmark-only profile simplification. Production does not import this module."""
from __future__ import annotations

import time
import numpy as np


def validate_arrays(mz, intensity):
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    if mz.ndim != 1 or intensity.ndim != 1 or len(mz) != len(intensity):
        raise ValueError("Expected aligned one-dimensional arrays")
    if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
        raise ValueError("Nonfinite measurements")
    if len(mz) and (mz[0] <= 0 or np.any(np.diff(mz) <= 0)):
        raise ValueError("Expected positive, strictly increasing m/z")
    return mz, intensity


def guard_indices(intensity, threshold=100.0, guards=1):
    """Union of strict >threshold source indices and up to N neighbors per edge.

    Expansion is from the original mask, never recursively from added points.
    A neighbor is an existing profile point (including source framing zeros).
    """
    if guards not in (0, 1, 2):
        raise ValueError("Benchmark supports zero, one or two guards")
    intensity = np.asarray(intensity)
    if intensity.ndim != 1 or not np.isfinite(intensity).all():
        raise ValueError("Expected finite one-dimensional intensity")
    above = intensity > threshold
    keep = above.copy()
    for distance in range(1, guards + 1):
        keep[distance:] |= above[:-distance]
        keep[:-distance] |= above[distance:]
    return np.flatnonzero(keep)


def event_peak_indices(intensity, region_units=2, trigger_units=20, min_points=3):
    """Independent loop oracle for Sparse Profile v2; returns (indices, event unit).

    The unit is the most frequent positive value (smallest on ties). Keep each
    maximal run above region_units*unit with at least min_points points and an
    apex above trigger_units*unit, plus one existing source neighbor per edge.
    """
    values = [float(v) for v in intensity]
    frequency = {}
    for v in values:
        if v > 0:
            frequency[v] = frequency.get(v, 0) + 1
    if not frequency:
        return np.zeros(0, dtype=int), 0.0
    unit = min(frequency, key=lambda v: (-frequency[v], v))
    keep, i, n = set(), 0, len(values)
    while i < n:
        if values[i] <= region_units * unit:
            i += 1
            continue
        j = i
        while j < n and values[j] > region_units * unit:
            j += 1
        if j - i >= min_points and max(values[i:j]) > trigger_units * unit:
            keep.update(range(max(i - 1, 0), min(j + 1, n)))
        i = j
    return np.array(sorted(keep), dtype=int), unit


def event_regions(mz, intensity, region_units=2, rescue_units=8, min_points=3):
    """Loop oracle for Sparse Profile v3 candidates of one spectrum.

    Returns (unit, point count, [(start, end, apex, first m/z, last m/z)]) for
    maximal runs above region_units*unit with at least min_points points and an
    apex above rescue_units*unit; end is exclusive.
    """
    values, masses = [float(v) for v in intensity], [float(m) for m in mz]
    frequency = {}
    for v in values:
        if v > 0:
            frequency[v] = frequency.get(v, 0) + 1
    if not frequency:
        return 0.0, len(values), []
    unit = min(frequency, key=lambda v: (-frequency[v], v))
    regions, i, n = [], 0, len(values)
    while i < n:
        if values[i] <= region_units * unit:
            i += 1
            continue
        j = i
        while j < n and values[j] > region_units * unit:
            j += 1
        apex = max(values[i:j])
        if j - i >= min_points and apex > rescue_units * unit:
            regions.append((i, j, apex, masses[i], masses[j - 1]))
        i = j
    return unit, n, regions


def event_rescue(sequence, trigger_units=20):
    """Breadth-first v3 selection over consecutive spectra of one MS1 experiment.

    ``sequence`` holds event_regions results. Regions in neighboring spectra
    are linked when their closed m/z spans overlap; every region reachable from
    one whose apex exceeds trigger_units*unit is selected. Returns sets of
    selected region positions, one per spectrum.
    """
    from collections import deque
    links = {}
    for j in range(len(sequence) - 1):
        a, b = sequence[j][2], sequence[j + 1][2]
        p = q = 0
        while p < len(a) and q < len(b):
            if a[p][4] < b[q][3]:
                p += 1
            elif b[q][4] < a[p][3]:
                q += 1
            else:
                links.setdefault((j, p), []).append((j + 1, q))
                links.setdefault((j + 1, q), []).append((j, p))
                if a[p][4] < b[q][4]:
                    p += 1
                else:
                    q += 1
    chosen = [set() for _ in sequence]
    queue = deque((j, k) for j, (unit, _, regions) in enumerate(sequence)
                  for k, region in enumerate(regions) if region[2] > trigger_units * unit)
    for j, k in queue:
        chosen[j].add(k)
    while queue:
        node = queue.popleft()
        for j, k in links.get(node, ()):
            if k not in chosen[j]:
                chosen[j].add(k)
                queue.append((j, k))
    return chosen


def event_guarded(spectrum_regions, chosen):
    """Source indices of chosen regions plus one existing neighbor per edge."""
    _, n, regions = spectrum_regions
    keep = set()
    for k in chosen:
        start, end = regions[k][:2]
        keep.update(range(max(start - 1, 0), min(end + 1, n)))
    return np.array(sorted(keep), dtype=int)


def event_rescue_plan(spectra):
    """Loop oracle over a run: (key, MS1 experiment, mz, intensity) items in file order.

    Returns {key: retained source indices}; each experiment is one sequence.
    """
    sequences = {}
    for key, experiment, mz, intensity in spectra:
        sequences.setdefault(experiment, []).append((key, event_regions(mz, intensity)))
    plan = {}
    for items in sequences.values():
        for (key, regions), chosen in zip(items, event_rescue([r for _, r in items])):
            plan[key] = event_guarded(regions, chosen)
    return plan


def event_rescue_indices(spectra, region_units=2, trigger_units=20, rescue_units=8, min_points=3):
    """Loop oracle for Sparse Profile v3 over one sequence of (mz, intensity).

    Returns [(indices, event unit)] in sequence order.
    """
    sequence = [event_regions(mz, y, region_units, rescue_units, min_points) for mz, y in spectra]
    return [(event_guarded(s, c), s[0]) for s, c in zip(sequence, event_rescue(sequence, trigger_units))]


def gap_connections(mz, factor=4.0, radius=8):
    """Return N-1 booleans: whether to draw each neighboring segment.

    Tested scope: positive profile TOF spectra. sqrt(m/z) removes the quadratic
    TOF calibration's smoothly changing m/z step. For each candidate interval,
    exclude that interval and take the lower quartile of up to eight intervals
    on each side. Use the larger supported side estimate. A side needs at least
    three intervals and two within 20% of its estimate. No source indices,
    intensities, threshold, peak labels or reference file enter the detector.
    With no supported local estimate, keep the segment (conservative fallback).
    Gradual spacing variation and uniformly sparse sampling remain connected.
    """
    mz = np.asarray(mz, dtype=np.float64)
    if mz.ndim != 1 or not np.isfinite(mz).all() or (len(mz) and mz[0] <= 0) or np.any(np.diff(mz) <= 0):
        raise ValueError("Expected positive ordered finite m/z")
    if factor <= 1 or radius < 3:
        raise ValueError("Gap factor must exceed one; radius must be at least three")
    delta = np.diff(np.sqrt(mz))
    if not len(delta):
        return np.ones(0, dtype=bool)
    padded = np.pad(delta, (radius, radius), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    estimates = []
    for side in (windows[:, :radius], windows[:, radius + 1:]):
        # Replace unavailable samples with +inf for sorting, without nan warnings.
        valid = np.isfinite(side)
        count = valid.sum(axis=1)
        ordered = np.sort(np.where(valid, side, np.inf), axis=1)
        rank = .25 * np.maximum(count - 1, 0)
        lo, hi = np.floor(rank).astype(int), np.ceil(rank).astype(int)
        a, b = ordered[np.arange(len(delta)), lo], ordered[np.arange(len(delta)), hi]
        estimate = np.where(count > 0, a, 0).copy()
        use = count > 0
        estimate[use] = a[use] + (b[use] - a[use]) * (rank[use] - lo[use])
        supported = (count >= 3) & (np.sum(valid & (side >= .8 * estimate[:, None]) & (side <= 1.2 * estimate[:, None]), axis=1) >= 2)
        estimates.append(np.where(supported, estimate, 0.0))
    local = np.maximum(*estimates)
    return (local == 0) | (delta <= factor * local)


def plotted_trace(grid, mz, intensity, connections):
    """Benchmark evaluation of drawn segments; blanks stay NaN, never observations."""
    grid = np.asarray(grid)
    result = np.full(len(grid), np.nan)
    if not len(mz):
        return result
    if len(mz) == 1:
        result[grid == mz[0]] = intensity[0]
        return result
    where = np.searchsorted(mz, grid, side="right") - 1
    clipped = np.clip(where, 0, len(mz) - 2)
    covered = (grid >= mz[0]) & (grid <= mz[-1]) & connections[clipped]
    result[covered] = np.interp(grid[covered], mz, intensity)
    exact = np.searchsorted(grid, mz)
    valid = exact < len(grid)
    exact, masses, values = exact[valid], mz[valid], intensity[valid]
    valid = grid[exact] == masses
    result[exact[valid]] = values[valid]
    return result


def major_apices(mz, intensity, limit=20):
    if len(mz) < 3:
        return np.array([], dtype=int)
    maxima = np.flatnonzero((intensity[1:-1] > intensity[:-2]) & (intensity[1:-1] >= intensity[2:])) + 1
    maxima = maxima[intensity[maxima] > 100]
    # Fixed separation selects distinct major peak regions, not shoulder noise.
    chosen = []
    for at in sorted(maxima, key=lambda i: (-intensity[i], mz[i])):
        if all(abs(mz[at] - mz[p]) >= .05 for p in chosen):
            chosen.append(int(at))
            if len(chosen) == limit:
                break
    return np.array(chosen, dtype=int)


def half_width(mz, intensity, apex, connections):
    """Apparent FWHM only when both half-height crossings are drawn and bounded."""
    half = intensity[apex] / 2
    crossings = []
    for direction in (-1, 1):
        at = apex
        for _ in range(len(mz)):
            neighbor = at + direction
            if neighbor < 0 or neighbor >= len(mz) or not connections[min(at, neighbor)]:
                return None
            if intensity[neighbor] <= half:
                fraction = (half - intensity[at]) / (intensity[neighbor] - intensity[at])
                crossings.append(float(mz[at] + fraction * (mz[neighbor] - mz[at])))
                break
            at = neighbor
        else:
            return None
    return crossings[1] - crossings[0]


def shape_metrics(mz, intensity, indices, connections):
    """Metrics on source knots + source midpoints, solely for trace comparisons.

    Blank-as-zero error measures missing ink, not inferred baseline measurements.
    Covered-only error and coverage are reported alongside it to avoid rewarding
    omission. Integrals use trapezoids on this irregular common grid.
    """
    start = time.perf_counter()
    m, y = mz[indices], intensity[indices]
    grid = np.sort(np.concatenate((mz, (mz[1:] + mz[:-1]) / 2)))
    reference = np.interp(grid, mz, intensity) if len(mz) else np.array([])
    trace = plotted_trace(grid, m, y, connections)
    blank = ~np.isfinite(trace)
    ink = np.nan_to_num(trace, nan=0.0)
    error = ink - reference
    ref_area = float(np.trapezoid(np.abs(reference), grid))
    width = float(mz[-1] - mz[0]) if len(mz) > 1 else 0
    covered_width = float(np.sum(np.diff(m) * connections)) if len(m) > 1 else 0
    weights = np.zeros(len(grid))
    if len(grid) > 1:
        weights[:-1] += np.diff(grid) / 2
        weights[1:] += np.diff(grid) / 2
    square = float(np.sum(error ** 2 * weights))
    apex = float(intensity.max(initial=0))
    crossing = np.flatnonzero((intensity[:-1] > 100) != (intensity[1:] > 100))
    edge_mask = np.zeros(len(mz), dtype=bool)
    for offset in (-1, 0, 1, 2):
        edge_mask[np.clip(crossing + offset, 0, max(0, len(mz) - 1))] = True
    knot_error = np.abs(error[::2])
    # grid alternates source knots/midpoints for strictly ordered input.
    deleted = np.diff(indices) > 1
    bridging = deleted & connections
    bridge_width = np.diff(m)[bridging]
    apices = major_apices(mz, intensity)
    peak_mask = np.zeros(len(grid), dtype=bool)
    for at in apices:
        peak_mask |= abs(grid - mz[at]) <= .05
    retained_fraction = len(indices) / len(mz) if len(mz) else 0
    result = {
        "profile_points": len(mz), "retained_points": len(indices), "retained_fraction": retained_fraction,
        "above_threshold_points": int((intensity > 100).sum()),
        "above_threshold_retained": int((y > 100).sum()), "added_guard_points": int((y <= 100).sum()),
        "mz_max_absolute_deviation": 0.0, "mz_median_absolute_deviation": 0.0,
        "intensity_max_relative_deviation": 0.0, "intensity_median_relative_deviation": 0.0,
        "ink_iad": float(np.sum(np.abs(error) * weights)),
        "ink_iad_normalized": float(np.sum(np.abs(error) * weights) / ref_area) if ref_area else 0.0,
        "ink_nrmse_apex": float(np.sqrt(square / width) / apex) if width and apex else 0.0,
        "drawn_mz_fraction": covered_width / width if width else 0.0,
        "covered_only_mae_counts": float(np.mean(abs(error[~blank]))) if np.any(~blank) else None,
        "omitted_reference_area_fraction": float(np.sum(reference[blank] * weights[blank]) / ref_area) if ref_area else 0.0,
        "threshold_edge_mae_counts": float(np.mean(knot_error[edge_mask])) if np.any(edge_mask) else None,
        "peak_region_nrmse_apex": float(np.sqrt(np.mean(error[peak_mask] ** 2)) / apex) if np.any(peak_mask) and apex else None,
        "deleted_regions_bridged": int(bridging.sum()), "bridge_total_mz": float(bridge_width.sum()),
        "bridge_max_mz": float(bridge_width.max(initial=0)),
        "bridge_max_endpoint_intensity": float(np.maximum(y[:-1], y[1:])[bridging].max(initial=0)) if len(y)>1 else 0.0,
        "render_discontinuities": int((~connections).sum()),
        "breaks_over_deleted_points": int((deleted & ~connections).sum()),
        "breaks_between_source_neighbors": int((~deleted & ~connections).sum()),
        "breaks_between_source_neighbors_with_signal": int((~deleted & ~connections & ((y[:-1]>0) | (y[1:]>0))).sum()),
        "isolated_render_points": int(np.sum(~np.r_[False, connections] & ~np.r_[connections, False])) if len(indices) else 0,
        "array_memory_bytes": int(m.nbytes + y.nbytes), "connection_memory_bytes": int(connections.nbytes),
    }
    peaks = []
    all_connections = np.ones(max(0, len(mz) - 1), dtype=bool)
    for at in apices:
        position = int(np.searchsorted(indices, at))
        retained = position < len(indices) and indices[position] == at
        ref_width = half_width(mz, intensity, at, all_connections)
        candidate_width = half_width(m, y, position, connections) if retained else None
        peaks.append({"source_index": int(at), "apex_mz": float(mz[at]), "apex_intensity": float(intensity[at]),
                      "apex_retained": retained, "apex_mz_deviation": float(m[position] - mz[at]) if retained else None,
                      "apex_intensity_deviation": float(y[position] - intensity[at]) if retained else None,
                      "reference_fwhm": ref_width, "apparent_fwhm": candidate_width,
                      "relative_apex_height_source": float(intensity[at]/apex) if apex else None,
                      "relative_apex_height_retained": float(y[position]/y.max()) if retained and y.max() else None,
                      "fwhm_relative_error": (candidate_width - ref_width) / ref_width if candidate_width is not None and ref_width else None})
    widths = [abs(p["fwhm_relative_error"]) for p in peaks if p["fwhm_relative_error"] is not None]
    result["major_apices"] = len(peaks)
    result["major_apices_retained"] = sum(p["apex_retained"] for p in peaks)
    result["fwhm_incomplete"] = sum(p["apparent_fwhm"] is None for p in peaks)
    result["median_absolute_fwhm_relative_error"] = float(np.median(widths)) if widths else None
    result["metric_seconds"] = time.perf_counter() - start
    return result, peaks
