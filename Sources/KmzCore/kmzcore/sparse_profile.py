"""Sparse Profile: versioned MS1 source subsets, renderer and fail-closed mzML contract.

KmzConvert writes v3 (event-normalized peak regions with chromatographic edge
rescue). v1 (fixed >100 counts) and v2 (per-spectrum peak regions) files remain
recognized. The historical v1 benchmark remains an independent oracle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET

import numpy as np
from .mass_correction import valid_method_params
from .mzml import require_current_input

V1, V2, V3 = "sparse-profile-v1", "sparse-profile-v2", "sparse-profile-v3"
FORMAT = V3
FAMILY = "sciex-positive-tof-framed-profile-v1"
SAMPLING_PARAM = "KmzSuite source sampling check"
REGION_UNITS, TRIGGER_UNITS, MIN_REGION_POINTS, RESCUE_UNITS = 2, 20, 3, 8
_COMMON = {
    "KmzSuite representation": "sparse-profile",
    "KmzSuite boundary guard rule": "one-source-neighbor-each-side",
    "KmzSuite centroiding": "false",
    "KmzSuite sampling family": FAMILY,
    "KmzSuite rendering": "sqrt-mz-local-spacing-v1",
}
METHOD_PARAMS = {
    V1: {**_COMMON, "KmzSuite sparse profile version": "1",
         "KmzSuite MS1 intensity comparator": ">", "KmzSuite MS1 intensity threshold": "100"},
    V2: {**_COMMON, "KmzSuite sparse profile version": "2",
         "KmzSuite MS1 event unit": "most-frequent-positive-source-intensity",
         "KmzSuite MS1 region rule": f"contiguous-source-points-above-{REGION_UNITS}-event-units",
         "KmzSuite MS1 region trigger": f"apex-above-{TRIGGER_UNITS}-event-units",
         "KmzSuite MS1 minimum region points": str(MIN_REGION_POINTS)},
    V3: {**_COMMON, "KmzSuite sparse profile version": "3",
         "KmzSuite MS1 event unit": "most-frequent-positive-source-intensity",
         "KmzSuite MS1 region rule": f"contiguous-source-points-above-{REGION_UNITS}-event-units",
         "KmzSuite MS1 region trigger": f"apex-above-{TRIGGER_UNITS}-event-units",
         "KmzSuite MS1 minimum region points": str(MIN_REGION_POINTS),
         "KmzSuite MS1 rescue floor": f"apex-above-{RESCUE_UNITS}-event-units",
         "KmzSuite MS1 rescue rule": "linked-to-trigger-region-by-overlapping-mz-spans-in-consecutive-ms1-spectra-of-one-experiment"},
}
PARAMS = METHOD_PARAMS[FORMAT]
_COUNTS = {
    V1: {"source_points", "retained_points", "above_threshold_points", "added_guard_points", "source_signal_breaks"},
    V2: {"source_points", "retained_points", "selected_points", "selected_regions", "added_guard_points",
         "source_signal_breaks"},
}
_COUNTS[V3] = _COUNTS[V2] | {"rescued_regions"}
_KEYS = {V1: _COUNTS[V1] | {"sampling_family"}, V2: _COUNTS[V2] | {"sampling_family", "event_unit"},
         V3: _COUNTS[V3] | {"sampling_family", "event_unit"}}
HELP = "MS1 is sparsified for inspection. Use the full-profile mzML for full-fidelity analysis."


def output_path(path, profile="full"):
    """Normalize an explicitly chosen sparse name; preserve full explicit paths."""
    path = Path(path)
    if profile == "full":
        return path
    if profile != "sparse":
        raise ValueError("Unknown output profile")
    name = path.name
    if name.lower().endswith(".mzml"):
        name = name[:-5]
    if name.lower().endswith(".sp"):
        name = name[:-3]
    while name.lower().endswith(".mzml"):
        name = name[:-5]
    return path.with_name(name + ".sp.mzML")


def validate_arrays(mz, intensity):
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    if mz.ndim != 1 or intensity.ndim != 1 or len(mz) != len(intensity):
        raise ValueError("Expected aligned one-dimensional arrays")
    if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
        raise ValueError("Nonfinite measurements")
    if len(mz) and (mz[0] <= 0 or np.any(np.diff(mz) <= 0)):
        raise ValueError("Expected positive, strictly increasing m/z")
    return mz, intensity


def retained_indices(intensity):
    """v1: strict >100 source indices plus one available original neighbor per edge.

    Expansion is from the original mask, never recursively from added points.
    A neighbor is an existing profile point (including source framing zeros).
    """
    intensity = np.asarray(intensity)
    if intensity.ndim != 1 or not np.isfinite(intensity).all():
        raise ValueError("Expected finite one-dimensional intensity")
    return np.flatnonzero(_guard(intensity > 100))


def _guard(mask):
    keep = mask.copy()
    keep[1:] |= mask[:-1]
    keep[:-1] |= mask[1:]
    return keep


def _runs(mask):
    """Half-open [start, end) bounds of contiguous True runs."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def event_unit(intensity):
    """v2 per-spectrum unit: most frequent positive intensity (smallest on ties).

    X500 MS1 intensities are scaled event counts whose scale varies by spectrum;
    the most frequent value is the low-count detector-event level. 0 when the
    spectrum has no positive intensity.
    """
    intensity = np.asarray(intensity)
    positive = intensity[intensity > 0]
    if not len(positive):
        return 0.0
    values, counts = np.unique(positive, return_counts=True)
    return float(values[np.argmax(counts)])


def peak_regions(intensity, unit):
    """v2 retained regions: maximal runs >2u with apex >20u and at least three points."""
    intensity = np.asarray(intensity)
    if unit <= 0:
        return np.zeros(0, int), np.zeros(0, int)
    starts, ends = _runs(intensity > REGION_UNITS * unit)
    if not len(starts):
        return starts, ends
    apex = np.maximum.reduceat(intensity, starts)
    keep = (ends - starts >= MIN_REGION_POINTS) & (apex > TRIGGER_UNITS * unit)
    return starts[keep], ends[keep]


def region_indices(length, starts, ends):
    """Selected half-open regions plus one original neighbor per edge; (indices, selected points)."""
    marks = np.zeros(length + 1, dtype=np.int64)
    np.add.at(marks, starts, 1)
    np.add.at(marks, ends, -1)
    selected = np.cumsum(marks)[:-1] > 0
    return np.flatnonzero(_guard(selected)), int(selected.sum())


def peak_indices(intensity):
    """v2 source indices: selected peak regions plus one original neighbor per edge.

    Returns (indices, unit, selected point count, selected region count).
    """
    intensity = np.asarray(intensity)
    if intensity.ndim != 1 or not np.isfinite(intensity).all():
        raise ValueError("Expected finite one-dimensional intensity")
    unit = event_unit(intensity)
    starts, ends = peak_regions(intensity, unit)
    indices, selected = region_indices(len(intensity), starts, ends)
    return indices, unit, selected, len(starts)


class Regions(NamedTuple):
    """v3 candidate regions of one MS1 spectrum, ordered by m/z."""
    unit: float
    starts: np.ndarray
    ends: np.ndarray
    apex: np.ndarray
    lower: np.ndarray   # m/z of each region's first point
    upper: np.ndarray   # m/z of each region's last point


def candidate_regions(mz, intensity):
    """v3 candidates: maximal runs >2u with at least three points and apex >8u."""
    mz, intensity = validate_arrays(mz, intensity)
    unit = event_unit(intensity)
    starts, ends = _runs(intensity > REGION_UNITS * unit) if unit > 0 else (np.zeros(0, int), np.zeros(0, int))
    apex = np.maximum.reduceat(intensity, starts) if len(starts) else np.zeros(0)
    keep = (ends - starts >= MIN_REGION_POINTS) & (apex > RESCUE_UNITS * unit)
    starts, ends, apex = starts[keep], ends[keep], apex[keep]
    return Regions(unit, starts, ends, apex, mz[starts], mz[ends - 1])


def rescue_selection(sequence):
    """v3 run-level choice for consecutive MS1 spectra of one experiment.

    Candidates in neighboring spectra are linked when their closed m/z spans
    overlap. A candidate is selected when its linked component contains a
    trigger region (apex >20u), so a peak that reaches the trigger keeps its
    chromatographic edges while they stay above 8u. Returns one boolean array
    per spectrum, aligned with its candidate regions.
    """
    sizes = [len(r.starts) for r in sequence]
    offsets = np.cumsum([0, *sizes])
    parent = list(range(int(offsets[-1])))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for j in range(len(sequence) - 1):
        a, b = sequence[j], sequence[j + 1]
        if not len(a.starts) or not len(b.starts):
            continue
        # Spans within one spectrum are disjoint and ordered, so the overlapping
        # neighbors of each span form one contiguous range.
        first = np.searchsorted(b.upper, a.lower, "left")
        counts = np.maximum(np.searchsorted(b.lower, a.upper, "right") - first, 0)
        left = np.repeat(np.arange(len(a.starts)), counts) + offsets[j]
        right = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
                 + np.repeat(first, counts) + offsets[j + 1])
        for x, y in zip(left.tolist(), right.tolist()):
            x, y = find(x), find(y)
            if x != y:
                parent[max(x, y)] = min(x, y)
    seeded = {find(i) for r, base in zip(sequence, offsets)
              for i in (np.flatnonzero(r.apex > TRIGGER_UNITS * r.unit) + base).tolist()}
    chosen = np.array([find(i) in seeded for i in range(len(parent))], dtype=bool)
    return [chosen[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def gap_connections(mz):
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
    factor, radius = 4.0, 8
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


def source_sampling_check(mz, intensity, *, sampling_family, sparse_format=FORMAT, selection=None):
    """Attest the full source before filtering, never infer family from sparsity.

    v3 ``selection`` is this spectrum's slice of ``rescue_selection`` over the
    run. Without it the spectrum is treated as a one-spectrum run (seeds only).
    """
    if sampling_family != FAMILY:
        raise ValueError("Unqualified Sparse Profile source family")
    if sparse_format not in METHOD_PARAMS:
        raise ValueError("Unknown Sparse Profile format")
    if selection is not None and sparse_format != V3:
        raise ValueError("Run-level selection applies only to Sparse Profile v3")
    mz, intensity = validate_arrays(mz, intensity)
    connections = gap_connections(mz)
    signal = (intensity[:-1] > 0) | (intensity[1:] > 0)
    if np.any(~connections & signal):
        raise ValueError("Sparse Profile gap rule breaks a source-neighbor signal interval")
    if sparse_format == V1:
        indices = retained_indices(intensity)
        above = int(np.count_nonzero(intensity > 100))
        check = dict(sampling_family=FAMILY, source_points=len(mz),
                     retained_points=len(indices), above_threshold_points=above,
                     added_guard_points=len(indices) - above, source_signal_breaks=0)
        return indices, check
    if sparse_format == V2:
        indices, unit, selected, regions = peak_indices(intensity)
        check = dict(sampling_family=FAMILY, source_points=len(mz), event_unit=unit,
                     retained_points=len(indices), selected_points=selected, selected_regions=regions,
                     added_guard_points=len(indices) - selected, source_signal_breaks=0)
        return indices, check
    candidates = candidate_regions(mz, intensity)
    chosen = rescue_selection([candidates])[0] if selection is None else np.asarray(selection)
    if chosen.dtype != bool or chosen.shape != candidates.starts.shape:
        raise ValueError("Sparse Profile v3 selection does not match the spectrum")
    indices, selected = region_indices(len(intensity), candidates.starts[chosen], candidates.ends[chosen])
    rescued = int(np.count_nonzero(chosen & (candidates.apex <= TRIGGER_UNITS * candidates.unit)))
    check = dict(sampling_family=FAMILY, source_points=len(mz), event_unit=candidates.unit,
                 retained_points=len(indices), selected_points=selected, selected_regions=int(chosen.sum()),
                 rescued_regions=rescued, added_guard_points=len(indices) - selected, source_signal_breaks=0)
    return indices, check


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate provenance field")
        result[key] = value
    return result


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _children(element, tag):
    return [e for e in element if _tag(e) == tag]


def _params(element):
    return _unique((p.get("name"), p.get("value")) for p in _children(element, "userParam")
                   if p.get("name", "").startswith("KmzSuite "))


class Provenance:
    """Resolve processing/software references from metadata, independently of names."""

    def __init__(self, root=None, default_ref=None):
        self.methods = set()
        self.formats = {}
        self.default_ref = default_ref
        if root is None:
            return
        try:
            if _tag(root) == "indexedmzML":
                root, = _children(root, "mzML")
            software_list, = _children(root, "softwareList")
            software = _unique((e.get("id"), e) for e in _children(software_list, "software"))
            processing_list, = _children(root, "dataProcessingList")
            processing = _unique((e.get("id"), e) for e in _children(processing_list, "dataProcessing"))
            if any(not ident or not ident.strip() for ident in (*software, *processing)):
                return
            for ident, dp in processing.items():
                methods = _children(dp, "processingMethod")
                orders = [m.get("order") for m in methods]
                if orders not in (["0", "1"], ["0", "1", "2"]):
                    continue
                conversion, method = methods[0], methods[-1]
                if len(methods) == 3:
                    correction = methods[1]
                    actions = _children(correction, "cvParam")
                    if (len(actions) != 1 or actions[0].get("accession") != "MS:1001485" or actions[0].get("cvRef") != "MS"
                            or correction.get("softwareRef") != method.get("softwareRef")
                            or not valid_method_params(_params(correction))):
                        continue
                params = _params(method)
                sparse_format = next((f for f, expected in METHOD_PARAMS.items() if params == expected), None)
                if sparse_format is None:
                    continue
                sw = software.get(method.get("softwareRef"))
                if sw is None or not sw.get("version", "").strip() or conversion.get("softwareRef") != method.get("softwareRef"):
                    continue
                if not any(p.get("accession") == "MS:1000799" and p.get("value") == "KmzConvert" for p in _children(sw, "cvParam")):
                    continue
                conversion_cv = _children(conversion, "cvParam")
                if len(conversion_cv) != 1 or conversion_cv[0].get("accession") != "MS:1000544" or conversion_cv[0].get("cvRef") != "MS":
                    continue
                actions = _children(method, "cvParam")
                if any(p.get("cvRef") != "MS" for p in actions):
                    continue
                if sparse_format == V1:
                    if len(actions) != 2 or {p.get("accession") for p in actions} != {"MS:1001486", "MS:1000787"}:
                        continue
                    threshold = next(p for p in actions if p.get("accession") == "MS:1000787")
                    if threshold.get("value") != "100" or threshold.get("unitAccession") != "MS:1000131" or threshold.get("unitCvRef") != "MS":
                        continue
                # v2/v3 thresholds are per-spectrum event units, recorded in userParams and attestations.
                elif len(actions) != 1 or actions[0].get("accession") != "MS:1001486":
                    continue
                self.methods.add(ident)
                self.formats[ident] = sparse_format
        except (ValueError, TypeError, AttributeError):
            self.methods.clear()
            self.formats.clear()

    def sampling(self, spectrum):
        """Recognize a supported per-spectrum contract; arrays are checked separately."""
        try:
            ref = spectrum.get("dataProcessingRef") or self.default_ref
            if not ref or ref not in self.methods:
                return None
            cvs = _children(spectrum, "cvParam")
            flags = {p.get("accession") for p in cvs}
            levels = [p for p in cvs if p.get("accession") == "MS:1000511"]
            if len(levels) != 1 or levels[0].get("value") != "1":
                return None
            if not {"MS:1000128", "MS:1000130"} <= flags or flags & {"MS:1000127", "MS:1000129"}:
                return None
            params = _params(spectrum)
            if set(params) != {SAMPLING_PARAM}:
                return None
            check = json.loads(params[SAMPLING_PARAM], object_pairs_hook=_unique)
            sparse_format = self.formats[ref]
            if not isinstance(check, dict) or set(check) != _KEYS[sparse_format]:
                return None
            counts = _COUNTS[sparse_format]
            if check["sampling_family"] != FAMILY or any(type(check[k]) is not int or check[k] < 0 for k in counts):
                return None
            if not (check["source_points"] >= check["retained_points"] == int(spectrum.get("defaultArrayLength"))
                    and check["source_signal_breaks"] == 0):
                return None
            if sparse_format == V1:
                above, guards = check["above_threshold_points"], check["added_guard_points"]
                return check if check["retained_points"] == above + guards and guards <= 2 * above else None
            unit, selected, regions = check["event_unit"], check["selected_points"], check["selected_regions"]
            if type(unit) not in (int, float) or not np.isfinite(unit) or unit < 0 or (unit == 0 and regions):
                return None
            if not (check["retained_points"] == selected + check["added_guard_points"]
                    and check["added_guard_points"] <= 2 * regions
                    and selected >= MIN_REGION_POINTS * regions and (regions == 0) == (selected == 0)):
                return None
            if sparse_format == V3 and check["rescued_regions"] > regions:
                return None
            return check
        except (ValueError, TypeError, AttributeError):
            return None


def read_provenance(path, *, is_cancelled=None):
    """Read only the mzML metadata prefix; binary arrays are never parsed here."""
    require_current_input(path, is_cancelled=is_cancelled)
    try:
        with open(path, "rb") as stream:
            parser = ET.iterparse(stream, events=("start",))
            root = None
            for _, element in parser:
                if root is None:
                    root = element
                if _tag(element) == "spectrumList":
                    return Provenance(root, element.get("defaultDataProcessingRef"))
                if stream.tell() > 4 * 1024 * 1024:
                    return Provenance()
    except (ET.ParseError, ValueError, OSError):
        pass
    return Provenance()


def rendering_connections(check, mz, intensity):
    """Fail closed on incompatible arrays, preserving ordinary rendering as fallback."""
    if check is None:
        return None
    try:
        mz, intensity = validate_arrays(mz, intensity)
        if len(mz) != check["retained_points"] or np.any(intensity < 0):
            return None
        if set(check) in (_KEYS[V2], _KEYS[V3]):
            # Every retained run above 2u must be one attested, rule-satisfying region.
            # v3 rescue links are cross-spectrum; one spectrum attests only their count.
            unit = check["event_unit"]
            starts, ends = _runs(intensity > REGION_UNITS * unit)
            if len(starts) != check["selected_regions"] or int((ends - starts).sum()) != check["selected_points"]:
                return None
            apex = np.maximum.reduceat(intensity, starts) if len(starts) else np.zeros(0)
            if np.any(ends - starts < MIN_REGION_POINTS):
                return None
            if "rescued_regions" not in check:
                if np.any(apex <= TRIGGER_UNITS * unit):
                    return None
            elif (np.any(apex <= RESCUE_UNITS * unit)
                  or int(np.count_nonzero(apex <= TRIGGER_UNITS * unit)) != check["rescued_regions"]):
                return None
        elif int(np.count_nonzero(intensity > 100)) != check["above_threshold_points"]:
            return None
        return gap_connections(mz)
    except (ValueError, TypeError, KeyError):
        return None
