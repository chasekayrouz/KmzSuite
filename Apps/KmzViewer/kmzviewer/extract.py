"""TIC, BPC and EIC calculation from native mzML peaks.

An EIC point is the sum of intensities whose m/z falls inside the window,
including both endpoints. Empty windows contribute zero. No centroiding,
interpolation, or smoothing is applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from kmzcore.cancel import Cancelled
from kmzviewer.model import ScanTable, Trace
from kmzviewer.spectra import ArrayCache, SpectrumReader, read_ahead, read_arrays, sum_intensities

try:  # Optional native checks; see _native.c.
    from kmzviewer._native import finite_max as _finite_max, ordered as _ordered
except ImportError:
    _finite_max = _ordered = None

_NUMBER = r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RANGE = re.compile(rf"^\s*({_NUMBER})\s*-\s*({_NUMBER})\s*$")
_MASS = re.compile(rf"^{_NUMBER}$")


@dataclass(frozen=True)
class MassList:
    masses: tuple[float, ...]
    ppm: float


@dataclass(frozen=True)
class MzRange:
    lower: float
    upper: float


def ppm_window(mz: float, ppm: float) -> tuple[float, float]:
    """Symmetric window. 10 ppm uses a factor of 1e-5, written 10e-6 in the spec."""
    if ppm < 0:
        raise ValueError("ppm tolerance cannot be negative.")
    delta = ppm * 1e-6
    return mz * (1.0 - delta), mz * (1.0 + delta)


def sum_interval(mz: np.ndarray, intensity: np.ndarray, lower: float, upper: float) -> float:
    """Sum native intensities with lower <= m/z <= upper."""
    if mz.size == 0 or intensity.size == 0 or upper < lower:
        return 0.0
    if mz.shape[0] != intensity.shape[0]:
        raise ValueError("m/z and intensity arrays have different lengths.")
    mz, intensity = _prepare(mz, intensity)
    left = int(np.searchsorted(mz, lower, side="left"))
    right = int(np.searchsorted(mz, upper, side="right"))
    if left >= right:
        return 0.0
    chunk = intensity[left:right]
    finite = chunk[np.isfinite(chunk)]
    if finite.size == 0:
        return 0.0
    return float(np.sum(finite, dtype=np.float64))


def _prepare(mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite masses in nondecreasing order. The native check returns the same arrays
    as the NumPy expressions below, in one pass that keeps the GIL."""
    if _ordered is not None and mz.dtype == np.float64 and mz.ndim == 1 and mz.flags.c_contiguous:
        state = _ordered(mz)
        if state == 2:
            return mz, intensity
        if state == 1:
            return _sorted_pairs(mz, intensity)
    finite = np.isfinite(mz)
    if not np.all(finite):
        mz, intensity = mz[finite], intensity[finite]
    return _sorted_pairs(mz, intensity)


def _sorted_pairs(mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mz.size <= 1:
        return mz, intensity
    if mz[0] <= mz[-1] and np.all(mz[1:] >= mz[:-1]):
        return mz, intensity
    order = np.argsort(mz, kind="mergesort")
    return mz[order], intensity[order]


def _sum_windows(mz: np.ndarray, intensity: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Prepare a scan once and locate all inclusive windows in one search.

    Each window retains the native finite slice and NumPy's float64 reduction,
    exactly as ``sum_interval``. Prefix-sum subtraction would lose small peaks
    following large ones and change summation roundoff, so it is not used.
    """
    sums = np.zeros(len(bounds), dtype=np.float64)
    if not mz.size or not intensity.size or not len(bounds):
        return sums
    if mz.shape[0] != intensity.shape[0]:
        raise ValueError("m/z and intensity arrays have different lengths.")
    mz, intensity = _prepare(mz, intensity)
    left = np.searchsorted(mz, bounds[:, 0], side="left")
    right = np.searchsorted(mz, bounds[:, 1], side="right")
    for index in np.flatnonzero(left < right):
        chunk = intensity[left[index]:right[index]]
        sums[index] = np.sum(chunk[np.isfinite(chunk)], dtype=np.float64)
    return sums


def parse_targets(text: str, ppm: float) -> MassList | MzRange:
    raw = text.strip()
    if not raw:
        raise ValueError("Enter an m/z, a comma-separated list, or a range such as 250-300.")
    if ppm < 0:
        raise ValueError("ppm tolerance cannot be negative.")
    range_match = _RANGE.fullmatch(raw)
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        if upper < lower:
            raise ValueError("The end of the m/z range is lower than the start.")
        return MzRange(lower, upper)
    masses: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            raise ValueError("The m/z list has an empty entry.")
        if not _MASS.fullmatch(token):
            raise ValueError(
                f"Could not read '{token}'. Use commas between m/z values, or a single range such as 250-300."
            )
        masses.append(float(token))
    if not masses:
        raise ValueError("Enter at least one m/z.")
    return MassList(tuple(masses), ppm)


def describe_targets(target: MassList | MzRange, polarity: str, merge: bool, file_count: int) -> str:
    polarity_text = {"all": "all polarities", "positive": "positive scans", "negative": "negative scans"}[polarity]
    if isinstance(target, MzRange):
        return (
            f"{file_count} file(s). One MS1 chromatogram per file, summing m/z "
            f"{format_number(target.lower)} through {format_number(target.upper)} inclusive. "
            "ppm is not applied to an explicit range."
        )
    if len(target.masses) > 1 and merge:
        listed = ", ".join(format_number(mass) for mass in target.masses)
        return (
            f"{file_count} file(s). One merged MS1 chromatogram per file from {listed}, "
            f"each ±{format_number(target.ppm)} ppm, {polarity_text}."
        )
    count = len(target.masses)
    noun = "chromatogram" if count == 1 else "chromatograms"
    return (
        f"{file_count} file(s). {count} MS1 {noun} per file, "
        f"±{format_number(target.ppm)} ppm, {polarity_text}."
    )


def format_number(value: float) -> str:
    if value == 0:
        return "0"
    absolute = abs(value)
    if absolute >= 1e6 or absolute < 1e-4:
        return f"{value:.6g}"
    return f"{value:.10f}".rstrip("0").rstrip(".")


def rows_for_polarity(scans: ScanTable, polarity: str) -> np.ndarray:
    if polarity == "all":
        return np.arange(len(scans), dtype=np.int32)
    wanted = {"positive": 1, "negative": -1}.get(polarity)
    if wanted is None:
        raise ValueError(f"Unknown polarity filter: {polarity}")
    return np.flatnonzero(scans.polarity == wanted).astype(np.int32, copy=False)


def make_tic_trace(trace_id: int, color: str, scans: ScanTable) -> Trace:
    rows = np.arange(len(scans), dtype=np.int32)
    return Trace(
        trace_id=trace_id,
        file_id=0,
        kind="tic",
        label="TIC MS1",
        rt=np.array(scans.rt_min, dtype=np.float64, copy=True),
        y=np.array(scans.tic, dtype=np.float64, copy=True),
        row=rows,
        detail="MS1 total ion current from each scan",
        color=color,
        extraction_key=("tic",),
    )


def fill_missing_tic(
    path: str,
    scans: ScanTable,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> None:
    missing = np.flatnonzero(~np.isfinite(scans.tic))
    if missing.size == 0:
        return
    with SpectrumReader(path) as reader:
        read_ahead(reader, None, path, scans.spectrum_index[missing])
        total = int(missing.size)
        for done, row in enumerate(missing, start=1):
            if is_cancelled is not None and is_cancelled():
                raise Cancelled()
            _mz, intensity = reader.read(int(scans.spectrum_index[int(row)]))
            if is_cancelled is not None and is_cancelled():
                raise Cancelled()
            scans.tic[int(row)] = sum_intensities(intensity)
            if on_progress is not None:
                on_progress(done, total)


def trace_plan(target: MassList | MzRange, polarity: str, merge: bool) -> list[tuple[str, str, list[tuple[float, float]]]]:
    """Return (label, detail, windows) for each chromatogram a file will gain."""
    polarity_text = {"all": "all polarities", "positive": "positive", "negative": "negative"}[polarity]
    if isinstance(target, MzRange):
        label = f"EIC {format_number(target.lower)}-{format_number(target.upper)}"
        detail = (
            f"MS1, {polarity_text}, m/z {format_number(target.lower)} through "
            f"{format_number(target.upper)} inclusive. ppm is not applied."
        )
        return [(label, detail, [(target.lower, target.upper)])]
    windows = [ppm_window(mass, target.ppm) for mass in target.masses]
    if len(windows) == 1 or not merge:
        planned = []
        for mass, window in zip(target.masses, windows):
            label = f"EIC {format_number(mass)} ±{format_number(target.ppm)} ppm"
            detail = (
                f"MS1, {polarity_text}, {format_number(mass)} ±{format_number(target.ppm)} ppm, "
                f"inclusive m/z {format_number(window[0])} to {format_number(window[1])}."
            )
            planned.append((label, detail, [window]))
        return planned
    shown = " + ".join(format_number(mass) for mass in target.masses[:4])
    if len(target.masses) > 4:
        shown = f"{len(target.masses)} masses"
    label = f"EIC {shown} ±{format_number(target.ppm)} ppm"
    parts = [
        f"{format_number(mass)} [{format_number(window[0])}, {format_number(window[1])}]"
        for mass, window in zip(target.masses, windows)
    ]
    detail = (
        f"MS1, {polarity_text}, merged ±{format_number(target.ppm)} ppm windows: "
        + "; ".join(parts)
        + ". A point that falls in two windows is counted in both."
    )
    return [(label, detail, windows)]


def extract_eics(
    path: str,
    scans: ScanTable,
    target: MassList | MzRange,
    polarity: str,
    merge: bool,
    trace_ids: list[int],
    colors: list[str],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    cache: ArrayCache | None = None,
) -> list[Trace]:
    rows = rows_for_polarity(scans, polarity)
    plan = trace_plan(target, polarity, merge)
    if len(trace_ids) != len(plan) or len(colors) != len(plan):
        raise ValueError("Trace ids do not match the extraction plan.")
    rt = np.asarray(scans.rt_min[rows], dtype=np.float64)
    totals = [np.zeros(rows.shape[0], dtype=np.float64) for _ in plan]
    windows = {}
    window_indices = [[windows.setdefault(window, len(windows)) for window in entry[2]] for entry in plan]
    bounds = np.asarray(list(windows), dtype=np.float64).reshape(-1, 2)
    scan_total = int(rows.shape[0])
    with SpectrumReader(path) as reader:
        read_ahead(reader, cache, path, scans.spectrum_index[rows])
        for done, row in enumerate(rows, start=1):
            if is_cancelled is not None and is_cancelled():
                raise Cancelled()
            spectrum_index = int(scans.spectrum_index[int(row)])
            mz, intensity = read_arrays(cache, path, spectrum_index, lambda: reader.read(spectrum_index))
            if is_cancelled is not None and is_cancelled():
                raise Cancelled()
            sums = _sum_windows(mz, intensity, bounds)
            for index, indices in enumerate(window_indices):
                extracted = 0.0
                for window_index in indices:
                    # Preserve input order and multiplicity for merged windows.
                    extracted += float(sums[window_index])
                totals[index][done - 1] = extracted
            if on_progress is not None and (done == scan_total or done % 10 == 0):
                on_progress(done, scan_total)
    traces = []
    for index, ((label, detail, _windows), values, trace_id, color) in enumerate(zip(plan, totals, trace_ids, colors)):
        if isinstance(target, MzRange):
            key = ("eic", polarity, "range", float(target.lower), float(target.upper))
        else:
            masses = tuple(sorted(map(float, target.masses))) if merge else (float(target.masses[index]),)
            key = ("eic", polarity, "masses", masses, float(target.ppm))
        traces.append(
            Trace(
                trace_id=trace_id,
                file_id=0,
                kind="eic",
                label=label,
                rt=rt.copy(),
                y=values,
                row=np.array(rows, copy=True),
                detail=detail,
                color=color,
                extraction_key=key,
                mz_windows=tuple(_windows),
            )
        )
    return traces


def extract_bpc(path, scans, polarity, trace_id, color, *, cache=None,
                on_progress=None, is_cancelled=None) -> Trace:
    """Highest finite native ion intensity per selected MS1 scan; empty scans are zero."""
    from kmzviewer.mz_filter import extract_filtered_values

    rows = rows_for_polarity(scans, polarity)
    polarity_text = {"all": "all polarities", "positive": "positive scans", "negative": "negative scans"}[polarity]
    trace = Trace(trace_id, 0, "bpc", "BPC MS1", scans.rt_min[rows].copy(),
                  np.zeros(len(rows)), rows, f"MS1 base peak intensity, {polarity_text}; native points, no centroiding.",
                  color, extraction_key=("bpc", polarity))
    trace.y = extract_filtered_values(path, scans, [trace], None, cache=cache,
                                     on_progress=on_progress, is_cancelled=is_cancelled)[trace_id]
    return trace
