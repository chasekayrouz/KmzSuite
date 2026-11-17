"""Two streaming passes: discover once, then measure every feature in every file."""
from contextlib import closing
import hashlib
from pathlib import Path
import time

import numpy as np

from .alignment import estimate_shifts, group_features
from .detection import Detector, trace_metrics
from .model import Measurement, Result, Sample, Settings
from .reader import iter_scans
from .sources import InputSource, native_scans
from . import METHOD, __version__


HASHERS = 4


def file_hash(path, checkpoint=lambda: None):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        while block := handle.read(1024*1024):
            checkpoint()
            digest.update(block)
        return digest.hexdigest()


def sample_scans(sample, checkpoint):
    if sample.path.suffix.lower() in {'.wiff', '.wiff2'}:
        return native_scans(InputSource(sample.path, sample.native_sample_id), checkpoint)
    return iter_scans(sample.path)


def unacquired(feature, sample, sample_index):
    observed = any(c.sample == sample_index for c in feature.members)
    return Measurement(feature.id, sample.id, observed, "not_acquired",
                       flags="No same-polarity MS1 scans covering this m/z–RT window")


def summarize(feature, sample, sample_index, trace, settings, *, missing_mass_scans=0):
    if not trace:
        return unacquired(feature, sample, sample_index)
    times, values, masses, known = (np.array(column)[None] for column in zip(*trace))
    return summarize_traces([feature], sample, sample_index, times, values, masses, known, settings,
                            [missing_mass_scans])[0]


def summarize_traces(features, sample, sample_index, times, values, masses, known, settings, missing):
    """``summarize`` of features whose traces share one length: rows of 2-D
    (times, values, masses, known) arrays, with each row's missing-mass count."""
    count = times.shape[1]
    metrics = trace_metrics(times, values, settings)
    steps = np.diff(times, axis=1)
    spacings = np.median(steps, axis=1).tolist() if count > 1 else [settings.max_scan_gap]*len(features)
    gaps = steps > settings.max_scan_gap
    background = np.array([metric["background"] for metric in metrics])
    subtracted = np.maximum(0., values-background[:, None])
    areas = .5*(subtracted[:, 1:]+subtracted[:, :-1]) * steps * ~gaps
    results = []
    for row, (feature, metric, spacing, gapped, complete, missing_mass_scans) in enumerate(zip(
            features, metrics, spacings, np.any(gaps, axis=1).tolist(), np.all(known, axis=1).tolist(), missing)):
        observed = any(c.sample == sample_index for c in feature.members)
        apex = metric.pop("apex")
        supported = metric.pop("supported")
        shift = sample.shifts.get(feature.polarity, 0.)
        flags = []
        # Allow one normal sampling interval at each window edge. Missing mass
        # coverage or acquisition gaps cannot be disguised by trapezoidal bridging.
        if count < settings.min_points:
            flags.append("too_few_scans")
        if times[row, 0] > feature.left+shift+spacing*1.5 or times[row, -1] < feature.right+shift-spacing*1.5:
            flags.append("partial_rt_window")
        if gapped:
            flags.append("acquisition_gap")
        if not complete:
            flags.append("mass_coverage_unknown")
        if missing_mass_scans:
            flags.append("partial_mass_coverage")
        if feature.ambiguous:
            flags.append("overlapping_feature_windows")
        if apex in (0, count-1) and metric["height"] > 0:
            flags.append("apex_at_window_edge")
        if metric["height"] > 0 and abs(times[row, apex]-shift-feature.rt) > settings.rt_tolerance:
            flags.append("apex_outside_rt_tolerance")
            supported = False
        if 0 < metric["height"] < settings.min_height:
            flags.append("below_evidence_height")
        if flags and any(f in flags for f in ("too_few_scans", "partial_rt_window", "acquisition_gap", "mass_coverage_unknown", "partial_mass_coverage")):
            status = "insufficient_coverage"
        elif feature.ambiguous:
            status = "ambiguous"
        elif metric["height"] == 0:
            status = "no_signal"
        elif supported:
            status = "supported"
        else:
            status = "weak_signal"
        results.append(Measurement(feature.id, sample.id, observed, status, **metric, points=count,
                                   window_area=float(np.sum(areas[row])),
                                   apex_rt=float(times[row, apex]) if metric["height"] else None,
                                   apex_mz=float(masses[row, apex]) if metric["height"] else None,
                                   flags="; ".join(flags)))
    return results


def window_maxima(mz, intensity, starts, stops):
    """Each [start, stop) window's first maximal intensity (np.argmax) and its m/z;
    an empty window gives 0 and NaN."""
    values, masses = np.zeros(len(starts)), np.full(len(starts), np.nan)
    filled = np.flatnonzero(starts < stops)
    if len(filled):
        first = starts[filled]
        lengths = stops[filled] - first
        offsets = np.cumsum(lengths) - lengths
        local = np.arange(int(offsets[-1] + lengths[-1])) - np.repeat(offsets, lengths)
        points = intensity[np.repeat(first, lengths) + local]
        top = np.repeat(np.maximum.reduceat(points, offsets), lengths)
        apex = first + np.minimum.reduceat(np.where(points == top, local, len(local)), offsets)
        values[filled], masses[filled] = intensity[apex], mz[apex]
    return values, masses


def quantify(sample, sample_index, features, settings, progress, checkpoint=lambda: None):
    # RTs strictly increase within a polarity, so each feature measures the
    # consecutive scans with left <= aligned RT < right: features join in order
    # of left and leave once the aligned RT reaches right.
    mz = np.array([f.mz for f in features], dtype=float)
    lefts = np.array([f.left for f in features], dtype=float)
    rights = np.array([f.right for f in features], dtype=float)
    polarities = np.array([f.polarity for f in features], dtype=int)
    lower, upper = mz*(1-settings.mz_ppm*1e-6), mz*(1+settings.mz_ppm*1e-6)
    queues, cursors, active = {}, {}, {}
    for polarity in (-1, 0, 1):
        members = np.flatnonzero(polarities == polarity)
        members = members[np.argsort(lefts[members], kind="stable")]
        queues[polarity], cursors[polarity] = (members, lefts[members]), 0
        active[polarity] = members[:0]
    missing_mass = np.zeros(len(features), dtype=int)
    found, scans, values, masses = [], [], [], []
    for scan_index, scan in enumerate(sample_scans(sample, checkpoint)):
        checkpoint()
        if scan_index % 250 == 0:
            progress(f"Measure {sample.id}: {scan_index}/{sample.scans} scans")
        shift = sample.shifts.get(scan.polarity, 0.)
        rt = scan.rt-shift
        members, starts = queues[scan.polarity]
        cursor = cursors[scan.polarity]
        joined = max(cursor, int(np.searchsorted(starts, rt, side="right")))
        selected = np.concatenate((active[scan.polarity], members[cursor:joined]))
        cursors[scan.polarity] = joined
        selected = active[scan.polarity] = selected[rt < rights[selected]]
        if not len(selected):
            continue
        lo, hi = lower[selected], upper[selected]
        if scan.windows:
            covered = np.zeros(len(selected), dtype=bool)
            for left, right in scan.windows:
                covered |= (left <= lo) & (hi <= right)
            missing_mass[selected[~covered]] += 1
            selected, lo, hi = selected[covered], lo[covered], hi[covered]
        value, mass = window_maxima(scan.mz, scan.intensity, np.searchsorted(scan.mz, lo),
                                    np.searchsorted(scan.mz, hi, side="right"))
        found.append(selected)
        scans.append((scan.rt, bool(scan.windows), len(selected)))
        values.append(value)
        masses.append(mass)
    # One trace per feature in scan order: (times, values, masses, known).
    found = np.concatenate(found) if found else np.zeros(0, dtype=int)
    order = np.argsort(found, kind="stable")
    times, known, counts = zip(*scans) if scans else ((), (), ())
    columns = [np.repeat(np.array(times, dtype=float), counts)[order],
               np.concatenate(values)[order] if values else np.zeros(0),
               np.concatenate(masses)[order] if masses else np.zeros(0),
               np.repeat(np.array(known, dtype=bool), counts)[order]]
    bounds = np.searchsorted(found[order], np.arange(len(features)+1))
    lengths = np.diff(bounds)
    results = [None] * len(features)
    for count in np.unique(lengths).tolist():
        rows = np.flatnonzero(lengths == count).tolist()
        if not count:
            for i in rows:
                results[i] = unacquired(features[i], sample, sample_index)
            continue
        index = bounds[rows][:, None] + np.arange(count)
        measured = summarize_traces([features[i] for i in rows], sample, sample_index,
                                    *(column[index] for column in columns), settings,
                                    missing_mass[rows].tolist())
        for i, measurement in zip(rows, measured):
            results[i] = measurement
    return results


def analyze(paths, settings=None, *, progress=lambda message: None, checkpoint=lambda: None):
    settings = settings or Settings()
    start = time.monotonic()
    sources = []
    for value in paths:
        checkpoint()
        sources.append((value if isinstance(value, InputSource) else InputSource(Path(value))).resolved())
    if not sources:
        raise ValueError("At least one mzML or WIFF input is required")
    identities = [source.identity() for source in sources]
    if len(set(identities)) != len(paths):
        raise ValueError("Duplicate input file (including aliases/hard links)")
    samples = []
    fingerprints = {}
    for i, source in enumerate(sources):
        checkpoint()
        path = source.path
        progress(f"Discover S{i+1:03d}")
        for item in source.files():
            if item not in fingerprints:
                fingerprints[item] = dict(path=str(item), name=item.name, size_bytes=item.stat().st_size,
                                          sha256=file_hash(item, checkpoint))
        sample = Sample(path, f"S{i+1:03d}", fingerprints[path]['sha256'], path.stat().st_size,
                        native_sample_id=source.sample_id,
                        source_files=[dict(fingerprints[item]) for item in source.files()])
        detector = Detector(i, settings)
        polarities = set()
        for scan in sample_scans(sample, checkpoint):
            checkpoint()
            sample.scans += 1
            sample.rt_min = min(sample.rt_min, scan.rt)
            sample.rt_max = max(sample.rt_max, scan.rt)
            sample.modes.add(scan.mode)
            sample.units.add(scan.intensity_unit)
            polarities.add(scan.polarity)
            detector.add(scan)
            if sample.scans % 250 == 0:
                progress(f"Discover {sample.id}: {sample.scans} scans")
        if not sample.scans:
            raise ValueError(f"{sample.id}: no MS1 scans")
        if 0 in polarities:
            sample.warnings.append("Unknown-polarity scans remain a separate group")
        if "unknown" in sample.units:
            sample.warnings.append("Intensity units were not declared in all MS1 scans")
        sample.candidates = detector.complete()
        if not sample.candidates:
            sample.warnings.append("No discovery candidates; measurements from other files are still attempted")
        samples.append(sample)
    if len(set.union(*(s.modes for s in samples))) != 1:
        raise ValueError("Mixed profile/centroid intensity comparisons are not qualified; analyze them separately")
    if len(set.union(*(s.units for s in samples))) != 1:
        raise ValueError("MS1 intensity units differ within/across inputs; cannot compare native heights")
    progress("Align candidates")
    checkpoint()
    estimate_shifts(samples, settings)
    features = group_features(samples, settings)
    checkpoint()
    # Fail before creating a partial workbook. Excel details use one row per pair.
    if len(samples)+4 > 16384 or len(features)*len(samples)+1 > 1048576:
        raise ValueError("Result exceeds Excel worksheet limits; use fewer files or a stricter discovery threshold")
    measurements = [quantify(sample, i, features, settings, progress, checkpoint) for i, sample in enumerate(samples)]
    for source, sample in zip(sources, samples):
        if [str(p) for p in source.files()] != [f['path'] for f in sample.source_files]:
            raise ValueError('Input sidecar set changed during analysis; result discarded')
    # Digests are computed ahead on helper threads (hashlib releases the GIL);
    # each file is still checked, and any failure raised, in this order.
    from kmzconvert_core.native import ordered
    with closing(ordered(lambda path: file_hash(path, checkpoint), list(fingerprints), HASHERS)) as digests:
        for path, before in fingerprints.items():
            checkpoint()
            if path.stat().st_size != before['size_bytes'] or next(digests) != before['sha256']:
                raise ValueError(f"{path.name}: input changed during analysis; result discarded")
    return Result(settings, samples, features, measurements, time.monotonic()-start,
                  method=METHOD, suite_version=__version__)
