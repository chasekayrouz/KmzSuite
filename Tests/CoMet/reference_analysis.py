"""main@0a039d5's discovery, measurement and Excel writer, kept verbatim as the
independent reference for test_analysis_exact.py, with main's mzML scan reader
(only the module boundary is new: ``quantify`` reads ``sample_scans``, which tests
replace; the writer reuses the current module's DEFINITIONS; ``iter_scans`` uses the
current ``UnitMzML`` and the suite's pinned PSI-MS release, v4.2.17: the release main
downloaded when the pin was introduced)."""
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
import os
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from kmzcore.psi_ms import vocabulary
from kmzcomet import METHOD, __version__
from kmzcomet.reader import UnitMzML, require_current_input, read_provenance
from kmzcomet.model import Scan
from kmzcomet.excel import DEFINITIONS
from kmzcomet.model import Candidate, Measurement, Settings
from kmzcomet.output import new_output


def sample_scans(sample, checkpoint):
    raise NotImplementedError("tests supply the scans")


def signal_metrics(times, values, settings: Settings):
    """Unsmooth native-height evidence. MAD of first differences estimates noise."""
    t, y = np.asarray(times), np.asarray(values)
    apex = int(np.argmax(y))
    height = float(y[apex])
    background = float(np.quantile(y, .2))
    delta = np.diff(y)
    noise = max(1., float(1.4826 * np.median(np.abs(delta - np.median(delta))) / np.sqrt(2))) if len(delta) else 1.
    above = max(0., height - background)
    threshold = background + max(2 * noise, .1 * above)
    left = right = apex
    if height > threshold:
        while left > 0 and y[left-1] > threshold and t[left]-t[left-1] <= settings.max_scan_gap:
            left -= 1
        while right+1 < len(y) and y[right+1] > threshold and t[right+1]-t[right] <= settings.max_scan_gap:
            right += 1
        support = right - left + 1
    else:
        support = 0
    contrast = height / max(background, 1.)
    snr = above / noise
    supported = (height >= settings.min_height and support >= settings.min_points and snr >= settings.min_snr and
                 contrast >= settings.min_contrast and 0 < apex < len(y)-1)
    return dict(apex=apex, height=height, background=background, noise=noise,
                height_above_background=above, snr=snr, contrast=contrast,
                support_points=support, supported=supported)


def mass_peaks(scan, threshold):
    y = scan.intensity
    if scan.mode == "centroid":
        return np.flatnonzero(y >= threshold)
    # Profile endpoints cannot establish a local maximum. Plateaus choose their
    # right edge. Native apex coordinates/heights are retained, not centroid areas.
    peaks = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]) &
                           (y[1:-1] >= threshold)) + 1
    # A shoulder ripple is not a resolved ion. Adjacent maxima need a valley
    # at or below half the smaller apex; otherwise retain the taller maximum.
    # This filters discovery only: measurement still uses every native point.
    kept = []
    for peak in peaks:
        while kept and np.min(y[kept[-1]:peak+1]) > .5*min(y[kept[-1]], y[peak]):
            if y[kept[-1]] >= y[peak]:
                break
            kept.pop()
        else:
            kept.append(peak)
    return np.asarray(kept, dtype=int)


@dataclass
class Track:
    mass: float
    polarity: int
    times: list[float] = field(default_factory=list)
    masses: list[float] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)


class Detector:
    def __init__(self, sample, settings):
        self.sample, self.settings = sample, settings
        self.active = {-1: [], 0: [], 1: []}
        self.serial = 0
        self.candidates = []

    def add(self, scan):
        s = self.settings
        active = self.active[scan.polarity]
        retained = []
        for entry in active:
            if scan.rt - entry[2].times[-1] > s.max_scan_gap:
                self.finish(entry[2])
            else:
                retained.append(entry)
        active[:] = retained
        assigned = set()
        peaks = mass_peaks(scan, s.discovery_height)
        # Highest-first makes one-to-one conflicts deterministic.
        peaks = peaks[np.argsort(-scan.intensity[peaks], kind="stable")]
        for index in peaks:
            mass, height = float(scan.mz[index]), float(scan.intensity[index])
            radius = mass * s.mz_ppm * 1e-6
            a = bisect_left(active, (mass-radius,))
            b = bisect_right(active, (mass+radius, float("inf")))
            matches = [entry for entry in active[a:b] if entry[1] not in assigned]
            if matches:
                entry = min(matches, key=lambda e: (abs(e[0]-mass), e[1]))
                track = entry[2]
            else:
                track = Track(mass, scan.polarity)
                entry = (mass, self.serial, track)
                self.serial += 1
                insort(active, entry)
            assigned.add(entry[1])
            track.times.append(scan.rt)
            track.masses.append(mass)
            track.heights.append(height)

    def finish(self, track):
        s = self.settings
        if len(track.times) < s.min_points:
            return
        times, y = np.array(track.times), np.array(track.heights)
        # Discovery gaps are below-threshold measurements, not quantitative zeroes.
        # Regular resampling is deliberately avoided. Split large scan gaps first.
        cuts = np.r_[0, np.flatnonzero(np.diff(times) > s.max_scan_gap)+1, len(y)]
        for start, stop in zip(cuts[:-1], cuts[1:]):
            if stop-start < s.min_points:
                continue
            t, v = times[start:stop], y[start:stop]
            smooth = np.convolve(np.pad(v, 1, mode="edge"), [.25, .5, .25], mode="valid")
            peaks = np.flatnonzero((smooth >= np.r_[-np.inf, smooth[:-1]]) &
                                   (smooth > np.r_[smooth[1:], -np.inf]))
            accepted = []
            for peak in sorted(peaks, key=lambda i: (-smooth[i], i)):
                window = max(s.discovery_window, s.rt_window)
                lo, hi = np.searchsorted(t, [t[peak]-window, t[peak]+window], side="left")
                local = v[lo:hi]
                if len(local) < s.min_points:
                    continue
                # Require a real rise and fall in the observed high-signal
                # track. A constant contaminant is not a chromatographic peak.
                if not lo < peak < hi-1:
                    continue
                floor = max(float(v[lo:peak+1].min()), float(v[peak:hi].min()))
                if v[peak] < s.min_contrast * max(1., floor):
                    continue
                # The same long tail can fluctuate beyond rt_tolerance. Require
                # a deep valley to every retained maximum on this mass track,
                # rather than creating another feature when a timer expires.
                if any(smooth[min(peak, old):max(peak, old)+1].min() >
                       .5*min(smooth[peak], smooth[old]) for old in accepted):
                    continue
                support = np.count_nonzero(local >= max(s.discovery_height, .1*v[peak]))
                if support < s.min_points:
                    continue
                accepted.append(int(peak))
                near = (np.abs(t-t[peak]) <= s.rt_tolerance/2) & (v >= .5*v[peak])
                mass = float(np.median(np.array(track.masses[start:stop])[near]))
                self.candidates.append(Candidate(self.sample, mass, float(t[peak]),
                                                 float(v[peak]), track.polarity))

    def complete(self):
        for active in self.active.values():
            for _, _, track in active:
                self.finish(track)
        # Fragmented tracks may converge to the same peak; collapse only within
        # one scan-scale RT distance and the configured mass tolerance.
        result = []
        for c in sorted(self.candidates, key=lambda c: (-c.height, c.mz, c.rt)):
            if not any(c.polarity == old.polarity and abs(c.rt-old.rt) <= self.settings.max_scan_gap
                       and abs(c.mz-old.mz)/old.mz*1e6 <= self.settings.mz_ppm for old in result):
                result.append(c)
        return sorted(result, key=lambda c: (c.polarity, c.mz, c.rt))


def summarize(feature, sample, sample_index, trace, settings, *, missing_mass_scans=0):
    observed = any(c.sample == sample_index for c in feature.members)
    result = Measurement(feature.id, sample.id, observed, "not_acquired")
    if not trace:
        result.flags = "No same-polarity MS1 scans covering this m/z–RT window"
        return result
    times, values, masses, known = map(np.array, zip(*trace))
    metrics = signal_metrics(times, values, settings)
    apex = metrics.pop("apex")
    supported = metrics.pop("supported")
    shift = sample.shifts.get(feature.polarity, 0.)
    spacing = float(np.median(np.diff(times))) if len(times) > 1 else settings.max_scan_gap
    flags = []
    # Allow one normal sampling interval at each window edge. Missing mass
    # coverage or acquisition gaps cannot be disguised by trapezoidal bridging.
    if len(times) < settings.min_points:
        flags.append("too_few_scans")
    if times[0] > feature.left+shift+spacing*1.5 or times[-1] < feature.right+shift-spacing*1.5:
        flags.append("partial_rt_window")
    gaps = np.diff(times) > settings.max_scan_gap
    if np.any(gaps):
        flags.append("acquisition_gap")
    if not np.all(known):
        flags.append("mass_coverage_unknown")
    if missing_mass_scans:
        flags.append("partial_mass_coverage")
    if feature.ambiguous:
        flags.append("overlapping_feature_windows")
    if apex in (0, len(values)-1) and metrics["height"] > 0:
        flags.append("apex_at_window_edge")
    if metrics["height"] > 0 and abs(times[apex]-shift-feature.rt) > settings.rt_tolerance:
        flags.append("apex_outside_rt_tolerance")
        supported = False
    if 0 < metrics["height"] < settings.min_height:
        flags.append("below_evidence_height")
    background_subtracted = np.maximum(0., values-metrics["background"])
    area = float(np.sum(.5*(background_subtracted[1:]+background_subtracted[:-1]) *
                        np.diff(times) * ~gaps))
    if flags and any(f in flags for f in ("too_few_scans", "partial_rt_window", "acquisition_gap", "mass_coverage_unknown", "partial_mass_coverage")):
        status = "insufficient_coverage"
    elif feature.ambiguous:
        status = "ambiguous"
    elif metrics["height"] == 0:
        status = "no_signal"
    elif supported:
        status = "supported"
    else:
        status = "weak_signal"
    return replace(result, status=status, **metrics, points=len(times), window_area=area,
                   apex_rt=float(times[apex]) if metrics["height"] else None,
                   apex_mz=float(masses[apex]) if metrics["height"] else None,
                   flags="; ".join(flags))


def quantify(sample, sample_index, features, settings, progress, checkpoint=lambda: None):
    traces = [[] for _ in features]
    missing_mass = [0 for _ in features]
    ordered = {}
    for polarity in (-1, 0, 1):
        ordered[polarity] = sorted([i for i, f in enumerate(features) if f.polarity == polarity],
                                   key=lambda i: features[i].left)
    next_index = defaultdict(int)
    active = defaultdict(list)
    for scan_index, scan in enumerate(sample_scans(sample, checkpoint)):
        checkpoint()
        if scan_index % 250 == 0:
            progress(f"Measure {sample.id}: {scan_index}/{sample.scans} scans")
        shift = sample.shifts.get(scan.polarity, 0.)
        rt = scan.rt-shift
        queue = ordered[scan.polarity]
        cursor = next_index[scan.polarity]
        while cursor < len(queue) and features[queue[cursor]].left <= rt:
            active[scan.polarity].append(queue[cursor])
            cursor += 1
        next_index[scan.polarity] = cursor
        active[scan.polarity] = [i for i in active[scan.polarity] if rt < features[i].right]
        selected = active[scan.polarity]
        if not selected:
            continue
        masses = np.array([features[i].mz for i in selected])
        lo, hi = masses*(1-settings.mz_ppm*1e-6), masses*(1+settings.mz_ppm*1e-6)
        starts, stops = np.searchsorted(scan.mz, lo), np.searchsorted(scan.mz, hi, side="right")
        for i, lower, upper, a, b in zip(selected, lo, hi, starts, stops):
            if scan.windows and not any(left <= lower and upper <= right for left, right in scan.windows):
                missing_mass[i] += 1
                continue
            if a < b:
                apex = a + int(np.argmax(scan.intensity[a:b]))
                value, mass = float(scan.intensity[apex]), float(scan.mz[apex])
            else:
                value, mass = 0., float("nan")
            traces[i].append((scan.rt, value, mass, bool(scan.windows)))
    return [summarize(f, sample, sample_index, trace, settings, missing_mass_scans=missing)
            for f, trace, missing in zip(features, traces, missing_mass)]


def write_workbook(result, destination, *, checkpoint=lambda: None):
    checkpoint()
    destination = Path(destination)
    if destination.suffix.lower() != ".xlsx":
        raise ValueError("Output must have an .xlsx extension")
    if os.path.lexists(destination):
        raise FileExistsError(f"Output already exists: {destination.name}")
    if len(result.features)*len(result.samples)+1 > 1048576 or len(result.samples)+4 > 16384:
        raise ValueError("Result exceeds Excel worksheet limits")
    destination.parent.mkdir(parents=True, exist_ok=True)
    book = Workbook(write_only=True)
    book.properties.creator = "KmzCoMet"
    book.properties.title = "Comparative MS1 ion features"
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Arial", size=10, color="172234")
    header_fill = PatternFill("solid", fgColor="243B57")

    def sheet(name, headers, widths, freeze):
        ws = book.create_sheet(name)
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = freeze
        ws.row_dimensions[1].height = 32
        for i, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
        cells = []
        for value in headers:
            cell = WriteOnlyCell(ws, value=value)
            cell.data_type = "s"  # Untrusted filenames must never become formulas.
            cell.font, cell.fill = header_font, header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cells.append(cell)
        ws.append(cells)
        return ws

    def append(ws, values, formats=None, *, wrap=False):
        cells = []
        for i, value in enumerate(values):
            cell = WriteOnlyCell(ws, value=value)
            if isinstance(value, str):
                cell.data_type = "s"
            cell.font = body_font
            cell.alignment = Alignment(vertical="center", wrap_text=wrap,
                                       horizontal="right" if isinstance(value, (float, int)) else "left")
            if formats and i < len(formats):
                cell.number_format = formats[i]
            cells.append(cell)
        ws.append(cells)

    headers = ["Feature ID", "m/z", "RT (min)", "Polarity"] + [s.id for s in result.samples]
    widths = [16, 16, 15, 13] + [20]*len(result.samples)
    height = sheet("Heights", headers, widths, "E2")
    evidence = sheet("Evidence", headers, widths[:4]+[25]*len(result.samples), "E2")
    height.sheet_properties.tabColor = "243B57"
    for i, feature in enumerate(result.features):
        common = [feature.id, feature.mz, feature.rt, {1:"positive",-1:"negative",0:"unknown"}[feature.polarity]]
        append(height, common+[column[i].height for column in result.measurements],
               ["@", "0.000000", "0.0000", "@"]+["0.00E+00"]*len(result.samples))
        append(evidence, common+[column[i].status for column in result.measurements],
               ["@", "0.000000", "0.0000", "@"])
    for ws in (height, evidence):
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(result.features)+1}"
    names = [f.name for f in fields(Measurement)]
    labels = ["Feature ID", "Sample ID", "Discovered", "Evidence", "Height", "Height above background",
              "Window area (intensity × min)", "Native apex RT (min)", "Apex m/z", "Background", "Noise", "SNR",
              "Contrast", "Scans", "Support scans", "Flags"]
    details = sheet("Measurements", labels, [16,14,14,25]+[20]*9+[12,15,48], "C2")
    formats = ["@", "@", "General", "@", "0.00E+00", "0.00E+00", "0.00E+00",
               "0.0000", "0.000000", "0.00E+00", "0.00E+00", "0.0", "0.0", "0", "0", "@"]
    for column in result.measurements:
        for measurement in column:
            append(details, [getattr(measurement, name) for name in names], formats)
    details.auto_filter.ref = f"A1:P{len(result.features)*len(result.samples)+1}"
    run = sheet("Run", ["Record", "Field", "Value"], [16,29,110], "A2")
    rows = [("Analysis", "Method", result.method), ("Analysis", "Suite version", result.suite_version or __version__),
            ("Analysis", "Created UTC", result.created_utc),
            ("Analysis", "Analysis ID", result.analysis_id),
            ("Analysis", "Elapsed seconds", result.elapsed_seconds),
            ("Analysis", "Reference sample", result.samples[0].id),
            ("Analysis", "Feature count", len(result.features))]
    rows.extend(("Settings", name, value) for name, value in asdict(result.settings).items())
    for sample in result.samples:
        for name, value in [("Filename", sample.path.name), ("SHA-256", sample.sha256),
                            ("Size bytes", sample.size_bytes), ("MS1 scans", sample.scans),
                            ("Representation", ", ".join(sorted(sample.modes))),
                            ("Intensity unit", ", ".join(sorted(sample.units))),
                            ("RT start (min)", sample.rt_min), ("RT end (min)", sample.rt_max),
                            ("Discovery candidates", len(sample.candidates)),
                            ("Warnings", "; ".join(sample.warnings) or "None")]:
            rows.append((sample.id, name, value))
        if sample.native_sample_id is not None:
            rows.append((sample.id, 'Stored WIFF sample ID', sample.native_sample_id))
        for source in sample.source_files:
            if source['path'] == str(sample.path):
                continue  # Primary filename/size/digest are already recorded.
            rows.append((sample.id, 'Source file', source['name']))
            rows.append((sample.id, 'Source SHA-256', source['sha256']))
            rows.append((sample.id, 'Source bytes', source['size_bytes']))
        for polarity, shift in sorted(sample.shifts.items()):
            rows.extend([(sample.id, f"RT shift, polarity {polarity} (min)", shift),
                         (sample.id, f"Anchors, polarity {polarity}", sample.anchors[polarity]),
                         (sample.id, f"Anchor MAD, polarity {polarity} (min)", sample.alignment_mad[polarity])])
    rows.extend(("Definitions", name, text) for name, text in DEFINITIONS.items())
    for number, row in enumerate(rows, 2):
        run.row_dimensions[number].height = max(30, 15*(1+len(str(row[2]))//100))
        append(run, row, wrap=True)
    try:
        with new_output(destination, checkpoint=checkpoint) as temporary:
            book.save(temporary)
    finally:
        book.close()
    return destination


def iter_scans(path: Path):
    require_current_input(path)
    if read_provenance(path).methods:
        raise ValueError("Sparse Profile input is not qualified for KmzCoMet; use Full Profile")
    previous = {}
    with UnitMzML(str(path), use_index=False, iterative=True, cv=vocabulary()) as reader:
        for spectrum in reader:
            if spectrum.get("ms level") != 1:
                continue
            ident = spectrum.get("id", "unknown scan")
            scans = spectrum.get("scanList", {}).get("scan", [])
            if len(scans) != 1:
                raise ValueError(f"{ident}: require exactly one MS1 scan/time entry")
            scan = scans[0]
            rt = scan.get("scan start time")
            unit = str(getattr(rt, "unit_info", "")).lower()
            if unit in {"minute", "minutes", "uo:0000031"}:
                rt = float(rt)
            elif unit in {"second", "seconds", "uo:0000010"}:
                rt = float(rt) / 60
            else:
                raise ValueError(f"{ident}: missing/unsupported RT unit {unit!r}")
            if not np.isfinite(rt) or rt < 0:
                raise ValueError(f"{ident}: invalid RT")
            positive, negative = "positive scan" in spectrum, "negative scan" in spectrum
            if positive and negative:
                raise ValueError(f"{ident}: conflicting scan polarities")
            polarity = int(positive) - int(negative)
            if rt <= previous.get(polarity, -1):
                raise ValueError(f"{ident}: MS1 times must strictly increase within each polarity")
            previous[polarity] = rt
            profile, centroid = "profile spectrum" in spectrum, "centroid spectrum" in spectrum
            if profile == centroid:
                raise ValueError(f"{ident}: require explicit profile or centroid spectrum metadata")
            if "m/z array" not in spectrum or "intensity array" not in spectrum:
                raise ValueError(f"{ident}: missing MS1 arrays")
            m = np.asarray(spectrum["m/z array"], dtype=float)
            y = np.asarray(spectrum["intensity array"], dtype=float)
            if (m.ndim != 1 or m.shape != y.shape or not np.isfinite(m).all() or
                    not np.isfinite(y).all() or np.any(m <= 0) or
                    np.any(np.diff(m) <= 0) or np.any(y < 0)):
                raise ValueError(f"{ident}: require ordered finite masses and nonnegative intensities")
            if int(spectrum.get("defaultArrayLength", len(m))) != len(m):
                raise ValueError(f"{ident}: array length disagrees with metadata")
            windows = []
            for window in scan.get("scanWindowList", {}).get("scanWindow", []):
                lo = float(window.get("scan window lower limit", float("nan")))
                hi = float(window.get("scan window upper limit", float("nan")))
                if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
                    raise ValueError(f"{ident}: invalid acquisition mass window")
                windows.append((lo, hi))
            yield Scan(rt, polarity, "profile" if profile else "centroid", m, y,
                       tuple(windows), spectrum.get("intensity unit", "unknown"))
