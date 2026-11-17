"""Reference-guided MS1 calibration: tested full-profile detection, constant ppm only.

Native reader guards remain authoritative. No production import of benchmark code.
"""
from bisect import bisect_left, bisect_right, insort
from contextlib import closing
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from kmzcore.mass_correction import ESTIMATOR, WARNING, provenance_params
from kmzcore.qualified import qualified

from . import native


@dataclass(frozen=True)
class Reference:
    id: str
    expected_mz: float
    expected_rt: float

    def __post_init__(self):
        if not self.id or not np.isfinite([self.expected_mz, self.expected_rt]).all():
            raise ValueError("Reference needs an ID and finite mass/RT")
        if self.expected_mz <= 0 or self.expected_rt < 0:
            raise ValueError("Reference mass must be positive and RT nonnegative")

    @property
    def pair(self):
        return f"{self.expected_mz:.10g};{self.expected_rt:.10g}"


def parse_references(entries):
    references, seen = [], set()
    for entry in entries:
        try:
            parts = entry.strip().split(";")
            if len(parts) != 2 or any(not p.strip() for p in parts):
                raise ValueError("use expected_mz;RT_minutes")
            mz, rt = (float(p) for p in parts)
            ref = Reference(f"reference-{len(references)+1}", mz, rt)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid mass reference {entry!r}: {error}") from error
        pair = (ref.expected_mz, ref.expected_rt)
        if pair in seen:
            raise ValueError(f"Duplicate mass reference {ref.pair}")
        seen.add(pair)
        references.append(ref)
    if not references:
        raise ValueError("Enabled mass correction requires at least one expected_mz;RT_minutes reference")
    return references


@dataclass(frozen=True)
class SearchSettings:
    rt_radius: float = .45
    broad_ppm: float = 1000.
    track_ppm: float = 40.
    lobe_ppm: float = 100.
    high_fraction: float = .5
    minimum_high_scans: int = 3
    minimum_prominence_ratio: float = 3.
    consensus_sd_ppm: float = 150.
    ambiguity_margin: float = .15
    rt_score_weight: float = 2.
    precision_score_scale_ppm: float = 10.

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in self.__dict__.values()):
            raise ValueError("Search settings must be positive and finite")
        if self.track_ppm >= self.broad_ppm or self.high_fraction > 1 or self.minimum_high_scans < 2:
            raise ValueError("Invalid search settings")


@dataclass
class ProfileScan:
    id: str
    rt: float
    mz: np.ndarray
    intensity: np.ndarray

    def __post_init__(self):
        if (self.mz.ndim != 1 or self.intensity.shape != self.mz.shape or
                not np.isfinite(self.mz).all() or not np.isfinite(self.intensity).all() or
                not np.isfinite(self.rt) or np.any(np.diff(self.mz) <= 0) or np.any(self.intensity < 0)):
            raise ValueError("Profile scans require finite ordered masses and aligned nonnegative counts")


def local_mass(mz, intensity, seed_mz, settings, *, interpolate=False):
    """Bound one lobe by minima/1%-height endpoints within a finite mass radius.

    Centroid is sum(mz*counts)/sum(counts), without smoothing or background
    subtraction. Three-point quadratic uses actual nonuniform mass coordinates.
    Failure/edge/truncated-lobe flags accompany every measurement.
    """
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    low, high = seed_mz * (1 + np.array([-1, 1]) * settings.track_ppm * 1e-6)
    a, b = np.searchsorted(mz, [low, high], side="left")
    if a == b or not np.any(intensity[a:b] > 0):
        return None
    return _lobe(mz, intensity, mz, intensity, a + int(np.argmax(intensity[a:b])), seed_mz, settings, interpolate)


def _scan_mass(scan, values, seed_mz, settings, interpolate):
    """``local_mass`` for a ProfileScan (finite, increasing, nonnegative) and its ``tolist()`` values.

    Python floats repeat NumPy's scalar operations exactly: the same window
    bounds, insertion points, first maximum and walk.
    """
    mz, intensity = values
    low, high = (seed_mz * (1 + sign * settings.track_ppm * 1e-6) for sign in (-1, 1))
    a, b = bisect_left(mz, low), bisect_left(mz, high)
    if a == b:
        return None
    window = intensity[a:b]
    top = max(window)
    if not top > 0:
        return None
    return _lobe(scan.mz, scan.intensity, mz, intensity, a + window.index(top), seed_mz, settings, interpolate)


def _lobe(array_mz, array_intensity, mz, intensity, peak, seed_mz, settings, interpolate):
    # mz/intensity index like the arrays (themselves, or their tolist() values).
    height = float(intensity[peak])
    # A window ending on the shoulder of a larger peak is not a new ion.
    if (peak == 0 or peak == len(mz)-1 or
            intensity[peak] < max(intensity[peak-1], intensity[peak+1])):
        return None
    bounds = []
    truncated = False
    for direction in (-1, 1):
        j = peak
        while 0 <= j + direction < len(mz):
            nxt = j + direction
            if abs(mz[nxt]-mz[peak]) / mz[peak] * 1e6 > settings.lobe_ppm:
                truncated = True
                break
            if intensity[nxt] > intensity[j] and j != peak and intensity[j] < height * .2:
                break
            j = nxt
            if intensity[j] <= height * .01:
                break
        if j in (0, len(mz)-1) and intensity[j] > height * .01:
            truncated = True
        bounds.append(j)
    left, right = bounds
    weights = array_intensity[left:right+1]
    total = weights.sum()
    centroid = float(np.dot(array_mz[left:right+1], weights)/total)
    parabolic = None
    if interpolate and 0 < peak < len(mz)-1 and intensity[peak-1] > 0 and intensity[peak+1] > 0:
        x = array_mz[peak-1:peak+2] - array_mz[peak]
        c2, c1, _ = np.linalg.solve(np.column_stack([x*x, x, np.ones(3)]), array_intensity[peak-1:peak+2])
        if c2 < 0:
            vertex = -c1/(2*c2)
            if x[0] <= vertex <= x[-1]:
                parabolic = float(array_mz[peak] + vertex)
    return dict(sampled_mz=float(mz[peak]), centroid_mz=centroid, parabola_mz=parabolic,
                height=height, sum_intensity=float(total), left_mz=float(mz[left]),
                right_mz=float(mz[right]), points=int(right-left+1), truncated=truncated,
                seed_distance_ppm=float((mz[peak]-seed_mz)/seed_mz*1e6))


def _median(values):
    """``float(np.median(values))`` of finite floats: the middle value, or NumPy's mean of the middle two."""
    values = sorted(values)
    half = len(values) // 2
    return values[half] if len(values) % 2 else (values[half-1] + values[half]) / 2


def _numpy_candidates(reference, scans, settings, *, interpolate=False, cancel_check=lambda: None):
    """The original NumPy-scalar implementation (exact for every accepted dtype)."""
    selected = [s for s in scans if abs(s.rt-reference.expected_rt) <= settings.rt_radius]
    if len(selected) < settings.minimum_high_scans:
        return [], dict(failure="Too few MS1 scans in the RT window")
    # Local spectral maxima vote in a fixed coarse ppm histogram. This finds
    # mass tracks before chromatographic scoring; a strong unrelated ion does
    # not automatically become the assignment.
    bin_ppm = settings.track_ppm / 2
    votes = {}
    for scan in selected:
        cancel_check()
        m, y = scan.mz, scan.intensity
        peaks = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0)) + 1
        error = (m[peaks]-reference.expected_mz)/reference.expected_mz*1e6
        for p, e in zip(peaks, error):
            if abs(e) <= settings.broad_ppm:
                key = int(np.floor(e/bin_ppm))
                votes[key] = votes.get(key, 0.) + float(y[p])
    proposals = sorted(votes, key=lambda k: (-votes[k], k))[:48]
    records = []
    rt = np.array([s.rt for s in selected])
    for key in proposals:
        cancel_check()
        seed = reference.expected_mz*(1+(key+.5)*bin_ppm*1e-6)
        measurements = []
        for s in selected:
            cancel_check()
            measurements.append(local_mass(s.mz, s.intensity, seed, settings, interpolate=interpolate))
        trace = np.array([r["sum_intensity"] if r else 0. for r in measurements])
        # Every local chromatographic maximum is considered, not just the
        # largest EIC in a broad window. Endpoints are retained as failed
        # candidates so an RT-window-clipped peak is visible.
        apices = np.flatnonzero((trace[1:-1] >= trace[:-2]) & (trace[1:-1] > trace[2:])) + 1
        apices = sorted(np.unique(np.r_[apices, np.argmax(trace)]), key=lambda i: -trace[i])
        for apex in apices:
            r = measurements[apex]
            if not r or trace[apex] <= 0:
                continue
            lo = hi = int(apex)
            while lo > 0 and trace[lo-1] >= settings.high_fraction*trace[apex]:
                lo -= 1
            while hi+1 < len(trace) and trace[hi+1] >= settings.high_fraction*trace[apex]:
                hi += 1
            floor = float(np.percentile(trace, 20))
            contrast = float(trace[apex]/max(floor, 1.))
            high = list(range(lo, hi+1))
            mass = float(np.median([measurements[i]["centroid_mz"] for i in high if measurements[i]]))
            if any(abs(mass-c["observed_mz"])/mass*1e6 < bin_ppm and
                   (abs(rt[apex]-c["rt"]) < .03 or set(high).intersection(c["high_indices"])) for c in records):
                continue
            flags = []
            if lo == 0 or hi == len(trace)-1:
                flags.append("chromatographic peak reaches RT-window edge")
            if len(high) < settings.minimum_high_scans:
                flags.append("too few high-intensity scans")
            if contrast < settings.minimum_prominence_ratio:
                flags.append("weak chromatographic contrast")
            if any(measurements[i] and measurements[i]["truncated"] for i in high):
                flags.append("truncated/overlapping local spectral lobe")
            dispersion = float(np.std([measurements[i]["centroid_mz"] for i in high], ddof=1)/reference.expected_mz*1e6) if len(high)>1 else None
            # No absolute intensity comparison across compounds. RT distance,
            # chromatographic contrast and repeated mass precision are evidence.
            precision_penalty = min((dispersion/settings.precision_score_scale_ppm)**2, 2) if dispersion is not None else 10.
            score = (settings.rt_score_weight*(1-abs(rt[apex]-reference.expected_rt)/settings.rt_radius)
                     + min(np.log10(max(contrast, 1)), 2) + min(len(high)/5, 1) - precision_penalty)
            records.append(dict(reference_id=reference.id, seed_mz=seed, observed_mz=mass, rt=float(rt[apex]),
                                ppm_error=float((mass-reference.expected_mz)/reference.expected_mz*1e6),
                                da_error=mass-reference.expected_mz, apex_id=selected[apex].id,
                                apex_index=int(apex), high_indices=high, intensity=float(trace[apex]),
                                contrast=contrast, centroid_sd_ppm=dispersion, score=float(score), flags=flags,
                                trace=trace.tolist(), measurements=measurements))
    records.sort(key=lambda c: (bool(c["flags"]), -c["score"], c["rt"], c["observed_mz"]))
    for i, c in enumerate(records):
        c["candidate_id"] = f"{reference.id}-c{i+1}"
    return records, dict(scan_ids=[s.id for s in selected], rt=rt.tolist(), broad_ppm=settings.broad_ppm)


BEAM_WIDTH = 256
MAX_CANDIDATES = 12


def candidates(reference, scans, settings, *, interpolate=False, cancel_check=lambda: None):
    selected = [s for s in scans if abs(s.rt-reference.expected_rt) <= settings.rt_radius]
    # The Python-float fast path repeats NumPy scalar arithmetic (and np.median) exactly
    # only for float64 and the qualified NumPy.
    if not qualified("numpy") or any(s.mz.dtype != np.float64 or s.intensity.dtype != np.float64 for s in selected):
        return _numpy_candidates(reference, scans, settings, interpolate=interpolate, cancel_check=cancel_check)
    if len(selected) < settings.minimum_high_scans:
        return [], dict(failure="Too few MS1 scans in the RT window")
    # Local spectral maxima vote in a fixed coarse ppm histogram. This finds
    # mass tracks before chromatographic scoring; a strong unrelated ion does
    # not automatically become the assignment.
    bin_ppm = settings.track_ppm / 2
    votes = {}
    for scan in selected:
        cancel_check()
        m, y = scan.mz, scan.intensity
        peaks = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > 0)) + 1
        error = (m[peaks]-reference.expected_mz)/reference.expected_mz*1e6
        for p, e in zip(y[peaks].tolist(), error.tolist()):
            if abs(e) <= settings.broad_ppm:
                key = math.floor(e/bin_ppm)
                votes[key] = votes.get(key, 0.) + p
    proposals = sorted(votes, key=lambda k: (-votes[k], k))[:48]
    records = []
    observed = []  # (observed_mz, record index) in mass order, for the duplicate test
    rt = np.array([s.rt for s in selected])
    values = [(s.mz.tolist(), s.intensity.tolist()) for s in selected]
    for key in proposals:
        cancel_check()
        seed = reference.expected_mz*(1+(key+.5)*bin_ppm*1e-6)
        measurements = []
        for s, v in zip(selected, values):
            cancel_check()
            measurements.append(_scan_mass(s, v, seed, settings, interpolate))
        trace = np.array([r["sum_intensity"] if r else 0. for r in measurements])
        heights = trace.tolist()
        floor = float(np.percentile(trace, 20))
        # Every local chromatographic maximum is considered, not just the
        # largest EIC in a broad window. Endpoints are retained as failed
        # candidates so an RT-window-clipped peak is visible.
        apices = np.flatnonzero((trace[1:-1] >= trace[:-2]) & (trace[1:-1] > trace[2:])) + 1
        apices = sorted(np.unique(np.r_[apices, np.argmax(trace)]), key=lambda i: -trace[i])
        for apex in apices:
            r = measurements[apex]
            if not r or heights[apex] <= 0:
                continue
            lo = hi = int(apex)
            limit = settings.high_fraction*heights[apex]
            while lo > 0 and heights[lo-1] >= limit:
                lo -= 1
            while hi+1 < len(heights) and heights[hi+1] >= limit:
                hi += 1
            contrast = float(heights[apex]/max(floor, 1.))
            high = list(range(lo, hi+1))
            mass = _median([measurements[i]["centroid_mz"] for i in high if measurements[i]])
            # The exact duplicate rule, for records inside twice its mass tolerance.
            reach = 2e-6*bin_ppm*mass
            near = observed[bisect_left(observed, (mass-reach,)):bisect_right(observed, (mass+reach, math.inf))]
            if any(abs(mass-m)/mass*1e6 < bin_ppm and
                   (abs(rt[apex]-records[c]["rt"]) < .03 or set(high).intersection(records[c]["high_indices"])) for m, c in near):
                continue
            flags = []
            if lo == 0 or hi == len(trace)-1:
                flags.append("chromatographic peak reaches RT-window edge")
            if len(high) < settings.minimum_high_scans:
                flags.append("too few high-intensity scans")
            if contrast < settings.minimum_prominence_ratio:
                flags.append("weak chromatographic contrast")
            if any(measurements[i] and measurements[i]["truncated"] for i in high):
                flags.append("truncated/overlapping local spectral lobe")
            dispersion = float(np.std([measurements[i]["centroid_mz"] for i in high], ddof=1)/reference.expected_mz*1e6) if len(high)>1 else None
            # No absolute intensity comparison across compounds. RT distance,
            # chromatographic contrast and repeated mass precision are evidence.
            precision_penalty = min((dispersion/settings.precision_score_scale_ppm)**2, 2) if dispersion is not None else 10.
            score = (settings.rt_score_weight*(1-abs(rt[apex]-reference.expected_rt)/settings.rt_radius)
                     + min(np.log10(max(contrast, 1)), 2) + min(len(high)/5, 1) - precision_penalty)
            records.append(dict(reference_id=reference.id, seed_mz=seed, observed_mz=mass, rt=float(rt[apex]),
                                ppm_error=float((mass-reference.expected_mz)/reference.expected_mz*1e6),
                                da_error=mass-reference.expected_mz, apex_id=selected[apex].id,
                                apex_index=int(apex), high_indices=high, intensity=float(trace[apex]),
                                contrast=contrast, centroid_sd_ppm=dispersion, score=float(score), flags=flags,
                                trace=trace.tolist(), measurements=measurements))
            insort(observed, (mass, len(records)-1))
    records.sort(key=lambda c: (bool(c["flags"]), -c["score"], c["rt"], c["observed_mz"]))
    for i, c in enumerate(records):
        c["candidate_id"] = f"{reference.id}-c{i+1}"
    return records, dict(scan_ids=[s.id for s in selected], rt=rt.tolist(), broad_ppm=settings.broad_ppm)


BEAM_WIDTH = 256
MAX_CANDIDATES = 12


def select_candidates(references, candidate_lists, contexts, settings, cancel_check=lambda: None):
    """Beam search with at most references * beam width * candidate limit expansions.

    The coarse common ppm-or-Da prior remains independent of model residuals.
    A per-pair ambiguity margin avoids dilution as the reference count grows.
    """
    eligible = []
    for ref, rows, context in zip(references, candidate_lists, contexts):
        good = [r for r in rows if not r["flags"]][:MAX_CANDIDATES]
        if not good:
            reason = context.get("failure") or ("; ".join(rows[0]["flags"]) if rows else "no supported chromatographic/profile peak in search window")
            raise ValueError(f"Mass reference {ref.pair}: {reason}")
        eligible.append(good)
    beam = [(0., (), 0.)]
    considered = 0
    for rows in eligible:
        expanded = []
        for _, previous, _ in beam:
            cancel_check()
            for row in rows:
                assignment = (*previous, row)
                ppm = np.array([c["ppm_error"] for c in assignment])
                da = np.array([c["da_error"] for c in assignment])
                masses = np.array([c["observed_mz"] for c in assignment])
                spread = min(float(np.std(ppm)), float(np.std(da)/np.mean(masses)*1e6))
                score = float(np.mean([c["score"] for c in assignment])-(spread/settings.consensus_sd_ppm)**2)
                expanded.append((score, assignment, spread))
                considered += 1
        expanded.sort(key=lambda item: -item[0])
        beam = expanded[:BEAM_WIDTH]
    if beam[0][2] > 2*settings.consensus_sd_ppm:
        pairs = ", ".join(r.pair for r in references)
        raise ValueError(f"Mass references {pairs}: no coherent common coarse mass-error regime")
    margin = (beam[0][0]-beam[1][0])*max(1., len(references)/4) if len(beam)>1 else None
    if margin is not None and margin < settings.ambiguity_margin:
        affected = [r.pair for r, a, b in zip(references, beam[0][1], beam[1][1]) if a["candidate_id"] != b["candidate_id"]]
        raise ValueError(f"Mass reference {', '.join(affected)}: ambiguous competing peaks (score margin {margin:.3f}); edit the reference and retry")
    return list(beam[0][1]), dict(beam_width=BEAM_WIDTH, candidate_limit=MAX_CANDIDATES,
                                assignments_considered=considered, score_margin=margin)


@dataclass(frozen=True)
class CalibrationObservation:
    reference: Reference
    observed_mz: float
    observed_rt: float
    high_scans: int
    centroid_sd_ppm: float
    apex_scan_id: str
    intensity: float

    @property
    def error_ppm(self):
        return (self.observed_mz-self.reference.expected_mz)/self.reference.expected_mz*1e6


@dataclass(frozen=True)
class MassCorrection:
    offset_ppm: float
    observations: tuple
    settings: SearchSettings

    def __post_init__(self):
        if not np.isfinite(self.offset_ppm) or self.denominator <= 0:
            raise ValueError("Invalid constant-ppm correction denominator")

    @property
    def denominator(self):
        return 1 + self.offset_ppm*1e-6

    def correct(self, mz):
        mz = np.asarray(mz)
        if mz.ndim != 1 or not np.isfinite(mz).all() or np.any(mz <= 0) or np.any(np.diff(mz) <= 0):
            raise ValueError("Invalid original MS1 mass axis")
        corrected = mz/self.denominator
        if not np.isfinite(corrected).all() or np.any(corrected <= 0) or np.any(np.diff(corrected) <= 0):
            raise ValueError("Invalid corrected MS1 mass axis")
        return corrected

    def calibration_residual(self, observation):
        return (observation.observed_mz/self.denominator-observation.reference.expected_mz)/observation.reference.expected_mz*1e6

    def payload(self):
        rows = []
        for o in self.observations:
            rows.append(dict(expected_mz=o.reference.expected_mz, expected_rt=o.reference.expected_rt,
                             observed_mz=o.observed_mz, observed_rt=o.observed_rt, error_ppm=o.error_ppm,
                             calibration_residual_ppm=self.calibration_residual(o), high_scans=o.high_scans,
                             centroid_sd_ppm=o.centroid_sd_ppm, apex_scan_id=o.apex_scan_id, intensity=o.intensity))
        return dict(offset_ppm=self.offset_ppm, references=rows, search_settings=asdict(self.settings),
                    consistency="unavailable-single-reference" if len(rows)==1 else "coarse-checked")

    def provenance_params(self):
        return provenance_params(self.payload())


def fit_constant_ppm(observations, settings=SearchSettings()):
    observations = tuple(observations)
    if not observations:
        raise ValueError("Mass correction requires at least one matched reference")
    correction = MassCorrection(float(np.median([o.error_ppm for o in observations])), observations, settings)
    correction.provenance_params()  # Validate observations and the complete v1 contract before writing.
    return correction


def estimate_mass_correction(reader, references, log=lambda message: None, cancel_check=lambda: None):
    """Estimate only from original full MS1 arrays, before any output is opened."""
    references = list(references)
    if not references:
        raise ValueError("Mass correction requires at least one reference")
    settings = SearchSettings()
    all_candidates, contexts = [], []
    for i, reference in enumerate(references):
        log(f"Estimating mass reference {i+1}/{len(references)}: {reference.pair}")
        window = []
        for scan in reader.scans:
            cancel_check()
            if reader.experiments[scan.experiment-1].ms_level != 1:
                continue
            rt = scan.rt_seconds/60
            if abs(rt-reference.expected_rt) > settings.rt_radius:
                continue
            window.append((scan, rt))
        scans = []
        # Native calibration AND framing; no filtering. The converter's reader decodes
        # ahead on worker threads; others decode each scan after its cancellation check.
        decoded = native.ordered(lambda item: native.reader_arrays(reader, item[0]), window, native.workers(reader))
        with closing(decoded):
            for scan, rt in window:
                cancel_check()
                mz, intensity = next(decoded)
                radius = (settings.broad_ppm+2*settings.lobe_ppm)*1e-6
                a, b = np.searchsorted(mz, reference.expected_mz*(1+np.array([-radius, radius])))
                scans.append(ProfileScan(scan.native_id, rt, mz[a:b].copy(), intensity[a:b].copy()))
        rows, context = candidates(reference, scans, settings, cancel_check=cancel_check)
        all_candidates.append(rows)
        contexts.append(context)
    selected, evidence = select_candidates(references, all_candidates, contexts, settings, cancel_check)
    observations = []
    for reference, candidate in zip(references, selected):
        cancel_check()
        measurements = [candidate["measurements"][j] for j in candidate["high_indices"]]
        values = [m["centroid_mz"] for m in measurements if m is not None and not m["truncated"]]
        if len(values) != len(measurements) or len(values) < settings.minimum_high_scans:
            raise ValueError(f"Mass reference {reference.pair}: insufficient valid high-scan centroids")
        observations.append(CalibrationObservation(reference, float(np.median(values)), candidate["rt"],
            len(values), float(np.std(values, ddof=1)/reference.expected_mz*1e6), candidate["apex_id"], candidate["intensity"]))
    correction = fit_constant_ppm(observations, settings)
    log(f"Fitted global MS1 offset: {correction.offset_ppm:+.6f} ppm (equal-reference median; {len(observations)} references)")
    for o in observations:
        residual = correction.calibration_residual(o)
        log(f"Mass reference {o.reference.pair}: expected m/z={o.reference.expected_mz:.7f}, RT={o.reference.expected_rt:.5f} min; "
            f"observed m/z={o.observed_mz:.7f}, RT={o.observed_rt:.5f} min; raw error={o.error_ppm:+.3f} ppm; "
            f"calibration residual={residual:+.3f} ppm; high scans={o.high_scans}, centroid SD={o.centroid_sd_ppm:.3f} ppm")
        if abs(residual) > 20:
            log(f"WARNING: 20 ppm target NOT MET for mass reference {o.reference.pair}: calibration residual {residual:+.3f} ppm. Beta output retained; accuracy across the run is not guaranteed.")
    if len(observations)==1:
        log("Single-reference calibration residual is zero by construction; predictive accuracy unverified. Cross-reference consistency unavailable.")
    else:
        log("These are fitting-reference calibration residuals; predictive accuracy across the run is unverified.")
    log(f"Bounded reference selection: beam={evidence['beam_width']}, candidates/reference≤{evidence['candidate_limit']}.")
    return correction


def default_output_name(input_path, profile="full", *, corrected=False):
    return Path(input_path).stem + (".mc" if corrected else "") + (".sp.mzML" if profile=="sparse" else ".mzML")
