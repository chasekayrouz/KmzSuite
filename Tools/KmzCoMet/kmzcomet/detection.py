"""Fixed-mass tracks and bounded chromatographic local maxima for discovery only."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from itertools import compress

import numpy as np

from .model import Candidate, Settings


def signal_metrics(times, values, settings: Settings):
    """Unsmooth native-height evidence. MAD of first differences estimates noise."""
    return trace_metrics(np.asarray(times)[None], np.asarray(values)[None], settings)[0]


def trace_metrics(times, values, settings: Settings):
    """``signal_metrics`` of each row of equal-length traces (2-D arrays). Row-wise
    reductions take the same order statistics, and apply the same elementwise
    arithmetic, as each trace alone."""
    apex = np.argmax(values, axis=1)
    height = values[np.arange(len(values)), apex]
    background = np.quantile(values, .2, axis=1)
    delta = np.diff(values, axis=1)
    if delta.shape[1]:
        noise = np.maximum(1., 1.4826 * np.median(np.abs(delta - np.median(delta, axis=1)[:, None]), axis=1)
                           / np.sqrt(2))
    else:
        noise = np.ones(len(values))
    above = np.maximum(0., height - background)
    threshold = background + np.maximum(2 * noise, .1 * above)
    contrast = height / np.maximum(background, 1.)
    snr = above / noise
    metrics = []
    for t, y, apex, height, background, noise, above, threshold, contrast, snr in zip(
            times.tolist(), values.tolist(), apex.tolist(), height.tolist(), background.tolist(), noise.tolist(),
            above.tolist(), threshold.tolist(), contrast.tolist(), snr.tolist()):
        left = right = apex
        if height > threshold:
            while left > 0 and y[left-1] > threshold and t[left]-t[left-1] <= settings.max_scan_gap:
                left -= 1
            while right+1 < len(y) and y[right+1] > threshold and t[right+1]-t[right] <= settings.max_scan_gap:
                right += 1
            support = right - left + 1
        else:
            support = 0
        supported = (height >= settings.min_height and support >= settings.min_points and snr >= settings.min_snr and
                     contrast >= settings.min_contrast and 0 < apex < len(y)-1)
        metrics.append(dict(apex=apex, height=float(height), background=float(background), noise=float(noise),
                            height_above_background=float(above), snr=float(snr), contrast=float(contrast),
                            support_points=support, supported=supported))
    return metrics


def mass_peaks(scan, threshold):
    y = scan.intensity
    if scan.mode == "centroid":
        return np.flatnonzero(y >= threshold)
    # Profile endpoints cannot establish a local maximum. Plateaus choose their
    # right edge. Native apex coordinates/heights are retained, not centroid areas.
    inner = np.flatnonzero(y[1:-1] >= threshold) + 1
    peaks = inner[(y[inner] >= y[inner-1]) & (y[inner] > y[inner+1])]
    # A shoulder ripple is not a resolved ion. Adjacent maxima need a valley
    # at or below half the smaller apex; otherwise retain the taller maximum.
    # This filters discovery only: measurement still uses every native point.
    if len(peaks) < 2:
        return peaks
    heights = y[peaks]
    # spans[j] is the minimum of y over [peaks[j], peaks[j+1]]; the minimum over
    # any run of consecutive candidates is the minimum of their spans.
    spans = np.minimum(np.minimum.reduceat(y, peaks)[:-1], heights[1:])
    if np.all(spans <= .5*np.minimum(heights[:-1], heights[1:])):
        return peaks  # Every neighbour is resolved: nothing merges.
    kept, valleys, low = [], [], None
    apex, spans = heights.tolist(), spans.tolist()
    for b, height in enumerate(apex):
        if kept:  # low becomes the minimum from the last kept maximum through b.
            low = spans[b-1] if low is None else min(low, spans[b-1])
        while kept and low > .5*min(apex[kept[-1]], height):
            if apex[kept[-1]] >= height:
                break
            kept.pop()
            valley = valleys.pop()
            if kept:
                low = min(valley, low)
        else:
            valleys.append(low)
            kept.append(b)
            low = None
    return peaks[kept]


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
        # Per polarity, active tracks in (mass, serial) order as parallel lists:
        # masses, serials, tracks and each track's last time.
        self.active = {polarity: ([], [], [], []) for polarity in (-1, 0, 1)}
        self.serial = 0
        self.candidates = []

    def add(self, scan):
        s = self.settings
        rt = scan.rt
        masses, serials, tracks, lasts = self.active[scan.polarity]
        if lasts:
            expired = np.flatnonzero(rt - np.array(lasts) > s.max_scan_gap)
            if len(expired):
                live = np.ones(len(lasts), dtype=bool)
                live[expired] = False
                for index in expired.tolist():
                    self.finish(tracks[index])
                live = live.tolist()
                for values in (masses, serials, tracks, lasts):
                    values[:] = compress(values, live)
        assigned = set()
        peaks = mass_peaks(scan, s.discovery_height)
        # Highest-first makes one-to-one conflicts deterministic.
        peaks = peaks[np.argsort(-scan.intensity[peaks], kind="stable")]
        for mass, height in zip(scan.mz[peaks].tolist(), scan.intensity[peaks].tolist()):
            radius = mass * s.mz_ppm * 1e-6
            # The nearest unassigned track within the radius; ties take the older.
            best = None
            for index in range(bisect_left(masses, mass-radius), bisect_right(masses, mass+radius)):
                if serials[index] not in assigned:
                    key = (abs(masses[index]-mass), serials[index])
                    if best is None or key < best:
                        best, match = key, index
            if best is None:
                track = Track(mass, scan.polarity)
                match = bisect_right(masses, mass)
                masses.insert(match, mass)
                serials.insert(match, self.serial)
                tracks.insert(match, track)
                lasts.insert(match, rt)
                self.serial += 1
            else:
                track = tracks[match]
                lasts[match] = rt
            assigned.add(serials[match])
            track.times.append(rt)
            track.masses.append(mass)
            track.heights.append(height)

    def finish(self, track):
        s = self.settings
        if len(track.times) < s.min_points:
            return
        times, y = np.array(track.times), np.array(track.heights)
        # Discovery gaps are below-threshold measurements, not quantitative zeroes.
        # Regular resampling is deliberately avoided. Split large scan gaps first.
        cuts = [0, *(np.flatnonzero(np.diff(times) > s.max_scan_gap)+1).tolist(), len(y)]
        window = max(s.discovery_window, s.rt_window)
        for start, stop in zip(cuts[:-1], cuts[1:]):
            if stop-start < s.min_points:
                continue
            t, v = times[start:stop], y[start:stop]
            smooth = np.convolve(np.concatenate((v[:1], v, v[-1:])), [.25, .5, .25], mode="valid")
            peaks = np.flatnonzero((smooth >= np.concatenate(([-np.inf], smooth[:-1]))) &
                                   (smooth > np.concatenate((smooth[1:], [-np.inf]))))
            # Scalar tests on the same float values as lists (no per-peak NumPy calls).
            tl, vl, sl = t.tolist(), v.tolist(), smooth.tolist()
            accepted = []
            for peak in sorted(peaks.tolist(), key=lambda i: (-sl[i], i)):
                lo, hi = bisect_left(tl, tl[peak]-window), bisect_left(tl, tl[peak]+window)
                if hi-lo < s.min_points:
                    continue
                # Require a real rise and fall in the observed high-signal
                # track. A constant contaminant is not a chromatographic peak.
                if not lo < peak < hi-1:
                    continue
                floor = max(min(vl[lo:peak+1]), min(vl[peak:hi]))
                if vl[peak] < s.min_contrast * max(1., floor):
                    continue
                # The same long tail can fluctuate beyond rt_tolerance. Require
                # a deep valley to every retained maximum on this mass track,
                # rather than creating another feature when a timer expires.
                if any(min(sl[min(peak, old):max(peak, old)+1]) >
                       .5*min(sl[peak], sl[old]) for old in accepted):
                    continue
                level = max(s.discovery_height, .1*vl[peak])
                if sum(value >= level for value in vl[lo:hi]) < s.min_points:
                    continue
                accepted.append(peak)
                near = (np.abs(t-t[peak]) <= s.rt_tolerance/2) & (v >= .5*v[peak])
                mass = float(np.median(np.array(track.masses[start:stop])[near]))
                self.candidates.append(Candidate(self.sample, mass, float(t[peak]),
                                                 float(v[peak]), track.polarity))

    def complete(self):
        for _, _, tracks, _ in self.active.values():
            for track in tracks:
                self.finish(track)
        # Fragmented tracks may converge to the same peak; collapse only within
        # one scan-scale RT distance and the configured mass tolerance.
        result = []
        gap, ppm = self.settings.max_scan_gap, self.settings.mz_ppm
        # Kept candidates by polarity in mass order. The exact test below only
        # reads those inside a mass interval at least twice its tolerance.
        margin = max(2 * ppm * 1e-6, 1e-9)
        kept = {}
        for c in sorted(self.candidates, key=lambda c: (-c.height, c.mz, c.rt)):
            keys, olds = kept.setdefault(c.polarity, ([], []))
            nearby = olds[bisect_left(keys, c.mz*(1-margin)):bisect_right(keys, c.mz*(1+margin))]
            if not any(c.polarity == old.polarity and abs(c.rt-old.rt) <= gap
                       and abs(c.mz-old.mz)/old.mz*1e6 <= ppm for old in nearby):
                result.append(c)
                index = bisect_right(keys, c.mz)
                keys.insert(index, c.mz)
                olds.insert(index, c)
        return sorted(result, key=lambda c: (c.polarity, c.mz, c.rt))
