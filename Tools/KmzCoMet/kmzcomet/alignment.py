"""Conservative constant RT shifts, complete-link groups, and disjoint RT windows."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import math

import numpy as np

from .model import Feature


def mass_key(mass, ppm):
    return int(math.log(mass) / math.log1p(ppm * 1e-6))


def estimate_shifts(samples, settings):
    reference = samples[0]
    for sample in samples:
        polarities = sorted({c.polarity for c in sample.candidates} |
                            {c.polarity for c in reference.candidates})
        for polarity in polarities:
            sample.shifts[polarity] = 0.
            sample.anchors[polarity] = 0
            sample.alignment_mad[polarity] = None
            if sample is reference or settings.alignment == "none":
                continue
            ref = sorted([c for c in reference.candidates if c.polarity == polarity], key=lambda c: c.mz)
            current = [c for c in sample.candidates if c.polarity == polarity]
            masses = [c.mz for c in ref]
            proposals = []
            counts = defaultdict(int)
            for candidate in current:
                radius = candidate.mz * settings.mz_ppm * 1e-6
                a, b = bisect_left(masses, candidate.mz-radius), bisect_right(masses, candidate.mz+radius)
                matches = [i for i in range(a, b) if abs(candidate.rt-ref[i].rt) <= settings.max_rt_shift]
                if len(matches) == 1:
                    index = matches[0]
                    proposals.append((index, candidate.rt-ref[index].rt))
                    counts[index] += 1
            delta = np.array([d for i, d in proposals if counts[i] == 1])
            if len(delta):
                center = np.median(delta)
                mad = np.median(np.abs(delta-center))
                keep = np.abs(delta-center) <= min(settings.rt_tolerance, max(.01, 3*1.4826*mad))
                delta = delta[keep]
            if len(delta) >= settings.min_anchors:
                shift = float(np.median(delta))
                mad = float(np.median(np.abs(delta-shift)))
                if mad <= settings.rt_tolerance/2:
                    sample.shifts[polarity] = shift
                    sample.anchors[polarity] = len(delta)
                    sample.alignment_mad[polarity] = mad
                    continue
            sample.warnings.append(f"Polarity {polarity}: insufficient consistent RT anchors; zero shift used")


def group_features(samples, settings):
    groups = []
    index = defaultdict(list)
    candidates = [c for sample in samples for c in sample.candidates]

    def rt(c):
        return c.rt - samples[c.sample].shifts.get(c.polarity, 0.)

    for c in sorted(candidates, key=lambda c: (-c.height, c.polarity, c.mz, rt(c), c.sample)):
        key = mass_key(c.mz, settings.mz_ppm)
        options = []
        for k in (key-1, key, key+1):
            for g in index[c.polarity, k]:
                if any(member.sample == c.sample for member in groups[g]):
                    continue
                if all(abs(c.mz-m.mz)/min(c.mz,m.mz)*1e6 <= settings.mz_ppm and
                       abs(rt(c)-rt(m)) <= settings.rt_tolerance for m in groups[g]):
                    options.append(g)
        if options:
            best = min(options, key=lambda g: (abs(rt(c)-rt(groups[g][0]))/settings.rt_tolerance +
                       abs(c.mz-groups[g][0].mz)/c.mz*1e6/settings.mz_ppm, g))
            groups[best].append(c)
        else:
            index[c.polarity, key].append(len(groups))
            groups.append([c])
    features = [Feature("", float(np.median([c.mz for c in group])),
                        float(np.median([rt(c) for c in group])), group[0].polarity, group)
                for group in groups]
    features.sort(key=lambda f: (f.polarity, f.mz, f.rt))
    index.clear()
    for i, f in enumerate(features):
        f.id = f"F{i+1:06d}"
        f.left, f.right = f.rt-settings.rt_window, f.rt+settings.rt_window
        key = mass_key(f.mz, settings.mz_ppm)
        for k in range(key-3, key+4):
            for old_index in index[f.polarity, k]:
                old = features[old_index]
                # If mass extraction windows overlap, divide the time overlap at
                # the midpoint. Coeluting unresolved masses are flagged instead.
                if abs(old.mz-f.mz) <= (old.mz+f.mz)*settings.mz_ppm*1e-6:
                    if abs(old.rt-f.rt) < settings.max_scan_gap:
                        old.ambiguous = f.ambiguous = True
                    else:
                        earlier, later = sorted((old, f), key=lambda x: x.rt)
                        middle = (earlier.rt+later.rt)/2
                        earlier.right = min(earlier.right, middle)
                        later.left = max(later.left, middle)
        index[f.polarity, key].append(i)
    return features
