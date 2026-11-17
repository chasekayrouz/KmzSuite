"""The optimized estimator equals the original interpreted one exactly (synthetic inputs).

``oracle_local_mass`` and ``oracle_candidates`` are main@0a039d5's implementations,
kept verbatim (renamed) as the independent reference.
"""
import numpy as np
import pytest

from kmzconvert_core.mass_correction import ProfileScan, Reference, SearchSettings, candidates, local_mass


def oracle_local_mass(mz, intensity, seed_mz, settings, *, interpolate=False):
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
    peak = a + int(np.argmax(intensity[a:b]))
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
    weights = intensity[left:right+1]
    centroid = float(np.dot(mz[left:right+1], weights)/weights.sum())
    parabolic = None
    if interpolate and 0 < peak < len(mz)-1 and intensity[peak-1] > 0 and intensity[peak+1] > 0:
        x = mz[peak-1:peak+2] - mz[peak]
        c2, c1, _ = np.linalg.solve(np.column_stack([x*x, x, np.ones(3)]), intensity[peak-1:peak+2])
        if c2 < 0:
            vertex = -c1/(2*c2)
            if x[0] <= vertex <= x[-1]:
                parabolic = float(mz[peak] + vertex)
    return dict(sampled_mz=float(mz[peak]), centroid_mz=centroid, parabola_mz=parabolic,
                height=height, sum_intensity=float(weights.sum()), left_mz=float(mz[left]),
                right_mz=float(mz[right]), points=int(right-left+1), truncated=truncated,
                seed_distance_ppm=float((mz[peak]-seed_mz)/seed_mz*1e6))


def oracle_candidates(reference, scans, settings, *, interpolate=False, cancel_check=lambda: None):
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
            measurements.append(oracle_local_mass(s.mz, s.intensity, seed, settings, interpolate=interpolate))
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


def scans(rng, reference, count=60):
    """Gaussian ion tracks over a noisy floor, with interferences, ties, gaps and clipped windows."""
    m = np.sort(rng.uniform(reference.expected_mz*.9988, reference.expected_mz*1.0012, int(rng.integers(40, 900))))
    m = np.unique(np.round(m, int(rng.choice([4, 6, 9]))))
    rts = np.sort(rng.uniform(reference.expected_rt-.5, reference.expected_rt+.5, count))
    tracks = [(reference.expected_mz*(1+rng.uniform(-900, 900)*1e-6), rng.uniform(-.4, .4), rng.choice([1e2, 1e4, 1e6]))
              for _ in range(int(rng.integers(1, 6)))]
    result = []
    for i, rt in enumerate(rts):
        y = rng.choice([0., 0., 1., 7.], len(m))
        for center, offset, height in tracks:
            width = reference.expected_mz*rng.choice([3e-6, 6e-6, 4e-5])
            y = y + height*np.exp(-.5*((rt-reference.expected_rt-offset)/.05)**2)*np.exp(-.5*((m-center)/width)**2)
        if rng.random() < .3:
            y = np.rint(y)
        result.append(ProfileScan(f"s{i}", float(rt), m, y))
    return result


@pytest.mark.parametrize("seed", range(30))
def test_candidates_and_local_mass_equal_the_original(seed):
    rng = np.random.default_rng(seed)
    reference = Reference("reference-1", float(rng.uniform(150, 900)), float(rng.uniform(1, 9)))
    settings = SearchSettings() if seed % 3 else SearchSettings(track_ppm=40, lobe_ppm=60., high_fraction=.3)
    data = scans(rng, reference, int(rng.integers(2, 80)))
    for interpolate in (False, True):
        assert repr(candidates(reference, data, settings, interpolate=interpolate)) == repr(
            oracle_candidates(reference, data, settings, interpolate=interpolate))
    for scan in data[:10]:
        for seed_mz in rng.uniform(reference.expected_mz*.999, reference.expected_mz*1.001, 20):
            for interpolate in (False, True):
                assert repr(local_mass(scan.mz, scan.intensity, seed_mz, settings, interpolate=interpolate)) == repr(
                    oracle_local_mass(scan.mz, scan.intensity, seed_mz, settings, interpolate=interpolate))


def test_unqualified_numpy_uses_the_original_estimator(monkeypatch, unqualified):
    from kmzconvert_core import mass_correction
    rng = np.random.default_rng(7)
    reference = Reference("reference-1", float(rng.uniform(150, 900)), float(rng.uniform(1, 9)))
    data = scans(rng, reference, 40)
    expected = repr(oracle_candidates(reference, data, SearchSettings()))
    original, calls = mass_correction._numpy_candidates, []
    monkeypatch.setattr(mass_correction, "_numpy_candidates", lambda *a, **k: calls.append(a) or original(*a, **k))
    assert repr(candidates(reference, data, SearchSettings())) == expected and not calls
    unqualified("numpy")
    assert repr(candidates(reference, data, SearchSettings())) == expected and len(calls) == 1
