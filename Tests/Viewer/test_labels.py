"""Adaptive m/z labels depend on height and on-screen spacing."""

import numpy as np

from kmzcore.labels import format_mz_label, select_peak_labels


def _profile():
    mz = np.arange(100.0, 110.0, 0.01)
    intensity = np.zeros_like(mz)
    for center, height in ((101.0, 100.0), (101.2, 60.0), (108.0, 90.0), (104.0, 3.0)):
        index = int(round((center - 100.0) / 0.01))
        intensity[index - 1] = height * 0.4
        intensity[index] = height
        intensity[index + 1] = height * 0.5
    return mz, intensity


def test_label_text_has_four_decimals():
    assert format_mz_label(273.1559) == "273.1559"
    assert format_mz_label(300) == "300.0000"


def _has(masses, target: float) -> bool:
    return any(abs(mass - target) < 1e-6 for mass in masses)


def test_profile_labels_prefer_tall_peaks_and_open_up_when_zoomed():
    mz, intensity = _profile()
    wide = select_peak_labels(mz, intensity, 100.0, 110.0, centroid=False, view_width_px=200)
    wide_mz = [mass for mass, _height in wide]
    assert _has(wide_mz, 101.0)
    assert _has(wide_mz, 108.0)
    assert not _has(wide_mz, 101.2)
    assert not _has(wide_mz, 104.0)
    close = select_peak_labels(mz, intensity, 100.8, 101.5, centroid=False, view_width_px=400)
    close_mz = [mass for mass, _height in close]
    assert _has(close_mz, 101.0)
    assert _has(close_mz, 101.2)


def test_centroid_peaks_are_candidates_without_a_local_maximum_requirement():
    mz = np.array([100.0, 100.05, 100.1, 105.0])
    intensity = np.array([10.0, 50.0, 20.0, 40.0])
    labels = select_peak_labels(mz, intensity, 99.0, 106.0, centroid=True, view_width_px=220)
    masses = [mass for mass, _height in labels]
    assert masses[0] == 100.05
    assert 105.0 in masses
    assert 100.0 not in masses
    assert 100.1 not in masses


def test_labels_outside_the_window_are_omitted():
    mz = np.array([50.0, 273.1559, 400.0])
    intensity = np.array([80.0, 100.0, 90.0])
    labels = select_peak_labels(mz, intensity, 200.0, 300.0, centroid=True, view_width_px=400)
    assert [mass for mass, _height in labels] == [273.1559]


def _greedy_reference(mz, intensity, xmin, xmax, width, separation):
    """The original per-candidate loop over the same visible, strong maxima."""
    visible = (mz >= xmin) & (mz <= xmax) & np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    mz, intensity = mz[visible], intensity[visible]
    if not mz.size:
        return []
    strong = intensity >= float(np.max(intensity)) * 0.05
    mz, intensity = mz[strong], intensity[strong]
    accepted, pixels = [], []
    for index in np.argsort(-intensity, kind="stable"):
        pixel = (float(mz[index]) - float(xmin)) / float(xmax - xmin) * float(width)
        if any(abs(pixel - other) < separation for other in pixels):
            continue
        accepted.append((float(mz[index]), float(intensity[index])))
        pixels.append(pixel)
    return accepted


def test_vectorized_separation_matches_greedy_loop_exactly():
    rng = np.random.default_rng(7)
    for trial in range(300):
        size = int(rng.integers(1, 1500))
        mz = np.round(np.sort(rng.uniform(100, 400, size)), 1 + trial % 3)
        intensity = rng.integers(1, 6, size).astype(np.float64) if trial % 2 else rng.exponential(50, size)
        xmin = float(rng.uniform(90, 300))
        xmax = xmin + float(rng.choice([1.0, 20.0, 400.0]))
        width = float(rng.choice([30.0, 400.0, 1400.0]))
        separation = float(rng.choice([0.0, 10.0, 72.0]))
        labels = select_peak_labels(mz, intensity, xmin, xmax, centroid=True, view_width_px=width,
                                    min_separation_px=separation)
        expected = _greedy_reference(mz, intensity, xmin, xmax, width, separation)
        assert [(m.hex(), y.hex()) for m, y in labels] == [(m.hex(), y.hex()) for m, y in expected]
