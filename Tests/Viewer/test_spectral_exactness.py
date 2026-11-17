"""Vectorized spectral paths reproduce the scalar/library results bit for bit."""
import json
import math

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from kmzcore.cancel import Cancelled
from kmzviewer import spectral_matching, spectral_motifs
from kmzviewer.reference_pack import ReferencePack
from kmzviewer.spectral_analysis import save_spectral_analysis
from kmzviewer.spectral_matching import SpectralSettings, align_peaks, search_reference_pack


def scalar_alignment(query_peaks, reference_peaks, query_precursor, reference_precursor, tolerance, mode):
    """The original per-peak implementation."""
    q, r = np.asarray(query_peaks, dtype=float), np.asarray(reference_peaks, dtype=float)
    delta = float(query_precursor - reference_precursor)
    shifts = [("direct", 0.)] if mode != "neutral_loss" else []
    if mode in {"hybrid", "neutral_loss"} and (mode == "neutral_loss" or abs(delta) > tolerance):
        shifts.append(("shifted", delta))
    candidates = []
    for i, (mz, intensity) in enumerate(q):
        for kind, shift in shifts:
            left, right = np.searchsorted(r[:, 0], [mz - shift - tolerance, mz - shift + tolerance])
            while right < len(r) and r[right, 0] <= mz - shift + tolerance:
                right += 1
            for j in range(left, right):
                candidates.append((float(intensity * r[j, 1]), i, j, kind, float(mz - r[j, 0] - shift)))
    candidates.sort(key=lambda x: (-x[0], x[3] != "direct", abs(x[4]), x[1], x[2]))
    used_q, used_r, pairs = set(), set(), []
    for _, i, j, kind, error in candidates:
        if i in used_q or j in used_r:
            continue
        used_q.add(i)
        used_r.add(j)
        pairs.append(dict(query_index=i, reference_index=j, query_mz=float(q[i, 0]), reference_mz=float(r[j, 0]),
                          kind=kind, error_da=error, query_intensity=float(q[i, 1]), reference_intensity=float(r[j, 1])))
    return dict(pairs=pairs, direct_count=sum(p["kind"] == "direct" for p in pairs),
                shifted_count=sum(p["kind"] == "shifted" for p in pairs),
                query_coverage=float(sum(q[i, 1] for i in used_q) / q[:, 1].sum()),
                reference_coverage=float(sum(r[i, 1] for i in used_r) / r[:, 1].sum()),
                description="Local one-to-one intensity-product alignment; coverage uses prepared summed intensity.")


def exact(value):
    """Type-, order- and bit-preserving comparison form."""
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, dict):
        return ("dict", [(k, exact(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [exact(v) for v in value])
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.strides, value.tobytes())
    return (type(value).__name__, value)


def test_alignment_matches_scalar_form_with_boundary_ties_and_equal_products():
    rng = np.random.default_rng(7)
    for trial in range(300):
        grid = .005 * rng.integers(1, 40000, size=(rng.integers(1, 40), 1))
        q = np.column_stack((np.sort(grid[:, 0]), rng.choice([.25, .5, 1., 2.], size=len(grid))))
        ref = q[:, 0] + .005 * rng.integers(-6, 7, size=len(q)) - (14. if trial % 3 else 0.)
        r = np.column_stack((np.sort(ref), rng.choice([.25, .5, 1.], size=len(q)))).astype(np.float32)
        for mode in ("hybrid", "direct", "neutral_loss"):
            args = (q, r, 400., 400. - (14. if trial % 3 else rng.choice([0., .01, .03])), rng.choice([.005, .01, .02]), mode)
            assert exact(align_peaks(*args)) == exact(scalar_alignment(*args))


def synthetic_pack(seed=3, count=400):
    """Duplicated spectra/precursors force structured-sort ties; mixed charges give negative losses."""
    rng = np.random.default_rng(seed)
    metadata, arrays = [], []
    for i in range(count):
        n = int(rng.choice([1, 2, 7, 8, 9, 30, 129, 300]))
        if i % 5 == 4:
            peaks = arrays[-1].copy()
            if i % 20 != 4:  # otherwise an exact duplicate: ties reach the spectrum order
                peaks[:, 1] *= rng.uniform(.5, 2, size=len(peaks)).astype(np.float32)
            precursor = metadata[-1]["precursor_mz"] if i % 10 == 4 else round(float(rng.uniform(150, 900)), 4)
        else:
            peaks = np.column_stack((np.sort(np.round(rng.uniform(40, 1200, n), 3)),
                                     rng.choice([1., 3., 10., 100.], n) if i % 2 else rng.uniform(1, 100, n))).astype(np.float32)
            precursor = round(float(rng.uniform(150, 900)), 4)
        metadata.append(dict(name=f"Synthetic {i}", accession=f"SYN{i}", usi="", library="Synthetic", provenance="Test",
                             smiles="", inchi="", structure_key=f"synthetic:{i // 3}", adduct="", quality="",
                             instrument="", collision_energy="", precursor_mz=precursor, polarity=1 if i % 7 else -1,
                             charge=[None, 1, -1, 2, 3][i % 5]))
        arrays.append(peaks)
    offsets = np.r_[0, np.cumsum([len(a) for a in arrays])].astype(np.int64)
    return ReferencePack(metadata, np.concatenate(arrays), offsets, {"schema": "test"})


def library_engine(pack, settings, polarity):
    from ms_entropy import FlashEntropySearch, clean_spectrum
    spectra = []
    for index, row in enumerate(pack.metadata):
        if row["polarity"] != polarity:
            continue
        cutoff = row["precursor_mz"] - 1.6 if row["charge"] is None or abs(row["charge"]) <= 1 else -1
        peaks = clean_spectrum(pack.spectrum(index).copy(), min_mz=0, max_mz=cutoff, noise_threshold=settings.relative_noise,
                               min_ms2_difference_in_da=spectral_matching._separation(settings), normalize_intensity=True)
        if len(peaks):
            spectra.append(dict(precursor_mz=row["precursor_mz"], peaks=peaks, pack_index=index))
    engine = FlashEntropySearch(max_ms2_tolerance_in_da=settings.fragment_da, mz_index_step=.001)
    return engine, engine.build_index(spectra, clean_spectra=False, max_indexed_mz=2000.00005)


@pytest.mark.parametrize("settings", [SpectralSettings(), SpectralSettings(relative_noise=0.),
                                      SpectralSettings(fragment_da=.05, relative_noise=.05)])
@pytest.mark.parametrize("polarity", [1, -1])
def test_vectorized_index_is_byte_identical_to_ms_entropy(settings, polarity):
    pack = synthetic_pack()
    reference, expected = library_engine(pack, settings, polarity)
    query = dict(peaks=pack.spectrum(3).astype(float).tolist(), precursor_mz=1250.,
                 charge=1, polarity=polarity)
    search_reference_pack(query, settings, pack)
    _, engine, spectra, corpus = pack._search_cache
    assert corpus["vector"] and engine.metadata is None  # the vectorized build ran
    assert [s["pack_index"] for s in spectra] == [s["pack_index"] for s in expected]
    for got, want in zip(engine.entropy_search.index, reference.entropy_search.index, strict=True):
        assert exact(got) == exact(want)
    assert exact(engine.precursor_mz_array) == exact(reference.precursor_mz_array)
    q = spectral_matching.prepare_search_peaks(query, settings)["peaks"]
    for method in ("open_search", "neutral_loss_search", "hybrid_search"):
        args = (q,) if method == "open_search" else (query["precursor_mz"], q)
        assert exact(getattr(engine, method)(*args, ms2_tolerance_in_da=settings.fragment_da)) == \
            exact(getattr(reference, method)(*args, ms2_tolerance_in_da=settings.fragment_da))


def without_timing(result):
    result["provenance"].pop("search_seconds")
    return exact(result)


@pytest.mark.parametrize("name", ["ms-entropy", "numpy"])
def test_library_fallback_and_tolerance_reuse_give_identical_results(unqualified, name):
    pack = synthetic_pack(seed=11)
    query = dict(peaks=pack.spectrum(3).astype(float).tolist(), precursor_mz=pack.metadata[3]["precursor_mz"],
                 charge=None, polarity=1)
    filters = SpectralSettings(required_fragments=(float(pack.spectrum(3)[0, 0]),), required_losses=(14.,), min_peaks=1)
    fast = [without_timing(search_reference_pack(query, s, pack)) for s in (SpectralSettings(), filters)]
    index = pack._search_cache[1].entropy_search.index
    # A tolerance with the same merge separation shares the prepared index.
    reused = without_timing(search_reference_pack(query, SpectralSettings(fragment_da=.01), pack))
    assert pack._search_cache[1].entropy_search.index is index
    unqualified(name)
    slow_pack = synthetic_pack(seed=11)
    slow = [without_timing(search_reference_pack(query, s, slow_pack)) for s in (SpectralSettings(), filters)]
    assert slow_pack._search_cache[1].metadata is not None  # ms-entropy's own build_index ran
    assert slow == fast
    assert without_timing(search_reference_pack(query, SpectralSettings(fragment_da=.01), slow_pack)) == reused
    assert slow_pack._search_cache[1].entropy_search.index is not index


def test_corpus_filters_match_per_spectrum_conditions():
    pack = synthetic_pack(seed=5)
    query = dict(peaks=pack.spectrum(2).astype(float).tolist(), precursor_mz=500., charge=1, polarity=1)
    search_reference_pack(query, SpectralSettings(), pack)
    _, _, spectra, corpus = pack._search_cache
    masses = np.concatenate([s["peaks"][:, 0] for s in spectra])
    for target in [float(masses[7]), float(masses[7] + np.float32(.02)), float(masses[100]) - .02, 333.]:
        for settings in (SpectralSettings(required_fragments=(target,)),
                         SpectralSettings(required_losses=(float(spectra[9]["precursor_mz"] - masses[50]), target)),
                         SpectralSettings(fragment_da=.001, required_fragments=(target, float(masses[3])))):
            expected = [spectral_matching._required_match(s["peaks"], s["precursor_mz"], settings) for s in spectra]
            assert spectral_matching._required_matches(corpus, settings).tolist() == expected


def scalar_motif_align(motif, peaks, original_indices, precursor, tolerance, allow_losses):
    """The original per-feature motif alignment."""
    features = [("fragment", mz, weight) for mz, weight in motif["fragments"]]
    features.extend(("loss", mz, weight) for mz, weight in motif["losses"])
    total_query = float(peaks[:, 1].sum())
    candidates = []
    for fi, (kind, mz, weight) in enumerate(features):
        if kind == "loss" and not allow_losses:
            continue
        distances = np.abs((peaks[:, 0] if kind == "fragment" else precursor - peaks[:, 0]) - mz)
        for qi in np.flatnonzero(distances <= tolerance):
            candidates.append((-(weight * math.sqrt(peaks[qi, 1] / total_query)), float(distances[qi]), fi, int(qi)))
    candidates.sort()
    assigned_features, assigned_query, matches = set(), set(), []
    for _, error, fi, qi in candidates:
        if fi in assigned_features or qi in assigned_query:
            continue
        assigned_features.add(fi)
        assigned_query.add(qi)
        kind, mz, weight = features[fi]
        matches.append({"type": kind, "reference_mz": mz, "reference_weight": weight,
                        "query_index": int(original_indices[qi]), "query_mz": float(peaks[qi, 0]),
                        "query_intensity": float(peaks[qi, 1]), "error_da": error,
                        "query_difference_mz": float(precursor - peaks[qi, 0]) if kind == "loss" else None})
    total_weight = sum(x[2] for x in features)
    motif_coverage = sum(x["reference_weight"] for x in matches) / total_weight if total_weight else 0.
    query_coverage = float(peaks[list(assigned_query), 1].sum()) / total_query if assigned_query else 0.
    return matches, motif_coverage, query_coverage


def motif_candidates(library, peaks, precursor, tolerance, allow):
    """All batches merged; batches cover consecutive, non-overlapping motif positions."""
    merged, covered = {}, 0
    for following, batch in spectral_motifs._candidate_batches(
            library, "positive ionisation mode", peaks, precursor, tolerance, allow):
        assert all(covered <= k < following for k in batch) and following > covered
        merged.update(batch)
        covered = following
    return merged


@pytest.mark.parametrize("bound", [1, 64, spectral_motifs._BATCH_WINDOWS])
def test_motif_candidates_match_scalar_alignment_on_bundled_library(bound, monkeypatch):
    monkeypatch.setattr(spectral_motifs, "_BATCH_WINDOWS", bound)
    library = spectral_motifs._load_library()
    rng = np.random.default_rng(13)
    motifs = [m for m in library["motifs"] if m["analysis_polarity"] == "positive ionisation mode"]
    features = sorted({mz for m in motifs for mz, _ in m["fragments"]})
    for trial in range(12):
        # Peaks at, near and exactly one tolerance away from public feature masses.
        tolerance = [.02, .005, .3][trial % 3]
        chosen = rng.choice(features, size=60)
        masses = np.unique(np.round(chosen + rng.choice([0., tolerance, -tolerance, .001], size=60), 4))
        peaks = np.column_stack((masses, rng.uniform(1, 100, len(masses))))
        precursor = float(masses.max() + 20)
        allow = trial % 2 == 0
        candidates = motif_candidates(library, peaks, precursor, tolerance, allow)
        indices = np.arange(len(peaks)) * 2
        for k, motif in enumerate(motifs):
            expected = scalar_motif_align(motif, peaks, indices, precursor, tolerance, allow)
            got = spectral_motifs._align(motif, candidates[k], peaks, indices, precursor) if k in candidates else ([], 0., 0.)
            assert exact(got[0]) == exact(expected[0])
            if got[0]:
                assert exact(got[1:]) == exact(expected[1:])


@pytest.mark.parametrize("bound", [1, 64, spectral_motifs._BATCH_WINDOWS])
def test_motif_candidates_match_scalar_alignment_at_exact_boundaries_and_ties(bound, monkeypatch):
    monkeypatch.setattr(spectral_motifs, "_BATCH_WINDOWS", bound)
    # Binary-exact masses and a power-of-two tolerance hit |error| == tolerance
    # exactly; equal intensities and weights tie merits, exercising every key.
    rng = np.random.default_rng(17)
    tolerance = 1 / 64
    motifs = [dict(analysis_polarity="positive ionisation mode",
                   fragments=[[float(k) / 1024, float(w)] for k, w in zip(rng.integers(102400, 106496, 12), rng.choice([.5, 1.], 12))],
                   losses=[[float(k) / 1024, float(w)] for k, w in zip(rng.integers(10240, 30720, 6), rng.choice([.5, 1.], 6))])
              for _ in range(60)]
    library = dict(motifs=motifs)
    for trial in range(20):
        masses = np.unique(rng.integers(102400, 106496, 80)) / 1024
        peaks = np.column_stack((masses, rng.choice([1., 4.], len(masses))))
        precursor = 150.
        allow = trial % 2 == 0
        candidates = motif_candidates(library, peaks, precursor, tolerance, allow)
        indices = np.arange(len(peaks))
        for k, motif in enumerate(motifs):
            expected = scalar_motif_align(motif, peaks, indices, precursor, tolerance, allow)
            got = spectral_motifs._align(motif, candidates[k], peaks, indices, precursor) if k in candidates else ([], 0., 0.)
            assert exact(got[0]) == exact(expected[0])
            if got[0]:
                assert exact(got[1:]) == exact(expected[1:])


def test_dense_motif_query_builds_bounded_cancellable_batches(monkeypatch):
    # Duplicate masses near a common fragment and water loss: every matching
    # feature's window spans all peaks, so whole-table construction is large.
    query = dict(peaks=[[145.082, 1.]] * 3000, precursor_mz=163.094, charge=1, polarity=1)
    settings = dict(fragment_da=.02, relative_noise=0, min_peaks=1)
    monkeypatch.setattr(spectral_motifs, "_BATCH_WINDOWS", 1 << 30)
    expected = spectral_motifs.search_motifs(query, settings)
    monkeypatch.setattr(spectral_motifs, "_BATCH_WINDOWS", 8000)
    actual = spectral_motifs._candidates
    sizes = []

    def candidates(rows, *args):
        motif, counts = rows[0], rows[6]
        assert counts.sum() <= 8000 or len(np.unique(motif)) == 1
        sizes.append(int(counts.sum()))
        return actual(rows, *args)

    monkeypatch.setattr(spectral_motifs, "_candidates", candidates)
    assert exact(spectral_motifs.search_motifs(query, settings)) == exact(expected)
    assert len(sizes) > 10 and sum(sizes) > 8000
    sizes.clear()
    with pytest.raises(Cancelled):
        spectral_motifs.search_motifs(query, settings, cancel=lambda: bool(sizes))
    assert len(sizes) == 1


def test_save_encodes_once_with_unchanged_errors(tmp_path):
    from Tests.Viewer.test_spectral_gui import analysis
    data = analysis()
    path = tmp_path / "a.kmzspectra"
    save_spectral_analysis(path, data)
    assert path.read_bytes() == json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="incomplete or malformed"):
        save_spectral_analysis(path, dict(data, extra=object()))
    with pytest.raises(ValueError, match="not JSON compliant"):
        save_spectral_analysis(path, dict(data, extra=[float("inf")]))


def test_repopulating_visible_table_keeps_view_state():
    app = QApplication.instance() or QApplication([])
    from kmzviewer.gui.spectral import SpectralWindow, table
    widget = table(["Reference molecule", "Score", "Ions", "Query %", "Library"])
    widget.resize(500, 300)
    widget.show()
    rows = [[f"Reference {i}", f"{i / 60:.3f}", i % 9, i, "Library"] for i in range(60)]
    SpectralWindow._populate(widget, rows)
    app.processEvents()
    widget.selectRow(4)
    widget.verticalScrollBar().setValue(20)
    app.processEvents()
    before = (widget.currentRow(), widget.verticalScrollBar().value(),
              [widget.horizontalHeader().sectionSize(c) for c in range(5)])
    SpectralWindow._populate(widget, [[f"Longer replacement name {i}", "0.500", 3, i, "Other library"] for i in range(60)])
    app.processEvents()
    assert (widget.currentRow(), widget.verticalScrollBar().value()) == before[:2]
    assert widget.item(59, 0).text() == "Longer replacement name 59"
    assert widget.horizontalHeader().sectionSize(4) > before[2][4]  # resized to new contents
    widget.close()
