"""Local spectral-neighbor retrieval with separate, interpretable score channels.

Flash Entropy scores rank spectra; they are not identification probabilities.
The displayed one-to-one peak alignment is independently calculated evidence,
not a claim to expose the scorer's internal peak assignment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from kmzcore.cancel import Cancelled
from kmzcore.qualified import qualified


@dataclass(frozen=True)
class SpectralSettings:
    fragment_da: float = .02
    relative_noise: float = .01
    min_peaks: int = 3
    required_fragments: tuple[float, ...] = ()
    required_losses: tuple[float, ...] = ()
    expanded: bool = False

    def validate(self):
        if not math.isfinite(self.fragment_da) or not .001 <= self.fragment_da <= .1:
            raise ValueError("Fragment tolerance must be between 0.001 and 0.1 Da.")
        if not math.isfinite(self.relative_noise) or not 0 <= self.relative_noise <= .1:
            raise ValueError("Relative noise cutoff must be between 0 and 0.1.")
        if type(self.min_peaks) is not int or not 1 <= self.min_peaks <= 100:
            raise ValueError("Minimum matched peaks must be an integer from 1 to 100.")
        if type(self.expanded) is not bool:
            raise ValueError("Expanded-library selection must be true or false.")
        for values in (self.required_fragments, self.required_losses):
            if not isinstance(values, (tuple, list)) or len(values) > 20 or any(not math.isfinite(v) or not 0 < v <= 10000 for v in values):
                raise ValueError("Fragment and loss filters accept at most 20 positive finite masses.")
        return self


def _check(cancel):
    if cancel is not None and cancel():
        raise Cancelled()


def _progress(callback, fraction, message):
    if callback is not None:
        callback(float(fraction), message)


def _separation(settings):
    return max(.05, 2 * settings.fragment_da + .00001)


def prepare_search_peaks(query, settings):
    from ms_entropy import clean_spectrum
    settings.validate()
    peaks = np.asarray(query["peaks"], dtype=np.float64)
    precursor = float(query["precursor_mz"])
    if (peaks.ndim != 2 or peaks.shape[1] != 2 or not len(peaks) or len(peaks) > 100_000
            or not np.all(np.isfinite(peaks)) or np.any(peaks <= 0)
            or not math.isfinite(precursor) or precursor <= 0):
        raise ValueError("A finite positive precursor and MS2 peak list are required.")
    charge = query.get("charge")
    if charge == 0:
        charge = None  # Zero is the conventional unreported-charge sentinel.
    if charge is not None and (type(charge) is not int or not 1 <= abs(charge) <= 100):
        raise ValueError("The reported precursor charge is invalid.")
    # Fragments of multiply charged precursors can exceed precursor m/z.
    cutoff = precursor - 1.6 if charge is None or abs(charge) <= 1 else -1
    prepared = clean_spectrum(peaks.copy(), min_mz=0, max_mz=cutoff,
                              noise_threshold=settings.relative_noise,
                              min_ms2_difference_in_da=_separation(settings), normalize_intensity=True)
    if not len(prepared):
        raise ValueError("No MS2 fragments remain after the search preparation settings.")
    warnings = []
    if query.get("representation") == "profile":
        warnings.append("Profile apex samples are experimental inputs. Search-only peak merging does not establish qualified centroid masses.")
    if charge is None:
        warnings.append("Charge is unreported. Precursor-difference matching is exploratory and assumes compatible singly charged ions; differences are in m/z, not confirmed neutral masses.")
    elif abs(charge) > 1:
        warnings.append("Multiple charge reported: precursor-difference and hybrid searches are disabled; direct fragments remain available.")
    description = (f"Search copy: merge neighboring peaks within {_separation(settings):.5g} Da, "
                   f"retain ≥{settings.relative_noise * 100:.3g}% of the merged base peak, "
                   + ("remove ions above precursor − 1.6 Da, " if cutoff > 0 else "")
                   + "normalize summed intensity. Original spectrum is unchanged.")
    return dict(peaks=prepared, original_count=len(peaks), retained_count=len(prepared), description=description, warnings=warnings)


def align_peaks(query_peaks, reference_peaks, query_precursor, reference_precursor, tolerance, mode="hybrid"):
    """Greedy one-to-one alignment, prioritizing intensity products.

    Returned coverage is a fraction of summed *prepared* intensity. A shifted
    pair means equal precursor-to-fragment m/z differences within tolerance.
    Reference peaks are m/z-sorted, as prepared peaks are.
    """
    q = np.asarray(query_peaks, dtype=float)
    r = np.asarray(reference_peaks, dtype=float)
    delta = float(query_precursor - reference_precursor)
    shifts = [("direct", 0.)] if mode != "neutral_loss" else []
    if mode in {"hybrid", "neutral_loss"} and (mode == "neutral_loss" or abs(delta) > tolerance):
        shifts.append(("shifted", delta))
    # Candidate windows for every query peak at once; the right bound includes
    # an exact upper-bound match. Arithmetic order matches the scalar form.
    parts = []
    for kind, shift in shifts:
        centre = q[:, 0] - shift
        left = np.searchsorted(r[:, 0], centre - tolerance)
        counts = np.searchsorted(r[:, 0], centre + tolerance, side="right") - left
        i = np.repeat(np.arange(len(q)), counts)
        j = np.arange(len(i)) + np.repeat(left - (np.cumsum(counts) - counts), counts)
        parts.append((i, j, np.full(len(i), kind != "direct"), q[i, 1] * r[j, 1], q[i, 0] - r[j, 0] - shift))
    i, j, shifted, product, error = (np.concatenate(x) for x in zip(*parts))
    # Keys are unique per (i, j, kind), so this equals the stable tuple sort on
    # (-product, kind != "direct", abs(error), i, j).
    order = np.lexsort((j, i, np.abs(error), shifted, -product))
    qmz, qy, rmz, ry = q[:, 0].tolist(), q[:, 1].tolist(), r[:, 0].tolist(), r[:, 1].tolist()
    used_q, used_r, pairs = set(), set(), []
    for i, j, shifted, error in zip(*(x[order].tolist() for x in (i, j, shifted, error))):
        if i in used_q or j in used_r:
            continue
        used_q.add(i)
        used_r.add(j)
        pairs.append(dict(query_index=i, reference_index=j, query_mz=qmz[i], reference_mz=rmz[j],
                          kind="shifted" if shifted else "direct", error_da=error,
                          query_intensity=qy[i], reference_intensity=ry[j]))
    return dict(pairs=pairs, direct_count=sum(p["kind"] == "direct" for p in pairs),
                shifted_count=sum(p["kind"] == "shifted" for p in pairs),
                query_coverage=float(sum(q[i, 1] for i in used_q) / q[:, 1].sum()),
                reference_coverage=float(sum(r[i, 1] for i in used_r) / r[:, 1].sum()),
                description="Local one-to-one intensity-product alignment; coverage uses prepared summed intensity.")


def _required_match(peaks, precursor, settings):
    masses = peaks[:, 0]
    for target in settings.required_fragments:
        if not np.any(np.abs(masses - target) <= settings.fragment_da):
            return False
    for target in settings.required_losses:
        if not np.any(np.abs(precursor - masses - target) <= settings.fragment_da):
            return False
    return True


def _required_matches(corpus, settings):
    """``_required_match`` for the whole corpus with the same float32 arithmetic."""
    masses, starts = corpus["peaks"][:, 0], corpus["starts"]
    eligible = np.ones(len(starts), dtype=bool)
    for target in settings.required_fragments:
        eligible &= np.logical_or.reduceat(np.abs(masses - target) <= settings.fragment_da, starts)
    if settings.required_losses:
        precursor = np.repeat(corpus["precursor"], corpus["counts"])
        for target in settings.required_losses:
            eligible &= np.logical_or.reduceat(np.abs(precursor - masses - target) <= settings.fragment_da, starts)
    return eligible


def _corpus(pack, spectra):
    """Concatenated prepared peaks (spectra keep row views) plus per-spectrum facts."""
    counts = np.fromiter((len(s["peaks"]) for s in spectra), dtype=np.int64, count=len(spectra))
    starts = np.cumsum(counts) - counts
    charges = [pack.metadata[s["pack_index"]].get("charge") for s in spectra]
    corpus = dict(counts=counts, starts=starts, single=np.array([c is None or abs(c) == 1 for c in charges], dtype=bool),
                  vector=all(type(s["precursor_mz"]) is float and s["peaks"].dtype == np.float32 for s in spectra))
    if corpus["vector"]:
        # float32 precursors reproduce the weak Python-float arithmetic of the scalar forms.
        peaks = np.concatenate([s["peaks"] for s in spectra])
        for s, start, count in zip(spectra, starts.tolist(), counts.tolist()):
            s["peaks"] = peaks[start:start + count]
        within = np.ones(len(peaks) - 1, dtype=bool)
        within[starts[1:] - 1] = False
        gaps = (peaks[1:, 0] - peaks[:-1, 0])[within]
        corpus.update(peaks=peaks, precursor=np.array([s["precursor_mz"] for s in spectra], dtype=np.float32),
                      gap=gaps.min() if len(gaps) else None)
    return corpus


def _flash_ready(corpus, tolerance):
    # _flash_index reproduces the qualified ms-entropy index layout and NumPy's sorts.
    # ms-entropy's build/search input check: adjacent prepared peaks > 2 × tolerance.
    return (qualified("ms-entropy", "numpy") and corpus["vector"]
            and (corpus["gap"] is None or corpus["gap"] > tolerance * 2))


def _order_key(values):
    """uint32 key ordered as the float32 values (−0 joins +0); NaN is excluded by callers."""
    bits = (values + np.float32(0)).view(np.uint32)
    return np.where(bits >> 31, ~bits, bits | np.uint32(1 << 31))


def _stable_order(keys):
    """Stable argsort over uint32 keys, least significant first (16-bit radix passes)."""
    order = None
    for key in keys:
        for shift in (0, 16):
            step = np.argsort(((key if order is None else key[order]) >> shift).astype(np.uint16), kind="stable")
            order = step if order is None else order[step]
    return order


def _flash_index(corpus, tolerance, max_indexed_mz):
    """Vectorized ``FlashEntropySearch.build_index(clean_spectra=False)`` (ms-entropy 1.5.3).

    Input is already precursor-ordered. Index arrays, dtypes and strides are
    byte-identical, including the structured sorts' tie order (remaining fields
    in dtype order, then spectrum order). Pickled per-spectrum metadata is
    omitted: this private engine only scores. None means use the library.
    """
    from ms_entropy import FlashEntropySearch
    if not _flash_ready(corpus, tolerance) or len(corpus["counts"]) >= 2**32 - 1:
        return None
    peaks, counts, starts = corpus["peaks"], corpus["counts"], corpus["starts"]
    # The library asserts summed intensity within 1e-4 of one; leave doubt to it.
    if np.any(np.abs(np.add.reduceat(peaks[:, 1].astype(np.float64), starts) - 1) >= 5e-5):
        return None
    # apply_weight_to_intensity per spectrum, grouped by peak count. Rows keep the
    # strided intensity column and reduce with the same per-row pairwise sums.
    intensity = peaks[:, 1].copy()
    for n in np.unique(counts):
        cells = starts[counts == n][:, None] + np.arange(n)
        block = peaks[cells]
        entropy = -np.sum(block[:, :, 1] * np.log(block[:, :, 1]), axis=1)
        low = entropy < 3
        if low.any():
            block = block[low]
            block[:, :, 1] = np.power(block[:, :, 1], (.25 + .25 * entropy[low])[:, None])
            block[:, :, 1] /= np.sum(block[:, :, 1], axis=1)[:, None]
            intensity[cells[low]] = block[:, :, 1]
    ion = peaks[:, 0]
    intensity /= 2
    nl = np.repeat(corpus["precursor"], counts) - ion
    spec = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)
    if not (np.isfinite(ion).all() and np.isfinite(nl).all() and np.isfinite(intensity).all()):
        return None
    # peak_data.sort(order="ion_mz"), then .sort(order="nl_mass") on ion-ordered rows.
    order = _stable_order((_order_key(intensity), _order_key(nl), _order_key(ion)))
    ion, nl, intensity, spec = ion[order], nl[order], intensity[order], spec[order]
    order = _stable_order((_order_key(nl),))
    data = np.zeros(len(ion), dtype=np.dtype([("ion_mz", np.float32), ("nl_mass", np.float32), ("intensity", np.float32),
                                              ("spec_idx", np.uint32), ("peak_idx", np.uint64)], align=True))
    data["ion_mz"], data["nl_mass"], data["intensity"], data["spec_idx"] = ion[order], nl[order], intensity[order], spec[order]
    data["peak_idx"] = order
    engine = FlashEntropySearch(max_ms2_tolerance_in_da=tolerance, mz_index_step=.001)
    engine.precursor_mz_array, engine.metadata = corpus["precursor"], None
    core = engine.entropy_search
    core.total_spectra_num, core.total_peaks_num = len(counts), len(ion)
    starts = []
    for values in (ion, data["nl_mass"]):
        max_mz = min(np.max(values), max_indexed_mz)
        starts.append(np.searchsorted(values, np.arange(0.0, max_mz, core.mz_index_step), side="left").astype(np.int64))
    core.index = [starts[0], ion, intensity, spec, starts[1], data["nl_mass"], data["intensity"], data["spec_idx"], data["peak_idx"]]
    return engine


def _with_tolerance(engine, corpus, tolerance):
    """The same index under another tolerance; ms-entropy uses it only in input checks."""
    from ms_entropy import FlashEntropySearch
    if not _flash_ready(corpus, tolerance):
        return None
    shared = FlashEntropySearch(max_ms2_tolerance_in_da=tolerance, mz_index_step=.001)
    shared.__dict__.update((k, v) for k, v in vars(engine).items() if k != "entropy_search")
    for name in ("index", "total_spectra_num", "total_peaks_num"):
        setattr(shared.entropy_search, name, getattr(engine.entropy_search, name))
    return shared


def _engine(pack, settings, polarity, cancel, progress):
    """Prepared corpus and index, reused while the reference preparation is unchanged.

    Preparation depends on merge separation, noise and polarity; tolerances that
    keep the same separation share it.
    """
    from ms_entropy import FlashEntropySearch, clean_spectrum
    key = (_separation(settings), settings.relative_noise, polarity)
    cached = getattr(pack, "_search_cache", None)
    if cached is not None and cached[0] == key:
        _, engine, spectra, corpus = cached
        if engine.entropy_search.max_ms2_tolerance_in_da == settings.fragment_da:
            return engine, spectra, corpus
        engine = _with_tolerance(engine, corpus, settings.fragment_da)
        if engine is not None:
            pack._search_cache = (key, engine, spectra, corpus)
            return engine, spectra, corpus
    spectra = []
    for index, metadata in enumerate(pack.metadata):
        if index % 256 == 0:
            _check(cancel)
            _progress(progress, .65 * index / max(1, len(pack.metadata)), "Preparing reference spectra for local search")
        if metadata["polarity"] != polarity:
            continue
        charge = metadata.get("charge")
        cutoff = metadata["precursor_mz"] - 1.6 if charge is None or abs(charge) <= 1 else -1
        peaks = clean_spectrum(pack.spectrum(index).copy(), min_mz=0, max_mz=cutoff,
                               noise_threshold=settings.relative_noise,
                               min_ms2_difference_in_da=_separation(settings), normalize_intensity=True)
        if len(peaks):
            spectra.append(dict(precursor_mz=metadata["precursor_mz"], peaks=peaks, pack_index=index))
    _check(cancel)
    if not spectra:
        raise ValueError("The reference pack has no usable spectra with the query polarity.")
    spectra.sort(key=lambda x: x["precursor_mz"])  # build_index's stable precursor order
    corpus = _corpus(pack, spectra)
    _progress(progress, .7, "Indexing public fragments and precursor differences")
    engine = _flash_index(corpus, settings.fragment_da, 2000.00005)
    if engine is None:
        engine = FlashEntropySearch(max_ms2_tolerance_in_da=settings.fragment_da, mz_index_step=.001)
        # Only locally constructed metadata enter the scorer. Its Python object
        # representation is never persisted or read from external serialized objects.
        spectra = engine.build_index(spectra, clean_spectra=False, max_indexed_mz=2000.00005)
    _check(cancel)
    pack._search_cache = (key, engine, spectra, corpus)
    return engine, spectra, corpus


def search_reference_pack(query, settings, pack, cancel=None, progress=None):
    """Search the full polarity-compatible corpus, then group the best evidence.

    Required fragment/loss conditions apply before top-hit truncation. They use
    the same prepared reference peaks and tolerance as matching; this is a local
    AND filter, not an invocation of the separate MassQL engine.
    """
    import ms_entropy
    settings.validate()
    _check(cancel)
    polarity = query.get("polarity")
    if polarity not in (-1, 1):
        raise ValueError("A reported positive or negative polarity is required for reference matching.")
    prepared = prepare_search_peaks(query, settings)
    q = prepared["peaks"]
    qp = float(query["precursor_mz"])
    engine, spectra, corpus = _engine(pack, settings, polarity, cancel, progress)
    _progress(progress, .8, "Searching direct fragments and precursor differences")
    started = time.monotonic()
    direct = engine.open_search(q, ms2_tolerance_in_da=settings.fragment_da)
    _check(cancel)
    charge = query.get("charge") or None
    allow_shift = charge is None or abs(charge) == 1
    if not allow_shift and settings.required_losses:
        raise ValueError("Precursor-difference filters require singly charged ions; the query reports multiple charge.")
    scores = {"direct": direct}
    if allow_shift:
        scores["neutral_loss"] = engine.neutral_loss_search(qp, q, ms2_tolerance_in_da=settings.fragment_da)
        _check(cancel)
        scores["hybrid"] = engine.hybrid_search(qp, q, ms2_tolerance_in_da=settings.fragment_da)
    search_seconds = time.monotonic() - started
    _check(cancel)
    shifted_eligible = corpus["single"]
    eligible = np.ones(len(spectra), dtype=bool)
    if settings.required_fragments or settings.required_losses:
        eligible = (_required_matches(corpus, settings) if corpus["vector"] else
                    np.array([_required_match(s["peaks"], s["precursor_mz"], settings) for s in spectra], dtype=bool))
        if settings.required_losses:
            eligible &= shifted_eligible
        _check(cancel)
    methods = {"direct": [], "hybrid": [], "neutral_loss": []}
    references = {}
    for mode, values in scores.items():
        seen = set()
        # Spectrum ranks before the first nonpositive score, restricted to eligible spectra.
        order = np.argsort(-values, kind="stable")
        stop = np.flatnonzero(values[order] <= 0)
        order = order[:stop[0] if len(stop) else len(order)]
        ranks = np.flatnonzero((eligible if mode == "direct" else eligible & shifted_eligible)[order]) + 1
        for visited, (rank, i) in enumerate(zip(ranks.tolist(), order[ranks - 1].tolist())):
            if visited % 128 == 0:
                _check(cancel)
            spectrum = spectra[i]
            metadata = pack.metadata[spectrum["pack_index"]]
            key = metadata["structure_key"]
            if key in seen:
                continue
            alignment = align_peaks(q, spectrum["peaks"], qp, spectrum["precursor_mz"], settings.fragment_da, mode)
            if len(alignment["pairs"]) < settings.min_peaks:
                continue
            seen.add(key)
            accession = metadata["accession"]
            hit = dict(metadata, scores={m: float(v[i]) if m == "direct" or shifted_eligible[i] else None for m, v in scores.items()},
                       score=float(values[i]), mode=mode, spectrum_rank=rank, rank=len(methods[mode]) + 1,
                       matched_peaks=len(alignment["pairs"]), alignment=alignment,
                       coverage=alignment["query_coverage"], delta_mass=qp - spectrum["precursor_mz"])
            methods[mode].append(hit)
            references[accession] = dict(metadata, peaks=spectrum["peaks"].tolist(),
                                         original_peak_count=len(pack.spectrum(spectrum["pack_index"])),
                                         preparation="Prepared reference peaks used for local scoring; original public spectrum is available through its accession.")
            if len(methods[mode]) >= 50:
                break
    _check(cancel)
    _progress(progress, 1., "Local spectral matching complete")
    return dict(methods=methods, references=references, prepared_query=q.tolist(),
                preparation={key: value for key, value in prepared.items() if key != "peaks"},
                provenance=dict(pack_schema=pack.provenance.get("schema", ""), pack_created=pack.provenance.get("created", ""),
                                libraries=pack.provenance.get("libraries", []), reference_spectra=len(spectra),
                                filtered_reference_spectra=int(eligible.sum()), polarity=polarity,
                                engine="Flash Entropy Search", engine_version=ms_entropy.__version__,
                                search_seconds=search_seconds, settings=asdict(settings),
                                grouping="Best qualifying spectrum per provided InChIKey, InChI, or SMILES string; distinct energies remain searchable in the corpus.",
                                score_interpretation="Spectral similarity, not identity or a calibrated probability.",
                                difference_interpretation="Precursor-to-fragment m/z differences; neutral losses require compatible singly charged ions."))
