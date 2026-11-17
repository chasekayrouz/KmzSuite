"""Local, interpretable fragment/loss overlap with a pinned public MotifDB.

This is an exploratory alignment of existing motifs, not MS2LDA topic inference,
chemical classification, or a calibrated identification probability. It needs no
model, account, network request, or compound identity. Acquisitions and the saved
query remain unchanged.
"""
from __future__ import annotations

from functools import lru_cache
import gzip
import hashlib
from importlib.resources import files
import json
import math
from urllib.parse import quote

import numpy as np

from kmzviewer.gnps import check_cancel, finite, validate_peaks

_RESOURCE = "data/motifdb.json.gz"
_RESOURCE_SHA256 = "6ca7efe04e30b66b90b97cca777d651b420861346a59ed8e4ba55c7efab9c78c"
_MAX_HITS = 50


def _setting(settings, name, default):
    return settings.get(name, default) if isinstance(settings, dict) else getattr(settings, name, default)


@lru_cache(maxsize=1)
def _load_library():
    payload = files("kmzviewer").joinpath(_RESOURCE).read_bytes()
    if hashlib.sha256(payload).hexdigest() != _RESOURCE_SHA256:
        raise ValueError("The bundled MotifDB resource failed its integrity check.")
    library = json.loads(gzip.decompress(payload))
    if library.get("schema") != "kmzviewer-motifdb-v1" or len(library.get("motifs", [])) != 996:
        raise ValueError("Unsupported bundled MotifDB resource.")
    return library


def _prepare(query, relative_noise):
    peaks = np.asarray(validate_peaks(query["peaks"]), dtype=np.float64)
    precursor = query.get("precursor_mz")
    if not finite(precursor) or precursor <= 0:
        raise ValueError("Motif search requires a positive precursor m/z.")
    # Preserve source-derived masses and intensities: no centroiding or merging.
    indices = np.flatnonzero((peaks[:, 0] < precursor - 1.) &
                             (peaks[:, 1] >= relative_noise * peaks[:, 1].max()))
    preparation = {
        "description": "Native query peaks filtered on a copy; no merging or centroiding. "
                       "Peaks within 1 Da of, or above, the precursor are excluded.",
        "source_preparation": query.get("preparation", ""),
        "original_count": len(peaks), "retained_count": len(indices),
        "relative_noise": relative_noise, "precursor_exclusion_da": 1.,
        "required_features_scope": "Selected fragment/loss requirements filter reference spectra, not motif suggestions.",
    }
    return peaks[indices].copy(), indices, preparation


def _features(motif):
    features = [("fragment", mz, weight) for mz, weight in motif["fragments"]]
    features.extend(("loss", mz, weight) for mz, weight in motif["losses"])
    return features


_tables = {}


def _table(library, polarity):
    """Flattened features of one polarity: motif position, feature index, loss flag, m/z, weight."""
    cached = _tables.get(polarity)
    if cached is None or cached[0] is not library:
        rows = [(k, fi, kind == "loss", mz, weight) for k, motif in
                enumerate(m for m in library["motifs"] if m["analysis_polarity"] == polarity)
                for fi, (kind, mz, weight) in enumerate(_features(motif))]
        columns = [np.array(c, dtype=t) for c, t in zip(zip(*rows), (np.int64, np.int64, bool, float, float))] if rows else \
            [np.zeros(0, dtype=t) for t in (np.int64, np.int64, bool, float, float)]
        cached = _tables[polarity] = (library, columns)
    return cached[1]


# Query-window entries examined per candidate batch. Dense queries never build
# the whole polarity's table at once, and cancellation is checked between batches.
_BATCH_WINDOWS = 1 << 16


def _candidates(rows, peaks, differences, total_query, tolerance):
    """Every (feature, query ion) within tolerance, per motif position, in assignment order.

    Equivalent to scanning each feature's distances: sorted query masses give
    padded windows, then the scalar form's exact distance test applies. Order is
    (-merit, mass error, feature, query ion), with merit as in the scalar form.
    """
    motif, feature, loss, mz, weight, left, counts = rows
    f = np.repeat(np.arange(len(mz)), counts)
    qi = np.arange(len(f)) + np.repeat(left - (np.cumsum(counts) - counts), counts)
    distance = np.abs(np.where(loss[f], differences[qi], peaks[qi, 0]) - mz[f])
    keep = distance <= tolerance
    f, qi, distance = f[keep], qi[keep], distance[keep]
    merit = weight[f] * np.sqrt(peaks[qi, 1] / total_query)
    order = np.lexsort((qi, feature[f], distance, -merit, motif[f]))
    positions, fi, qi, distance = motif[f][order], feature[f][order].tolist(), qi[order].tolist(), distance[order].tolist()
    bounds = (np.flatnonzero(np.diff(positions)) + 1).tolist()
    return {int(positions[s]): (fi[s:e], qi[s:e], distance[s:e])
            for s, e in zip([0] + bounds, bounds + [len(fi)]) if s < e}


def _candidate_batches(library, polarity, peaks, precursor, tolerance, allow_losses):
    """Yield (next batch's first motif position, candidates) over consecutive motifs.

    Nothing is computed until the first batch is requested. Batches hold whole
    motifs, so each motif keeps its exact candidate order and global query
    indices; a motif with more windows than the bound forms its own batch.
    """
    rows = _table(library, polarity)
    if not allow_losses:
        rows = [a[~rows[2]] for a in rows]
    motif, _, loss, mz, _ = rows
    masses, differences = peaks[:, 0], precursor - peaks[:, 0]
    centre = np.where(loss, precursor - mz, mz)
    left = np.searchsorted(masses, centre - tolerance - 1e-6)
    counts = np.searchsorted(masses, centre + tolerance + 1e-6, side="right") - left
    rows = [*rows, left, counts]
    total_query = float(peaks[:, 1].sum())
    edges = np.concatenate(([0], np.flatnonzero(np.diff(motif)) + 1, [len(motif)]))
    windows = np.concatenate(([0], np.cumsum(counts)))[edges]
    i = 0
    while i < len(edges) - 1:
        j = max(i + 1, int(np.searchsorted(windows, windows[i] + _BATCH_WINDOWS, side="right")) - 1)
        s, e = edges[i], edges[j]
        yield (int(motif[e]) if e < len(motif) else math.inf,
               _candidates([a[s:e] for a in rows], peaks, differences, total_query, tolerance))
        i = j


def _align(motif, candidates, peaks, original_indices, precursor):
    """Greedy weighted one-to-one alignment across both feature channels.

    A query ion can contribute only once, even if it matches a fragment and a
    precursor-relative loss. Weights and coverage are counted only for assigned
    pairs. Ties use mass error then stable source/query order.
    """
    features = _features(motif)
    total_query = float(peaks[:, 1].sum())
    assigned_features, assigned_query, matches = set(), set(), []
    for fi, qi, error in zip(*candidates):
        if fi in assigned_features or qi in assigned_query:
            continue
        assigned_features.add(fi)
        assigned_query.add(qi)
        kind, mz, weight = features[fi]
        matches.append({
            "type": kind, "reference_mz": mz, "reference_weight": weight,
            "query_index": int(original_indices[qi]), "query_mz": float(peaks[qi, 0]),
            "query_intensity": float(peaks[qi, 1]), "error_da": error,
            "query_difference_mz": float(precursor - peaks[qi, 0]) if kind == "loss" else None,
        })
    total_weight = sum(x[2] for x in features)
    motif_coverage = sum(x["reference_weight"] for x in matches) / total_weight if total_weight else 0.
    query_coverage = float(peaks[list(assigned_query), 1].sum()) / total_query if assigned_query else 0.
    return matches, motif_coverage, query_coverage


def search_motifs(query, settings, cancel=None):
    """Return ranked motif-overlap evidence from a GNPS-style local query dict.

    ``settings`` accepts either a mapping or an object exposing ``fragment_da``,
    ``relative_noise`` and ``min_peaks``. Scores rank exploration only. The result
    includes portable reference features and aligned pairs for manual inspection.
    Unknown polarity returns an explicit skipped result; no polarity is inferred.
    """
    check_cancel(cancel)
    tolerance = _setting(settings, "fragment_da", .02)
    noise = _setting(settings, "relative_noise", .01)
    minimum = _setting(settings, "min_peaks", 3)
    if not finite(tolerance) or not 0 < tolerance <= 1:
        raise ValueError("Motif mass tolerance must be greater than zero and at most 1 Da.")
    if not finite(noise) or not 0 <= noise < 1:
        raise ValueError("Relative noise must be between zero (inclusive) and one (exclusive).")
    if type(minimum) is not int or not 1 <= minimum <= 1000:
        raise ValueError("Minimum matched peaks must be an integer from 1 to 1000.")
    polarity = query.get("polarity", 0)
    if polarity not in (-1, 0, 1, None):
        raise ValueError("Invalid query polarity.")
    peaks, original_indices, preparation = _prepare(query, noise)
    library = _load_library()
    provenance = {k: library[k] for k in ("source", "source_url", "source_commit", "source_sha256", "retrieved", "license")}
    provenance.update({
        "method": "Motif overlap", "method_version": 1,
        "score_definition": "motif weight coverage × square root of unique query intensity coverage",
        "score_interpretation": "Exploratory overlap rank, not a probability or MS2LDA inference.",
        "fragment_da": tolerance, "min_peaks": minimum,
        "database_records": len(library["motifs"]), "searched_records": 0,
    })
    result = dict(hits=[], provenance=provenance, preparation=preparation, warnings=[])
    if not polarity:
        result["warnings"].append("Motif search skipped because the scan polarity is unknown.")
        return result
    charge = query.get("charge")
    if charge is not None and (type(charge) is not int or abs(charge) > 100):
        raise ValueError("Invalid query precursor charge.")
    allow_losses = charge in (None, 0, -1, 1)
    if charge in (None, 0):
        result["warnings"].append("Loss matches are precursor-to-fragment m/z differences; a neutral-mass interpretation assumes singly charged ions.")
    elif not allow_losses:
        result["warnings"].append("Loss alignment is disabled for a multiply charged precursor because fragment charges are unknown.")
    if not len(peaks):
        result["warnings"].append("No peaks remain after the motif search filters.")
        return result
    expected_polarity = "positive ionisation mode" if polarity == 1 else "negative ionisation mode"
    batches = _candidate_batches(library, expected_polarity, peaks, query["precursor_mz"], tolerance, allow_losses)
    covered, candidates = 0, {}
    hits = []
    for motif in library["motifs"]:
        check_cancel(cancel)
        if motif["analysis_polarity"] != expected_polarity:
            continue
        position = provenance["searched_records"]
        provenance["searched_records"] += 1
        # The next batch is built only after the cancellation check above.
        while position >= covered:
            covered, candidates = next(batches, (math.inf, {}))
        # A motif without candidate pairs has no matches; minimum is at least one.
        pairs = candidates.get(position)
        if pairs is None:
            continue
        matches, motif_coverage, query_coverage = _align(motif, pairs, peaks, original_indices, query["precursor_mz"])
        if len(matches) < minimum:
            continue
        annotation = str(motif.get("annotation") or motif.get("short_annotation") or "Unannotated motif")
        set_name = motif["motifset"]
        source_url = f"{library['source_url']}/" + quote(f"Motifset {set_name}.json", safe="")
        hit = {
            "motif_id": motif["motif_id"], "motifset": set_name,
            "name": annotation, "annotation": annotation,
            "score": min(1., motif_coverage * math.sqrt(query_coverage)),
            "matched_peaks": len(matches), "matched_fragments": sum(x["type"] == "fragment" for x in matches),
            "matched_losses": sum(x["type"] == "loss" for x in matches),
            "motif_coverage": min(1., motif_coverage), "query_coverage": min(1., query_coverage),
            "motif_feature_count": len(motif["fragments"]) + len(motif["losses"]),
            "source_url": source_url, "annotation_url": str(motif.get("paper_url") or source_url),
            "instrument": str(motif.get("analysis_massspectrometer") or ""),
            "collision_energy": str(motif.get("collision_energy") or ""),
            "source_information": str(motif.get("other_information") or ""),
            "fragment_peaks": [list(p) for p in motif["fragments"]],
            "loss_peaks": [list(p) for p in motif["losses"]], "matches": matches,
            "interpretation": "Source motif annotation is a hypothesis to inspect; matching part of a motif does not establish its complete named scaffold.",
        }
        hits.append(hit)
    hits.sort(key=lambda h: (-h["score"], h["motifset"], h["motif_id"]))
    provenance["eligible_hits"] = len(hits)
    provenance["display_limit"] = _MAX_HITS
    for rank, hit in enumerate(hits[:_MAX_HITS], 1):
        hit["rank"] = rank
    result["hits"] = hits[:_MAX_HITS]
    check_cancel(cancel)
    return result
