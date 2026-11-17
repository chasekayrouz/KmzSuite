"""Explicit, portable spectral-exploration snapshots; never an automatic history."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

from kmzcore.version import __version__
from kmzviewer.gnps import SearchSettings, new_analysis, validate_analysis, validate_peaks, timestamp

SCHEMA = "kmzviewer-spectral-v1"
MAX_BYTES = 64 * 1024 * 1024


def new_spectral_analysis(query, settings, library, motifs, learned=None):
    return validate_spectral_analysis(dict(schema=SCHEMA, version=__version__, created=timestamp(),
        query=query, settings=asdict(settings), library=library, motifs=motifs, learned=learned or {}, structures={}))


def _number(value, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or minimum is not None and value < minimum or maximum is not None and value > maximum):
        raise ValueError("Invalid numeric match evidence.")


def _text(value, maximum=16384):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("Invalid saved match metadata.")


def _validate_hit(hit, *, motif=False, score_mode=None):
    if not isinstance(hit, dict):
        raise ValueError("Invalid saved match.")
    _text(hit.get("name"))
    for key in ("library", "accession", "usi", "adduct", "smiles", "provenance", "motifset"):
        if key in hit:
            _text(hit[key])
    if "score" in hit:
        _number(hit["score"], -1 if not motif else 0, 1.000001)
    if "precursor_mz" in hit:
        _number(hit["precursor_mz"], 0)
    if "matched_peaks" in hit:
        if type(hit["matched_peaks"]) is not int or not 0 <= hit["matched_peaks"] <= 100_000:
            raise ValueError("Invalid matched-ion count.")
    for key in ("query_coverage", "reference_coverage", "motif_coverage"):
        if key in hit:
            _number(hit[key], 0, 1.000001)
    scores = hit.get("scores", {})
    if not isinstance(scores, dict) or set(scores) - {"direct", "hybrid", "neutral_loss", "dreams"}:
        raise ValueError("Invalid match scores.")
    for value in scores.values():
        if value is not None:
            _number(value, -1, 1.000001)
    if score_mode is not None:
        _number(scores.get(score_mode, hit.get("score")), -1, 1.000001)
    elif motif:
        _number(hit.get("score"), 0, 1.000001)
    alignment = hit.get("alignment", {})
    if not isinstance(alignment, dict):
        raise ValueError("Invalid peak alignment.")
    for key in ("query_coverage", "reference_coverage"):
        if key in alignment:
            _number(alignment[key], 0, 1.000001)
    for key in ("direct_count", "shifted_count"):
        if key in alignment and (type(alignment[key]) is not int or not 0 <= alignment[key] <= 100_000):
            raise ValueError("Invalid peak count.")
    pairs = hit.get("matches", []) if motif else alignment.get("pairs", [])
    if not isinstance(pairs, list) or len(pairs) > 100_000:
        raise ValueError("Invalid peak alignment.")
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Invalid peak pair.")
        _number(pair.get("query_mz"), 0)
        _number(pair.get("reference_mz"), 0)
        _number(pair.get("error_da", 0))
        if pair.get("type" if motif else "kind") not in (("fragment", "loss") if motif else ("direct", "shifted")):
            raise ValueError("Unknown peak relationship.")
    for key in ("fragment_peaks", "loss_peaks"):
        if key in hit:
            validate_peaks(hit[key], allow_empty=True)


def validate_spectral_analysis(data):
    _validate(data, lambda value: json.dumps(value, allow_nan=False))
    return data


def _validate(data, encode):
    from kmzviewer.spectral_matching import SpectralSettings
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("This is not a supported KmzViewer spectral analysis.")
    try:
        # Reuse the qualified query validator without changing its older format.
        validate_analysis(new_analysis(data["query"], SearchSettings()))
        settings = dict(data["settings"])
        for key in ("required_fragments", "required_losses"):
            settings[key] = tuple(settings.get(key, ()))
        SpectralSettings(**settings).validate()
        library, motifs = data["library"], data["motifs"]
        if not isinstance(library, dict) or not isinstance(motifs, dict):
            raise ValueError("Invalid saved evidence.")
        methods = library.get("methods", {})
        if not isinstance(methods, dict) or set(methods) - {"direct", "hybrid", "neutral_loss"}:
            raise ValueError("Unknown spectral search method.")
        learned = data.get("learned", {})
        if not isinstance(learned, dict):
            raise ValueError("Invalid learned-neighbor evidence.")
        lists = [(h, False, key) for key, h in methods.items()] + [(motifs.get("hits", []), True, None), (learned.get("hits", []), False, "dreams")]
        for hits, is_motif, mode in lists:
            if not isinstance(hits, list) or len(hits) > 1000:
                raise ValueError("Too many saved matches.")
            for hit in hits:
                _validate_hit(hit, motif=is_motif, score_mode=mode)
        for result in (library, learned):
            references = result.get("references", {})
            if not isinstance(references, dict) or len(references) > 3000:
                raise ValueError("Invalid saved reference spectra.")
            for ref in references.values():
                validate_peaks(ref["peaks"], allow_empty=True)
                if "comparison_peaks" in ref:
                    validate_peaks(ref["comparison_peaks"], allow_empty=True)
            if "prepared_query" in result:
                validate_peaks(result["prepared_query"], allow_empty=True)
            if "error" in result:
                _text(result["error"])
        for result in (library, motifs, learned):
            if not isinstance(result.get("provenance", {}), dict) or not isinstance(result.get("preparation", {}), dict):
                raise ValueError("Invalid saved preparation metadata.")
            if "warnings" in result and (not isinstance(result["warnings"], list) or any(not isinstance(x, str) for x in result["warnings"])):
                raise ValueError("Invalid saved warnings.")
        structures = data.get("structures", {})
        if not isinstance(structures, dict) or len(structures) > 1000:
            raise ValueError("Invalid saved structures.")
        import base64
        for key, value in structures.items():
            if not isinstance(key, str) or not isinstance(value, str) or len(value) > 2_800_000:
                raise ValueError("Invalid structure image.")
            if not base64.b64decode(value, validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Invalid structure image.")
        # Reject nonfinite values at every nesting level, and non-JSON objects.
        return encode(data)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("The spectral analysis is incomplete or malformed.") from exc


def save_spectral_analysis(path, data):
    # The snapshot encoding is the final validation check; encode only once.
    payload = _validate(data, lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False,
                                                       separators=(",", ":"))).encode()
    if len(payload) > MAX_BYTES:
        raise ValueError("The analysis exceeds the 64 MiB snapshot limit.")
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".spectral-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_spectral_analysis(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("The analysis exceeds the 64 MiB snapshot limit.")
    try:
        return validate_spectral_analysis(json.loads(payload))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The spectral analysis is not valid JSON.") from exc
