"""Local GNPS query preparation and portable, explicit analysis snapshots.

Profile queries use native local-maximum samples, not inferred centroid masses.
All transformations are confined to the query copy; acquisitions remain untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from kmz2viewer.index import index_ms2
from kmzcore.cancel import Cancelled
from kmzcore.version import __version__

SCHEMA = "kmzviewer-gnps-v1"
MAX_SNAPSHOT = 32 * 1024 * 1024
MAX_PEAKS = 100_000


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def check_cancel(cancel):
    if cancel is not None and cancel():
        raise Cancelled()


def finite(value):
    return value is not None and math.isfinite(float(value))


@dataclass(frozen=True)
class SearchSettings:
    precursor_da: float = .05
    fragment_da: float = .05
    min_cosine: float = .7
    min_peaks: int = 6
    analog_below: float = 100.
    analog_above: float = 100.

    def validate(self):
        for name in ("precursor_da", "fragment_da"):
            if not finite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError("Mass tolerances must be greater than zero and at most 1 Da.")
        if not finite(self.min_cosine) or not 0 <= self.min_cosine <= 1:
            raise ValueError("Cosine threshold must be between zero and one.")
        if type(self.min_peaks) is not int or not 1 <= self.min_peaks <= 1000:
            raise ValueError("Minimum matched peaks must be an integer from 1 to 1000.")
        for value in (self.analog_below, self.analog_above):
            if not finite(value) or not 0 <= value <= 300:
                raise ValueError("Analog mass ranges must be between 0 and 300 Da.")
        return self


def validate_peaks(peaks, *, allow_empty=False):
    if not isinstance(peaks, (list, tuple)) or len(peaks) > MAX_PEAKS:
        raise ValueError("Invalid or oversized peak list.")
    if not peaks:
        if allow_empty:
            return []
        raise ValueError("The spectrum has no positive peaks to search.")
    data = np.asarray(peaks, dtype=np.float64)
    if (data.ndim != 2 or data.shape[1] != 2 or not np.all(np.isfinite(data))
            or np.any(data[:, 0] <= 0) or np.any(data[:, 1] <= 0)
            or np.any(np.diff(data[:, 0]) < 0)):
        raise ValueError("Peak lists must contain sorted, positive, finite m/z–intensity pairs.")
    return data.tolist()


def prepare_query(scan, *, target_mz=None, target_rt=None):
    mz = np.asarray(scan.mz, dtype=np.float64)
    y = np.asarray(scan.intensity, dtype=np.float64)
    if mz.ndim != 1 or mz.shape != y.shape or not np.all(np.isfinite(mz)) or not np.all(np.isfinite(y)):
        raise ValueError("MS2 arrays must be equal-length finite vectors.")
    if np.any(mz <= 0):
        raise ValueError("MS2 m/z values must be positive.")
    if not finite(scan.precursor_mz) or scan.precursor_mz <= 0:
        raise ValueError("This MS2 scan has no valid precursor mass.")
    representation = scan.representation
    if representation == "centroid":
        indices = np.flatnonzero(y > 0)
        indices = indices[np.argsort(mz[indices], kind="stable")]
        preparation = "Native centroid peaks"
    elif representation == "profile":
        if np.any(np.diff(mz) < 0):
            raise ValueError("Profile MS2 m/z values must be sorted to extract local maxima.")
        # One native sample per local maximum; flat maxima use their middle
        # sample. Endpoints can be maxima. No interpolation, merging or smoothing.
        starts = np.r_[0, np.flatnonzero(np.diff(y) != 0) + 1] if y.size else np.array([], dtype=int)
        ends = np.r_[starts[1:] - 1, y.size - 1] if starts.size else starts
        indices = []
        for left, right in zip(starts, ends):
            value = y[left]
            if value > 0 and (left == 0 or value > y[left - 1]) and (right == y.size - 1 or value > y[right + 1]):
                indices.append((int(left) + int(right)) // 2)
        indices = np.asarray(indices, dtype=int)
        preparation = "Profile local maxima (experimental; native apex samples, not centroid masses)"
    else:
        raise ValueError("The mzML does not unambiguously identify this MS2 spectrum as profile or centroid.")
    peaks = validate_peaks(np.column_stack((mz[indices], y[indices])).tolist())
    return {
        "precursor_mz": float(scan.precursor_mz), "charge": scan.charge,
        "polarity": scan.polarity, "peaks": peaks, "preparation": preparation,
        "representation": representation, "precursor_source": scan.precursor_source,
        "source_name": scan.file_name, "scan_id": scan.scan_id, "spectrum_index": scan.spectrum_index,
        "rt_min": float(scan.rt_min) if finite(scan.rt_min) else None,
        "collision_energy": float(scan.collision_energy) if finite(scan.collision_energy) else None,
        "target_mz": target_mz, "target_rt": target_rt if finite(target_rt) else None,
    }


def matching_scans(path, mass, rt, polarity, *, ppm=20., cancel=None, progress=None):
    """Decode only precursor candidates; rank by RT, never pool scans/energies."""
    if not finite(mass) or mass <= 0:
        raise ValueError("A positive MS1 target mass is required.")
    path = Path(path)
    before = path.stat()
    width = mass * ppm * 1e-6

    def accept(meta):
        precursor = meta["precursor_mz"]
        return (finite(precursor) and mass - width <= precursor <= mass + width
                and (not polarity or not meta["polarity"] or polarity == meta["polarity"]))

    scans = index_ms2(path, accept_header=accept, is_cancelled=cancel, on_progress=progress)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("The source file changed while reading MS2 scans. Reopen it and try again.")
    check_cancel(cancel)
    scans.sort(key=lambda scan: (abs(scan.rt_min - rt) if finite(rt) and finite(scan.rt_min) else math.inf,
                                 abs(scan.precursor_mz - mass), scan.spectrum_index))
    return scans


def new_analysis(query, settings):
    settings.validate()
    return dict(schema=SCHEMA, version=__version__, created=timestamp(), query=query,
                settings=asdict(settings), searches={}, references={})


def validate_analysis(data):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("This is not a supported KmzViewer GNPS analysis.")
    try:
        query = data["query"]
        for key in ("preparation", "source_name", "scan_id"):
            if not isinstance(query[key], str) or len(query[key]) > 16384:
                raise ValueError("Invalid query metadata.")
        validate_peaks(query["peaks"])
        if not finite(query["precursor_mz"]) or query["precursor_mz"] <= 0:
            raise ValueError("Invalid query precursor mass.")
        if query.get("polarity", 0) not in (-1, 0, 1):
            raise ValueError("Invalid query polarity.")
        charge = query.get("charge")
        if charge is not None and (type(charge) is not int or abs(charge) > 100):
            raise ValueError("Invalid query charge.")
        if set(data["settings"]) != set(asdict(SearchSettings())):
            raise ValueError("Incomplete search settings.")
        SearchSettings(**data["settings"]).validate()
        searches, references = data["searches"], data["references"]
        if not isinstance(searches, dict) or not isinstance(references, dict):
            raise ValueError("Invalid saved results.")
        for mode, result in searches.items():
            if mode not in {"standard", "analog", "datasets"} or not isinstance(result, dict):
                raise ValueError("Unknown GNPS search type.")
            if not isinstance(result.get("hits"), list) or len(result["hits"]) > 10000:
                raise ValueError("Invalid saved matches.")
            for hit in result["hits"]:
                if (not isinstance(hit, dict) or not isinstance(hit.get("usi"), str)
                        or not hit["usi"].startswith("mzspec:") or len(hit["usi"]) > 4096):
                    raise ValueError("Invalid saved match.")
                if any(not isinstance(hit.get(key), str) or len(hit[key]) > 16384 for key in ("name", "dataset", "adduct")):
                    raise ValueError("Invalid saved match metadata.")
                if not finite(hit.get("score")) or not 0 <= hit["score"] <= 1.000001:
                    raise ValueError("Invalid saved score.")
                if type(hit.get("matched_peaks")) is not int or hit["matched_peaks"] < 0:
                    raise ValueError("Invalid saved matched-peak count.")
                if not finite(hit.get("delta_mass")):
                    raise ValueError("Invalid saved mass difference.")
        if len(references) > 10000:
            raise ValueError("Too many saved reference spectra.")
        for reference in references.values():
            validate_peaks(reference["peaks"], allow_empty=True)
            encoded = reference.get("structure_png", "")
            if not isinstance(encoded, str) or len(encoded) > 2_800_000:
                raise ValueError("Oversized saved structure image.")
            if encoded and not base64.b64decode(encoded, validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Invalid saved structure image.")
            metadata = reference.get("metadata", {})
            if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
                raise ValueError("Invalid reference metadata.")
            warnings = reference.get("warnings", [])
            if not isinstance(warnings, list) or any(not isinstance(v, str) for v in warnings):
                raise ValueError("Invalid reference warnings.")
        # Strict JSON rejects NaN/Infinity anywhere, including service metadata.
        json.dumps(data, allow_nan=False)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("The saved GNPS analysis is incomplete or malformed.") from exc
    return data


def save_analysis(path, data):
    """Atomic replacement only at the explicitly selected destination."""
    validate_analysis(data)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_SNAPSHOT:
        raise ValueError("The analysis exceeds the 32 MiB snapshot limit.")
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".gnps-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_analysis(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_SNAPSHOT + 1)
    if len(payload) > MAX_SNAPSHOT:
        raise ValueError("The analysis exceeds the 32 MiB snapshot limit.")
    try:
        return validate_analysis(json.loads(payload))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The GNPS analysis is not valid JSON.") from exc
