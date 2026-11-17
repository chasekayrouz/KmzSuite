"""Explicit, bounded downloads of public reference spectra for local searches.

Only fixed public-library URLs are requested. Query spectra never enter this
module. Durable packs use numeric arrays and JSON, never executable pickles.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

from kmzcore.cancel import Cancelled

PACK_SCHEMA = "kmzviewer-public-references-v1"
MAX_DOWNLOAD = 512 * 1024 * 1024
MAX_PACK_BYTES = 768 * 1024 * 1024
LIBRARIES = (
    ("GNPS-LIBRARY", "Community annotations", 143_000_000),
    ("MASSBANK", "Imported experimental library", 80_000_000),
    ("MASSBANKEU", "Imported experimental library", 1_600_000),
    ("GNPS-SCIEX-LIBRARY", "Experimental reference collection", 300_000),
    ("TUEBINGEN-NATURAL-PRODUCT-COLLECTION", "Experimental reference collection", 2_300_000),
    ("GNPS-PRESTWICKPHYTOCHEM", "Commercial compound collection", 3_100_000),
)
EXPANDED_LIBRARY = ("BMDMS-NP", "Experimental standards (GNPS import)", 197_000_000)
HOST = "https://external.gnps2.org/gnpslibrary/"


def default_cache_dir():
    root = os.environ.get("KMZ_LOCAL_STATE_ROOT")
    if not root:
        root = str(Path.home() / "Library/Caches") if sys.platform == "darwin" else os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(root) / "org.kmzsuite.viewer" / "spectral-references"


def _check(cancel):
    if cancel is not None and cancel():
        raise Cancelled()


def _notify(progress, fraction, message):
    if progress is not None:
        progress(float(fraction), message)


def _destination(cache_dir, expanded):
    return Path(cache_dir) / ("expanded-v1" if expanded else "base-v1")


def pack_status(cache_dir=None, *, expanded=False):
    """Fast local status; full integrity is checked when loading the pack."""
    destination = _destination(cache_dir or default_cache_dir(), expanded)
    libraries = LIBRARIES + ((EXPANDED_LIBRARY,) if expanded else ())
    result = dict(available=False, path=str(destination), estimated_download_bytes=sum(x[2] for x in libraries),
                  libraries=[x[0] for x in libraries], expanded=expanded)
    try:
        path = destination / "manifest.json"
        if path.stat().st_size > 256_000:
            return result
        manifest = json.loads(path.read_text())
        if manifest.get("schema") == PACK_SCHEMA and all((destination / name).is_file() for name in ("peaks.npz", "metadata.json.gz")):
            result.update(available=True, spectra=manifest.get("spectra", 0), created=manifest.get("created", ""))
    except (OSError, ValueError, TypeError):
        pass
    return result


@dataclass
class ReferencePack:
    metadata: list[dict]
    peaks: np.ndarray
    offsets: np.ndarray
    provenance: dict
    path: Path | None = None

    def spectrum(self, index):
        return self.peaks[self.offsets[index]:self.offsets[index + 1]]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_TEXT_FIELDS = ("name", "accession", "usi", "library", "provenance", "smiles", "inchi", "structure_key", "adduct")
_STR = (str,) * len(_TEXT_FIELDS)


def load_reference_pack(path, *, cancel=None):
    path = Path(path)
    _check(cancel)
    if (path / "manifest.json").stat().st_size > 256_000:
        raise ValueError("Oversized reference-pack manifest.")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != PACK_SCHEMA:
        raise ValueError("Unsupported reference-pack format.")
    for name in ("peaks.npz", "metadata.json.gz"):
        _check(cancel)
        if (path / name).stat().st_size > MAX_PACK_BYTES or _sha256(path / name) != manifest.get("sha256", {}).get(name):
            raise ValueError("Reference-pack integrity check failed. Download a fresh reference pack.")
    with gzip.open(path / "metadata.json.gz", "rb") as stream:
        payload = stream.read(MAX_PACK_BYTES + 1)
    if len(payload) > MAX_PACK_BYTES:
        raise ValueError("Oversized reference metadata.")
    metadata = json.loads(payload)
    with zipfile.ZipFile(path / "peaks.npz") as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_PACK_BYTES:
            raise ValueError("Oversized reference peak arrays.")
    with np.load(path / "peaks.npz", allow_pickle=False) as arrays:
        peaks, offsets = arrays["peaks"], arrays["offsets"]
    if (not isinstance(metadata, list) or len(metadata) > 1_000_000 or peaks.ndim != 2 or peaks.shape[1] != 2
            or offsets.dtype.kind not in "iu" or offsets.shape != (len(metadata) + 1,)
            or offsets[0] != 0 or offsets[-1] != len(peaks) or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or not np.all(np.isfinite(peaks)) or np.any(peaks <= 0)):
        raise ValueError("Malformed reference spectrum arrays.")
    for row in metadata:
        if not isinstance(row, dict):
            raise ValueError("Malformed reference metadata.")
        precursor, texts = row.get("precursor_mz"), list(map(row.get, _TEXT_FIELDS))
        if ((not (math.isfinite(precursor) and precursor > 0) if type(precursor) is float else
             not isinstance(precursor, (float, int)) or not np.isfinite(precursor) or precursor <= 0)
                or row.get("polarity") not in (-1, 1)
                or (row.get("charge") is not None and (type(row["charge"]) is not int or not 1 <= abs(row["charge"]) <= 20))
                or not all(map(isinstance, texts, _STR)) or max(map(len, texts)) > 4096):
            raise ValueError("Malformed reference metadata.")
    _check(cancel)
    return ReferencePack(metadata, peaks.astype(np.float32, copy=False), offsets.astype(np.int64, copy=False), manifest, path)


def _download(url, destination, *, cancel=None, progress=None):
    from PySide6.QtCore import QEventLoop, QTimer, QUrl
    from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
    import shiboken6
    # Fixed HTTPS hosts and platform TLS also work in independently installed apps.
    allowed = {HOST + row[0] + ".mgf" for row in LIBRARIES + (EXPANDED_LIBRARY,)}
    if url not in allowed:
        raise ValueError("Unrecognized public reference endpoint.")
    _check(cancel)
    manager = QNetworkAccessManager()
    request = QNetworkRequest(QUrl(url))
    request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute, QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
    request.setRawHeader(b"User-Agent", b"KmzViewer-References/1")
    reply = manager.get(request)
    loop, timer = QEventLoop(), QTimer()
    timer.setInterval(40)
    digest = hashlib.sha256()
    count = 0
    expected = 0
    reasons = []
    started = last_data = time.monotonic()
    try:
        with Path(destination).open("wb") as output:
            def read():
                nonlocal count, expected, last_data
                try:
                    expected = int(bytes(reply.rawHeader("Content-Length")) or b"0")
                    while reply.bytesAvailable():
                        block = bytes(reply.read(256 * 1024))
                        count += len(block)
                        if count > MAX_DOWNLOAD or expected > MAX_DOWNLOAD:
                            reasons.append("The public reference download exceeds the supported size.")
                            reply.abort()
                            return
                        output.write(block)
                        digest.update(block)
                        last_data = time.monotonic()
                    if progress is not None:
                        progress(count, expected)
                except (OSError, ValueError, TypeError) as exc:
                    reasons.append(str(exc))
                    reply.abort()

            def tick():
                if cancel is not None and cancel():
                    reasons.append("cancel")
                    reply.abort()
                elif time.monotonic() - last_data > 45 or time.monotonic() - started > 600:
                    reasons.append("Reference download timed out. Try again later.")
                    reply.abort()

            reply.readyRead.connect(read)
            reply.finished.connect(loop.quit)
            timer.timeout.connect(tick)
            timer.start()
            if not reply.isFinished():
                loop.exec()
            timer.stop()
            if reply.isOpen():
                read()
        _check(cancel)
        if reasons:
            if reasons[0] == "cancel":
                raise Cancelled()
            raise ValueError(reasons[0])
        if reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) != 200:
            raise ValueError("The public reference download did not return HTTP 200.")
        if reply.error() != QNetworkReply.NetworkError.NoError:
            raise ValueError("Could not download public references securely. Check the connection and try again.")
        if expected and count != expected:
            raise ValueError("The public reference download was incomplete.")
        return dict(url=url, bytes=count, sha256=digest.hexdigest(), last_modified=bytes(reply.rawHeader("Last-Modified")).decode("ascii", errors="replace"))
    finally:
        # Delete these worker-thread Qt objects here, even if a traceback keeps this frame
        # alive: the GUI-thread garbage collector must never be the one to destroy them.
        timer.stop()
        for item in (timer, loop, manager):  # The reply is the manager's child.
            shiboken6.delete(item)


def _text(value, maximum=4096):
    return str(value or "").strip()[:maximum]


def _structure_key(metadata):
    for key in ("inchikey", "inchi", "smiles"):
        value = _text(metadata.get(key))
        if value.upper() not in {"", "N/A", "NA", "NONE", "NULL"}:
            return key + ":" + value.removeprefix("InChI=")
    return "accession:" + metadata["accession"]


def _read_mgf(path, library, provenance, *, cancel=None):
    from pyteomics import mgf
    metadata, arrays = [], []
    skipped = dict(unknown_polarity=0, invalid=0, non_ms2=0)
    with mgf.MGF(str(path), read_charges=False) as reader:
        for ordinal, record in enumerate(reader):
            if ordinal % 256 == 0:
                _check(cancel)
            raw = record["params"]
            mode = str(raw.get("ionmode", "")).strip().lower()
            polarity = 1 if mode in {"positive", "pos", "+"} else -1 if mode in {"negative", "neg", "-"} else 0
            if not polarity:
                skipped["unknown_polarity"] += 1
                continue
            if str(raw.get("mslevel", "")) != "2":
                skipped["non_ms2"] += 1
                continue
            try:
                precursor = float(raw.get("pepmass", [0])[0])
                peaks = np.column_stack((record["m/z array"], record["intensity array"])).astype(np.float32)
                if not np.isfinite(precursor) or precursor <= 0 or not np.all(np.isfinite(peaks)):
                    raise ValueError()
                peaks = peaks[(peaks[:, 0] > 0) & (peaks[:, 1] > 0)]
                if not len(peaks):
                    raise ValueError()
                peaks = peaks[np.argsort(peaks[:, 0], kind="stable")]
            except (ValueError, TypeError, IndexError):
                skipped["invalid"] += 1
                continue
            accession = _text(raw.get("spectrumid")) or f"{library}:{ordinal}"
            charge = None
            charge_text = str(raw.get("charge", "")).strip()
            charge_values = re.findall(r"[-+]?\d+", charge_text)
            if len(charge_values) == 1:
                parsed = int(charge_values[0])
                if charge_text.endswith("-"):
                    parsed = -abs(parsed)
                if parsed and abs(parsed) <= 20:
                    # Preserve the reported magnitude; polarity remains separate.
                    charge = parsed
            row = dict(name=_text(raw.get("name")), accession=accession,
                       usi=_text(raw.get("usi")), precursor_mz=precursor, polarity=polarity, charge=charge,
                       library=library, provenance=provenance, smiles=_text(raw.get("smiles")),
                       inchi=_text(raw.get("inchi")), inchikey=_text(raw.get("inchikey")),
                       adduct=_text(raw.get("adduct")), quality=_text(raw.get("libraryquality")),
                       instrument=_text(raw.get("source_instrument")), collision_energy=_text(raw.get("collision_energy")))
            row["structure_key"] = _structure_key(row)
            metadata.append(row)
            arrays.append(peaks)
    return metadata, arrays, skipped


def ensure_reference_pack(cache_dir=None, *, allow_download=False, expanded=False, cancel=None, progress=None, downloader=None):
    """Return the local pack, downloading public data only after explicit consent."""
    from datetime import datetime, timezone
    cache_dir = Path(cache_dir or default_cache_dir())
    destination = _destination(cache_dir, expanded)
    _check(cancel)
    if destination.exists():
        _notify(progress, 0., "Checking local reference pack")
        return load_reference_pack(destination, cancel=cancel)
    if not allow_download:
        raise FileNotFoundError("The public reference pack has not been downloaded. Choose Download references & search.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".reference-download-", dir=cache_dir))
    libraries = LIBRARIES + ((EXPANDED_LIBRARY,) if expanded else ())
    try:
        metadata, arrays, receipts = [], [], []
        base_path = _destination(cache_dir, False)
        if expanded and base_path.exists():
            # Expansion reuses the qualified native public peaks and their
            # receipts; only the additional collection needs downloading.
            base = load_reference_pack(base_path, cancel=cancel)
            metadata = list(base.metadata)
            arrays = [base.spectrum(i) for i in range(len(base.metadata))]
            receipts = list(base.provenance["libraries"])
            libraries = (EXPANDED_LIBRARY,)
        for index, (library, provenance, _) in enumerate(libraries):
            _check(cancel)
            raw_path = staging / (library + ".mgf")
            def report(done, total):
                fraction = done / total if total else 0.
                _notify(progress, .8 * (index + fraction) / len(libraries), f"Downloading {library}: {done / 1048576:.1f} MiB")
            receipt = (downloader or _download)(HOST + library + ".mgf", raw_path, cancel=cancel, progress=report)
            rows, peaks, skipped = _read_mgf(raw_path, library, provenance, cancel=cancel)
            metadata.extend(rows)
            arrays.extend(peaks)
            receipt.update(library=library, provenance=provenance, spectra=len(rows), skipped=skipped)
            receipts.append(receipt)
            raw_path.unlink()
        _check(cancel)
        if not metadata:
            raise ValueError("No usable spectra were present in the public reference collections.")
        _notify(progress, .85, "Preparing the local reference pack")
        offsets = np.r_[0, np.cumsum([len(a) for a in arrays], dtype=np.int64)]
        peaks = np.concatenate(arrays)
        if peaks.nbytes > MAX_PACK_BYTES:
            raise ValueError("The reference peak arrays exceed the supported pack size.")
        np.savez_compressed(staging / "peaks.npz", peaks=peaks, offsets=offsets)
        with gzip.open(staging / "metadata.json.gz", "wt", encoding="utf8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        _check(cancel)
        manifest = dict(schema=PACK_SCHEMA, created=datetime.now(timezone.utc).isoformat(), expanded=expanded,
                        spectra=len(metadata), libraries=receipts,
                        sha256={name: _sha256(staging / name) for name in ("peaks.npz", "metadata.json.gz")})
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        _check(cancel)
        # A concurrent window may finish first. Reuse its complete pack.
        if destination.exists():
            return load_reference_pack(destination, cancel=cancel)
        try:
            os.rename(staging, destination)
        except OSError:
            if destination.exists():
                return load_reference_pack(destination, cancel=cancel)
            raise
        _notify(progress, 1., "Public reference pack ready")
        return ReferencePack(metadata, peaks, offsets, manifest, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
