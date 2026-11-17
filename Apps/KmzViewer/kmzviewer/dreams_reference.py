"""Versioned public DreaMS reference cache; contains no user spectra.

The one-time HDF5 conversion is streamed. Search uses memory-mapped arrays and
SQLite metadata, without PyTorch, pickle, or loading the reference into RAM.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

import numpy as np

from kmzviewer.gnps import check_cancel

REFERENCE_REVISION = "d67f74258999ab2dba9f23df6df6b65a7c5ca7c1"
REFERENCE_SHA256 = "cd7a4660d25517870795dcd1ab56f74c391be4f4c87ca49ff07c6ee1602e94d5"
REFERENCE_BYTES = 1540090256
REFERENCE_COUNT = 231104
DIMENSIONS = 1024
REFERENCE_URL = ("https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/"
                 + REFERENCE_REVISION + "/data/auxiliary/MassSpecGym_DreaMS.hdf5")
REFERENCE_PAGE = "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym"
CACHE_NAME = "dreams-reference-v1-" + REFERENCE_SHA256[:12]
FILES = ("embeddings.npy", "offsets.npy", "mz.npy", "intensity.npy", "metadata.sqlite")
_verified = {}


class ReferenceError(ValueError):
    pass


def _hash(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            check_cancel(cancel)
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(cache_dir):
    return Path(cache_dir) / CACHE_NAME


def _manifest(path):
    path = Path(path)
    try:
        raw = (path / "manifest.json").read_bytes()
        if len(raw) > 65536:
            raise ValueError()
        manifest = json.loads(raw)
        if (manifest.get("schema") != "kmz-dreams-reference" or manifest.get("version") != 1
                or manifest.get("dimensions") != DIMENSIONS
                or not isinstance(manifest.get("count"), int) or not 1 <= manifest["count"] <= 500000
                or set(manifest.get("files", {})) != set(FILES)
                or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("source_sha256", ""))):
            raise ValueError()
        for name, details in manifest["files"].items():
            file = path / name
            if (file.is_symlink() or not file.is_file() or details["bytes"] != file.stat().st_size
                    or not re.fullmatch(r"[0-9a-f]{64}", details.get("sha256", ""))):
                raise ValueError()
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ReferenceError("The DreaMS reference cache is missing or incomplete.") from exc


def cache_ready(cache_dir):
    try:
        return _manifest(cache_path(cache_dir))["source_sha256"] == REFERENCE_SHA256
    except ReferenceError:
        return False


def _text(value, maximum=8192):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    if len(value) > maximum or "\x00" in value:
        raise ReferenceError("The public reference contains invalid metadata.")
    return value


def _peak_mask(spectra):
    if spectra.ndim != 3 or spectra.shape[1] != 2 or spectra.shape[2] > 4096:
        raise ReferenceError("The public reference has an unexpected spectrum layout.")
    if not np.isfinite(spectra).all() or np.any(spectra < 0):
        raise ReferenceError("The public reference contains invalid peaks.")
    return (spectra[:, 0] > 0) & (spectra[:, 1] > 0)


def build_reference(source, destination, *, expected_sha256=REFERENCE_SHA256,
                    expected_count=REFERENCE_COUNT, cancel=None, progress=None):
    """Convert a verified public HDF5 file atomically; never remove the source."""
    import h5py

    check_cancel(cancel)
    source, destination = Path(source), Path(destination)
    if progress:
        progress("Verifying the public DreaMS reference download…")
    if _hash(source, cancel) != expected_sha256:
        raise ReferenceError("DreaMS reference checksum did not match. No cache was accepted.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ("." + destination.name + ".building-" + uuid.uuid4().hex)
    staging.mkdir()
    arrays = []
    try:
        with h5py.File(source, "r") as original:
            required = {"DreaMS_embedding", "spectrum", "IDENTIFIER", "INCHIKEY", "smiles", "name", "precursor_mz"}
            if not required.issubset(original):
                raise ReferenceError("The public DreaMS reference format has changed.")
            count = len(original["IDENTIFIER"])
            if count != expected_count or original["DreaMS_embedding"].shape != (count, DIMENSIONS):
                raise ReferenceError("The public DreaMS reference dimensions have changed.")
            for key in required:
                if len(original[key]) != count:
                    raise ReferenceError("The public reference rows are inconsistent.")
            offsets = np.zeros(count + 1, dtype=np.int64)
            for lo in range(0, count, 1024):
                check_cancel(cancel)
                mask = _peak_mask(original["spectrum"][lo:lo + 1024])
                offsets[lo + 1:lo + len(mask) + 1] = mask.sum(axis=1)
            np.cumsum(offsets, out=offsets)
            np.save(staging / "offsets.npy", offsets, allow_pickle=False)
            embeddings = np.lib.format.open_memmap(staging / "embeddings.npy", mode="w+", dtype="float16", shape=(count, DIMENSIONS))
            mz = np.lib.format.open_memmap(staging / "mz.npy", mode="w+", dtype="float64", shape=(int(offsets[-1]),))
            intensity = np.lib.format.open_memmap(staging / "intensity.npy", mode="w+", dtype="float32", shape=mz.shape)
            arrays = [embeddings, mz, intensity]
            with sqlite3.connect(staging / "metadata.sqlite") as database:
                database.execute("CREATE TABLE records (idx INTEGER PRIMARY KEY, accession TEXT NOT NULL, inchikey TEXT NOT NULL, name TEXT NOT NULL, smiles TEXT NOT NULL, precursor REAL NOT NULL, metadata TEXT NOT NULL)")
                metadata_keys = [k for k in ("adduct", "FORMULA", "INSTRUMENT_TYPE", "COLLISION_ENERGY", "FOLD") if k in original]
                for lo in range(0, count, 1024):
                    check_cancel(cancel)
                    hi = min(lo + 1024, count)
                    block = np.asarray(original["DreaMS_embedding"][lo:hi], dtype=np.float32)
                    norms = np.linalg.norm(block, axis=1, keepdims=True)
                    if not np.isfinite(block).all() or np.any(norms <= 0):
                        raise ReferenceError("The public reference contains invalid embeddings.")
                    embeddings[lo:hi] = block / norms
                    spectra = original["spectrum"][lo:hi]
                    mask = _peak_mask(spectra)
                    mz[offsets[lo]:offsets[hi]] = spectra[:, 0][mask]
                    intensity[offsets[lo]:offsets[hi]] = spectra[:, 1][mask]
                    columns = {k: original[k][lo:hi] for k in ["IDENTIFIER", "INCHIKEY", "name", "smiles", "precursor_mz"] + metadata_keys}
                    rows = []
                    for local in range(hi - lo):
                        accession = _text(columns["IDENTIFIER"][local], 64)
                        precursor = float(columns["precursor_mz"][local])
                        if not re.fullmatch(r"MassSpecGymID\d+", accession) or not math.isfinite(precursor) or precursor <= 0:
                            raise ReferenceError("The public reference has invalid identifiers or precursor masses.")
                        metadata = {}
                        for key in metadata_keys:
                            value = columns[key][local]
                            if isinstance(value, (float, np.floating)):
                                metadata[key] = float(value) if math.isfinite(value) else None
                            else:
                                metadata[key] = _text(value)
                        rows.append((lo + local, accession, _text(columns["INCHIKEY"][local], 64),
                                     _text(columns["name"][local]), _text(columns["smiles"][local]), precursor,
                                     json.dumps(metadata, allow_nan=False, separators=(",", ":"))))
                    database.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", rows)
                    if progress:
                        progress(f"Preparing DreaMS public reference · {hi:,}/{count:,} spectra…")
                database.commit()
            for array in arrays:
                array.flush()
                array._mmap.close()
            arrays = []
        manifest = dict(schema="kmz-dreams-reference", version=1, count=count, dimensions=DIMENSIONS,
                        source_sha256=expected_sha256, source_revision=REFERENCE_REVISION,
                        source_url=REFERENCE_URL, source_page=REFERENCE_PAGE,
                        embedding_storage="unit-normalized float16; cosine re-normalized during search",
                        files={name: dict(bytes=(staging / name).stat().st_size, sha256=_hash(staging / name, cancel)) for name in FILES})
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        check_cancel(cancel)
        _manifest(staging)
        backup = destination.parent / ("." + destination.name + ".replaced-" + uuid.uuid4().hex)
        if destination.exists():
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    finally:
        for array in arrays:
            array._mmap.close()
        if staging.exists():
            shutil.rmtree(staging)


def _verified_manifest(path, cancel=None):
    manifest = _manifest(path)
    fingerprint = tuple((name, (path / name).stat().st_size, (path / name).stat().st_mtime_ns,
                         (path / name).stat().st_ino, manifest["files"][name]["sha256"]) for name in FILES)
    if _verified.get(str(path)) != fingerprint:
        for name in FILES:
            if _hash(path / name, cancel) != manifest["files"][name]["sha256"]:
                raise ReferenceError("The DreaMS reference cache failed its integrity check.")
        _verified[str(path)] = fingerprint
    return manifest


def search_reference(embedding, path, *, top_k=50, cancel=None, progress=None):
    """Return unique connectivity groups from a blind full-library cosine search."""
    check_cancel(cancel)
    path = Path(path)
    manifest = _verified_manifest(path, cancel)
    query = np.asarray(embedding, dtype=np.float32)
    if query.shape != (DIMENSIONS,) or not np.isfinite(query).all() or np.linalg.norm(query) <= 0:
        raise ReferenceError("DreaMS returned an invalid spectral embedding.")
    if not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("DreaMS result count must be between 1 and 100.")
    query = query / np.linalg.norm(query)
    arrays = []
    started = time.monotonic()
    try:
        arrays = [np.load(path / name, mmap_mode="r", allow_pickle=False) for name in ("embeddings.npy", "offsets.npy", "mz.npy", "intensity.npy")]
        embeddings, offsets, mz, intensity = arrays
        count = manifest["count"]
        if (embeddings.shape != (count, DIMENSIONS) or embeddings.dtype != np.float16
                or offsets.shape != (count + 1,) or offsets.dtype != np.int64
                or mz.ndim != 1 or mz.dtype != np.float64 or intensity.shape != mz.shape
                or intensity.dtype != np.float32 or offsets[0] != 0 or offsets[-1] != len(mz)
                or np.any(np.diff(offsets) < 0)):
            raise ReferenceError("The DreaMS reference cache has invalid arrays.")
        scores = np.empty(count, dtype=np.float32)
        for lo in range(0, count, 4096):
            check_cancel(cancel)
            block = np.asarray(embeddings[lo:lo + 4096], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(block).all() or np.any(norms <= 0):
                raise ReferenceError("The DreaMS reference cache has invalid embeddings.")
            scores[lo:lo + len(block)] = (block @ query) / norms
            if progress:
                progress(f"Finding learned spectral neighbors · {min(lo + len(block), count):,}/{count:,}…")
        order = np.argsort(-scores, kind="stable")
        hits, references, seen = [], {}, set()
        uri = path.joinpath("metadata.sqlite").resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as database:
            for rank, index in enumerate(order, 1):
                check_cancel(cancel)
                row = database.execute("SELECT accession,inchikey,name,smiles,precursor,metadata FROM records WHERE idx=?", (int(index),)).fetchone()
                if row is None:
                    raise ReferenceError("DreaMS reference metadata is incomplete.")
                accession, key, name, smiles, precursor, metadata_json = row
                group = key or smiles or accession
                if group in seen:
                    continue
                seen.add(group)
                left, right = int(offsets[index]), int(offsets[index + 1])
                peaks = np.column_stack((mz[left:right], intensity[left:right]))
                peaks = peaks[np.argsort(peaks[:, 0], kind="stable")]
                metadata = json.loads(metadata_json)
                title = name.strip() or accession
                hit = dict(accession=accession, name=title, precursor_mz=precursor, smiles=smiles,
                           inchikey=key, scores={"dreams": float(np.clip(scores[index], -1, 1))},
                           rank=len(hits) + 1, spectrum_rank=rank, reference_key=accession,
                           source="MassSpecGym / DreaMS", url=REFERENCE_PAGE)
                reference = dict(accession=accession, name=title, precursor_mz=precursor,
                                 smiles=smiles, inchikey=key, peaks=peaks.tolist(), metadata=metadata,
                                 source="MassSpecGym / DreaMS", url=REFERENCE_PAGE,
                                 adduct=metadata.get("adduct", ""), structure_png="", warnings=[])
                hits.append(hit)
                references[accession] = reference
                if len(hits) >= top_k:
                    break
        return dict(hits=hits, references=references, provenance=dict(reference_sha256=manifest["source_sha256"],
                    reference_revision=manifest["source_revision"], reference_spectra=count,
                    reference_url=REFERENCE_URL, reference_page=REFERENCE_PAGE,
                    ranking="cosine of DreaMS embeddings; unique reference connectivity keys",
                    embedding_storage=manifest["embedding_storage"], local_search_seconds=time.monotonic() - started))
    except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise ReferenceError("The DreaMS reference cache could not be read safely.") from exc
    finally:
        for array in arrays:
            array._mmap.close()
