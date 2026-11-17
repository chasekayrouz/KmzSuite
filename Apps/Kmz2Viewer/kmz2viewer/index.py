"""Index MS2 scans from mzML. MS1 peak arrays are skipped.

Common zlib and uncompressed float arrays are decoded in the same pass.
Anything else is filled afterwards by pyteomics for that scan only.
"""

from __future__ import annotations

import base64
import re
import zlib
from pathlib import Path
from typing import Callable

import numpy as np

from kmz2viewer.model import Ms2Scan
from kmzcore.cancel import Cancelled


def _check_cancel(callback):
    if callback is not None and callback():
        raise Cancelled()
from kmz2viewer.peaks import base_peak, local_maxima_mask, total_intensity
from kmzcore.mzml import looks_like_mzml, require_current_input

_OPEN = b"<spectrum"
_CLOSE = b"</spectrum>"
_CLOSE_LEN = len(_CLOSE)
_CV = re.compile(r"<cvParam\b([^>]*)/?>")
_ATTR = re.compile(r'([A-Za-z_:][\w:.-]*)="([^"]*)"')
_ARRAY = re.compile(r"<binaryDataArray\b.*?</binaryDataArray>", re.S)
_MINUTE_UNITS = {"uo:0000031", "minute", "minutes", "min"}
_SECOND_UNITS = {"uo:0000010", "second", "seconds", "sec", "s"}


def index_ms2(
    path: str | Path,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    accept_header: Callable[[dict], bool] | None = None,
) -> list[Ms2Scan]:
    path = Path(path)
    if not looks_like_mzml(path):
        raise ValueError(f"{path.name} is not an mzML file.")
    require_current_input(path, is_cancelled=is_cancelled)
    size = path.stat().st_size
    scanner = _Ms2Scanner(str(path), path.name, is_cancelled, accept_header)
    read = 0
    with path.open("rb") as handle:
        while True:
            if is_cancelled is not None and is_cancelled():
                from kmzcore.cancel import Cancelled

                raise Cancelled()
            chunk = handle.read(chunk_size)
            read += len(chunk)
            scanner.feed(chunk, final=not chunk)
            if on_progress is not None:
                on_progress(min(read, size), size)
            if not chunk:
                break
    if scanner.pending:
        _fill_with_pyteomics(path, scanner.pending, is_cancelled)
    _check_cancel(is_cancelled)
    return scanner.scans


class _Ms2Scanner:
    def __init__(self, path: str, file_name: str, is_cancelled=None, accept_header=None) -> None:
        self.is_cancelled = is_cancelled
        self.accept_header = accept_header
        self.path = path
        self.file_name = file_name
        self.buf = bytearray()
        self.skipping = False
        self.holding: bytearray | None = None
        self.scans: list[Ms2Scan] = []
        self.pending: list[Ms2Scan] = []

    def feed(self, data: bytes, final: bool = False) -> None:
        if data:
            self.buf.extend(data)
        while True:
            _check_cancel(self.is_cancelled)
            if self.holding is not None:
                if not self._finish_held(final):
                    return
                continue
            if self.skipping:
                if not self._skip(final):
                    return
                continue
            start = _find_spectrum_start(self.buf)
            if start < 0:
                if len(self.buf) > 64:
                    del self.buf[:-64]
                return
            if start > 0:
                del self.buf[:start]
            binary_at = self.buf.find(b"<binary")
            # No </spectrum> can straddle '<binary', so only arrays need memchr hops.
            close_at = self.buf.find(_CLOSE, 0, binary_at) if binary_at >= 0 else -1
            if close_at < 0:
                close_at = _find(self.buf, _CLOSE, max(binary_at, 0))
            if binary_at < 0 and close_at < 0:
                if final:
                    raise ValueError("The mzML file ends inside a spectrum header.")
                return
            header_end = binary_at if binary_at >= 0 else close_at
            header = bytes(self.buf[:header_end])
            level = _ms_level(header)
            if level != 2:
                if close_at >= 0:
                    del self.buf[: close_at + _CLOSE_LEN]
                    continue
                tail = bytes(self.buf[-(_CLOSE_LEN - 1) :])
                self.buf.clear()
                self.buf.extend(tail)
                self.skipping = True
                if final:
                    raise ValueError("The mzML file ends inside a spectrum.")
                return
            if close_at >= 0:
                block = bytes(self.buf[: close_at + _CLOSE_LEN])
                del self.buf[: close_at + _CLOSE_LEN]
                self._store(header, block)
                continue
            self.holding = bytearray(self.buf)
            self.buf.clear()
            if final:
                raise ValueError("The mzML file ends inside an MS2 spectrum.")
            return

    def _finish_held(self, final: bool) -> bool:
        assert self.holding is not None
        self.holding.extend(self.buf)
        self.buf.clear()
        close_at = _find(self.holding, _CLOSE)
        if close_at < 0:
            if final:
                raise ValueError("The mzML file ends inside an MS2 spectrum.")
            return False
        block = bytes(self.holding[: close_at + _CLOSE_LEN])
        rest = self.holding[close_at + _CLOSE_LEN :]
        self.holding = None
        self.buf.extend(rest)
        header_end = block.find(b"<binary")
        header = block[: header_end if header_end >= 0 else len(block)]
        self._store(header, block)
        return True

    def _skip(self, final: bool) -> bool:
        close_at = _find(self.buf, _CLOSE)
        if close_at < 0:
            if len(self.buf) > _CLOSE_LEN:
                del self.buf[:-_CLOSE_LEN]
            if final:
                raise ValueError("The mzML file ends inside a spectrum.")
            return False
        del self.buf[: close_at + _CLOSE_LEN]
        self.skipping = False
        return True

    def _store(self, header: bytes, block: bytes) -> None:
        _check_cancel(self.is_cancelled)
        meta = _parse_ms2_header(header)
        if self.accept_header is not None and not self.accept_header(meta):
            return
        decoded = _decode_arrays(block.decode("utf-8", "replace"))
        _check_cancel(self.is_cancelled)
        scan = _make_scan(self.path, self.file_name, meta, decoded)
        if decoded is None:
            self.pending.append(scan)
        self.scans.append(scan)


def _make_scan(path: str, file_name: str, meta: dict, arrays: tuple[np.ndarray, np.ndarray] | None) -> Ms2Scan:
    if arrays is None:
        mz = np.array([], dtype=np.float64)
        intensity = np.array([], dtype=np.float64)
    else:
        mz, intensity = arrays
    peak_mz, peak_intensity = base_peak(mz, intensity)
    stored_tic = meta.get("tic")
    tic = float(stored_tic) if stored_tic is not None and np.isfinite(stored_tic) else total_intensity(intensity)
    if meta["centroid"]:
        n_peaks = int(np.sum(np.isfinite(intensity) & (intensity > 0)))
    else:
        n_peaks = int(np.sum(local_maxima_mask(intensity)))
    return Ms2Scan(
        path=path,
        file_name=file_name,
        spectrum_index=int(meta["spectrum_index"]),
        scan_id=str(meta["scan_id"]),
        rt_min=float(meta["rt_min"]) if meta["rt_min"] is not None else float("nan"),
        precursor_mz=meta["precursor_mz"],
        charge=meta["charge"],
        precursor_intensity=meta["precursor_intensity"],
        collision_energy=meta["collision_energy"],
        centroid=bool(meta["centroid"]),
        mz=mz,
        intensity=intensity,
        base_peak_mz=peak_mz,
        base_peak_intensity=peak_intensity,
        tic=tic,
        n_peaks=n_peaks,
        polarity=meta["polarity"],
        representation=meta["representation"],
        precursor_source=meta["precursor_source"],
    )


def _fill_with_pyteomics(path: Path, pending: list[Ms2Scan], is_cancelled=None) -> None:
    from pyteomics import mzml
    from kmzcore import psi_ms

    _check_cancel(is_cancelled)
    with mzml.MzML(str(path), cv=psi_ms.vocabulary()) as reader:
        for scan in pending:
            _check_cancel(is_cancelled)
            spectrum = reader.get_by_index(scan.spectrum_index)
            _check_cancel(is_cancelled)
            scan.mz = np.asarray(spectrum.get("m/z array", []), dtype=np.float64)
            scan.intensity = np.asarray(spectrum.get("intensity array", []), dtype=np.float64)
            peak_mz, peak_intensity = base_peak(scan.mz, scan.intensity)
            scan.base_peak_mz = peak_mz
            scan.base_peak_intensity = peak_intensity
            if not np.isfinite(scan.tic):
                scan.tic = total_intensity(scan.intensity)
            if scan.centroid:
                scan.n_peaks = int(np.sum(np.isfinite(scan.intensity) & (scan.intensity > 0)))
            else:
                scan.n_peaks = int(np.sum(local_maxima_mask(scan.intensity)))


def _ms_level(header: bytes) -> int | None:
    text = header.decode("utf-8", "replace")
    for match in _CV.finditer(text):
        attrs = dict(_ATTR.findall(match.group(1)))
        if attrs.get("accession") == "MS:1000511" and attrs.get("value"):
            return int(float(attrs["value"]))
    return None


def _parse_ms2_header(header: bytes) -> dict:
    text = header.decode("utf-8", "replace")
    opener = text.split(">", 1)[0]
    open_attrs = dict(_ATTR.findall(opener))
    rt = None
    rt_unit = ""
    tic = None
    centroid = False
    saw_centroid = False
    saw_profile = False
    precursor = None
    isolation_target = None
    polarity = 0
    charge = None
    precursor_intensity = None
    collision_energy = None
    for match in _CV.finditer(text):
        attrs = dict(_ATTR.findall(match.group(1)))
        accession = attrs.get("accession", "")
        value = attrs.get("value")
        if accession == "MS:1000016" and rt is None and value:
            rt = float(value)
            rt_unit = attrs.get("unitAccession") or attrs.get("unitName") or ""
        elif accession == "MS:1000285" and value:
            tic = float(value)
        elif accession == "MS:1000127":
            saw_centroid = True
        elif accession == "MS:1000128":
            saw_profile = True
        elif accession == "MS:1000130":
            polarity = 1
        elif accession == "MS:1000129":
            polarity = -1
        elif accession == "MS:1000744" and precursor is None and value:
            precursor = float(value)
        elif accession == "MS:1000827" and isolation_target is None and value:
            isolation_target = float(value)
        elif accession == "MS:1000041" and charge is None and value:
            charge = int(float(value))
        elif accession == "MS:1000042" and precursor_intensity is None and value:
            precursor_intensity = float(value)
        elif accession == "MS:1000045" and collision_energy is None and value:
            collision_energy = float(value)
    if saw_centroid and not saw_profile:
        centroid = True
    index_text = open_attrs.get("index")
    spectrum_index = int(index_text) if index_text not in (None, "") else -1
    rt_min = None if rt is None else _to_minutes(rt, rt_unit)
    scan_id = open_attrs.get("id") or (f"index {spectrum_index}" if spectrum_index >= 0 else "")
    return {
        "spectrum_index": spectrum_index,
        "scan_id": scan_id,
        "rt_min": rt_min,
        "tic": tic,
        "centroid": centroid,
        "precursor_mz": precursor if precursor is not None else isolation_target,
        "precursor_source": "selected ion" if precursor is not None else "isolation target" if isolation_target is not None else "unknown",
        "polarity": polarity,
        "representation": "centroid" if centroid else "profile" if saw_profile and not saw_centroid else "unknown",
        "charge": charge,
        "precursor_intensity": precursor_intensity,
        "collision_energy": collision_energy,
    }


def _to_minutes(value: float, unit: str) -> float:
    token = (unit or "").strip().lower()
    if token in _SECOND_UNITS:
        return value / 60.0
    if token in _MINUTE_UNITS or token == "":
        return value
    raise ValueError(f"Unsupported scan start time unit: {unit}")


def _decode_arrays(block: str) -> tuple[np.ndarray, np.ndarray] | None:
    found: dict[str, np.ndarray] = {}
    for body in _array_bodies(block):
        if "MS:1000514" in body:
            kind = "mz"
        elif "MS:1000515" in body:
            kind = "intensity"
        else:
            continue
        if "MS:1002746" in body or "MS:1002747" in body or "MS:1002748" in body:
            return None
        if "MS:1000523" in body:
            dtype = np.dtype("<f8")
        elif "MS:1000521" in body:
            dtype = np.dtype("<f4")
        else:
            return None
        if "MS:1000574" not in body and "MS:1000576" not in body and "compression" in body.lower():
            return None
        binary = re.search(r"<binary>([^<]*)</binary>", body)
        payload = "" if binary is None else binary.group(1).strip()
        if not payload:
            found[kind] = np.array([], dtype=np.float64)
            continue
        raw = base64.b64decode(payload)
        if "MS:1000574" in body:
            raw = zlib.decompress(raw)
        count = len(raw) // dtype.itemsize
        found[kind] = np.frombuffer(raw[: count * dtype.itemsize], dtype=dtype).astype(np.float64, copy=True)
    if "mz" not in found or "intensity" not in found:
        return None
    return found["mz"], found["intensity"]


def _array_bodies(block: str):
    """The _ARRAY.finditer() matches, without lazy-regex scanning of payloads."""
    start = 0
    while True:
        found = block.find("<binaryDataArray", start)
        if found < 0:
            return
        after = found + 16
        if after < len(block) and (block[after].isalnum() or block[after] == "_"):
            start = found + 1  # not at a \b word boundary
            continue
        end = block.find("</binaryDataArray>", after)
        if end < 0:
            return
        yield block[found : end + 18]
        start = end + 18


def _find(buf: bytearray, token: bytes, start: int = 0) -> int:
    """buf.find(token, start) for a token starting with '<', hopping by memchr.

    Base64 payloads contain no '<', so large arrays are skipped at memchr speed."""
    find, startswith = buf.find, buf.startswith
    while True:
        found = find(b"<", start)
        if found < 0 or startswith(token, found):
            return found
        start = found + 1


def _find_spectrum_start(buf: bytearray) -> int:
    token = _OPEN
    token_len = len(token)
    start = 0
    limit = len(buf)
    while True:
        found = _find(buf, token, start)
        if found < 0:
            return -1
        marker = found + token_len
        if marker >= limit:
            return -1
        if buf[marker : marker + 1] in (b" ", b">", b"\t", b"\n", b"\r", b"/"):
            return found
        start = found + token_len
