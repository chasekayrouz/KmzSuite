"""MS1 header index for mzML files.

The scanner reads spectrum metadata and skips peak arrays. MS2 spectra are
counted so spectrum indexes stay aligned with the file, then discarded.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import numpy as np

from kmzcore.cancel import Cancelled
from kmzcore.mzml import looks_like_mzml
from kmzcore.sparse_profile import Provenance, read_provenance
from kmzviewer.extract import fill_missing_tic
from kmzviewer.model import ScanTable, DadScanTable
from kmzviewer.mzml_direct import IndexRecorder, find_native

_OPEN = b"<spectrum"
_CLOSE = b"</spectrum>"
_CLOSE_LEN = len(_CLOSE)
_CV = re.compile(r"<cvParam\b([^>]*)/?>")
_ATTR = re.compile(r'([A-Za-z_:][\w:.-]*)="([^"]*)"')
_MINUTE_UNITS = {"uo:0000031", "minute", "minutes", "min"}
_SECOND_UNITS = {"uo:0000010", "second", "seconds", "sec", "s"}


def load_ms1(
    path: str | Path,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ScanTable:
    """Index MS1 headers and fill any missing per-scan total ion current."""
    path = Path(path)
    if not looks_like_mzml(path):
        raise ValueError(f"{path.name} is not an mzML file.")
    table = index_ms1(
        path,
        chunk_size=chunk_size,
        on_progress=on_progress,
        is_cancelled=is_cancelled,
    )
    if len(table) == 0 and table.dad is None:
        raise ValueError(f"No MS1 scans in {path.name}.")
    if np.any(~np.isfinite(table.tic)):
        fill_missing_tic(
            str(path),
            table,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
        )
    return table


def index_ms1(
    path: str | Path,
    *,
    chunk_size: int = 4 * 1024 * 1024,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ScanTable:
    """Return one row per MS1 scan. Retention times are in minutes."""
    path = Path(path)
    size = path.stat().st_size
    scanner = _HeaderScanner(is_cancelled, read_provenance(path, is_cancelled=is_cancelled))
    read = 0
    with path.open("rb") as handle:
        # Spectrum readers reuse this pass instead of scanning the file again.
        recorder = IndexRecorder.start(str(path), handle)
        while True:
            if is_cancelled is not None and is_cancelled():
                raise Cancelled()
            chunk = handle.read(chunk_size)
            read += len(chunk)
            scanner.feed(chunk, final=not chunk)
            if recorder is not None:
                recorder.feed(chunk, final=not chunk)
            if on_progress is not None:
                on_progress(min(read, size), size)
            if not chunk:
                break
        if recorder is not None:
            recorder.finish()
    table = _table_from_records(scanner.records)
    if scanner.dad_records:
        records = scanner.dad_records
        table.dad = DadScanTable(np.array([r['spectrum_index'] for r in records], dtype=np.int32),
                                 np.array([r['rt_min'] for r in records]), [r['scan_id'] for r in records])
    table.dad_error = scanner.dad_error
    return table


def _table_from_records(records: list[dict]) -> ScanTable:
    count = len(records)
    return ScanTable(
        spectrum_index=np.array([row["spectrum_index"] for row in records], dtype=np.int32),
        rt_min=np.array([row["rt_min"] for row in records], dtype=np.float64),
        tic=np.array([row["tic"] for row in records], dtype=np.float64),
        polarity=np.array([row["polarity"] for row in records], dtype=np.int8),
        centroid=np.array([row["centroid"] for row in records], dtype=np.int8),
        scan_id=[row["scan_id"] for row in records],
        sparse_profile=[row.get("sparse_profile") for row in records],
    ) if count else ScanTable(
        spectrum_index=np.array([], dtype=np.int32),
        rt_min=np.array([], dtype=np.float64),
        tic=np.array([], dtype=np.float64),
        polarity=np.array([], dtype=np.int8),
        centroid=np.array([], dtype=np.int8),
        scan_id=[],
    )


class _HeaderScanner:
    def __init__(self, is_cancelled=None, provenance=None) -> None:
        self.is_cancelled = is_cancelled
        self.provenance = provenance or Provenance()
        self.buf = bytearray()
        self.skipping = False
        self.ordinal = 0
        self.records: list[dict] = []
        self.dad_records: list[dict] = []
        self.dad_error = ""

    def feed(self, data: bytes, final: bool = False) -> None:
        if data:
            self.buf.extend(data)
        while True:
            if self.is_cancelled is not None and self.is_cancelled():
                raise Cancelled()
            if self.skipping:
                if not self._skip_rest_of_spectrum(final):
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
            close_at = _find_close(self.buf, binary_at)
            if binary_at < 0 and close_at < 0:
                if len(self.buf) > 2_000_000:
                    raise ValueError("An mzML spectrum header exceeded 2 MB.")
                if final:
                    raise ValueError("The mzML file ends inside a spectrum header.")
                return
            header_end = binary_at if binary_at >= 0 else close_at
            header = bytes(self.buf[:header_end])
            optical = b'MS:1000806' in header or b'MS:1000620' in header
            if optical:
                try:
                    record = parse_spectrum_header(header, self.ordinal, optical=True)
                    if record is None:
                        raise ValueError("DAD spectrum has no retention time")
                    if not np.isfinite(record['rt_min']):
                        raise ValueError("DAD spectrum has a nonfinite retention time")
                    self.dad_records.append(record)
                except (ValueError, OverflowError) as exc:
                    self.dad_error = str(exc)
                record = None
            else:
                record = parse_spectrum_header(header, self.ordinal)
            self.ordinal += 1
            if record is not None:
                record["sparse_profile"] = None
                if self.provenance.methods:
                    # Only the spectrum ParamGroup is needed, before nested metadata.
                    prefix = header.split(b"<scanList", 1)[0]
                    try:
                        record["sparse_profile"] = self.provenance.sampling(ET.fromstring(prefix + b"</spectrum>"))
                    except ET.ParseError:
                        pass
                self.records.append(record)
            if close_at >= 0:
                del self.buf[: close_at + _CLOSE_LEN]
                continue
            # Closing tag may be split across the next read. Keep a short tail.
            tail = bytes(self.buf[-(_CLOSE_LEN - 1) :])
            self.buf.clear()
            self.buf.extend(tail)
            self.skipping = True
            if final:
                raise ValueError("The mzML file ends inside a spectrum.")
            return

    def _skip_rest_of_spectrum(self, final: bool) -> bool:
        close_at = _find_close(self.buf, 0)
        if close_at < 0:
            if len(self.buf) > _CLOSE_LEN:
                del self.buf[:-_CLOSE_LEN]
            if final:
                raise ValueError("The mzML file ends inside a spectrum.")
            return False
        del self.buf[: close_at + _CLOSE_LEN]
        self.skipping = False
        return True


def parse_spectrum_header(header: bytes, ordinal: int, *, optical=False) -> dict | None:
    """Parse one spectrum header. Return None unless the scan is MS1 with a time."""
    text = header.decode("utf-8", "replace")
    opener = text.split(">", 1)[0]
    open_attrs = dict(_ATTR.findall(opener))
    level = None
    polarity = 0
    centroid = -1
    tic = None
    rt = None
    rt_unit = ""
    for source in _CV.findall(text):
        found = _cv_memo.get(source)
        if found is None:
            found = _cv_param(source)
        accession, value, unit = found
        if accession == "MS:1000511" and level is None and value:
            level = int(float(value))
        elif accession == "MS:1000130":
            polarity = 1
        elif accession == "MS:1000129":
            polarity = -1
        elif accession == "MS:1000127":
            centroid = 1
        elif accession == "MS:1000128":
            centroid = 0
        elif accession == "MS:1000285" and value:
            tic = float(value)
        elif accession == "MS:1000016" and rt is None and value:
            rt = float(value)
            rt_unit = unit
    if (not optical and level != 1) or rt is None:
        return None
    index_text = open_attrs.get("index")
    spectrum_index = int(index_text) if index_text not in (None, "") else ordinal
    tic_value = np.nan if tic is None else tic
    return {
        "spectrum_index": spectrum_index,
        "scan_id": open_attrs.get("id", ""),
        "rt_min": _to_minutes(rt, rt_unit),
        "tic": tic_value,
        "polarity": polarity,
        "centroid": centroid,
    }


_USED = re.compile("MS:1000(?:511|130|129|127|128|285|016)")
_IGNORED = ("", None, "")
_cv_memo: dict[str, tuple] = {}


def _cv_param(source: str) -> tuple:
    """(accession, value, unit) of one cvParam's attributes; most strings recur verbatim.

    Attribute values are raw substrings, so a source naming no used accession is
    ignored without parsing.
    """
    if _USED.search(source):
        attrs = dict(_ATTR.findall(source))
        found = (attrs.get("accession", ""), attrs.get("value"),
                 attrs.get("unitAccession") or attrs.get("unitName") or "")
    else:
        found = _IGNORED
    if len(_cv_memo) >= 1024:
        _cv_memo.clear()
    _cv_memo[source] = found
    return found


def _to_minutes(value: float, unit: str) -> float:
    token = (unit or "").strip().lower()
    if token in _SECOND_UNITS:
        return value / 60.0
    if token in _MINUTE_UNITS or token == "":
        return value
    raise ValueError(f"Unsupported scan start time unit: {unit}")


def _find_close(buf: bytearray, after: int) -> int:
    """Exactly ``buf.find(_CLOSE)``, stepping between tags after ``after`` at memchr speed.

    The closing tag holds no other '<', so no match can straddle the '<' at ``after``.
    """
    if find_native is not None:
        return find_native(buf, _CLOSE)
    found = buf.find(_CLOSE, 0, after) if after > 0 else -1
    if found >= 0:
        return found
    found = buf.find(b"<", max(after, 0))
    while found >= 0 and not buf.startswith(_CLOSE, found):
        found = buf.find(b"<", found + 1)
    return found


def _find_spectrum_start(buf: bytearray) -> int:
    token = _OPEN
    token_len = len(token)
    start = 0
    limit = len(buf)
    while True:
        found = buf.find(token, start)
        if found < 0:
            return -1
        marker = found + token_len
        if marker >= limit:
            return -1
        if buf[marker : marker + 1] in (b" ", b">", b"\t", b"\n", b"\r", b"/"):
            return found
        start = found + token_len
