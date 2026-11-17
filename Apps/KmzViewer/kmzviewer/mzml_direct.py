"""Exact direct decoding of plainly written mzML spectra.

pyteomics remains the reference reader. This module reproduces its spectrum byte
index and, for spectra written in a strict, recognised markup subset, the result of
its array decoding: strict base64, zlib or no compression, and native
``numpy.frombuffer``. Any other markup, value or failure returns ``None``, so the
caller uses pyteomics unchanged, including its errors. So do pyteomics or NumPy
versions other than those qualified. Decoding is thread-safe.
"""

from __future__ import annotations

import binascii
import os
import re
import threading
import zlib
from bisect import bisect_right
from collections import OrderedDict

import numpy as np

from kmzcore.qualified import qualified

try:  # Optional native helpers; see _native.c.
    from kmzviewer._native import find as find_native, indexed_tags as _indexed_tags, \
        inflate_base64 as _inflate_base64
except ImportError:
    find_native = _indexed_tags = _inflate_base64 = None
ACCELERATED = _inflate_base64 is not None

# pyteomics' ByteCountingXMLScanner: each '<'-delimited chunk is matched separately,
# its first 1,000,000-byte block drops empty chunks, and the final chunk is unread.
_INDEXED = re.compile(rb"<(?:\w+:)?(spectrum|chromatogram)\s")
_PARTIAL = re.compile(rb"<\w*(?::\w*)?")
_LOOSE_ATTRS = re.compile(rb"(\S+)=[\"']([^\"']*)[\"']")
_ENTITY = re.compile(r"&(quot|amp|apos|lt|gt);")
_ENTITIES = {"quot": '"', "amp": "&", "apos": "'", "lt": "<", "gt": ">"}
_FIRST_BLOCK = 1_000_000
_READ_BLOCK = 4 * 1024 * 1024

_S = rb"[ \t\r\n]"
_NAME = rb"[A-Za-z_][A-Za-z0-9_.-]*"
_VALUE = rb'"(?:[^"<&\x00-\x1f\x7f-\xff]|&(?:quot|amp|lt|gt|apos);)*"'
_PLAIN_VALUE = rb'"[^"<&\x00-\x1f\x7f-\xff]*"'  # The same language without '&'; faster.


def _tag_patterns(value: bytes) -> tuple[re.Pattern, re.Pattern]:
    attr = rb"%s+%s%s*=%s*%s" % (_S, _NAME, _S, _S, value)
    return (re.compile(rb"%s*<(/?)(%s)((?:%s)*)%s*(/?)>" % (_S, _NAME, attr, _S)),
            re.compile(rb"(?:%s*</?%s(?:%s)*%s*/?>)*%s*" % (_S, _NAME, attr, _S, _S)))


_TAG, _TAGS = _tag_patterns(_VALUE)
_PLAIN_TAG, _PLAIN_TAGS = _tag_patterns(_PLAIN_VALUE)
_ATTRS = re.compile(rb"%s+(%s)%s*=%s*\"([^\"]*)\"" % (_S, _NAME, _S, _S))
_DIGITS = re.compile(rb"[0-9]+")

_PARAMS = {b"cvParam", b"userParam"}
_CONTAINERS = {b"scanList", b"scan", b"scanWindowList", b"scanWindow"}
_INTEGERS = {b"count", b"index", b"defaultArrayLength", b"arrayLength", b"encodedLength"}
# Keys with special meaning in pyteomics' element flattening and binary handling.
_STRUCTURAL = {"binary", "binaryDataArray", "binaryDataArrayList", "referenceableParamGroupRef"}
_STRUCTURAL_ATTRS = {name.encode() for name in _STRUCTURAL}
# pyteomics merges same-named params into element attributes and then converts
# schema fields, so such collisions can fail there; leave them to pyteomics.
_ATTRIBUTE_NAMES = {name.decode() for name in _INTEGERS} | {
    "id", "dataProcessingRef", "sourceFileRef", "spotID", "instrumentConfigurationRef"}
_OPTICAL = {"absorption spectrum", "PDA spectrum"}
_RESERVED = {"m/z array", "intensity array", "wavelength array",
             "intensity array unit", "wavelength array unit"} | _STRUCTURAL
_DTYPES = {"32-bit float": np.float32, "64-bit float": np.float64,
           "32-bit integer": np.int32, "64-bit integer": np.int64}
_COMPRESSION = {"zlib compression", "no compression"}
_UNIT_ARRAYS = {"wavelength array", "intensity array"}
_ARRAY_ATTRS = {b"encodedLength", b"arrayLength", b"dataProcessingRef"}
# libxml2 rejects text nodes over 10,000,000 bytes without pyteomics' huge_tree.
_MAX_TEXT = 8 * 1024 * 1024
_MAX_SPECTRUM = 64 * 1024 * 1024
_MAX_HINT = 64 * 1024 * 1024
_MEMO = 1024

# Accessions verified in the pinned PSI-MS vocabulary (see Tests/Viewer). pyteomics
# looks up valued PSI-MS cvParams there and fails on unknown terms; other
# accessions load that vocabulary on first use.
_KNOWN_ACCESSIONS = {
    "MS:1000016", "MS:1000127", "MS:1000128", "MS:1000129", "MS:1000130",
    "MS:1000285", "MS:1000294", "MS:1000500", "MS:1000501", "MS:1000504",
    "MS:1000505", "MS:1000511", "MS:1000512", "MS:1000514", "MS:1000515",
    "MS:1000519", "MS:1000521", "MS:1000522", "MS:1000523", "MS:1000527",
    "MS:1000528", "MS:1000574", "MS:1000576", "MS:1000579", "MS:1000580",
    "MS:1000616", "MS:1000617", "MS:1000618", "MS:1000619", "MS:1000620",
    "MS:1000795", "MS:1000796", "MS:1000806", "MS:1000927",
}
_accessions: dict[str, bool] = {}

_indexes: OrderedDict[tuple, tuple | None] = OrderedDict()
_indexes_lock = threading.Lock()
_INDEX_LIMIT = 8


def _known(accession: str) -> bool:
    if accession in _KNOWN_ACCESSIONS:
        return True
    found = _accessions.get(accession)
    if found is None:
        from kmzviewer.spectra import _local_vocabulary
        try:
            _local_vocabulary()[accession]
            found = True
        except Exception:
            found = False
        _accessions[accession] = found
    return found


def _text(value: bytes) -> str:
    if b"&" in value:
        value = (value.replace(b"&lt;", b"<").replace(b"&gt;", b">").replace(b"&quot;", b'"')
                 .replace(b"&apos;", b"'").replace(b"&amp;", b"&"))
    return value.decode("ascii")


def _attributes(blob: bytes) -> dict[bytes, bytes] | None:
    found = {}
    for name, value in _ATTRS.findall(blob):
        if name in found or name in _STRUCTURAL_ATTRS:
            return None
        if name in _INTEGERS and not _DIGITS.fullmatch(value):
            return None
        found[name] = value
    return found


def open_direct(path: str) -> DirectSpectra | None:
    """Return a decoder sharing a validated index, or None if pyteomics must read."""
    if not qualified("pyteomics", "numpy"):
        return None  # The index and arrays reproduce the qualified pyteomics and NumPy.
    path = str(path)
    if _saved_index(path):
        return None  # pyteomics would load this saved index instead of scanning.
    stat = os.stat(path)
    key = _identity(path, stat)
    with _indexes_lock:
        found = _indexes.get(key, False)
        if found is not False:
            _indexes.move_to_end(key)
    if found is False:
        found = _build_index(path)
        if _identity(path, os.stat(path)) != key:
            return None  # Replaced while indexing: never cache or use a mixed index.
        _remember(key, found)
    if found is None:
        return None
    direct = DirectSpectra(path, *found, stat.st_size)
    if _identity(path, os.fstat(direct._fd)) != key:
        direct.close()  # The opened descriptor is not the indexed file version.
        return None
    return direct


def _saved_index(path: str) -> bool:
    name, ext = os.path.splitext(path)
    return os.path.exists(f"{name}-{ext[1:]}-byte-offsets.json")


def _identity(path: str, stat: os.stat_result) -> tuple:
    # ctime also changes on same-size rewrites that restore mtime.
    return path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _remember(key: tuple, found: tuple | None) -> None:
    with _indexes_lock:
        _indexes[key] = found
        _indexes.move_to_end(key)
        while len(_indexes) > _INDEX_LIMIT:
            _indexes.popitem(last=False)


class IndexRecorder:
    """Build the index from a caller's sequential read of an open file.

    The caller passes every chunk read from offset 0 to feed() and calls finish()
    after end of file. The result is shared only if the path and descriptor
    still name the version indexed.
    """

    def __init__(self, path: str, handle) -> None:
        self.path = str(path)
        self._handle = handle
        self._key = _identity(self.path, os.fstat(handle.fileno()))
        self._scan = _IndexScan()
        self._failed = False
        try:
            _read_version(handle)
        except Exception:
            self._failed = True
        finally:
            handle.seek(0)

    @classmethod
    def start(cls, path: str, handle) -> IndexRecorder | None:
        """A recorder when the native scanner can index unseen, current files."""
        path = str(path)
        if _indexed_tags is None or not qualified("pyteomics", "numpy") or _saved_index(path):
            return None
        try:
            key = _identity(path, os.fstat(handle.fileno()))
            if _identity(path, os.stat(path)) != key:
                return None
            with _indexes_lock:
                if key in _indexes:
                    return None
            return cls(path, handle)
        except Exception:
            handle.seek(0)
            return None

    def feed(self, chunk: bytes, final: bool = False) -> None:
        self._scan.feed(chunk, final)

    def finish(self) -> None:
        if not self._scan.done:
            return
        found = None if self._failed else _index_from_entries(self._scan.entries)
        try:
            current = (_identity(self.path, os.fstat(self._handle.fileno())) == self._key
                       == _identity(self.path, os.stat(self.path)))
        except OSError:
            return
        if current:
            _remember(self._key, found)


def _build_index(path: str) -> tuple | None:
    """Spectrum ids/offsets exactly as pyteomics' MzML constructor would index them."""
    try:
        with open(path, "rb") as handle:
            _read_version(handle)
            handle.seek(0)
            scan = _IndexScan()
            while not scan.done:
                block = handle.read(_READ_BLOCK)
                scan.feed(block, final=not block)
    except Exception:
        return None
    return _index_from_entries(scan.entries)


def _index_from_entries(entries: list | None) -> tuple | None:
    if entries is None:
        return None
    tables = {}
    for tag, key, offset in entries:
        if key is None:
            return None
        try:
            key = key.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if "&" in key:
            key = _ENTITY.sub(lambda match: _ENTITIES[match.group(1)], key)
        tables.setdefault(tag, {})[key] = offset
    spectra = tables.get(b"spectrum", {})
    # pyteomics resolves ids across element types in first-seen type order.
    if spectra.keys() & tables.get(b"chromatogram", {}).keys():
        return None
    return tuple(spectra), tuple(spectra.values()), tuple(offset for *_, offset in entries)


def _read_version(handle) -> None:
    """Raise wherever pyteomics' constructor fails while locating the mzML root."""
    from lxml import etree
    for _event, element in etree.iterparse(handle, events=("start",), remove_comments=True, huge_tree=False):
        if element.tag.rpartition("}")[2] == "mzML":
            return


def _chunks(buffer: bytes) -> tuple:
    """((offset, next '<') for each '<' but the last, the last '<' or -1).

    The native helper lists only chunks that match _INDEXED; callers re-check.
    """
    if _indexed_tags is not None:
        return _indexed_tags(buffer)
    return _python_chunks(buffer), buffer.rfind(b"<")


def _python_chunks(buffer: bytes):
    at = buffer.find(b"<")
    following = buffer.find(b"<", at + 1) if at >= 0 else -1
    while following >= 0:
        yield at, following
        at, following = following, buffer.find(b"<", following + 1)


class _IndexScan:
    """Lexical index of consecutive file bytes, scanned in fixed _READ_BLOCK units.

    Re-blocking makes the result independent of the caller's chunk sizes.
    ``entries`` becomes None when pyteomics must index; ``done`` follows end of file.
    """

    def __init__(self) -> None:
        self.entries: list | None = []
        self._final = False
        self._pending = bytearray()
        self._carry = b""
        self._base = 0

    @property
    def done(self) -> bool:
        return self._final or self.entries is None

    def feed(self, data: bytes, final: bool = False) -> None:
        if self.done:
            return
        if len(data) == _READ_BLOCK and not self._pending:
            self._block(data, False)
        else:
            self._pending += data
            while len(self._pending) >= _READ_BLOCK and self.entries is not None:
                block = bytes(self._pending[:_READ_BLOCK])
                del self._pending[:_READ_BLOCK]
                self._block(block, False)
            if final and self._pending and self.entries is not None:
                block = bytes(self._pending)
                self._pending.clear()
                self._block(block, False)
        if final and self.entries is not None:
            self._block(b"", True)
            self._final = True

    def _block(self, block: bytes, final: bool) -> None:
        buffer = self._carry + block if self._carry else block
        base = self._base
        if base < _FIRST_BLOCK and b"<<" in buffer[:_FIRST_BLOCK - base + 1]:
            self.entries = None
            return
        chunks, last = _chunks(buffer)
        carry = b""
        if last >= 0:
            for at, following in chunks:
                match = _INDEXED.match(buffer, at, following)
                if match is not None:
                    key = dict(_LOOSE_ATTRS.findall(buffer, at, following)).get(b"id")
                    self.entries.append((match.group(1), key, base + at))
            if not final and (_INDEXED.match(buffer, last) or _PARTIAL.fullmatch(buffer, last)):
                if len(buffer) - last > _MAX_SPECTRUM:
                    self.entries = None
                    return
                carry = buffer[last:]
        self._carry = carry
        self._base = base + len(buffer) - len(carry)


class DirectSpectra:
    """One file's validated index and descriptor; decode() is safe across threads."""

    def __init__(self, path, ids, offsets, bounds, size) -> None:
        self.ids = ids
        self.offsets = offsets
        self._bounds = bounds
        self._size = size
        self._fd = os.open(path, os.O_RDONLY)
        self._params: dict[bytes, tuple | None] = {}
        self._plain: dict[bytes, bool] = {}

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def decode(self, ordinal: int) -> dict | None:
        """pyteomics' relevant spectrum keys, or None when it must decode instead."""
        try:
            return self._decode(int(ordinal))
        except Exception:
            return None

    def _decode(self, ordinal: int) -> dict | None:
        if not 0 <= ordinal < len(self.ids):
            return None
        start = self.offsets[ordinal]
        following = bisect_right(self._bounds, start)
        stop = self._bounds[following] if following < len(self._bounds) else self._size
        data = os.pread(self._fd, min(stop - start, _MAX_SPECTRUM), start)
        # Attribute values hold no '<', so the start tag cannot contain this text.
        listed = data.find(b"<binaryDataArrayList")
        if listed < 0:
            return None
        tag, tags = (_TAG, _TAGS) if data.find(b"&", 0, listed) >= 0 else (_PLAIN_TAG, _PLAIN_TAGS)
        opener = tag.match(data, 0, listed)
        if opener is None or opener.group(1) or opener.group(4) or opener.group(2) != b"spectrum":
            return None
        attrs = _attributes(opener.group(3))
        if attrs is None or b"id" not in attrs or _text(attrs[b"id"]) != self.ids[ordinal]:
            return None
        if not tags.fullmatch(data, opener.end(), listed):
            return None
        spectrum = {}
        open_tags = []
        for closing, name, blob, empty in tag.findall(data, opener.end(), listed):
            if closing:
                if blob or empty or not open_tags or open_tags.pop() != name:
                    return None
            elif name in _PARAMS:
                param = self._param(blob) if empty else None
                if param is None:
                    return None
                if not open_tags:
                    if param[0] in _RESERVED:
                        return None
                    if param[0] in _OPTICAL:
                        spectrum[param[0]] = True
            elif name in _CONTAINERS and self._container(blob):
                if not empty:
                    open_tags.append(name)
            else:
                return None
        if open_tags:
            return None
        count = attrs.get(b"defaultArrayLength")
        return self._arrays(data, listed, spectrum, int(count) if count is not None else 0)

    def _arrays(self, data: bytes, at: int, spectrum: dict, count: int) -> dict | None:
        tag = _TAG.match(data, at)
        if tag is None or tag.group(1) or tag.group(4) or tag.group(2) != b"binaryDataArrayList":
            return None
        attrs = _attributes(tag.group(3))
        if attrs is None or set(attrs) - {b"count"}:
            return None
        at = tag.end()
        while True:
            tag = _TAG.match(data, at)
            if tag is None:
                return None
            closing, name, blob, empty = tag.groups()
            at = tag.end()
            if closing:
                break
            if name != b"binaryDataArray" or empty:
                return None
            attrs = _attributes(blob)
            if attrs is None or set(attrs) - _ARRAY_ATTRS:
                return None
            at = self._array(data, at, spectrum, count)
            if at is None:
                return None
        if name != b"binaryDataArrayList" or blob or empty:
            return None
        tag = _TAG.match(data, at)
        if tag is None or tag.groups() != (b"/", b"spectrum", b"", b""):
            return None
        return spectrum

    def _array(self, data: bytes, at: int, spectrum: dict, count: int) -> int | None:
        dtype = compression = array = None
        while True:
            tag = _TAG.match(data, at)
            if tag is None:
                return None
            closing, name, blob, empty = tag.groups()
            at = tag.end()
            if name == b"binary" and not (closing or blob or empty):
                break
            param = self._param(blob) if name == b"cvParam" and empty and not closing else None
            if param is None:
                return None
            label, blank, unit = param
            if label in _DTYPES and dtype is None:
                dtype = _DTYPES[label]
            elif label in _COMPRESSION and compression is None:
                compression = label
            elif label.endswith(" array") and label != "non-standard data array" and blank and array is None:
                array = label, unit
            else:
                return None
        if dtype is None or compression is None or array is None:
            return None
        compressed = compression == "zlib compression"
        hint = count * np.dtype(dtype).itemsize if compressed else 0
        if _inflate_base64 is not None and (0 < hint <= _MAX_HINT or not compressed):
            # The text end and, off the GIL, the helper's private bytearray or None.
            raw, end = _inflate_base64(data, at, _MAX_TEXT, compressed, hint)
        else:
            raw, end = None, data.find(b"<", at)
        if end < 0 or end - at > _MAX_TEXT:
            return None
        text = memoryview(data)[at:end]
        for expected in (b"binary", b"binaryDataArray"):
            tag = _TAG.match(data, end)
            if tag is None or tag.groups() != (b"/", expected, b"", b""):
                return None
            end = tag.end()
        if len(text):
            if raw is None:  # binascii and zlib decide, as without the helper.
                raw = binascii.a2b_base64(text, strict_mode=True)
                if compressed:
                    raw = zlib.decompress(raw, zlib.MAX_WBITS, hint if 0 < hint <= _MAX_HINT else zlib.DEF_BUF_SIZE)
                raw = bytearray(raw)
            # pyteomics returns writable arrays over a private copy of these bytes.
            values = np.frombuffer(raw, dtype=dtype)
        else:
            values = np.array([], dtype=dtype)
        label, unit = array
        spectrum[label] = values
        if label in _UNIT_ARRAYS:
            spectrum[label + " unit"] = unit
        return end

    def _param(self, blob: bytes) -> tuple | None:
        """(name, value is empty, pyteomics unit info) for an accepted cv/userParam."""
        found = self._params.get(blob, False)
        if found is not False:
            return found
        found = None
        attrs = _attributes(blob)
        if attrs is not None and b"name" in attrs and b"id" not in attrs:
            name = _text(attrs[b"name"])
            accession = attrs.get(b"accession")
            unit = None
            if b"unitCvRef" in attrs or b"unitName" in attrs:
                unit = attrs.get(b"unitName") or attrs.get(b"unitAccession")
                unit = None if unit is None else _text(unit)
            needs_unit_term = b"unitName" not in attrs and b"unitAccession" in attrs and unit is not None
            needs_term = (accession is not None and b"value" in attrs
                          and attrs.get(b"cvRef") in (b"MS", b"PSI-MS"))
            if (name not in _STRUCTURAL and name not in _ATTRIBUTE_NAMES and not needs_unit_term
                    and (not needs_term or _known(_text(accession)))):
                found = name, not attrs.get(b"value"), unit
        if len(self._params) >= _MEMO:
            self._params.clear()
        self._params[blob] = found
        return found

    def _container(self, blob: bytes) -> bool:
        found = self._plain.get(blob)
        if found is None:
            attrs = _attributes(blob)
            found = attrs is not None and b"id" not in attrs
            if len(self._plain) >= _MEMO:
                self._plain.clear()
            self._plain[blob] = found
        return found
