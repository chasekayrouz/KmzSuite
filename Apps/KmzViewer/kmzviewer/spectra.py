"""Lazy spectrum decoding and a memory-capped cache of arrays already read."""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kmzcore import psi_ms
from kmzviewer import mzml_direct
from kmzviewer.mzml_direct import open_direct

# Two profile-mode LC-QTOF files in this project decode to roughly 800 MB each.
# The budget is large enough to keep both, so a later EIC does not decompress them again.
# When a new file still does not fit, older files are dropped whole. Scans already stored
# for the file being read are kept, which avoids evicting the prefix of a sequential pass.
DEFAULT_CACHE_BUDGET = 2 * 1024 * 1024 * 1024

# zlib inflation releases the GIL; markup checks do not. Without the native helper
# base64 holds it too, so four helpers saturate a sequential pass; with it, eight.
_DECODERS = max(1, min(8 if mzml_direct.ACCELERATED else 4, (os.cpu_count() or 1) - 1))
_AHEAD = 4 * _DECODERS
_MISSING = object()

# A spectrum whose two float64 arrays reach this size gets private anonymous pages
# instead of malloc memory. macOS malloc keeps freed 64 KiB-1 MiB blocks committed,
# so an evicted or closed file's decoded arrays would otherwise stay resident.
_PAGED_MIN = 64 * 1024
_PAGE_FLAGS = {"flags": mmap.MAP_PRIVATE} if hasattr(mmap, "MAP_PRIVATE") else {}
# Frozen pages are mapped read-only and made writable only while being filled, with
# libc's mprotect. Where it is unavailable (Windows), frozen arrays use bytes.
try:
    _mprotect = ctypes.CDLL(None).mprotect
    _mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    _mprotect.restype = ctypes.c_int
except (AttributeError, OSError, TypeError):
    _mprotect = None


class _Pages(mmap.mmap):
    """Private pages behind one spectrum's arrays; only _paged makes them."""
    __slots__ = ()


class _Frozen(tuple):
    """An (m/z, intensity) pair over memory nothing can write; only _freeze makes one.

    SpectrumReader.read returns one only to read_arrays, which hands the cache this
    new pair instead of a copy. ArrayCache.put copies whatever it is given. The
    cache stores plain tuples, which the GC stops tracking; this subclass it cannot.
    """
    __slots__ = ()


class _Raw:
    """A writable NumPy view of read-only pages while _paged fills them."""
    __slots__ = ("__array_interface__",)

    def __init__(self, address: int, count: int) -> None:
        self.__array_interface__ = {"data": (address, False), "shape": (count,),
                                    "typestr": np.dtype(np.float64).str, "version": 3}


def _halves(block: np.ndarray, mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return block[:mz.size].reshape(mz.shape), block[mz.size:].reshape(intensity.shape)


def _fill(block: np.ndarray, mz: np.ndarray, intensity: np.ndarray) -> None:
    # Assignment applies the same exact casts as np.asarray(..., float64).
    first, second = _halves(block, mz, intensity)
    first[...] = mz
    second[...] = intensity


def _paged(mz: np.ndarray, intensity: np.ndarray, frozen: bool = False) -> np.ndarray | None:
    """Both arrays as float64 in one block of new private pages, or None when
    small, unusual or unmappable.

    A frozen block's pages are mapped read-only, so NumPy cannot make arrays over
    them writable, and accept writes in hardware only while a raw view fills them.
    """
    count = mz.size + intensity.size
    if (8 * count < _PAGED_MIN or mz.dtype.kind not in "fiu" or intensity.dtype.kind not in "fiu"
            or frozen and _mprotect is None):
        return None
    try:
        pages = _Pages(-1, 8 * count, **({"prot": mmap.PROT_READ} if frozen else {}), **_PAGE_FLAGS)
    except (OSError, ValueError):
        return None  # For example a mapping limit: use NumPy memory as before.
    # The block's base is the mapping itself; only it is tracked by the GC.
    block = np.ndarray((count,), dtype=np.float64, buffer=pages)
    if not frozen:
        _fill(block, mz, intensity)
        return block
    address = block.__array_interface__["data"][0]
    if _mprotect(address, len(pages), mmap.PROT_READ | mmap.PROT_WRITE):
        return None
    _fill(np.asarray(_Raw(address, count)), mz, intensity)
    _mprotect(address, len(pages), mmap.PROT_READ)
    return block


def _float64_pair(mz, intensity, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The writable float64 arrays np.asarray returns, in one page block when large.

    Small, unusual or unmappable input stays in ordinary NumPy memory, copied on request.
    """
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    block = _paged(mz, intensity)
    if block is not None:
        return _halves(block, mz, intensity)
    if copy:
        return mz.astype(np.float64, order="C"), intensity.astype(np.float64, order="C")
    return np.asarray(mz, dtype=np.float64), np.asarray(intensity, dtype=np.float64)


def _freeze(mz, intensity) -> _Frozen:
    """Float64 copies, cast as by np.asarray, over memory nothing can write.

    Large spectra get frozen pages and others bytes. Both export read-only
    buffers, so NumPy refuses setflags(write=True) on any array over them.
    """
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    block = _paged(mz, intensity, frozen=True)
    if block is None:
        mz, intensity = np.asarray(mz, dtype=np.float64), np.asarray(intensity, dtype=np.float64)
        raw = b"".join((np.ascontiguousarray(mz), np.ascontiguousarray(intensity)))
        block = np.frombuffer(raw, dtype=np.float64)
    return _Frozen(_halves(block, mz, intensity))


def _decode_ahead(direct, index: int, prepare=_float64_pair):
    """Helper-thread decode. MS spectra become the arrays read() returns here, so
    the raw decoded buffers are released rather than queued."""
    spectrum = direct.decode(index)
    if (spectrum is None or "wavelength array" in spectrum
            or "absorption spectrum" in spectrum or "PDA spectrum" in spectrum):
        return spectrum
    return prepare(spectrum.get("m/z array", []), spectrum.get("intensity array", []))


def _decode_frozen(direct, index: int):
    """_decode_ahead for spectra planned for read_arrays: MS arrays are _Frozen."""
    return _decode_ahead(direct, index, _freeze)


# Set on a thread while read_arrays' loader runs: SpectrumReader.read then returns
# MS arrays as a new _Frozen pair, which read_arrays stores without copying.
_filling = threading.local()


# pyteomics and psims take ~0.2 s to import and only fallback reads need them, so
# _UnitMzML is bound on first use instead of during application startup. Internal
# calls read it back from the module, so replacing the attribute still changes
# what is used, as with the former eager import.
_fallback_lock = threading.Lock()


def _local_vocabulary():
    """The suite's pinned PSI-MS vocabulary, loaded once without remote I/O."""
    return psi_ms.vocabulary()


def _unit_mzml():
    """Return the module's _UnitMzML, defining it once on first use."""
    with _fallback_lock:
        if "_UnitMzML" not in globals():
            from pyteomics import mzml

            class _UnitMzML(mzml.MzML):
                """Keep optical array units that pyteomics otherwise discards on decoding."""
                def _handle_binary(self, info, **kwargs):
                    units = {f"{name} unit": getattr(info[name], "unit_info", None)
                             for name in ("wavelength array", "intensity array") if name in info}
                    result = super()._handle_binary(info, **kwargs)
                    result.update(units)
                    return result

            globals()["_UnitMzML"] = _UnitMzML
        return globals()["_UnitMzML"]


def __getattr__(name):
    # The former module attributes are bound on first access, not at startup.
    if name == "_UnitMzML":
        return _unit_mzml()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class SpectrumReader:
    """Random-access reader. One instance is not shared across threads.

    Plainly written spectra decode directly with pyteomics-identical results; any
    other spectrum uses pyteomics on the calling thread. Spectra planned with
    ``read_ahead`` decode on private helper threads, which ``close`` joins.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._reader = None  # _UnitMzML, opened only if a spectrum needs pyteomics
        self._direct = None
        self._pool: ThreadPoolExecutor | None = None
        self._plan: list[int] = []
        self._planned: dict[int, int] = {}
        self._next = 0
        self._pending: OrderedDict[int, object] = OrderedDict()
        self._frozen = False  # planned spectra are decoded for read_arrays

    def read_ahead(self, indices: Iterable[int]) -> None:
        """Decode these spectra ahead of read() calls made in the same order."""
        self._read_ahead(indices, frozen=False)

    def _read_ahead(self, indices: Iterable[int], frozen: bool) -> None:
        self._cancel_pending()
        if self._open_direct() is None:
            return
        self._frozen = frozen
        self._plan = [int(index) for index in indices]
        self._planned = {index: position for position, index in enumerate(self._plan)}
        self._next = 0
        self._fill()

    def read(self, spectrum_index: int) -> tuple[np.ndarray, np.ndarray]:
        """New writable arrays, except _Frozen MS arrays for read_arrays' loader."""
        index = int(spectrum_index)
        frozen = getattr(_filling, "active", False)
        spectrum = self._take(index)
        # MS arrays a helper thread already prepared, for read_arrays or plain reads.
        if type(spectrum) is _Frozen:
            return spectrum if frozen else _float64_pair(*spectrum, copy=True)
        if type(spectrum) is tuple:
            return _freeze(*spectrum) if frozen else spectrum
        if spectrum is _MISSING:
            direct = self._open_direct()
            spectrum = None if direct is None else direct.decode(index)
        if spectrum is None:
            if self._reader is None:
                self._reader = _unit_mzml()(self.path, cv=_local_vocabulary())
            spectrum = self._reader.get_by_index(index)
        if ('absorption spectrum' in spectrum or 'PDA spectrum' in spectrum) and 'wavelength array' not in spectrum:
            raise ValueError('DAD spectrum has no wavelength array')
        if "wavelength array" in spectrum:
            wavelength = np.asarray(spectrum['wavelength array'], dtype=np.float64)
            values = np.asarray(spectrum.get('intensity array', []), dtype=np.float64)
            xunit = str(spectrum.get('wavelength array unit', '')).lower()
            yunit = str(spectrum.get('intensity array unit', '')).lower()
            if xunit not in {'nanometer', 'nanometre', 'nm', 'uo:0000018'}:
                raise ValueError(f"Unsupported DAD wavelength unit: {xunit}")
            if yunit in {'absorbance unit', 'au', 'uo:0000269'}:
                values = values * 1000.
            elif yunit not in {'mau', 'milliabsorbance unit', 'milliabsorbance units'}:
                raise ValueError(f"Unsupported DAD absorbance unit: {yunit}")
            if (wavelength.ndim != 1 or values.shape != wavelength.shape
                    or not np.isfinite(wavelength).all() or not np.isfinite(values).all()
                    or np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0)):
                raise ValueError("Invalid DAD wavelength/absorbance arrays")
            return wavelength, values
        prepare = _freeze if frozen else _float64_pair
        return prepare(spectrum.get("m/z array", []), spectrum.get("intensity array", []))

    def close(self) -> None:
        self._cancel_pending()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        if self._direct:
            self._direct.close()
        self._direct = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _open_direct(self):
        if self._direct is None:
            try:
                self._direct = open_direct(self.path) or False
            except Exception:
                self._direct = False  # pyteomics reports the error.
        return self._direct or None

    def _take(self, index: int):
        position = self._planned.get(index)
        if position is None:
            return _MISSING
        # Earlier planned spectra were not requested, for example cached meanwhile.
        while self._pending:
            first = next(iter(self._pending))
            if self._planned[first] >= position:
                break
            self._pending.pop(first).cancel()
        self._next = max(self._next, position + 1)
        future = self._pending.pop(index, None)
        self._fill()
        return _MISSING if future is None else future.result()

    def _fill(self) -> None:
        while self._next < len(self._plan) and len(self._pending) < _AHEAD:
            index = self._plan[self._next]
            self._next += 1
            if index not in self._pending:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(_DECODERS, thread_name_prefix="kmzviewer-decode")
                decode = _decode_frozen if self._frozen else _decode_ahead
                self._pending[index] = self._pool.submit(decode, self._direct, index)

    def _cancel_pending(self) -> None:
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        self._plan, self._planned, self._next = [], {}, 0

    def __enter__(self) -> SpectrumReader:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class SpectrumCache:
    """Small LRU used by tests and callers that only need a handful of scans."""

    def __init__(self, limit: int = 32) -> None:
        self.limit = limit
        self._items: OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, path: str, spectrum_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        key = (path, spectrum_index)
        found = self._items.get(key)
        if found is None:
            return None
        self._items.move_to_end(key)
        return found

    def put(self, path: str, spectrum_index: int, arrays: tuple[np.ndarray, np.ndarray]) -> None:
        key = (path, spectrum_index)
        self._items[key] = arrays
        self._items.move_to_end(key)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def drop_path(self, path: str) -> None:
        for key in [key for key in self._items if key[0] == path]:
            self._items.pop(key, None)


class ArrayCache:
    """Decoded m/z and intensity arrays for spectra that have already been read."""

    def __init__(self, budget_bytes: int = DEFAULT_CACHE_BUDGET) -> None:
        self.budget_bytes = int(budget_bytes)
        self._files: OrderedDict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = OrderedDict()
        self._file_bytes: dict[str, int] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def contains(self, path: str, spectrum_index: int) -> bool:
        """Membership only; hit counts and file recency are unchanged."""
        with self._lock:
            spectra = self._files.get(path)
            return spectra is not None and int(spectrum_index) in spectra

    def get(self, path: str, spectrum_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            spectra = self._files.get(path)
            found = None if spectra is None else spectra.get(int(spectrum_index))
            if found is None:
                self.misses += 1
                return None
            self.hits += 1
            self._files.move_to_end(path)
            return found

    def put(self, path: str, spectrum_index: int, arrays: tuple[np.ndarray, np.ndarray]) -> bool:
        """Store read-only copies, so arrays a caller still holds never alias them."""
        return self._store(path, spectrum_index, arrays, adopt=False) is not None

    def _store(self, path: str, spectrum_index: int, arrays, adopt: bool):
        """put(), returning the stored pair or None. With adopt, a _Frozen pair
        that read_arrays just received from SpectrumReader.read is kept as is."""
        adopt = adopt and type(arrays) is _Frozen
        mz = np.asarray(arrays[0], dtype=np.float64)
        intensity = np.asarray(arrays[1], dtype=np.float64)
        nbytes = int(mz.nbytes + intensity.nbytes)
        index = int(spectrum_index)
        with self._lock:
            spectra = self._files.get(path)
            if spectra is not None and index in spectra:
                return spectra[index]
            if nbytes <= 0 or nbytes > self.budget_bytes:
                return None
            self._evict_other_files(path, nbytes)
            if self._bytes + nbytes > self.budget_bytes:
                return None
            if spectra is None:
                spectra = {}
                self._files[path] = spectra
                self._file_bytes[path] = 0
            # Copy only after acceptance. A full cache must not copy and
            # immediately discard every remaining profile scan.
            stored = spectra[index] = tuple(arrays if adopt else _freeze(mz, intensity))
            self._file_bytes[path] += nbytes
            self._bytes += nbytes
            self._files.move_to_end(path)
            return stored

    def drop_path(self, path: str) -> None:
        with self._lock:
            self._drop_unlocked(path)

    def _evict_other_files(self, keep_path: str, incoming: int) -> None:
        while self._bytes + incoming > self.budget_bytes:
            victim = next((path for path in self._files if path != keep_path), None)
            if victim is None:
                return
            self._drop_unlocked(victim)

    def _drop_unlocked(self, path: str) -> None:
        if path not in self._files:
            return
        self._files.pop(path, None)
        self._bytes -= self._file_bytes.pop(path, 0)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes

    def stored_scans(self, path: str) -> int:
        with self._lock:
            spectra = self._files.get(path)
            return 0 if spectra is None else len(spectra)


def read_arrays(
    cache: ArrayCache | None,
    path: str,
    spectrum_index: int,
    read: Callable[[], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached arrays or decode them once and return the stored result.

    SpectrumReader.read calls that ``read`` makes on this thread return a new
    _Frozen pair, which the cache keeps instead of the copy put() makes.
    """
    if cache is None:
        return read()
    found = cache.get(path, spectrum_index)
    if found is not None:
        return found
    outer = getattr(_filling, "active", False)
    _filling.active = True
    try:
        arrays = read()
    finally:
        _filling.active = outer
    stored = cache._store(path, spectrum_index, arrays, adopt=True)
    return arrays if stored is None else stored


def read_ahead(reader, cache: ArrayCache | None, path: str, indices: Iterable[int]) -> None:
    """Plan the uncached spectra a sequential pass will read; substitutes skip it.

    With a cache, helpers decode into the _Frozen arrays read_arrays stores."""
    if isinstance(reader, SpectrumReader):
        reader._read_ahead([int(index) for index in indices
                            if cache is None or not cache.contains(path, int(index))],
                           frozen=cache is not None)


def sum_intensities(intensity: np.ndarray) -> float:
    finite = intensity[np.isfinite(intensity)] if intensity.size else intensity
    if finite.size == 0:
        return 0.0
    return float(np.sum(finite, dtype=np.float64))
