"""Stream native arrays through Pyteomics using the suite's pinned PSI-MS vocabulary."""
from collections import deque
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
import threading

import numpy as np

from kmzcore import psi_ms
from kmzcore.mzml import require_current_input
from kmzcore.qualified import qualified
from kmzcore.sparse_profile import read_provenance
from .model import Scan
from .sources import ordered_finite

# pyteomics and psims take ~0.2 s to import, so they load on the first mzML read
# rather than when the CLI or desktop window starts. UnitMzML is bound on first
# use and read back from the module, so replacing it still changes the reader used;
# deferred decoding is composed over whichever class it is on each read.
_unit_lock = threading.Lock()
_deferred = {}


def vocabulary():
    # The suite's pinned PSI-MS release, read offline. Pyteomics looks up every valued
    # MS cvParam, so this release decides which accessions an mzML may carry.
    return psi_ms.vocabulary()


def _unit_mzml():
    """Return the module's UnitMzML, defined once even for simultaneous first reads."""
    with _unit_lock:
        if "UnitMzML" not in globals():
            from pyteomics import mzml

            class UnitMzML(mzml.MzML):
                def _handle_binary(self, info, **kwargs):
                    unit = str(getattr(info.get("intensity array"), "unit_info", "") or "unknown")
                    result = super()._handle_binary(info, **kwargs)
                    result["intensity unit"] = unit
                    return result

            globals()["UnitMzML"] = UnitMzML
        return globals()["UnitMzML"]


def __getattr__(name):
    # The former module attribute is bound on first access, not at startup.
    if name == "UnitMzML":
        return _unit_mzml()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _Decoding:
    """One binary array that pyteomics decodes on a helper thread."""
    __slots__ = ("future", "key")

    def __init__(self, future):
        self.future, self.key = future, None


class _DeferredDecoding:
    """Mixin for a ``UnitMzML`` whose base64/zlib decoding of larger arrays (pyteomics'
    own ``decode_data_array``, a pure function of the element's text) runs on ``pool``.
    Parsed spectra hold placeholders, listed in ``decoding`` in parse order, until
    ``finish`` converts each decoded result exactly as an immediate decode would have."""

    def __init__(self, *args, pool, **kwargs):
        self._pool, self.decoding = pool, []
        super().__init__(*args, **kwargs)

    def decode_data_array(self, source, compression_type=None, dtype=np.float64):
        if len(source) < DEFER_CHARS:  # Not worth a thread hand-off: decode now.
            return super().decode_data_array(source, compression_type, dtype)
        array = _Decoding(self._pool.submit(super().decode_data_array, source, compression_type, dtype))
        self.decoding.append(array)
        return array

    def _convert_array(self, k, array):
        if type(array) is _Decoding:
            array.key = k
            return array
        return super()._convert_array(k, array)

    def finish(self, array, data):
        return super()._convert_array(array.key, data)


def _deferred_mzml(unit):
    """``unit`` with deferred decoding, composed once per reader class."""
    with _unit_lock:
        if unit not in _deferred:
            _deferred[unit] = type("DeferredMzML", (_DeferredDecoding, unit), {})
        return _deferred[unit]


# Array-decoding threads (zlib releases the GIL), spectra parsed ahead of use and
# the smallest encoded array decoded on a thread.
DECODERS, AHEAD, DEFER_CHARS = 2, 3, 4096


def spectra(path):
    """pyteomics spectrum dicts in document order, as ``UnitMzML`` yields them, with
    arrays decoded on helper threads up to ``AHEAD`` spectra ahead. Before a
    spectrum is used, or a parse error raised in its place, the arrays parsed with
    it are decoded in order, so every error surfaces in the original order.

    As in the converter's ``ordered``, SIGINT is only recorded from before the first
    decoder starts until the last is joined, and raised as KeyboardInterrupt at this
    generator's interruption points; closing drops queued decodes and joins the rest.
    The vocabulary loads first, so a slow download stays directly interruptible."""
    unit, cv = _unit_mzml(), vocabulary()
    # A replacement reader factory is used as-is. Deferral hooks pyteomics' internals,
    # so other pyteomics versions decode while parsing, as before.
    if not isinstance(unit, type) or not qualified("pyteomics"):
        with unit(str(path), use_index=False, iterative=True, cv=cv) as reader:
            yield from reader
        return
    from kmzconvert_core.native import deferred_interrupts, interruption_point

    def result(future):
        # Poll with wait(), not result(timeout=...): a decoder's own TimeoutError must propagate.
        while not future.done():
            wait((future,), timeout=.05)
            interruption_point()
        return future.result()

    with deferred_interrupts():
        pool = ThreadPoolExecutor(DECODERS, thread_name_prefix="kmzcomet-mzml")
        try:
            with _deferred_mzml(unit)(str(path), use_index=False, iterative=True, cv=cv, pool=pool) as reader:
                parsed, queue, end = iter(reader), deque(), None
                while True:
                    interruption_point()
                    while end is None and len(queue) < AHEAD:
                        try:
                            spectrum = next(parsed)
                        except StopIteration as stop:
                            spectrum, end = None, stop
                        except Exception as error:
                            spectrum, end = None, error
                        arrays, reader.decoding = reader.decoding, []
                        queue.append((spectrum, arrays))
                    spectrum, arrays = queue.popleft()
                    decoded = {id(array): reader.finish(array, result(array.future)) for array in arrays}
                    if spectrum is None:
                        if isinstance(end, StopIteration):
                            return
                        raise end
                    for key, value in spectrum.items():
                        if type(value) is _Decoding:
                            spectrum[key] = decoded[id(value)]
                    yield spectrum
        finally:
            pool.shutdown(cancel_futures=True)


def iter_scans(path: Path):
    require_current_input(path)
    if read_provenance(path).methods:
        raise ValueError("Sparse Profile input is not qualified for KmzCoMet; use Full Profile")
    previous = {}
    with closing(spectra(path)) as reader:
        for spectrum in reader:
            if spectrum.get("ms level") != 1:
                continue
            ident = spectrum.get("id", "unknown scan")
            scans = spectrum.get("scanList", {}).get("scan", [])
            if len(scans) != 1:
                raise ValueError(f"{ident}: require exactly one MS1 scan/time entry")
            scan = scans[0]
            rt = scan.get("scan start time")
            unit = str(getattr(rt, "unit_info", "")).lower()
            if unit in {"minute", "minutes", "uo:0000031"}:
                rt = float(rt)
            elif unit in {"second", "seconds", "uo:0000010"}:
                rt = float(rt) / 60
            else:
                raise ValueError(f"{ident}: missing/unsupported RT unit {unit!r}")
            if not np.isfinite(rt) or rt < 0:
                raise ValueError(f"{ident}: invalid RT")
            positive, negative = "positive scan" in spectrum, "negative scan" in spectrum
            if positive and negative:
                raise ValueError(f"{ident}: conflicting scan polarities")
            polarity = int(positive) - int(negative)
            if rt <= previous.get(polarity, -1):
                raise ValueError(f"{ident}: MS1 times must strictly increase within each polarity")
            previous[polarity] = rt
            profile, centroid = "profile spectrum" in spectrum, "centroid spectrum" in spectrum
            if profile == centroid:
                raise ValueError(f"{ident}: require explicit profile or centroid spectrum metadata")
            if "m/z array" not in spectrum or "intensity array" not in spectrum:
                raise ValueError(f"{ident}: missing MS1 arrays")
            m = np.asarray(spectrum["m/z array"], dtype=float)
            y = np.asarray(spectrum["intensity array"], dtype=float)
            if not ordered_finite(m, y):
                raise ValueError(f"{ident}: require ordered finite masses and nonnegative intensities")
            if int(spectrum.get("defaultArrayLength", len(m))) != len(m):
                raise ValueError(f"{ident}: array length disagrees with metadata")
            windows = []
            for window in scan.get("scanWindowList", {}).get("scanWindow", []):
                lo = float(window.get("scan window lower limit", float("nan")))
                hi = float(window.get("scan window upper limit", float("nan")))
                if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
                    raise ValueError(f"{ident}: invalid acquisition mass window")
                windows.append((lo, hi))
            yield Scan(rt, polarity, "profile" if profile else "centroid", m, y,
                       tuple(windows), spectrum.get("intensity unit", "unknown"))
