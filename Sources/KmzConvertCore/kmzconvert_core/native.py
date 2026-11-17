"""Optional exact native WIFF codec and the converter's fused spectrum kernels.

``decode_tokens`` and ``scan_metadata`` equal ``binary.decode_tokens`` and
``binary.scan_metadata`` (CoMet EICs share the decoder). For the converter's
``NativeWiffReader`` only, ``reader_arrays``, ``measured_regions`` and
``encode_spectrum`` decode, measure, frame, correct, sample (Sparse Profile v3)
and compress one spectrum in a single GIL-free call over its immutable block. They
equal the interpreted composition (``WiffReader.arrays``, ``MassCorrection.correct``,
``kmzcore.sparse_profile``, ``zlib``/``base64``) and fall back to it without the
extension or wherever the reference must decide. Every other reader, including
the plain ``WiffReader`` that validators use, takes the interpreted path. So does
every reader with a NumPy other than the qualified one, whose semantics the
kernels reproduce; compression additionally requires CPython's zlib.
"""
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import cache
from itertools import batched
import os
import signal
import threading
import zlib

import numpy as np
from kmzcore.qualified import qualified
from kmzcore.sparse_profile import FAMILY, TRIGGER_UNITS, Regions, candidate_regions, rescue_selection as _python_rescue

from .binary import FormatError, decode_tokens as _python_decode, scan_metadata as _python_metadata
from .reader import WiffReader

try:
    from . import _native
except ImportError:
    _native = None

ACCELERATED = _native is not None  # Built; distribution builds require it.
if not qualified("numpy"):
    _native = None  # Every kernel declines: the interpreted reference decides.
ENGAGED = _native is not None  # Built and qualified: the kernels run.


def scan_metadata(data, start, expected_end):
    """``binary.scan_metadata``; the reference parses (and rejects) all but the qualified layout."""
    parsed = None if _native is None else _native.scan_metadata(data, start, expected_end)
    return _python_metadata(data, start, expected_end) if parsed is None else parsed


class NativeWiffReader(WiffReader):
    """The converter's ``WiffReader``: the writer and mass-reference estimation take the
    native and threaded path, and scan preambles parse natively."""
    scan_metadata = staticmethod(scan_metadata)


def applies(reader):
    """Only the converter's reader is accelerated; validators and adapters keep the reference."""
    return isinstance(reader, NativeWiffReader)


def decode_tokens(block):
    """Return coordinates and intensities, retaining ownership through NumPy bases."""
    if _native is None:
        return _python_decode(block)
    # Snapshot mutable buffer inputs before releasing the GIL in the extension.
    block = bytes(block)
    try:
        coordinates, intensities = _native.decode_tokens(block)
    except ValueError as error:
        raise FormatError(str(error)) from error
    return np.frombuffer(coordinates, dtype=np.int64), np.frombuffer(intensities, dtype=np.float64)


def rescue_selection(sequence):
    """``kmzcore.sparse_profile.rescue_selection`` with the span links and union-find in C."""
    if _native is None or not sequence or not all(
            all(type(a) is np.ndarray and a.ndim == 1 and len(a) == len(r.starts) for a in (r.apex, r.lower, r.upper))
            and r.lower.dtype == r.upper.dtype == np.float64 for r in sequence):
        return _python_rescue(sequence)
    sizes = [len(r.starts) for r in sequence]
    # Private immutable copies: the extension never reads caller-owned arrays.
    chosen = _native.rescue_selection(np.array(sizes, dtype=np.int64).tobytes(),
                                      np.concatenate([r.lower for r in sequence]).tobytes(),
                                      np.concatenate([r.upper for r in sequence]).tobytes(),
                                      np.concatenate([r.apex > TRIGGER_UNITS * r.unit for r in sequence]).tobytes())
    if chosen is None:
        return _python_rescue(sequence)
    chosen, offsets = np.frombuffer(chosen, dtype=bool), np.cumsum([0, *sizes])
    return [chosen[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


@cache
def _same_zlib():
    """The extension links CPython's zlib: same runtime version and identical streams on a probe."""
    if _native is None or _native.zlib_runtime() != zlib.ZLIB_RUNTIME_VERSION:
        return False
    probe = np.sqrt(np.arange(1, 4097, dtype=np.float64))
    probe = np.concatenate([probe, np.where(np.arange(4096) % 7, 0., probe[::-1])]).tobytes()
    encoded = _native.deflate_base64(probe)
    return encoded is not None and bytes(encoded[0]) == base64.b64encode(zlib.compress(probe))


def deflate_base64(*arrays):
    """``base64.b64encode(zlib.compress(array))`` for up to four arrays, as bytes-like objects."""
    data = [np.ascontiguousarray(array).tobytes() for array in arrays]
    encoded = _native.deflate_base64(*data) if _native is not None and _same_zlib() else None
    return [base64.b64encode(zlib.compress(item)) for item in data] if encoded is None else encoded


def _block(reader, scan, correction):
    """The fused kernels' arguments for a production reader, else None."""
    denominator = 0.0 if correction is None else correction.denominator
    if (_native is None or not applies(reader) or (correction is not None and not denominator > 0)
            or not all(type(v) is float for v in (scan.cal_a, scan.cal_b, reader.tdc, denominator))):
        return None
    block = reader.block(scan)
    return None if type(block) is not bytes else (block, scan.cal_a, scan.cal_b, reader.tdc, denominator)


def _guarded(reader, scan, kernel, arguments):
    """A fused kernel's tail after the reader's guards and the correction check, or None."""
    try:
        measured = kernel(*arguments)
    except ValueError as error:
        raise FormatError(str(error)) from error
    if measured is None:
        return None
    total, top, first, last, valid, status, tail = measured
    reader.check_measured(scan, total, top, (first, last), valid)
    if status:
        raise ValueError("Invalid original MS1 mass axis" if status == 1 else "Invalid corrected MS1 mass axis")
    return tail


def reader_arrays(reader, scan, correction=None):
    """Framed ``reader.arrays(scan)``, then ``correction.correct`` of its m/z when given.

    Production readers decode, measure, frame and correct natively in one call
    and keep the reader's guards (``check_measured``).
    """
    arguments = _block(reader, scan, correction)
    tail = None if arguments is None else _guarded(reader, scan, _native.profile_spectrum, arguments)
    if tail is not None:
        return tuple(np.frombuffer(part, dtype=np.float64) for part in tail)
    mz, intensity = reader.arrays(scan, decoder=decode_tokens) if applies(reader) else reader.arrays(scan)
    return (mz if correction is None else correction.correct(mz)), intensity


def measured_regions(reader, scan, correction=None):
    """``candidate_regions(*reader_arrays(reader, scan, correction))``."""
    arguments = _block(reader, scan, correction)
    tail = None if arguments is None else _guarded(reader, scan, _native.profile_regions, arguments)
    if tail is None:
        return candidate_regions(*reader_arrays(reader, scan, correction))
    unit, found, parts = tail
    parts = [np.frombuffer(parts, dtype, found, 8*found*k) for k, dtype in enumerate((np.int64,) * 2 + (np.float64,) * 3)]
    return Regions(unit, *parts)


def encode_spectrum(reader, scan, correction=None, selection=None, sparse=False):
    """(written points, base64 zlib m/z, base64 zlib intensity, Sparse Profile check or None)
    of a production reader's spectrum: ``deflate_base64`` of the arrays ``reader_arrays``
    gives, after the ``source_sampling_check`` subset for Sparse Profile v3 (``selection``
    is the run-level choice). None where the reference path must decide."""
    if not _same_zlib() or (sparse and (getattr(reader, "sparse_profile_sampling_family", None) != FAMILY
                                        or type(selection) is not np.ndarray or selection.dtype != bool or selection.ndim != 1)):
        return None
    arguments = _block(reader, scan, correction)
    tail = None if arguments is None else _guarded(
        reader, scan, _native.encode_spectrum, (*arguments, selection.tobytes() if sparse else None))
    if tail is None:
        return None
    source, points, mz, intensity, unit, selected, regions, rescued = tail
    check = None if not sparse else dict(
        sampling_family=FAMILY, source_points=source, event_unit=unit, retained_points=points,
        selected_points=selected, selected_regions=regions, rescued_regions=rescued,
        added_guard_points=points - selected, source_signal_breaks=0)
    return points, mz, intensity, check


def workers(reader):
    """Production readers convert spectra on worker threads (native/zlib work releases the GIL)."""
    return min(16, os.process_cpu_count() or 1) if applies(reader) else 0


class _Interrupts:
    """Main-thread SIGINT state while converter worker threads may be live.

    ThreadPoolExecutor registers a worker only after starting it, and an
    interrupted shutdown join would leave workers running after the reader
    closes. Inside ``deferred_interrupts()`` SIGINT is only recorded; it is raised
    as KeyboardInterrupt at an ``interruption_point()`` (where cleanup follows) or
    re-delivered when the outermost section ends.
    """

    def __init__(self):
        self.depth, self.handler, self.received = 0, None, False

    def record(self, *_):
        self.received = True


_INTERRUPTS = _Interrupts()


@contextmanager
def deferred_interrupts():
    """Record SIGINT instead of raising it anywhere while worker threads start, run or join."""
    state = _INTERRUPTS
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if not state.depth:
        handler = signal.getsignal(signal.SIGINT)
        if handler is None or handler is signal.SIG_IGN:  # Not installed from Python, or ignored.
            yield
            return
        state.handler, state.received = handler, False
        signal.signal(signal.SIGINT, state.record)
    state.depth += 1
    try:
        yield
    finally:
        state.depth -= 1
        if not state.depth:
            signal.signal(signal.SIGINT, state.handler)
            if state.received:
                state.received = False
                signal.raise_signal(signal.SIGINT)


def interruption_point():
    """Raise a SIGINT recorded under the default handler as KeyboardInterrupt."""
    state = _INTERRUPTS
    if (state.received and state.handler is signal.default_int_handler
            and threading.current_thread() is threading.main_thread()):
        state.received = False
        raise KeyboardInterrupt


def _result(future):
    # Poll with wait(), not result(timeout=...): a worker's own TimeoutError must propagate.
    while not wait((future,), timeout=.05).done:
        interruption_point()
    return future.result()


def ordered(function, items, workers, chunk=1):
    """Yield function(item) in input order. Each task takes up to ``chunk`` items, at most
    2*workers tasks are in flight, and close() cancels pending tasks and joins all workers.
    SIGINT is deferred from before the first worker starts until the last is joined."""
    if not workers:
        yield from map(function, items)
        return
    with deferred_interrupts(), ThreadPoolExecutor(workers, thread_name_prefix="kmzconvert") as pool:
        pending = deque()
        try:
            for batch in batched(items, chunk):
                interruption_point()
                pending.append(pool.submit(_each, function, batch))
                if len(pending) >= 2 * workers:
                    yield from _result(pending.popleft())
            while pending:
                yield from _result(pending.popleft())
        finally:
            for future in pending:
                future.cancel()


def _each(function, batch):
    return [function(item) for item in batch]
