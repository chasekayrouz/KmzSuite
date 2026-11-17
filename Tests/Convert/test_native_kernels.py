"""Exact parity of the shared native codec, preamble parser and rescue selection with their
interpreted references, and the extension's input contract (synthetic inputs)."""
import base64
import random
import struct
import sys
import zlib

import numpy as np
import pytest
from kmzconvert_core import _native, binary, native
from kmzcore import sparse_profile as sp


def outcome(function, *args):
    try:
        return 'ok', function(*args)
    except (ValueError, OverflowError, IndexError, TypeError) as error:
        return type(error), str(error)


def same(actual, expected):
    if isinstance(expected, np.ndarray):
        return (isinstance(actual, np.ndarray) and actual.dtype == expected.dtype and actual.shape == expected.shape
                and actual.tobytes() == expected.tobytes())
    if isinstance(expected, (tuple, list)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(map(same, actual, expected))
    if type(expected) is float:  # Bit equality: signed zeros differ, NaNs match.
        return type(actual) is float and (struct.pack('<d', actual) == struct.pack('<d', expected)
                                          or (actual != actual and expected != expected))
    return type(actual) is type(expected) and actual == expected


def parity(reference, accelerated, *args):
    """Equal values, dtypes and bytes, or the same exception type and message."""
    expected, actual = outcome(reference, *args), outcome(accelerated, *args)
    assert actual[0] == expected[0] and same(actual[1], expected[1]), (args, expected, actual)
    return expected


def test_accelerated():
    assert native.ACCELERATED and native.ENGAGED


def block(body, seed=0):
    return b'\xff' * 4 + struct.pack('<I', seed) + b'\0' + body + b'\xff'


def test_decoded_blocks_round_trip():
    rng = random.Random(17)
    for _ in range(500):
        # Signal tokens with short TOF deltas; a trailing delta prefix gets its intensity.
        body = bytes(rng.choice([1, 2, 7, 64, 123]) if rng.random() < .7 else rng.choice([129, 130, 200, 251])
                     for _ in range(rng.randrange(1, 120)))
        body += b'\x01' if body[-1] >= 128 else b''
        parity(binary.decode_tokens, native.decode_tokens, block(body, rng.choice([0, 4, 4096])))


def test_import_fallback(monkeypatch):
    import importlib.util
    import kmzconvert_core
    monkeypatch.setitem(sys.modules, 'kmzconvert_core._native', None)
    monkeypatch.delattr(kmzconvert_core, '_native')
    spec = importlib.util.spec_from_file_location('kmzconvert_core._fallback_test', native.__file__)
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    assert not fallback.ACCELERATED and not fallback._same_zlib()
    data = block(b'\x01\x02')
    assert same(fallback.decode_tokens(data), binary.decode_tokens(data))
    sequence = [spans(np.random.default_rng(2), 5, np.arange(0., 60.))]
    assert same(fallback.rescue_selection(sequence), sp.rescue_selection(sequence))
    array = np.arange(9.)
    assert [bytes(t) for t in fallback.deflate_base64(array)] == [base64.b64encode(zlib.compress(array.tobytes()))]



def test_unqualified_numpy_takes_the_interpreted_reference(unqualified):
    import importlib.util
    unqualified('numpy')
    spec = importlib.util.spec_from_file_location('kmzconvert_core._unqualified_test', native.__file__)
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    # Still built (distribution builds require it), but no kernel runs.
    assert fallback.ACCELERATED and not fallback.ENGAGED and fallback._native is None and not fallback._same_zlib()
    data = block(b'\x01\x02')
    assert same(fallback.decode_tokens(data), binary.decode_tokens(data))
    sequence = [spans(np.random.default_rng(2), 5, np.arange(0., 60.))]
    assert same(fallback.rescue_selection(sequence), sp.rescue_selection(sequence))
    array = np.arange(9.)
    assert [bytes(t) for t in fallback.deflate_base64(array)] == [base64.b64encode(zlib.compress(array.tobytes()))]
    assert fallback.encode_spectrum(None, None) is None

def test_gil_free_kernels_take_only_immutable_bytes():
    """Kernels that release the GIL never read caller-mutable memory: other buffers are refused."""
    data = block(b'\x01\x02\x03')
    for mutable in (bytearray(data), memoryview(bytearray(data)), np.frombuffer(bytearray(data), np.uint8)):
        for call in (lambda: _native.decode_tokens(mutable),
                     lambda: _native.profile_spectrum(mutable, .5, .1, .2, 0.),
                     lambda: _native.profile_regions(mutable, .5, .1, .2, 0.),
                     lambda: _native.encode_spectrum(mutable, .5, .1, .2, 0., None),
                     lambda: _native.encode_spectrum(data, .5, .1, .2, 0., bytearray(1)),
                     lambda: _native.deflate_base64(data, mutable),
                     lambda: _native.rescue_selection(mutable, bytes(8), bytes(8), b'\x01')):
            with pytest.raises(TypeError):
                call()
    with pytest.raises(ValueError):
        _native.rescue_selection(bytes(8), bytes(8), bytes(16), b'\x01')


def unaligned(values, dtype):
    array = np.ndarray((len(values),), dtype=dtype, buffer=bytearray(8 * len(values) + 1), offset=1)
    array[:] = values
    assert not array.flags.aligned and array.flags.c_contiguous
    return array


def test_unaligned_strided_and_byteswapped_inputs_match():
    rng = np.random.default_rng(4)
    sequence = [spans(rng, int(rng.integers(1, 8)), np.arange(0., 60.)) for _ in range(12)]
    expected = sp.rescue_selection(sequence)
    for transform in (lambda a: unaligned(a, a.dtype), lambda a: np.repeat(a, 2)[::2], lambda a: a.astype('>f8')):
        changed = [r._replace(lower=transform(r.lower), upper=transform(r.upper), apex=transform(r.apex)) for r in sequence]
        assert same(native.rescue_selection(changed), expected)
    values = np.sqrt(np.arange(1., 300.))
    for array in (unaligned(values, 'f8'), values[::3], values.astype('>f8'), values[:299].reshape(13, 23)):
        assert [bytes(t) for t in native.deflate_base64(array)] == [base64.b64encode(zlib.compress(array.tobytes()))]


def varint(value):
    data = bytearray()
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    return bytes(data) + bytes([value])


def pair(x, y, tags=(b'\x08', b'\x10')):
    content = tags[0] + varint(x) + tags[1] + varint(y)
    return b'\x12' + varint(len(content)) + content


def preamble(rng, checkpoints=3, extra=b''):
    cal = b'\x09' + struct.pack('<d', rng.random()) + b'\x11' + struct.pack('<d', rng.random() - .5)
    body = b'\x0a' + varint(len(cal)) + cal
    body += b''.join(pair(int(rng.integers(0, 2**63)), int(rng.integers(0, 2**20))) for _ in range(checkpoints)) + extra
    return varint(len(body)) + body


def test_scan_metadata_layout_mutations_and_fallback():
    from kmzconvert_core import _native
    rng = np.random.default_rng(11)
    qualified = [preamble(rng, n) for n in (0, 1, 3, 70)] + [preamble(rng, 1, pair(2**64 - 1, 0))]
    for data in qualified:
        assert _native.scan_metadata(data, 0, len(data)) is not None
    # Non-canonical varint tags/lengths are valid for the reference: the fast path may decline them.
    examples = qualified + [b'\x16' + b'\x8a\x00\x92\x00' + b'\x09' + bytes(8) + b'\x11' + bytes(8),
                            preamble(rng, 1, b'\x12\x0d\x08' + b'\xff' * 9 + b'\x02\x10\x00'),  # 2**64
                            preamble(rng, 1, b'\x1a\x00'), preamble(rng, 1, pair(1, 2, (b'\x10', b'\x08'))),
                            preamble(rng, 0, b'\x12\x02\x08\x01'), preamble(rng, 1, pair(1, 2) + b'\x0d\x00\x00\x00\x00')]
    pyrng = random.Random(23)
    for data in list(examples):
        for _ in range(300):
            mutated = bytearray(data)
            for _ in range(pyrng.randrange(1, 4)):
                mutated[pyrng.randrange(len(mutated))] = pyrng.choice([0, 8, 9, 10, 16, 17, 18, 127, 128, 255, pyrng.randrange(256)])
            examples.append(bytes(mutated))
        examples += [data[:end] for end in range(len(data))]
    for data in examples:
        padded = b'\x00' * 5 + data + b'\x01'
        parity(binary.scan_metadata, native.scan_metadata, padded, 5, 5 + len(data))
        parity(binary.scan_metadata, native.scan_metadata, data, 0, len(data))
    parity(binary.scan_metadata, native.scan_metadata, examples[0], 3, 2)
    parity(binary.scan_metadata, native.scan_metadata, examples[0], 0, len(examples[0]) + 4)


def spans(rng, count, grid):
    """Ordered disjoint candidate spans on a coarse grid, so equal (touching) endpoints occur."""
    edges = np.sort(rng.choice(grid, 2 * count, replace=False)).astype(float)
    lower, upper = edges[0::2], edges[1::2]
    unit = float(rng.choice([1., 10., 7.5]))
    apex = rng.choice([9., 15., 21., 40.], count) * unit
    starts = np.arange(count, dtype=np.int64) * 4
    return sp.Regions(unit, starts, starts + 3, apex, lower, upper)


@pytest.mark.parametrize('seed', range(40))
def test_rescue_selection(seed):
    rng = np.random.default_rng(seed)
    grid = np.arange(0., 60.)
    sequence = [spans(rng, int(rng.integers(0, 12)), grid) for _ in range(int(rng.integers(1, 30)))]
    # Shift neighbors by whole grid steps so closed spans touch, overlap or miss.
    sequence = [r._replace(lower=r.lower + k, upper=r.upper + k)
                for r, k in zip(sequence, rng.integers(-1, 2, len(sequence)))]
    parity(sp.rescue_selection, native.rescue_selection, sequence)
    if any(len(r.starts) > 1 for r in sequence):
        bad = [r._replace(lower=r.lower[::-1].copy()) if len(r.starts) > 1 else r for r in sequence]
        parity(sp.rescue_selection, native.rescue_selection, bad)
        bad = [r._replace(upper=np.r_[r.upper[:-1], np.nan]) if len(r.starts) else r for r in sequence]
        parity(sp.rescue_selection, native.rescue_selection, bad)
    parity(sp.rescue_selection, native.rescue_selection, [r._replace(apex=r.apex.astype(np.int64)) for r in sequence])
    parity(sp.rescue_selection, native.rescue_selection, [])


MUTATION = '''
import threading
import time
import numpy as np
from kmzconvert_core import native
from kmzcore.sparse_profile import Regions
n = 5_000
lower = np.arange(n, dtype=float); upper = lower + .5; apex = np.full(n, 100.); starts = np.arange(n, dtype=np.int64)
half = n // 2
sequence = [Regions(1., starts[:half], starts[:half] + 3, apex[:half], lower[:half], upper[:half]),
            Regions(1., starts[half:], starts[half:] + 3, apex[half:], lower[half:], upper[half:])]
block = bytearray(b'\\xff' * 4 + bytes(5) + b'\\x01' * n + b'\\xff')
stop = threading.Event()
def mutate():
    while not stop.is_set():
        lower[:] = -lower        # Large NumPy writes run without the GIL.
        upper[::-1] = upper
        block[9:-1] = bytes([block[9] ^ 0x7e]) * n
threading.Thread(target=mutate).start()
try:
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        native.rescue_selection(sequence)
        native.deflate_base64(lower, upper)
        try:
            native.decode_tokens(block)
        except ValueError:
            pass
finally:
    stop.set()
'''


def test_concurrent_mutation_of_inputs_is_memory_safe():
    """The review's race (mutable indices changed after validation, with the GIL released)
    cannot recur: inputs are snapshotted to immutable or private memory first. Crash-isolated."""
    import subprocess
    process = subprocess.run([sys.executable, '-c', MUTATION], capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr[-2000:]
