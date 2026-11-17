"""Exact codec parity using synthetic inputs only."""
import gc
import random
import re
import struct
import sys
import tracemalloc

import numpy as np
import pytest
from kmzconvert_core.binary import decode_tokens as reference
from kmzcomet import native


def block(body, seed=0, padding=1):
    return b'\xff'*4 + struct.pack('<I', seed) + b'\0' + body + b'\xff'*padding


def parity(data):
    try:
        expected = reference(data)
    except (ValueError, OverflowError) as error:
        with pytest.raises(type(error), match='^' + re.escape(str(error)) + '$'):
            native.decode_tokens(data)
    else:
        actual = native.decode_tokens(data)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype


def test_accelerated():
    assert native.ACCELERATED


@pytest.mark.parametrize('seed', [0, 4, 0xfffffffc])
@pytest.mark.parametrize('padding', [1, 2, 3, 4])
def test_tokens(seed, padding):
    bodies = [bytes([i]) for i in range(128)]
    bodies += [bytes([i, 1]) for i in range(128, 252)]
    for prefix, n in [(252, 1), (253, 2), (254, 4)]:
        bodies += [bytes([prefix]) + value.to_bytes(n, 'little') + b'\x01'
                   for value in (0, 2**(8*n)-1)]
    for prefix, n in [(124, 1), (125, 2), (126, 4)]:
        bodies += [bytes([prefix]) + value.to_bytes(n, 'little')
                   for value in (0, 2**(8*n)-1)]
    for body in bodies:
        parity(block(body, seed, padding))


def test_truncations_and_random():
    examples = [block(b'\xfe\xff\xff\xff\xff\x7e\xff\xff\xff\xff'),
                block(b'\x01'), block(b'\xff'*8), block(b'\x01', seed=3)]
    for data in examples:
        for end in range(len(data)+1):
            parity(data[:end])
    rng = random.Random(491)
    for _ in range(10000):
        body = rng.randbytes(rng.randrange(80))
        parity(block(body, rng.choice([0, 4, 0xfffffffc])))
        parity(body)


def test_lifetime_and_mutable_input():
    source = bytearray(block(b'\x01\x02'))
    x, y = native.decode_tokens(source)
    source[:] = b'\0'*len(source)
    gc.collect()
    np.testing.assert_array_equal(x, [4, 8])
    np.testing.assert_array_equal(y, [1, 2])
    assert isinstance(x.base, bytes) and isinstance(y.base, bytes)
    for array in (x, y):
        with pytest.raises(ValueError):
            array[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_import_fallback(monkeypatch):
    import importlib.util
    import kmzconvert_core
    from kmzconvert_core import native as shared
    # CoMet re-exports the converter's shared codec.
    assert native.decode_tokens is shared.decode_tokens and native.ACCELERATED is shared.ACCELERATED
    assert native.ENGAGED is shared.ENGAGED is True  # Built and qualified here.
    monkeypatch.setitem(sys.modules, 'kmzconvert_core._native', None)
    monkeypatch.delattr(kmzconvert_core, '_native')
    spec = importlib.util.spec_from_file_location('kmzconvert_core._fallback_test', shared.__file__)
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    assert not fallback.ACCELERATED and not fallback.ENGAGED
    for actual, expected in zip(fallback.decode_tokens(block(b'\x01')), reference(block(b'\x01'))):
        np.testing.assert_array_equal(actual, expected)


def test_error_and_success_buffers_released():
    tracemalloc.start()
    try:
        gc.collect()
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(2000):
            native.decode_tokens(block(b'\x01'*100))
            try:
                native.decode_tokens(block(b'\x7f'*100))
            except ValueError:
                pass
        gc.collect()
        assert tracemalloc.get_traced_memory()[0] - before < 65536
    finally:
        tracemalloc.stop()
