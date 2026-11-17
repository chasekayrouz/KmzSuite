import struct

import numpy as np
import pytest

from kmzconvert_core.binary import FormatError, calibrate, decode_tokens, frame_profile, protobuf_fields, varint


def block(tokens, seed=40):
    return b"\xff"*4 + struct.pack("<I", seed) + b"\0" + tokens + b"\xff"*4


def test_delta_and_intensity_escape_widths():
    # Hand-built independent record, including the X500 four-byte forms.
    payload = (b"\x05" + b"\x81\x7c\xfa" + b"\xfc\xff\x7d\x2c\x01"
               + b"\xfd\xff\x01\x0a" + b"\xfe" + struct.pack("<I", 70000)
               + b"\x7e" + struct.pack("<I", 100000))
    coordinates, intensity = decode_tokens(block(payload))
    assert coordinates.tolist() == [44, 52, 1076, 3124, 283128]
    assert intensity.tolist() == [5, 250, 300, 10, 100000]


@pytest.mark.parametrize("payload", [b"\xfc", b"\xfd\x01", b"\xfe\x01\x02", b"\x7e\x01\x02", b"\x7f"])
def test_truncation_and_unknown_prefix_fail_closed(payload):
    raw = b"\xff"*4 + struct.pack("<I", 40) + b"\0" + payload
    with pytest.raises(FormatError):
        decode_tokens(raw)


def test_terminator_required_and_empty_not_fabricated():
    with pytest.raises(FormatError):
        decode_tokens(block(b""))
    with pytest.raises(FormatError):
        decode_tokens(block(b"\x0a")[:-4])


def test_sparse_profile_framing_preserves_signal_and_merges_shared_zeros():
    # Adjacent peaks, one shared zero between peaks, and a larger gap.
    x = np.array([40, 44, 52, 100])
    y = np.array([10., 20., 30., 40.])
    mz, signal = frame_profile(x, y, .1, 2., .25)
    expected_grid = np.array([36, 40, 44, 48, 52, 56, 96, 100, 104])
    np.testing.assert_array_equal(signal, [0, 10, 20, 0, 30, 0, 0, 40, 0])
    np.testing.assert_allclose(mz, [.49, .64, .81, 1, 1.21, 1.44, 4.84, 5.29, 5.76], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(mz[signal != 0], calibrate(x, .1, 2., .25))
    assert signal.sum() == y.sum() and len(mz) == len(expected_grid)


def test_bounded_protobuf_wire_types():
    assert varint(b"\xa2\x04", 0) == (546, 2)
    assert protobuf_fields(b"\x08\x96\x01\x12\x02ab") == [(1, 0, 150), (2, 2, b"ab")]
    for bad in [b"\x00", b"\x12\x08ab", b"\x09\0\0", b"\x08"+b"\x80"*11]:
        with pytest.raises(FormatError):
            protobuf_fields(bad)
