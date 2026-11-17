"""Small, bounded primitives for the observed X500 scan-block encoding."""
import struct

import numpy as np


class FormatError(ValueError):
    """The input does not satisfy the supported on-disk layout."""


def varint(data, pos, end=None):
    end = len(data) if end is None else end
    result = 0
    for shift in range(0, 70, 7):
        if pos >= end:
            raise FormatError("Truncated varint")
        b = data[pos]
        pos += 1
        result |= (b & 127) << shift
        if b < 128:
            if result >= 1 << 64:
                raise FormatError("Oversized varint")
            return result, pos
    raise FormatError("Oversized varint")


def protobuf_fields(data):
    """Decode protobuf wire types, without assuming a vendor message schema."""
    result = []
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise FormatError("Protobuf field zero")
        if wire == 0:
            val, pos = varint(data, pos)
        elif wire in (1, 5):
            n = 8 if wire == 1 else 4
            if pos + n > len(data):
                raise FormatError("Truncated fixed protobuf field")
            val = data[pos:pos+n]
            pos += n
        elif wire == 2:
            n, pos = varint(data, pos)
            if pos + n > len(data):
                raise FormatError("Truncated protobuf bytes")
            val = data[pos:pos+n]
            pos += n
        else:
            raise FormatError(f"Unsupported protobuf wire type {wire}")
        result.append((number, wire, val))
    return result


def scan_metadata(data, start, expected_end):
    size, body = varint(data, start, expected_end)
    if body + size != expected_end:
        raise FormatError("Scan metadata length does not reach indexed block")
    fields = protobuf_fields(data[body:expected_end])
    if not fields or fields[0][:2] != (1, 2):
        raise FormatError("Missing scan calibration protobuf")
    cal = protobuf_fields(fields[0][2])
    if [c[:2] for c in cal] != [(1, 1), (2, 1)]:
        raise FormatError("Unrecognized scan calibration fields")
    a, b = (struct.unpack("<d", c[2])[0] for c in cal)
    checkpoints = []
    for number, wire, val in fields[1:]:
        if (number, wire) != (2, 2):
            raise FormatError("Unexpected scan metadata field")
        pair = protobuf_fields(val)
        if [c[:2] for c in pair] != [(1, 0), (2, 0)]:
            raise FormatError("Unexpected checkpoint fields")
        checkpoints.append((pair[0][2], pair[1][2]))
    return a, b, checkpoints


def decode_tokens(block):
    """Return raw TOF coordinates and integer intensities, with strict bounds.

    The prefix grammar was informed by sciexwiff's MIT-licensed work (see
    THIRD_PARTY.md). X500's coordinate scaling, seed, calibration and index
    framing were independently established against the local oracle.
    """
    if len(block) < 11 or block[:4] != b"\xff" * 4 or block[8] != 0:
        raise FormatError("Unsupported X500 spectrum block header")
    coordinate = struct.unpack_from("<I", block, 4)[0]
    if coordinate % 4:
        raise FormatError("Unaligned TOF seed")
    pos, end = 9, len(block)
    coordinates, intensities = [], []
    terminated = False
    while pos < end:
        # Vendor aligns the body with a final 1..4 FF bytes, not peak tokens.
        if end - pos <= 4 and block[pos:] == b"\xff" * (end - pos):
            terminated = True
            break
        c = block[pos]
        pos += 1
        delta = 1
        if c >= 128:
            if c in (252, 253, 254):
                n = {252: 1, 253: 2, 254: 4}[c]
                if pos + n >= end:
                    raise FormatError("Truncated TOF delta")
                delta = int.from_bytes(block[pos:pos+n], "little") + 1
                pos += n
            else:
                if c == 255:
                    raise FormatError("Unexpected 0xff delta outside terminator")
                delta = c - 127
            if pos >= end:
                raise FormatError("Missing intensity")
            c = block[pos]
            pos += 1
        if c in (124, 125, 126):
            n = {124: 1, 125: 2, 126: 4}[c]
            if pos + n > end:
                raise FormatError("Truncated intensity")
            c = int.from_bytes(block[pos:pos+n], "little")
            pos += n
        elif c == 127:
            raise FormatError("Unsupported intensity prefix 0x7f")
        coordinate += delta * 4
        coordinates.append(coordinate)
        intensities.append(c)
    if not terminated or not coordinates:
        raise FormatError("Missing spectrum terminator or empty spectrum")
    return np.asarray(coordinates, dtype=np.int64), np.asarray(intensities, dtype=np.float64)


def calibrate(coordinates, a, b, tdc):
    return np.square(a * (coordinates * tdc - b))


def measure_spectrum(coordinates, intensity, a, b, tdc):
    """Reader guard inputs: intensity sum and maximum, calibrated m/z, measured validity."""
    if not len(intensity):
        # Keep the reader's guard order: an empty spectrum fails the TIC/BPI checks, not an index.
        return float(intensity.sum()), float("nan"), calibrate(coordinates, a, b, tdc), False
    mz = calibrate(coordinates, a, b, tdc)
    valid = bool(np.isfinite(mz).all() and np.all(np.diff(mz) > 0) and mz[0] > 0 and np.all(intensity > 0))
    return float(intensity.sum()), float(intensity.max()), mz, valid


def frame_profile(coordinates, intensity, a, b, tdc):
    """Add neighboring zero bins around sparse profile signal islands.

    Use the raw acquisition grid, calibrated identically for signal and zero
    bins. Vendor-generated framing zeros sometimes use linear extrapolation
    instead (a ~5e-9 m/z second difference here). Validation reports that
    residual separately; measured signal coordinates are not altered.
    """
    if np.any(intensity <= 0):
        raise FormatError("Stored zero/negative signal is outside the validated codec")
    grid = np.unique(np.concatenate((coordinates - 4, coordinates, coordinates + 4)))
    mz = calibrate(grid, a, b, tdc)
    y = np.zeros(len(grid), dtype=np.float64)
    at = np.searchsorted(grid, coordinates)
    y[at] = intensity
    return mz, y
