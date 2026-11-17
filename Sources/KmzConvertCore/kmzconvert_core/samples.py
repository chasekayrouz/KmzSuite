"""Bounded sample discovery and segment framing for qualified WIFF sidecars."""
from dataclasses import dataclass
from pathlib import Path
import re
import struct

import olefile

from .binary import FormatError

STREAM_HEADER = bytes.fromhex('0000000004000000') + bytes(24)
SAMPLE_INDEX = struct.Struct('<IQI')


def require(condition, message):
    if not condition:
        raise FormatError(message)


@dataclass(frozen=True)
class SampleInfo:
    id: int
    segment_offset: int


def sample_index(data, names):
    require(len(data) > 32 and data[:32] == STREAM_HEADER
            and (len(data)-32) % SAMPLE_INDEX.size == 0, 'Malformed WIFF sample index')
    samples = []
    for ident, offset, reserved in SAMPLE_INDEX.iter_unpack(data[32:]):
        require(ident > 0 and reserved == 0 and offset >= 20, 'Invalid WIFF sample index record')
        samples.append(SampleInfo(ident, offset))
    ids = [s.id for s in samples]
    stored = {int(p.split('/')[1][6:]) for p in names if re.match(r'SampleSubtree/Sample[1-9]\d*/', p)}
    require(len(set(ids)) == len(ids) and set(ids) == stored, 'WIFF sample identifiers disagree')
    require([s.segment_offset for s in samples] == sorted({s.segment_offset for s in samples}),
            'WIFF sample offsets are duplicated or unordered')
    return samples


def list_samples(entry):
    entry = Path(entry)
    require(entry.suffix.lower() in ('.wiff', '.wiff2'), 'Input must be a .wiff2 or .wiff file')
    require(entry.is_file(), f'Missing entry point: {entry}')
    with olefile.OleFileIO(entry.with_suffix('.wiff')) as ole:
        names = ['/'.join(p) for p in ole.listdir()]
        require('SampleSubtree/SampleIdxTable' in names, 'Missing WIFF sample index')
        samples = sample_index(ole.openstream('SampleSubtree/SampleIdxTable').read(), names)
    return sorted(samples, key=lambda s: s.id)


def scan_segments(data):
    """Map stored sample IDs to (header offset, payload start, payload end)."""
    require(len(data) >= 44, 'Truncated scan file')
    require(struct.unpack_from('<I', data)[0] == 2200, 'Unsupported scan-file version')
    require(struct.unpack_from('<Q', data, 4)[0] == len(data), 'Scan file size/header mismatch')
    require(data[12:20] == bytes(8), 'Unsupported scan container header')
    position, result = 20, {}
    while position < len(data):
        require(position+24 <= len(data), 'Truncated scan segment header')
        require(data[position:position+4] == b'\x11'*4, 'Unsupported scan container framing')
        version, ident, length, reserved = struct.unpack_from('<IIQI', data, position+4)
        require(version == 2200 and ident > 0 and ident not in result and reserved == 0,
                'Unsupported or duplicate scan segment identifier')
        start, end = position+24, position+24+length
        require(length > 0 and end <= len(data), 'Scan segment out of bounds')
        result[ident] = (position, start, end)
        position = end
    return result


def sample_output_path(path, sample_id):
    """Insert the metadata ID before the complete .sp.mzML / .mzML suffix."""
    path = Path(path)
    suffix = '.sp.mzML' if path.name.lower().endswith('.sp.mzml') else path.suffix
    stem = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(f'{stem}_{sample_id}{suffix}')
