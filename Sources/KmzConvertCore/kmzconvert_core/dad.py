"""Guarded SCIEX OS 2.x DAD sidecars, corroborated by the WIFF DAD streams.

This is a qualified layout, not a general DAD decoder. No vendor software or
reference export is used. The KmzConvert manual lists the remaining limits.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import numpy as np

from .binary import FormatError
from .samples import scan_segments

INDEX = struct.Struct("<IQIHddd")
STREAM_HEADER = bytes.fromhex("0000000004000000") + bytes(24)


@dataclass
class DadData:
    wavelength: np.ndarray
    rt_min: np.ndarray
    absorbance_mau: np.ndarray
    total_mau: np.ndarray
    sources: list[tuple[Path, str]]
    sample_id: int = 1

    def __len__(self):
        return len(self.rt_min)


def require(condition, message):
    if not condition:
        raise FormatError(message)


def read_dad(entry, streams, sample_id=1):
    """Return (complete DAD or None, status, user-facing diagnostic).

    Missing/unqualified DAD never prevents MS-only conversion. Both sidecars
    are required even when lower-precision duplicate data remains in the WIFF.
    """
    paths = [Path(entry).with_suffix(ext) for ext in (".dad.scan", ".dad.sidx")]
    if not any(p.exists() or p.is_symlink() for p in paths):
        return None, "absent", "No DAD sidecars supplied; converting MS data only."
    raw, problems = [], []
    for path in paths:
        try:
            raw.append(path.read_bytes())
        except OSError as exc:
            raw.append(None)
            problems.append(f"{path.name}: {exc.strerror or str(exc)}")
    if problems:
        return None, "unreadable", "; ".join(problems) + ". Continuing with MS data only."
    try:
        data = decode_dad(*raw, streams, sample_id=sample_id)
    except (FormatError, ValueError, struct.error, KeyError) as exc:
        return None, "unreadable", f"Cannot use {paths[0].name} / {paths[1].name}: {exc}. Continuing with MS data only."
    data.sources = [(path, hashlib.sha1(content).hexdigest()) for path, content in zip(paths, raw)]
    return data, "included", f"Including {len(data)} DAD spectra ({len(data.wavelength)} wavelengths)."


def decode_dad(scan, index, streams, sample_id=1):
    keys = [k for k in streams if k.startswith(f'SampleSubtree/Sample{sample_id}/') and k.endswith('/DAD/DADSampleHeader')]
    require(len(keys) == 1, "expected one WIFF DAD metadata header")
    prefix = keys[0].removesuffix('DADSampleHeader')
    header = streams[keys[0]]
    embedded = streams[prefix + 'DADRealTimeData']
    embedded_index = streams[prefix + 'DADIdx']
    require(len(header) == 86 and header[:32] == STREAM_HEADER, "unsupported WIFF DAD header")
    require(header[32:34] == bytes(2) and header[62:82] == bytes(20), "unsupported DAD header fields")
    count, width, channels, last, lower, upper, step = struct.unpack_from('<IIIffff', header, 34)
    first = struct.unpack_from('<f', header, 82)[0]
    require(count > 0 and width > 1 and channels == 1, "unsupported DAD dimensions/channel layout")
    require(np.isfinite([lower, upper, step, first, last]).all() and 0 < lower < upper and step > 0,
            "invalid DAD wavelength/time metadata")
    require(lower + (width - 1) * step == upper, "DAD wavelength endpoints/count disagree")
    segments = scan_segments(scan)
    require(sample_id in segments, 'DAD scan segment missing for selected sample')
    _, start, end = segments[sample_id]
    require(end-start == count * width * 8, ".dad.scan length disagrees with WIFF metadata")
    require(len(index) >= 52 and index[:28] == struct.pack('<8sIQII', b'@dadsidx', 1002, len(index), 1, 0),
            'unsupported .dad.sidx framing')
    position, indices = 28, {}
    while position < len(index):
        require(position+24 <= len(index), 'truncated .dad.sidx segment')
        version, ident, stop, zero1, zero2 = struct.unpack_from('<IIQII', index, position)
        begin = position+24
        require(version == 1002 and ident > 0 and ident not in indices and zero1 == zero2 == 0
                and begin < stop <= len(index) and (stop-begin) % INDEX.size == 0,
                'unsupported .dad.sidx segment')
        indices[ident] = (begin, stop)
        position = stop
    require(set(indices) == set(segments), 'DAD scan/index sample identifiers disagree')
    begin, stop = indices[sample_id]
    require(stop-begin == count * INDEX.size, '.dad.sidx length disagrees with WIFF metadata')
    records = list(INDEX.iter_unpack(index[begin:stop]))
    rt = np.array([r[4] for r in records])
    total = np.array([r[5] for r in records])
    for i, (number, offset, size, kind, _rt, _total, reserved) in enumerate(records):
        require((number, offset, size, kind, reserved) == (i, i * width * 8, width * 8, 7, 0),
                f"unsupported .dad.sidx record {i + 1}")
    require(np.isfinite(rt).all() and np.all(rt >= 0) and np.all(np.diff(rt) >= 0), "invalid DAD retention times")
    require(np.float32(rt[0]) == first and np.float32(rt[-1]) == last, "DAD time endpoints disagree")
    absorbance = np.frombuffer(scan, dtype='<f8', offset=start, count=count*width).reshape(count, width)
    require(np.isfinite(absorbance).all() and np.isfinite(total).all(), "nonfinite DAD absorbance")
    require(len(embedded) == 32 + count * width * 4 and embedded[:32] == STREAM_HEADER,
            "unsupported WIFF DAD spectrum copy")
    require(len(embedded_index) == 32 + count * 8 and embedded_index[:32] == STREAM_HEADER,
            "unsupported WIFF DAD index copy")
    duplicate = np.frombuffer(embedded, dtype='<f4', offset=32).reshape(count, width)
    duplicate_index = np.frombuffer(embedded_index, dtype='<f4', offset=32).reshape(count, 2)
    require(np.array_equal(absorbance.astype('<f4'), duplicate), ".dad.scan disagrees with WIFF DAD measurements")
    require(np.array_equal(rt.astype('<f4'), duplicate_index[:, 0]), ".dad.sidx disagrees with WIFF DAD times")
    require(np.array_equal(total.astype('<f4'), duplicate_index[:, 1]), ".dad.sidx disagrees with WIFF DAD totals")
    # Different addition orders may round differently, especially around zero.
    tolerance = np.maximum(1e-9, np.sum(np.abs(absorbance), axis=1) * 1e-12)
    require(np.all(np.abs(absorbance.sum(axis=1) - total) <= tolerance), "DAD totals disagree with spectrum sums")
    return DadData(lower + step * np.arange(width), rt, absorbance, total, [], sample_id)
