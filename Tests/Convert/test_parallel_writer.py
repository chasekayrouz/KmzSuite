"""Threaded native conversion equals sequential interpreted conversion; interruption leaves nothing behind.

Synthetic many-spectrum single-run containers only; no private WIFF measurements.
"""
from io import BytesIO
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from kmzconvert_core import native, writer
from kmzconvert_core.binary import FormatError, decode_tokens
from kmzconvert_core.mass_correction import CalibrationObservation, Reference, fit_constant_ppm
from kmzconvert_core.reader import INDEX, WiffReader
from kmzconvert_core.samples import STREAM_HEADER
from Tests.Convert.test_samples import framed_segments

CYCLES, A, B, SEED = 240, .000025, 0., 4_000_000


def varint(value):
    data = bytearray()
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    return bytes(data) + bytes([value])


def tokens(rng, gaps):
    """Valid peaks over a low event level, including multi-byte deltas and intensities."""
    body = bytearray()
    for gap in gaps:
        if 1 < gap <= 124:
            body.append(127 + gap)
        elif gap > 124:
            size = 1 if gap <= 256 else 2 if gap <= 65536 else 4
            body += bytes([251 + size if size < 4 else 254]) + (gap - 1).to_bytes(size, 'little')
        value = int(rng.choice([7, 7, 7, 7, 14, 21, 70, 120, 160, 300, 2000, 70000]))
        body += (bytes([value]) if value < 124 else bytes([124, value]) if value < 256
                 else bytes([125]) + value.to_bytes(2, 'little') if value < 65536 else bytes([126]) + value.to_bytes(4, 'little'))
    return bytes(body)


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    rng = np.random.default_rng(29)
    streams = {'FileRec_Str': 'SCIEX OS 2.2.0.0'.encode('utf-16le')}
    period = bytearray(78); struct.pack_into('<I', period, 74, 2)
    method = {'Period0/PeriodStream': period}
    for exp, kind in enumerate((8, 9)):
        prefix = f'Period0/Experiment{exp}/'
        header = bytearray(98); struct.pack_into('<I', header, 94, kind)
        method[prefix+'ExperimentHeader'] = header
        method[prefix+'ExperimentHeaderEx'] = bytes(66) + struct.pack('<ddd', 50., 1600., 200.) + bytes(4)
        name = 'IS'.encode('utf-16le')
        method[prefix+'IonSourceParamsTable/Parameter0/ParameterData'] = bytes(34)+struct.pack('<H', len(name))+name+struct.pack('<ff', 5500., 5500.)
        name = 'CE'.encode('utf-16le')
        method[prefix+'MassRangeEx/MassRangeEx'] = bytes(32)+struct.pack('<II', 1, 1)+bytes(22)+struct.pack('<H', len(name))+name+struct.pack('<ffI', 30., 30., 0)
    streams.update({'MethodSubtree/Method1/DeviceMethod0/'+k: bytes(v) for k, v in method.items()})
    cal = b'\x09' + struct.pack('<d', A) + b'\x11' + struct.pack('<d', B)
    checkpoint = b'\x08' + varint(SEED) + b'\x10\x08'
    metadata = b'\x0a' + varint(len(cal)) + cal + b'\x12' + varint(len(checkpoint)) + checkpoint
    metadata = varint(len(metadata)) + metadata
    payload, index, dde = bytearray(), [], bytearray(32)
    # MS1 spectra share a TOF grid so candidate regions link across neighbors (v3 rescue).
    grid = [int(g) for g in rng.choice([1, 1, 1, 2, 3, 40, 300, 70000], 400)]
    for slot in range(2 * CYCLES):
        ms1 = slot % 2 == 0
        gaps = grid if ms1 else [int(g) for g in rng.choice([1, 2, 40, 300], 30)]
        block = b'\xff'*4 + struct.pack('<I', SEED) + b'\x00' + tokens(rng, gaps) + b'\xff' * int(rng.integers(1, 5))
        _, y = decode_tokens(block)
        payload += metadata + block
        index.append(INDEX.pack(len(payload) - len(block), len(block), slot * 50., 6, float(y.sum()), float(y.max()), 0., 0., 0))
        if not ms1:
            dde += struct.pack('<dddd', 100. + slot, 0., 1000. + slot, 0.)
    prefix = 'SampleSubtree/Sample1/'
    streams[prefix+'Idx'] = STREAM_HEADER + b''.join(index)
    streams[prefix+'TOFCalibrationData'] = bytes(52) + struct.pack('<I', len(index)) + struct.pack('<ddI', A, B, 0) * len(index)
    streams[prefix+'TDCInfo'] = bytes(32) + struct.pack('<d', .1)
    streams[prefix+'DDERealTimeData'] = bytes(dde)
    streams[prefix+'Log'] = 'Mass Spectrometer: X500 QTOF'.encode('utf-16le')
    scan, offsets = framed_segments([(1, bytes(payload))])
    streams['SampleSubtree/SampleIdxTable'] = STREAM_HEADER + b''.join(struct.pack('<IQI', *r) for r in offsets)
    path = tmp_path / 'many.wiff'
    path.write_bytes(b'synthetic OLE stand-in')
    path.with_suffix('.wiff.scan').write_bytes(scan)

    class Ole:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def listdir(self): return [k.split('/') for k in streams]
        def openstream(self, key): return BytesIO(streams['/'.join(key) if isinstance(key, list) else key])
    monkeypatch.setattr('olefile.OleFileIO', Ole)
    return path


def correction():
    observation = CalibrationObservation(Reference('reference-1', 100.2, .1), 100.2004, .1, 3, .1, 'synthetic', 10.)
    return fit_constant_ppm([observation])


def workers():
    return [t for t in threading.enumerate() if t.name.startswith('kmzconvert')]


@pytest.mark.parametrize('profile', ['full', 'sparse'])
@pytest.mark.parametrize('corrected', [False, True])
def test_threaded_native_output_equals_sequential_reference(acquisition, tmp_path, monkeypatch, profile, corrected):
    assert native.ACCELERATED
    options = dict(profile=profile, mass_correction=correction() if corrected else None)
    with native.NativeWiffReader(acquisition) as reader:
        assert native.workers(reader) > 1
        fast = writer.write_mzml(reader, tmp_path / 'native.mzML', **options)
        overhangs = list(reader.nominal_window_overhangs)
    # The plain WiffReader (validators, adapters) takes the sequential interpreted path.
    with WiffReader(acquisition) as reader:
        assert native.workers(reader) == 0
        reference = writer.write_mzml(reader, tmp_path / 'reference.mzML', **options)
        assert list(reader.nominal_window_overhangs) == overhangs
    with monkeypatch.context() as patched:
        # Without the extension, threaded conversion uses the interpreted functions.
        patched.setattr(native, '_native', None)
        with native.NativeWiffReader(acquisition) as reader:
            assert native.workers(reader) > 1
            fallback = writer.write_mzml(reader, tmp_path / 'fallback.mzML', **options)
    assert fast.read_bytes() == reference.read_bytes() == fallback.read_bytes()
    assert not workers()


@pytest.mark.parametrize('profile', ['full', 'sparse'])
def test_interrupt_joins_workers_and_removes_unfinished_output(acquisition, tmp_path, profile):
    calls = []
    def interrupt(done, total):
        calls.append(done)
        if done:
            assert workers()
            raise KeyboardInterrupt
    with native.NativeWiffReader(acquisition) as reader:
        with pytest.raises(KeyboardInterrupt):
            writer.write_mzml(reader, tmp_path / 'cancelled.mzML', interrupt, profile=profile)
        assert not workers()
        assert not reader._mapping.closed
    assert reader._mapping.closed and calls[-1] == 1
    assert not list(tmp_path.glob('*.mzML')) and not list(tmp_path.glob('.kmzconvert-*.tmp'))


@pytest.mark.parametrize('profile', ['full', 'sparse'])
def test_worker_failures_and_interrupts_propagate_in_order(acquisition, tmp_path, monkeypatch, profile):
    # Every decoded spectrum passes the reader's guards on a worker, whichever kernel decoded it.
    original = WiffReader.check_measured
    for error in (FormatError('synthetic decoding failure'), KeyboardInterrupt()):
        calls = []
        def fail(*args):
            calls.append(None)
            if len(calls) == 37:
                raise error
            return original(*args)
        monkeypatch.setattr(WiffReader, 'check_measured', fail)
        with native.NativeWiffReader(acquisition) as reader:
            with pytest.raises(type(error)):
                writer.write_mzml(reader, tmp_path / 'failed.mzML', profile=profile)
            assert not workers()
        assert not list(tmp_path.glob('*.mzML')) and not list(tmp_path.glob('.kmzconvert-*.tmp'))


@pytest.mark.integration
def test_cli_sigint_during_threaded_conversion_removes_output(tmp_path):
    entry = Path(os.environ.get('KMZCONVERT_FIXTURES', Path(__file__).resolve().parents[2] / 'TestData/Local')) / 'acquisition-01.wiff'
    if not entry.is_file():
        pytest.skip('Optional local acquisition unavailable')
    root = Path(__file__).resolve().parents[2]
    paths = [str(root / p) for p in ('Sources/KmzCore', 'Sources/KmzConvertCore', 'Tools/KmzConvert')]
    code = f'import sys; sys.path[:0] = {paths!r}; from kmzconvert.cli import main; raise SystemExit(main(sys.argv[1:]))'
    process = subprocess.Popen([sys.executable, '-c', code, str(entry), '-o', str(tmp_path / 'out.mzML')],
                               stderr=subprocess.PIPE, cwd=tmp_path)
    seen = b''
    deadline = time.monotonic() + 60
    while b'Converting 1/' not in seen and time.monotonic() < deadline:
        seen += process.stderr.read1(4096)
    process.send_signal(signal.SIGINT)
    seen += process.stderr.read()
    assert process.wait(timeout=60) == 130, seen.decode(errors='replace')
    assert b'cancelled' in seen and b'Traceback' not in seen
    assert not list(tmp_path.iterdir())
