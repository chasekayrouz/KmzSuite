"""Fused native spectrum paths and templates equal the interpreted reference (synthetic inputs)."""
import base64
from functools import partial
import random
import struct
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest
from kmzconvert_core import _native, native, writer
from kmzconvert_core.binary import FormatError, decode_tokens
from kmzconvert_core.mass_correction import MassCorrection, SearchSettings
from kmzconvert_core.reader import Experiment, Scan, WiffReader
from kmzcore import sparse_profile as sp
from Tests.Convert.test_native_kernels import outcome, same
from Tests.Convert.test_parallel_writer import tokens

EXPERIMENTS = [Experiment(1, 'positive', 400., 900., 200., None, {}), Experiment(2, 'positive', 50., 1000., 50., 30., {})]


class Mapping:
    closed = False


def acquisition(rng, count=24):
    """A production reader over synthetic blocks: valid, corrupted, wrongly totalled and odd calibrations."""
    grid = [int(g) for g in rng.choice([1, 1, 1, 2, 3, 40, 300, 70000], 300)]
    data, scans = bytearray(), []
    for i in range(count):
        ms1 = i % 4 != 3
        gaps = grid if ms1 and rng.random() < .8 else [int(g) for g in rng.choice([1, 2, 40, 300], int(rng.integers(1, 60)))]
        body = tokens(rng, gaps)
        if rng.random() < .1:
            body = body[:int(rng.integers(0, len(body)))]  # Truncated payloads raise FormatError.
        block = b'\xff' * 4 + struct.pack('<I', 4_000_000) + b'\0' + body + b'\xff' * int(rng.integers(1, 5))
        try:
            x, y = decode_tokens(block)
            tic, bpi = float(y.sum()), float(y.max())
        except FormatError:
            tic = bpi = 1.
        if rng.random() < .05:
            tic += 1.
        # A vertex inside the TOF range gives non-increasing measured or framing m/z.
        a, b = (2.5e-5, 0.) if rng.random() < .85 else (2.5e-5, float(rng.choice([4_000_000 * .1, 4_000_010 * .1])))
        scans.append(Scan(i, i // 4 + 1, 1 if ms1 else 2, i * 3., tic, bpi, len(data), len(block), a, b,
                          None if ms1 else 300. + i, None if ms1 else 1000., 1))
        data += block
    reader = object.__new__(native.NativeWiffReader)
    reader.__dict__.update(_mapping=Mapping(), data=bytes(data), tdc=.1, scans=scans, experiments=EXPERIMENTS,
                           nominal_window_overhangs={}, dad=None)
    return reader


def copy(reader, kind=None):
    """The same acquisition as a fresh reader (a plain WiffReader takes the interpreted path)."""
    other = object.__new__(kind or type(reader))
    other.__dict__.update(reader.__dict__, nominal_window_overhangs={})
    return other


def corrections():
    return [None, MassCorrection(4.713609144661427, (), SearchSettings()), MassCorrection(-250., (), SearchSettings())]


@pytest.mark.parametrize('seed', range(12))
def test_fused_arrays_and_regions_equal_the_reference(seed):
    rng = np.random.default_rng(seed)
    fused = acquisition(rng)
    reference = copy(fused, WiffReader)
    for scan in fused.scans:
        for correction in corrections():
            def arrays(reader):
                mz, y = reader.arrays(scan)  # Interpreted codec, measurement and framing.
                return (mz if correction is None else correction.correct(mz)), y
            expected = outcome(arrays, reference)
            assert same(outcome(native.reader_arrays, fused, scan, correction), expected), scan
            regions = outcome(lambda: sp.candidate_regions(*expected[1])) if expected[0] == 'ok' else expected
            actual = outcome(native.measured_regions, fused, scan, correction)
            assert actual[0] == regions[0] and (actual[0] != 'ok' or all(map(same, actual[1], regions[1]))), scan
    assert fused.nominal_window_overhangs == reference.nominal_window_overhangs


def reference_spectrum(reader, scan, index, profile, correction, selection):
    assert not native.applies(reader)  # Interpreted arrays, sampling check and ElementTree serialization.
    chosen = {} if selection is None else {'selection': selection}
    return ET.tostring(writer.spectrum(reader, scan, index, profile, correction, **chosen), encoding='utf-8') + b'\n'



@pytest.mark.parametrize('seed', range(12))
def test_templates_write_the_reference_spectra(seed):
    rng = np.random.default_rng(seed)
    fused = acquisition(rng)
    reference = copy(fused, WiffReader)
    templates = writer.Templates(fused)
    encoded = 0
    for i, scan in enumerate(fused.scans):
        for profile in ('full', 'sparse'):
            for correction in corrections():
                selection = None
                if profile == 'sparse' and scan.experiment == 1:
                    found = outcome(lambda: sp.candidate_regions(*native.reader_arrays(copy(fused), scan, correction)))
                    if found[0] == 'ok':
                        n = len(found[1].starts)
                        options = [sp.rescue_selection([found[1]])[0], rng.random(n) < .5, np.ones(n + 1, bool),
                                   np.ones(n, np.int8)]
                        selection = options[int(rng.choice(4, p=[.6, .2, .1, .1]))]
                expected = outcome(reference_spectrum, reference, scan, i, profile, correction, selection)
                assert same(outcome(templates.spectrum, scan, i, profile, correction, selection), expected), (scan, profile)
                ms1 = scan.experiment == 1
                encoded += native.encode_spectrum(copy(fused), scan, correction if ms1 else None, selection,
                                                  profile == 'sparse' and ms1) is not None if expected[0] == 'ok' else 0
    assert fused.nominal_window_overhangs == reference.nominal_window_overhangs
    assert encoded  # The fused kernel, not only its fallback, wrote spectra.


def test_templates_escape_like_elementtree():
    mz, y = np.linspace(100., 101., 50), np.r_[np.zeros(20), np.full(10, 500.), np.zeros(20)]
    scan = SimpleNamespace(native_id='a & <b> "c"\n\r\t é', experiment=2, precursor_mz=250.125, tic=float('inf'),
                           rt_seconds=1e-300)
    reader = SimpleNamespace(experiments=EXPERIMENTS, arrays=lambda _: (mz, y), nominal_window_overhangs={})
    element = writer.spectrum(reader, scan, 7)
    assert writer.Templates(reader).spectrum(scan, 7) == ET.tostring(element, encoding='utf-8') + b'\n'


def test_dad_templates_equal_elementtree():
    from Tests.dad_fixtures import sidecars
    dad = sidecars()[3]
    dad.sample_id = 'x & "y"'
    templates = writer.Templates(SimpleNamespace(dad=dad))
    for row in range(len(dad)):
        element = writer.dad_spectrum(dad, row, 40 + row)
        assert templates.dad_spectrum(row, 40 + row) == (element.get('id'), ET.tostring(element, encoding='utf-8') + b'\n')


@pytest.mark.parametrize('size', [0, 1, 7, 8, 1000, 65536, 300001])
def test_deflate_base64_equals_zlib(size):
    rng = random.Random(size)
    payloads = [np.frombuffer(rng.randbytes(size), np.uint8), np.zeros(size, np.uint8),
                np.sqrt(np.arange(1., size // 8 + 2))[:size // 8]]
    for data in payloads:
        texts = native.deflate_base64(data, data)
        assert [bytes(t) for t in texts] == [base64.b64encode(zlib.compress(data.tobytes()))] * 2
    assert native._same_zlib()


def test_deflate_falls_back_to_cpython_zlib(monkeypatch):
    native._same_zlib.cache_clear()
    monkeypatch.setattr(_native, 'zlib_runtime', lambda: 'another zlib')
    try:
        assert not native._same_zlib()
        assert native.deflate_base64(np.arange(3.)) == [base64.b64encode(zlib.compress(np.arange(3.).tobytes()))]
        reader = acquisition(np.random.default_rng(1))
        assert native.encode_spectrum(reader, reader.scans[0]) is None
    finally:
        native._same_zlib.cache_clear()


@pytest.mark.parametrize('chunk', [1, 3, 8])
def test_ordered_chunks_keep_order_and_propagate_failures(chunk):
    assert list(native.ordered(lambda i: i * i, range(100), 4, chunk)) == [i * i for i in range(100)]
    def fail(i):
        if i == 37:
            raise KeyboardInterrupt
        return i
    seen = []
    with pytest.raises(KeyboardInterrupt):
        for value in native.ordered(fail, range(100), 4, chunk):
            seen.append(value)
    assert seen == list(range(len(seen))) and len(seen) <= 37
    assert not [t for t in threading.enumerate() if t.name.startswith('kmzconvert')]


def peaks(values, gaps=None):
    """A block of explicit integer intensities (TOF deltas of one step unless given)."""
    body = bytearray()
    for i, value in enumerate(values):
        gap = 1 if gaps is None else gaps[i]
        body += bytes([127 + gap]) if gap > 1 else b''
        body += (bytes([value]) if value < 124 else bytes([124, value]) if value < 256
                 else bytes([125]) + value.to_bytes(2, 'little') if value < 65536 else bytes([126]) + value.to_bytes(4, 'little'))
    return b'\xff' * 4 + struct.pack('<I', 4_000_000) + b'\0' + bytes(body) + b'\xff'


def reader_of(blocks, tic=None):
    """A production reader over explicit MS1 blocks, with matching TIC/BPI unless given."""
    data, scans = bytearray(), []
    for i, block in enumerate(blocks):
        _, y = decode_tokens(block)
        total, top = (float(y.sum()), float(y.max())) if tic is None else tic
        scans.append(Scan(i, i + 1, 1, i * 3., total, top, len(data), len(block), 2.5e-5, 0., None, None, 1))
        data += block
    reader = object.__new__(native.NativeWiffReader)
    reader.__dict__.update(_mapping=Mapping(), data=bytes(data), tdc=.1, scans=scans, experiments=EXPERIMENTS,
                           nominal_window_overhangs={}, dad=None)
    return reader


def test_zero_intensities_and_inexact_sums_defer_to_numpy():
    # A zero maximum's sign and every guard outcome come from NumPy, as in the reference.
    blocks = [peaks([0]), peaks([5, 0, 7]), peaks([0, 0, 0])]
    fused = reader_of(blocks)
    reference = copy(fused, WiffReader)
    for scan, block in zip(fused.scans, blocks):
        assert _native.profile_spectrum(block, scan.cal_a, scan.cal_b, fused.tdc, 0.) is None
        assert same(outcome(native.reader_arrays, fused, scan), outcome(reference.arrays, scan))
    # Sums at or above 2**53 are not proven exact in integers: NumPy measures them.
    huge = peaks([2**32 - 1] * (2**21 + 1))
    fused = reader_of([huge])
    scan = fused.scans[0]
    assert _native.profile_spectrum(huge, scan.cal_a, scan.cal_b, fused.tdc, 0.) is None
    assert same(native.reader_arrays(fused, scan), copy(fused, WiffReader).arrays(scan, decoder=native.decode_tokens))


def test_event_unit_ties_and_many_distinct_values():
    rng = np.random.default_rng(3)
    # Ties within and across the integer tallies (< 4096) and the table of other values.
    cases = [[30] * 5 + [10] * 5 + [500] * 3, list(rng.integers(1, 4000, 3000)), list(rng.integers(1, 2**32, 3000)),
             [7] * 2990 + [900] * 10, [5000] * 3 + [7] * 3 + [9000] * 2, [4096] * 3 + [4095] * 3 + [1000] * 3,
             [70000] * 4 + [2**32 - 1] * 4 + [100] * 3, list(rng.choice([1, 3, 4095, 4096, 70000, 2**31], 5000))]
    blocks = [peaks([int(v) for v in values], [int(g) for g in rng.choice([1, 1, 2, 30], len(values))]) for values in cases]
    fused = reader_of(blocks)
    reference = copy(fused, WiffReader)
    for scan in fused.scans:
        expected = sp.candidate_regions(*reference.arrays(scan))
        assert same(tuple(native.measured_regions(fused, scan)), tuple(expected)), scan
