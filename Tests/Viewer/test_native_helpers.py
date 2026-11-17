"""Native viewer helpers equal the Python expressions they replace, or decline to them."""
import base64
import binascii
import random
import zlib

import numpy as np
import pytest

from kmzviewer import _native, extract, mz_filter, mzml_direct, spectra
from kmzviewer.extract import _sorted_pairs, extract_eics, make_tic_trace, parse_targets, sum_interval
from kmzviewer.model import Trace
from kmzviewer.mz_filter import extract_filtered_values, mz_range
from kmzviewer.mzml_index import index_ms1, load_ms1
from kmzviewer.spectra import ArrayCache, SpectrumReader
from Tests.Viewer.mzml_factory import write_mzml
from Tests.Viewer.test_direct_decode import assert_same_as_pyteomics, peaks, VARIANTS


def test_accelerated():
    assert mzml_direct.ACCELERATED and spectra._DECODERS >= 1


def reference_decode(data, start, compressed, size):
    end = data.find(b"<", start)
    raw = binascii.a2b_base64(data[start:end], strict_mode=True)
    return zlib.decompress(raw, zlib.MAX_WBITS, size) if compressed else raw


def test_inflate_equals_binascii_and_zlib():
    rng = np.random.default_rng(3)
    for size in [*range(0, 40), 1000, 65537]:
        payload = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        for compressed in (False, True):
            raw = zlib.compress(payload) if compressed else payload
            if not raw:
                continue
            data = b"<binary>" + base64.b64encode(raw) + b"</binary>"
            for hint in (size, size + 9) if compressed else (0,):
                if compressed and hint == 0:
                    continue
                found, end = _native.inflate_base64(data, 8, 1 << 30, compressed, hint)
                assert end == data.find(b"<", 8)
                assert isinstance(found, bytearray) and bytes(found) == reference_decode(data, 8, compressed, hint)


@pytest.mark.parametrize("text, compressed, hint", [
    (b"QUJD\nREVG", False, 0),          # whitespace: strict binascii rejects it
    (b"QUJDRA=", False, 0),             # incomplete quantum
    (b"QUJDRA==QQ==", False, 0),        # data after padding
    (b"QR==", False, 0),                # nonzero discarded bits: left to binascii
    (b"=QUJ", False, 0),
    (base64.b64encode(zlib.compress(b"abc") + b"x"), True, 3),   # trailing data after the stream
    (base64.b64encode(zlib.compress(b"abcdef")), True, 3),        # more output than the hint
    (base64.b64encode(zlib.compress(b"abcdef")[:-2]), True, 6),   # truncated stream
    (base64.b64encode(b"\x78\x9cnot zlib"), True, 6),
])
def test_inflate_declines_to_reference(text, compressed, hint):
    data = b"<binary>" + text + b"</binary>"
    found, end = _native.inflate_base64(data, 8, 1 << 30, compressed, hint)
    assert found is None and end == data.find(b"<", 8)


def test_inflate_limits_and_missing_end():
    data = b"<binary>QUJD"
    assert _native.inflate_base64(data, 8, 100, False, 0) == (None, -1)
    assert _native.inflate_base64(data + b"<", 8, 3, False, 0) == (None, 12)
    assert _native.inflate_base64(data + b"<", 8, 4, False, 0) == (bytearray(b"ABC"), 12)
    assert _native.inflate_base64(b"<binary><", 8, 4, False, 0) == (None, 8)


def numpy_state(mz):
    if not np.isfinite(mz).all():
        return 0
    return 2 if mz.size <= 1 or (mz[0] <= mz[-1] and np.all(mz[1:] >= mz[:-1])) else 1


def test_ordered_and_finite_max_equal_numpy():
    rng = np.random.default_rng(11)
    specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 5e-324, 1e308])
    for size in [*range(0, 20), 41, 1000]:
        for trial in range(30):
            values = np.sort(rng.normal(size=size)) if trial % 2 else rng.normal(size=size)
            if size and trial % 3 == 0:
                values[rng.integers(0, size, 2)] = rng.choice(specials, 2)
            if size and trial % 5 == 0:
                values[:] = rng.choice([0.0, -0.0], size)
            assert _native.ordered(values) == numpy_state(values)
            count, top = _native.finite_max(values)
            finite = values[np.isfinite(values)]
            assert count == finite.size
            if top:
                assert np.float64(top).tobytes() == np.max(finite).tobytes()


def test_prepare_returns_the_same_arrays_as_numpy(monkeypatch):
    rng = np.random.default_rng(2)
    cases = [np.array([]), np.array([3.]), np.array([1., 1., 2.]), np.array([2., 1., 3.]),
             np.array([1., np.nan, 2.]), np.array([np.inf, 1.]), np.array([0.0, -0.0, 0.0]),
             rng.normal(size=500), np.sort(rng.normal(size=500))]
    for mz in cases:
        intensity = rng.normal(size=mz.size)
        native = extract._prepare(mz, intensity)
        monkeypatch.setattr(extract, "_ordered", None)
        expected = extract._prepare(mz, intensity)
        monkeypatch.undo()
        finite = np.isfinite(mz)
        assert [a.tobytes() for a in native] == [a.tobytes() for a in expected] == [
            a.tobytes() for a in _sorted_pairs(mz[finite], intensity[finite])]
        if np.isfinite(mz).all() and numpy_state(mz) == 2:
            assert native[0] is mz and native[1] is intensity


def test_indexed_tags_and_find_equal_python():
    rng = random.Random(7)
    pieces = [b"<", b"<<", b"<spectrum ", b"<spectrum>", b"<spectrum\x0b", b"<chromatogram\n", b"<ns:spectrum ",
              b"<a:b:spectrum ", b"<:spectrum ", b"<spectrumX ", b"<spectrumx:spectrum ", b' id="s"', b"AAAA",
              b"</spectrum>", b"<spect", b"rum ", b"\xff", b"<binary>", b"<s", b"<sp:"]
    for _ in range(5000):
        data = b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
        want = [(at, following) for at, following in mzml_direct._python_chunks(data)
                if mzml_direct._INDEXED.match(data, at, following)]
        assert _native.indexed_tags(data) == (want, data.rfind(b"<"))
        needle = rng.choice([b"</spectrum>", b"<", b"", b"spectrum", b"<<"])
        start = rng.randint(-5, len(data) + 5)
        assert _native.find(bytearray(data), needle, start) == data.find(needle, start)


@pytest.mark.parametrize("chunk", [7, 64, 4096, 4 * 1024 * 1024])
def test_open_records_the_reader_index(tmp_path, monkeypatch, chunk):
    path = tmp_path / "sample.mzML"
    write_mzml(path, peaks(), chromatogram=True)
    mzml_direct._indexes.clear()
    monkeypatch.setattr(mzml_direct, "_READ_BLOCK", 977)
    index_ms1(path, chunk_size=chunk)
    key = mzml_direct._identity(str(path), path.stat())
    assert mzml_direct._indexes[key] == mzml_direct._build_index(str(path))
    monkeypatch.setattr(mzml_direct, "_build_index", lambda _path: pytest.fail("Reader rescanned the file"))
    assert_same_as_pyteomics(path, 12)


def test_recorded_index_follows_file_identity_and_complete_reads(tmp_path, monkeypatch):
    path = tmp_path / "sample.mzML"
    write_mzml(path, peaks(), chromatogram=True)
    mzml_direct._indexes.clear()
    load_ms1(path)
    stale = mzml_direct._identity(str(path), path.stat())
    # A same-size rewrite changes ctime, so readers index the new version again.
    text = path.read_text().replace('id="scan=2"', 'id="scan=Z"')
    path.write_text(text)
    assert mzml_direct._identity(str(path), path.stat()) != stale
    assert_same_as_pyteomics(path, 12)
    # Failed or interrupted opens and the interpreted path record nothing.
    for change in ("truncated", "interpreted"):
        mzml_direct._indexes.clear()
        if change == "truncated":
            data = path.read_bytes()
            path.write_bytes(data[:data.rindex(b"</spectrum>") - 40])
            with pytest.raises(ValueError):
                load_ms1(path)
        else:
            write_mzml(path, peaks(), chromatogram=True)
            monkeypatch.setattr(mzml_direct, "_indexed_tags", None)
            load_ms1(path)
        assert not mzml_direct._indexes
    monkeypatch.undo()


@pytest.mark.parametrize("variant", ["wrapped base64", "bad padding", "corrupt zlib", "float32 uncompressed"])
def test_interpreted_decoder_equals_pyteomics(sample_path, monkeypatch, variant):
    monkeypatch.setattr(mzml_direct, "_inflate_base64", None)
    sample_path.write_text(VARIANTS[variant](sample_path.read_text()))
    assert_same_as_pyteomics(sample_path, 12)


@pytest.fixture
def sample_path(tmp_path):
    path = tmp_path / "sample.mzML"
    write_mzml(path, peaks(), chromatogram=True)
    return path


def signed_zero_file(path):
    rows = [{"mz": np.array([100., 200., 300.]), "intensity": np.array(values), "rt": i / 10}
            for i, values in enumerate([[0.0, -0.0, -1.0], [-0.0, 0.0, -2.0], [-0.0, -0.0, np.nan],
                                        [np.nan, np.inf, -np.inf], [3.0, -0.0, np.nan], [-5.0, -1.0, -3.0]])]
    write_mzml(path, rows)


def test_native_reductions_equal_numpy_reductions(tmp_path, monkeypatch):
    path = tmp_path / "zeros.mzML"
    signed_zero_file(path)
    scans = load_ms1(path)
    rows = np.arange(len(scans), dtype=np.int32)

    def run():
        bpc = Trace(1, 0, "bpc", "BPC", scans.rt_min.copy(), np.zeros(len(rows)), rows, "", "#000")
        traces = [make_tic_trace(2, "#000", scans), bpc] + extract_eics(
            str(path), scans, parse_targets("100, 200, 300", 1e5), "all", False, [3, 4, 5], ["#000"] * 3)
        out = [extract_filtered_values(str(path), scans, traces, bounds, cache=ArrayCache())
               for bounds in (None, mz_range(150, 400))]
        with SpectrumReader(path) as reader:
            sums = [np.float64(sum_interval(*reader.read(j), 150, 400)).tobytes() for j in range(len(scans))]
        return [(key, value.tobytes()) for result in out for key, value in sorted(result.items())] + [
            trace.y.tobytes() for trace in traces[2:]] + sums

    native = run()
    monkeypatch.setattr(mz_filter, "_finite_max", None)
    monkeypatch.setattr(extract, "_ordered", None)
    assert run() == native


def test_inflate_rejects_mutable_buffers_released_off_the_gil():
    # A bytearray could change while the GIL is released; only immutable bytes are accepted.
    with pytest.raises(TypeError):
        _native.inflate_base64(bytearray(b"AAAA<"), 0, 100, False, 0)
    assert _native.inflate_base64(b"AAAA<", 0, 100, False, 0) == (bytearray(3), 4)


def test_smoke_module_imports():
    import importlib
    importlib.import_module("kmzcore.runtime.smoke")
