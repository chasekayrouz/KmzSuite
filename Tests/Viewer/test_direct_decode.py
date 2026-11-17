"""Direct spectrum decoding equals pyteomics byte for byte, or declines to it."""
import base64
import threading
import zlib

import numpy as np
import pytest
from pyteomics import mzml

from kmzcore.cancel import Cancelled
from kmzcore.trace_style import _lab, _linear, _rgb, chromatogram_palette
from kmzconvert_core.writer import write_mzml as write_native
from kmzviewer import mzml_direct
from kmzviewer.extract import make_tic_trace
from kmzviewer.mz_filter import extract_filtered_values
from kmzviewer.mzml_index import _CLOSE, _find_close, load_ms1
from kmzviewer.spectra import ArrayCache, SpectrumReader, _local_vocabulary, read_ahead
from Tests.dad_fixtures import acquisition
from Tests.Viewer.mzml_factory import write_mzml

FLOAT = '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>'
ZLIB = '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>'
MZ = ('<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value="" '
      'unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>')


def peaks(count=12):
    rng = np.random.default_rng(1729)
    rows = []
    for index in range(count):
        size = (300, 7, 1, 0)[index % 4]
        mz = np.sort(rng.uniform(100, 900, size))
        intensity = rng.normal(size=size) * 1e6
        if size > 5:
            intensity[2], intensity[3], mz[4] = np.nan, -np.inf, np.nan
        ms2 = index % 5 == 4
        rows.append({"mz": mz, "intensity": intensity, "rt": .01 * index, "tic": 1. + index,
                     "polarity": (1, -1)[index % 2], "ms_level": 2 if ms2 else 1,
                     "precursor_mz": 400.5 if ms2 else None})
    return rows


def outcome(reader, index):
    try:
        arrays = reader.read(index)
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return [(a.dtype.str, a.shape, a.tobytes(), a.flags.writeable, a.flags.c_contiguous) for a in arrays]


def assert_same_as_pyteomics(path, count):
    direct, reference = SpectrumReader(path), SpectrumReader(path)
    reference._direct = False
    try:
        for index in [*range(-1, count + 1), 0, count - 1]:
            assert outcome(direct, index) == outcome(reference, index), index
    finally:
        direct.close()
        reference.close()


def pyteomics_index(path):
    with mzml.MzML(str(path), cv=_local_vocabulary()) as reader:
        found = reader._offset_index["spectrum"]
        return tuple(found), tuple(found.values())


def direct_count(path, count):
    decoder = mzml_direct.open_direct(str(path))
    try:
        return sum(decoder.decode(index) is not None for index in range(count))
    finally:
        decoder.close()


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.mzML"
    write_mzml(path, peaks(), chromatogram=True)
    return path


@pytest.mark.parametrize("block", [5, 64, 977, 4 * 1024 * 1024])
def test_index_matches_pyteomics_across_read_blocks(sample, monkeypatch, block):
    monkeypatch.setattr(mzml_direct, "_READ_BLOCK", block)
    text = sample.read_text()
    # Duplicate, entity-bearing and prefixed ids follow pyteomics' lexical index.
    text = text.replace('id="scan=3"', 'id="scan=1"').replace('id="scan=4"', 'id="a&amp;b"')
    start = text.index('<spectrum index="5"')
    end = text.index("</spectrum>", start)
    text = (text[:start] + "<x:" + text[start + 1:end] + "</x:spectrum>" + text[end + 11:]).replace(
        '<mzML xmlns=', '<mzML xmlns:x="urn:x" xmlns=')
    sample.write_text(text)
    found = mzml_direct._build_index(str(sample))
    assert found[:2] == pyteomics_index(sample)
    assert list(found[2]) == sorted(found[2])


@pytest.mark.parametrize("change", ["double bracket", "saved index", "missing id", "shared id"])
def test_index_declines_when_pyteomics_would_differ_or_fail(sample, change):
    text = sample.read_text()
    if change == "double bracket":
        text = text.replace("<run", "<<run", 1)
    elif change == "saved index":
        (sample.parent / "sample-mzML-byte-offsets.json").write_text("{}")
    elif change == "missing id":
        text = text.replace(' id="scan=2"', "")
    else:
        text = text.replace('id="TIC"', 'id="scan=2"')
    sample.write_text(text)
    # A saved pyteomics index is checked before every cache lookup, not while indexing.
    assert (mzml_direct.open_direct(str(sample)) if change == "saved index" else mzml_direct._build_index(str(sample))) is None
    assert_same_as_pyteomics(sample, 12)


def test_direct_reads_equal_pyteomics_and_skip_only_precursor_scans(sample):
    assert_same_as_pyteomics(sample, 12)
    assert direct_count(sample, 12) == 12 - 2



@pytest.mark.parametrize("name", ["pyteomics", "numpy"])
def test_unqualified_versions_read_through_pyteomics(sample, unqualified, name):
    direct = SpectrumReader(sample)
    try:
        expected = [outcome(direct, index) for index in range(12)]
        assert direct._direct
    finally:
        direct.close()
    unqualified(name)
    mzml_direct._indexes.clear()
    load_ms1(sample)
    assert not mzml_direct._indexes  # MS1 indexing records nothing for the direct decoder.
    assert mzml_direct.open_direct(str(sample)) is None
    reader = SpectrumReader(sample)
    try:
        assert [outcome(reader, index) for index in range(12)] == expected
        assert reader._direct is False
    finally:
        reader.close()

def test_native_dad_and_ms1_reads_are_direct_and_equal(tmp_path):
    path = write_native(acquisition(tmp_path), tmp_path / "native.mzML")
    scans = load_ms1(path)
    count = len(scans) + len(scans.dad)
    assert direct_count(path, count) == count
    assert_same_as_pyteomics(path, count)
    original = path.read_bytes()
    for absorbance in (b'unitAccession="UO:0000000" unitName="unqualified"', b'unitName="milliabsorbance unit"'):
        path.write_bytes(original.replace(b'unitAccession="UO:0000269" unitName="absorbance unit"', absorbance))
        assert_same_as_pyteomics(path, count)


def array_xml(values, dtype, compression, name=MZ):
    raw = np.asarray(values, dtype=dtype).tobytes()
    if compression == "zlib compression":
        raw = zlib.compress(raw)
    names = {"<f4": "32-bit float", "<f8": "64-bit float", "<i4": "32-bit integer", "<i8": "64-bit integer"}
    accession = {"zlib compression": "MS:1000574", "no compression": "MS:1000576"}[compression]
    return (f'<binaryDataArray encodedLength="1"><cvParam cvRef="MS" accession="MS:1000523" '
            f'name="{names[dtype]}" value=""/><cvParam cvRef="MS" accession="{accession}" '
            f'name="{compression}" value=""/>{name}<binary>{base64.b64encode(raw).decode()}</binary>'
            '</binaryDataArray>')


VARIANTS = {
    "float32 uncompressed": lambda t: t.replace(t[t.index("<binaryDataArray "):t.index("</binaryDataArray>") + 18],
                                                array_xml([1.5, 2.25, np.nan], "<f4", "no compression"), 1),
    "integers": lambda t: t.replace(t[t.index("<binaryDataArray "):t.index("</binaryDataArray>") + 18],
                                    array_xml([7, -3, 2**40], "<i8", "zlib compression"), 1),
    "empty binary": lambda t: t.replace(t[t.index("<binary>"):t.index("</binary>")], "<binary>", 1),
    "entity title": lambda t: t.replace(
        '<scanList count="1">', '<userParam name="title" value="A &quot;b&quot; &amp; c"/><scanList count="1">', 1),
    "wrapped base64": lambda t: t.replace("<binary>eJ", "<binary>e\nJ", 1),
    "comment": lambda t: t.replace("<scanList", "<!-- c --><scanList", 1),
    "unknown valued term": lambda t: t.replace('accession="MS:1000511"', 'accession="MS:9999999"', 1),
    "unit by accession": lambda t: t.replace(' unitAccession="UO:0000031" unitName="minute"',
                                             ' unitAccession="UO:0000031"', 1),
    "nested id": lambda t: t.replace("<scan>", '<scan id="scan=0">', 1),
    "numpress": lambda t: t.replace('name="zlib compression"', 'name="MS-Numpress linear prediction compression"', 1),
    "corrupt zlib": lambda t: t.replace("<binary>eJ", "<binary>eA", 1),
    "bad padding": lambda t: t.replace("</binary>", "A</binary>", 1),
    "array param group": lambda t: t.replace(FLOAT, '<referenceableParamGroupRef ref="g"/>', 1).replace(
        "<run", '<referenceableParamGroupList count="1"><referenceableParamGroup id="g">'
                f'{FLOAT}</referenceableParamGroup></referenceableParamGroupList><run', 1),
    "attribute order and spacing": lambda t: t.replace('<spectrum index="1" id="scan=1"',
                                                       '<spectrum\n id="scan=1"\tindex = "1" ', 1),
    "single quotes": lambda t: t.replace('<spectrum index="1" id="scan=1"', "<spectrum index='1' id='scan=1'", 1),
    "two intensity arrays": lambda t: t.replace("</binaryDataArrayList>", array_xml(
        [9., 8.], "<f8", "no compression", MZ.replace("m/z array", "intensity array").replace(
            "MS:1000514", "MS:1000515")) + "</binaryDataArrayList>", 1),
}


DIRECT = {"float32 uncompressed", "integers", "empty binary", "entity title",
          "attribute order and spacing", "two intensity arrays"}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_markup_variants_equal_pyteomics(sample, variant):
    sample.write_text(VARIANTS[variant](sample.read_text()))
    assert_same_as_pyteomics(sample, 12)
    # Each variant edits one MS1 spectrum; only recognised markup stays direct.
    assert direct_count(sample, 12) == 10 - (variant not in DIRECT)


def test_truncated_file_fails_like_pyteomics(sample):
    data = sample.read_bytes()
    sample.write_bytes(data[:data.rindex(b"<binary>") + 40])
    assert_same_as_pyteomics(sample, 12)


def test_known_accessions_resolve_in_pinned_vocabulary():
    vocabulary = _local_vocabulary()
    for accession in mzml_direct._KNOWN_ACCESSIONS:
        assert vocabulary[accession].id == accession


def test_direct_reads_need_no_vocabulary_or_network(sample, monkeypatch):
    from kmzcore import psi_ms
    from psims.controlled_vocabulary import controlled_vocabulary as cv
    monkeypatch.setattr(psi_ms, "_vocabulary", None)
    monkeypatch.setattr(cv, "urlopen", lambda *_a, **_k: pytest.fail("Reader attempted network I/O"))
    with SpectrumReader(sample) as reader:
        assert reader.read(1)[0].size == 7
        assert reader._reader is None and psi_ms._vocabulary is None


def decoder_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith("kmzviewer-decode")]


def test_read_ahead_keeps_order_skips_and_joins_helpers(tmp_path):
    path = tmp_path / "many.mzML"
    rows = [{"mz": np.arange(50.) + i, "intensity": np.arange(50.) * i, "rt": i / 10} for i in range(60)]
    write_mzml(path, rows)
    reference = SpectrumReader(path)
    reference._direct = False
    reader = SpectrumReader(path)
    try:
        plan = list(range(0, 60, 2))
        reader.read_ahead(plan)
        assert decoder_threads()
        # Skipped planned spectra and unplanned reads still return exact arrays.
        for index in [0, 2, 10, 11, 12, 58, 3]:
            assert outcome(reader, index) == outcome(reference, index)
        reader.read_ahead([5, 6])
        assert outcome(reader, 6) == outcome(reference, 6)
    finally:
        reader.close()
        reference.close()
    assert not decoder_threads()


def test_read_ahead_plans_only_uncached_reads_and_cancellation_joins(sample):
    scans = load_ms1(sample)
    cache = ArrayCache()
    cache.put(str(sample), int(scans.spectrum_index[1]), (np.zeros(2), np.zeros(2)))
    with SpectrumReader(sample) as reader:
        read_ahead(reader, cache, str(sample), scans.spectrum_index)
        assert reader._plan == [int(i) for i in scans.spectrum_index if i != scans.spectrum_index[1]]
    reads = iter(range(3))
    traces = [make_tic_trace(1, "#0072b2", scans)]
    with pytest.raises(Cancelled):
        extract_filtered_values(str(sample), scans, traces, None, is_cancelled=lambda: next(reads, 3) >= 3)
    assert not decoder_threads()


def test_close_search_equals_bytearray_find():
    rng = np.random.default_rng(5)
    alphabet = np.frombuffer(b"ab</spectrum>cdinry<", dtype=np.uint8)
    for size in range(0, 400, 7):
        buffer = bytearray(rng.choice(alphabet, size).tobytes())
        for token in rng.integers(0, max(size, 1), 2):
            buffer[token:token] = rng.choice([_CLOSE, b"<binary", b"</spectru", b"<binaryD"])
        expected = buffer.find(_CLOSE)
        assert _find_close(buffer, buffer.find(b"<binary")) == expected
        for after in [i for i, byte in enumerate(buffer) if byte == ord("<")][:12]:
            assert _find_close(buffer, after) == expected


def test_palette_equals_original_greedy_maximin():
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc3377"]
    candidates = set()
    from colorsys import hsv_to_rgb
    for hue in range(0, 360, 5):
        for saturation in (.60, .80, 1.0):
            for value in (.65, .82, 1.0):
                rgb = hsv_to_rgb(hue / 360, saturation, value)
                color = "#" + "".join(f"{round(c * 255):02x}" for c in rgb)
                luminance = sum(w * _linear(c) for w, c in zip((.2126, .7152, .0722), _rgb(color)))
                if 1.05 / (luminance + .05) >= 3:
                    candidates.add(color)
    points = {color: _lab(color) for color in sorted(candidates.difference(colors))}
    while len(colors) < 24:
        selected = [_lab(color) for color in colors]
        chosen = max(points, key=lambda color: min(
            sum((a - b) ** 2 for a, b in zip(points[color], other)) for other in selected))
        colors.append(chosen)
        del points[chosen]
    assert chromatogram_palette() == tuple(colors)
