"""MS2 streaming scan helpers must equal the plain find/regex searches they replace."""

import random

import pytest

import kmz2viewer.index as index
from Tests.Viewer.mzml_factory import write_mzml

PIECES = ["<", "</spectrum>", "</spectrum", "<spectrum ", "<spectrumList", "<binary>", "<binaryDataArray", "</binaryDataArray>",
          "<binaryDataArrayList>", "<binaryDataArray_", "<binaryDataArrayé", "<binaryDataArray١", "<binaryDataArray>",
          "eJzt", "QUJD+/==", " ", "\n", "a", ">"]


def test_find_hops_equal_bytearray_find():
    rng = random.Random(1)
    for _ in range(400):
        buf = bytearray("".join(rng.choice(PIECES) for _ in range(rng.randrange(40))).encode())
        for token in (index._CLOSE, index._OPEN, b"<binary"):
            for start in range(len(buf) + 2):
                assert index._find(buf, token, start) == buf.find(token, start)


def test_array_bodies_equal_regex_matches():
    rng = random.Random(2)
    for _ in range(3000):
        block = "".join(rng.choice(PIECES) for _ in range(rng.randrange(30)))
        assert list(index._array_bodies(block)) == [m.group(0) for m in index._ARRAY.finditer(block)]


def _spectra(rng):
    spectra = []
    for i in range(40):
        peaks = rng.randrange(0, 40)
        mz = sorted(rng.uniform(50, 900) for _ in range(peaks))
        spectra.append({"id": f"s{i}", "ms_level": 2 if i == 7 else rng.choice((1, 1, 2)), "mz": mz,
                        "intensity": [round(rng.uniform(0, 1e5), 3) for _ in range(peaks)], "rt": i * .1,
                        "tic": None, "centroid": rng.random() < .5, "precursor_mz": 300.0 + i})
    return spectra


@pytest.mark.parametrize("chunk", [17, 64, 211, 4096, 4 * 1024 * 1024])
def test_scan_matches_plain_find_and_regex_at_any_chunk_size(tmp_path, monkeypatch, chunk):
    path = tmp_path / "mixed.mzML"
    write_mzml(path, _spectra(random.Random(3)), chromatogram=True)
    text = path.read_text()
    opener = text.index(">", text.index('<spectrum index="5"')) + 1  # '<binary' inside a header comment
    text = text[:opener] + "<!-- <binaryDataArrayList> -->" + text[opener:]
    start = text.index("<binaryDataArrayList", text.index('<spectrum index="7"'))  # no arrays at all
    path.write_text(text[:start] + text[text.index("</binaryDataArrayList>", start) + 21:])
    new = index.index_ms2(path, chunk_size=chunk)
    monkeypatch.setattr(index, "_find", lambda buf, token, start=0: buf.find(token, start))
    monkeypatch.setattr(index, "_array_bodies", lambda block: (m.group(0) for m in index._ARRAY.finditer(block)))
    old = index.index_ms2(path, chunk_size=chunk)
    assert len(new) == len(old) > 0
    for a, b in zip(new, old):
        assert {k: v for k, v in vars(a).items() if k not in ("mz", "intensity")} == \
               {k: v for k, v in vars(b).items() if k not in ("mz", "intensity")}
        assert a.mz.tobytes() == b.mz.tobytes() and a.intensity.tobytes() == b.intensity.tobytes()
