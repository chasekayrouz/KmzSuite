"""Independent review regressions: expected equivalence, including errors.

The cache and metadata regressions intentionally fail on the submitted patch.
Fixtures are synthetic and only written beneath pytest's temporary directory.
"""
import os
import threading

import numpy as np
import pytest

from kmzviewer import mzml_direct
from kmzviewer.spectra import SpectrumReader, _UnitMzML, _local_vocabulary
from Tests.Viewer.mzml_factory import _spectrum_xml, write_mzml
from Tests.Convert.test_writer_cli import synthetic


def outcome(path, baseline=False, index=0):
    with SpectrumReader(path) as reader:
        if baseline:
            reader._direct = False
        try:
            arrays = reader.read(index)
            return ('ok', tuple((a.dtype.str, a.shape, a.tobytes(), a.flags.writeable)
                                for a in arrays))
        except Exception as exc:
            return ('error', type(exc).__name__, str(exc))


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / 'review.mzML'
    write_mzml(path, [dict(mz=[100., 200.], intensity=[4., 8.], rt=.1),
                      dict(mz=[300., 400.], intensity=[9., 10.], rt=.2)])
    return path


def test_reopen_after_same_size_edit_with_preserved_mtime(tmp_path):
    path = tmp_path / 'reordered.mzML'
    a = _spectrum_xml(0, dict(id='a', mz=[100., 200.], intensity=[4., 8.], rt=.1))
    b = _spectrum_xml(1, dict(id='b', mz=[300., 400.], intensity=[9., 10.], rt=.2))
    prefix = '<mzML version="1.1.0"><run><spectrumList count="2">'
    suffix = '</spectrumList></run></mzML>'
    path.write_text(prefix + ' ' * len(b) + a + b + suffix)
    assert outcome(path) == outcome(path, True)
    stat = path.stat()
    path.write_text(prefix + b.replace('index="1"', 'index="0"')
                    + a.replace('index="0"', 'index="1"') + ' ' * len(b) + suffix)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_ino == stat.st_ino
    assert outcome(path) == outcome(path, True)


def test_saved_index_created_after_direct_cache_warmed(sample):
    assert outcome(sample) == outcome(sample, True)
    with _UnitMzML(str(sample), cv=_local_vocabulary()) as reader:
        reader._offset_index['spectrum'].move_to_end('scan=0')
        reader.write_byte_offsets()
    assert outcome(sample) == outcome(sample, True)


@pytest.mark.parametrize('replacement', [
    ('<scanList', '<userParam name="index" value="bad"/><scanList'),
    ('<scan>', '<userParam name="count" value="bad"/><scan>'),
])
def test_metadata_conversion_errors_fall_back(sample, replacement):
    sample.write_text(sample.read_text().replace(*replacement, 1))
    assert outcome(sample) == outcome(sample, True)


def test_ordinary_file_edits_and_replacement_invalidate(sample):
    assert outcome(sample) == outcome(sample, True)
    text = sample.read_text()
    sample.write_text(text.replace('id="scan=1"', 'id="scan=0"'))
    assert outcome(sample) == outcome(sample, True)
    stat = sample.stat()
    other = sample.with_name('replacement.mzML')
    other.write_text(text)
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    other.replace(sample)
    assert outcome(sample) == outcome(sample, True)


def test_inflight_helpers_join_before_descriptor_closes(sample, monkeypatch):
    entered, release, closing, finished = (threading.Event() for _ in range(4))
    actual = mzml_direct.DirectSpectra.decode
    descriptor_checks = []

    def blocked(decoder, index):
        entered.set()
        assert release.wait(5)
        descriptor_checks.append(os.fstat(decoder._fd).st_size)
        return actual(decoder, index)

    monkeypatch.setattr(mzml_direct.DirectSpectra, 'decode', blocked)

    def owner():
        with SpectrumReader(sample) as reader:
            reader.read_ahead([0, 1])
            assert entered.wait(5)
            closing.set()
        finished.set()

    thread = threading.Thread(target=owner, name='review-reader-owner')
    thread.start()
    try:
        assert closing.wait(5)
        assert not finished.is_set()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and finished.is_set()
    assert descriptor_checks
    assert not [t for t in threading.enumerate() if t.name.startswith('kmzviewer-decode')]


@pytest.mark.parametrize('variant', ['utf16', 'latin1', 'crlf', 'prefixed', 'centroid', 'none', '32bit', 'numpress', 'odd-index'])
def test_additional_format_variants_match_reference(sample, variant):
    text = sample.read_text()
    if variant == 'utf16':
        sample.write_bytes(text.replace('utf-8', 'utf-16').encode('utf-16'))
    elif variant == 'latin1':
        text = text.replace('utf-8', 'ISO-8859-1').replace('<scanList', '<userParam name="title" value="café"/><scanList', 1)
        sample.write_bytes(text.encode('latin1'))
    else:
        if variant == 'crlf': text = text.replace('\n', '\r\n')
        if variant == 'prefixed':
            text = text.replace('<mzML ', '<mzML xmlns:s="urn:review" ').replace('<spectrum ', '<s:spectrum ').replace('</spectrum>', '</s:spectrum>')
        if variant == 'centroid': text = text.replace('MS:1000128', 'MS:1000127').replace('profile spectrum', 'centroid spectrum')
        if variant == 'odd-index':
            text = text.replace('<mzML ', '<indexedmzML><mzML ', 1)
            text += '<indexList count="1"><index name="spectrum"><offset idRef="scan=0">99999999</offset></index></indexList><indexListOffset>0</indexListOffset></indexedmzML>'
        if variant in {'none', '32bit', 'numpress'}:
            import base64
            import re
            import zlib
            def convert(match):
                raw = zlib.decompress(base64.b64decode(match[1]))
                if variant == '32bit': raw = np.frombuffer(raw, dtype='<f8').astype('<f4').tobytes()
                if variant != 'none': raw = zlib.compress(raw)
                return '<binary>' + base64.b64encode(raw).decode() + '</binary>'
            text = re.sub(r'<binary>([^<]*)</binary>', convert, text)
            if variant == 'none': text = text.replace('zlib compression', 'no compression').replace('MS:1000574', 'MS:1000576')
            if variant == '32bit': text = text.replace('64-bit float', '32-bit float').replace('MS:1000523', 'MS:1000521')
            if variant == 'numpress':
                # The qualified environment lacks pynumpress: preserve its failure
                # on an unsupported declared encoding, rather than claim decoding.
                text = text.replace('zlib compression', 'MS-Numpress linear prediction compression').replace('MS:1000574', 'MS:1002312')
        sample.write_bytes(text.encode())
    for index in [0, 1, -1, 2]:
        assert outcome(sample, index=index) == outcome(sample, True, index)


@pytest.mark.parametrize('version', [1, 2, 3])
@pytest.mark.parametrize('empty', [False, True])
def test_sparse_versions_with_precursors_and_empty_arrays(synthetic, tmp_path, version, empty):
    from kmzconvert_core.writer import write_mzml as write_native
    from kmzcore.sparse_profile import FAMILY
    from kmzviewer.mzml_index import load_ms1
    synthetic.sparse_profile_sampling_family = FAMILY
    y = np.array([] if empty else [0., 10., 10., 30., 250., 300., 30., 10., 0.])
    m = (10 + np.arange(len(y)) * .0001) ** 2
    synthetic.arrays = lambda _: (m.copy(), y.copy())
    path = write_native(synthetic, tmp_path / 'sparse.mzML', profile='sparse',
                        sparse_format=f'sparse-profile-v{version}')
    scans = load_ms1(path)
    assert scans.sparse_profile[0] is not None
    for index in [0, 1]:  # Synthetic acquisition includes an MS2/precursor scan.
        assert outcome(path, index=index) == outcome(path, True, index)


def test_array_buffers_are_writable_independent_and_survive_close(sample):
    for baseline in [False, True]:
        with SpectrumReader(sample) as reader:
            if baseline: reader._direct = False
            first = reader.read(0)
            second = reader.read(0)
        for a, b in zip(first, second):
            assert a.flags.writeable and b.flags.writeable
            assert not np.shares_memory(a, b)
            a[:] = -1
            assert not np.all(b == -1)


def test_shared_index_retention_is_bounded(tmp_path):
    for i in range(12):
        path = tmp_path / f'bound-{i}.mzML'
        write_mzml(path, [dict(mz=[100.], intensity=[4.], rt=.1)])
        assert outcome(path) == outcome(path, True)
    assert len(mzml_direct._indexes) <= mzml_direct._INDEX_LIMIT
