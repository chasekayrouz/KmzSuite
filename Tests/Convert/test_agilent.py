"""Agilent MassHunter .d reader, decoders and CLI on synthetic bundles (no vendor data)."""
import json
import struct

import numpy as np
import pytest
from pyteomics import mzml

from kmzconvert.cli import main
from kmzconvert_core import agilent
from kmzconvert_core.agilent import AgilentReader, Calibration, decode_profile, lzf_decompress, rle_decode
from kmzconvert_core.binary import FormatError
from kmzconvert_core.formats import AGILENT, WIFF, detect

XSD = """<?xml version="1.0" encoding="utf-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:complexType name="ScanRecordType"><xs:sequence>
    <xs:element name="ScanID" type="xs:int"/><xs:element name="ScanMethodID" type="xs:int"/>
    <xs:element name="TimeSegmentID" type="xs:int"/><xs:element name="ScanTime" type="xs:double"/>
    <xs:element name="MSLevel" type="xs:int"/><xs:element name="ScanType" type="xs:int"/>
    <xs:element name="TIC" type="xs:double"/><xs:element name="BasePeakMZ" type="xs:double"/>
    <xs:element name="BasePeakValue" type="xs:double"/><xs:element name="CalibrationID" type="xs:int" minOccurs="0"/>
    <xs:element name="IonPolarity" type="xs:int" minOccurs="0"/><xs:element name="CollisionEnergy" type="xs:double" minOccurs="0"/>
    <xs:element name="MzOfInterest" type="xs:double" minOccurs="0"/><xs:element name="ChargeState" type="xs:int" minOccurs="0"/>
    <xs:element name="MassCalOffset" type="xs:long" minOccurs="0"/>
    <xs:element name="DataDependentScanParamType" type="DataDependentScanParamType" minOccurs="0"/>
    <xs:element name="SpectrumParamValues" type="SpectrumParamsType" maxOccurs="unbounded"/>
  </xs:sequence></xs:complexType>
  <xs:complexType name="SpectrumParamsType"><xs:sequence>
    <xs:element name="SpectrumFormatID" type="xs:int"/><xs:element name="SpectrumOffset" type="xs:long"/>
    <xs:element name="ByteCount" type="xs:int"/><xs:element name="PointCount" type="xs:int"/>
    <xs:element name="UncompressedByteCount" type="xs:int" minOccurs="0"/>
    <xs:element name="MinX" type="xs:double" minOccurs="0"/><xs:element name="MaxX" type="xs:double" minOccurs="0"/>
  </xs:sequence></xs:complexType>
  <xs:complexType name="DataDependentScanParamType"><xs:sequence>
    <xs:element name="DDScanID" type="xs:int"/><xs:element name="DDScanParamMz" type="xs:double"/>
  </xs:sequence></xs:complexType>
</xs:schema>
"""
RECORD = np.dtype([("ScanID", "<i4"), ("ScanMethodID", "<i4"), ("TimeSegmentID", "<i4"), ("ScanTime", "<f8"),
                   ("MSLevel", "<i4"), ("ScanType", "<i4"), ("TIC", "<f8"), ("BasePeakMZ", "<f8"),
                   ("BasePeakValue", "<f8"), ("CalibrationID", "<i4"), ("IonPolarity", "<i4"),
                   ("CollisionEnergy", "<f8"), ("MzOfInterest", "<f8"), ("ChargeState", "<i4"), ("MassCalOffset", "<i8"),
                   ("DDScanID", "<i4"), ("DDScanParamMz", "<f8")]
                  + [(f"b{i}_{n}", t) for i in range(2) for n, t in
                     (("format", "<i4"), ("offset", "<i8"), ("bytes", "<i4"), ("points", "<i4"),
                      ("uncompressed", "<i4"), ("minx", "<f8"), ("maxx", "<f8"))])
# MassHunter B.06 schemas declare ChromScaleFactor as xs:double; its records hold a 4-byte float.
XSD_CHROM_SCALE = XSD.replace('<xs:element name="MassCalOffset"',
                              '<xs:element name="ChromScaleFactor" type="xs:double" minOccurs="0"/><xs:element name="MassCalOffset"')


def record_dtype(chrom_scale=None):
    if chrom_scale is None:
        return RECORD
    fields = []
    for name, kind in RECORD.descr:
        if name == "MassCalOffset":
            fields.append(("ChromScaleFactor", chrom_scale))
        fields.append((name, kind))
    return np.dtype(fields)


TRADITIONAL = (3.5e-4, 1005.0)
POLYNOMIAL = (30000.0, 120000.0, 2e-3, -1e-7)
FLAGS = 3  # powers 0 and 1


def default_cal():
    return Calibration(TRADITIONAL[0], TRADITIONAL[1], POLYNOMIAL[0], POLYNOMIAL[1], POLYNOMIAL[2:], (0, 1))


def lzf_literals(data):
    """A valid raw LZF stream of literal runs only."""
    out = bytearray()
    for i in range(0, len(data), 32):
        chunk = data[i:i + 32]
        out += bytes([len(chunk) - 1]) + chunk
    return bytes(out)


def rle_block(x0, dx, y):
    """Run-length block with 4-byte values; zero runs become skip controls (width code 2 keeps 4 bytes)."""
    first = int(np.flatnonzero(y)[0])
    stream, pending = bytearray(), 0
    for value in y[first:]:
        if value == 0:
            pending += 1
            continue
        if pending:
            stream += struct.pack("<i", -((pending << 2) | 2) - 1)
            pending = 0
        stream += struct.pack("<i", int(value))
    return struct.pack("<ddIi", x0, dx, (0x90 << 24) | len(y), -first) + bytes(stream)


def make_bundle(root, name="Run.d", *, encoding="lzf", device="QTOF", per_scan_cal=False, tamper=None, extra=(),
                cal_offsets=None, levels=(1, 2, 1), centroid=False, chrom_scale=None):
    """A synthetic bundle; ``centroid`` stores only MSPeak.bin centroids (empty MSProfile.bin) and
    ``chrom_scale`` adds a ChromScaleFactor record field of that type, declared xs:double."""
    bundle = root / name
    acq = bundle / "AcqData"
    acq.mkdir(parents=True)
    points, x0, dx = 400, 30000.0, 0.5
    scans = [  # (id, rt, level, scan type, polarity, ce, precursor, charge, peaks)
        (11, 0.10, 1, 1, 1, 0.0, 0.0, 0, {50: 900, 51: 1500, 52: 700, 200: 40000}),
        (12, 0.11, 2, 512, 1, 20.0, 412.25, 2, {120: 300, 121: 25, 300: 5}),
        (13, 0.20, 1, 1, 1, 0.0, 0.0, 0, {60: 100, 61: 200}),
    ]
    profile = bytearray(struct.pack("<I", 258) + bytes(64))
    peak = bytearray(struct.pack("<I", 259) + bytes(64))
    records, cal = [], default_cal()
    scans = [s for s, keep in zip(scans, levels) if keep] if levels != (1, 2, 1) else scans
    for index, (sid, rt, level, kind, polarity, ce, precursor, charge, peaks) in enumerate(scans):
        y = np.zeros(points, dtype=np.int64)
        for k, v in peaks.items():
            y[k] = v
        if encoding == "lzf":
            block = lzf_literals(struct.pack("<dd", x0, dx) + y.astype("<i4").tobytes())
        else:
            block = rle_block(x0, dx, y)
        top = int(np.argmax(y))
        bp_mz = float(cal.mz(np.array([x0 + top * dx]))[0])
        r = np.zeros((), record_dtype(chrom_scale))
        t_axis = x0 + dx * np.arange(points)
        values = dict(ScanID=sid, ScanMethodID=1, TimeSegmentID=1, ScanTime=rt, MSLevel=level, ScanType=kind,
                      TIC=float(y.sum()), BasePeakMZ=bp_mz, BasePeakValue=float(y.max()), CalibrationID=1,
                      IonPolarity=polarity, CollisionEnergy=ce, MzOfInterest=precursor, ChargeState=charge,
                      MassCalOffset={"reverse": 72 + 84 * (len(scans) - 1 - index), "zero": 0, None: 72 + 84 * index}[cal_offsets]
                      if per_scan_cal or cal_offsets == "reverse" else 0,
                      DDScanID=sid, b0_format=1, b0_offset=len(profile), b0_bytes=len(block), b0_points=points,
                      b0_uncompressed=16 + 4 * points, b0_minx=float(cal.mz(t_axis[:1])[0]),
                      b0_maxx=float(cal.mz(t_axis[-1:])[0]), b1_format=2)
        if chrom_scale is not None:
            values["ChromScaleFactor"] = 1.0
        if centroid:
            at = np.array(sorted(peaks))
            centroids = (x0 + dx * at).astype("<f8").tobytes() + np.array([peaks[k] for k in at], "<f4").tobytes()
            mz = cal.mz(x0 + dx * at)
            values.update(b0_format=2, b0_offset=len(peak), b0_bytes=len(centroids), b0_points=len(at),
                          b0_uncompressed=0, b0_minx=float(mz[0]), b0_maxx=float(mz[-1]), b1_format=0)
            peak += centroids
        for k, v in values.items():
            r[k] = v
        if tamper:
            tamper(sid, r)
        profile += block
        records.append(r.tobytes())
    header = bytearray(struct.pack("<I", 257) + bytes(0x60 - 4))
    struct.pack_into("<I", header, 0x58, len(header))
    (acq / "MSScan.bin").write_bytes(bytes(header) + b"".join(records))
    (acq / "MSScan.xsd").write_text(XSD if chrom_scale is None else XSD_CHROM_SCALE)
    (acq / "MSProfile.bin").write_bytes(b"" if centroid else bytes(profile))
    if centroid:
        (acq / "MSPeak.bin").write_bytes(bytes(peak))
    (acq / "MSTS.xml").write_text(f"<TimeSegments><TimeSegment TimeSegmentID=\"1\"><NumOfScans>{len(scans)}</NumOfScans></TimeSegment></TimeSegments>")
    (acq / "Devices.xml").write_text(f"<Devices><Device DeviceID=\"1\"><Name>{device}</Name><ModelNumber>G0000X</ModelNumber>"
                                     "<SerialNumber>SYNTH</SerialNumber></Device></Devices>")
    (acq / "Contents.xml").write_text("<Contents><AcqSoftwareVersion>Synthetic 1.0</AcqSoftwareVersion></Contents>")
    values = "".join(f'<Value Number="{i + 1}">{v!r}</Value>' for i, v in enumerate(POLYNOMIAL + (0.0,) * 4))
    (acq / "DefaultMassCal.xml").write_text(
        '<DefaultMassCalibration><DefaultCalibrations><DefaultCalibration DefaultCalibrationID="1">'
        f'<Step Number="1"><CalibrationFormula>Traditional</CalibrationFormula><ValueUseFlags>0</ValueUseFlags>'
        f'<Values><Value Number="1">{TRADITIONAL[0]!r}</Value><Value Number="2">{TRADITIONAL[1]!r}</Value></Values></Step>'
        f'<Step Number="2"><CalibrationFormula>Polynomial</CalibrationFormula><ValueUseFlags>{FLAGS}</ValueUseFlags>'
        f'<Values>{values}</Values></Step></DefaultCalibration></DefaultCalibrations></DefaultMassCalibration>')
    if per_scan_cal:
        rows = b"".join(struct.pack("<i10d", 10, TRADITIONAL[0] * (1 + 1e-6 * i), TRADITIONAL[1], *POLYNOMIAL, 0, 0, 0, 0)
                        for i in range(3))
        (acq / "MSMassCal.bin").write_bytes(struct.pack("<I", 260) + bytes(64) + struct.pack("<I", 1) + rows)
    for item in extra:
        (acq / item).write_bytes(b"")
    return bundle


def test_lzf_reference_and_native_agree_on_published_vector_and_backreferences():
    vector = bytes.fromhex("07000000 0080d6d4 40200620 0001e03f".replace(" ", ""))
    assert struct.unpack("<dd", lzf_decompress(vector, 16)) == (21338.0, 0.5)
    overlapping = bytes([2]) + b"abc" + bytes([(1 << 5) | 0, 2])  # literal "abc", then copy 3 from distance 3
    assert lzf_decompress(overlapping, 6) == b"abcabc"
    long = bytes([0]) + b"z" + bytes([7 << 5, 10, 0])  # length 7+10+2 = 19 repeats of the previous byte
    assert lzf_decompress(long, 20) == b"z" * 20
    for bad, size, message in [(bytes([5]) + b"ab", 6, "Truncated"), (bytes([1 << 5, 5]), 3, "Invalid"),
                               (bytes([1]) + b"ab", 3, "shorter"), (bytes([3]) + b"abcd", 2, "exceeds")]:
        with pytest.raises(FormatError, match=message):
            lzf_decompress(bad, size)
        if agilent._agilent is not None:
            with pytest.raises(ValueError, match=message):
                agilent._agilent.lzf_decompress(bad, size)
    if agilent._agilent is not None:
        for data, size in [(vector, 16), (overlapping, 6), (long, 20)]:
            assert agilent._agilent.lzf_decompress(data, size) == lzf_decompress(data, size)


def test_run_length_reference_and_native_agree_on_published_prefix():
    # From the public format notes: first point 16872, width changes and zero skips.
    stream = bytes.fromhex("ffffffff 04 df 06 ef 01 16 19 fe 8300 ffff 4e".replace(" ", ""))
    y = rle_decode(stream, 16872, 17000)
    nonzero = {int(i): int(y[i]) for i in np.flatnonzero(y)}
    assert nonzero == {16872: 4, 16881: 6, 16886: 1, 16887: 22, 16888: 25, 16889: 131, 16890: 78}
    if agilent._agilent is not None:
        native = np.frombuffer(agilent._agilent.rle_decode(stream, 16872, 17000), dtype=np.int64)
        assert np.array_equal(native, y)
    for bad, start, count, message in [(b"\x05\x00", 0, 4, "Truncated"), (b"\x01\x00\x00\x00", 4, 4, "exceeds"),
                                       (b"", 5, 4, "start")]:
        with pytest.raises(FormatError, match=message):
            rle_decode(bad, start, count)
        if agilent._agilent is not None:
            with pytest.raises(ValueError, match=message):
                agilent._agilent.rle_decode(bad, start, count)


@pytest.mark.parametrize("encoding", ["lzf", "rle"])
@pytest.mark.parametrize("native", [True, False])
def test_reader_decodes_calibrates_and_guards_both_encodings(tmp_path, monkeypatch, encoding, native):
    if not native:
        monkeypatch.setattr(agilent, "_agilent", None)
    reader = AgilentReader(make_bundle(tmp_path, encoding=encoding))
    assert reader.profile_encoding() == ("LZF" if encoding == "lzf" else "run-length")
    assert [s.native_id for s in reader.scans] == ["scanId=11", "scanId=12", "scanId=13"]
    assert [reader.experiments[s.experiment - 1].polarity for s in reader.scans] == ["negative"] * 3
    ms2 = reader.scans[1]
    assert (ms2.ms_level, ms2.precursor_mz, ms2.precursor_charge) == (2, 412.25, 2)
    assert reader.experiments[ms2.experiment - 1].collision_energy == 20.0
    mz, y = reader.arrays(reader.scans[0])
    t = 30000.0 + 0.5 * np.array([49, 50, 51, 52, 53, 199, 200, 201])
    assert np.array_equal(y, [0, 900, 1500, 700, 0, 0, 40000, 0])
    assert np.array_equal(mz, default_cal().mz(t))
    times, totals, levels = reader.tic_arrays()
    assert np.array_equal(times, [0.10, 0.11, 0.20]) and list(levels) == [1, 2, 1] and totals[0] == 43100
    reader.close()


def test_per_scan_calibration_is_used_for_mz_and_default_for_base_peak_guard(tmp_path):
    reader = AgilentReader(make_bundle(tmp_path, per_scan_cal=True))
    assert reader.calibration_source == "MSMassCal.bin"
    scan = reader.scans[2]
    mz, _ = reader.arrays(scan)
    expected = Calibration(3.5e-4 * (1 + 2e-6), 1005.0, *POLYNOMIAL[:2], POLYNOMIAL[2:], (0, 1))
    assert np.array_equal(mz, expected.mz(30000.0 + 0.5 * np.array([59, 60, 61, 62])))
    reader.close()


@pytest.mark.parametrize("field, value, message", [
    ("TIC", 1.0, "TIC"), ("BasePeakValue", 1.0, "base peak intensity"), ("BasePeakMZ", 999.0, "base-peak m/z"),
    ("BasePeakMZ", -5.0, "Invalid stored base-peak m/z"),
])
def test_summary_disagreement_is_refused(tmp_path, field, value, message):
    def tamper(sid, record):
        if sid == 13:
            record[field] = value
    reader = AgilentReader(make_bundle(tmp_path, tamper=tamper))
    with pytest.raises(FormatError, match=message):
        reader.arrays(reader.scans[2])
    reader.close()


@pytest.mark.parametrize("kwargs, message", [
    (dict(device="TandemQuadrupole"), "Only Agilent TOF and Q-TOF"),
    (dict(extra=("IMSFrame.bin",)), "ion-mobility"),
    (dict(tamper=lambda sid, r: r.__setitem__("b0_format", 2)), "centroid-only"),
    (dict(tamper=lambda sid, r: r.__setitem__("CollisionEnergy", 5.0) if sid == 11 else None), "All Ions"),
    (dict(tamper=lambda sid, r: r.__setitem__("ScanType", 4) if sid == 13 else None), "scan type"),
    (dict(tamper=lambda sid, r: r.__setitem__("ScanTime", 0.01) if sid == 13 else None), "Retention times decrease"),
])
def test_unsupported_bundles_are_refused(tmp_path, kwargs, message):
    with pytest.raises(FormatError, match=message):
        AgilentReader(make_bundle(tmp_path, **kwargs))


def test_bundle_without_profile_or_centroid_data_is_refused(tmp_path):
    bundle = make_bundle(tmp_path)
    (bundle / "AcqData/MSProfile.bin").unlink()
    with pytest.raises(FormatError, match="neither profile"):
        AgilentReader(bundle)
    bundle = make_bundle(tmp_path / "centroid", centroid=True)
    (bundle / "AcqData/MSPeak.bin").unlink()
    with pytest.raises(FormatError, match="neither profile"):
        AgilentReader(bundle)


@pytest.mark.parametrize("centroid", [False, True])
@pytest.mark.parametrize("chrom_scale", ["<f4", "<f8"])
def test_chrom_scale_factor_may_be_stored_as_float_despite_the_schema(tmp_path, centroid, chrom_scale):
    reader = AgilentReader(make_bundle(tmp_path, centroid=centroid, chrom_scale=chrom_scale))
    assert reader.records.dtype["ChromScaleFactor"] == np.dtype(chrom_scale)
    assert (reader.records["ChromScaleFactor"] == 1).all()
    assert [s.precursor_mz for s in reader.scans] == [None, 412.25, None]
    for scan in reader.scans:
        reader.arrays(scan)
    reader.close()


def test_narrowed_chrom_scale_factor_must_read_as_one(tmp_path):
    # Another four-byte disagreement with the schema must not pass as the B.06 float field.
    def scaled(sid, record):
        record["ChromScaleFactor"] = 0.5
    with pytest.raises(FormatError, match="record size"):
        AgilentReader(make_bundle(tmp_path, chrom_scale="<f4", tamper=scaled))
    AgilentReader(make_bundle(tmp_path / "double", chrom_scale="<f8", tamper=scaled)).close()


def test_centroid_only_bundle_yields_the_stored_centroids(tmp_path):
    reader = AgilentReader(make_bundle(tmp_path, centroid=True))
    assert reader.centroided and reader.limitations == agilent.CENTROID_LIMITATIONS
    assert "MSPeak.bin" in reader.files and "MSProfile.bin" not in reader.files
    mz, y = reader.arrays(reader.scans[0])
    assert np.array_equal(y, [900, 1500, 700, 40000])
    assert np.array_equal(mz, default_cal().mz(30000.0 + 0.5 * np.array([50, 51, 52, 200])))
    assert all(reader.experiments[s.experiment - 1].lower_mz is None for s in reader.scans)
    inspected = reader.inspect()
    assert inspected["spectra_representation"] == "centroid" and inspected["profile_encoding"] is None
    reader.close()


def test_run_without_kept_centroids_converts_to_empty_spectra(tmp_path, capsys):
    def unkept(sid, record):
        record["b0_points"], record["b0_bytes"], record["b0_offset"] = 0, 0, 0
    bundle = make_bundle(tmp_path, centroid=True, tamper=unkept)
    path = bundle / "AcqData/MSPeak.bin"
    path.write_bytes(path.read_bytes()[:68])  # header only
    output = tmp_path / "empty.mzML"
    assert main([str(bundle), "-o", str(output)]) == 0
    spectra = list(mzml.MzML(str(output)))
    assert [len(s["m/z array"]) for s in spectra] == [0, 0, 0]
    assert [s["total ion current"] for s in spectra] == [43100, 330, 300]


@pytest.mark.parametrize("tamper, message", [
    (lambda r: r.__setitem__("b0_offset", 68), "out of bounds"),
    (lambda r: r.__setitem__("TIC", float("nan")), "Invalid stored TIC"),
    (lambda r: r.__setitem__("TIC", -1.0), "Invalid stored TIC"),
    (lambda r: r.__setitem__("BasePeakValue", 1e9), "Invalid stored TIC"),
])
def test_empty_centroid_block_must_be_the_empty_sentinel(tmp_path, tamper, message):
    def damage(sid, record):
        if sid == 13:
            record["b0_points"], record["b0_bytes"], record["b0_offset"] = 0, 0, 0
            tamper(record)
    with pytest.raises(FormatError, match=message):
        AgilentReader(make_bundle(tmp_path, centroid=True, tamper=damage))


def test_empty_centroid_block_with_stored_tic_is_an_empty_spectrum(tmp_path):
    def unkept(sid, record):
        if sid == 13:
            record["b0_points"], record["b0_bytes"], record["b0_offset"] = 0, 0, 0
    bundle = make_bundle(tmp_path, centroid=True, tamper=unkept)
    path = bundle / "AcqData/MSPeak.bin"
    with pytest.raises(FormatError, match="outside the scans' centroid blocks"):
        AgilentReader(bundle)  # scan 13's centroids are still in the file
    path.write_bytes(path.read_bytes()[:-24])  # its two centroids, 12 bytes each
    reader = AgilentReader(bundle)
    scan = reader.scans[2]
    mz, y = reader.arrays(scan)
    assert len(mz) == len(y) == 0 and scan.tic == 300
    reader.close()


def test_overlapping_centroid_blocks_are_refused(tmp_path):
    def overlap(sid, record):
        if sid == 13:
            record["b0_offset"] -= 12
    with pytest.raises(FormatError, match="overlap or leave gaps"):
        AgilentReader(make_bundle(tmp_path, centroid=True, tamper=overlap))


def test_missing_profile_file_is_named_when_centroids_do_not_reproduce_summaries(tmp_path):
    def profile_scan(sid, record):
        record["b1_format"], record["b1_bytes"] = 1, 100  # points at profile data the folder lacks
        if sid == 13:
            record["TIC"] = 999.0
    reader = AgilentReader(make_bundle(tmp_path, centroid=True, tamper=profile_scan))
    reader.arrays(reader.scans[0])
    with pytest.raises(FormatError, match="MSProfile.bin is absent or empty although the scans reference profile"):
        reader.arrays(reader.scans[2])
    reader.close()


@pytest.mark.parametrize("tamper, message", [
    (lambda r: r.__setitem__("TIC", 1.0), "TIC"),
    (lambda r: r.__setitem__("BasePeakValue", 1.0), "base peak intensity"),
    (lambda r: r.__setitem__("BasePeakMZ", r["BasePeakMZ"] * (1 + 1e-7)), "base-peak m/z"),
    (lambda r: r.__setitem__("b0_bytes", r["b0_bytes"] + 4), "centroid block size"),
    (lambda r: r.__setitem__("b0_offset", 10 ** 6), "out of bounds"),
    (lambda r: r.__setitem__("b0_format", 1), "no centroid spectrum"),
])
def test_centroid_guards(tmp_path, tamper, message):
    def damage(sid, record):
        if sid == 13:
            tamper(record)
    with pytest.raises(FormatError, match=message):
        reader = AgilentReader(make_bundle(tmp_path, centroid=True, tamper=damage))
        reader.arrays(reader.scans[2])


def test_centroid_base_peak_tolerance(tmp_path):
    def shift(sid, record):
        if sid == 13:
            record["BasePeakMZ"] *= 1 + 3e-8  # 0.03 ppm: within the centroid tolerance only
    reader = AgilentReader(make_bundle(tmp_path, centroid=True, tamper=shift))
    reader.arrays(reader.scans[2])
    reader.close()
    reader = AgilentReader(make_bundle(tmp_path / "profile", tamper=shift))
    with pytest.raises(FormatError, match="base-peak m/z"):
        reader.arrays(reader.scans[2])
    reader.close()


def test_mass_cal_offsets_select_records_and_missing_file_is_refused(tmp_path):
    # Offsets address the per-scan records in reverse order: scan 13 uses record 0 (A unchanged).
    reader = AgilentReader(make_bundle(tmp_path, per_scan_cal=True, cal_offsets="reverse"))
    assert reader.calibration(reader.scans[2]).a == 3.5e-4 and reader.calibration(reader.scans[0]).a == 3.5e-4 * (1 + 2e-6)
    reader.close()
    bundle = make_bundle(tmp_path / "missing", cal_offsets="reverse")
    with pytest.raises(FormatError, match="MSMassCal.bin is missing"):
        AgilentReader(bundle)
    bundle = make_bundle(tmp_path / "unreferenced", per_scan_cal=True, cal_offsets="zero")
    with pytest.raises(FormatError, match="no scan references"):
        AgilentReader(bundle)


def test_calibration_cache_is_keyed_by_calibration_id(tmp_path):
    # Scans 11 and 13 share identical MSMassCal.bin rows but name calibrations with different powers.
    def tamper(sid, record):
        if sid == 13:
            record["CalibrationID"] = 2
    bundle = make_bundle(tmp_path, per_scan_cal=True, tamper=tamper)
    rows = bundle / "AcqData/MSMassCal.bin"
    data = bytearray(rows.read_bytes())
    data[72 + 2 * 84:72 + 3 * 84] = data[72:72 + 84]
    rows.write_bytes(bytes(data))
    path = bundle / "AcqData/DefaultMassCal.xml"
    text = path.read_text()
    entry = text[text.index("<DefaultCalibration "):text.index("</DefaultCalibration>") + len("</DefaultCalibration>")]
    second = entry.replace('DefaultCalibrationID="1"', 'DefaultCalibrationID="2"').replace(
        f"<ValueUseFlags>{FLAGS}</ValueUseFlags>", "<ValueUseFlags>6</ValueUseFlags>")
    path.write_text(text.replace(entry, entry + second))
    reader = AgilentReader(bundle)
    assert reader.calibration(reader.scans[0]).powers == (0, 1)
    assert reader.calibration(reader.scans[2]).powers == (1, 2)
    reader.close()


def test_ms1_only_output_declares_only_ms1_and_writer_refuses_unqualified_modes(tmp_path):
    from kmzconvert_core.writer import write_mzml
    bundle = make_bundle(tmp_path, levels=(1, 0, 1))
    reader = AgilentReader(bundle)
    output = write_mzml(reader, tmp_path / "ms1.mzML")
    head = output.read_text()[:4000]
    assert 'name="MS1 spectrum"' in head and 'name="MSn spectrum"' not in head.split("</fileContent>")[0]
    for kwargs in (dict(profile="sparse"), dict(mass_correction=object())):
        with pytest.raises(ValueError, match="Full Profile"):
            write_mzml(reader, tmp_path / "other.mzML", **kwargs)
    with pytest.raises(ValueError, match="inside the source acquisition"):
        write_mzml(reader, bundle / "AcqData/out.mzML")
    assert not (bundle / "AcqData/out.mzML").exists()
    reader.close()


def test_detection(tmp_path):
    bundle = make_bundle(tmp_path)
    assert detect(bundle) == AGILENT and detect(tmp_path / "x.wiff") == WIFF and detect(tmp_path / "x.WIFF2") == WIFF
    (tmp_path / "Other.d").mkdir()
    (tmp_path / "file.d").write_bytes(b"")
    for path, message in [(tmp_path / "Other.d", "not an Agilent MassHunter"), (tmp_path / "file.d", "not a folder"),
                          (tmp_path / "x.mzML", "Input must be")]:
        with pytest.raises(FormatError, match=message):
            detect(path)


@pytest.mark.parametrize("encoding", ["lzf", "rle"])
def test_cli_converts_agilent_bundle_to_valid_indexed_mzml(tmp_path, capsys, encoding):
    bundle = make_bundle(tmp_path, encoding=encoding)
    output = tmp_path / "out.mzML"
    assert main([str(bundle), "-o", str(output)]) == 0
    assert "Wrote" in capsys.readouterr().out
    spectra = list(mzml.MzML(str(output)))
    assert [s["id"] for s in spectra] == ["scanId=11", "scanId=12", "scanId=13"]
    assert all("negative scan" in s and "profile spectrum" in s for s in spectra)
    ion = spectra[1]["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
    assert ion["selected ion m/z"] == 412.25 and ion["charge state"] == 2
    assert spectra[0]["scanList"]["scan"][0]["scan start time"] == 0.10
    text = output.read_text()
    assert 'accession="MS:1001509"' in text and 'accession="MS:1000490"' in text and 'defaultSourceFileRef="MSScan"' in text
    assert "WIFF" not in text


def test_cli_agilent_list_inspect_and_refusals(tmp_path, capsys):
    bundle = make_bundle(tmp_path)
    assert main([str(bundle), "--list-samples"]) == 0
    assert json.loads(capsys.readouterr().out) == {"samples": [{"id": 1}]}
    assert main([str(bundle), "--inspect"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["spectra"] == 3 and inspected["ms_levels"] == {"1": 2, "2": 1}
    for args, message in [(["--profile", "sparse"], "Sparse Profile is not yet qualified"),
                          (["--sample", "1"], "omit --sample"),
                          (["--mass-reference", "100;1"], "mass correction is not yet qualified")]:
        assert main([str(bundle), "-o", str(tmp_path / "x.mzML"), *args]) == 1
        err = capsys.readouterr().err
        assert message in err and "Traceback" not in err
    assert not (tmp_path / "x.mzML").exists() and not (tmp_path / "x.sp.mzML").exists()


@pytest.mark.parametrize("change, message", [
    (('<xs:element name="ScanID" type="xs:int"/>', '<xs:choice/><xs:element name="ScanID" type="xs:int"/>'), "construct"),
    (('<xs:element name="SpectrumParamValues" type="SpectrumParamsType" maxOccurs="unbounded"/>',
      '<xs:element name="SpectrumParamValues" type="SpectrumParamsType" maxOccurs="unbounded"/>'
      '<xs:element name="Late" type="xs:int"/>'), "follow the spectrum blocks"),
])
def test_unsupported_schema_constructs_are_refused(tmp_path, change, message):
    bundle = make_bundle(tmp_path)
    xsd = bundle / "AcqData/MSScan.xsd"
    xsd.write_text(xsd.read_text().replace(*change))
    with pytest.raises(FormatError, match=message):
        AgilentReader(bundle)


def _rewrite(path, old, new):
    path.write_text(path.read_text().replace(old, new))


@pytest.mark.parametrize("damage, message", [
    (lambda b: _rewrite(b / "AcqData/MSScan.xsd", 'name="ChargeState" type="xs:int"', 'name="ChargeState" type="xs:decimal"'),
     "older MassHunter"),
    (lambda b: _rewrite(b / "AcqData/MSTS.xml", "<NumOfScans>3<", "<NumOfScans>4<"), "scan count|record size"),
    (lambda b: _rewrite(b / "AcqData/Devices.xml", "</Devices>",
                        '<Device DeviceID="2"><Name>TOF</Name></Device></Devices>'), "More than one"),
    (lambda b: (b / "AcqData/Devices.xml").unlink(), "Devices.xml is missing"),
    (lambda b: _rewrite(b / "AcqData/DefaultMassCal.xml", '<ValueUseFlags>3</ValueUseFlags>', ''), "Malformed DefaultMassCal.xml"),
    (lambda b: _rewrite(b / "AcqData/DefaultMassCal.xml", '<Value Number="3">', '<Value Number="3">x'), "Malformed DefaultMassCal.xml"),
    (lambda b: (b / "AcqData/MSMassCal.bin").write_bytes((b / "AcqData/MSMassCal.bin").read_bytes()[:-84]),
     "does not address"),
    (lambda b: (b / "AcqData/MSProfile.bin").write_bytes((b / "AcqData/MSProfile.bin").read_bytes()[:-10]), "out of bounds"),
])
def test_damaged_bundles_are_refused_cleanly(tmp_path, damage, message):
    bundle = make_bundle(tmp_path, per_scan_cal=True)
    damage(bundle)
    with pytest.raises(FormatError, match=message):
        AgilentReader(bundle)


def test_per_scan_polynomial_must_match_default(tmp_path):
    bundle = make_bundle(tmp_path, per_scan_cal=True)
    path = bundle / "AcqData/MSMassCal.bin"
    data = bytearray(path.read_bytes())
    struct.pack_into("<d", data, 72 + 4 + 8 * 4, 9.0)  # first polynomial coefficient of record 0
    path.write_bytes(bytes(data))
    with pytest.raises(FormatError, match="polynomial differs"):
        AgilentReader(bundle)


def test_cli_writes_centroid_only_bundle_as_centroid_spectra(tmp_path, capsys):
    bundle = make_bundle(tmp_path, centroid=True, chrom_scale="<f4")
    output = tmp_path / "centroid.mzML"
    assert main([str(bundle), "-o", str(output)]) == 0
    captured = capsys.readouterr()
    assert "stored centroids" in captured.err and "Wrote" in captured.out
    spectra = list(mzml.MzML(str(output)))
    assert all("centroid spectrum" in s and "profile spectrum" not in s for s in spectra)
    assert all("scanWindowList" not in s["scanList"]["scan"][0] for s in spectra)
    assert list(spectra[0]["intensity array"]) == [900, 1500, 700, 40000]
    head = output.read_text().split("</fileContent>")[0]
    assert 'name="centroid spectrum"' in head and 'name="profile spectrum"' not in head
    text = output.read_text()
    assert 'name="MSPeak.bin"' in text and 'name="MSProfile.bin"' not in text


def test_positive_polarity_is_written(tmp_path):
    def positive(sid, record):
        record["IonPolarity"] = 0
    bundle = make_bundle(tmp_path, tamper=positive)
    output = tmp_path / "pos.mzML"
    assert main([str(bundle), "-o", str(output)]) == 0
    spectra = list(mzml.MzML(str(output)))
    assert all("positive scan" in s and "negative scan" not in s for s in spectra)


@pytest.mark.skipif(agilent._agilent is None, reason="Agilent decoders not built")
def test_native_decoders_equal_references_on_random_streams():
    rng = np.random.default_rng(20261121)
    for _ in range(400):
        data = rng.integers(0, 256, rng.integers(0, 64)).astype(np.uint8).tobytes()
        size = int(rng.integers(0, 128))
        try:
            expected = lzf_decompress(data, size)
        except FormatError as error:
            with pytest.raises(ValueError) as native:
                agilent._agilent.lzf_decompress(data, size)
            assert str(native.value) == str(error)
        else:
            assert agilent._agilent.lzf_decompress(data, size) == expected
        count = int(rng.integers(0, 64))
        start = int(rng.integers(0, count + 1))
        try:
            expected = rle_decode(data, start, count)
        except FormatError as error:
            with pytest.raises(ValueError) as native:
                agilent._agilent.rle_decode(data, start, count)
            assert str(native.value) == str(error)
        else:
            assert np.array_equal(np.frombuffer(agilent._agilent.rle_decode(data, start, count), dtype=np.int64), expected)
