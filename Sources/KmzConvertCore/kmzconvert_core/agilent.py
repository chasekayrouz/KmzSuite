"""Narrow Agilent MassHunter ``.d`` reader for TOF/Q-TOF profile acquisitions.

Clean-room decoding from the bundle's own files: ``MSScan.bin`` records are laid
out from the bundle's ``MSScan.xsd``; ``MSProfile.bin`` blocks are raw LZF
(MassHunter Acquisition B.0x) or run-length coded (10.x); flight times become m/z
with ``MSMassCal.bin`` per-scan records or, without them, ``DefaultMassCal.xml``.
Every profile spectrum must reproduce the stored TIC, base-peak intensity and
base-peak m/z. Anything else (triple quadrupole, ion mobility, centroid-only data,
unknown layouts) is refused. No vendor library or network access is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

from .binary import FormatError
from .reader import Experiment

try:
    from . import _agilent
except ImportError:
    _agilent = None

ACCELERATED = _agilent is not None

XS = "{http://www.w3.org/2001/XMLSchema}"
XSD_TYPES = {"xs:int": "<i4", "xs:short": "<i2", "xs:long": "<i8", "xs:double": "<f8", "xs:float": "<f4"}
PROFILE, CENTROID = 1, 2
# Run-length blocks: bytes 16-19 hold the point count in 24 bits under a 0x90 marker byte.
RLE_MARKER = 0x90
# MassHunter Devices.xml device types for the supported mass analyzers.
SUPPORTED_DEVICES = {"QTOF": "Q-TOF", "TOF": "TOF"}
# ScanType values for full-scan MS1 and product-ion records (qualified: targeted MS/MS).
SUPPORTED_SCAN_TYPES = {1: 1, 512: 2}
# BasePeakMZ is stored to the instrument's double precision with the default calibration.
BASE_PEAK_PPM = 0.01
SOURCE_FILES = ("MSScan.bin", "MSScan.xsd", "MSProfile.bin", "MSMassCal.bin", "DefaultMassCal.xml",
                "MSTS.xml", "Devices.xml", "Contents.xml")
LIMITATIONS = (
    "Experimental Agilent MassHunter TOF/Q-TOF profile reader. Isolation widths, activation type, "
    "parent-scan links and absolute acquisition timestamp are not decoded. "
    "Stored profile points are written with one zero-intensity neighbour on each side of signal; "
    "longer zero runs are omitted. Centroid, DAD/UV and auxiliary data are not exported."
)


def require(condition, message):
    if not condition:
        raise FormatError(message)


def is_masshunter(path):
    """True for a directory that holds an Agilent MassHunter scan index."""
    path = Path(path)
    if not path.is_dir():
        return False
    acq = _child(path, "AcqData")
    return acq is not None and _child(acq, "MSScan.bin") is not None


def _child(directory, name):
    """Case-insensitive child lookup: MassHunter versions differ in file-name case."""
    try:
        matches = [p for p in Path(directory).iterdir() if p.name.lower() == name.lower()]
    except OSError:
        return None
    require(len(matches) <= 1, f"Ambiguous {name} entries in {directory}")
    return matches[0] if matches else None


def _xml(path):
    try:
        return ET.fromstring(path.read_bytes())
    except ET.ParseError as error:
        raise FormatError(f"Malformed {path.name}: {error}") from error


def record_layout(xsd):
    """Packed NumPy fields of one ScanRecord and its repeated SpectrumParams block."""
    root = _xml(xsd)
    types = {c.get("name"): c for c in root.iter(XS + "complexType") if c.get("name")}
    require("ScanRecordType" in types and "SpectrumParamsType" in types, "Unsupported MSScan.xsd layout")

    def fields(name, prefix=""):
        sequence = types[name].find(XS + "sequence")
        require(sequence is not None, f"Unsupported MSScan.xsd type {name}")
        out, repeated = [], None
        children = [c for c in sequence if isinstance(c.tag, str)]
        require(all(c.tag == XS + "element" for c in children), f"Unsupported MSScan.xsd construct in {name}")
        for element in children:
            require(repeated is None, "Unsupported MassHunter scan layout (fields follow the spectrum blocks); "
                                      "acquisitions from older MassHunter versions are not supported")
            ident, kind = element.get("name"), element.get("type")
            if kind in XSD_TYPES:
                require(element.get("maxOccurs") in (None, "1"), f"Unsupported repeated field {ident}")
                out.append((prefix + ident, XSD_TYPES[kind]))
            elif kind == "SpectrumParamsType" and element.get("maxOccurs") == "unbounded" and not prefix:
                require(repeated is None, "Unsupported MSScan.xsd layout")
                repeated, _ = fields(kind)
            elif kind in types and kind != "SpectrumParamsType":
                nested, inner = fields(kind, prefix + ident + ".")
                require(inner is None, "Unsupported MSScan.xsd nesting")
                out += nested
            else:
                raise FormatError(f"Unsupported MassHunter scan layout (MSScan.xsd element {ident}); "
                                  "acquisitions from older MassHunter versions are not supported")
        return out, repeated

    fixed, block = fields("ScanRecordType")
    require(block is not None, "MSScan.xsd declares no spectrum blocks")
    # MzOfInterest is required only for MS/MS records (checked per scan).
    needed = {"ScanID", "ScanMethodID", "TimeSegmentID", "ScanTime", "MSLevel", "ScanType", "TIC",
              "BasePeakMZ", "BasePeakValue", "CalibrationID", "IonPolarity", "CollisionEnergy"}
    names = {name for name, _ in fixed}
    require(needed <= names, f"Unsupported MassHunter scan layout (MSScan.xsd lacks {', '.join(sorted(needed - names))})")
    block_names = {name for name, _ in block}
    require({"SpectrumFormatID", "SpectrumOffset", "ByteCount", "PointCount", "UncompressedByteCount",
             "MinX", "MaxX"} <= block_names, "MSScan.xsd spectrum block lacks required fields")
    return np.dtype(fixed), np.dtype(block)


@dataclass(frozen=True)
class Calibration:
    a: float
    t0: float
    t_min: float
    t_max: float
    coefficients: tuple
    powers: tuple

    def mz(self, t):
        t = np.asarray(t, dtype=np.float64)
        mz = (self.a * (t - self.t0)) ** 2
        if self.coefficients:
            clamped = np.clip(t, self.t_min, self.t_max)
            correction = np.zeros_like(clamped)
            for c, p in zip(self.coefficients, self.powers):
                correction += c * clamped ** p
            mz = mz - correction
        return mz


def _powers(flags):
    return tuple(bit for bit in range(32) if flags >> bit & 1)


def default_calibrations(path):
    """DefaultMassCal.xml: {DefaultCalibrationID: (traditional[2], polynomial[8], flags)}."""
    try:
        return _default_calibrations(path)
    except (TypeError, ValueError, KeyError) as error:
        if isinstance(error, FormatError):
            raise
        raise FormatError(f"Malformed DefaultMassCal.xml: {error}") from error


def _default_calibrations(path):
    calibrations = {}
    for entry in _xml(path).iter("DefaultCalibration"):
        steps = {s.findtext("CalibrationFormula"): s for s in entry.iter("Step")}
        require(set(steps) == {"Traditional", "Polynomial"}, "Unsupported DefaultMassCal.xml calibration steps")
        values = {}
        for formula, step in steps.items():
            numbered = sorted((int(v.get("Number")), float(v.text)) for v in step.iter("Value"))
            require([n for n, _ in numbered] == list(range(1, len(numbered) + 1)), "Malformed DefaultMassCal.xml values")
            values[formula] = [v for _, v in numbered]
        require(len(values["Traditional"]) == 2 and len(values["Polynomial"]) == 8,
                "Unsupported DefaultMassCal.xml coefficient counts")
        flags = int(steps["Polynomial"].findtext("ValueUseFlags"))
        require(len(_powers(flags)) <= 6, "Unsupported DefaultMassCal.xml polynomial flags")
        calibrations[int(entry.get("DefaultCalibrationID"))] = (values["Traditional"], values["Polynomial"], flags)
    require(calibrations, "DefaultMassCal.xml has no calibrations")
    return calibrations


def calibration(traditional, polynomial, flags):
    powers = _powers(flags)
    coefficients = tuple(polynomial[2:2 + len(powers)])
    require(not any(polynomial[2 + len(powers):]), "Polynomial calibration uses undeclared coefficients")
    a, t0 = traditional
    require(np.isfinite([a, t0, *polynomial]).all() and a > 0, "Invalid mass calibration")
    return Calibration(float(a), float(t0), float(polynomial[0]), float(polynomial[1]), coefficients, powers)


def _close(a, b):
    return all(abs(x - y) <= 1e-12 * max(abs(x), abs(y)) for x, y in zip(a, b, strict=True))


# Decoders: the C extension when built, else these reference implementations.

def lzf_decompress(data, size):
    """Raw LZF stream (no header) to exactly ``size`` bytes."""
    data = bytes(data)
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        control = data[i]
        i += 1
        if control < 32:
            run = control + 1
            require(i + run <= n, "Truncated LZF literal")
            out += data[i:i + run]
            i += run
        else:
            length = control >> 5
            if length == 7:
                require(i < n, "Truncated LZF back-reference")
                length += data[i]
                i += 1
            require(i < n, "Truncated LZF back-reference")
            ref = len(out) - ((control & 0x1f) << 8) - data[i] - 1
            i += 1
            require(ref >= 0, "Invalid LZF back-reference")
            for _ in range(length + 2):
                out.append(out[ref])
                ref += 1
        require(len(out) <= size, "LZF output exceeds the declared size")
    require(len(out) == size, "LZF output is shorter than the declared size")
    return bytes(out)


def rle_decode(stream, start, count):
    """Run-length profile stream to ``count`` int64 intensities (zeros where unreached)."""
    stream = bytes(stream)
    out = np.zeros(count, dtype=np.int64)
    formats = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}
    i, width, point, n = 0, 4, start, len(stream)
    require(0 <= start <= count, "Invalid run-length start point")
    while i < n:
        require(i + width <= n, "Truncated run-length stream")
        value = struct.unpack_from(formats[width], stream, i)[0]
        i += width
        if value >= 0:
            require(point < count, "Run-length stream exceeds the point count")
            out[point] = value
            point += 1
        else:
            u = -value - 1
            point += u >> 2
            width = 1 << (u & 3)
            require(point <= count, "Run-length stream exceeds the point count")
    return out


def is_run_length(block, points):
    if len(block) < 24:
        return False
    word = struct.unpack_from("<I", block, 16)[0]
    return word >> 24 == RLE_MARKER and word & 0xffffff == points


def decode_profile(block, points, uncompressed):
    """(x0, dx, int64 intensities) of one MSProfile.bin block."""
    require(uncompressed == 16 + 4 * points, "Unsupported profile block size")
    if is_run_length(block, points):
        x0, dx = struct.unpack_from("<dd", block)
        start = -struct.unpack_from("<i", block, 20)[0]
        require(np.isfinite(x0) and np.isfinite(dx) and x0 > 0 and dx > 0, "Invalid run-length profile axis")
        if _agilent is not None:
            try:
                y = np.frombuffer(_agilent.rle_decode(bytes(block[24:]), start, points), dtype=np.int64)
            except ValueError as error:
                raise FormatError(str(error)) from error
        else:
            y = rle_decode(block[24:], start, points)
        return x0, dx, y
    if _agilent is not None:
        try:
            raw = _agilent.lzf_decompress(bytes(block), uncompressed)
        except ValueError as error:
            raise FormatError(str(error)) from error
    else:
        raw = lzf_decompress(block, uncompressed)
    x0, dx = struct.unpack_from("<dd", raw)
    require(np.isfinite(x0) and np.isfinite(dx) and x0 > 0 and dx > 0, "Invalid LZF profile axis")
    return x0, dx, np.frombuffer(raw, dtype="<i4", offset=16).astype(np.int64)


@dataclass
class Scan:
    scan_id: int
    method: int
    rt_minutes: float
    ms_level: int
    experiment: int
    tic: float
    base_peak_mz: float
    base_peak_intensity: float
    calibration_id: int
    precursor_mz: float | None
    precursor_charge: int | None
    offset: int
    size: int
    points: int
    uncompressed: int

    @property
    def native_id(self):
        return f"scanId={self.scan_id}"

    @property
    def rt_seconds(self):
        return self.rt_minutes * 60

    @property
    def preset(self):
        return self.method


class AgilentReader:
    """Reader contract used by the mzML writer and CLI, for one MassHunter ``.d``."""
    format_name = "Agilent MassHunter"
    source_ref = "MSScan"
    limitations = LIMITATIONS
    calibration_resolutions = ()
    dad = None

    def __init__(self, entry):
        self.entry = Path(entry).resolve()
        self.nominal_window_overhangs = {}
        self._mapping = None
        self._file = None
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._mapping is not None and not self._mapping.closed:
            self._mapping.close()
        if self._file is not None:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _load(self):
        require(self.entry.is_dir(), f"Not a directory: {self.entry}")
        acq = _child(self.entry, "AcqData")
        require(acq is not None, f"{self.entry.name} has no AcqData folder; not an Agilent MassHunter acquisition")
        self.acq = acq
        files = {name: _child(acq, name) for name in SOURCE_FILES}
        self.files = {name: path for name, path in files.items() if path is not None}
        require(files["MSScan.bin"] is not None and files["MSScan.xsd"] is not None,
                "No MassHunter MS scan index (MSScan.bin/MSScan.xsd)")
        require(_child(acq, "IMSFrame.bin") is None, "Agilent ion-mobility (IM-QTOF) acquisitions are not supported")
        require(files["Devices.xml"] is not None, "Devices.xml is missing; the instrument cannot be identified")
        self.instrument = self._device(files["Devices.xml"])
        require(files["MSTS.xml"] is not None, "MSTS.xml is missing")
        contents = _xml(files["Contents.xml"]) if files["Contents.xml"] is not None else None
        self.acquisition_software = (contents.findtext("AcqSoftwareVersion") or "").strip() if contents is not None else ""
        record, block = record_layout(files["MSScan.xsd"])
        data = files["MSScan.bin"].read_bytes()
        require(len(data) >= 0x5c, "Truncated MSScan.bin")
        header = struct.unpack_from("<I", data, 0x58)[0]
        segments = [int(n) for n in re.findall(rb"<NumOfScans>\s*(\d+)\s*</NumOfScans>", files["MSTS.xml"].read_bytes())]
        count = sum(segments)
        require(segments and count > 0, "MSTS.xml records no scans")
        payload = len(data) - header
        require(header >= 0x5c and payload > 0 and payload % count == 0,
                "MSScan.bin size disagrees with the MSTS.xml scan count")
        stride = payload // count
        extra = stride - record.itemsize
        require(extra > 0 and extra % block.itemsize == 0 and extra // block.itemsize <= 4,
                f"MSScan.bin record size {stride} does not match MSScan.xsd")
        layout = np.dtype(record.descr + [(f"block{i}", block) for i in range(extra // block.itemsize)])
        self.records = records = np.frombuffer(data, layout, count=count, offset=header)
        blocks = [records[f"block{i}"] for i in range(extra // block.itemsize)]
        self._calibrations(files, records)
        self._profile_file(files)
        self._parse(records, blocks)

    def _device(self, path):
        devices = [(d.findtext("Name", "").strip(), d) for d in _xml(path).iter("Device")]
        ms = [(name, d) for name, d in devices if name in SUPPORTED_DEVICES]
        names = ", ".join(name for name, _ in devices) or "none"
        if not ms:
            if any(name in ("TandemQuadrupole", "Quadrupole", "IM-QTOF") or "Quad" in name for name, _ in devices):
                raise FormatError(f"Only Agilent TOF and Q-TOF acquisitions are supported (devices: {names})")
            raise FormatError(f"No supported Agilent TOF/Q-TOF mass spectrometer in Devices.xml (devices: {names})")
        require(len(ms) == 1, "More than one mass spectrometer in Devices.xml")
        name, device = ms[0]
        return {"Name": SUPPORTED_DEVICES[name],
                "Model": device.findtext("ModelNumber", "").strip(),
                "Serial Number": device.findtext("SerialNumber", "").strip(),
                "Firmware": device.findtext("FirmwareVersion", "").strip()}

    def _calibrations(self, files, records):
        require(files["DefaultMassCal.xml"] is not None, "DefaultMassCal.xml is missing; flight times cannot be calibrated")
        defaults = default_calibrations(files["DefaultMassCal.xml"])
        ids = sorted({int(i) for i in records["CalibrationID"]})
        require(set(ids) <= set(defaults), "Scans name calibrations absent from DefaultMassCal.xml")
        self.default_calibration = {i: calibration(*defaults[i]) for i in ids}
        self.calibration_source = "DefaultMassCal.xml"
        self.scan_calibration = None
        path = files["MSMassCal.bin"]
        # MassCalOffset is each scan's byte offset in MSMassCal.bin (0: no per-scan record).
        offsets = records["MassCalOffset"].astype(np.int64) if "MassCalOffset" in records.dtype.names else None
        referenced = offsets is not None and bool((offsets != 0).any())
        if path is None:
            require(not referenced, "MSMassCal.bin is missing but scans reference per-scan calibrations")
            return
        data = path.read_bytes()
        require(len(data) >= 72 and (len(data) - 72) % 84 == 0, "Malformed MSMassCal.bin")
        table = np.frombuffer(data, np.dtype([("n", "<i4"), ("values", "<f8", 10)]), offset=72)
        require(offsets is None or referenced,
                "MSMassCal.bin is present but no scan references a per-scan calibration")
        if referenced:
            require(((offsets - 72) % 84 == 0).all() and (offsets >= 72).all() and (offsets < len(data)).all(),
                    "MassCalOffset does not address MSMassCal.bin records")
            rows = (offsets - 72) // 84
        else:
            require(len(table) == len(records), "MSMassCal.bin record count disagrees with MSScan.bin")
            rows = np.arange(len(records))
        require((table["n"][rows] == 10).all(), "Unsupported MSMassCal.bin record")
        cache, per_scan = {}, []
        for row, cal_id in zip(table["values"][rows], records["CalibrationID"]):
            key = (int(cal_id), row.tobytes())
            if key not in cache:
                traditional, polynomial, flags = defaults[int(cal_id)]
                require(_close(row[2:], polynomial), "MSMassCal.bin polynomial differs from DefaultMassCal.xml")
                cache[key] = calibration(row[:2], row[2:].tolist(), flags)
            per_scan.append(cache[key])
        self.scan_calibration = per_scan
        self.calibration_source = "MSMassCal.bin"

    def _profile_file(self, files):
        path = files["MSProfile.bin"]
        require(path is not None and path.stat().st_size > 68,
                "No profile data (MSProfile.bin); centroid-only Agilent acquisitions are not supported yet")
        self._file = path.open("rb")
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def _parse(self, records, blocks):
        self.scans, self.experiments = [], []
        experiments = {}
        size = len(self._mapping)
        previous = -np.inf
        for i, r in enumerate(records):
            ident = f"scanId={int(r['ScanID'])}"
            profile = [b for b in blocks if b[i]["SpectrumFormatID"] == PROFILE]
            require(len(profile) == 1,
                    f"{ident} has no profile spectrum; centroid-only Agilent acquisitions are not supported yet")
            p = profile[0][i]
            scan_type = int(r["ScanType"])
            require(scan_type in SUPPORTED_SCAN_TYPES, f"Unsupported Agilent scan type {scan_type} at {ident}")
            level = int(r["MSLevel"])
            require(level == SUPPORTED_SCAN_TYPES[scan_type], f"MS level disagrees with scan type at {ident}")
            polarity = {0: "positive", 1: "negative"}.get(int(r["IonPolarity"]))
            require(polarity is not None, f"Unknown ion polarity at {ident}")
            rt = float(r["ScanTime"])
            require(np.isfinite(rt) and rt >= previous, f"Retention times decrease at {ident}")
            previous = rt
            offset, nbytes = int(p["SpectrumOffset"]), int(p["ByteCount"])
            require(68 <= offset and nbytes > 0 and offset + nbytes <= size, f"Profile block out of bounds at {ident}")
            fields = records.dtype.names
            ce = float(r["CollisionEnergy"])
            if level == 1:
                # All Ions MS/MS stores fragmentation scans as MS1 records with a collision energy.
                require(ce == 0, f"MS1 scan with collision energy (All Ions MS/MS) is not supported: {ident}")
                precursor, charge = None, None
            else:
                require("MzOfInterest" in fields, f"MS/MS record without precursor fields at {ident}")
                precursor = float(r["MzOfInterest"])
                require(np.isfinite(precursor) and precursor > 0, f"Missing precursor m/z at {ident}")
                charge = int(r["ChargeState"]) if "ChargeState" in fields else 0
                charge = charge if charge > 0 else None
            lower, upper = float(p["MinX"]), float(p["MaxX"])
            require(0 < lower < upper, f"Invalid scan window at {ident}")
            key = (level, polarity, lower, upper, ce if level > 1 else None)
            if key not in experiments:
                self.experiments.append(Experiment(level, polarity, lower, upper, 0.0, key[4], {}))
                experiments[key] = len(self.experiments)
            self.scans.append(Scan(int(r["ScanID"]), int(r["ScanMethodID"]), rt, level, experiments[key],
                                   float(r["TIC"]), float(r["BasePeakMZ"]), float(r["BasePeakValue"]),
                                   int(r["CalibrationID"]), precursor, charge, offset, nbytes,
                                   int(p["PointCount"]), int(p["UncompressedByteCount"])))
        ids = [s.scan_id for s in self.scans]
        require(len(set(ids)) == len(ids), "Duplicate MassHunter scan IDs")
        self._index = {id(s): i for i, s in enumerate(self.scans)}

    def calibration(self, scan):
        if self.scan_calibration is not None:
            return self.scan_calibration[self._index[id(scan)]]
        return self.default_calibration[scan.calibration_id]

    def block(self, scan):
        require(self._mapping is not None and not self._mapping.closed, "Reader is closed")
        return self._mapping[scan.offset:scan.offset + scan.size]

    def decoded(self, scan):
        """Guarded (flight time, intensity) of every stored profile point."""
        x0, dx, y = decode_profile(self.block(scan), scan.points, scan.uncompressed)
        require(len(y) == scan.points, f"Profile point count disagrees at {scan.native_id}")
        require((y >= 0).all(), f"Negative profile intensity at {scan.native_id}")
        t = x0 + dx * np.arange(scan.points, dtype=np.float64)
        total, top = float(y.sum()), float(y.max()) if len(y) else 0.0
        require(total == scan.tic, f"Decoded intensity sum disagrees with TIC: {scan.native_id}")
        require(top == scan.base_peak_intensity, f"Decoded maximum disagrees with base peak intensity: {scan.native_id}")
        if top > 0:
            require(np.isfinite(scan.base_peak_mz) and scan.base_peak_mz > 0,
                    f"Invalid stored base-peak m/z: {scan.native_id}")
            # The stored base-peak m/z uses the default calibration, before reference-mass adjustment.
            at = t[y == y.max()]
            mz = self.default_calibration[scan.calibration_id].mz(at)
            ppm = np.min(np.abs(mz - scan.base_peak_mz)) / scan.base_peak_mz * 1e6
            require(ppm <= BASE_PEAK_PPM, f"Decoded base-peak m/z disagrees ({ppm:.3g} ppm): {scan.native_id}")
        return t, y

    def arrays(self, scan, framing=True):
        """Calibrated m/z and intensity; framing keeps one zero neighbour beside each signal run."""
        t, y = self.decoded(scan)
        if framing:
            signal = y > 0
            keep = signal.copy()
            keep[1:] |= signal[:-1]
            keep[:-1] |= signal[1:]
            t, y = t[keep], y[keep]
        mz = self.calibration(scan).mz(t)
        require(np.isfinite(mz).all() and (len(mz) < 2 or (np.diff(mz) > 0).all()),
                f"Calibrated m/z axis is not increasing: {scan.native_id}")
        # MinX/MaxX are the instrument's acquisition range, not exact bounds of the stored
        # axis (which extends slightly beyond it), so no nominal-window overhangs are reported.
        return mz, y.astype(np.float64)

    def tic_arrays(self):
        times = np.array([s.rt_minutes for s in self.scans], dtype=np.float64)
        intensities = np.array([s.tic for s in self.scans], dtype=np.float64)
        levels = np.array([s.ms_level for s in self.scans], dtype=np.int64)
        return times, intensities, levels

    def inspect(self):
        from collections import Counter
        return {"format": self.format_name, "spectra": len(self.scans),
                "ms_levels": dict(Counter(s.ms_level for s in self.scans)),
                "polarities": dict(Counter(self.experiments[s.experiment - 1].polarity for s in self.scans)),
                "instrument": self.instrument, "software": self.acquisition_software,
                "calibration": self.calibration_source,
                "profile_encoding": self.profile_encoding(),
                "limitations": LIMITATIONS}

    def profile_encoding(self):
        if not self.scans:
            return None
        first = self.scans[0]
        return "run-length" if is_run_length(self.block(first), first.points) else "LZF"
