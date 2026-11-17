"""Narrow X500 SCIEX OS 2.x reader through WIFF + WIFF.SCAN sidecars.

The opaque WIFF2 payload is not decoded. WIFF2 or WIFF can be the discovery
entry point. No reference mzML, vendor library, or network access is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path
import re
import struct

import numpy as np
import olefile

from .binary import FormatError, decode_tokens, frame_profile, measure_spectrum, scan_metadata
from .samples import sample_index, scan_segments

INDEX = struct.Struct("<IIdHddddI")


@dataclass
class Experiment:
    ms_level: int
    polarity: str
    lower_mz: float
    upper_mz: float
    accumulation_ms: float
    collision_energy: float | None
    source_parameters: dict


@dataclass
class Scan:
    slot: int
    cycle: int
    experiment: int
    rt_seconds: float
    tic: float
    base_peak_intensity: float
    offset: int
    size: int
    cal_a: float
    cal_b: float
    precursor_mz: float | None
    precursor_intensity: float | None
    sample_id: int = 1

    @property
    def native_id(self):
        return f"sample={self.sample_id} period=1 cycle={self.cycle} experiment={self.experiment}"


def require(condition, message):
    if not condition:
        raise FormatError(message)


def validate_calibration_records(table, embedded, n_exp):
    """Validate current-slot WIFF coefficients; return explained startup conflicts.

    Vendor references establish that the slot table, not the first populated
    cycle's stale scan preamble, supplies the effective calibration. Acquisitions
    recorded from the first cycle (no empty prefix) can instead carry one stale
    preamble on several consecutive startup scans; ion m/z continuity across the
    transition favours the table there too. Accept a disagreement only as an
    experiment's run of consecutive populated scans from the first populated
    cycle, sharing one stale preamble, whose constant table value (including
    any empty prefix) is matched exactly by both records of the immediately
    following cycle. This is not a general permission to prefer the table or
    shift calibration by cycles. Returns {resolved slot: corroborating slot}.
    """
    first_cycle = min(embedded) // n_exp if embedded else None
    conflicts = {}
    for slot, (ma, mb) in embedded.items():
        cycle, exp = divmod(slot, n_exp)
        a, b, extra = table[slot]
        location = f"cycle={cycle+1} experiment={exp+1}"
        require(0 < a < 1 and np.isfinite([a, b]).all() and extra == 0,
                f"Invalid calibration record at {location}")
        require(0 < ma < 1 and np.isfinite([ma, mb]).all(),
                f"Invalid scan calibration record at {location}")
        if (a, b) != (ma, mb):
            conflicts.setdefault(exp, []).append(slot)
    resolved = {}
    for exp, slots in sorted(conflicts.items(), key=lambda item: item[1][0]):
        start = slots[0]
        run = 1
        while run < len(slots) and slots[run] == start + run*n_exp:
            run += 1
        # Report the first scan the startup rule cannot explain.
        failed = slots[run] if run < len(slots) else start
        following = start + run*n_exp
        a, b, _ = table[start]
        require(run == len(slots) and start // n_exp == first_cycle
                and all(embedded[s] == embedded[start] for s in slots)
                and following in embedded
                and table[following] == (a, b, 0)
                and embedded[following] == (a, b)
                and all(table[i] == (a, b, 0) for i in range(exp, following, n_exp)),
                f"Calibration disagreement at cycle={failed//n_exp+1} experiment={exp+1}: "
                "not a corroborated startup table/scan transition; "
                "refusing to choose unverified coefficients")
        resolved.update(dict.fromkeys(slots, following))
    return dict(sorted(resolved.items()))


def named_source_parameter(data):
    require(len(data) >= 44, "Short source parameter")
    size = struct.unpack_from("<H", data, 34)[0]
    require(size % 2 == 0 and 36 + size + 8 <= len(data), "Invalid source parameter name")
    name = data[36:36+size].decode("utf-16le")
    val, copy = struct.unpack_from("<ff", data, 36+size)
    require(val == copy and np.isfinite(val), "Non-static source parameter")
    return name, val


def mass_range_parameters(data):
    """Two static parameter groups, each containing named float32 ranges."""
    require(len(data) >= 40, "Short MassRangeEx")
    groups, count = struct.unpack_from("<II", data, 32)
    require(0 < groups <= 100 and 0 < count <= 100, "Unsupported MassRangeEx table")
    pos, result = 40, []
    for _ in range(groups):
        # Five float/u32 fields and a length-prefixed optional range label.
        require(pos + 22 <= len(data), "Truncated mass-range prefix")
        label_size = struct.unpack_from("<H", data, pos + 20)[0]
        pos += 22 + label_size
        params = {}
        for _ in range(count):
            require(pos + 2 <= len(data), "Truncated method parameter")
            size = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            require(size % 2 == 0 and pos + size + 12 <= len(data), "Invalid method parameter name")
            name = data[pos:pos+size].decode("utf-16le")
            pos += size
            lo, hi, reserved = struct.unpack_from("<ffI", data, pos)
            pos += 12
            require(reserved == 0 and np.isfinite([lo, hi]).all(), "Unsupported method parameter encoding")
            params[name] = (lo, hi)
        result.append(params)
    require(pos == len(data), "Unconsumed method parameter bytes")
    return result


class WiffReader:
    # This reader accepts only the qualified X500 positive SCIEX OS 2.x layout.
    sparse_profile_sampling_family = "sciex-positive-tof-framed-profile-v1"
    # Subclasses may supply an exact equivalent scan-preamble parser.
    scan_metadata = staticmethod(scan_metadata)

    def __init__(self, entry, sample_id=None):
        self.entry = Path(entry).resolve()
        self.sample_id = sample_id
        self._mapping = None
        self.nominal_window_overhangs = {}
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._mapping is not None:
            self._mapping.close()

    def __enter__(self):
        require(self._mapping is not None and not self._mapping.closed, "Reader is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def _load(self):
        require(self.entry.suffix.lower() in (".wiff2", ".wiff"), "Input must be a .wiff2 or .wiff file")
        require(self.entry.is_file(), f"Missing entry point: {self.entry}")
        self.wiff_path = self.entry.with_suffix(".wiff")
        self.scan_path = self.entry.with_suffix(".wiff.scan")
        for path in (self.wiff_path, self.scan_path):
            require(path.is_file(), f"Required native-reader sidecar missing: {path.name}")
        if self.entry.suffix.lower() == ".wiff2":
            with self.entry.open("rb") as f:
                header = f.read(24)
            require(header[16:24] == bytes.fromhex("100001010c402020"), "Unrecognized WIFF2 page header")
        with self.scan_path.open("rb") as handle:
            require(len(handle.read(44)) == 44, "Truncated scan file")
            self._mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.data = self._mapping
        segments = scan_segments(self.data)
        with olefile.OleFileIO(self.wiff_path) as ole:
            names = ["/".join(p) for p in ole.listdir()]
            self.streams = {p: ole.openstream(p).read() for p in names}
        try:
            self.samples = sample_index(self.streams['SampleSubtree/SampleIdxTable'], names)
            require(set(segments) == {s.id for s in self.samples}, 'WIFF/scan sample identifiers disagree')
            require(all(segments[s.id][0] == s.segment_offset for s in self.samples),
                    'WIFF sample index disagrees with scan segment offsets')
            if self.sample_id is None:
                require(len(self.samples) == 1,
                        f'This WIFF contains {len(self.samples)} samples/runs. Select samples in KmzConvert '
                        'or use --list-samples and --sample ID.')
                self.sample_id = self.samples[0].id
            require(type(self.sample_id) is int and self.sample_id in segments,
                    f'Unknown WIFF sample identifier: {self.sample_id}')
            _, self.segment_start, self.segment_end = segments[self.sample_id]
            # Sample-to-method links are not yet independently decoded. Multiple
            # methods are safe only when every MS method stream is byte-identical.
            methods = sorted({p.split('/')[1] for p in names if p.startswith('MethodSubtree/')})
            require(methods, 'Missing WIFF acquisition method')
            method_streams = [{p.removeprefix(f'MethodSubtree/{m}/'): value
                               for p, value in self.streams.items()
                               if p.startswith(f'MethodSubtree/{m}/DeviceMethod0/')}
                              for m in methods]
            require(method_streams[0] and all(m == method_streams[0] for m in method_streams),
                    'Different MS methods across samples are not yet qualified; refusing to guess sample/method association')
            self.method_name = methods[0]
            self._parse()
        except KeyError as e:
            raise FormatError(f"Missing required WIFF metadata stream: {e.args[0]}") from e
        except (struct.error, UnicodeError) as e:
            raise FormatError(f"Truncated or invalid WIFF metadata: {e}") from e
        from .dad import read_dad
        self.dad, self.dad_status, self.dad_message = read_dad(self.entry, self.streams, self.sample_id)

    def _parse(self):
        s = self.streams
        sample = f"SampleSubtree/Sample{self.sample_id}/"
        method_root = f"MethodSubtree/{self.method_name}/DeviceMethod0/"
        method = method_root + "Period0/"
        log = s[sample + "Log"].decode("utf-16le", errors="replace")
        instrument = re.search(r"Mass Spectrometer: ([^\r\n]+)", log)
        require(instrument is not None, "Missing instrument identity")
        self.instrument = dict(part.split(":", 1) for part in instrument[0].split(",") if ":" in part)
        self.instrument = {k.strip(): v.strip() for k, v in self.instrument.items()}
        require(self.instrument.get("Mass Spectrometer") == "X500 QTOF", "Only X500 QTOF is validated")
        software = re.search(r"SCIEX OS [\d.]+", s["FileRec_Str"].decode("utf-16le", errors="replace"))
        self.acquisition_software = software[0] if software else None
        require(self.acquisition_software is not None and self.acquisition_software.startswith("SCIEX OS 2."), "Only SCIEX OS 2.x layout is validated")
        periods = {p.split("/")[3] for p in s if re.match(re.escape(method_root) + r"Period\d+/", p)}
        require(periods == {"Period0"}, "Only single-period acquisitions are validated")
        period = s[method + "PeriodStream"]
        require(len(period) == 78, "Unsupported period record")
        n_exp = struct.unpack_from("<I", period, 74)[0]
        require(1 < n_exp <= 100, "Unsupported number of experiments")
        self.experiments = []
        for exp in range(n_exp):
            prefix = method + f"Experiment{exp}/"
            header = s[prefix + "ExperimentHeader"]
            # The scan-type follows the variable-length UTF16 experiment name.
            name_size = struct.unpack_from("<H", header, 90)[0]
            type_pos = 94 + name_size
            require(type_pos + 4 <= len(header), "Invalid experiment header")
            scan_type = struct.unpack_from("<I", header, type_pos)[0]
            require(scan_type in (8, 9), f"Unsupported scan type {scan_type}")
            require((scan_type == 8) == (exp == 0), "Expected TOFMS followed by dependent TOFMSMS slots")
            extended = s[prefix + "ExperimentHeaderEx"]
            require(len(extended) == 94, "Unsupported experiment range record")
            lo, hi, dwell = struct.unpack_from("<ddd", extended, 66)
            require(0 < lo < hi and dwell > 0 and np.isfinite([lo, hi, dwell]).all(), "Invalid mass range")
            params = dict(named_source_parameter(content) for path, content in s.items()
                          if path.startswith(prefix + "IonSourceParamsTable/") and path.endswith("/ParameterData"))
            voltage = params.get("IS", params.get("ISVF", 0))
            require(voltage > 0, "Only the positive-polarity X500 layout is validated")
            groups = mass_range_parameters(s[prefix + "MassRangeEx/MassRangeEx"])
            ces = {pair for group in groups for name, pair in group.items() if name == "CE"}
            ce = None
            if len(ces) == 1:
                pair = next(iter(ces))
                if pair[0] == pair[1] and pair[0] > 0:
                    ce = pair[0]
            require(scan_type == 8 or ce is not None, "Dynamic collision energy is not decoded")
            self.experiments.append(Experiment(1 if scan_type == 8 else 2, "positive", lo, hi, dwell, ce, params))
        self.tdc = struct.unpack_from("<d", s[sample + "TDCInfo"], 32)[0]
        require(0 < self.tdc < 1, "Unsupported TDC interval")
        idx = s[sample + "Idx"]
        require(len(idx) >= 32 and (len(idx)-32) % INDEX.size == 0, "Malformed scan index length")
        self.index = list(INDEX.iter_unpack(idx[32:]))
        require(len(self.index) % n_exp == 0, "Incomplete cycle slots")
        cal = s[sample + "TOFCalibrationData"]
        require(len(cal) == 56 + len(self.index) * 20, "Calibration/scan index count mismatch")
        require(struct.unpack_from("<I", cal, 52)[0] == len(self.index), "Calibration count field mismatch")
        calibration = list(struct.iter_unpack("<ddI", cal[56:]))
        embedded_calibration = {}
        dde = s[sample + "DDERealTimeData"]
        require(len(dde) == 32 + (len(self.index)//n_exp)*(n_exp-1)*32, "Unsupported DDE record layout")
        self.scans = []
        base = self.segment_start
        cursor = base
        self.checkpoint_count = 0
        for slot, record in enumerate(self.index):
            offset, size, rt_ms, flags, tic, bpi, _, _, reserved = record
            cycle, exp = divmod(slot, n_exp)
            require(flags == 6 and reserved == 0, "Unsupported scan-index flags")
            require(np.isfinite([rt_ms, tic, bpi]).all() and min(rt_ms, tic, bpi) >= 0,
                    "Invalid scan time/TIC/BPI")
            if size == 0:
                # Empty slots retain the preceding payload's end offset.
                require(offset + base == cursor and tic == 0, "Invalid empty scan slot")
                continue
            require(size > 9 and cursor < offset+base and offset+base+size <= self.segment_end, "Scan range out of bounds")
            a, b, _ = calibration[slot]
            ma, mb, checkpoints = self.scan_metadata(self.data, cursor, offset+base)
            embedded_calibration[slot] = (ma, mb)
            require(checkpoints and checkpoints[0] == (struct.unpack_from("<I", self.data, offset+base+4)[0], 8), "Missing initial scan checkpoint")
            require(all(0 <= y < size for _, y in checkpoints), "Checkpoint outside scan block")
            self.checkpoint_count += len(checkpoints)
            cursor = offset + base + size
            precursor, precursor_intensity = None, None
            if exp:
                dde_pos = 32 + (cycle*(n_exp-1) + exp-1)*32
                precursor, u1, precursor_intensity, u2 = struct.unpack_from("<dddd", dde, dde_pos)
                require(np.isfinite([precursor, precursor_intensity]).all() and
                        precursor > 0 and precursor_intensity > 0 and u1 == 0 and u2 == 0, "Unrecognized precursor record")
            self.scans.append(Scan(slot, cycle+1, exp+1, rt_ms/1000, tic, bpi,
                                   offset+base, size, a, b, precursor, precursor_intensity, self.sample_id))
        require(cursor == self.segment_end, "Unindexed trailing scan payload")
        require(self.scans and all(a.rt_seconds <= b.rt_seconds for a, b in zip(self.scans, self.scans[1:])), "Empty or non-monotonic acquisition")
        resolved = validate_calibration_records(calibration, embedded_calibration, n_exp)
        self.calibration_resolutions = [
            {"native_id": scan.native_id, "slot": scan.slot,
             "source": "WIFF TOFCalibrationData current slot",
             "corroborating_slot": resolved[scan.slot],
             "rule": "constant startup table, one stale preamble on consecutive first scans "
                     "and exact following-cycle table/preamble agreement"}
            for scan in self.scans if scan.slot in resolved]

    def block(self, scan):
        require(self._mapping is not None and not self._mapping.closed, "Reader is closed")
        return self.data[scan.offset:scan.offset+scan.size]

    def arrays(self, scan, framing=True, *, decoder=None):
        # A caller may supply an equivalent codec; integrity/calibration checks
        # remain owned here and apply to every decoded spectrum.
        decode = decode_tokens if decoder is None else decoder
        x, y = decode(self.block(scan))
        total, top, mz, valid = measure_spectrum(x, y, scan.cal_a, scan.cal_b, self.tdc)
        self.check_measured(scan, total, top, mz, valid)
        if framing:
            return frame_profile(x, y, scan.cal_a, scan.cal_b, self.tdc)
        return mz, y

    def check_measured(self, scan, total, top, mz, valid):
        """Guard one decoded spectrum; ``mz`` needs only its first and last measured value."""
        require(total == scan.tic, f"Decoded intensity sum disagrees with TIC: {scan.native_id}")
        require(top == scan.base_peak_intensity, f"Decoded maximum disagrees with BPI: {scan.native_id}")
        require(valid, f"Invalid measured spectrum: {scan.native_id}")
        exp = self.experiments[scan.experiment-1]
        if mz[0] < exp.lower_mz or mz[-1] > exp.upper_mz:
            self.nominal_window_overhangs[scan.native_id] = (float(mz[0]), float(mz[-1]))

    def tic_arrays(self):
        # The vendor TIC includes empty slots, but ends at the last acquired slot.
        records = self.index[:self.scans[-1].slot+1]
        times = np.array([r[2]/60000 for r in records], dtype=np.float64)
        intensities = np.array([r[4] for r in records], dtype=np.float64)
        levels = np.array([self.experiments[i % len(self.experiments)].ms_level for i in range(len(records))], dtype=np.int64)
        return times, intensities, levels
