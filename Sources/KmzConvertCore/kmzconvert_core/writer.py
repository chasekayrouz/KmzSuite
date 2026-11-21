"""Streaming indexed mzML 1.1 writer using stdlib XML and little-endian arrays."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import errno
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import _escape_attrib
import zlib

import numpy as np
from kmzcore.sparse_profile import (FORMAT, METHOD_PARAMS, SAMPLING_PARAM, V1, V3, canonical, output_path,
                                   rescue_selection, source_sampling_check)

from . import __version__, native

MS_URI = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
UO_URI = "https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"
MZ = ("MS", "MS:1000040", "m/z")
COUNTS = ("MS", "MS:1000131", "number of detector counts")
MINUTES = ("UO", "UO:0000031", "minute")
EV = ("UO", "UO:0000266", "electronvolt")
NM = ("UO", "UO:0000018", "nanometer")
AU = ("UO", "UO:0000269", "absorbance unit")
# Sparse spectra are small: workers take several per task to amortize hand-offs.
SPARSE_BATCH = 4
LIMITATIONS = (
    "Experimental X500 sidecar reader. Precursor charge, isolation widths, "
    "activation type, and absolute acquisition timestamp are not decoded. "
    "Instrument model is preserved as X500 QTOF; R/B subtype is unresolved. "
    "Framing zeros use calibrated TOF bins. Qualified DAD sidecars are included when available; "
    "auxiliary time-series data are not exported."
)


def cv(parent, accession, name, value="", unit=None):
    attributes = dict(cvRef=accession.split(":")[0], accession=accession, name=name, value=str(value))
    if unit:
        attributes.update(unitCvRef=unit[0], unitAccession=unit[1], unitName=unit[2])
    return ET.SubElement(parent, "cvParam", attributes)


def user(parent, name, value):
    return ET.SubElement(parent, "userParam", name=name, value=str(value))


def binary_array(parent, array, kind):
    integer = kind == "ms level"
    array = np.asarray(array, dtype="<i8" if integer else "<f8")
    encoded = base64.b64encode(zlib.compress(array.tobytes())).decode("ascii")
    parent.append(array_element(kind, str(len(encoded)), encoded))


def array_element(kind, length, encoded):
    integer = kind == "ms level"
    element = ET.Element("binaryDataArray", encodedLength=length)
    cv(element, "MS:1000522" if integer else "MS:1000523", "64-bit integer" if integer else "64-bit float")
    cv(element, "MS:1000574", "zlib compression")
    if kind == "mz":
        cv(element, "MS:1000514", "m/z array", unit=MZ)
    elif kind == "intensity":
        cv(element, "MS:1000515", "intensity array", unit=COUNTS)
    elif kind == "time":
        cv(element, "MS:1000595", "time array", unit=MINUTES)
    elif kind == "ms level":
        cv(element, "MS:1000786", "non-standard data array", "ms level")
    elif kind == "wavelength":
        cv(element, "MS:1000617", "wavelength array", unit=NM)
    elif kind == "absorbance":
        cv(element, "MS:1000515", "intensity array", unit=AU)
    else:
        raise ValueError(f"Unsupported array {kind}")
    ET.SubElement(element, "binary").text = encoded
    return element


def metadata(reader, profile="full", mass_correction=None, *, sparse_format=FORMAT):
    container = ET.Element("metadata")
    cvs = ET.SubElement(container, "cvList", count="2")
    ET.SubElement(cvs, "cv", id="MS", fullName="Proteomics Standards Initiative Mass Spectrometry Ontology", version="4.2.2", URI=MS_URI)
    ET.SubElement(cvs, "cv", id="UO", fullName="Unit Ontology", URI=UO_URI)
    description = ET.SubElement(container, "fileDescription")
    content = ET.SubElement(description, "fileContent")
    agilent = getattr(reader, "format_name", None) == "Agilent MassHunter"
    levels = {s.ms_level for s in reader.scans} if agilent else {1, 2}
    if 1 in levels:
        cv(content, "MS:1000579", "MS1 spectrum")
    if levels - {1}:
        cv(content, "MS:1000580", "MSn spectrum")
    cv(content, "MS:1000128", "profile spectrum")
    cv(content, "MS:1000235", "total ion current chromatogram")
    dad = getattr(reader, "dad", None)
    if dad is not None:
        cv(content, "MS:1000806", "absorption spectrum")
    if agilent:
        _agilent_metadata(container, description, reader)
        return list(container)
    if hasattr(reader, 'sample_id'):
        user(content, "KmzConvert WIFF sample identifier", reader.sample_id)
    paths = [("WIFF", reader.wiff_path), ("WIFFSCAN", reader.scan_path)]
    if reader.entry.suffix.lower() == ".wiff2":
        paths.insert(0, ("WIFF2", reader.entry))
    sources = ET.SubElement(description, "sourceFileList", count=str(len(paths)))
    for ident, path in paths:
        source = ET.SubElement(sources, "sourceFile", id=ident, name=path.name, location=path.parent.as_uri())
        cv(source, "MS:1000770", "WIFF nativeID format")
        if ident == "WIFF2":
            cv(source, "MS:1003448", "SCIEX WIFF2 format")
        else:
            cv(source, "MS:1000562", "ABI WIFF format")
        with path.open("rb") as f:
            cv(source, "MS:1000569", "SHA-1", hashlib.file_digest(f, "sha1").hexdigest())
        if ident == "WIFF2":
            user(source, "KmzConvert source role", "Discovery/provenance only; opaque database not decoded")
        elif ident == "WIFFSCAN":
            user(source, "KmzConvert source role", "WIFF spectral payload sidecar")
    if dad is not None:
        sources.set("count", str(len(paths) + len(dad.sources)))
        for i, (path, digest) in enumerate(dad.sources):
            source = ET.SubElement(sources, "sourceFile", id=f"DAD{i}", name=path.name, location=path.parent.as_uri())
            cv(source, "MS:1000562", "ABI WIFF format")
            cv(source, "MS:1000770", "WIFF nativeID format")
            cv(source, "MS:1000569", "SHA-1", digest)
            user(source, "KmzConvert source role", "DAD spectral sidecar" if i == 0 else "DAD index sidecar")
    software = ET.SubElement(container, "softwareList", count="2")
    acquisition = ET.SubElement(software, "software", id="acquisition", version=reader.acquisition_software.removeprefix("SCIEX OS "))
    cv(acquisition, "MS:1003446", "SCIEX OS")
    converter = ET.SubElement(software, "software", id="kmzconvert_core", version=__version__)
    cv(converter, "MS:1000799", "custom unreleased software tool", "KmzConvert")
    instruments = ET.SubElement(container, "instrumentConfigurationList", count="2" if dad is not None else "1")
    instrument = ET.SubElement(instruments, "instrumentConfiguration", id="IC1")
    cv(instrument, "MS:1000121", "SCIEX instrument model")
    serial = reader.instrument.get("Serial Number")
    if serial:
        cv(instrument, "MS:1000529", "instrument serial number", serial)
    for key, val in reader.instrument.items():
        if key != "Serial Number":
            user(instrument, "SCIEX " + key, val)
    ET.SubElement(instrument, "softwareRef", ref="acquisition")
    if dad is not None:
        optical = ET.SubElement(instruments, "instrumentConfiguration", id="IC-DAD")
        cv(optical, "MS:1000031", "instrument model")
        user(optical, "KmzConvert instrument role", "Diode array detector; model not decoded")
        ET.SubElement(optical, "softwareRef", ref="acquisition")
    processing = ET.SubElement(container, "dataProcessingList", count="1")
    dp = ET.SubElement(processing, "dataProcessing", id="conversion")
    method = ET.SubElement(dp, "processingMethod", order="0", softwareRef="kmzconvert_core")
    cv(method, "MS:1000544", "Conversion to mzML")
    user(method, "KmzConvert decoding limitations", LIMITATIONS)
    if dad is not None:
        user(method, "KmzConvert DAD decoding", "sciex-os2-dad-v1; WIFF duplicate arrays/times/totals corroborated; native mAU divided by 1000 to AU")
    def correction_method(parent, order):
        method = ET.SubElement(parent, "processingMethod", order=str(order), softwareRef="kmzconvert_core")
        cv(method, "MS:1001485", "m/z calibration")
        for name, value in mass_correction.provenance_params().items():
            user(method, name, value)
    if mass_correction is not None:
        processing.set("count", "2")
        corrected = ET.SubElement(processing, "dataProcessing", id="mass-correction-v1")
        conversion = ET.SubElement(corrected, "processingMethod", order="0", softwareRef="kmzconvert_core")
        cv(conversion, "MS:1000544", "Conversion to mzML")
        correction_method(corrected, 1)
    if profile == "sparse":
        processing.set("count", "3" if mass_correction is not None else "2")
        sparse = ET.SubElement(processing, "dataProcessing", id=sparse_format)
        conversion = ET.SubElement(sparse, "processingMethod", order="0", softwareRef="kmzconvert_core")
        cv(conversion, "MS:1000544", "Conversion to mzML")
        if mass_correction is not None:
            correction_method(sparse, 1)
        method = ET.SubElement(sparse, "processingMethod", order="2" if mass_correction is not None else "1", softwareRef="kmzconvert_core")
        cv(method, "MS:1001486", "data filtering")
        if sparse_format == V1:
            cv(method, "MS:1000787", "inclusive low intensity threshold", 100, COUNTS)
        for name, value in METHOD_PARAMS[sparse_format].items():
            user(method, name, value)
    return list(container)


def _agilent_metadata(container, description, reader):
    """File, software, instrument and processing description of an Agilent MassHunter conversion (Full Profile)."""
    sources = ET.SubElement(description, "sourceFileList", count=str(len(reader.files)))
    for name, path in reader.files.items():
        ident = reader.source_ref if name == "MSScan.bin" else name.replace(".", "_")
        source = ET.SubElement(sources, "sourceFile", id=ident, name=path.name, location=path.parent.as_uri())
        cv(source, "MS:1001508", "Agilent MassHunter nativeID format")
        cv(source, "MS:1001509", "Agilent MassHunter format")
        with path.open("rb") as f:
            cv(source, "MS:1000569", "SHA-1", hashlib.file_digest(f, "sha1").hexdigest())
    software = ET.SubElement(container, "softwareList", count="2")
    acquisition = ET.SubElement(software, "software", id="acquisition", version=reader.acquisition_software or "unknown")
    cv(acquisition, "MS:1000678", "MassHunter Data Acquisition")
    converter = ET.SubElement(software, "software", id="kmzconvert_core", version=__version__)
    cv(converter, "MS:1000799", "custom unreleased software tool", "KmzConvert")
    instruments = ET.SubElement(container, "instrumentConfigurationList", count="1")
    instrument = ET.SubElement(instruments, "instrumentConfiguration", id="IC1")
    cv(instrument, "MS:1000490", "Agilent instrument model")
    if reader.instrument.get("Serial Number"):
        cv(instrument, "MS:1000529", "instrument serial number", reader.instrument["Serial Number"])
    for key, val in reader.instrument.items():
        if key != "Serial Number" and val:
            user(instrument, "Agilent " + key, val)
    ET.SubElement(instrument, "softwareRef", ref="acquisition")
    processing = ET.SubElement(container, "dataProcessingList", count="1")
    dp = ET.SubElement(processing, "dataProcessing", id="conversion")
    method = ET.SubElement(dp, "processingMethod", order="0", softwareRef="kmzconvert_core")
    cv(method, "MS:1000544", "Conversion to mzML")
    user(method, "KmzConvert decoding limitations", reader.limitations)
    user(method, "KmzConvert mass calibration source", reader.calibration_source)


def dad_element(dad, index, ident, rt):
    """A DAD spectrum without its arrays."""
    element = ET.Element("spectrum", index=str(index), id=ident, defaultArrayLength=str(len(dad.wavelength)))
    cv(element, "MS:1000806", "absorption spectrum")
    cv(element, "MS:1000128", "profile spectrum")
    cv(element, "MS:1000619", "lowest observed wavelength", dad.wavelength[0], NM)
    cv(element, "MS:1000618", "highest observed wavelength", dad.wavelength[-1], NM)
    scans = ET.SubElement(element, "scanList", count="1")
    cv(scans, "MS:1000795", "no combination")
    scan = ET.SubElement(scans, "scan", instrumentConfigurationRef="IC-DAD")
    cv(scan, "MS:1000016", "scan start time", rt, MINUTES)
    return element


def dad_id(dad, row):
    return f"sample={dad.sample_id} period=1 cycle={row+1} experiment=0"


def dad_spectrum(dad, row, index):
    element = dad_element(dad, index, dad_id(dad, row), dad.rt_min[row])
    arrays = ET.SubElement(element, "binaryDataArrayList", count="2")
    binary_array(arrays, dad.wavelength, "wavelength")
    binary_array(arrays, dad.absorbance_mau[row] / 1000., "absorbance")
    return element


def measured_arrays(reader, scan, mass_correction=None):
    """Arrays as written before any sparse subset: MS1 m/z on the corrected axis when enabled."""
    ms1 = reader.experiments[scan.experiment-1].ms_level == 1
    return native.reader_arrays(reader, scan, mass_correction if ms1 else None)


def sparse_plan(reader, mass_correction=None):
    """Sparse Profile v3 run-level selection, keyed by scan index.

    Each MS1 experiment's spectra form one sequence in file order. Only small
    per-spectrum region summaries are retained; arrays are decoded again when
    written, and the writer rejects any mismatch with this plan.
    """
    sequences = {}
    ms1 = [(index, scan) for index, scan in enumerate(reader.scans) if reader.experiments[scan.experiment-1].ms_level == 1]
    regions = native.ordered(lambda item: native.measured_regions(reader, item[1], mass_correction), ms1,
                             native.workers(reader), SPARSE_BATCH)
    with closing(regions):
        for (index, scan), found in zip(ms1, regions, strict=True):
            sequences.setdefault(scan.experiment, []).append((index, found))
    plan = {}
    select = native.rescue_selection if native.applies(reader) else rescue_selection
    for items in sequences.values():
        for (index, _), chosen in zip(items, select([regions for _, regions in items]), strict=True):
            plan[index] = chosen
    return plan


def spectrum_arrays(reader, scan, profile="full", mass_correction=None, *, sparse_format=FORMAT, selection=None):
    """Arrays as written, with the spectrum's dataProcessingRef and Sparse Profile check (or None)."""
    mz, intensity = measured_arrays(reader, scan, mass_correction)
    ms1 = reader.experiments[scan.experiment-1].ms_level == 1
    ref = check = None
    if mass_correction is not None and ms1:
        ref = "mass-correction-v1"
    if profile == "sparse" and ms1:
        indices, check = source_sampling_check(mz, intensity, sampling_family=getattr(reader, "sparse_profile_sampling_family", None),
                                               sparse_format=sparse_format, selection=selection)
        mz, intensity = mz[indices], intensity[indices]
        ref = sparse_format
    return mz, intensity, ref, check


def spectrum_element(scan, experiment, index, length, rt, ref=None, check=None):
    """An MS spectrum without its arrays; ``check`` is the canonical Sparse Profile check."""
    e = ET.Element("spectrum", index=str(index), id=scan.native_id, defaultArrayLength=str(length))
    if ref is not None:
        e.set("dataProcessingRef", ref)
    cv(e, "MS:1000511", "ms level", experiment.ms_level)
    cv(e, "MS:1000579" if experiment.ms_level == 1 else "MS:1000580", "MS1 spectrum" if experiment.ms_level == 1 else "MSn spectrum")
    if experiment.polarity == "negative":
        cv(e, "MS:1000129", "negative scan")
    else:
        cv(e, "MS:1000130", "positive scan")
    cv(e, "MS:1000128", "profile spectrum")
    cv(e, "MS:1000285", "total ion current", scan.tic)
    if check is not None:
        user(e, SAMPLING_PARAM, check)
    scans = ET.SubElement(e, "scanList", count="1")
    cv(scans, "MS:1000795", "no combination")
    acq = ET.SubElement(scans, "scan")
    cv(acq, "MS:1000016", "scan start time", rt, MINUTES)
    cv(acq, "MS:1000616", "preset scan configuration", getattr(scan, "preset", scan.experiment))
    windows = ET.SubElement(acq, "scanWindowList", count="1")
    window = ET.SubElement(windows, "scanWindow")
    cv(window, "MS:1000501", "scan window lower limit", experiment.lower_mz, MZ)
    cv(window, "MS:1000500", "scan window upper limit", experiment.upper_mz, MZ)
    if scan.precursor_mz is not None:
        precursors = ET.SubElement(e, "precursorList", count="1")
        precursor = ET.SubElement(precursors, "precursor")
        isolation = ET.SubElement(precursor, "isolationWindow")
        cv(isolation, "MS:1000827", "isolation window target m/z", scan.precursor_mz, MZ)
        ions = ET.SubElement(precursor, "selectedIonList", count="1")
        ion = ET.SubElement(ions, "selectedIon")
        cv(ion, "MS:1000744", "selected ion m/z", scan.precursor_mz, MZ)
        charge = getattr(scan, "precursor_charge", None)
        if charge is not None:
            cv(ion, "MS:1000041", "charge state", charge)
        # DDE also stores a signal value, but its intensity convention is not
        # confirmed. Preserve it in forensic reports rather than label it here.
        activation = ET.SubElement(precursor, "activation")
        # The PSI mapping explicitly permits this generic parent category.
        # It names no mechanism: CID/activation labels remain unrecovered.
        cv(activation, "MS:1000044", "dissociation method")
        if experiment.collision_energy is not None:
            cv(activation, "MS:1000045", "collision energy", experiment.collision_energy, EV)
        user(activation, "KmzConvert activation method", "Not decoded; generic CV category only")
    return e


def spectrum(reader, scan, index, profile="full", mass_correction=None, *, sparse_format=FORMAT, selection=None):
    mz, intensity, ref, check = spectrum_arrays(reader, scan, profile, mass_correction,
                                                sparse_format=sparse_format, selection=selection)
    rt = scan.rt_minutes if hasattr(scan, "rt_minutes") else scan.rt_seconds/60
    e = spectrum_element(scan, reader.experiments[scan.experiment-1], index, len(mz), rt,
                         ref, None if check is None else canonical(check))
    arrays = ET.SubElement(e, "binaryDataArrayList", count="2")
    binary_array(arrays, mz, "mz")
    binary_array(arrays, intensity, "intensity")
    return e


def _hole(name):
    return f"\0{name}\0"


class _Template:
    """Serialized text of an element built with ``_hole`` values, filled per spectrum.

    Attribute values are filled ``_escape_attrib``-escaped, as ElementTree escapes
    them. The binary and arrays holes take base64 and serialized array elements,
    which text escaping leaves unchanged.
    """

    def __init__(self, element, tail=b""):
        pieces = ET.tostring(element, encoding="utf-8").split(b"\0")
        self.text, self.names = pieces[0::2], [name.decode() for name in pieces[1::2]]
        self.text[-1] += tail

    def fill(self, parts, values):
        parts.append(self.text[0])
        for name, text in zip(self.names, self.text[1:]):
            value = values[name]
            if type(value) is list:
                parts += value
            else:
                parts.append(value)
            parts.append(text)


class _Holes:
    """Stands in for a scan while building its template."""

    def __init__(self, **values):
        self.__dict__.update(values)


def _attribute(value):
    return _escape_attrib(str(value)).encode("utf-8", "xmlcharrefreplace")


class Templates:
    """Production serialization: ``serialized(spectrum(...))`` and ``serialized(dad_spectrum(...))``
    bytes from templates of the same elements, compressing arrays from their buffers."""

    def __init__(self, reader):
        self.reader = reader
        self.spectra = {}
        self.kinds = {kind: _Template(array_element(kind, _hole("length"), _hole("binary")))
                      for kind in ("mz", "intensity", "wavelength", "absorbance")}
        dad = getattr(reader, "dad", None)
        if dad is not None:
            element = dad_element(dad, _hole("index"), _hole("id"), _hole("rt"))
            ET.SubElement(element, "binaryDataArrayList", count="2").text = _hole("arrays")
            self.dad = _Template(element, b"\n")
            self.wavelength = self.arrays(("wavelength", dad.wavelength))

    def arrays(self, *arrays):
        """Serialized binaryDataArray parts for (kind, array) pairs, compressed together."""
        encoded = native.deflate_base64(*(np.ascontiguousarray(array, dtype="<f8") for _, array in arrays))
        return self.encoded(*((kind, text) for (kind, _), text in zip(arrays, encoded, strict=True)))

    def encoded(self, *texts):
        """Serialized binaryDataArray parts for (kind, base64 zlib text) pairs."""
        parts = []
        for kind, text in texts:
            self.kinds[kind].fill(parts, {"length": str(len(text)).encode(), "binary": text})
        return parts

    def spectrum(self, scan, index, profile="full", mass_correction=None, selection=None):
        ms1 = self.reader.experiments[scan.experiment-1].ms_level == 1
        sparse = profile == "sparse" and ms1
        encoded = native.encode_spectrum(self.reader, scan, mass_correction if ms1 else None, selection, sparse)
        if encoded is None:
            mz, intensity, ref, check = spectrum_arrays(self.reader, scan, profile, mass_correction, selection=selection)
            length, arrays = len(mz), self.arrays(("mz", mz), ("intensity", intensity))
        else:
            length, mz, intensity, check = encoded
            ref = FORMAT if sparse else "mass-correction-v1" if mass_correction is not None and ms1 else None
            arrays = self.encoded(("mz", mz), ("intensity", intensity))
        key = (scan.experiment, scan.precursor_mz is None, ref, check is None)
        template = self.spectra.get(key)
        if template is None:
            holes = _Holes(native_id=_hole("id"), tic=_hole("tic"), experiment=scan.experiment,
                           precursor_mz=None if scan.precursor_mz is None else _hole("precursor"))
            element = spectrum_element(holes, self.reader.experiments[scan.experiment-1], _hole("index"), _hole("length"),
                                       _hole("rt"), ref, None if check is None else _hole("check"))
            ET.SubElement(element, "binaryDataArrayList", count="2").text = _hole("arrays")
            template = self.spectra[key] = _Template(element, b"\n")
        values = {"index": str(index).encode(), "id": _attribute(scan.native_id), "length": str(length).encode(),
                  "tic": _attribute(scan.tic), "rt": _attribute(scan.rt_seconds/60), "arrays": arrays}
        if check is not None:
            values["check"] = _attribute(canonical(check))
        if scan.precursor_mz is not None:
            values["precursor"] = _attribute(scan.precursor_mz)
        parts = []
        template.fill(parts, values)
        return b"".join(parts)

    def dad_spectrum(self, row, index):
        dad = self.reader.dad
        arrays = self.wavelength + self.arrays(("absorbance", dad.absorbance_mau[row] / 1000.))
        ident = dad_id(dad, row)
        parts = []
        self.dad.fill(parts, {"index": str(index).encode(), "id": _attribute(ident), "rt": _attribute(dad.rt_min[row]),
                              "arrays": arrays})
        return ident, b"".join(parts)


_UNSUPPORTED = {errno.ENOTSUP, errno.EOPNOTSUPP}


def _finalize_output(temporary, destination):
    """Prefer atomic no-clobber publication; copy exclusively without hard links."""
    try:
        os.link(temporary, destination)
        return
    except OSError as error:
        if error.errno not in _UNSUPPORTED:
            raise

    created = False
    try:
        with temporary.open("rb") as source, destination.open("xb") as target:
            created = True
            expected_size = os.fstat(source.fileno()).st_size
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            try:
                os.fsync(target.fileno())
            except OSError as error:
                if error.errno not in _UNSUPPORTED:
                    raise
            if os.fstat(target.fileno()).st_size != expected_size:
                raise OSError(errno.EIO, "Output copy size verification failed")
            source.seek(0)
            expected_digest = hashlib.file_digest(source, "sha256").digest()
        with destination.open("rb") as copied:
            if hashlib.file_digest(copied, "sha256").digest() != expected_digest:
                raise OSError(errno.EIO, "Output copy checksum verification failed")
    except BaseException as error:
        if created:
            try:
                destination.unlink()
            except OSError as cleanup_error:
                error.add_note(f"Incomplete output could not be removed at {destination}: {cleanup_error}")
        raise


def write_mzml(reader, destination, progress=None, *, profile="full", mass_correction=None, sparse_format=FORMAT):
    """Publish only a completely written file, never overwrite existing files.

    ``sparse_format`` exists for legacy-compatibility tests; production writes FORMAT.
    """
    if sparse_format not in METHOD_PARAMS:
        raise ValueError("Unknown Sparse Profile format")
    # Benchmarks wrap metadata/spectrum with their historical call shapes.
    legacy = {} if sparse_format == FORMAT else {"sparse_format": sparse_format}
    destination = output_path(destination, profile).absolute()
    if destination.is_symlink():
        raise FileExistsError(f"Refusing to write through output symlink {destination}")
    destination = destination.resolve()
    if getattr(reader, "format_name", None) == "Agilent MassHunter":
        # Full Profile only until other representations are qualified for Agilent data.
        if profile != "full" or mass_correction is not None:
            raise ValueError("Agilent data supports Full Profile without mass correction only")
        if destination.is_relative_to(reader.entry):
            raise ValueError(f"Refusing to write inside the source acquisition {reader.entry.name}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if destination.suffix.lower() != ".mzml":
        raise ValueError("Output must have an .mzML extension")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")
    temporary = None
    completed = False
    published = False
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=".kmzconvert-", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            digest = hashlib.sha1()
            def emit(data):
                f.write(data)
                digest.update(data)
            offsets = []
            emit(b'<?xml version="1.0" encoding="utf-8"?>\n')
            emit(b'<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">\n')
            emit(b'<mzML version="1.1.0" id="KmzConvert">\n')
            describe = lambda: (metadata(reader, profile, mass_correction, **legacy) if mass_correction is not None
                                else metadata(reader) if profile == "full" else metadata(reader, profile, **legacy))
            workers = native.workers(reader)
            dad = getattr(reader, "dad", None)
            total_spectra = len(reader.scans) + (len(dad) if dad is not None else 0)
            sparse_v3 = profile == "sparse" and sparse_format == V3
            # Threaded v3 conversions checksum the sources (metadata) during the plan pass.
            overlap = sparse_v3 and workers
            elements = None if overlap else describe()
            plan = {}
            if sparse_v3:
                # v3 links MS1 spectra across the run, so select before writing.
                if progress:
                    progress(0, total_spectra)
                with native.deferred_interrupts(), ThreadPoolExecutor(1, thread_name_prefix="kmzconvert-metadata") as helper:
                    described = helper.submit(describe) if overlap else None
                    plan = sparse_plan(reader, mass_correction)
                    elements = elements if described is None else described.result()
            for element in elements:
                emit(ET.tostring(element, encoding="utf-8") + b"\n")
            source_ref = getattr(reader, "source_ref", "WIFF")
            emit(f'<run id="run1" defaultInstrumentConfigurationRef="IC1" defaultSourceFileRef="{source_ref}">\n'.encode())
            emit(f'<spectrumList count="{total_spectra}" defaultDataProcessingRef="conversion">\n'.encode())
            # Production readers serialize through templates of the same elements;
            # other readers (and patched writer functions) keep stdlib serialization.
            templates = Templates(reader) if native.applies(reader) and not legacy else None
            tostring = lambda element: ET.tostring(element, encoding="utf-8") + b"\n"
            count = len(reader.scans)
            def encode(i):
                if i >= count:
                    if templates is not None:
                        return templates.dad_spectrum(i - count, i)
                    element = dad_spectrum(dad, i - count, i)
                    return element.get("id"), tostring(element)
                scan = reader.scans[i]
                if templates is not None:
                    return templates.spectrum(scan, i, profile, mass_correction, plan.get(i))
                chosen = {"selection": plan[i]} if i in plan else {}
                element = (spectrum(reader, scan, i, profile, mass_correction, **legacy, **chosen) if mass_correction is not None
                           else spectrum(reader, scan, i) if profile == "full"
                           else spectrum(reader, scan, i, profile, **legacy, **chosen))
                return tostring(element)
            overhangs = reader.nominal_window_overhangs if workers else {}
            # MS spectra, then DAD rows, as one stream so workers never drain between them.
            with closing(native.ordered(encode, range(total_spectra), workers, SPARSE_BATCH if profile == "sparse" else 1)) as encoded:
                for i, data in enumerate(encoded):
                    if i < count:
                        ident = reader.scans[i].native_id
                        if ident in overhangs:
                            # Workers record overhangs as they finish; keep file order.
                            overhangs[ident] = overhangs.pop(ident)
                        last = i + 1 == count
                    else:
                        ident, data = data
                        last = i + 1 == total_spectra
                    offsets.append((ident, f.tell()))
                    emit(data)
                    if progress and ((i if i < count else i - count) % 250 == 0 or last):
                        progress(i+1, total_spectra)
            emit(b'</spectrumList>\n<chromatogramList count="1" defaultDataProcessingRef="conversion">\n')
            times, intensities, levels = reader.tic_arrays()
            tic = ET.Element("chromatogram", index="0", id="TIC", defaultArrayLength=str(len(times)))
            cv(tic, "MS:1000235", "total ion current chromatogram")
            arrays = ET.SubElement(tic, "binaryDataArrayList", count="3")
            for a, kind in [(times, "time"), (intensities, "intensity"), (levels, "ms level")]:
                binary_array(arrays, a, kind)
            tic_offset = f.tell()
            emit(ET.tostring(tic, encoding="utf-8") + b"\n")
            emit(b'</chromatogramList>\n</run>\n</mzML>\n')
            index_offset = f.tell()
            indices = ET.Element("indexList", count="2")
            for kind, records in [("spectrum", offsets), ("chromatogram", [("TIC", tic_offset)])]:
                index = ET.SubElement(indices, "index", name=kind)
                for ident, offset in records:
                    ET.SubElement(index, "offset", idRef=ident).text = str(offset)
            # OpenMS's DOM index walker advances past the first child before
            # reading it. Leading whitespace keeps the first offset visible.
            ET.indent(indices, space="  ")
            emit(ET.tostring(indices, encoding="utf-8") + b"\n")
            emit(f'<indexListOffset>{index_offset}</indexListOffset>\n<fileChecksum>'.encode())
            # mzML hashes through the end of the opening fileChecksum tag.
            emit(digest.hexdigest().encode() + b'</fileChecksum>\n</indexedmzML>\n')
            f.flush()
            os.fsync(f.fileno())
        completed = True
        # Both paths refuse files/symlinks that appeared during conversion.
        # os.replace would clobber them, and ordinary rename can do so too.
        _finalize_output(temporary, destination)
        published = True
        temporary.unlink()
        temporary = None
    except BaseException as error:
        if completed and not isinstance(error, FileExistsError):
            detail = f"Completed temporary output retained at {temporary}."
            if published:
                detail += f" Final output is complete at {destination}; only temporary cleanup failed."
            error.add_note(detail)
            temporary = None  # Keep successful conversion work for recovery.
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
