"""Explicit qualification outputs through the unchanged full-profile writer."""
import json
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from lxml import etree

from kmzconvert_core import writer
from kmzconvert_core.reference import NS, read_arrays
from scripts.validation.check_mzml import check_index

from .models import correct_scan


class CorrectedReader:
    def __init__(self, reader, model):
        self.reader, self.model = reader, model

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def arrays(self, scan):
        mz, intensity = self.reader.arrays(scan)
        if self.reader.experiments[scan.experiment-1].ms_level != 1:
            return mz, intensity
        corrected, original_intensity, _ = correct_scan(self.model, mz, intensity, scan.rt_seconds/60)
        return corrected, original_intensity


def write_experimental(reader, destination, model):
    original = writer.metadata
    original_spectrum = writer.spectrum

    def metadata(adapted):
        elements = original(adapted)
        processing = next(e for e in elements if e.tag == "dataProcessingList")
        processing.set("count", str(int(processing.get("count"))+1))
        dp = ET.SubElement(processing,"dataProcessing",id="qualification-ms1-mass-correction")
        conversion = ET.SubElement(dp,"processingMethod",order="0",softwareRef="kmzconvert_core")
        writer.cv(conversion,"MS:1000544","Conversion to mzML")
        method = ET.SubElement(dp,"processingMethod",order="1",softwareRef="kmzconvert_core")
        writer.cv(method, "MS:1001485", "m/z calibration")
        writer.user(method, "KmzSuite mass correction qualification",
                    "EXPERIMENTAL: measured MS1 mass arrays only; MS2 and all acquisition metadata unchanged")
        writer.user(method, "KmzSuite qualification mass model", json.dumps(model.__dict__, sort_keys=True))
        writer.user(method, "KmzSuite qualification error convention", "observed=true*(1+ppm(rt,true_mz)/1e6); constant_da: observed=true+offset")
        return elements

    def spectrum(adapted, scan, index):
        element = original_spectrum(adapted,scan,index)
        if adapted.experiments[scan.experiment-1].ms_level == 1:
            element.set("dataProcessingRef","qualification-ms1-mass-correction")
        return element

    # A standalone sequential development driver; production functions are
    # restored even if writing fails. No CLI/GUI or writer source is modified.
    with patch.object(writer, "metadata", metadata), patch.object(writer, "spectrum", spectrum):
        path = writer.write_mzml(CorrectedReader(reader, model), destination)
    return path


def verify_experimental(reader, path, model):
    """Independent XML decode of EVERY output spectrum and TIC, plus indices."""
    by_id = {s.native_id: s for s in reader.scans}
    seen = set()
    levels = {1: 0, 2: 0}
    for _, element in etree.iterparse(str(path), events=("end",), huge_tree=True):
        if element.tag == NS+"spectrum":
            ident = element.get("id")
            if ident in seen or ident not in by_id:
                raise ValueError("Duplicate/unexpected spectrum identity")
            seen.add(ident)
            scan = by_id[ident]
            raw_m, raw_y = reader.arrays(scan)
            arrays = read_arrays(element)
            level = reader.experiments[scan.experiment-1].ms_level
            assert element.get("dataProcessingRef") == ("qualification-ms1-mass-correction" if level == 1 else None)
            expected = model.correct(raw_m, scan.rt_seconds/60) if level == 1 else raw_m
            np.testing.assert_array_equal(arrays["mz"], expected)
            np.testing.assert_array_equal(arrays["intensity"], raw_y)
            assert len(expected) == int(element.get("defaultArrayLength"))
            assert np.all(np.diff(arrays["mz"]) > 0)
            rt = float(element.find('.//'+NS+'cvParam[@accession="MS:1000016"]').get("value"))
            assert rt == scan.rt_seconds/60
            assert float(element.find(NS+'cvParam[@accession="MS:1000285"]').get("value")) == scan.tic
            exp = reader.experiments[scan.experiment-1]
            for accession, value in (("MS:1000501", exp.lower_mz), ("MS:1000500", exp.upper_mz),
                                     ("MS:1000827", scan.precursor_mz), ("MS:1000744", scan.precursor_mz)):
                item = element.find('.//'+NS+f'cvParam[@accession="{accession}"]')
                if value is not None:
                    assert item is not None and float(item.get("value")) == value
            levels[level] += 1
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
        elif element.tag == NS+"chromatogram":
            arrays = read_arrays(element)
            for kind, expected in zip(("time", "intensity", "ms level"), reader.tic_arrays()):
                np.testing.assert_array_equal(arrays[kind], expected)
            element.clear()
    assert seen == set(by_id)
    return dict(spectra_by_level=levels, all_arrays_and_metadata_verified=True, index=check_index(path))
