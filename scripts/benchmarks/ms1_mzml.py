"""Isolated mzML experiments: preserve metadata/MS2, rebuild indices/checksum."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from unittest.mock import patch
import zlib
import base64

from lxml import etree
import numpy as np

from kmzconvert_core import writer
from kmzconvert_core.reference import read_arrays, NS
from scripts.benchmarks.ms1_algorithms import guard_indices

MARKER = "BENCHMARK ONLY: MS1 strict intensity >100; source-neighbor guards={guards}; no centroiding; vendor TIC retained"


class InspectionReader:
    """Adapter around the unchanged raw reader, modifying only returned MS1 arrays."""
    def __init__(self, reader, guards):
        self.reader, self.guards = reader, guards

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def arrays(self, scan):
        mz, intensity = self.reader.arrays(scan)
        if self.reader.experiments[scan.experiment - 1].ms_level != 1:
            return mz, intensity
        indices = guard_indices(intensity, guards=self.guards)
        return mz[indices], intensity[indices]


def write_raw_inspection(reader, destination, guards):
    original = writer.metadata
    def metadata(adapted):
        elements = original(adapted)
        processing = next(e for e in elements if e.tag == "dataProcessingList")
        writer.user(processing.find(".//processingMethod"), "KmzSuite spectrum benchmark", MARKER.format(guards=guards))
        return elements
    # This sequential standalone benchmark process restores the function even on
    # failure. No production source, CLI options or installed runtime is changed.
    with patch.object(writer, "metadata", metadata):
        return writer.write_mzml(InspectionReader(reader, guards), destination)


def _plain(element):
    # stdlib serialization drops inherited namespace declarations. The output
    # wrapper supplies the mzML default namespace; offsets point to plain tags.
    element = ET.fromstring(etree.tostring(element, with_tail=False))
    for node in element.iter():
        if isinstance(node.tag, str) and node.tag.startswith(NS):
            node.tag = node.tag[len(NS):]
    return ET.tostring(element, encoding="utf-8")


def _opening(tag, attributes):
    node = etree.Element(tag, attributes)
    return etree.tostring(node, encoding="utf-8")[:-2] + b">\n"


@contextmanager
def _exclusive_output(destination):
    with destination.open("xb") as output:
        try:
            yield output
        except BaseException:
            output.close()
            destination.unlink()
            raise


def filter_profile_mzml(source, destination, guards):
    """Stream a real profile mzML into a new indexed MS1 inspection copy.

    Metadata/chromatograms and other-level spectral values are retained. Numeric
    encoding stays at its source precision. No in-place writes or source edits.
    The experimental processing marker explicitly disclaims quantitative use.
    """
    destination = Path(destination)
    digest = hashlib.sha1()
    offsets = {"spectrum": [], "chromatogram": []}
    with _exclusive_output(destination) as output:
        def emit(data):
            output.write(data)
            digest.update(data)
        emit(b'<?xml version="1.0" encoding="utf-8"?>\n<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">\n')
        for event, element in etree.iterparse(str(source), events=("start", "end"), huge_tree=True):
            tag = etree.QName(element).localname
            parent = element.getparent()
            parent_tag = etree.QName(parent).localname if parent is not None else None
            if event == "start":
                if tag in ("mzML", "run", "spectrumList", "chromatogramList"):
                    emit(_opening(tag, dict(element.attrib)))
                continue
            if tag == "spectrum":
                level = element.find(NS + 'cvParam[@accession="MS:1000511"]')
                if level is not None and level.get("value") == "1":
                    arrays = read_arrays(element)
                    indices = guard_indices(arrays["intensity"], guards=guards)
                    element.set("defaultArrayLength", str(len(indices)))
                    for binary in element.findall(".//" + NS + "binaryDataArray"):
                        accessions = {p.get("accession") for p in binary.findall(NS + "cvParam")}
                        kind = "mz" if "MS:1000514" in accessions else "intensity" if "MS:1000515" in accessions else None
                        if kind is None:
                            raise ValueError("Aligned auxiliary spectrum arrays require a separate qualification")
                        raw = arrays[kind][indices].tobytes()
                        if "MS:1000574" in accessions:
                            raw = zlib.compress(raw)
                        encoded = base64.b64encode(raw).decode()
                        binary.set("encodedLength", str(len(encoded)))
                        if "arrayLength" in binary.attrib:
                            binary.set("arrayLength", str(len(indices)))
                        binary.find(NS + "binary").text = encoded
                offsets[tag].append((element.get("id"), output.tell()))
                emit(_plain(element) + b"\n")
            elif tag == "chromatogram":
                offsets[tag].append((element.get("id"), output.tell()))
                emit(_plain(element) + b"\n")
            elif tag in ("mzML", "run", "spectrumList", "chromatogramList"):
                emit(f"</{tag}>\n".encode())
            elif parent_tag in ("mzML", "run"):
                if tag == "dataProcessingList":
                    for method in element.findall(".//" + NS + "processingMethod"):
                        etree.SubElement(method, NS + "userParam", name="KmzSuite spectrum benchmark", value=MARKER.format(guards=guards))
                emit(_plain(element) + b"\n")
            else:
                continue
            element.clear()
            if parent is not None:
                while element.getprevious() is not None:
                    del parent[0]
        index_offset = output.tell()
        index_list = ET.Element("indexList", count=str(sum(bool(v) for v in offsets.values())))
        for tag, records in offsets.items():
            if records:
                index = ET.SubElement(index_list, "index", name=tag)
                for ident, offset in records:
                    ET.SubElement(index, "offset", idRef=ident).text = str(offset)
        ET.indent(index_list, space="  ")
        emit(ET.tostring(index_list, encoding="utf-8") + b"\n")
        emit(f"<indexListOffset>{index_offset}</indexListOffset>\n<fileChecksum>".encode())
        output.write(digest.hexdigest().encode() + b"</fileChecksum>\n</indexedmzML>\n")
    return destination
