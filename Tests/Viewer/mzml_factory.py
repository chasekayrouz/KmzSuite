"""Tiny mzML writer for tests. Peak arrays are zlib-compressed float64."""

from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np


def _precursor_xml(spectrum: dict) -> str:
    if spectrum.get("precursor_mz") is None:
        return ""
    ions = [
        f'<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="{spectrum["precursor_mz"]}"/>'
    ]
    if spectrum.get("charge") is not None:
        ions.append(
            f'<cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="{spectrum["charge"]}"/>'
        )
    if spectrum.get("precursor_intensity") is not None:
        ions.append(
            '<cvParam cvRef="MS" accession="MS:1000042" name="peak intensity" '
            f'value="{spectrum["precursor_intensity"]}"/>'
        )
    activation = ""
    if spectrum.get("collision_energy") is not None:
        activation = (
            "<activation>"
            '<cvParam cvRef="MS" accession="MS:1000045" name="collision energy" '
            f'value="{spectrum["collision_energy"]}"/>'
            "</activation>"
        )
    return (
        '<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>'
        + "".join(ions)
        + "</selectedIon></selectedIonList>"
        + activation
        + "</precursor></precursorList>"
    )


def encode_float64(values: np.ndarray) -> str:
    raw = np.ascontiguousarray(values, dtype="<f8").tobytes()
    return base64.b64encode(zlib.compress(raw)).decode("ascii")


def write_mzml(path: str | Path, spectra: list[dict], chromatogram: bool = False) -> None:
    blocks = []
    for index, spectrum in enumerate(spectra):
        blocks.append(_spectrum_xml(index, spectrum))
    chrom = _chromatogram_xml() if chromatogram else ""
    document = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run id="run">
    <spectrumList count="{len(spectra)}">
      {''.join(blocks)}
    </spectrumList>
    {chrom}
  </run>
</mzML>
"""
    Path(path).write_text(document, encoding="utf-8")


def _spectrum_xml(index: int, spectrum: dict) -> str:
    mz = np.asarray(spectrum["mz"], dtype=np.float64)
    intensity = np.asarray(spectrum["intensity"], dtype=np.float64)
    mz_text = encode_float64(mz)
    intensity_text = encode_float64(intensity)
    level = int(spectrum.get("ms_level", 1))
    polarity = spectrum.get("polarity", 1)
    polarity_xml = ""
    if polarity == 1:
        polarity_xml = '<cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>'
    elif polarity == -1:
        polarity_xml = '<cvParam cvRef="MS" accession="MS:1000129" name="negative scan" value=""/>'
    mode = spectrum.get("centroid", False)
    mode_xml = (
        '<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>'
        if mode
        else '<cvParam cvRef="MS" accession="MS:1000128" name="profile spectrum" value=""/>'
    )
    tic_xml = ""
    if "tic" in spectrum and spectrum["tic"] is not None:
        tic_xml = (
            f'<cvParam cvRef="MS" accession="MS:1000285" name="total ion current" '
            f'value="{spectrum["tic"]}"/>'
        )
    unit = spectrum.get("rt_unit", "minute")
    if unit == "second":
        unit_attrs = 'unitCvRef="UO" unitAccession="UO:0000010" unitName="second"'
    else:
        unit_attrs = 'unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"'
    scan_id = spectrum.get("id", f"scan={index}")
    precursor_xml = _precursor_xml(spectrum)
    return f"""
      <spectrum index="{index}" id="{scan_id}" defaultArrayLength="{mz.size}">
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{level}"/>
        {mode_xml}
        {polarity_xml}
        {tic_xml}
        <scanList count="1">
          <scan>
            <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{spectrum["rt"]}" {unit_attrs}/>
          </scan>
        </scanList>
        {precursor_xml}
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="{len(mz_text)}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value="" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>
            <binary>{mz_text}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="{len(intensity_text)}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value="" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>
            <binary>{intensity_text}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </spectrum>
"""


def _chromatogram_xml() -> str:
    time = encode_float64(np.array([0.0, 1.0, 2.0]))
    intensity = encode_float64(np.array([9.0, 9.0, 9.0]))
    return f"""
    <chromatogramList count="1">
      <chromatogram index="0" id="TIC" defaultArrayLength="3">
        <cvParam cvRef="MS" accession="MS:1000235" name="total ion current chromatogram" value=""/>
        <binaryDataArrayList count="2">
          <binaryDataArray encodedLength="{len(time)}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000595" name="time array" value="" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>
            <binary>{time}</binary>
          </binaryDataArray>
          <binaryDataArray encodedLength="{len(intensity)}">
            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
            <binary>{intensity}</binary>
          </binaryDataArray>
        </binaryDataArrayList>
      </chromatogram>
    </chromatogramList>
"""
