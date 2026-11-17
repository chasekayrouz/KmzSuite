"""mzML oracle/validation utilities; never imported by the raw reader.

Binary arrays are decoded to numbers, independently of XML formatting.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np

NS = "{http://psi.hupo.org/ms/mzml}"


def cv(element):
    return {p.get("accession"): dict(p.attrib) for p in element.iter(NS + "cvParam")}


def value(params, accession, default=None):
    return params.get(accession, {}).get("value", default)


def read_arrays(element):
    result = {}
    for b in element.findall(".//" + NS + "binaryDataArray"):
        params = cv(b)
        kinds = [("MS:1000514", "mz"), ("MS:1000515", "intensity"),
                 ("MS:1000595", "time"), ("MS:1000617", "wavelength")]
        kind = next((name for accession, name in kinds if accession in params), "unknown")
        if "MS:1000523" in params:
            dtype = "<f8"
        elif "MS:1000521" in params:
            dtype = "<f4"
        elif "MS:1000522" in params:
            dtype = "<i8"
        elif "MS:1000519" in params:
            dtype = "<i4"
        else:
            raise ValueError(f"Unsupported binary numeric representation: {params}")
        raw = base64.b64decode(b.findtext(NS + "binary") or "", validate=True)
        if "MS:1000574" in params:
            raw = zlib.decompress(raw)
        elif "MS:1000576" not in params:
            raise ValueError("Unsupported mzML compression")
        if kind == "unknown" and "MS:1000786" in params:
            kind = params["MS:1000786"]["value"]
        result[kind] = np.frombuffer(raw, dtype=dtype)
    return result


def iter_records(path, tag="spectrum"):
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != NS + tag:
            if element.tag in (NS + 'spectrum', NS + 'chromatogram'):
                element.clear()
            continue
        params = cv(element)
        arrays = read_arrays(element)
        record = dict(element.attrib)
        record["cv"] = params
        record["ms_level"] = int(value(params, "MS:1000511", 0))
        rt = params.get("MS:1000016")
        record["rt_seconds"] = (float(rt["value"]) * (60 if rt.get("unitAccession") == "UO:0000031" else 1)) if rt else None
        record["polarity"] = "positive" if "MS:1000130" in params else "negative" if "MS:1000129" in params else None
        record["representation"] = "profile" if "MS:1000128" in params else "centroid" if "MS:1000127" in params else None
        record["precursors"] = [
            {"attributes": dict(p.attrib), "cv": cv(p)}
            for p in element.findall(".//" + NS + "precursor")
        ]
        record["arrays"] = arrays
        yield record
        element.clear()


def array_summary(a):
    return {"length": len(a), "dtype": str(a.dtype),
            "min": float(a.min()) if len(a) else None,
            "max": float(a.max()) if len(a) else None,
            "sum": float(a.sum(dtype=np.float64)),
            "zeros": int(np.count_nonzero(a == 0)),
            "sha256_le_f64": hashlib.sha256(a.astype("<f8").tobytes()).hexdigest(),
            "first": a[:5].tolist(), "last": a[-5:].tolist()}


def inventory(path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels, polarities, representations, windows = Counter(), Counter(), Counter(), Counter()
    n, points, first, last = 0, 0, None, None
    lengths, retention_times = [], []
    with (output / "spectra.jsonl").open("w") as f:
        for record in iter_records(path):
            arrays = record.pop("arrays")
            record["arrays"] = {k: array_summary(a) for k, a in arrays.items()}
            assert all(len(a) == int(record["defaultArrayLength"]) for a in arrays.values())
            n += 1
            points += len(arrays.get("mz", []))
            lengths.append(int(record["defaultArrayLength"]))
            retention_times.append(record["rt_seconds"])
            levels[record["ms_level"]] += 1
            polarities[record["polarity"]] += 1
            representations[record["representation"]] += 1
            windows[(value(record["cv"], "MS:1000501"), value(record["cv"], "MS:1000500"))] += 1
            first = first or record["id"]
            last = record["id"]
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    chromatograms = []
    for r in iter_records(path, "chromatogram"):
        r["arrays"] = {k: array_summary(a) for k, a in r["arrays"].items()}
        chromatograms.append(r)
    metadata = {}
    for event, e in ET.iterparse(path, events=("start", "end")):
        if event == "start" and e.tag == NS + "run":
            metadata["run"] = e.attrib
            break
        if event == "end" and e.tag in [NS + t for t in ("fileDescription", "instrumentConfigurationList", "softwareList", "dataProcessingList")]:
            metadata[e.tag[len(NS):]] = ET.tostring(e, encoding="unicode")
    summary = {"source": str(Path(path).resolve()), "spectrum_count": n,
               "ms_levels": levels, "polarities": polarities,
               "representations": representations, "total_mz_points": points,
               "array_length_range": [min(lengths), max(lengths)],
               "rt_seconds_range": [min(retention_times), max(retention_times)],
               "first_native_id": first, "last_native_id": last,
               "scan_windows": [{"bounds": list(k), "count": v} for k, v in windows.items()],
               "chromatograms": chromatograms, "metadata": metadata}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    result = inventory(sys.argv[1], sys.argv[2])
    print(json.dumps({k: v for k, v in result.items() if k not in ("metadata", "chromatograms")}, indent=2))
