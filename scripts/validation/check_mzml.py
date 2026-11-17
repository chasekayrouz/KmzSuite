"""Validate mzML against a local official XSD and PSI-MS controlled vocabulary."""
import json
import hashlib
from pathlib import Path
import re
import sys

from lxml import etree


def check_index(path):
    """Independently verify each indexed byte offset and the specified SHA-1."""
    with Path(path).open('rb') as f:
        header = f.read(4096)
        if b'<indexedmzML' not in header:
            return {'indexed': False}
        f.seek(max(0, Path(path).stat().st_size-4096))
        tail_start = f.tell()
        tail = f.read()
        offset_match = re.search(rb'<indexListOffset>([0-9]+)</indexListOffset>', tail)
        checksum_match = re.search(rb'<fileChecksum>([0-9a-f]{40})</fileChecksum>', tail)
        if offset_match is None or checksum_match is None:
            raise ValueError('Missing indexed mzML footer')
        index_offset = int(offset_match[1])
        f.seek(index_offset)
        # The index is small metadata, not the spectral payload.
        data = f.read(tail_start + offset_match.start() - index_offset)
        if not data.startswith(b'<indexList '):
            raise ValueError('indexListOffset does not point to indexList')
        indices = etree.fromstring(data)
        assert int(indices.get('count')) == len(indices)
        counts = {}
        seen = set()
        for index in indices:
            kind = index.get('name')
            assert kind in ('spectrum', 'chromatogram') and kind not in counts
            counts[kind] = len(index)
            previous = -1
            for item in index:
                offset, ident = int(item.text), item.get('idRef')
                assert previous < offset < index_offset and (kind, ident) not in seen
                previous = offset
                seen.add((kind, ident))
                f.seek(offset)
                opening = f.read(4096).split(b'>', 1)[0] + b'/>'
                element = etree.fromstring(opening)
                assert element.tag == kind and element.get('id') == ident, 'Indexed ID/offset mismatch'
        end = tail_start + checksum_match.start(1)
        f.seek(0)
        digest = hashlib.sha1()
        remaining = end
        while remaining:
            chunk = f.read(min(1024*1024, remaining))
            if not chunk:
                raise ValueError('Truncated indexed mzML')
            digest.update(chunk)
            remaining -= len(chunk)
        assert digest.hexdigest() == checksum_match[1].decode(), 'mzML checksum mismatch'
        return {'indexed': True, 'offsets_verified': counts, 'checksum_valid': True}


def check(path, schema_path, obo_path):
    terms = {}
    for block in Path(obo_path).read_text().split("[Term]"):
        ident = re.search(r"^id: (\S+)$", block, re.M)
        name = re.search(r"^name: (.+)$", block, re.M)
        if ident and name:
            terms[ident[1]] = name[1]
    schema = etree.XMLSchema(etree.parse(str(schema_path)))
    problems, seen = set(), set()
    for _, element in etree.iterparse(str(path), events=("end",), schema=schema, huge_tree=True):
        if etree.QName(element).localname == "cvParam":
            accession, name = element.get("accession"), element.get("name")
            seen.add(accession)
            if accession.startswith("MS:") and terms.get(accession) != name:
                problems.add((accession, name, terms.get(accession)))
            unit = element.get("unitAccession")
            if unit and unit.startswith("MS:") and terms.get(unit) != element.get("unitName"):
                problems.add((unit, element.get("unitName"), terms.get(unit)))
        if etree.QName(element).localname in ("spectrum", "chromatogram"):
            element.clear()
    return {"xsd_valid": True, "psi_ms_names_valid": not problems, 'index': check_index(path),
            "unique_cv_terms": sorted(seen), "cv_problems": sorted(problems),
            "scope": "Official mzML 1.1.0 XSD plus PSI-MS accessions/names; not the complete PSI semantic mapping validator."}


if __name__ == "__main__":
    result = check(*sys.argv[1:4])
    Path(sys.argv[4]).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["psi_ms_names_valid"] else 1)
