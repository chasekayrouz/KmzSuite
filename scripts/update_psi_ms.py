#!/usr/bin/env python3
"""Pin a PSI-MS controlled-vocabulary release for offline mzML reading.

Downloads the named HUPO-PSI release's psi-ms.obo, checks its data-version and
rewrites kmzcore/data/psi-ms.obo.gz (deterministic gzip), its attribution and
kmzcore.psi_ms.VERSION. Then run ./scripts/test.sh: Tests/test_psi_ms.py checks
that every accession the suite uses still resolves.

    .venv/bin/python scripts/update_psi_ms.py 4.2.18
"""
import argparse
import datetime
import gzip
import hashlib
import io
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Sources/KmzCore/kmzcore/data"
MODULE = ROOT / "Sources/KmzCore/kmzcore/psi_ms.py"
URL = "https://github.com/HUPO-PSI/psi-ms-CV/releases/download/v{0}/psi-ms.obo"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="PSI-MS release, for example 4.2.17")
    version = parser.parse_args().version
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        parser.error("version must look like 4.2.17")
    with urllib.request.urlopen(URL.format(version), timeout=60) as response:
        data = response.read()
    header = data[:4096].decode("utf-8", "replace")
    found = re.search(r"^data-version: (\S+)$", header, re.M)
    if not found or found.group(1) != version:
        raise SystemExit(f"The downloaded file is not PSI-MS release {version}")
    day, month, year = re.search(r"^date: (\d\d):(\d\d):(\d{4})", header, re.M).groups()
    from psims.controlled_vocabulary.controlled_vocabulary import ControlledVocabulary
    if ControlledVocabulary.from_obo(io.BytesIO(data)).version != version:
        raise SystemExit(f"psims does not read PSI-MS release {version} from the download")
    digest = hashlib.sha256(data).hexdigest()
    # Prepare every output before writing any, so a failure leaves the previous pin intact.
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0, compresslevel=9) as out:
        out.write(data)
    attribution = DATA / "PSI-MS-ATTRIBUTION.txt"
    text = attribution.read_text()
    previous = re.search(r"^data-version: (\S+)", text, re.M).group(1)
    text = text.replace(f"v{previous}", f"v{version}")
    text = re.sub(r"^data-version: .*$", f"data-version: {version} (released {year}-{month}-{day})", text, flags=re.M)
    text = re.sub(r"^[0-9a-f]{64}$", digest, text, flags=re.M)
    text = re.sub(r"^Retrieved: .*$", f"Retrieved: {datetime.date.today().isoformat()}", text, flags=re.M)
    module = re.sub(r'^VERSION = ".*"$', f'VERSION = "{version}"', MODULE.read_text(), flags=re.M)
    for path, content in ((DATA / "psi-ms.obo.gz", compressed.getvalue()), (attribution, text.encode()),
                          (MODULE, module.encode())):
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(content)
        os.replace(partial, path)
    print(f"Pinned PSI-MS {version} (SHA-256 {digest}). Run ./scripts/test.sh next.")


if __name__ == "__main__":
    main()
