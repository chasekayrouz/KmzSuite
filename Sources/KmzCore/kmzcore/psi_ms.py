"""The suite's pinned PSI-MS controlled vocabulary, loaded once without network access.

Pyteomics types valued PSI-MS cvParams from this vocabulary and rejects accessions
it lacks, so the release decides which mzML files read. Shipping one release keeps
reading offline and identical across KmzViewer, Kmz2Viewer and KmzCoMet. Refresh it with
``scripts/update_psi_ms.py``.
"""
import gzip
from pathlib import Path
import threading

VERSION = "4.2.17"
URI = "http://purl.obolibrary.org/obo/ms/psi-ms.obo"
PATH = Path(__file__).with_name("data") / "psi-ms.obo.gz"

_vocabulary = None
_lock = threading.Lock()


def vocabulary():
    """Return the shared vocabulary; the lock covers simultaneous first reads."""
    global _vocabulary
    with _lock:
        if _vocabulary is None:
            _vocabulary = load()
        return _vocabulary


def load():
    """Parse the pinned release. Imports, if any, resolve to psims' bundled copies."""
    # psims takes ~0.1 s to import, so it loads with the first mzML read.
    from psims.controlled_vocabulary.controlled_vocabulary import ControlledVocabulary, OBOCache
    imports = OBOCache(enabled=False, use_remote=False)
    with gzip.open(PATH, "rb") as handle:
        result = ControlledVocabulary.from_obo(handle, import_resolver=imports.load)
    if result.version != VERSION:
        raise RuntimeError(f"Packaged PSI-MS vocabulary is {result.version}, expected {VERSION}")
    return result
