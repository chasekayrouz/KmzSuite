"""The pinned PSI-MS vocabulary: provenance, offline loading and coverage of suite accessions."""
import gzip
import hashlib
import socket
import subprocess
from pathlib import Path

import pytest

from kmzcore import psi_ms

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_file_is_the_attributed_release():
    attribution = (psi_ms.PATH.parent / "PSI-MS-ATTRIBUTION.txt").read_text()
    digest = hashlib.sha256(gzip.decompress(psi_ms.PATH.read_bytes())).hexdigest()
    assert digest in attribution and f"v{psi_ms.VERSION}" in attribution
    assert f"data-version: {psi_ms.VERSION}" in attribution


def test_loads_once_offline(monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError("network access")
    monkeypatch.setattr(socket, "create_connection", offline)
    monkeypatch.setattr(psi_ms, "_vocabulary", None)
    first = psi_ms.vocabulary()
    assert first is psi_ms.vocabulary() and first.version == psi_ms.VERSION


def test_rejects_a_mismatched_release(monkeypatch):
    monkeypatch.setattr(psi_ms, "VERSION", "0.0.0")
    with pytest.raises(RuntimeError, match="expected 0.0.0"):
        psi_ms.load()


def test_every_accession_in_suite_code_resolves():
    """Accessions the converter writes and the readers recognize stay in the release."""
    paths = ["Apps/*.py", "Apps/*.c", "Sources/*.py", "Sources/*.c", "Tools/*.py"]
    source = subprocess.run(["git", "grep", "-h", "-o", "-E", r"MS:[0-9]{7}", "--", *paths],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    accessions = set(source.split())
    assert len(accessions) > 30
    terms = psi_ms.vocabulary().terms
    assert sorted(a for a in accessions if a not in terms) == []
