"""DreaMS upload privacy, service failure semantics, and public-cache science."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from kmzcore.cancel import Cancelled
from kmzviewer.dreams_client import (
    DreamsError, REFERENCE_BYTES, SPACE, SPACE_INFO, SPACE_REVISION,
    _safe_url, ensure_reference, query_mgf, remote_embedding, search_dreams,
)
from kmzviewer.dreams_reference import DIMENSIONS, ReferenceError, build_reference, search_reference


def query(**overrides):
    value = dict(precursor_mz=300.1234, charge=None, polarity=1,
                 peaks=[[100., 5.], [130., 100.], [144., 20.], [150., .5]],
                 representation="centroid", preparation="Native centroid values",
                 source_name="private research.mzML", path="/private/secret/data.mzML", scan_id="secret_scan")
    value.update(overrides)
    return value


def test_payload_is_spectrum_only_and_unknown_charge_stays_unknown():
    value = query()
    before = copy.deepcopy(value)
    mgf, preparation = query_mgf(value)
    assert value == before
    assert b"CHARGE" not in mgf
    assert not any(word in mgf for word in (b"private", b"secret", b"mzML", b"TITLE", b"RTINSECONDS"))
    assert b"150.0000000000" not in mgf
    assert preparation["submitted_peaks"] == 3
    assert preparation["input_peaks"] == 4
    assert b"CHARGE=1+" in query_mgf(query(charge=1))[0]


@pytest.mark.parametrize("change", [dict(polarity=-1), dict(polarity=0), dict(charge=2),
                                     dict(precursor_mz=float("nan")), dict(peaks=[[100., 2.]])])
def test_unsupported_input_rejected_before_any_network(change, tmp_path):
    calls = []
    def transport(*args, **kwargs):
        calls.append(args)
        raise AssertionError("network should not be used")
    with pytest.raises(ValueError):
        search_dreams(query(**change), tmp_path, transport=transport)
    assert not calls


def fake_transport(*, event="complete", result_url=None, embedding=None, version=SPACE_REVISION, error=None):
    calls = []
    embedding = embedding if embedding is not None else [1.] + [0.] * (DIMENSIONS - 1)
    library_path = "/tmp/gradio/" + "a" * 64 + "/MassSpecGym_DreaMS.hdf5"
    upload_path = "/tmp/gradio/" + "b" * 64 + "/query.mgf"
    result_url = result_url or SPACE + "/gradio_api/file=/tmp/gradio/" + "c" * 64 + "/query_hits.tsv"
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["DreaMS_embedding", "DreaMS_similarity", "ref_IDENTIFIER"], delimiter="\t")
    writer.writeheader()
    writer.writerow(dict(DreaMS_embedding=json.dumps(embedding), DreaMS_similarity=.9, ref_IDENTIFIER="MassSpecGymID000001"))

    def transport(url, **kwargs):
        calls.append((url, kwargs))
        if url == SPACE_INFO:
            body = json.dumps({"sha": version}).encode()
        elif url.endswith("/config"):
            body = json.dumps({"components": [{"type": "file", "props": {"value": {"path": library_path,
                "orig_name": "MassSpecGym_DreaMS.hdf5", "size": REFERENCE_BYTES}}}],
                "dependencies": [{"id": 2, "api_name": "predict"}]}).encode()
        elif url.endswith("/upload"):
            body = json.dumps([upload_path]).encode()
        elif url.endswith("/queue/join"):
            body = json.dumps({"event_id": "d" * 32}).encode()
        elif "/queue/data?" in url:
            value = [{}, {"value": {"url": result_url}}] if event == "complete" else None
            response = dict(msg="process_completed", event_id="d" * 32, success=event == "complete",
                            output=dict(data=value, error=error))
            body = ("data: " + json.dumps(response) + "\n\n").encode()
        else:
            assert url == result_url
            body = output.getvalue().encode()
        return dict(status=200, body=body)
    return transport, calls


def test_remote_protocol_only_uploads_generic_mgf_and_extracts_embedding():
    transport, calls = fake_transport()
    result = remote_embedding(query(), transport=transport)
    assert result["embedding"].shape == (DIMENSIONS,)
    assert result["remote_accession"] == "MassSpecGymID000001"
    bodies = [kwargs["body"] for _, kwargs in calls if kwargs.get("body")]
    assert len(bodies) == 2
    assert b'filename="query.mgf"' in bodies[0]
    assert not any(token in b" ".join(bodies) for token in (b"private", b"secret", b"mzML", b"scan_id"))
    request = json.loads(bodies[1])
    assert request["data"][2:] == [.5, False, False]


def test_remote_error_is_not_silently_reported_as_zero_matches():
    transport, _ = fake_transport(event="error")
    with pytest.raises(DreamsError, match="could not complete"):
        remote_embedding(query(), transport=transport)


def test_public_gpu_quota_failure_has_specific_actionable_message():
    transport, _ = fake_transport(event="error", error="You have exceeded your ZeroGPU runs limit.")
    with pytest.raises(DreamsError, match="free public GPU quota is exhausted"):
        remote_embedding(query(), transport=transport)


def test_changed_service_version_fails_before_upload():
    transport, calls = fake_transport(version="changed")
    with pytest.raises(DreamsError, match="version has changed"):
        remote_embedding(query(), transport=transport)
    assert len(calls) == 1


@pytest.mark.parametrize("url", ["http://anton-bushuiev-dreams.hf.space/x", "https://attacker.invalid/x",
                                 "https://anton-bushuiev-dreams.hf.space.attacker.invalid/x",
                                 "https://user:secret@anton-bushuiev-dreams.hf.space/x",
                                 "https://huggingface.co:444/x"])
def test_service_urls_do_not_allow_host_or_scheme_escape(url):
    with pytest.raises(DreamsError):
        _safe_url(url)


def test_result_file_is_confined_to_expected_service_and_generic_name():
    transport, calls = fake_transport(result_url="https://attacker.invalid/leak.tsv")
    with pytest.raises(DreamsError):
        remote_embedding(query(), transport=transport)
    assert all(not url.startswith("https://attacker") for url, _ in calls)


@pytest.mark.parametrize("embedding", [[1., 2.], [float("nan")] * DIMENSIONS, [0.] * DIMENSIONS])
def test_invalid_remote_embeddings_are_not_scored(embedding):
    transport, _ = fake_transport(embedding=embedding)
    with pytest.raises(DreamsError, match="invalid learned"):
        remote_embedding(query(), transport=transport)


def reference_fixture(tmp_path):
    source = tmp_path / "public-fixture.hdf5"
    embedding = np.zeros((3, DIMENSIONS), dtype=np.float32)
    embedding[0, :2] = [1., 0.]
    embedding[1, :2] = [.9, .1]
    embedding[2, :2] = [.6, .8]
    spectra = np.array([[[100., 144., 0.], [10., 20., 0.]],
                        [[120., 144., 0.], [5., 40., 0.]],
                        [[50., 150., 0.], [8., 30., 0.]]])
    with h5py.File(source, "w") as file:
        file["DreaMS_embedding"] = embedding
        file["spectrum"] = spectra
        string = h5py.string_dtype("utf-8")
        for key, values in dict(IDENTIFIER=["MassSpecGymID000001", "MassSpecGymID000002", "MassSpecGymID000003"],
                                INCHIKEY=["same", "same", "different"], name=["Reference one", "", "Reference two"],
                                smiles=["CC", "CC", "CCC"], adduct=["[M+H]+"] * 3).items():
            file.create_dataset(key, data=np.asarray(values, dtype=object), dtype=string)
        file["precursor_mz"] = [200., 200., 400.]
        file["COLLISION_ENERGY"] = [10., np.nan, 30.]
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_cache_preserves_native_masses_and_deduplicates_only_after_blind_ranking(tmp_path):
    source, checksum = reference_fixture(tmp_path)
    target = tmp_path / "cache"
    manifest = build_reference(source, target, expected_sha256=checksum, expected_count=3)
    assert manifest["count"] == 3
    result = search_reference([1.] + [0.] * (DIMENSIONS - 1), target)
    assert [h["accession"] for h in result["hits"]] == ["MassSpecGymID000001", "MassSpecGymID000003"]
    assert result["hits"][1]["scores"]["dreams"] == pytest.approx(.6, abs=.0002)
    assert result["references"]["MassSpecGymID000001"]["peaks"] == [[100., 10.], [144., 20.]]
    assert source.exists()
    assert not list(tmp_path.glob(".*building*"))


def test_wrong_download_checksum_and_cancel_preserve_existing_cache(tmp_path):
    source, checksum = reference_fixture(tmp_path)
    target = tmp_path / "cache"
    build_reference(source, target, expected_sha256=checksum, expected_count=3)
    original = (target / "manifest.json").read_bytes()
    with pytest.raises(ReferenceError, match="checksum"):
        build_reference(source, target, expected_sha256="0" * 64, expected_count=3)
    with pytest.raises(Cancelled):
        build_reference(source, target, expected_sha256=checksum, expected_count=3, cancel=lambda: True)
    assert (target / "manifest.json").read_bytes() == original
    with pytest.raises(Cancelled):
        search_reference([1.] * DIMENSIONS, target, cancel=lambda: True)
    stopped = [False]
    def progress(message):
        if message.startswith("Preparing"):
            stopped[0] = True
    with pytest.raises(Cancelled):
        build_reference(source, target, expected_sha256=checksum, expected_count=3,
                        cancel=lambda: stopped[0], progress=progress)
    assert (target / "manifest.json").read_bytes() == original
    assert not list(tmp_path.glob(".*building*"))


def test_untrusted_download_redirect_does_not_escape_or_leave_partial_data(tmp_path):
    calls = []
    def transport(url, **kwargs):
        calls.append(url)
        Path(kwargs["output"]).write_bytes(b"redirect")
        return dict(status=302, location="https://attacker.invalid/reference.hdf5")
    with pytest.raises(DreamsError, match="Unrecognized"):
        ensure_reference(tmp_path, transport=transport)
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.hdf5"))


def test_cancelled_download_does_not_leave_partial_public_cache(tmp_path):
    def transport(url, **kwargs):
        Path(kwargs["output"]).write_bytes(b"partial")
        raise Cancelled()
    with pytest.raises(Cancelled):
        ensure_reference(tmp_path, transport=transport)
    assert not list(tmp_path.glob("*.hdf5"))


def test_corrupt_cache_is_rejected_before_returning_candidates(tmp_path):
    source, checksum = reference_fixture(tmp_path)
    target = tmp_path / "cache"
    build_reference(source, target, expected_sha256=checksum, expected_count=3)
    path = target / "embeddings.npy"
    with path.open("r+b") as stream:
        stream.seek(-2, 2)
        stream.write(b"xx")
    with pytest.raises(ReferenceError, match="integrity"):
        search_reference([1.] * DIMENSIONS, target)


def test_model_reference_drift_fails_instead_of_mislabeling_matches(tmp_path, monkeypatch):
    monkeypatch.setattr("kmzviewer.dreams_client.ensure_reference", lambda *a, **kw: tmp_path)
    monkeypatch.setattr("kmzviewer.dreams_client.remote_embedding", lambda *a, **kw: dict(
        embedding=np.ones(DIMENSIONS), remote_accession="MassSpecGymID000001", remote_score=.9,
        preparation={}, source_revision=SPACE_REVISION))
    monkeypatch.setattr("kmzviewer.dreams_client.search_reference", lambda *a, **kw: dict(
        hits=[dict(accession="MassSpecGymID000002", scores={"dreams": .9})], references={}, provenance={}))
    with pytest.raises(DreamsError, match="no longer agree"):
        search_dreams(query(), tmp_path)


def test_validated_float16_ties_can_select_another_spectrum_of_same_structure(tmp_path, monkeypatch):
    monkeypatch.setattr("kmzviewer.dreams_client.ensure_reference", lambda *a, **kw: tmp_path)
    monkeypatch.setattr("kmzviewer.dreams_client.remote_embedding", lambda *a, **kw: dict(
        embedding=np.ones(DIMENSIONS), remote_accession="MassSpecGymID000001", remote_score=.9,
        remote_inchikey="same", preparation={}, source_revision=SPACE_REVISION))
    monkeypatch.setattr("kmzviewer.dreams_client.search_reference", lambda *a, **kw: dict(
        hits=[dict(accession="MassSpecGymID000002", inchikey="same", scores={"dreams": .90001})],
        references={}, provenance={}))
    assert search_dreams(query(), tmp_path)["hits"][0]["accession"] == "MassSpecGymID000002"
