"""Optional DreaMS learned-neighbor search with an explicit spectrum-only upload.

Public reference data may be cached. Query MGF, remote result and embedding stay
in memory; persistence belongs to the user's explicit Save action in the GUI.
"""
from __future__ import annotations

import csv
import fcntl
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlsplit
import uuid

import numpy as np

from kmzcore.cancel import Cancelled
from kmzviewer.gnps import check_cancel, timestamp, validate_peaks
from kmzviewer.dreams_reference import (
    DIMENSIONS, REFERENCE_BYTES, REFERENCE_SHA256, REFERENCE_URL,
    ReferenceError, build_reference, cache_path, cache_ready, search_reference,
)

SPACE = "https://anton-bushuiev-dreams.hf.space"
SPACE_PAGE = "https://huggingface.co/spaces/anton-bushuiev/DreaMS"
SPACE_INFO = "https://huggingface.co/api/spaces/anton-bushuiev/DreaMS"
SPACE_REVISION = "c0baf9817879d36b1f07fc6c818bd3ef50283db6"
DOWNLOAD_HOSTS = {"huggingface.co", "us.aws.cdn.hf.co", "cdn-lfs.huggingface.co",
                  "cdn-lfs-us-1.hf.co", "cas-bridge.xethub.hf.co"}
HOSTS = DOWNLOAD_HOSTS | {"anton-bushuiev-dreams.hf.space"}


class DreamsError(ValueError):
    pass


def _safe_url(url, *, upload=False):
    try:
        parts = urlsplit(url)
        valid = (parts.scheme == "https" and parts.hostname in HOSTS and not parts.username
                 and not parts.password and parts.port in {None, 443})
    except (TypeError, ValueError):
        valid = False
    if not valid or (upload and parts.hostname != "anton-bushuiev-dreams.hf.space"):
        raise DreamsError("Unrecognized DreaMS service address.")
    return parts


def qt_http(url, *, body=None, content_type=None, cancel=None, timeout=45.,
            max_bytes=16 * 1024 * 1024, output=None, progress=None, sse=False):
    """One bounded request, with manual redirects and streaming public downloads."""
    from PySide6.QtCore import QByteArray, QEventLoop, QTimer, QUrl
    from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
    import shiboken6

    _safe_url(url, upload=body is not None)
    check_cancel(cancel)
    stream = Path(output).open("wb") if output else None  # Before any Qt object exists.
    manager = QNetworkAccessManager()
    request = QNetworkRequest(QUrl(url))
    request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                         QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
    request.setRawHeader(b"User-Agent", b"KmzViewer-DreaMS/1")
    if content_type:
        request.setRawHeader(b"Content-Type", content_type.encode("ascii"))
    reply = manager.post(request, QByteArray(body)) if body is not None else manager.get(request)
    loop, timer = QEventLoop(), QTimer()
    timer.setInterval(40)
    started, last_progress = time.monotonic(), [0.]
    received, total, reasons = bytearray(), [0], []
    event_buffer, completed_early = bytearray(), [False]

    def read():
        chunk = bytes(reply.readAll())
        total[0] += len(chunk)
        if total[0] > max_bytes:
            reasons.append("DreaMS returned an oversized response.")
            reply.abort()
            return
        try:
            if stream:
                stream.write(chunk)
            else:
                received.extend(chunk)
            if sse:
                event_buffer.extend(chunk)
                while b"\n" in event_buffer:
                    line, _, remaining = event_buffer.partition(b"\n")
                    event_buffer[:] = remaining
                    if not line.startswith(b"data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(event, dict) and event.get("msg") in {"process_completed", "unexpected_error", "close_stream"}:
                        completed_early[0] = True
                        reply.abort()
                        break
            if progress and time.monotonic() - last_progress[0] >= .25:
                last_progress[0] = time.monotonic()
                progress(f"Downloading public DreaMS references · {total[0] / 1e6:.0f} / {REFERENCE_BYTES / 1e6:.0f} MB…")
        except OSError:
            reasons.append("The DreaMS reference download could not be written.")
            reply.abort()

    def tick():
        if cancel is not None and cancel():
            reasons.append("cancel")
            reply.abort()
        elif time.monotonic() - started > timeout:
            reasons.append("DreaMS request timed out. Try again later.")
            reply.abort()

    try:
        reply.readyRead.connect(read)
        reply.finished.connect(loop.quit)
        timer.timeout.connect(tick)
        timer.start()
        if not reply.isFinished():
            loop.exec()
        timer.stop()
        if reply.isOpen():
            read()
        check_cancel(cancel)
        if reasons:
            if reasons[0] == "cancel":
                raise Cancelled()
            raise DreamsError(reasons[0])
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if status not in {200, 301, 302, 303, 307, 308}:
            raise DreamsError(f"DreaMS service returned HTTP {status or 'unavailable'}. Try again later.")
        if reply.error() != QNetworkReply.NetworkError.NoError and not completed_early[0]:
            raise DreamsError("Could not reach DreaMS securely. Check your connection and try again.")
        return dict(status=status, body=bytes(received), bytes=total[0],
                    location=bytes(reply.rawHeader("Location")).decode("utf-8", errors="strict"))
    finally:
        timer.stop()
        # Delete these worker-thread Qt objects here, even if a traceback keeps this frame
        # alive: the GUI-thread garbage collector must never be the one to destroy them.
        for item in (timer, loop, manager):  # The reply is the manager's child.
            shiboken6.delete(item)
        if stream:
            stream.close()


def _read(url, *, transport=qt_http, **kwargs):
    response = transport(url, **kwargs)
    if response.get("status") != 200:
        raise DreamsError("DreaMS redirected an analysis request. No redirect was followed.")
    return response["body"]


def _json(url, *, payload=None, transport=qt_http, **kwargs):
    if payload is not None:
        kwargs.update(body=json.dumps(payload, allow_nan=False).encode(), content_type="application/json")
    try:
        return json.loads(_read(url, transport=transport, **kwargs))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DreamsError("DreaMS returned an unreadable response.") from exc


def query_mgf(query):
    """Strict allowlist: peaks and precursor/known charge, never source metadata."""
    if query.get("polarity") != 1:
        raise DreamsError("DreaMS learned neighbors currently require known positive ion mode.")
    charge = query.get("charge")
    if charge is not None and charge != 1:
        raise DreamsError("DreaMS learned neighbors currently support singly charged positive ions.")
    precursor = float(query["precursor_mz"])
    if not math.isfinite(precursor) or precursor <= 0:
        raise DreamsError("A finite positive precursor mass is required for DreaMS.")
    peaks = validate_peaks(query["peaks"])
    maximum = max(p[1] for p in peaks)
    kept = [[mz, intensity / maximum * 100.] for mz, intensity in peaks if intensity >= .01 * maximum]
    if len(kept) < 3:
        raise DreamsError("DreaMS needs at least three peaks above 1% of the base peak.")
    lines = ["BEGIN IONS", f"PEPMASS={precursor:.10f}"]
    if charge is not None:
        lines.append("CHARGE=1+")
    lines += [f"{mz:.10f} {intensity:.10g}" for mz, intensity in kept]
    lines += ["END IONS", ""]
    preparation = dict(input_peaks=len(peaks), submitted_peaks=len(kept), relative_intensity_cutoff=.01,
                       intensity_normalization="base peak = 100", charge=charge, polarity=1,
                       source_preparation=query.get("preparation", ""),
                       server_preparation="DreaMS applies its model preprocessing; not a centroiding operation")
    return "\n".join(lines).encode("ascii"), preparation


def _result_file(data):
    if not isinstance(data, list) or len(data) != 2:
        raise DreamsError("DreaMS returned an unrecognized result.")
    file = data[1]
    file = file.get("value") if isinstance(file, dict) else None
    if file is None:
        return None
    url = file.get("url") if isinstance(file, dict) else None
    if not isinstance(url, str):
        raise DreamsError("DreaMS returned an invalid result file.")
    parts = _safe_url(url)
    if (parts.hostname != "anton-bushuiev-dreams.hf.space" or parts.query or parts.fragment
            or not re.fullmatch(r"/gradio_api/file=/tmp/gradio/[0-9a-f]{64}/query_[A-Za-z0-9_.-]+\.tsv", parts.path)):
        raise DreamsError("DreaMS returned an invalid result file address.")
    return url


def remote_embedding(query, *, cancel=None, progress=None, transport=qt_http):
    mgf, preparation = query_mgf(query)
    info = _json(SPACE_INFO, transport=transport, cancel=cancel, max_bytes=65536)
    if not isinstance(info, dict) or info.get("sha") != SPACE_REVISION:
        raise DreamsError("The DreaMS service version has changed and needs compatibility validation.")
    config = _json(SPACE + "/config", transport=transport, cancel=cancel, max_bytes=2_000_000)
    if not isinstance(config, dict) or not isinstance(config.get("components"), list) or len(config["components"]) > 1000:
        raise DreamsError("The DreaMS reference configuration has changed.")
    library = None
    for component in config.get("components", []):
        if not isinstance(component, dict):
            continue
        props = component.get("props", {})
        if not isinstance(props, dict):
            continue
        value = props.get("value")
        if component.get("type") == "file" and isinstance(value, dict) and value.get("orig_name") == "MassSpecGym_DreaMS.hdf5":
            path = value.get("path", "")
            if re.fullmatch(r"/tmp/gradio/[0-9a-f]{64}/MassSpecGym_DreaMS\.hdf5", path) and value.get("size") == REFERENCE_BYTES:
                library = dict(path=path, orig_name="MassSpecGym_DreaMS.hdf5", meta={"_type": "gradio.FileData"})
                break
    if library is None:
        raise DreamsError("The DreaMS reference configuration has changed.")
    dependencies = config.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise DreamsError("The DreaMS prediction interface has changed.")
    prediction = next((d for d in dependencies if isinstance(d, dict) and d.get("api_name") == "predict"), None)
    if not prediction or not isinstance(prediction.get("id"), int):
        raise DreamsError("The DreaMS prediction interface has changed.")
    check_cancel(cancel)
    boundary = "----------------Kmz" + uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="query.mgf"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n').encode() + mgf + f"\r\n--{boundary}--\r\n".encode()
    if progress:
        progress("Sending the selected spectrum to DreaMS…")
    try:
        uploaded = json.loads(_read(SPACE + "/gradio_api/upload", body=body,
                                   content_type="multipart/form-data; boundary=" + boundary,
                                   transport=transport, cancel=cancel, max_bytes=65536))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DreamsError("DreaMS did not accept the selected spectrum.") from exc
    if (not isinstance(uploaded, list) or len(uploaded) != 1 or not isinstance(uploaded[0], str)
            or not re.fullmatch(r"/tmp/gradio/[0-9a-f]{64}/query\.mgf", uploaded[0])):
        raise DreamsError("DreaMS did not return a valid upload identifier.")
    file = dict(path=uploaded[0], orig_name="query.mgf", meta={"_type": "gradio.FileData"})
    session = uuid.uuid4().hex
    job = _json(SPACE + "/gradio_api/queue/join", payload={"data": [library, file, .5, False, False],
                "fn_index": prediction["id"], "session_hash": session},
                transport=transport, cancel=cancel, max_bytes=65536)
    event_id = job.get("event_id") if isinstance(job, dict) else None
    if not isinstance(event_id, str) or not re.fullmatch(r"[0-9a-f]{32}", event_id):
        raise DreamsError("DreaMS did not return a valid job identifier.")
    if progress:
        progress("DreaMS is calculating the selected spectrum’s representation…")
    raw = _read(SPACE + "/gradio_api/queue/data?session_hash=" + session, transport=transport,
                cancel=cancel, timeout=180., max_bytes=16 * 1024 * 1024, sse=True)
    complete = None
    try:
        for line in raw.decode("utf-8").splitlines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if not isinstance(event, dict) or event.get("event_id") not in {None, event_id}:
                raise DreamsError("DreaMS returned an unexpected job event.")
            if event.get("msg") == "process_completed":
                output = event.get("output", {})
                if not event.get("success"):
                    error = str(output.get("error", "")) if isinstance(output, dict) else ""
                    if "ZeroGPU" in error and ("limit" in error or "quota" in error):
                        raise DreamsError("DreaMS’s free public GPU quota is exhausted for this connection. Local matching remains available; try the online option later.")
                    raise DreamsError("DreaMS could not complete this search. The public service may be busy; try again later.")
                complete = output.get("data") if isinstance(output, dict) else None
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DreamsError("DreaMS returned an unreadable result.") from exc
    url = _result_file(complete)
    if url is None:
        return dict(embedding=None, preparation=preparation, source_revision=info["sha"])
    tsv = _read(url, transport=transport, cancel=cancel, max_bytes=2_000_000)
    try:
        rows = list(csv.DictReader(io.StringIO(tsv.decode("utf-8")), delimiter="\t"))
        if len(rows) != 1:
            raise ValueError()
        row = rows[0]
        embedding = np.asarray(json.loads(row["DreaMS_embedding"]), dtype=np.float32)
        score = float(row["DreaMS_similarity"])
        accession = row["ref_IDENTIFIER"]
        if (embedding.shape != (DIMENSIONS,) or not np.isfinite(embedding).all()
                or np.linalg.norm(embedding) <= 0 or not math.isfinite(score) or not -1 <= score <= 1.00001
                or not re.fullmatch(r"MassSpecGymID\d+", accession)):
            raise ValueError()
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise DreamsError("DreaMS returned invalid learned match evidence.") from exc
    return dict(embedding=embedding, remote_accession=accession, remote_score=score,
                remote_inchikey=row.get("ref_INCHIKEY", ""),
                preparation=preparation, source_revision=info["sha"])


def ensure_reference(cache_dir, *, cancel=None, progress=None, transport=qt_http):
    if cache_ready(cache_dir):
        return cache_path(cache_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".dreams-reference.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DreamsError("Another analysis window is preparing DreaMS references. Try again when it finishes.") from exc
        if cache_ready(cache_dir):
            return cache_path(cache_dir)
        temporary = cache_dir / (".dreams-download-" + uuid.uuid4().hex + ".hdf5")
        try:
            url = REFERENCE_URL
            for _ in range(6):
                check_cancel(cancel)
                parts = _safe_url(url)
                if parts.hostname not in DOWNLOAD_HOSTS:
                    raise DreamsError("Unrecognized DreaMS reference download host.")
                result = transport(url, cancel=cancel, timeout=900., max_bytes=REFERENCE_BYTES,
                                   output=temporary, progress=progress)
                if result.get("status") == 200:
                    if not temporary.is_file() or temporary.stat().st_size != REFERENCE_BYTES:
                        raise DreamsError("The DreaMS public reference download is incomplete.")
                    break
                if result.get("status") not in {301, 302, 303, 307, 308} or not result.get("location"):
                    raise DreamsError("The DreaMS reference could not be downloaded.")
                url = urljoin(url, result["location"])
            else:
                raise DreamsError("Too many redirects while downloading DreaMS references.")
            build_reference(temporary, cache_path(cache_dir), cancel=cancel, progress=progress)
            return cache_path(cache_dir)
        finally:
            temporary.unlink(missing_ok=True)


def search_dreams(query, cache_dir, cancel=None, progress=None, *, transport=qt_http):
    """Call only after an explicit user Search with the online option selected."""
    # Reject unsupported input before a large public download or any submission.
    query_mgf(query)
    path = ensure_reference(cache_dir, cancel=cancel, progress=progress, transport=transport)
    remote = remote_embedding(query, cancel=cancel, progress=progress, transport=transport)
    if remote["embedding"] is None:
        return dict(hits=[], references={}, preparation=remote["preparation"], error="",
                    provenance=dict(service=SPACE_PAGE, service_revision=remote["source_revision"], completed=timestamp()),
                    warnings=["DreaMS returned no match above its 0.5 service threshold; no embedding was available for local ranking."])
    result = search_reference(remote["embedding"], path, cancel=cancel, progress=progress)
    leading = result["hits"][0] if result["hits"] else None
    # Float16 may exchange tied reference spectra, including spectra of the same
    # structure. Require the server's winner/group in the local top50 AND both
    # its score and the leading score to agree within the quantified rounding bound.
    compatible = [h for h in result["hits"] if h["accession"] == remote["remote_accession"]
                  or (remote.get("remote_inchikey") and h.get("inchikey") == remote["remote_inchikey"])]
    if (leading is None or not compatible
            or abs(compatible[0]["scores"]["dreams"] - remote["remote_score"]) > .0002
            or abs(leading["scores"]["dreams"] - remote["remote_score"]) > .0002):
        raise DreamsError("DreaMS model and reference cache no longer agree. No learned matches were accepted.")
    result["provenance"].update(service=SPACE_PAGE, service_revision=remote["source_revision"],
                                completed=timestamp(), remote_top1=remote["remote_accession"],
                                remote_top1_score=remote["remote_score"])
    result.update(preparation=remote["preparation"], error="", warnings=[
        "DreaMS similarity ranks structural neighbors; it is not an identification probability.",
        "MassSpecGym combines public reference collections with varying annotation provenance; a library structure is not proof of the query structure.",
        "Profile input remains experimental apex samples; model preprocessing does not validate centroid masses."] if query.get("representation") == "profile" else [
        "DreaMS similarity ranks structural neighbors; it is not an identification probability.",
        "MassSpecGym combines public reference collections with varying annotation provenance; a library structure is not proof of the query structure."])
    return result
