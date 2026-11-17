"""Bounded GNPS2 HTTPS client. Only the explicit query allowlist leaves the Mac.

Qt uses the platform TLS store in frozen apps and can abort requests immediately.
The transport is injectable so normal regressions never need Internet access.
"""
from __future__ import annotations

import base64
import json
import re
import time
from urllib.parse import urlencode, urlsplit

from kmzcore.cancel import Cancelled
from kmzviewer.gnps import check_cancel, finite, timestamp, validate_peaks

API = "https://api.fasst.gnps2.org"
LIBRARY = "gnpslibrary"
DATASETS = "metabolomicspanrepo_index_latest"
HOSTS = {"api.fasst.gnps2.org", "metabolomics-usi.gnps2.org", "external.gnps2.org", "structure.gnps2.org"}


class ServiceError(ValueError):
    pass


def qt_request(url, *, payload=None, cancel=None, timeout=30., max_bytes=16 * 1024 * 1024):
    from PySide6.QtCore import QByteArray, QEventLoop, QTimer, QUrl
    from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
    import shiboken6

    check_cancel(cancel)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ServiceError("Unrecognized GNPS service address.")
    data = None if payload is None else QByteArray(json.dumps(payload, allow_nan=False).encode())
    manager = QNetworkAccessManager()  # Fallible encoding is done before any Qt object exists.
    request = QNetworkRequest(QUrl(url))
    request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                         QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
    request.setRawHeader(b"User-Agent", b"KmzViewer-GNPS/1")
    request.setRawHeader(b"Accept", b"application/json, image/png")
    if payload is not None:
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        reply = manager.post(request, data)
    else:
        reply = manager.get(request)
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(40)
    started = time.monotonic()
    received = bytearray()
    reason = []

    def read():
        received.extend(bytes(reply.readAll()))
        if len(received) > max_bytes:
            reason.append("GNPS returned an oversized response.")
            reply.abort()

    def tick():
        if cancel is not None and cancel():
            reason.append("cancel")
            reply.abort()
        elif time.monotonic() - started > timeout:
            reason.append("GNPS request timed out. Try again later.")
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
        if reason:
            if reason[0] == "cancel":
                raise Cancelled()
            raise ServiceError(reason[0])
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if status != 200:
            # Never include request URLs, submitted peak lists or remote error bodies.
            raise ServiceError(f"GNPS service returned HTTP {status or 'unavailable'}. Try again later.")
        if reply.error() != QNetworkReply.NetworkError.NoError:
            raise ServiceError("Could not reach GNPS securely. Check your connection and try again.")
        return bytes(received)
    finally:
        # Delete these worker-thread Qt objects here, even if a traceback keeps this frame
        # alive: the GUI-thread garbage collector must never be the one to destroy them.
        timer.stop()
        for item in (timer, loop, manager):  # The reply is the manager's child.
            shiboken6.delete(item)


def request_payload(query, settings, mode):
    settings.validate()
    peaks = validate_peaks(query["peaks"])
    # Normalize only the submitted copy, preserving native values in the snapshot.
    maximum = max(y for _, y in peaks)
    spectrum = dict(peaks=[[mz, y / maximum * 100.] for mz, y in peaks],
                    precursor_mz=query["precursor_mz"], precursor_charge=query.get("charge") or 0)
    return dict(query_spectrum=json.dumps(spectrum, allow_nan=False, separators=(",", ":")),
                library=DATASETS if mode == "datasets" else LIBRARY,
                analog="Yes" if mode == "analog" else "No", cache="No",
                pm_tolerance=settings.precursor_da, fragment_tolerance=settings.fragment_da,
                cosine_threshold=settings.min_cosine, lower_delta=settings.analog_below,
                upper_delta=settings.analog_above)


def parse_hits(data, settings, mode):
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ServiceError("GNPS returned an unrecognized result. No matches were accepted.")
    if len(data["results"]) > 10000:
        raise ServiceError("GNPS returned over 10,000 matches. Increase the match thresholds.")
    hits = []
    seen = set()
    for row in data["results"]:
        try:
            score, count = float(row["Cosine"]), float(row["Matching Peaks"])
            usi = row["USI"]
            delta = float(row["Delta Mass"])
            if (not finite(score) or not 0 <= score <= 1.000001 or not finite(count)
                    or count < 0 or not count.is_integer() or not finite(delta)
                    or not isinstance(usi, str) or not usi.startswith("mzspec:") or len(usi) > 4096):
                raise ValueError()
        except (TypeError, ValueError, KeyError) as exc:
            raise ServiceError("GNPS returned malformed match evidence. No matches were accepted.") from exc
        if score < settings.min_cosine or count < settings.min_peaks or usi in seen:
            continue
        if mode == "analog" and abs(delta) <= settings.precursor_da:
            continue  # Same-mass matches belong in the standard group.
        seen.add(usi)
        hits.append(dict(usi=usi, accession=str(row.get("GNPSLibraryAccession", "")),
                         name=str(row.get("CompoundName") or "Unannotated spectrum"),
                         adduct=str(row.get("Adduct") or ""), dataset=str(row.get("Dataset") or "Unknown dataset"),
                         score=score, matched_peaks=int(count), delta_mass=delta,
                         charge=row.get("Charge")))
    return sorted(hits, key=lambda h: (-h["score"], -h["matched_peaks"], abs(h["delta_mass"]), h["usi"]))


class GnpsClient:
    def __init__(self, transport=qt_request):
        self.transport = transport

    def _json(self, url, *, cancel=None, payload=None):
        try:
            return json.loads(self.transport(url, payload=payload, cancel=cancel))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ServiceError("GNPS returned an unreadable response.") from exc

    def search(self, query, settings, mode, *, cancel=None, progress=None):
        if mode not in {"standard", "analog", "datasets"}:
            raise ValueError("Unknown search mode.")
        payload = request_payload(query, settings, mode)
        created = timestamp()
        submission = self._json(API + "/search", payload=payload, cancel=cancel)
        task = submission.get("id") if isinstance(submission, dict) else None
        if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,128}", task):
            raise ServiceError("GNPS did not return a valid search identifier.")
        deadline = time.monotonic() + 120
        while True:
            check_cancel(cancel)
            result = self._json(API + "/search/result/" + task, cancel=cancel)
            if isinstance(result, dict) and "results" in result:
                return dict(hits=parse_hits(result, settings, mode), error="", created=created,
                            completed=timestamp(), task=task, database=payload["library"],
                            endpoint=API, returned_count=len(result["results"]))
            if not isinstance(result, dict) or result.get("status") not in {"PENDING", "RUNNING", "STARTED", "QUEUED"}:
                raise ServiceError("GNPS could not complete this search. Try again later.")
            if time.monotonic() >= deadline:
                raise ServiceError("GNPS is still processing after two minutes. Try again later.")
            if progress:
                progress(f"Searching {mode} · {result['status'].lower()}…")
            # Worker-only sleep, with cooperative cancellation every 50 ms.
            for _ in range(20):
                check_cancel(cancel)
                time.sleep(.05)

    def reference(self, hit, *, cancel=None):
        usi = hit["usi"]
        if not isinstance(usi, str) or not usi.startswith("mzspec:") or len(usi) > 4096:
            raise ServiceError("This match has no valid public spectrum identifier.")
        resolved = self._json("https://metabolomics-usi.gnps2.org/json/?" + urlencode({"usi1": usi}), cancel=cancel)
        if not isinstance(resolved, dict):
            raise ServiceError("The reference spectrum could not be resolved.")
        try:
            peaks = sorted([list(map(float, pair)) for pair in resolved["peaks"] if float(pair[1]) > 0])
            validate_peaks(peaks, allow_empty=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceError("GNPS returned an invalid reference spectrum.") from exc
        reference = dict(peaks=peaks, precursor_mz=resolved.get("precursor_mz"),
                         charge=resolved.get("precursor_charge"), name=hit["name"],
                         adduct=hit.get("adduct", ""), structure_png="", metadata={}, warnings=[])
        accession = hit.get("accession", "")
        if re.fullmatch(r"CCMSLIB\d+", accession):
            try:
                record = self._json("https://external.gnps2.org/gnpsspectrum?" + urlencode({"SpectrumID": accession}), cancel=cancel)
                annotations = record.get("annotations", [])
                annotation = annotations[0] if annotations else {}
                keys = ("Compound_Name", "Adduct", "Ion_Mode", "Charge", "Smiles", "INCHI", "Instrument", "Library_Class")
                reference["metadata"] = {k: str(annotation[k]) for k in keys if k in annotation}
                reference["library"] = str(record.get("spectruminfo", {}).get("library_membership", ""))
                smiles, inchi = annotation.get("Smiles"), annotation.get("INCHI")
                invalid = {None, "", "N/A", "NA", "None", "null", "-"}
                structure = {"smiles": smiles} if smiles not in invalid else {"inchi": inchi} if inchi not in invalid else None
                if structure:
                    png = self.transport("https://structure.gnps2.org/structureimg?" + urlencode(structure), cancel=cancel, max_bytes=2_000_000)
                    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ServiceError("The reference structure image is unavailable.")
                    reference["structure_png"] = base64.b64encode(png).decode("ascii")
            except Cancelled:
                raise
            except (ServiceError, KeyError, TypeError, ValueError):
                reference["warnings"].append("Some reference metadata or the structure image could not be retrieved.")
        return reference


def public_url(hit, *, dataset=False):
    """Generate trusted public links; never open a URL supplied by a result file."""
    if dataset and re.fullmatch(r"MSV\d+", hit.get("dataset", "")):
        return "https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?" + urlencode({"accession": hit["dataset"]})
    return "https://metabolomics-usi.gnps2.org/dashinterface/?" + urlencode({"usi1": hit["usi"]})
