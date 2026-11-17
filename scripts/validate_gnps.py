#!/usr/bin/env python3
"""Opt-in live GNPS qualification using a fixed PUBLIC reference spectrum only.

No acquisition path or private query is accepted. Normal root tests remain offline.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from PySide6.QtWidgets import QApplication
from kmzcore.theme import apply_theme
from kmzcore.version import __version__
from kmzviewer.gnps import SearchSettings, new_analysis, save_analysis, timestamp
from kmzviewer.gnps_client import GnpsClient
from kmzviewer.gui.gnps import GnpsWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".local/gnps/live"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    apply_theme(app)
    client = GnpsClient()
    hit = dict(usi="mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001647",
               accession="CCMSLIB00000001647", name="Enterocin", adduct="M+Na")
    reference = client.reference(hit)
    query = dict(precursor_mz=reference["precursor_mz"], charge=reference["charge"], polarity=1,
                 peaks=reference["peaks"], preparation="Public GNPS reference peak list",
                 representation="centroid", precursor_source="public spectrum resolver",
                 source_name="Public GNPS reference", scan_id=hit["accession"], spectrum_index=0,
                 rt_min=None, collision_energy=None, target_mz=None, target_rt=None)
    settings = SearchSettings()
    analysis = new_analysis(query, settings)
    report = dict(version=__version__, created=timestamp(), public_query=hit["usi"],
                  settings=asdict(settings), query_peaks=len(query["peaks"]), modes={})
    for mode in ("standard", "analog", "datasets"):
        started = time.monotonic()
        result = client.search(query, settings, mode)
        analysis["searches"][mode] = result
        report["modes"][mode] = dict(seconds=round(time.monotonic() - started, 3),
                                     returned=result["returned_count"], accepted=len(result["hits"]),
                                     database=result["database"])
        print(mode, report["modes"][mode], flush=True)
    standard = analysis["searches"]["standard"]["hits"]
    assert any(h["accession"] == hit["accession"] and h["score"] >= .99 for h in standard), "Public self-match failed"
    analysis["references"][hit["usi"]] = reference
    if standard and standard[0]["usi"] != hit["usi"]:
        analysis["references"][standard[0]["usi"]] = client.reference(standard[0])
    snapshot = args.output / "public-reference.kmzgnps"
    save_analysis(snapshot, analysis)
    window = GnpsWindow()
    window.open_analysis(snapshot)
    window.show()
    app.processEvents()
    window.grab().save(str(args.output / "public-match.png"))
    assert len(window.mirror.listDataItems()) == 2
    report.update(ok=True, self_match=True, structure_loaded=bool(reference["structure_png"]),
                  offline_reopen=True)
    window.close()
    app.processEvents()
    # GNPSDataPackage's own positive repository-search reference exercises a
    # populated result, independently of the first reference's empty MASST result.
    dataset_hit = dict(usi="mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005435899",
                       accession="CCMSLIB00005435899", name="Public dataset-search reference", adduct="")
    dataset_reference = client.reference(dataset_hit)
    dataset_query = dict(peaks=dataset_reference["peaks"], precursor_mz=dataset_reference["precursor_mz"],
                         charge=dataset_reference["charge"])
    started = time.monotonic()
    populated = client.search(dataset_query, settings, "datasets")
    assert populated["hits"], "Public repository positive control returned no matches"
    report["dataset_positive_control"] = dict(public_query=dataset_hit["usi"],
        accepted=len(populated["hits"]), datasets=len({h["dataset"] for h in populated["hits"]}),
        seconds=round(time.monotonic() - started, 3))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("GNPS qualification passed", flush=True)


if __name__ == "__main__":
    main()
