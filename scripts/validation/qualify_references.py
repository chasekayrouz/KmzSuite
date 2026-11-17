"""Compare all local reference mzMLs with converted outputs.

Retains the original strict comparison fields, including failures for differing
zero framing, auxiliary chromatograms, or metadata. Separately qualifies measured
signal and checks each reference TIC against the raw index's two observed extents:
through the last populated slot, or through all slots. No source is modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import iter_records, value
from kmzconvert_core.validate import compare


def audit_framing(reference, actual):
    result = {"identical_intensity_arrays": 0, "only_two_outer_zero_bins_differ": 0,
              "other_intensity_layout_differences": 0, "max_aligned_zero_mz_error": 0.,
              "reference_missing_collision_energy": 0}
    for r, a in zip(iter_records(reference), iter_records(actual), strict=True):
        assert r["id"] == a["id"]
        ry, ay = r["arrays"]["intensity"], a["arrays"]["intensity"]
        rm, am = r["arrays"]["mz"], a["arrays"]["mz"]
        if r["ms_level"] == 2 and value(r["cv"], "MS:1000045") is None and value(a["cv"], "MS:1000045") is not None:
            result["reference_missing_collision_energy"] += 1
        if np.array_equal(ry, ay):
            result["identical_intensity_arrays"] += 1
        elif len(ay) == len(ry)+2 and ay[0] == ay[-1] == 0 and np.array_equal(ry, ay[1:-1]):
            result["only_two_outer_zero_bins_differ"] += 1
            am = am[1:-1]
        else:
            result["other_intensity_layout_differences"] += 1
            continue
        result["max_aligned_zero_mz_error"] = max(result["max_aligned_zero_mz_error"],
            float(abs(am[ry == 0]-rm[ry == 0]).max(initial=0)))
    return result


def audit_tic(reference, actual, entry):
    refs = [r for r in iter_records(reference, "chromatogram") if r["id"] == "TIC"]
    acts = list(iter_records(actual, "chromatogram"))
    assert len(refs) == len(acts) == 1 and acts[0]["id"] == "TIC"
    ra, aa = refs[0]["arrays"], acts[0]["arrays"]
    with WiffReader(entry) as reader:
        current = dict(zip(("time", "intensity", "ms level"), reader.tic_arrays()))
        full = {"time": np.array([r[2]/60000 for r in reader.index]),
                "intensity": np.array([r[4] for r in reader.index]),
                "ms level": np.array([reader.experiments[i % len(reader.experiments)].ms_level
                                       for i in range(len(reader.index))])}
        assert aa.keys() == ra.keys() == current.keys()
        assert all(np.array_equal(aa[k], current[k]) for k in aa)
        exact = all(np.array_equal(ra[k], aa[k]) for k in ra)
        full_index = all(np.array_equal(ra[k], full[k]) for k in ra)
        extra = len(ra["time"])-len(aa["time"])
        tail = reader.index[reader.scans[-1].slot+1:]
        empty_tail = full_index and extra == len(tail) and all(r[1] == r[4] == 0 for r in tail)
        return {"exact": exact, "reference_points": len(ra["time"]), "actual_points": len(aa["time"]),
                "reference_equals_full_raw_index": full_index,
                "extra_reference_slots": extra, "extra_slots_are_empty_raw_index_tail": empty_tail,
                "shared_prefix_exact": all(np.array_equal(ra[k][:len(aa[k])], aa[k]) for k in ra),
                "total_intensity_error": float(abs(ra["intensity"].sum()-aa["intensity"].sum())),
                "raw_index_contract_passed": bool(exact or empty_tail)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("converted", type=Path)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    args.reports.mkdir(parents=True, exist_ok=False)
    summaries = []
    for actual in sorted(args.converted.glob("*.mzML")):
        references = sorted(args.fixtures.glob(actual.stem+"_from_*.mzML"))
        original = args.fixtures/actual.name
        if original.exists():
            references.append(original)
        for reference in references:
            report = compare(reference, actual)
            framing = report["framing_audit"] = audit_framing(reference, actual)
            tic = report["tic_audit"] = audit_tic(reference, actual, args.fixtures/(actual.stem+".wiff"))
            report["additional_reference_chromatograms"] = [r["id"] for r in iter_records(reference, "chromatogram") if r["id"] != "TIC"]
            unexplained = dict(report["metadata_mismatches"])
            if unexplained.get("collision_energy") == framing["reference_missing_collision_energy"]:
                unexplained.pop("collision_energy", None)
            passed = (report["measured_signal"]["passed"] and tic["raw_index_contract_passed"]
                      and not framing["other_intensity_layout_differences"] and not unexplained)
            report["measured_signal_and_raw_tic_passed"] = passed
            (args.reports/(reference.stem+".json")).write_text(json.dumps(report, indent=2)+"\n")
            summaries.append({"reference": reference.name, "acquisition": actual.stem,
                              "spectra": report["actual_count"], "measured_signal_and_raw_tic_passed": passed,
                              "original_array_contract_passed": report["measured_spectra_pass"],
                              "full_chromatograms_passed": report["chromatograms_pass"]})
            print(reference.name, "signal/raw TIC:", passed, "original array contract:", report["measured_spectra_pass"],
                  "signal m/z error:", report["measured_signal"]["max_mz_absolute_error"], flush=True)
    (args.reports/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    return 0 if summaries and all(r["measured_signal_and_raw_tic_passed"] for r in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
