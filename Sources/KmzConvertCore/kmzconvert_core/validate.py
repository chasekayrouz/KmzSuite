"""Numerical mzML comparison. This module is not part of the conversion path."""
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

from .reference import iter_records, value


def compare(reference, actual):
    report = {"reference": str(Path(reference).resolve()), "actual": str(Path(actual).resolve()),
              "tolerances": {"signal_mz_absolute": 1e-10, "all_mz_absolute": 1e-9,
                             "rt_seconds_absolute": 1e-8, "intensity_absolute": 0., "precursor_mz_absolute": 1e-9},
              "reference_count": 0, "actual_count": 0, "paired_spectra": 0, "points": 0,
              "reference_ms_levels": Counter(), "actual_ms_levels": Counter(),
              "max_mz_absolute_error": 0., "max_signal_mz_absolute_error": 0.,
              "max_zero_mz_absolute_error": 0., "max_mz_ppm_error": 0.,
              "max_signal_mz_ppm_error": 0.,
              "max_intensity_absolute_error": 0., "max_rt_seconds_error": 0.,
              "max_tic_absolute_error": 0., "max_array_sum_vs_tic_error": 0.,
              "max_precursor_mz_absolute_error": 0., "length_mismatches": 0,
              "intensity_mismatched_points": 0, "signal_mz_mismatched_points": 0,
              "zero_mz_mismatched_points": 0, "order_mismatches": 0,
              "metadata_mismatches": Counter(), "metadata_missing": Counter(),
              "reference_charge_distribution": Counter(), "nonfinite_arrays": 0,
              "unsorted_mz_arrays": 0, "chromatograms": [],
              "measured_signal": {"reference_points": 0, "actual_points": 0,
                                  "length_mismatches": 0, "intensity_mismatched_points": 0,
                                  "mz_mismatched_points": 0, "max_mz_absolute_error": 0.,
                                  "max_mz_ppm_error": 0., "max_intensity_absolute_error": 0.}}
    ref_ids, act_ids = [], []
    comparisons = {
        "MS:1000501": "scan_window_lower", "MS:1000500": "scan_window_upper",
        "MS:1000827": "isolation_target", "MS:1000828": "isolation_lower_offset",
        "MS:1000829": "isolation_upper_offset", "MS:1000045": "collision_energy",
        "MS:1000041": "charge", "MS:1000422": "activation_type",
    }
    for r, a in zip_longest(iter_records(reference), iter_records(actual)):
        if r:
            ref_ids.append(r["id"])
            report["reference_count"] += 1
            report["reference_ms_levels"][r["ms_level"]] += 1
        if a:
            act_ids.append(a["id"])
            report["actual_count"] += 1
            report["actual_ms_levels"][a["ms_level"]] += 1
        if r is None or a is None:
            continue
        if r["id"] != a["id"]:
            report["order_mismatches"] += 1
            continue
        report["paired_spectra"] += 1
        for key in ("ms_level", "polarity", "representation"):
            if r[key] != a[key]:
                report["metadata_mismatches"][key] += 1
        for accession, name in comparisons.items():
            rv, av = value(r["cv"], accession), value(a["cv"], accession)
            if rv is not None and av is None:
                report["metadata_missing"][name] += 1
            elif (rv is None) != (av is None):
                report["metadata_mismatches"][name] += 1
            elif rv is not None and av is not None:
                if rv == av:
                    continue
                try:
                    agrees = abs(float(rv)-float(av)) <= 1e-9
                except ValueError:
                    agrees = False
                if not agrees:
                    report["metadata_mismatches"][name] += 1
        if r["ms_level"] == 2:
            report["reference_charge_distribution"][value(r["cv"], "MS:1000041", "absent")] += 1
            rp, ap = value(r["cv"], "MS:1000744"), value(a["cv"], "MS:1000744")
            if rp is None or ap is None:
                report["metadata_mismatches"]["precursor"] += 1
            else:
                report["max_precursor_mz_absolute_error"] = max(report["max_precursor_mz_absolute_error"], abs(float(rp)-float(ap)))
        report["max_rt_seconds_error"] = max(report["max_rt_seconds_error"], abs(r["rt_seconds"]-a["rt_seconds"]))
        rtic, atic = float(value(r["cv"], "MS:1000285")), float(value(a["cv"], "MS:1000285"))
        report["max_tic_absolute_error"] = max(report["max_tic_absolute_error"], abs(rtic-atic))
        rm, am = r["arrays"]["mz"], a["arrays"]["mz"]
        ry, ay = r["arrays"]["intensity"], a["arrays"]["intensity"]
        report["max_array_sum_vs_tic_error"] = max(report["max_array_sum_vs_tic_error"], abs(float(ay.sum())-atic))
        if not np.isfinite(am).all() or not np.isfinite(ay).all():
            report["nonfinite_arrays"] += 1
        if np.any(np.diff(am) <= 0):
            report["unsorted_mz_arrays"] += 1
        # Separate measured signal from vendor-specific zero framing. Preserve
        # the original full-array checks and separate pass/fail fields below.
        measured = report["measured_signal"]
        if len(rm) != len(ry) or len(am) != len(ay):
            report["length_mismatches"] += 1
            measured["length_mismatches"] += 1
            continue
        rsignal, asignal = ry != 0, ay != 0
        rm_signal, am_signal = rm[rsignal], am[asignal]
        ry_signal, ay_signal = ry[rsignal], ay[asignal]
        measured["reference_points"] += len(rm_signal)
        measured["actual_points"] += len(am_signal)
        if len(rm_signal) != len(am_signal):
            measured["length_mismatches"] += 1
        else:
            dm_signal, dy_signal = abs(am_signal-rm_signal), abs(ay_signal-ry_signal)
            measured["max_mz_absolute_error"] = max(measured["max_mz_absolute_error"], float(dm_signal.max(initial=0)))
            measured["max_mz_ppm_error"] = max(measured["max_mz_ppm_error"], float(np.max(dm_signal/rm_signal*1e6, initial=0)))
            measured["max_intensity_absolute_error"] = max(measured["max_intensity_absolute_error"], float(dy_signal.max(initial=0)))
            measured["intensity_mismatched_points"] += int(np.count_nonzero(dy_signal))
            measured["mz_mismatched_points"] += int(np.count_nonzero(dm_signal > 1e-10))
        if not (len(rm) == len(am) == len(ry) == len(ay) == int(a["defaultArrayLength"])):
            report["length_mismatches"] += 1
            continue
        report["points"] += len(am)
        dm, dy = np.abs(am-rm), np.abs(ay-ry)
        report["max_mz_absolute_error"] = max(report["max_mz_absolute_error"], float(dm.max(initial=0)))
        report["max_mz_ppm_error"] = max(report["max_mz_ppm_error"], float(np.max(dm/rm*1e6, initial=0)))
        report["max_intensity_absolute_error"] = max(report["max_intensity_absolute_error"], float(dy.max(initial=0)))
        report["intensity_mismatched_points"] += int(np.count_nonzero(dy))
        signal = ry != 0
        report["max_signal_mz_ppm_error"] = max(report["max_signal_mz_ppm_error"],
            float(np.max(dm[signal]/rm[signal]*1e6, initial=0)))
        for label, mask, tolerance in [("signal", ry != 0, 1e-10), ("zero", ry == 0, 1e-9)]:
            report[f"max_{label}_mz_absolute_error"] = max(report[f"max_{label}_mz_absolute_error"], float(dm[mask].max(initial=0)))
            report[f"{label}_mz_mismatched_points"] += int(np.count_nonzero(dm[mask] > tolerance))
    report["missing_scans"] = sorted(set(ref_ids)-set(act_ids))
    report["extra_scans"] = sorted(set(act_ids)-set(ref_ids))
    for r, a in zip_longest(iter_records(reference, "chromatogram"), iter_records(actual, "chromatogram")):
        if r is None or a is None:
            report["chromatograms"].append({"missing_or_extra": (r or a)["id"]})
            continue
        entry = {"reference_id": r["id"], "actual_id": a["id"], "arrays": {}}
        for key in r["arrays"].keys() | a["arrays"].keys():
            ra, aa = r["arrays"].get(key), a["arrays"].get(key)
            if ra is None or aa is None or len(ra) != len(aa):
                entry["arrays"][key] = {"length_mismatch": True}
            else:
                entry["arrays"][key] = {"length": len(aa), "max_absolute_error": float(np.abs(ra-aa).max(initial=0)),
                                         "exact": bool(np.array_equal(ra, aa))}
        report["chromatograms"].append(entry)
    report["chromatograms_pass"] = all(
        "missing_or_extra" not in c and c["reference_id"] == c["actual_id"] and
        all(a.get("exact", False) for a in c["arrays"].values())
        for c in report["chromatograms"])
    report["measured_spectra_pass"] = (
        report["paired_spectra"] == report["reference_count"] == report["actual_count"] and
        not any(report[k] for k in ["missing_scans", "extra_scans", "order_mismatches", "length_mismatches", "intensity_mismatched_points",
                                   "signal_mz_mismatched_points", "nonfinite_arrays", "unsorted_mz_arrays", "metadata_mismatches"]) and
        report["max_rt_seconds_error"] <= 1e-8 and report["max_tic_absolute_error"] == 0 and
        report["max_array_sum_vs_tic_error"] == 0 and report["max_precursor_mz_absolute_error"] <= 1e-9)
    measured = report["measured_signal"]
    measured["passed"] = (
        report["paired_spectra"] == report["reference_count"] == report["actual_count"] > 0 and
        not any(report[k] for k in ["missing_scans", "extra_scans", "order_mismatches", "nonfinite_arrays",
                                   "unsorted_mz_arrays"]) and
        not any(report["metadata_mismatches"][k] for k in ["ms_level", "polarity", "representation", "precursor"]) and
        not any(measured[k] for k in ["length_mismatches", "intensity_mismatched_points", "mz_mismatched_points"]) and
        report["max_rt_seconds_error"] <= 1e-8 and report["max_tic_absolute_error"] == 0 and
        report["max_array_sum_vs_tic_error"] == 0 and report["max_precursor_mz_absolute_error"] <= 1e-9)
    report["strict_all_mz_points_pass"] = (
        report["paired_spectra"] == report["reference_count"] == report["actual_count"] > 0 and
        report["length_mismatches"] == 0 and report["max_mz_absolute_error"] <= 1e-9)
    report["complete_metadata_equivalence"] = not report["metadata_missing"] and not report["metadata_mismatches"]
    report["known_differences"] = [
        "Framing zero coordinates are calibrated on the native TOF grid; the vendor sometimes linearly extrapolates them.",
        "Precursor charge and activation type are not decoded. They are omitted, not inferred.",
        "Instrument subtype and absolute acquisition timestamp remain unresolved; compare the metadata inventory separately.",
    ]
    return report


if __name__ == "__main__":
    result = compare(sys.argv[1], sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["measured_spectra_pass"] and result["chromatograms_pass"] else 1)
