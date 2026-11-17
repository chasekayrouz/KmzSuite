"""Strict version-1 constant-ppm beta processing contract shared with the viewer."""
import json
import math
from statistics import median

WARNING = "Beta mass correction: applies one global offset to measured MS1 masses. Accuracy across the run is not guaranteed. MS2 masses remain uncorrected."
ESTIMATOR = "high-scan-local-intensity-weighted-centroid-median"
PAYLOAD = "KmzSuite mass correction observations"
PARAMS = {
    "KmzSuite mass correction version": "1",
    "KmzSuite mass correction beta": "true",
    "KmzSuite mass correction scope": "MS1",
    "KmzSuite mass correction model": "constant-ppm",
    "KmzSuite mass correction estimator": ESTIMATOR,
}
SEARCH_KEYS = {"rt_radius", "broad_ppm", "track_ppm", "lobe_ppm", "high_fraction",
               "minimum_high_scans", "minimum_prominence_ratio", "consensus_sd_ppm",
               "ambiguity_margin", "rt_score_weight", "precision_score_scale_ppm"}
SEARCH_V1 = dict(rt_radius=.45, broad_ppm=1000., track_ppm=40., lobe_ppm=100.,
                 high_fraction=.5, minimum_high_scans=3, minimum_prominence_ratio=3.,
                 consensus_sd_ppm=150., ambiguity_margin=.15, rt_score_weight=2.,
                 precision_score_scale_ppm=10.)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_payload(payload):
    """Reject unsupported, inconsistent, nonfinite or incomplete provenance."""
    try:
        if set(payload) != {"offset_ppm", "references", "search_settings", "consistency"}:
            return False
        e = payload["offset_ppm"]
        if not finite(e) or 1 + e * 1e-6 <= 0:
            return False
        settings = payload["search_settings"]
        if set(settings) != SEARCH_KEYS or any(not finite(v) or v <= 0 for v in settings.values()):
            return False
        if settings != SEARCH_V1:
            return False  # Version 1 is the explicitly resolved tested beta method.
        rows = payload["references"]
        if not isinstance(rows, list) or not rows:
            return False
        pairs = set()
        for row in rows:
            if set(row) != {"expected_mz", "expected_rt", "observed_mz", "observed_rt", "error_ppm",
                            "calibration_residual_ppm", "high_scans", "centroid_sd_ppm", "apex_scan_id", "intensity"}:
                return False
            if any(not finite(v) for k, v in row.items() if k != "apex_scan_id"):
                return False
            if (row["expected_mz"] <= 0 or row["observed_mz"] <= 0 or row["expected_rt"] < 0 or
                    row["observed_rt"] < 0 or row["centroid_sd_ppm"] < 0 or row["intensity"] <= 0 or
                    type(row["high_scans"]) is not int or row["high_scans"] < 3 or
                    not isinstance(row["apex_scan_id"], str) or not row["apex_scan_id"]):
                return False
            pair = (row["expected_mz"], row["expected_rt"])
            if pair in pairs:
                return False
            pairs.add(pair)
            raw = (row["observed_mz"]-row["expected_mz"])/row["expected_mz"]*1e6
            residual = (row["observed_mz"]/(1+e*1e-6)-row["expected_mz"])/row["expected_mz"]*1e6
            if abs(raw-row["error_ppm"]) > 1e-7 or abs(residual-row["calibration_residual_ppm"]) > 1e-7:
                return False
        if abs(median(r["error_ppm"] for r in rows)-e) > 1e-7:
            return False
        return payload["consistency"] == ("unavailable-single-reference" if len(rows) == 1 else "coarse-checked")
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def provenance_params(payload):
    if not validate_payload(payload):
        raise ValueError("Invalid constant-ppm mass correction provenance")
    return {**PARAMS, PAYLOAD: json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)}


def valid_method_params(params):
    if set(params) != set(PARAMS) | {PAYLOAD} or any(params[k] != v for k, v in PARAMS.items()):
        return False
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("Duplicate provenance key")
            result[k] = v
        return result
    try:
        return validate_payload(json.loads(params[PAYLOAD], object_pairs_hook=unique))
    except (ValueError, TypeError):
        return False
