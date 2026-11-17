"""Pilot diagnostics and exhaustive four-reference consensus.

Profile detection/estimation primitives now live in the normal converter core.
Production never imports this development package.
"""
from itertools import product

import numpy as np
from kmzconvert_core.mass_correction import SearchSettings, ProfileScan
from kmzconvert_core.mass_correction import local_mass as _local_mass, candidates as _candidates
from .models import Observation

METHODS = ("apex_maximum", "apex_centroid", "multi_scan_centroid", "apex_parabola")


def local_mass(mz, intensity, seed_mz, settings):
    return _local_mass(mz, intensity, seed_mz, settings, interpolate=True)


def candidates(reference, scans, settings):
    return _candidates(reference, scans, settings, interpolate=True)


def consensus(candidate_lists, settings):
    """Rank complete assignments using broad common ppm OR Da regimes.

    This is a detection prior, never calibration-model training performance.
    Keep runner-up evidence; fail closed on insufficient separation. No
    reference is discarded to improve agreement.
    """
    eligible = [[c for c in rows if not c["flags"]][:12] for rows in candidate_lists]
    if any(not rows for rows in eligible):
        return None, dict(failure="At least one reference has no eligible candidate")
    ranked = []
    for assignment in product(*eligible):
        ppm = np.array([c["ppm_error"] for c in assignment])
        da = np.array([c["da_error"] for c in assignment])
        masses = np.array([c["observed_mz"] for c in assignment])
        spread = min(float(np.std(ppm)), float(np.std(da)/np.mean(masses)*1e6))
        penalty = (spread/settings.consensus_sd_ppm)**2
        score = float(np.mean([c["score"] for c in assignment])-penalty)
        ranked.append((score, assignment, spread))
    ranked.sort(key=lambda x: -x[0])
    top = ranked[:5]
    margin = top[0][0]-top[1][0] if len(top)>1 else None
    evidence = dict(assignments_considered=len(ranked), score_margin=margin,
                    best_assignments=[dict(score=s, candidate_ids=[c["candidate_id"] for c in a], common_regime_sd_ppm=d) for s,a,d in top])
    if top[0][2] > 2*settings.consensus_sd_ppm:
        evidence["failure"] = "No coherent common coarse error regime"
    elif margin is not None and margin < settings.ambiguity_margin:
        evidence["failure"] = "Ambiguous competing reference assignments; inspect diagnostics"
    return (None if "failure" in evidence else list(top[0][1])), evidence


def estimate(reference, candidate, context):
    rows = []
    for i in candidate["high_indices"]:
        measurement = candidate["measurements"][i]
        rows.append(dict(reference_id=reference.id, scan_id=context["scan_ids"][i], rt=context["rt"][i],
                         apex=i == candidate["apex_index"], **measurement))
    apex = next(r for r in rows if r["apex"])
    outputs = []
    for method, field in ((METHODS[0], "sampled_mz"), (METHODS[1], "centroid_mz"), (METHODS[2], "centroid_mz"), (METHODS[3], "parabola_mz")):
        values = [r[field] for r in rows if r[field] is not None]
        observed = float(np.median(values)) if method == METHODS[2] else apex[field]
        if observed is None:
            outputs.append(dict(reference_id=reference.id, method=method, failure="No stable concave three-point fit at apex"))
            continue
        observation = Observation(reference, observed, candidate["rt"], method)
        outputs.append(dict(reference_id=reference.id, method=method, observed_mz=observed,
                            ppm_error=observation.ppm_error, da_error=observation.da_error,
                            scans=len(rows), successful_scans=len(values),
                            scan_sd_ppm=float(np.std(values, ddof=1)/reference.expected_mz*1e6) if len(values)>1 else None,
                            scan_mad_ppm=float(1.4826*np.median(abs(np.array(values)-np.median(values)))/reference.expected_mz*1e6),
                            rt=candidate["rt"], intensity=candidate["intensity"]))
    return outputs, rows
