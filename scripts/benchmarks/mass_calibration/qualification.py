"""Reproducible single-acquisition development qualification orchestration."""
from dataclasses import asdict, replace
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from kmzconvert_core.reader import WiffReader

from .detection import METHODS, ProfileScan, SearchSettings, candidates, consensus, estimate
from .models import MODEL_NAMES, Model, Observation, Reference, benchmark, correct_scan, fit, residual
from .mzml import verify_experimental, write_experimental

ROOT = Path(__file__).resolve().parents[3]

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def save_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()})


def identity(path, id):
    with path.open("rb") as handle:
        return dict(id=id, bytes=path.stat().st_size, sha256=hashlib.file_digest(handle, "sha256").hexdigest())


def code_provenance():
    source = Path(__file__).resolve().parent
    files = [*sorted(source.glob("*.py")), source.parents[1]/"qualify-mass-correction.py"]
    result = dict(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    try:
        result["baseline_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
        result["branch"] = subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
        result["working_tree_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,stderr=subprocess.DEVNULL,text=True))
    except subprocess.CalledProcessError:
        result["git_unavailable"] = True
    return result


def read_neighborhoods(reader, references, settings):
    scans = [[] for _ in references]
    for scan in reader.scans:
        if reader.experiments[scan.experiment-1].ms_level != 1:
            continue
        rt = scan.rt_seconds/60
        wanted = [i for i,r in enumerate(references) if abs(rt-r.expected_rt) <= settings.rt_radius]
        if not wanted:
            continue
        mz, intensity = reader.arrays(scan)
        for i in wanted:
            reference = references[i]
            radius = (settings.broad_ppm + 2*settings.lobe_ppm)*1e-6
            a,b = np.searchsorted(mz, reference.expected_mz*(1+np.array([-radius, radius])))
            scans[i].append(ProfileScan(scan.native_id, rt, mz[a:b].copy(), intensity[a:b].copy()))
    return scans


def sensitivity(observations, estimates, chosen, contexts, references):
    result = dict(estimator_models={}, omissions=[], rt_perturbations=[], high_scan_fraction=[], apex_neighbors=[])
    for method in METHODS:
        rows = [e for e in estimates if e["method"] == method and "failure" not in e]
        if len(rows) != len(references):
            result["estimator_models"][method] = dict(failure="At least one estimator failed")
        else:
            observations_m = [Observation(r, e["observed_mz"], e["rt"], method) for r,e in zip(references, rows)]
            result["estimator_models"][method] = benchmark(observations_m)
    for name in MODEL_NAMES:
        for i,row in enumerate(observations):
            try:
                model = fit(name, [r for j,r in enumerate(observations) if j != i])
                result["omissions"].append(dict(model=name, omitted=row.reference.id, coefficients=asdict(model),
                    all_residuals_ppm=[residual(model,r) for r in observations], held_out_residual_ppm=residual(model,row)))
            except ValueError as error:
                result["omissions"].append(dict(model=name, omitted=row.reference.id, failure=str(error)))
    # RT uncertainty is a sensitivity scenario, not additional measured points.
    scenarios = [("supplied_approximate_rt", [r.expected_rt-o.rt for r,o in zip(references,observations)])]
    for delta in (.03, .05):
        scenarios.extend((f"independent_rt_{delta:g}_"+"".join("+" if s>0 else "-" for s in signs),
                          [delta*s for s in signs]) for signs in product((-1,1), repeat=len(observations)))
    for label, shifts in scenarios:
        changed = [replace(o, rt=o.rt+shift) for o,shift in zip(observations,shifts)]
        for record in benchmark(changed):
            if "full_model" not in record:
                continue
            model = Model(**record["full_model"])
            baseline = fit(record["model"], observations)
            deviations = [abs(residual(model,o)-residual(baseline,o)) for o in observations]
            result["rt_perturbations"].append(dict(scenario=label, shifts_minutes=shifts, model=record["model"],
                loo_metrics=record.get("loo_metrics"), max_correction_change_at_original_coordinates_ppm=max(deviations)))
    for fraction in (.3, .7):
        changed = []
        for ref,c,ctx in zip(references, chosen, contexts):
            trace = np.array(c["trace"])
            apex = c["apex_index"]
            lo = hi = apex
            while lo>0 and trace[lo-1] >= fraction*trace[apex]: lo-=1
            while hi+1<len(trace) and trace[hi+1] >= fraction*trace[apex]: hi+=1
            values = [c["measurements"][j]["centroid_mz"] for j in range(lo,hi+1) if c["measurements"][j]]
            changed.append(Observation(ref,float(np.median(values)),c["rt"],"multi_scan_centroid"))
        result["high_scan_fraction"].append(dict(fraction=fraction, observations=[asdict(o) for o in changed], models=benchmark(changed)))
    for method in METHODS:
        rows = [e for e in estimates if e["method"] == method and "failure" not in e]
        if len(rows) != len(references):
            continue
        base = [Observation(r,e["observed_mz"],e["rt"],method) for r,e in zip(references,rows)]
        for i,(ref,c,ctx) in enumerate(zip(references,chosen,contexts)):
            for direction in (-1,1):
                neighbor=c["apex_index"]+direction
                record=dict(method=method, perturbed_reference=ref.id, direction=direction)
                if neighbor not in c["high_indices"] or c["measurements"][neighbor] is None:
                    record["failure"]="No eligible high scan adjacent to apex"
                else:
                    measurement=c["measurements"][neighbor]
                    field={METHODS[0]:"sampled_mz",METHODS[1]:"centroid_mz",METHODS[3]:"parabola_mz"}.get(method)
                    mass=measurement[field] if field else base[i].observed_mz
                    if mass is None:
                        record["failure"]="Adjacent scan estimator failed"
                    else:
                        changed=base.copy()
                        changed[i]=Observation(ref,mass,ctx["rt"][neighbor],method)
                        record.update(scan_id=ctx["scan_ids"][neighbor],rt_change_minutes=ctx["rt"][neighbor]-c["rt"],
                                      raw_ppm_change=changed[i].ppm_error-base[i].ppm_error,models=benchmark(changed))
                result["apex_neighbors"].append(record)
    return result


def scan_validation(reader, models):
    results = []
    for model in models:
        start = time.perf_counter()
        minimum, maximum = float("inf"), -float("inf")
        shift_min, shift_max = float("inf"), -float("inf")
        scans = samples = ms2 = 0
        for scan in reader.scans:
            mz, intensity = reader.arrays(scan)
            if reader.experiments[scan.experiment-1].ms_level != 1:
                ms2 += 1
                continue
            corrected, signal, checks = correct_scan(model, mz, intensity, scan.rt_seconds/60)
            assert signal is intensity and len(corrected) == len(mz)
            scans += 1
            samples += len(mz)
            minimum = min(minimum, checks["minimum_spacing_ratio"])
            maximum = max(maximum, checks["maximum_spacing_ratio"])
            shifts = (corrected-mz)/mz*1e6
            shift_min = min(shift_min, float(shifts.min()))
            shift_max = max(shift_max, float(shifts.max()))
        results.append(dict(model=model.name, ms1_scans=scans, profile_samples=samples, ms2_scans_left_unchanged=ms2,
            all_axes_strictly_increasing=True, all_sample_counts_unchanged=True, all_intensities_unchanged=True,
            all_retention_times_unchanged=True, minimum_spacing_ratio=minimum, maximum_spacing_ratio=maximum,
            correction_min_ppm=shift_min, correction_max_ppm=shift_max, seconds=time.perf_counter()-start))
    return results


def ms2_inventory(reader, references, observations):
    rows = []
    for scan in reader.scans:
        if reader.experiments[scan.experiment-1].ms_level == 1:
            continue
        mz, intensity = reader.arrays(scan)
        order = np.argsort(intensity)[-5:][::-1]
        rows.append(dict(scan_id=scan.native_id, rt=scan.rt_seconds/60, experiment=scan.experiment,
            precursor_nominal_mz=scan.precursor_mz, profile_points=len(mz),
            strongest_sampled_mz=mz[order].tolist(), strongest_sampled_intensities=intensity[order].tolist(),
            near_reference_ids=[r.id for r in references if abs(scan.rt_seconds/60-r.expected_rt)<=.45]))
    return dict(scans=rows,
        evidence="Same decode_tokens/calibrate/frame_profile functions, separate per-scan cal_a/cal_b. Shared code does not establish shared residual mass error.",
        conclusion="No exact product-ion masses or standard identities were supplied. Fragment mass-error equivalence is unvalidated; all MS2 arrays and nominal precursor/isolation/window metadata remain unchanged.")


def run(entry, settings_path, destination, experimental=False):
    start = time.perf_counter()
    entry, destination = Path(entry).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(ROOT/".local") or destination.exists():
        raise ValueError("Use a new directory inside this repository's ignored .local/")
    settings_data = json.loads(Path(settings_path).read_text())
    references = [Reference(**r) for r in settings_data["references"]]
    if len(references) != 4 or len({r.id for r in references}) != 4:
        raise ValueError("This pilot requires exactly four distinct reference IDs")
    settings = SearchSettings(**settings_data.get("search", {}))
    destination.mkdir(parents=True)
    save_json(destination/"settings.json", dict(references=[asdict(r) for r in references], search=asdict(settings)))
    paths = [entry, entry.with_suffix(".wiff.scan")]
    before = [identity(p, f"input-{i+1}") for i,p in enumerate(paths) if p.is_file()]
    result = dict(status="not_qualified", input_identity=before, references=[asdict(r) for r in references],
                  settings=asdict(settings), runtime=dict(python=platform.python_version(), numpy=np.__version__),
                  code_provenance=code_provenance(),
                  default_estimator="multi_scan_centroid", outputs=[], source_unchanged=None)
    try:
        with WiffReader(entry) as reader:
            result["acquisition"] = dict(spectra=len(reader.scans), ms1=sum(reader.experiments[s.experiment-1].ms_level==1 for s in reader.scans),
                ms2=sum(reader.experiments[s.experiment-1].ms_level==2 for s in reader.scans),
                rt_range_minutes=[reader.scans[0].rt_seconds/60,reader.scans[-1].rt_seconds/60],
                calibration_pairs_by_experiment=[len({(s.cal_a,s.cal_b) for s in reader.scans if s.experiment==i+1}) for i in range(len(reader.experiments))])
            scans = read_neighborhoods(reader, references, settings)
            allc, contexts = [], []
            for reference, local in zip(references,scans):
                c, context = candidates(reference, local, settings)
                allc.append(c); contexts.append(context)
            chosen, evidence = consensus(allc, settings)
            result["detection"] = evidence
            save_json(destination/"detection.json", dict(candidates=allc, contexts=contexts, consensus=evidence))
            flat = [{k:v for k,v in c.items() if k not in ("trace", "measurements", "high_indices")} for rows in allc for c in rows]
            save_csv(destination/"candidates.csv", flat)
            # Render all references even on failure; no failed match disappears.
            from .report import reference_diagnostics, write_report
            reference_diagnostics(destination, references, scans, chosen or [rows[0] if rows else None for rows in allc], contexts, settings, matched=chosen is not None)
            if chosen is None:
                result["failure"] = evidence["failure"]
            else:
                estimates, scan_rows = [], []
                for reference,c,ctx in zip(references,chosen,contexts):
                    ee,rr = estimate(reference,c,ctx); estimates.extend(ee); scan_rows.extend(rr)
                result["estimates"] = estimates
                save_csv(destination/"mass_estimators.csv", estimates)
                save_csv(destination/"scan_measurements.csv", scan_rows)
                best_estimates = [e for e in estimates if e["method"] == result["default_estimator"]]
                observations = [Observation(r,e["observed_mz"],e["rt"],e["method"]) for r,e in zip(references,best_estimates)]
                table = [dict(expected_mz=o.reference.expected_mz, expected_rt=o.reference.expected_rt,
                              chosen_candidate_id=c["candidate_id"], **e) for o,c,e in zip(observations,chosen,best_estimates)]
                result["raw_reference_table"] = table
                save_csv(destination/"references.csv", table)
                models = benchmark(observations)
                result["models"] = models
                save_json(destination/"models.json", models)
                save_csv(destination/"models.csv", [dict(model=m["model"],parameters=m["parameters"],
                    **{"full_"+k:v for k,v in m.get("full_metrics",{}).items()}, **{"loo_"+k:v for k,v in m.get("loo_metrics",{}).items()}) for m in models])
                save_csv(destination/"held_out_predictions.csv", [dict(model_name=m["model"],**f) for m in models for f in m["folds"]])
                sensitivity_data = sensitivity(observations,estimates,chosen,contexts,references)
                result["sensitivity"] = sensitivity_data
                save_json(destination/"sensitivity.json", sensitivity_data)
                save_csv(destination/"omissions.csv", sensitivity_data["omissions"])
                save_csv(destination/"rt_sensitivity.csv", sensitivity_data["rt_perturbations"])
                result["ms2"] = ms2_inventory(reader,references,observations)
                save_json(destination/"ms2_inventory.json", result["ms2"])
                # Mechanically demonstrate ALL five models on complete MS1
                # scans. Only the simple baseline and lowest-CV candidate are
                # exported. Array checks do not qualify extrapolated accuracy.
                fitted = [fit(name,observations) for name in MODEL_NAMES]
                result["scan_validation"] = scan_validation(reader,fitted)
                save_csv(destination/"scan_validation.csv", result["scan_validation"])
                if experimental:
                    predictive = [m for m in models if "loo_metrics" in m]
                    lowest = min(predictive,key=lambda m:m["loo_metrics"]["rms_ppm"])["model"]
                    for name in dict.fromkeys(("constant_ppm",lowest)):
                        model = fit(name,observations)
                        path = write_experimental(reader,destination/f"qualification-MS1-{name}-profile.mzML",model)
                        validation = verify_experimental(reader,path,model)
                        result["outputs"].append(dict(name=path.name,bytes=path.stat().st_size,validation=validation))
                result["status"] = "pilot_measured_production_unqualified"
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        result["status"] = "qualification_failed"
    finally:
        after = [identity(p,f"input-{i+1}") for i,p in enumerate(paths) if p.is_file()]
        result["source_unchanged"] = before == after
        if not result["source_unchanged"]:
            result["status"] = "failed_input_identity_changed"
        result["elapsed_seconds"] = time.perf_counter()-start
        save_json(destination/"results.json",result)
        from .report import write_report
        write_report(destination,result)
    return result
