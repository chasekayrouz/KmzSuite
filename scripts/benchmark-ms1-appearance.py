#!/usr/bin/env python3
"""Regenerate isolated MS1 inspection qualification from local real inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/validation"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/state/matplotlib"))
os.environ.setdefault("KMZ_LOCAL_STATE_ROOT", str(ROOT / ".local/state"))

import numpy as np
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import iter_records
from kmzconvert_core.writer import write_mzml
from kmzviewer.spectra import SpectrumReader
from scripts.benchmarks.ms1_algorithms import guard_indices, gap_connections, validate_arrays, shape_metrics
from scripts.benchmarks.ms1_mzml import write_raw_inspection, filter_profile_mzml
from scripts.benchmarks.ms1_report import comparison_figure, actual_viewer_frame, html_report, technical_report
from check_mzml import check_index


def csv_write(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def identity(path):
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path":os.path.relpath(path, ROOT), "bytes":path.stat().st_size, "sha256":digest}


def source_scans(path):
    if path.suffix.lower() in (".wiff", ".wiff2"):
        with WiffReader(path) as reader:
            for index, scan in enumerate(reader.scans):
                if reader.experiments[scan.experiment-1].ms_level == 1:
                    mz, y = reader.arrays(scan)
                    yield {"dataset":path.stem, "index":index, "id":scan.native_id, "rt_seconds":scan.rt_seconds, "mz":mz, "y":y}
    else:
        for index, record in enumerate(iter_records(path)):
            if record["ms_level"] == 1:
                if record["representation"] != "profile":
                    raise ValueError(f"Nonprofile input: {path.name}, index {index}")
                yield {"dataset":path.stem, "index":index, "id":record["id"], "rt_seconds":record["rt_seconds"],
                       "mz":record["arrays"]["mz"], "y":record["arrays"]["intensity"]}


def describe(scan):
    mz, y = validate_arrays(scan["mz"], scan["y"])
    above = y > 100
    hard = guard_indices(y, guards=0)
    g1, g2 = guard_indices(y), guard_indices(y, guards=2)
    area = float(np.trapezoid(y,mz))
    hard_area = float(np.trapezoid(y[hard],mz[hard]))
    maxima = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))+1
    moderate = (y > 100) & (y <= 300)
    apex = int(np.argmax(y))
    clustered = np.sum((abs(mz[maxima] - mz[apex]) <= .25) & (y[maxima] >= max(100, y[apex]*.05)))
    return {k:scan[k] for k in ("dataset","index","id","rt_seconds")} | {
        "profile_points":len(y), "hard100_points":len(hard), "guard1_points":len(g1), "guard2_points":len(g2),
        "exactly_100_points":int((y==100).sum()), "moderate_points":int(moderate.sum()),
        "moderate_fraction":float(moderate.sum()/max(1,len(hard))),
        "density":len(hard)/max(1e-9,float(mz[-1]-mz[0])), "base_peak":float(y[apex]),
        "cluster_score":int(clustered), "area_distortion":abs(hard_area-area)/max(area,1e-9)}


def select_scans(rows):
    selected = defaultdict(list)
    def add(row, reason):
        selected[row["index"]].append(reason)
    ordered = sorted(rows,key=lambda r:(r["rt_seconds"],r["index"]))
    for fraction in (.1,.35,.65,.9):
        add(ordered[int(round((len(ordered)-1)*fraction))],f"RT fraction {fraction}")
    nonempty = [r for r in rows if r["hard100_points"]>=5] or rows
    for key, maximize, label in [("density",True,"dense"),("density",False,"sparse"),
                                 ("moderate_points",True,"many threshold-adjacent signals"),
                                 ("area_distortion",True,"largest control area distortion"),
                                 ("cluster_score",True,"nearby peak cluster")]:
        row = sorted(nonempty,key=lambda r:((-r[key] if maximize else r[key]),r["index"]))[0]
        add(row,label)
    isolated=[r for r in nonempty if r["cluster_score"]==1] or nonempty
    add(sorted(isolated,key=lambda r:(-r["base_peak"],r["index"]))[0],"strong isolated apex")
    empty=[r for r in rows if r["hard100_points"]==0]
    if empty:
        add(empty[len(empty)//2],"all measurements below cutoff")
    return dict(selected)


def measured_fidelity(source_mz, source_y, indices, mz, y):
    if len(indices)!=len(mz) or len(mz)!=len(y):
        raise AssertionError("Retained point count mismatch")
    dm=np.abs(mz-source_mz[indices]); dy=np.abs(y-source_y[indices])
    relative=np.divide(dy,np.abs(source_y[indices]),out=np.zeros(len(y)),where=source_y[indices]!=0)
    if np.any(dy[source_y[indices]==0]):
        raise AssertionError("A zero guard changed")
    assert not dm.any() and not dy.any(), "Retained source measurements changed"
    return {"mz_max_absolute_deviation":float(dm.max(initial=0)), "mz_median_absolute_deviation":float(np.median(dm)) if len(dm) else 0,
            "intensity_max_relative_deviation":float(relative.max(initial=0)), "intensity_median_relative_deviation":float(np.median(relative)) if len(relative) else 0}


def evaluate_selected(scan):
    metrics, peaks, sweep=[] ,[],[]
    mz,y=scan["mz"],scan["y"]
    prefix={k:scan[k] for k in ("dataset","index","id","rt_seconds")}
    indices={g:guard_indices(y,guards=g) for g in (0,1,2)}
    for condition,guards,gap in [("A",0,False),("B",1,False),("C",0,True),("D",1,True)]:
        idx=indices[guards]
        start=time.perf_counter()
        connections=gap_connections(mz[idx]) if gap else np.ones(max(0,len(idx)-1),dtype=bool)
        detector_seconds=time.perf_counter()-start
        row,p=shape_metrics(mz,y,idx,connections)
        row.update(measured_fidelity(mz,y,idx,mz[idx],y[idx]))
        metrics.append(prefix | {"condition":condition,"guard_increase_percent":100*(len(idx)-len(indices[0]))/len(indices[0]) if len(indices[0]) else 0,
                                 "gap_detector_seconds":detector_seconds} | row)
        peaks.extend(prefix | {"condition":condition} | entry for entry in p)
    for guards in (0,1,2):
        idx=indices[guards]
        for factor in (1.5,2.,4.,8.):
            row,_=shape_metrics(mz,y,idx,gap_connections(mz[idx],factor=factor))
            sweep.append(prefix | {"guards":guards,"gap_factor":factor} | row)
    return metrics,peaks,sweep


def verify_file(source, candidate, guards, raw=False):
    """Compare every scan and chromatogram, not only selected image scans."""
    checked=points=0
    if raw:
        reader=WiffReader(source)
        expected=((i,s.native_id,reader.experiments[s.experiment-1].ms_level,*reader.arrays(s)) for i,s in enumerate(reader.scans))
    else:
        reader=None
        expected=((i,r["id"],r["ms_level"],r["arrays"]["mz"],r["arrays"]["intensity"]) for i,r in enumerate(iter_records(source)))
    try:
        for original,record in zip(expected,iter_records(candidate),strict=True):
            index,ident,level,mz,y=original
            idx=guard_indices(y,guards=guards) if level==1 and guards is not None else np.arange(len(mz))
            assert record["id"]==ident and record["ms_level"]==level and record["representation"]=="profile"
            measured_fidelity(mz,y,idx,record["arrays"]["mz"],record["arrays"]["intensity"])
            checked+=1; points+=len(idx)
        if raw:
            records=list(iter_records(candidate,"chromatogram"))
            assert len(records)==1
            for kind,original in zip(("time","intensity","ms level"),reader.tic_arrays(),strict=True):
                np.testing.assert_array_equal(records[0]["arrays"][kind],original)
        else:
            for a,b in zip(iter_records(source,"chromatogram"),iter_records(candidate,"chromatogram"),strict=True):
                assert a["id"]==b["id"] and a["arrays"].keys()==b["arrays"].keys()
                for kind in a["arrays"]:
                    np.testing.assert_array_equal(a["arrays"][kind],b["arrays"][kind])
    finally:
        if reader is not None: reader.close()
    return {"spectra_verified":checked,"retained_measurements_verified":points,"arrays_exact":True,"chromatograms_exact":True,"index":check_index(candidate)}


def file_benchmark(path, output, raw):
    results=[]
    for condition,guards in ([("reference",None)] if raw else [])+[("A",0),("B",1)]:
        destination=output/"mzml"/f"{path.stem}-{condition}.mzML"
        start=time.perf_counter()
        if raw:
            with WiffReader(path) as reader:
                if guards is None: write_mzml(reader,destination)
                else: write_raw_inspection(reader,destination,guards)
        else:
            filter_profile_mzml(path,destination,guards)
        seconds=time.perf_counter()-start
        print(f"  wrote {destination.name}: {destination.stat().st_size/1e6:.1f} MB in {seconds:.2f}s",flush=True)
        verification=verify_file(path,destination,guards,raw=raw)
        start=time.perf_counter()
        count=0
        for record in iter_records(destination): count+=1
        parsing=time.perf_counter()-start
        results.append({"dataset":path.stem,"condition":condition,"path":str(destination.relative_to(output)),"bytes":destination.stat().st_size,
                        "conversion_seconds":seconds,"parse_seconds":parsing,"spectra":count,"verification":verification,"file_reused_from":None})
    if not raw:
        start=time.perf_counter()
        count=sum(1 for _ in iter_records(path))
        results.insert(0,{"dataset":path.stem,"condition":"reference","path":os.path.relpath(path,output),"bytes":path.stat().st_size,
                          "conversion_seconds":None,"parse_seconds":time.perf_counter()-start,"spectra":count,
                          "verification":{"role":"Existing independent profile input; not a WIFF pair"},"file_reused_from":None})
    a=next(r for r in results if r["condition"]=="A")
    ref=next((r["bytes"] for r in results if r["condition"]=="reference"),path.stat().st_size)
    for r in results:
        r["size_increase_over_A_percent"]=100*(r["bytes"]-a["bytes"])/a["bytes"]
        r["reduction_from_profile_percent"]=100*(1-r["bytes"]/ref)
    for condition,original in (("C","A"),("D","B")):
        results.append(next(r for r in results if r["condition"]==original) | {"condition":condition,"file_reused_from":original})
    return results


def verify_viewer_loading(costs, selected, output):
    """Independent pyteomics/KmzViewer indexed decode of selected file spectra."""
    results=[]
    for file in costs:
        if file.get('file_reused_from'):
            continue
        path=Path(file['path'])
        path=path if path.is_absolute() else output/path
        if not path.exists():
            # Existing independent profile inputs are identified relative to the
            # repository in retained metadata, rather than copied into results.
            path=ROOT/file['path']
        with SpectrumReader(path) as reader:
            for scan in (s for s in selected if s['dataset']==file['dataset']):
                start=time.perf_counter()
                mz,y=reader.read(scan['index'])
                seconds=time.perf_counter()-start
                guards={'reference':None,'A':0,'B':1}[file['condition']]
                idx=np.arange(len(scan['mz'])) if guards is None else guard_indices(scan['y'],guards=guards)
                fidelity=measured_fidelity(scan['mz'],scan['y'],idx,mz,y)
                results.append({'dataset':scan['dataset'],'index':scan['index'],'condition':file['condition'],
                                'indexed_load_seconds':seconds,'decoded_points':len(mz),**fidelity})
    csv_write(output/'viewer-loading.csv',results)
    return results


def aggregate(metrics):
    rows=[]
    for condition in "ABCD":
        group=[r for r in metrics if r["condition"]==condition]
        values=lambda key:[r[key] for r in group if r[key] is not None]
        rows.append({"condition":condition,"spectra":len(group),
                     "median_IAD_ratio":round(float(np.median(values("ink_iad_normalized"))),4),
                     "p90_IAD_ratio":round(float(np.quantile(values("ink_iad_normalized"),.9)),4),
                     "median_NRMSE_apex":round(float(np.median(values("ink_nrmse_apex"))),5),
                     "median_boundary_MAE_counts":round(float(np.median(values("threshold_edge_mae_counts"))),2),
                     "remaining_deleted_bridges":sum(values("deleted_regions_bridged")),
                     "breaks_between_source_neighbors":sum(values("breaks_between_source_neighbors")),
                     "incomplete_FWHM":sum(values("fwhm_incomplete")),
                     "major_apices":sum(values("major_apices")),
                     "median_guard_increase_percent":round(float(np.median(values("guard_increase_percent"))),2),
                     "median_gap_ms":round(float(np.median(values("gap_detector_seconds")))*1000,3)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",type=Path,default=ROOT/"TestData/Local")
    parser.add_argument("--output",type=Path,required=True,help="New local directory; never overwrite a prior run")
    parser.add_argument("--raw",type=Path,action="append",help="Explicit complete WIFF input; repeatable")
    parser.add_argument("--profile",type=Path,action="append",help="Explicit real profile mzML; repeatable")
    parser.add_argument("--skip-file-benchmark",action="store_true",help="Only post-conversion numerical/visual analysis; mark file costs unmeasured")
    args=parser.parse_args()
    paths=(args.raw or sorted(args.fixtures.glob("*.wiff")))+(args.profile or sorted(args.fixtures.glob("*.mzML")))
    paths=[p.resolve() for p in paths]
    if not paths: parser.error("No real raw/profile fixture available. Supply --raw with its sidecar, or --profile.")
    if len({p.stem for p in paths})!=len(paths): parser.error("Input stems must be distinct to identify datasets")
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output.resolve()
    for name in ("figures","mzml","selected-arrays"): (output/name).mkdir()
    inventory=[]; census=[]; selections={}; populations=[]
    for path in paths:
        raw=path.suffix.lower()==".wiff"
        files=[path,path.with_suffix(".wiff.scan")] if raw else [path]
        if not all(p.is_file() for p in files): parser.error(f"Incomplete fixture: {path.name}")
        inventory.append({"dataset":path.stem,"kind":"complete WIFF+SCAN" if raw else "retained real profile mzML", "files":[identity(p) for p in files]})
        started=time.perf_counter()
        rows=[describe(scan) for scan in source_scans(path)]
        if not rows: raise ValueError(f"No MS1 scans in {path.name}")
        census.extend(rows); selections[path.stem]=select_scans(rows)
        sums={key:sum(r[key] for r in rows) for key in ("profile_points","hard100_points","guard1_points","guard2_points","exactly_100_points")}
        populations.append({"dataset":path.stem,"ms1_spectra":len(rows),"rt_min":min(r["rt_seconds"] for r in rows)/60,"rt_max":max(r["rt_seconds"] for r in rows)/60,
                            "census_seconds":time.perf_counter()-started,**sums,
                            "guard1_increase_percent":100*(sums["guard1_points"]-sums["hard100_points"])/max(1,sums["hard100_points"]),
                            "guard2_increase_percent":100*(sums["guard2_points"]-sums["hard100_points"])/max(1,sums["hard100_points"])})
        print(f'{path.name}: {len(rows)} real MS1 scans, {len(selections[path.stem])} selected',flush=True)
    csv_write(output/"spectrum-census.csv",census); csv_write(output/"population-retention.csv",populations)
    metrics=[]; peaks=[]; sweep=[]; selected=[]
    for path in paths:
        for scan in source_scans(path):
            if scan["index"] not in selections[path.stem]: continue
            scan["key"]=f'{scan["dataset"]}-scan-{scan["index"]:05}'
            scan["reasons"]=selections[path.stem][scan["index"]]
            selected.append(scan)
            np.savez_compressed(output/"selected-arrays"/f'{scan["key"]}.npz',mz=scan["mz"],intensity=scan["y"])
            m,p,s=evaluate_selected(scan); metrics.extend(m); peaks.extend(p); sweep.extend(s)
        print(f'{path.stem}: selected-spectrum metrics and parameter sweep complete',flush=True)
    csv_write(output/"per-spectrum-metrics.csv",metrics); csv_write(output/"peak-metrics.csv",peaks); csv_write(output/"parameter-sweep.csv",sweep)
    summary=aggregate(metrics); csv_write(output/"summary.csv",summary)
    costs=[]
    if not args.skip_file_benchmark:
        raw_pop=[p for p in populations if any(path.stem==p["dataset"] and path.suffix.lower()==".wiff" for path in paths)]
        ordered=sorted(raw_pop,key=lambda p:(p["hard100_points"]/p["ms1_spectra"],p["dataset"]))
        raw_cost={ordered[i]["dataset"] for i in (0,len(ordered)//2,len(ordered)-1)} if ordered else set()
        for path in paths:
            raw=path.suffix.lower()==".wiff"
            if not raw or path.stem in raw_cost:
                print(f'File-cost and full-array verification: {path.name}',flush=True)
                costs.extend(file_benchmark(path,output,raw))
    csv_write(output/"file-costs.csv",[{k:v for k,v in r.items() if k!="verification"} for r in costs])
    viewer_loading=verify_viewer_loading(costs,selected,output)
    figures=[]; frames=[]
    visual=[]
    for path in paths:
        group=[s for s in selected if s["dataset"]==path.stem]
        labels=["largest control area distortion", "sparse"] if path.suffix.lower()==".wiff" else ["largest control area distortion","many threshold-adjacent signals","nearby peak cluster","strong isolated apex","RT fraction 0.35","RT fraction 0.9"]
        for label in labels:
            matches=[s for s in group if label in s["reasons"]]
            if matches and matches[0] not in visual: visual.append(matches[0])
    # Every quantitatively selected spectrum is shown; informative cases come
    # first, then the remainder in reproducible dataset/scan order.
    for scan in selected:
        if scan not in visual: visual.append(scan)
    for i,scan in enumerate(visual):
        # Vector exports for six informative, predetermined cases.
        vector=("largest control area distortion" in scan["reasons"] and (scan["dataset"] in {p.stem for p in paths[:1]+paths[-2:]})) or i in (1,3,5)
        figures.append(comparison_figure(scan,output/"figures",vector=vector))
        if i in (0,len(visual)-1): frames.extend(actual_viewer_frame(scan,output/"figures"))
        print(f'Figure {i+1}/{len(visual)}: {scan["key"]}',flush=True)
    csv_write(output/"viewer-frame-costs.csv",frames)
    metadata={"created_utc":datetime.now(timezone.utc).isoformat(),"baseline_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform(),"inputs":inventory,"populations":populations,
              "reference":"Unchanged current KmzConvert framed profile output; retained vendor-profile mzMLs are separate independent datasets, not paired with raw inputs",
              "control":"Reconstructed experimental MS1 intensity >100, no centroiding; not current or historical production behavior",
              "settings":{"threshold":100,"comparator":">","guards":1,"gap_factor":4,"radius":8,"coordinate":"sqrt(m/z)","guard_sweep":[0,1,2],"gap_sweep":[1.5,2,4,8],"ms2":"unchanged","tic":"vendor values unchanged"},
              "selection_criteria":"RT fractions .1,.35,.65,.9; dense/sparse; moderate points; area distortion; apex neighborhood clusters; strong isolated apex; empty cutoff where present; deterministic ties by index",
              "selected":[{k:v for k,v in s.items() if k not in ("mz","y")} for s in selected],"file_costs":costs,"figures":figures,"viewer_frames":frames,"viewer_loading":viewer_loading,
              "production_sources_modified":False,"prior_hard100_evidence_found":False,"input_identity_rechecked_at_end":all(identity(Path(ROOT/f["path"]))==f for entry in inventory for f in entry["files"])}
    assert metadata["input_identity_rechecked_at_end"]
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    narrative=technical_report(output,metadata,summary)
    html_report(output,summary,visual,figures,frames,narrative)
    print(f'COMPLETE: {len(census)} MS1 census, {len(selected)} selected, {len(figures)} aligned figures; {output/"report.html"}',flush=True)


if __name__=="__main__": main()
