#!/usr/bin/env python3
"""Known-truth, sensitivity and optional private-input checks for KmzCoMet."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kmzcomet.engine import analyze, file_hash
from kmzcomet.excel import write_workbook
from kmzcomet.model import Settings
from scripts.benchmarks.comet_fixtures import Truth, cohort, write_run


def truth_metrics(result, truth, shifts):
    matched, present, absent, recovered, missed, false_support = set(), 0, 0, 0, 0, 0
    ratios, duplicate, missing = [], 0, 0
    discovery_missing = 0
    for expected in truth:
        indices = [i for i, f in enumerate(result.features) if f.polarity == expected.polarity and
                   abs(f.mz-expected.mz)/expected.mz*1e6 < 5 and abs(f.rt-expected.rt) < .035]
        if not indices:
            missing += 1
            missed += sum(h > 0 for h in expected.heights)
            present += sum(h > 0 for h in expected.heights)
            absent += sum(h == 0 for h in expected.heights)
            continue
        duplicate += len(indices)-1
        matched.update(indices)
        index = indices[0]
        for column, amplitude in zip(result.measurements, expected.heights):
            measurement = column[index]
            if amplitude > 0:
                present += 1
                discovery_missing += not measurement.discovered
                if measurement.status != "supported":
                    missed += 1
                if not measurement.discovered and measurement.status == "supported":
                    recovered += 1
                if measurement.height is not None:
                    ratios.append(measurement.height/amplitude)
            else:
                absent += 1
                false_support += measurement.status == "supported"
    return dict(expected_features=len(truth), reported_features=len(result.features),
                missing_features=missing, duplicate_features=duplicate,
                unmatched_features=len(result.features)-len(matched),
                present_pairs=present, absent_pairs=absent, discovery_missing_pairs=discovery_missing,
                recovered_pairs=recovered, unsupported_present_pairs=missed,
                false_supported_absent_pairs=false_support,
                median_height_ratio=float(np.median(ratios)) if ratios else None,
                max_height_relative_error=float(np.max(np.abs(np.array(ratios)-1))) if ratios else None,
                max_rt_shift_error_min=max(abs(s.shifts.get(1,0)-d) for s,d in zip(result.samples,shifts)),
                elapsed_seconds=result.elapsed_seconds)


def verify_workbook(path, result):
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=False)
    rows = list(book["Heights"].iter_rows(values_only=True))
    assert len(rows) == len(result.features)+1
    assert len(rows[0]) == len(result.samples)+4
    for i,row in enumerate(rows[1:]):
        for j,actual in enumerate(row[4:]):
            expected = result.measurements[j][i].height
            assert (actual is None and expected is None) or np.isclose(actual,expected,rtol=1e-12,atol=1e-12)
    assert not any(c.data_type == "f" for ws in book for row in ws for c in row)
    book.close()


def independent_real_eics(result):
    """Use the existing viewer index/reader, not the CoMet extractor."""
    from psims.controlled_vocabulary.controlled_vocabulary import obo_cache
    obo_cache.use_remote = False
    from kmzviewer.mzml_index import index_ms1
    from kmzviewer.spectra import SpectrumReader
    if not result.features:
        return 0
    indices = np.unique(np.linspace(0,len(result.features)-1,min(10,len(result.features)),dtype=int))
    checked = 0
    for j,sample in enumerate(result.samples):
        table = index_ms1(sample.path)
        with SpectrumReader(sample.path) as reader:
            for i in indices:
                feature = result.features[i]
                shift = sample.shifts.get(feature.polarity,0)
                selected = np.flatnonzero((table.rt_min >= feature.left+shift) &
                                         (table.rt_min < feature.right+shift) &
                                         (table.polarity == feature.polarity))
                native_max = 0.
                for index in selected:
                    mz,intensity = reader.read(table.spectrum_index[index])
                    keep = np.abs(mz-feature.mz) <= feature.mz*result.settings.mz_ppm*1e-6
                    native_max = max(native_max,float(np.max(intensity[keep],initial=0)))
                actual = result.measurements[j][i].height
                assert actual is None if not len(selected) else np.isclose(actual,native_max,rtol=1e-12)
                checked += 1
    return checked


def figures(output, result, truth, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kmzcomet.reader import iter_scans
    examples = [truth[0],next(t for t in truth if t.id == "only-first")]
    fig,axes = plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    colors = ["#295A8C","#C77C28","#7759A6","#32785F"]
    for ax,example in zip(axes,examples):
        for j,path in enumerate(paths):
            t,y = [],[]
            for scan in iter_scans(path):
                rt = scan.rt-result.samples[j].shifts.get(1,0)
                if abs(rt-example.rt) <= .15:
                    selected = abs(scan.mz-example.mz)/example.mz*1e6 <= result.settings.mz_ppm
                    t.append(rt)
                    y.append(np.max(scan.intensity[selected],initial=0))
            ax.plot(t,y,label=f"S{j+1:03d}",color=colors[j],linewidth=1.7)
        ax.set_xlabel("Aligned RT (min)")
        ax.set_ylabel("Native intensity")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=8)
        ax.set_title("Shared peak, including a weak sample" if example.id.startswith("shared") else "Present only in S001")
    fig.savefig(output/"synthetic-eics.png",dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="New local output directory")
    parser.add_argument("--local",type=Path,nargs="*",default=[],help="Optional private mzML files (kept local)")
    parser.add_argument("--identity",action="store_true",help="Also compare the first local file with a temporary exact copy")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    settings = Settings()
    report = {"settings":asdict(settings),"cases":{},"limits":[
        "Synthetic truth is generic and favorable to the baseline; not an external biological validation.",
        "Real-file checks establish extraction consistency, not which metabolites are present.",
        "No MZmine execution or accuracy comparison was performed."]}
    progress = lambda message: print(message,flush=True)
    with tempfile.TemporaryDirectory(prefix="generated-",dir=args.output) as temporary:
        temporary = Path(temporary)
        for mode in ("centroid","profile"):
            paths,truth,shifts = cohort(temporary/mode,profile=mode=="profile")
            result = analyze(paths,settings,progress=progress)
            report["cases"][f"synthetic_{mode}"] = truth_metrics(result,truth,shifts)
            path = write_workbook(result,args.output/f"synthetic-{mode}.xlsx")
            verify_workbook(path,result)
            if mode == "centroid":
                figures(args.output,result,truth,paths)
                for name,modified in [("lower_discovery",replace(settings,discovery_height=300)),
                                      ("higher_discovery",replace(settings,discovery_height=2500)),
                                      ("narrow_mass",replace(settings,mz_ppm=5)),
                                      ("no_rt_alignment",replace(settings,alignment="none"))]:
                    report["cases"][name] = truth_metrics(analyze(paths,modified),truth,shifts)
        # Independent seeds and many known empty windows test false supporting
        # evidence; this is not a fitted or calibrated false-discovery estimate.
        for seed,noise_model in ((101,"uniform"),(203,"uniform"),(307,"uniform"),
                                 (409,"uniform"),(503,"exponential")):
            truth = [Truth(f"empty-{i}",120+i*7.31,.6+(i%12)*.3,(9000.,0.,0.,0.)) for i in range(60)]
            truth += [Truth(f"shared-{i}",700+i*11.77,.5+i*.45,(12000.,9000.,600.,5000.)) for i in range(9)]
            truth += [Truth(f"anchor-{i}",1000+i*13.11,.6+i*.6,(14000.,14000.,14000.,14000.)) for i in range(6)]
            shifts = (0.,.07,-.04,.11)
            paths = [write_run(temporary/f"seed-{seed}-{j}.mzML",truth,sample=j,shift=shifts[j],
                               noise=80,seed=seed+j,noise_model=noise_model) for j in range(4)]
            report["cases"][f"absence_seed_{seed}"] = truth_metrics(analyze(paths),truth,shifts)
        if args.local:
            result = analyze(args.local,settings,progress=progress)
            path = write_workbook(result,args.output/"local-comparison.xlsx")
            verify_workbook(path,result)
            report["real"] = dict(features=len(result.features),elapsed_seconds=result.elapsed_seconds,
                                  independent_eic_checks=independent_real_eics(result),
                                  samples=[dict(id=s.id,bytes=s.size_bytes,sha256=s.sha256,scans=s.scans,
                                                candidates=len(s.candidates),shifts=s.shifts,anchors=s.anchors,
                                                alignment_mad=s.alignment_mad,warnings=s.warnings,
                                                evidence=dict(Counter(m.status for m in column)))
                                           for s,column in zip(result.samples,result.measurements)],
                                  discovery_missing_pairs=sum(not m.discovered for col in result.measurements for m in col),
                                  recovered_supported_pairs=sum(not m.discovered and m.status=="supported"
                                                                for col in result.measurements for m in col))
            if args.identity:
                copied = temporary/"identity-copy.mzML"
                shutil.copyfile(args.local[0],copied)
                identity = analyze([args.local[0],copied],settings,progress=progress)
                a,b = identity.measurements
                assert len(a)>0 and all(x.height==y.height and x.status==y.status for x,y in zip(a,b))
                assert all(len(f.members)==2 for f in identity.features)
                report["identity"] = dict(features=len(identity.features),equal_heights=True,
                                           equal_evidence=True,all_features_discovered_in_both=True,
                                           elapsed_seconds=identity.elapsed_seconds)
    report["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=="darwin" else 1024)
    report["workbook_readback"] = "All exported height cells checked against engine results; no formula cells"
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    lines = ["# KmzCoMet baseline benchmark", "", "Known-truth synthetic cases; all processing was local.", "",
             "| Case | Features expected/reported | Present pairs | Missing in discovery | Recovered supported | Unsupported present | Absent pairs | False supported absent |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,c in report["cases"].items():
        lines.append(f"| {name} | {c['expected_features']}/{c['reported_features']} | {c['present_pairs']} | {c['discovery_missing_pairs']} | {c['recovered_pairs']} | {c['unsupported_present_pairs']} | {c['absent_pairs']} | {c['false_supported_absent_pairs']} |")
    lines += ["", "Settings and full aggregate metrics: `summary.json`.", "",
              "Synthetic workbooks are reproducible examples. Local workbooks contain private results and remain local.", "",
              *["- "+limit for limit in report["limits"]], "",
              "Temporary generated mzMLs and the identity copy were removed; original inputs were hash-verified by the engine."]
    (args.output/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
