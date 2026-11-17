"""Static scientific figures and a self-contained, offline benchmark report."""
from __future__ import annotations

import base64
import csv
from html import escape
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts.benchmarks.ms1_algorithms import guard_indices, gap_connections, plotted_trace

LABELS = ["Reference profile", "A · Hard-100 / current lines", "B · One guard / current lines", "C · Hard-100 / gap-aware", "D · One guard / gap-aware"]
COLORS = ["#25364b", "#ba4a35", "#2b6f95", "#8051a6", "#18795c"]


def representations(mz, y):
    hard, guard = guard_indices(y, guards=0), guard_indices(y)
    return [(np.arange(len(mz)), np.ones(max(0, len(mz)-1), dtype=bool)),
            (hard, np.ones(max(0, len(hard)-1), dtype=bool)),
            (guard, np.ones(max(0, len(guard)-1), dtype=bool)),
            (hard, gap_connections(mz[hard])), (guard, gap_connections(mz[guard]))]


def zooms(mz, y):
    if len(mz) < 2:
        return [(99, 101), (99, 101)]
    hard = guard_indices(y, guards=0)
    if len(hard) > 1:
        widths = np.diff(mz[hard])
        bridged_area = widths * (y[hard[:-1]] + y[hard[1:]]) / 2
        at = int(np.argmax(bridged_area * (np.diff(hard) > 1)))
        lo, hi = mz[hard[at]], mz[hard[at+1]]
        pad = max(.04, (hi-lo)*.12)
        artifact = (max(mz[0], lo-pad), min(mz[-1], hi+pad))
    else:
        at = int(np.argmax(y)); artifact = (mz[at]-.15, mz[at]+.15)
    maxima = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    moderate = maxima[(y[maxima] > 100) & (y[maxima] <= 300)]
    if len(moderate):
        at = int(moderate[np.argmin(abs(y[moderate]-150))])
    else:
        at = int(np.argmax(y))
    boundary = (max(mz[0], mz[at]-.08), min(mz[-1], mz[at]+.08))
    return [artifact, boundary]


def plot_line(ax, mz, y, indices, connections, color, gap=False, guards=False):
    m, v = mz[indices], y[indices]
    if len(m):
        # NaN separators are a drawing instruction only, never mzML values.
        x = np.repeat(m, 2)[1:]
        values = np.repeat(v, 2)[1:]
        if len(m)>1:
            values[1:-1:2][~connections] = np.nan
        ax.plot(x, values, lw=.75, color=color, rasterized=False)
        if gap:
            isolated = ~np.r_[False, connections] & ~np.r_[connections, False]
            ax.plot(m[isolated], v[isolated], ".", color=color, ms=2.5)
        if guards:
            added = v <= 100
            ax.plot(m[added], v[added], "o", mfc="white", mec="#d18b25", mew=.65, ms=2.7, alpha=.85)


def comparison_figure(scan, directory, vector=False):
    start = time.perf_counter()
    mz, y = scan["mz"], scan["y"]
    views = [(float(mz[0]), float(mz[-1]))] + zooms(mz, y)
    reps = representations(mz, y)
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":9, "svg.fonttype":"none", "path.simplify":False}):
        fig, axes = plt.subplots(5, 3, figsize=(16, 11.8), layout="constrained", sharex="col", sharey="col")
        for col, (lo, hi) in enumerate(views):
            values = y[(mz >= lo) & (mz <= hi)]
            grid = np.unique(np.r_[lo, mz[(mz>=lo)&(mz<=hi)], hi])
            heights = [float(values.max(initial=100))]
            for indices, connections in reps:
                trace = plotted_trace(grid, mz[indices], y[indices], connections)
                heights.append(float(np.nan_to_num(trace, nan=0).max(initial=0)))
            # Include crossing segments whose endpoints lie outside the zoom;
            # otherwise an artificial high baseline could be entirely clipped.
            ymax = max(heights) * 1.10
            for row, (indices, connections) in enumerate(reps):
                ax = axes[row, col]
                plot_line(ax, mz, y, indices, connections, COLORS[row], gap=row in (3,4), guards=row in (2,4) and col>0)
                ax.set(xlim=(lo, hi), ylim=(0, ymax), ylabel="Counts" if col==0 else "", xlabel="m/z" if row==4 else "")
                ax.grid(alpha=.15, lw=.4)
                ax.ticklabel_format(axis="x", useOffset=False)
                if col == 0:
                    ax.set_title(LABELS[row], loc="left", fontsize=10, color=COLORS[row])
                if row == 0 and col > 0:
                    ax.set_title(["", "Largest bridge by endpoint area", "Near-threshold peak boundary"][col], fontsize=10)
        fig.suptitle(f'{scan["dataset"]} · {scan["id"]} · RT {scan["rt_seconds"]/60:.3f} min\n'
                     'Identical axes within each column · amber circles = added source guards · dots = isolated observed points', fontsize=12)
        name = scan["key"]
        png = directory / f"{name}.png"
        fig.savefig(png, dpi=170)
        if vector:
            fig.savefig(directory / f"{name}.svg")
        plt.close(fig)
    return {"key":name, "views":[list(v) for v in views], "figure_seconds":time.perf_counter()-start,
            "png":png.name, "svg":f"{name}.svg" if vector else None}


def actual_viewer_frame(scan, directory):
    """Exercise the existing SpectrumPanel and patch curves only in this process."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    from kmzviewer.gui.plots import SpectrumPanel
    app = QApplication.instance() or QApplication([])
    mz, y = scan["mz"], scan["y"]
    lo, hi = zooms(mz,y)[1]
    grid = np.unique(np.r_[lo, mz[(mz>=lo)&(mz<=hi)], hi])
    ymax = max(float(y[(mz>=lo)&(mz<=hi)].max(initial=100)),
               *(float(np.nan_to_num(plotted_trace(grid,mz[i],y[i],c),nan=0).max(initial=0)) for i,c in representations(mz,y)))*1.1
    results = []
    for row, (indices, connected) in enumerate(representations(mz,y)):
        panel = SpectrumPanel()
        panel.resize(920, 320)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            panel.show_scan(mz[indices], y[indices], False, LABELS[row])
            # Fixed source-based labels/limits keep visual comparisons aligned.
            for label in panel._label_items:
                panel.plot.removeItem(label)
            panel._label_items.clear()
            if row in (3,4) and len(indices):
                panel.plot.listDataItems()[0].setData(mz[indices], y[indices], connect=np.r_[connected, False])
                isolated = ~np.r_[False,connected] & ~np.r_[connected,False]
                if np.any(isolated):
                    panel.plot.addItem(pg.ScatterPlotItem(mz[indices][isolated], y[indices][isolated], size=3, pen=None, brush="#1f4e79"))
            panel._guard = True
            panel.plot.setXRange(lo, hi, padding=0)
            panel.plot.setYRange(0, ymax, padding=0)
            panel._guard = False
            panel.show()
            app.processEvents()
            panel.grab()
            times.append(time.perf_counter()-start)
        path = directory / f'{scan["key"]}-qt-{row}.png'
        assert panel.grab().save(str(path))
        panel.close(); panel.deleteLater(); app.processEvents()
        results.append({"dataset":scan["dataset"], "spectrum_index":scan["index"], "condition":"reference" if row==0 else "ABCD"[row-1],
                        "frame_seconds_median":float(np.median(times)), "png":path.name})
    return results


def html_report(output, summary, scans, figures, frames, narrative):
    def table(rows):
        if not rows:
            return "<p>No observations.</p>"
        keys=list(rows[0])
        return '<div class="table"><table><thead><tr>'+''.join(f'<th>{escape(k)}</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{escape(str(row[k]))}</td>' for k in keys)+'</tr>' for row in rows)+'</tbody></table></div>'
    def embed(path, alt):
        data=base64.b64encode(path.read_bytes()).decode()
        return f'<img loading="lazy" src="data:image/png;base64,{data}" alt="{escape(alt)}">'
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>KmzSuite MS1 appearance benchmark</title><style>body{font:16px/1.55 system-ui;color:#203144;background:#f3f5f6;margin:0}main{max-width:1600px;margin:auto;padding:30px}h1{font-size:32px}h2{margin-top:36px}.notice{background:#fff0ce;border-left:5px solid #b47113;padding:18px}.table{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #d9e1e6;text-align:left}img{width:100%;height:auto;background:white}details{background:white;border:1px solid #d9e1e6;padding:12px;margin:16px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap}a{color:#176d9a}.qt{max-width:950px}</style><main>',
            '<h1>MS1 inspection-spectrum appearance</h1><p>Local experimental benchmark. No production conversion or viewer defaults changed.</p>',
            '<div class="notice"><strong>Experimental inspection control.</strong> Current production writes unfiltered framed profile arrays. A reconstructs strict intensity &gt;100 on MS1, without centroiding. The earlier inspection candidate was never production behavior; its qualification report and paired outputs were not retained. Results here are regenerated from current fixtures. The raw reference uses the unchanged production writer. MS2 values and vendor TICs remain unchanged in all experimental files.</div>',
            '<h2>Results and limitations</h2><pre>'+escape(narrative)+'</pre>', table(summary),
            '<h2>Aligned visual comparisons</h2><p>Full spectrum, largest endpoint-area bridge, and an automatically chosen near-threshold peak. No manual tuning. All five rows use identical limits, dimensions and linear scaling per column. Reference line follows the original profile representation, including its source framing zeros. Empty gaps in C/D contain no drawn measurements.</p>']
    costs_path=output/'file-costs.csv'
    if costs_path.exists() and costs_path.stat().st_size:
        costs=list(csv.DictReader(costs_path.open()))
        compact=[]
        for dataset in sorted({r['dataset'] for r in costs}):
            rows={r['condition']:r for r in costs if r['dataset']==dataset}
            a,b=rows['A'],rows['B']
            compact.append({'dataset':dataset,'A/C MB':round(float(a['bytes'])/1e6,2),'B/D MB':round(float(b['bytes'])/1e6,2),
                            'B/D increase vs A %':round(float(b['size_increase_over_A_percent']),2),
                            'B/D reduction from profile %':round(float(b['reduction_from_profile_percent']),2)})
        body.insert(-1,'<h2>File-size tradeoffs</h2><p>Decimal MB; unchanged MS2 included. C reuses A and D reuses B. Raw and post-conversion timings are recorded separately in the accompanying CSV.</p>'+table(compact))
    bykey={s["key"]:s for s in scans}
    for i,f in enumerate(figures):
        s=bykey[f["key"]]
        body.append(f'<details id="{escape(s["key"])}" {"open" if i==0 else ""}><summary>{escape(s["dataset"])} · scan index {s["index"]} · RT {s["rt_seconds"]/60:.3f} min · {escape(", ".join(s["reasons"]))}</summary>')
        body.append(embed(output/"figures"/f["png"], f'Five-condition comparison for {s["key"]}'))
        body.append('</details>')
    body.append('<h2>Actual KmzViewer plot-path checks</h2><p>Offscreen SpectrumPanel frames, same source-based limits. Experimental C/D change only the curve connection mask and show isolated observed points. No production source edits.</p>')
    for frame in frames:
        body.append('<details><summary>'+escape(f'{frame["dataset"]} scan {frame["spectrum_index"]} · {frame["condition"]}')+'</summary><div class="qt">'+embed(output/"figures"/frame["png"],"Actual KmzViewer frame")+'</div></details>')
    body.append('<h2>Reproducibility</h2><p>All images are embedded. This HTML uses no network resources or external scripts. CSVs, metadata.json, REPORT.md and indexed experimental mzML copies accompany it locally.</p><details><summary>Benchmark settings</summary><pre>'+escape(json.dumps({"threshold":"intensity > 100", "guards":1, "gap_factor":4, "local_radius":8, "gap_coordinate":"sqrt(m/z)", "guard_sweep":[0,1,2], "gap_sweep":[1.5,2,4,8], "quantitative_use":False},indent=2))+'</pre></details></main></html>')
    (output/"report.html").write_text(''.join(body))


def technical_report(output, metadata, summary):
    """Write an evidence-based local report; scientific results stay ignored."""
    def markdown_table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                         ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
    populations=metadata['populations']
    totals={k:sum(p[k] for p in populations) for k in ('ms1_spectra','profile_points','hard100_points','guard1_points','guard2_points','exactly_100_points')}
    guard_pct=100*(totals['guard1_points']-totals['hard100_points'])/max(1,totals['hard100_points'])
    guard2_pct=100*(totals['guard2_points']-totals['hard100_points'])/max(1,totals['hard100_points'])
    costs=metadata['file_costs']
    raw_count=sum(entry['kind']=='complete WIFF+SCAN' for entry in metadata['inputs'])
    verified_raw={r['dataset'] for r in costs if r['condition']=='reference' and r['verification'].get('arrays_exact')}
    raw_evidence=f"The unchanged production writer was run for {len(verified_raw)} acquisitions selected by low/median/high mean hard-100 point density; every scan value, native ID, level, chromatogram, index and checksum was checked against the current raw reader." if verified_raw else "Raw conversion/file verification was not measured in this run; profile post-conversion analysis cannot qualify that pathway."
    table=markdown_table(['Condition','Median normalized IAD','P90 normalized IAD','Median NRMSE / apex','Threshold-edge MAE (counts)','Deleted regions still connected','Incomplete FWHM / apices'],
                         [[r['condition'],r['median_IAD_ratio'],r['p90_IAD_ratio'],r['median_NRMSE_apex'],r['median_boundary_MAE_counts'],r['remaining_deleted_bridges'],f"{r['incomplete_FWHM']} / {r['major_apices']}"] for r in summary])
    fixtures=markdown_table(['Fixture set','Files used (relative to repository)','Input bytes'],
                            [[entry['dataset'],'; '.join(f['path'] for f in entry['files']),'; '.join(str(f['bytes']) for f in entry['files'])] for entry in metadata['inputs']])
    sizes=[]
    for dataset in sorted({r['dataset'] for r in costs}):
        rows={r['condition']:r for r in costs if r['dataset']==dataset}
        a,b=rows['A'],rows['B']
        reference=rows.get('reference')
        sizes.append([dataset,round(reference['bytes']/1e6,2) if reference else 'see input metadata',round(a['bytes']/1e6,2),round(b['bytes']/1e6,2),
                      round(b['size_increase_over_A_percent'],2),round(a['reduction_from_profile_percent'],2),round(b['reduction_from_profile_percent'],2),
                      f"{a['conversion_seconds']:.2f} / {b['conversion_seconds']:.2f}",f"{a['parse_seconds']:.3f} / {b['parse_seconds']:.3f}"])
    size_table=markdown_table(['Dataset','Profile MB','A / C MB','B / D MB','B increase vs A %','A reduction %','B reduction %','A / B write s','A / B parse s'],sizes)
    # Interpretation belongs to the reviewed run, not a preset favorable outcome.
    narrative=metadata.get('interpretation', 'Quantitative comparisons are generated. Review the aligned visual panels and failure cases before selecting a production candidate.')
    visual_observations=metadata.get('visual_observations', 'No visual interpretation has been recorded for this run. Inspect low-baseline slopes, long connections, singleton measurements, sampling variation and omitted below-cutoff signals in the aligned panels. A metric winner alone is not a production recommendation.')
    sweep_interpretation=metadata.get('sweep_interpretation', 'Review the complete zero/one/two-guard and 1.5/2/4/8 gap-factor sweep. Extra retained points, blank regions and missing half-height crossings must be considered alongside shape error. No full-file two-guard size measurement is implied.')
    recommendation=metadata.get('recommendation', 'No reviewed recommendation is recorded. Production defaults remain unchanged.')
    next_task=metadata.get('next_task', 'Any production inspection feature needs a separate decision, explicit lossy provenance and sampling/provenance qualification. Ordinary unfiltered conversion and viewer defaults remain authoritative.')
    report=f'''# MS1 inspection appearance qualification

{narrative}

## Inputs, discovery and baseline

{fixtures}

The {raw_count} raw sets contain both the WIFF entry and its required `.wiff.scan` payload; present raw sets opened successfully through the current guarded reader. No WIFF2 entry is needed. Retained viewer mzMLs are independent real profile datasets; no raw-to-viewer pairing is assumed. Input SHA-256 identities, relative paths and settings are in `metadata.json`; identities were rechecked after analysis.

Repository and normal local-fixture searches found no retained hard-100/profile pair, no earlier hard-100 qualification report and no old inspection implementation. `Tests/Convert/test_qualification.py`, `scripts/validation/qualify_references.py` and the previous ignored integration reports qualify different converter/calibration contracts. Their results are not reused as hard-100 evidence. `TestData/Regression/profile.mzML` and `centroid.mzML` and generated `.local/pytest`/lifecycle files are synthetic and excluded from scientific benchmarking. The sibling LCMS-Plotting fixture location also contains Test1/Test2 profile runs and Test3-centroid; see discovery metadata for duplicate/exclusion details. No external input was uploaded or newly fabricated.

Actual production: `WiffReader.arrays(scan)` calibrates native stored signal and uses `frame_profile` to provide its existing neighboring zero bins. `writer.spectrum` writes those full framed arrays as compressed 64-bit profile values. There is no intensity filter or centroiding. CLI `--inspect` reports metadata and writes no mzML. {raw_evidence} This establishes current implementation behavior where measured, not an independent vendor calibration oracle.

KmzViewer `SpectrumPanel.show_scan` calls `self.plot.plot(mz, intensity, pen=pen)` for profile spectra. The default pyqtgraph polyline connects neighboring retained points. The reference shows the source/profile line as currently rendered, including its existing zero framing.

## Experimental definitions

- A: retain source MS1 points with **intensity >100**, without centroiding; normal connected profile line. Exactly 100 is excluded consistently. {totals['exactly_100_points']:,} source points equal 100 across the census; B may recover some as boundary guards. This is a reconstructed experimental inspection control, never current or historical production behavior.
- B: start from A's original mask and union one source index immediately before/after each retained region, clipped to spectrum boundaries and deduplicated. Above-threshold indices are unchanged. Existing source zeros are valid guards; no coordinates, intensities or zero measurements are invented. Expansion is nonrecursive. Two guards are an internal sweep variant.
- C: use A's unchanged file. In sqrt(m/z), estimate spacing from the lower quartile of up to eight other intervals on each side, excluding the candidate interval. A supported side needs three intervals and two within 20% of its estimate. Use the larger supported side estimate; break intervals larger than four times it. With insufficient local support, keep the segment. sqrt(m/z) follows the inspected quadratic TOF grid, avoiding mass-dependent raw m/z spacing. It is not qualified for arbitrary analyzers.
- D: B's file with C's connection mask. Disconnected singleton measurements are plotted as dots at their exact observed coordinates/intensities. No vertical sticks or synthetic baseline endpoints are added. Mask breaks do not change mzML data.

All experimental behavior lives in `scripts/benchmarks` and the benchmark driver. The production reader/writer is reused through a local adapter and a restored, process-local metadata annotation. Retained-profile copies use streaming format-aware XML rewriting with original numeric precision and rebuilt indices/checksums. MS2 values and vendor TIC/chromatograms remain unchanged. This does not validate quantitative use of an inspection file; a vendor TIC does not become the sum of filtered measurements.

## Sampling and measurements

The census includes **{totals['ms1_spectra']:,} real MS1 spectra** and **{totals['profile_points']:,} profile points**. Detailed comparisons cover **{len(metadata['selected'])} deterministically selected spectra** across all twelve datasets. Selection combines RT fractions .1/.35/.65/.9, maximum/minimum retained density, moderate (100–300) signal counts, maximum control area distortion, a nearby-apex cluster criterion, strong isolated apices, and empty cutoff spectra where present. Ties use scan index. There were no retained known-problem scan IDs to supplement. Stress scans are deliberately included; these selected medians are not prevalence estimates for every acquired scan.

A retains {totals['hard100_points']:,} MS1 points ({100*totals['hard100_points']/totals['profile_points']:.2f}% of profile); B retains {totals['guard1_points']:,} ({100*totals['guard1_points']/totals['profile_points']:.2f}%), adding {totals['guard1_points']-totals['hard100_points']:,} sub-threshold/source-zero guards. Two guards retain {totals['guard2_points']:,}, {guard2_pct:.2f}% above A. All above-threshold measurements survive all four conditions. Maximum and median absolute m/z deviation and relative intensity deviation are **zero** for retained points. File verification checks every decoded value, not only images; selected apex m/z/intensity and relative apex heights remain unchanged.

The common evaluation grid is the source knots plus source-interval midpoints. Reference and candidate polylines are evaluated linearly solely to compare drawn shapes. Blank intervals remain NaN in trace evaluation. `ink_*` metrics treat blank ink as zero for a visual error convention, not an inferred measurement; coverage, omitted reference area and covered-only error are also exported. Trapezoidal integration on this irregular grid is approximate. Normalized IAD divides by the integrated source profile trace and can greatly exceed one when long false baselines dominate narrow peaks. Global apex-normalized RMSE dilutes local errors; neither metric is a quantitative assay qualification.

{table}

Peak CSVs compare up to twenty separated source local maxima per selected scan. FWHM is reported only when both half-height crossings lie on connected segments; missing crossings stay missing. This prevents an apparently good median over the remaining widths from concealing truncation. Threshold-adjacent peaks particularly expose that failure. Major/above-threshold apex preservation does not imply that sub-threshold peak shape or missing low peaks are recovered.

## Storage and performance

{size_table}

MB is decimal. C uses exactly A's file; D uses B's file. File sizes include unchanged MS2 and metadata, so MS2-rich raw conversions reduce less than the two retained profile runs. Conversion/rewrite and sequential parsing timings are single warm-host observations, with no claim of statistically significant speed differences. Three raw references use direct current production conversion; viewer profiles use post-conversion rewriting, not raw conversion timing. Different source metadata/encodings also limit cross-dataset size comparisons.

The selected median gap-mask cost is {next(r for r in summary if r['condition']=='C')['median_gap_ms']:.3f} ms for C and {next(r for r in summary if r['condition']=='D')['median_gap_ms']:.3f} ms for D. Actual offscreen KmzViewer checks exercise all five plot paths; their frame costs and screenshots are in `viewer-frame-costs.csv`/HTML. Three captures per path include set-data, event processing and grab, not interactive FPS. Independent pyteomics/KmzViewer indexed reads verified {len(metadata.get('viewer_loading',[]))} selected file spectra; `viewer-loading.csv` records individual timings and exact fidelity. The first read per file includes reader/index initialization. The connection mask costs approximately one byte per retained interval. B/D's cached two-array storage grows with the measured guard-point count. Total process RSS was not isolated or measured.

## Visual mechanism, limitations and decision

{visual_observations}

The guard sweep and gap factors 1.5/2/4/8 are fully exported in `parameter-sweep.csv`. {sweep_interpretation}

**Recommendation:** {recommendation}

**Next minimal production task, if approved separately:** {next_task}

## Artifacts and validation

`report.html` embeds aligned full/bridge/boundary views for every selected scan and actual KmzViewer frames; each column has identical axes across all conditions. Limits include segments crossing into a zoom so false high baselines cannot disappear off-scale. Selected vector SVGs and high-resolution PNGs are in `figures/`. The directory also contains census/population CSVs, summary, per-spectrum and peak metrics, parameter sweep, file/frame costs, source-array subsets, indexed experimental files and machine-readable metadata. Results review bundles include the reports, numerical tables, metadata, validation logs and representative figures. Raw inputs, mzML copies and source-array subsets remain excluded.

Focused tests cover exact source-mask boundaries, isolated/contiguous/spectrum-edge regions, nonrecursive two-guard expansion, strict cutoff semantics, deterministic gap detection, dense/gradual/sparse spacing, blank-versus-observed trace handling, incomplete widths, 32-bit precision, MS2/chromatogram preservation, failed-output cleanup, no-clobber, real raw adapter behavior and restoration of its annotation patch. Existing production tests are run unchanged. Recorded task validation: {json.dumps(metadata.get('validation', {'status':'See accompanying validation evidence; no test execution inferred from report generation'}))}.

Reproduce with the qualified repository environment:

```sh
.venv/bin/python scripts/benchmark-ms1-appearance.py --output .local/benchmarks/ms1-appearance-new
```

The destination must be new. Supply `--raw` with a complete WIFF/SCAN set or `--profile` for an independent real profile file; `--fixtures` selects the normal discovery directory. `--skip-file-benchmark` leaves conversion/storage qualification unmeasured. If raw inputs are absent, profile rewriting and viewer analyses remain possible, but raw conversion/calibration/time claims cannot be made. No synthetic test input is substituted as benchmark evidence.
'''
    (output/'REPORT.md').write_text(report)
    return narrative
