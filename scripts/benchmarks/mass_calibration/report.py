"""Compact offline HTML and static scientific diagnostics; no private uploads."""
import base64
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = dict(constant_da="Constant Da", constant_ppm="Constant ppm", rt_ppm="Linear RT ppm",
              mz_ppm="Linear m/z ppm", rt_mz_ppm="Additive RT + m/z ppm")


def table(rows, columns):
    def format_value(v):
        if v is None: return "—"
        if isinstance(v,float): return f"{v:.6g}"
        return str(v)
    return '<div class="table"><table><thead><tr>'+''.join(f'<th>{escape(label)}</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{escape(f"{row[key]:.7f}" if key in ("expected_mz","observed_mz") and isinstance(row.get(key),float) else format_value(row.get(key)))}</td>' for key,_ in columns)+'</tr>' for row in rows)+'</tbody></table></div>'


def image(path, caption):
    if not path.exists(): return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def reference_diagnostics(destination, references, scans, chosen, contexts, settings, matched):
    fig, axes = plt.subplots(len(references),3,figsize=(13,3*len(references)),layout="constrained",squeeze=False)
    for i,(ref,ss,c,ctx) in enumerate(zip(references,scans,chosen,contexts)):
        eic, spectrum, local = axes[i]
        if not ss or c is None:
            eic.text(.1,.5,"Reference not matched",transform=eic.transAxes); continue
        times = np.array([s.rt for s in ss])
        broad = [float(s.intensity[(s.mz>=ref.expected_mz*(1-settings.broad_ppm*1e-6)) & (s.mz<=ref.expected_mz*(1+settings.broad_ppm*1e-6))].sum()) for s in ss]
        eic.plot(times,broad,label=f"Broad ±{settings.broad_ppm:g} ppm",color="#98a5a9",lw=1)
        eic.plot(ctx["rt"],c["trace"],label="Selected local lobe",color="#127a61",lw=1.5)
        eic.axvline(ref.expected_rt,ls="--",color="#333",lw=.8,label="Expected RT")
        eic.axvline(c["rt"],color="#bc6132",lw=1,label="Observed apex")
        high_rt=[ctx["rt"][j] for j in c["high_indices"]]
        eic.axvspan(min(high_rt),max(high_rt),alpha=.12,color="#127a61")
        eic.set(title=f'{ref.id} · expected {ref.expected_mz:.4f} · '+("matched" if matched else "UNRESOLVED candidate"),xlabel="RT (min)",ylabel="Sum of profile counts")
        eic.legend(fontsize=7,loc="upper right")
        scan = next(s for s in ss if s.id==c["apex_id"])
        error = (scan.mz-ref.expected_mz)/ref.expected_mz*1e6
        spectrum.plot(error,scan.intensity,color="#394c56",lw=.8)
        spectrum.axvline(0,ls="--",color="#333",lw=.8,label="Expected mass")
        spectrum.axvline(c["ppm_error"],color="#127a61",lw=1,label="Median centroid")
        spectrum.set(xlim=(-settings.broad_ppm,settings.broad_ppm),xlabel="Mass error from expected (ppm)",ylabel="Profile counts",title="Apex spectrum · broad search context")
        spectrum.legend(fontsize=7)
        for j in c["high_indices"]:
            s = next(s for s in ss if s.id==ctx["scan_ids"][j])
            d=(s.mz-c["observed_mz"])/ref.expected_mz*1e6
            local.plot(d,s.intensity/max(s.intensity.max(),1),color="#91b6ad",alpha=.5,lw=.6)
        d=(scan.mz-c["observed_mz"])/ref.expected_mz*1e6
        local.plot(d,scan.intensity/scan.intensity.max(),".-",color="#243f4c",lw=1,ms=3,label="Apex sampled profile")
        apex=c["measurements"][c["apex_index"]]
        for field,color,label in (("sampled_mz","#a46035","Sampled maximum"),("centroid_mz","#127a61","Lobe centroid"),("parabola_mz","#6f58a1","3-point parabola")):
            if apex[field] is not None:local.axvline((apex[field]-c["observed_mz"])/ref.expected_mz*1e6,color=color,lw=1,label=label)
        local.set(xlim=(-70,70),xlabel="ppm relative to multi-scan centroid",ylabel="Normalized profile intensity",title=f'Local shape · {len(c["high_indices"])} high scans')
        local.legend(fontsize=7)
    fig.savefig(destination/"reference-diagnostics.png",dpi=140)
    plt.close(fig)


def error_plots(destination, rows):
    fig, axes = plt.subplots(2,2,figsize=(9,6),layout="constrained")
    for ax,(x,y,title) in zip(axes.flat,(("rt","ppm_error","ppm vs observed RT"),("expected_mz","ppm_error","ppm vs expected m/z"),
                                      ("rt","da_error","Da vs observed RT"),("expected_mz","da_error","Da vs expected m/z"))):
        ax.scatter([r[x] for r in rows],[r[y] for r in rows],color="#127a61")
        for r in rows:ax.annotate(r["reference_id"],(r[x],r[y]),xytext=(4,5),textcoords="offset points",fontsize=8)
        ax.set(title=title,xlabel="RT (min)" if x=="rt" else "Expected m/z",ylabel="ppm error" if y=="ppm_error" else "Da error")
        ax.grid(alpha=.2)
        ax.margins(.2)
    fig.savefig(destination/"error-characterization.png",dpi=140)
    plt.close(fig)


def model_plot(destination,result):
    models=result["models"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    positions=np.arange(len(models))
    axes[0].barh(positions-.18,[m["full_metrics"]["rms_ppm"] for m in models],height=.35,label="Full-fit RMS",color="#a8b9b3")
    axes[0].barh(positions+.18,[m.get("loo_metrics",{}).get("rms_ppm",np.nan) for m in models],height=.35,label="Held-out RMS",color="#127a61")
    axes[0].set(yticks=positions,yticklabels=[LABELS[m["model"]] for m in models],xlabel="Residual ppm RMS",title="Prediction vs training fit")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()
    for method,reports in result["sensitivity"]["estimator_models"].items():
        if not isinstance(reports,list): continue
        axes[1].plot(positions,[m.get("loo_metrics",{}).get("rms_ppm",np.nan) for m in reports],"o-",ms=4,label=method)
    axes[1].set(xticks=positions,xticklabels=["Da","ppm","RT","m/z","RT+m/z"],xlabel="Correction model",ylabel="Held-out RMS (ppm)",title="Estimator sensitivity")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=.2)
    fig.savefig(destination/"model-prediction.png",dpi=140)
    plt.close(fig)


def write_report(destination,result):
    content=['<h1>KmzConvert mass-correction pilot</h1>',
             '<p class="sub">One acquisition · four supplied ground-truth references · isolated development harness</p>',
             '<p class="notice"><strong>'+escape(result["status"])+'.</strong> Production conversion and GUI are unchanged. All source acquisition files are opened read-only. Input SHA-256 checks before/after: '+("unchanged" if result["source_unchanged"] else "FAILED")+'.</p>']
    if "failure" in result:
        content += ['<h2>Qualification incomplete</h2><p>'+escape(result["failure"])+'. No successful calibration conclusion is inferred from missing or ambiguous measurements.</p>']
    if all(k in result for k in ("raw_reference_table", "models", "sensitivity", "scan_validation", "ms2")):
        rows=result["raw_reference_table"]
        error_plots(destination,rows)
        model_plot(destination,result)
        errors=np.array([r["ppm_error"] for r in rows])
        models=result["models"]
        best=min([m for m in models if "loo_metrics" in m],key=lambda m:m["loo_metrics"]["rms_ppm"])
        const=next(m for m in models if m["model"]=="constant_ppm")
        da=next(m for m in models if m["model"]=="constant_da")
        regime = ('Constant absolute Da performs poorly in held-out prediction relative to constant ppm; this acquisition is predominantly consistent with a common proportional mass shift, with smaller unresolved drift/shape effects.'
                  if da.get("loo_metrics",{}).get("rms_ppm",0) > 2*const["loo_metrics"]["rms_ppm"]
                  else 'These four points do not establish a unique physical error law. Compare predictive performance, estimator sensitivity and model simplicity below.')
        margin=result["detection"].get("score_margin")
        margin_text=f"{margin:.3f}" if margin is not None else "no runner-up"
        content+=['<h2>Findings and recommendation</h2>',
            f'<p>The multi-scan local centroids have raw errors of <strong>{errors.min():.2f} to {errors.max():.2f} ppm</strong>. '
            f'The robust constant-ppm offset is {const["full_model"]["intercept"]:.3f} ppm. '
            +regime+'</p>',
            f'<p><strong>Numerical pilot winner: {LABELS[best["model"]]}</strong>, with held-out RMS {best["loo_metrics"]["rms_ppm"]:.3f} ppm '
            f'and worst error {best["loo_metrics"]["worst_abs_ppm"]:.3f} ppm for the default centroid definition. '
            f'<strong>Simple baseline: constant ppm</strong>, held-out RMS {const["loo_metrics"]["rms_ppm"]:.3f} ppm, worst {const["loo_metrics"]["worst_abs_ppm"]:.3f} ppm. '
            'The combined three-parameter model is a promising candidate, not a production-qualified physical law: each three-reference training fold is exactly determined and the full fit has only one residual degree of freedom. Four dependent folds provide no reliable confidence interval or independent acquisition validation.</p>',
            '<p>Keep constant ppm as the conservative development baseline and carry the combined model forward only as a challenger. This pilot does not support a universal correction across the full acquisition RT/mass range, across files or for product ions. A reference-centroid definition must be settled independently before interpreting the last few ppm as instrument error.</p>',
            '<h2>Raw references</h2>',table(rows,[("reference_id","ID"),("expected_mz","Expected m/z"),("expected_rt","Expected RT / min"),("rt","Observed RT / min"),
               ("observed_mz","Observed m/z"),("da_error","Da error"),("ppm_error","ppm error"),("intensity","Local EIC apex counts"),("method","Estimator")]),
            '<p>Observed RT is the maximum local-lobe EIC scan; mass is the median of independently computed centroids in contiguous scans at ≥50% of that apex. EIC intensity is the sum of local profile samples, not a single peak height.</p>',
            '<h2>Reference assignment evidence</h2><p>Search: RT ±'+str(result["settings"]["rt_radius"])+
            ' min; m/z ±'+str(result["settings"]["broad_ppm"])+
            ' ppm. Local spectral maxima propose mass tracks. The refined trace follows a bounded profile lobe; RT proximity, chromatographic contrast, repeated scan support and mass stability contribute to its score. A coarse common ppm-or-Da regime checks all four together before fitting any correction model. No point is dropped. Small valleys above 20% of peak height are crossed to avoid splitting noisy peak tops; lobe endpoints occur at deep minima or 1% height within 100 ppm. Shoulder-window maxima are excluded as separate ions.</p>',
            f'<p>Complete assignments considered: {result["detection"]["assignments_considered"]:,}; top-vs-runner-up score margin {margin_text}. '
            'The scoring weights were refined during development on this file, so detection generalization is unqualified. RT/mass consistency supports these assignments, but no compound identities, isotope patterns or exact fragments were supplied to establish independent chemical identity.</p>',
            image(destination/"reference-diagnostics.png","Each row: broad and local EIC; apex broad spectrum; high-scan local profile shapes. Normalized traces are diagnostic copies only. Raw intensities are unchanged."),
            '<h2>Reference mass estimators</h2>',table(result["estimates"],[("reference_id","ID"),("method","Method"),("observed_mz","Observed m/z"),("ppm_error","Raw ppm"),
                ("scan_sd_ppm","High-scan SD / ppm"),("scan_mad_ppm","Robust SD (1.4826 MAD) / ppm"),("successful_scans","Scans")]),
            '<p><strong>Default: multi-scan lobe centroid (median).</strong> It is simple, robust to individual scans and more stable here than sampled maxima. SD/MAD describe the underlying per-scan estimates; they are not confidence intervals for the median. Apex maximum/centroid/parabola use one scan but their SD is measured by repeating that estimator over the same high scans. Parabolic fitting uses only three actual, nonuniform profile coordinates; failed/convex/edge fits are explicit failures.</p>',
            '<p>The profiles are asymmetric: the sampled top and three-point vertex lie below the full-lobe centroid. Their several-ppm definition difference exceeds some model residuals and is a potential systematic bias. Synthetic symmetric-peak recovery cannot resolve the correct mass definition for these real asymmetric profiles. No elaborate fit is justified by this evidence.</p>',
            '<h2>Error characterization</h2>',image(destination/"error-characterization.png","Raw ppm and absolute Da errors against observed RT and expected m/z; four points are descriptive, not a validated error surface.")]
        coeluting=[]
        for i,r in enumerate(rows):
            for s in rows[i+1:]:
                if abs(r["expected_rt"]-s["expected_rt"])<1e-9:
                    differences=[]
                    for method in ("apex_maximum","apex_centroid","multi_scan_centroid","apex_parabola"):
                        e=[e for e in result["estimates"] if e["method"]==method and e["reference_id"] in (r["reference_id"],s["reference_id"]) and "ppm_error" in e]
                        if len(e)==2:differences.append(f'{method}: {e[1]["ppm_error"]-e[0]["ppm_error"]:+.3f} ppm')
                    coeluting.append(f'<p><strong>Same-time pair {r["expected_mz"]:.4f} / {s["expected_mz"]:.4f}:</strong> both observed at {r["rt"]:.5f} / {s["rt"]:.5f} min. '
                         f'Default-centroid ppm difference is {s["ppm_error"]-r["ppm_error"]:+.3f} ppm. At effectively the same acquisition time, this difference cannot be attributed to temporal drift; it indicates a mass-dependent residual or a compound/profile-estimator-dependent bias. '
                         'Estimator differences (second minus first): '+escape('; '.join(differences))+'. This estimator dependence limits a firm physical m/z-axis conclusion.</p>')
        content+=coeluting
        comparison=[]
        for m in models:
            comparison.append(dict(model=LABELS[m["model"]],parameters=m["parameters"],full_rms=m.get("full_metrics",{}).get("rms_ppm"),
                **m.get("loo_metrics",{})))
        content+=['<h2>Model prediction benchmark</h2>',
            '<p>Constant Da/ppm fits use the median; linear models use unweighted centered/scaled least squares. No silent outlier rejection. A failed/rank-deficient fold invalidates aggregate CV. Model ppm error is evaluated at true m/z; the correction is the exact inverse of observed = true × (1 + e(RT,true)/10⁶). Constant Da uses true = observed − offset. Linear m/z dependence yields a quadratic forward equation, inverted on its positive increasing branch.</p>',
            table(comparison,[("model","Model"),("parameters","Parameters"),("full_rms","FULL-FIT RMS ppm"),("mean_abs_ppm","LOO mean |ppm|"),("median_abs_ppm","LOO median |ppm|"),
                ("rms_ppm","LOO RMS ppm"),("worst_abs_ppm","LOO worst |ppm|"),("bias_ppm","LOO signed bias ppm")]),
            image(destination/"model-prediction.png","Training fit and held-out prediction are separate. Estimator sensitivity shows whether the preferred model changes with profile-center definition."),
            '<h3>Before → after: full fit and held-out errors</h3>']
        corrected=[]
        for m in models:
            for r,full,fold in zip(rows,m.get("full_residuals_ppm",[]),m["folds"]):
                corrected.append(dict(id=r["reference_id"],model=LABELS[m["model"]],raw=r["ppm_error"],full=full,loo=fold.get("residual_ppm")))
        content+=[table(corrected,[("id","ID"),("model","Model"),("raw","Raw ppm"),("full","After FULL FIT ppm"),("loo","After HELD-OUT ppm")]),
            '<h2>Omission and sensitivity</h2><p>Each omitted reference remains an explicit held-out prediction. The three retained references are refitted; no inconsistent point is silently rejected. The omission table exposes coefficient/held-out influence and lays the foundation for future robust detection. At four points, robust multi-parameter regression cannot credibly classify a chemical mismatch as an outlier.</p>']
        omission=[dict(model=LABELS[o["model"]],omitted=o["omitted"],held=o.get("held_out_residual_ppm"),
                       intercept=o.get("coefficients",{}).get("intercept"), rt_slope=o.get("coefficients",{}).get("rt_slope"),mz_slope=o.get("coefficients",{}).get("mz_slope")) for o in result["sensitivity"]["omissions"]]
        content+=[table(omission,[("model","Model"),("omitted","Omitted"),("held","Held-out ppm"),("intercept","Intercept at fold center"),("rt_slope","ppm/min"),("mz_slope","ppm/(m/z)")]),
            '<p>Centered intercepts have fold-specific origins in omissions.csv/JSON; they are not directly comparable global intercepts. Constant-Da intercepts are in Da. Linear coefficients and predictions, rather than intercept alone, measure influence.</p>']
        sensitivity_table=[]
        for m in models:
            scenarios=[r for r in result["sensitivity"]["rt_perturbations"] if r["model"]==m["model"]]
            rms=[s["loo_metrics"]["rms_ppm"] for s in scenarios if s["loo_metrics"]]
            sensitivity_table.append(dict(model=LABELS[m["model"]],min_rms=min(rms),max_rms=max(rms),
                max_change=max(s["max_correction_change_at_original_coordinates_ppm"] for s in scenarios)))
        content+=['<h3>Approximate RT uncertainty</h3><p>Refit with supplied approximate RTs and all 16 independent ±0.03 min and ±0.05 min perturbations of the selected apices. The largest changes are evaluated back at original measured coordinates. These are deterministic sensitivity scenarios, not new observations or probabilistic uncertainty estimates.</p>',
            table(sensitivity_table,[("model","Model"),("min_rms","Minimum LOO RMS ppm"),("max_rms","Maximum LOO RMS ppm"),("max_change","Max full-fit correction change ppm")]),
            '<h3>High-scan selection and estimator dependence</h3>']
        half_table=[]
        for scenario in result["sensitivity"]["high_scan_fraction"]:
            for m in scenario["models"]:
                half_table.append(dict(fraction=scenario["fraction"],model=LABELS[m["model"]],rms=m.get("loo_metrics",{}).get("rms_ppm")))
        content+=[table(half_table,[("fraction","Apex intensity fraction"),("model","Model"),("rms","LOO RMS ppm")]),
            '<p>Full four-estimator model results, including all failed fits, are retained in sensitivity.json. Small mass-estimator differences can change sub-ppm rankings; none is selected by its training residual. Sparse-profile estimation has not been attempted: every measurement uses the full decoded profile.</p>',
            '<h3>Moving the actual apex by one measured scan</h3>']
        neighbors=[]
        for method in result["sensitivity"]["estimator_models"]:
            scenarios=[s for s in result["sensitivity"].get("apex_neighbors",[]) if s["method"]==method and "models" in s]
            for name in ("constant_ppm","rt_mz_ppm"):
                scores=[m["loo_metrics"]["rms_ppm"] for s in scenarios for m in s["models"] if m["model"]==name and "loo_metrics" in m]
                if scores:
                    neighbors.append(dict(method=method,model=LABELS[name],min_rms=min(scores),max_rms=max(scores),
                                          max_mass_change=max(abs(s["raw_ppm_change"]) for s in scenarios)))
        content+=[table(neighbors,[("method","Estimator"),("model","Model"),("min_rms","Min LOO RMS ppm"),("max_rms","Max LOO RMS ppm"),("max_mass_change","Max raw mass change ppm")]),
            '<p>One reference at a time uses the actual preceding/following high scan for the apex estimators, retaining its measured RT. The multi-scan mass stays fixed while its RT anchor changes. This is more representative of apex picking than independent ±0.05-min label stresses. The latter span many measured scan intervals here and should not be mistaken for instrument RT precision.</p>',
            '<h2>Complete scan correction and mzML inspection artifacts</h2>',
            table(result["scan_validation"],[("model","Model"),("ms1_scans","MS1 scans"),("profile_samples","Profile samples"),("minimum_spacing_ratio","Min corrected/raw spacing"),
                 ("maximum_spacing_ratio","Max corrected/raw spacing"),("correction_min_ppm","Min correction ppm"),("correction_max_ppm","Max correction ppm")]),
            '<p>Every complete MS1 axis was checked for strict order and unchanged sample count. The same intensity array and unchanged scan RT are returned; peak ordinates and sample adjacency are untouched. Mass-dependent models change local spacing by the ratios above: this is an x-axis transformation, not intensity processing. A changed spacing also changes a numerical mass-domain peak area/Jacobian; quantitative suitability is not established. Entire-axis extrapolation is mechanically well behaved but scientifically unvalidated outside the four-reference domain.</p>']
        for output in result["outputs"]:
            content.append('<p><a href="'+escape(output["name"])+ '">'+escape(output["name"])+ '</a> · '+f'{output["bytes"]/1e6:.2f} MB'+
                ' · independent XML readback of every spectrum/TIC, exact MS1 corrected arrays, exact MS2/intensities/RT/nominal metadata, indexed offsets and checksum passed.</p>')
        content+=['<h2>MS1 versus MS/MS</h2><p>'+escape(result["ms2"]["evidence"])+ '</p><p>'+escape(result["ms2"]["conclusion"])+
                  f' This acquisition contains {result["acquisition"]["ms2"]} MS2 spectra; their RTs, precursor metadata and strongest measured profile samples are in ms2_inventory.json. A nominal isolation target is acquisition metadata, not fragment ground truth.</p>',
            '<h2>Next qualification experiment</h2><p>Use at least 12–20 independently identifiable references spanning the full useful m/z range and early/middle/late RTs, including several coeluting low/high-mass pairs. Break the RT/mass correlation deliberately. Acquire replicate injections and multiple days/calibration states, including genuinely large offsets. Reserve entire reference compounds, RT/mass regions and complete acquisitions as held-out tests. Include unspiked controls and known standards/isotopes/fragments to validate assignment and the profile-center convention. Compare constant ppm with RT, mass and additive challengers under preregistered error/ambiguity/extrapolation limits. Qualify MS2 separately with exact product-ion ground truth. Only then define production domain gates and profile/sparse integration.</p>']
    elif "reference-diagnostics.png" in [p.name for p in destination.iterdir()]:
        content.append(image(destination/"reference-diagnostics.png","Unresolved reference candidates; no successful calibration inferred."))
    content+=['<h2>Pipeline and future integration</h2><p>decode_tokens returns TOF coordinates and counts. WiffReader.arrays first obtains masses through calibrate(x, cal_a, cal_b, tdc), then frame_profile adds neighboring zero bins on that same TOF grid. MS1 and MS2 share these functions, with separately stored per-scan coefficients. writer.spectrum obtains full arrays before optional MS1 source_sampling_check/subset filtering. Future insertion: after native calibration/framing and before sparse filtering; estimation always reads full profiles. Nominal precursor/isolation targets and scan-window bounds must remain separate from measured analyzer arrays.</p>',
        '<h2>Reproduction and supporting files</h2><p>Run the isolated driver with the retained settings.json into a new ignored .local directory. CSV files preserve reference/scan/estimator measurements, candidate alternatives, model metrics, held-out predictions, omissions, RT sensitivity and complete-axis checks. models.json contains all centered coefficients, origins, rank/condition information and residuals. results.json contains input identities, acquisition census, settings/runtime, model and sensitivity provenance, output validations and any failures. Original spectra are not bundled.</p>',
        '<p>'+''.join('<a href="'+escape(p.name)+'">'+escape(p.name)+'</a> · ' for p in sorted(destination.iterdir()) if p.suffix in (".csv",".json"))+ '</p>',
        '<p class="sub">No install, production mode, source mutation, upload or chemical-identification claim is implied. Qualification accuracy is conditional on these four supplied references and the specified mass estimator.</p>']
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KmzConvert mass-correction pilot</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#23343b;background:#fafcfb;max-width:1200px;margin:36px auto;padding:0 24px}h1{font-size:32px}h2{margin-top:32px;border-top:1px solid #d5dfdb;padding-top:18px}h3{font-size:18px}.sub,figcaption{color:#53676d;font-size:13px}.notice{background:#e5f0ec;border-left:4px solid #127a61;padding:14px}.table{overflow-x:auto}table{border-collapse:collapse;font-size:12px;width:100%;margin:12px 0}th,td{padding:7px 8px;border-bottom:1px solid #dce5e1;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f0}tr:nth-child(even){background:#f1f6f3}img{width:100%;height:auto}figure{margin:18px 0}a{color:#127a61}code{background:#eaf0ec;padding:2px 4px}@media print{body{font-size:11px}h2{break-after:avoid}figure{break-inside:avoid}}</style></head><body>'''+''.join(content)+'</body></html>'
    (destination/"report.html").write_text(html)
