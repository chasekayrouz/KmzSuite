"""Synthetic numerical qualification; never substitute these for real evidence."""
from dataclasses import replace

import numpy as np
import pytest

from scripts.benchmarks.mass_calibration.models import (
    MODEL_NAMES, Model, Observation, Reference, benchmark, correct_scan, fit, residual,
)
from scripts.benchmarks.mass_calibration.detection import (
    ProfileScan, SearchSettings, candidates, consensus, estimate, local_mass,
)


def observations(model):
    refs = [Reference("r1", 250., 3.), Reference("r2", 350., 3.),
            Reference("r3", 200., 5.), Reference("r4", 300., 4.)]
    return [Observation(r, float(model.forward(r.expected_mz, r.expected_rt)), r.expected_rt, "synthetic") for r in refs]


@pytest.mark.parametrize("model", [Model("constant_da", .15), Model("constant_ppm", 650.),
    Model("rt_ppm", 450., 70.), Model("mz_ppm", 150., mz_slope=1.8),
    Model("rt_mz_ppm", 120., 45., 1.4)])
def test_known_forward_errors_recovered_in_full_fit_and_every_held_out_fold(model):
    rows = observations(model)
    fitted = fit(model.name, rows)
    assert max(abs(residual(fitted, r)) for r in rows) < 1e-8
    result = next(r for r in benchmark(rows) if r["model"] == model.name)
    assert result["loo_metrics"]["worst_abs_ppm"] < 1e-8
    mz = np.linspace(50, 1500, 1001)
    for rt in (0., 3.5, 12.):
        np.testing.assert_allclose(fitted.correct(model.forward(mz, rt), rt), mz, rtol=2e-14)


@pytest.mark.parametrize("slope", [0., 1e-9, -.4, 2.])
def test_exact_mass_dependent_inverse_and_spacing(slope):
    model = Model("mz_ppm", 700., mz_slope=slope)
    raw = model.forward(np.linspace(40, 1200, 100), 3.)
    y = np.arange(len(raw), dtype=float)
    source_copy = raw.copy()
    corrected, signal, checks = correct_scan(model, raw, y, 3.)
    assert signal is y
    np.testing.assert_array_equal(raw, source_copy)
    np.testing.assert_allclose(corrected, np.linspace(40, 1200, 100), atol=1e-11)
    assert checks["sample_count_unchanged"] and checks["strictly_increasing"]
    assert checks["minimum_spacing_ratio"] > 0


def test_degenerate_designs_and_invalid_inverse_fail_explicitly():
    rows = observations(Model("constant_ppm", 500.))
    same_rt = [replace(r, rt=3.) for r in rows]
    report = next(r for r in benchmark(same_rt) if r["model"] == "rt_mz_ppm")
    assert "full_failure" in report and "loo_failure" in report
    assert "loo_metrics" not in report
    with pytest.raises(ValueError, match="duplicate"):
        fit("constant_ppm", [rows[0], rows[0]])
    with pytest.raises(ValueError, match="inverse"):
        Model("mz_ppm", -2e6).correct([100.], 2.)
    with pytest.raises(ValueError, match="increasing"):
        correct_scan(Model("constant_ppm", 1.), [100., 99.], [2., 3.], 2.)
    with pytest.raises(ValueError, match="invalid masses"):
        Model("constant_da", 200.).correct([100.], 2.)


def test_no_silent_outlier_rejection_and_simplicity_visible():
    rows = observations(Model("constant_ppm", 650.))
    rows[-1] = replace(rows[-1], observed_mz=rows[-1].observed_mz+.1)
    reports = benchmark(rows)
    assert all(len(r["folds"]) == 4 for r in reports)
    assert all(len(f["training_ids"]) == 3 for r in reports for f in r["folds"])
    combined = reports[-1]
    assert combined["loo_metrics"]["rms_ppm"] > 0
    assert all(f["residual_degrees_of_freedom"] == 0 for f in combined["folds"])


def test_centroid_and_local_quadratic_on_known_profile():
    cfg = SearchSettings()
    center = 300.0037
    m = np.linspace(299.94, 300.06, 121)
    y = 1e5*np.exp(-.5*((m-center)/.004)**2)
    result = local_mass(m, y, center, cfg)
    assert abs(result["centroid_mz"]-center) < 3e-5
    assert abs(result["parabola_mz"]-center) < 1e-5
    assert not result["truncated"]
    assert result["points"] > 5


def synthetic_scans(reference, ppm=700., interference=False):
    scans = []
    center = reference.expected_mz*(1+ppm*1e-6)
    m = np.linspace(reference.expected_mz*.9988, reference.expected_mz*1.0012, 1201)
    for i, rt in enumerate(np.linspace(reference.expected_rt-.45, reference.expected_rt+.45, 61)):
        height = 1e5*np.exp(-.5*((rt-reference.expected_rt)/.055)**2)
        y = height*np.exp(-.5*((m-center)/(reference.expected_mz*6e-6))**2)
        if interference:
            y += 5e5*np.exp(-.5*((rt-reference.expected_rt-.18)/.055)**2)*np.exp(-.5*((m-reference.expected_mz*.9992)/(reference.expected_mz*6e-6))**2)
        scans.append(ProfileScan(f"s{i}", float(rt), m, y))
    return scans


@pytest.mark.parametrize("ppm",[-700.,700.])
def test_broad_search_uses_four_reference_consensus_against_stronger_interference(ppm):
    refs = [r.reference for r in observations(Model("constant_ppm", 700.))]
    cfg = SearchSettings()
    lists, contexts = [], []
    for i, ref in enumerate(refs):
        c, context = candidates(ref, synthetic_scans(ref,ppm=ppm,interference=i==0), cfg)
        lists.append(c)
        contexts.append(context)
    selected, evidence = consensus(lists, cfg)
    assert selected is not None, evidence
    assert all(abs(c["ppm_error"]-ppm) < .1 for c in selected)
    estimates, rows = estimate(refs[0], selected[0], contexts[0])
    assert len(estimates) == 4 and len(rows) >= 3
    assert max(abs(r["ppm_error"]-ppm) for r in estimates) < 3


def test_missing_and_competing_assignments_are_not_hidden():
    cfg = SearchSettings()
    assert consensus([[], [], [], []], cfg)[0] is None
    c = dict(flags=[], score=3., ppm_error=700., da_error=.2, observed_mz=300., candidate_id="a")
    other = dict(c, candidate_id="b")
    selected, evidence = consensus([[c, other], [c], [c], [c]], cfg)
    assert selected is None and "Ambiguous" in evidence["failure"]


def test_parabola_failure_is_reported_not_filled_with_sample_maximum():
    r = local_mass(np.array([99.99, 100., 100.01]), np.array([0., 100., 0.]), 100., SearchSettings())
    assert r["parabola_mz"] is None


def test_experimental_mzml_preserves_ms2_metadata_tic_and_normal_writer(tmp_path):
    from Tests.Convert.test_writer_cli import synthetic
    from kmzconvert_core import writer
    from kmzconvert_core.reference import NS
    from lxml import etree
    from scripts.benchmarks.mass_calibration.mzml import write_experimental, verify_experimental
    reader=synthetic.__wrapped__(tmp_path)
    normal=writer.write_mzml(reader,tmp_path/"before.mzML").read_bytes()
    old_metadata,old_spectrum=writer.metadata,writer.spectrum
    model=Model("constant_ppm",-600.)
    path=write_experimental(reader,tmp_path/"qualification.mzML",model)
    checks=verify_experimental(reader,path,model)
    assert checks["spectra_by_level"]=={1:1,2:1}
    assert checks["index"]["checksum_valid"]
    root=etree.parse(str(path))
    assert len(root.findall('.//'+NS+'dataProcessing'))==2
    assert root.find('.//'+NS+'spectrum').get("dataProcessingRef")=="qualification-ms1-mass-correction"
    assert writer.metadata is old_metadata and writer.spectrum is old_spectrum
    assert writer.write_mzml(reader,tmp_path/"after.mzML").read_bytes()==normal
    with pytest.raises(FileExistsError):write_experimental(reader,path,model)
    assert writer.metadata is old_metadata and writer.spectrum is old_spectrum


def test_full_synthetic_qualification_driver_and_report(tmp_path,monkeypatch):
    """Exercise serialization, sensitivity, complete arrays and HTML together."""
    import json
    from types import SimpleNamespace
    from kmzconvert_core.reader import Experiment,Scan
    from scripts.benchmarks.mass_calibration import qualification
    refs=[r.reference for r in observations(Model("constant_ppm",700.))]
    entry=tmp_path/"synthetic.wiff"
    entry.write_bytes(b"Synthetic test only, not WIFF")
    entry.with_suffix(".wiff.scan").write_bytes(b"Synthetic test only, not SCAN")
    settings=tmp_path/"references.json"
    settings.write_text(json.dumps({"references":[r.__dict__ for r in refs]}))
    grids=[np.linspace(m*.9988,m*1.0012,301) for m in sorted({r.expected_mz for r in refs})]
    mass=np.concatenate(grids)
    scans=[];signals={}
    for i,rt in enumerate(np.linspace(2.5,5.5,201)):
        y=np.zeros_like(mass)
        for ref in refs:
            y+=1e5*np.exp(-.5*((rt-ref.expected_rt)/.055)**2)*np.exp(-.5*((mass-ref.expected_mz*1.0007)/(ref.expected_mz*6e-6))**2)
        scan=Scan(i,i+1,1,float(rt*60),float(y.sum()),float(y.max()),0,0,.1,.2,None,None)
        scans.append(scan);signals[scan.native_id]=y
    class SyntheticReader(SimpleNamespace):
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def arrays(self,scan):return mass,signals[scan.native_id]
        def tic_arrays(self):return np.array([s.rt_seconds/60 for s in scans]),np.array([s.tic for s in scans]),np.ones(len(scans),dtype=int)
    reader=SyntheticReader(entry=entry,wiff_path=entry,scan_path=entry.with_suffix('.wiff.scan'),
        scans=scans,experiments=[Experiment(1,"positive",100.,1600.,200.,None,{})],
        acquisition_software="SCIEX OS 2.2",instrument={"Mass Spectrometer":"X500 QTOF"})
    monkeypatch.setattr(qualification,"ROOT",tmp_path)
    monkeypatch.setattr(qualification,"WiffReader",lambda _:reader)
    out=tmp_path/".local"/"results"
    result=qualification.run(entry,settings,out,experimental=True)
    assert "failure" not in result,result.get("failure")
    assert result["source_unchanged"]
    assert len(result["models"])==5 and len(result["outputs"])>=1
    assert max(abs(r["ppm_error"]-700) for r in result["raw_reference_table"])<.1
    assert all(r["loo_metrics"]["worst_abs_ppm"]<1e-7 for r in result["models"] if r["model"]=="constant_ppm")
    html=(out/"report.html").read_text()
    assert html.count("data:image/png;base64,")==3
    assert "HELD-OUT" in html and "Next qualification" in html
    assert (out/"held_out_predictions.csv").is_file()
    with pytest.raises(ValueError,match="new directory"):
        qualification.run(entry,settings,out)


def test_missing_input_report_never_fabricates_measurements(tmp_path,monkeypatch):
    import json
    from scripts.benchmarks.mass_calibration import qualification
    refs=[r.reference for r in observations(Model("constant_ppm",700.))]
    settings=tmp_path/"refs.json"
    settings.write_text(json.dumps({"references":[r.__dict__ for r in refs]}))
    entry=tmp_path/"absent.wiff"
    monkeypatch.setattr(qualification,"ROOT",tmp_path)
    out=tmp_path/".local"/"missing"
    result=qualification.run(entry,settings,out)
    assert result["status"]=="qualification_failed" and "failure" in result
    assert "raw_reference_table" not in result and "models" not in result
    assert "Qualification incomplete" in (out/"report.html").read_text()
