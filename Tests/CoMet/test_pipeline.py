from dataclasses import replace
from pathlib import Path
import os

from lxml import etree
import numpy as np
from openpyxl import load_workbook
import pytest

from kmzcomet.alignment import group_features
from kmzcomet.cli import main
from kmzcomet.engine import analyze, file_hash, summarize
from kmzcomet.excel import write_workbook
from kmzcomet.model import Candidate, Feature, Sample, Settings
from kmzcomet.model import Scan
from kmzcomet.detection import mass_peaks
from kmzcomet.reader import iter_scans
from scripts.benchmarks.comet_fixtures import Truth, cohort, write_run


@pytest.fixture(scope="module", params=[False, True], ids=["centroid", "profile"])
def benchmark(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("comet-cohort")
    paths, truth, shifts = cohort(root, profile=request.param)
    return analyze(paths), truth, shifts


def match(result, truth):
    found = [i for i, feature in enumerate(result.features) if feature.polarity == truth.polarity
             and abs(feature.mz-truth.mz)/truth.mz*1e6 < 5 and abs(feature.rt-truth.rt) < .035]
    assert len(found) == 1, (truth.id, found)
    return found[0]


def test_cohort_truth_recovery(benchmark):
    result, truth, shifts = benchmark
    assert len(result.features) == len(truth)
    for expected in truth:
        i = match(result, expected)
        for sample, amplitude in enumerate(expected.heights):
            measured = result.measurements[sample][i]
            if amplitude:
                assert measured.status == "supported", (expected.id, sample, measured)
                assert measured.height == pytest.approx(amplitude, rel=.08, abs=21)
            else:
                assert measured.status in {"weak_signal", "no_signal"}
                assert measured.height < 21
    for sample, shift in zip(result.samples, shifts):
        assert sample.shifts[1] == pytest.approx(shift, abs=.011)


def test_subthreshold_signal_is_measured(benchmark):
    result, truth, _ = benchmark
    for expected in truth:
        if expected.id.startswith("shared"):
            m = result.measurements[2][match(result, expected)]
            assert not m.discovered
            assert m.status == "supported"
            assert 500 < m.height < result.settings.discovery_height


def test_no_flat_or_single_scan_features(benchmark):
    result, _, _ = benchmark
    assert all(abs(f.mz-1100)>1 and abs(f.mz-1200)>1 for f in result.features)


def test_workbook_values_types_and_no_clobber(benchmark, tmp_path):
    result, truth, _ = benchmark
    output = write_workbook(result, tmp_path/"comparison.xlsx")
    original = file_hash(output)
    with pytest.raises(FileExistsError):
        write_workbook(result, output)
    assert file_hash(output) == original
    book = load_workbook(output, data_only=False)
    assert book.sheetnames == ["Heights", "Evidence", "Measurements", "Run"]
    assert book["Heights"].freeze_panes == "E2"
    assert book["Heights"].max_row == len(result.features)+1
    for i, row in enumerate(book["Heights"].iter_rows(min_row=2, values_only=True)):
        assert row[1] == pytest.approx(result.features[i].mz)
        assert list(row[4:]) == pytest.approx([c[i].height for c in result.measurements])
    assert not any(cell.data_type == "f" for ws in book for row in ws for cell in row)
    book.close()


def test_seconds_and_polarity_separation(tmp_path):
    truth = [Truth("pos",200.,1.,(9000.,)), Truth("neg",200.,1.,(12000.,),-1)]
    path = write_run(tmp_path/"seconds.mzML", truth, rt_stop=2., rt_unit="second")
    result = analyze([path])
    assert len(result.features) == 2
    assert {f.polarity for f in result.features} == {-1,1}
    assert all(f.rt == pytest.approx(1., abs=.011) for f in result.features)


def test_duplicate_alias_and_hardlink_rejected(tmp_path):
    path = write_run(tmp_path/"a.mzML", [Truth("p",200.,1.,(8000.,))], rt_stop=2.)
    alias = tmp_path/"alias.mzML"
    os.link(path, alias)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze([path, alias])


@pytest.mark.parametrize("change", ["time_unit", "missing_mode", "negative", "nonfinite", "unordered", "repeated_rt"])
def test_invalid_metadata_and_arrays_fail(tmp_path, change):
    path = write_run(tmp_path/"a.mzML", [Truth("p",200.,1.,(8000.,))], rt_stop=.03)
    tree = etree.parse(str(path))
    ns = {"m":"http://psi.hupo.org/ms/mzml"}
    if change == "time_unit":
        for p in tree.xpath('//m:cvParam[@accession="MS:1000016"]', namespaces=ns):
            p.attrib.pop("unitName")
            p.attrib.pop("unitAccession")
    elif change == "missing_mode":
        for p in tree.xpath('//m:cvParam[@accession="MS:1000127"]', namespaces=ns):
            p.getparent().remove(p)
    elif change == "repeated_rt":
        for p in tree.xpath('//m:cvParam[@accession="MS:1000016"]', namespaces=ns):
            p.set("value", "1")
    else:
        from Tests.Viewer.mzml_factory import encode_float64
        arrays = tree.xpath('//m:spectrum[1]/m:binaryDataArrayList/m:binaryDataArray', namespaces=ns)
        if change == "negative":
            arrays[1].find("m:binary", ns).text = encode_float64(np.array([-1.]))
        elif change == "nonfinite":
            arrays[1].find("m:binary", ns).text = encode_float64(np.array([np.nan]))
        else:
            arrays[0].find("m:binary", ns).text = encode_float64(np.array([200.,100.]))
            arrays[1].find("m:binary", ns).text = encode_float64(np.array([1.,1.]))
    tree.write(str(path))
    with pytest.raises(ValueError):
        list(iter_scans(path))


@pytest.mark.parametrize("kwargs", [{"mz_ppm":float("nan")}, {"min_points":2}, {"discovery_height":0},
                                    {"rt_tolerance":1}, {"alignment":"warp"}, {"min_anchors":3.2}])
def test_settings_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_acquisition_coverage_and_zero_are_distinct(tmp_path):
    truth = [Truth("p",200.,1.,(9000.,0.))]
    first = write_run(tmp_path/"a.mzML", truth, rt_stop=2., noise=0)
    zero = write_run(tmp_path/"zero.mzML", truth, sample=1, rt_stop=2., noise=0)
    partial = write_run(tmp_path/"partial.mzML", truth, rt_start=1.0, rt_stop=2., noise=0)
    outside = write_run(tmp_path/"outside.mzML", truth, rt_start=1.5, rt_stop=2., noise=0)
    gap = write_run(tmp_path/"gap.mzML", truth, rt_stop=2., gaps=[(.95,1.05)], noise=0)
    unknown = write_run(tmp_path/"unknown.mzML", truth, rt_stop=2., scan_windows=False)
    result = analyze([first,zero,partial,outside,gap,unknown], Settings(alignment="none"))
    i = match(result, truth[0])
    assert result.measurements[1][i].status == "no_signal"
    assert result.measurements[1][i].height == 0
    assert result.measurements[2][i].status == "insufficient_coverage"
    assert result.measurements[3][i].status == "not_acquired"
    assert result.measurements[3][i].height is None
    assert result.measurements[4][i].status == "insufficient_coverage"
    assert result.measurements[5][i].status == "insufficient_coverage"


def test_area_does_not_bridge_gap():
    sample = Sample(Path("unused"),"S001","",0)
    feature = Feature("F1",200.,1.,1,[],.8,1.2)
    trace = [(.8,0.,200.,True),(.81,10.,200.,True),(1.1,10.,200.,True),(1.11,0.,200.,True)]
    measured = summarize(feature,sample,0,trace,Settings())
    assert measured.window_area == pytest.approx(.1)
    assert "acquisition_gap" in measured.flags


def test_alignment_does_not_chain_or_merge_same_file():
    samples = [Sample(Path("unused"),f"S{i}","",0) for i in range(3)]
    for i,s in enumerate(samples):
        s.candidates = [Candidate(i,200.,1.+i*.09,9000.-i,1)]
    features = group_features(samples, Settings())
    assert len(features) == 2
    assert all(max(c.rt for c in f.members)-min(c.rt for c in f.members) <= .1 for f in features)
    samples[0].candidates.append(Candidate(0,200.,1.06,5000.,1))
    features = group_features(samples, Settings())
    assert all(len({c.sample for c in f.members}) == len(f.members) for f in features)


def test_unresolved_mass_windows_are_flagged(tmp_path):
    truth = [Truth("a",200.,1.,(10000.,)), Truth("b",200.003,1.,(8000.,))]
    result = analyze([write_run(tmp_path/"a.mzML",truth,rt_stop=2.)])
    assert len(result.features) == 2
    assert all(m.status == "ambiguous" for m in result.measurements[0])


def test_formula_like_filename_is_literal_and_cli_fails_safely(tmp_path):
    path = write_run(tmp_path/"=1+1.mzML", [Truth("p",200.,1.,(8000.,))], rt_stop=2.)
    output = tmp_path/"out.xlsx"
    assert main([str(path),"-o",str(output),"--quiet"]) == 0
    book = load_workbook(output)
    cells = [c for row in book["Run"] for c in row if c.value == path.name]
    assert len(cells)==1 and cells[0].data_type == "s"
    book.close()
    digest = file_hash(output)
    assert main([str(path),"-o",str(output)]) == 2
    assert file_hash(output) == digest
    assert main([str(path),"-o",str(tmp_path/"bad.xlsx"),"--mz-ppm","-1"]) == 2
    assert not (tmp_path/"bad.xlsx").exists()
    assert not list(tmp_path.glob(".kmzcomet-*"))


def test_no_discovery_still_writes_valid_empty_table(tmp_path):
    path = write_run(tmp_path/"a.mzML",[Truth("weak",200.,1.,(100.,))],rt_stop=2.)
    result = analyze([path])
    assert result.features == []
    book = load_workbook(write_workbook(result,tmp_path/"empty.xlsx"))
    assert book["Heights"].max_row == 1
    book.close()


def test_partial_mass_coverage_is_not_a_zero(tmp_path):
    truth = [Truth("p",200.,1.,(9000.,))]
    a = write_run(tmp_path/"a.mzML",truth,rt_stop=2.)
    b = write_run(tmp_path/"b.mzML",truth,rt_stop=2.)
    tree = etree.parse(str(b))
    ns = {"m":"http://psi.hupo.org/ms/mzml"}
    spectrum = tree.xpath('//m:spectrum[100]',namespaces=ns)[0]
    spectrum.xpath('.//m:cvParam[@accession="MS:1000501"]',namespaces=ns)[0].set("value","300")
    tree.write(str(b))
    result = analyze([a,b],Settings(alignment="none"))
    i = match(result,truth[0])
    assert result.measurements[1][i].status == "insufficient_coverage"
    assert "partial_mass_coverage" in result.measurements[1][i].flags


def test_mixed_representations_are_rejected(tmp_path):
    truth = [Truth("p",200.,1.,(9000.,))]
    a = write_run(tmp_path/"a.mzML",truth,rt_stop=2.)
    b = write_run(tmp_path/"b.mzML",truth,rt_stop=2.,profile=True)
    with pytest.raises(ValueError,match="Mixed profile/centroid"):
        analyze([a,b])


def test_publication_failure_removes_temporary_file(benchmark,tmp_path,monkeypatch):
    result,_,_ = benchmark
    def fail(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr(os,"link",fail)
    with pytest.raises(OSError,match="simulated"):
        write_workbook(result,tmp_path/"result.xlsx")
    assert not list(tmp_path.iterdir())


def test_input_changed_during_analysis_is_rejected(tmp_path,monkeypatch):
    import kmzcomet.engine as engine
    path = write_run(tmp_path/"a.mzML",[Truth("p",200.,1.,(9000.,))],rt_stop=2.)
    quantify = engine.quantify
    def mutate(*args):
        values = quantify(*args)
        with path.open("a") as stream:
            stream.write("\n")
        return values
    monkeypatch.setattr(engine,"quantify",mutate)
    with pytest.raises(ValueError,match="input changed"):
        engine.analyze([path])


def test_unsupported_mass_range_has_blank_height(tmp_path):
    truth = [Truth("p",200.,1.,(9000.,))]
    a = write_run(tmp_path/"a.mzML",truth,rt_stop=2.)
    b = write_run(tmp_path/"b.mzML",[Truth("other",500.,1.5,(8000.,))],rt_stop=2.)
    tree = etree.parse(str(b))
    for p in tree.xpath('//*[local-name()="cvParam"][@accession="MS:1000501"]'):
        p.set("value","300")
    tree.write(str(b))
    result = analyze([a,b])
    i = match(result,truth[0])
    assert result.measurements[1][i].status == "not_acquired"
    assert result.measurements[1][i].height is None


def test_profile_shoulders_require_a_resolving_valley():
    # Two ripples share one lobe; a lower, genuinely resolved ion is retained.
    y = np.array([0., 20., 100., 80., 90., 70., 85., 10., 40., 0.])
    scan = Scan(1., 1, 'profile', np.arange(len(y))+200., y, ((50., 1500.),), 'count')
    np.testing.assert_array_equal(mass_peaks(scan, 15.), [2, 8])
    scan.mode = 'centroid'
    np.testing.assert_array_equal(mass_peaks(scan, 15.), [1, 2, 3, 4, 5, 6, 8])


def test_profile_higher_shoulder_replaces_lower_and_plateau_is_stable():
    y = np.array([0., 40., 35., 60., 60., 40., 90., 0.])
    scan = Scan(1., 1, 'profile', np.arange(len(y))+200., y, (), 'count')
    np.testing.assert_array_equal(mass_peaks(scan, 10.), [6])


def test_discovery_context_can_widen_without_widening_measurement(tmp_path):
    truth = [Truth('broad', 200., 1., (10000.,), sigma=.14)]
    path = write_run(tmp_path/'broad.mzML', truth, rt_stop=2., noise=0.)
    narrow = analyze([path], Settings(rt_window=.15))
    wide = analyze([path], Settings(rt_window=.15, discovery_window=.5))
    assert not narrow.features
    assert len(wide.features) == 1
    assert wide.features[0].right-wide.features[0].left == pytest.approx(.3)
    assert wide.measurements[0][0].height == pytest.approx(10000.)


def test_evidence_floor_retains_low_signal_values(tmp_path):
    truth = [Truth('p', 200., 1., (9000., 60.))]
    paths = [write_run(tmp_path/f'{j}.mzML', truth, sample=j, rt_stop=2., noise=0.) for j in range(2)]
    result = analyze(paths, Settings(min_height=100.))
    m = result.measurements[1][0]
    assert m.height == pytest.approx(60.)
    assert m.status == 'weak_signal'
    assert 'below_evidence_height' in m.flags
    assert not m.discovered
    with pytest.raises(ValueError):
        Settings(min_height=-1.)


def test_distant_apex_is_not_support_for_requested_rt():
    sample = Sample(Path('unused'), 'S001', '', 0)
    feature = Feature('F1', 200., 1., 1, [], .7, 1.3)
    times = np.arange(.7, 1.3, .01)
    values = 10000.*np.exp(-.5*((times-1.22)/.025)**2)
    trace = [(float(t), float(y), 200., True) for t,y in zip(times, values)]
    measured = summarize(feature, sample, 0, trace, Settings(rt_window=.3))
    assert measured.height == pytest.approx(10000.)
    assert measured.status == 'weak_signal'
    assert 'apex_outside_rt_tolerance' in measured.flags


def test_wide_discovery_keeps_separated_same_mass_peaks(tmp_path):
    truth = [Truth('early', 200., 1., (9000., 0.)), Truth('late', 200., 1.35, (0., 9000.))]
    paths = [write_run(tmp_path/f'{j}.mzML', truth, sample=j, rt_stop=2., noise=0.) for j in range(2)]
    result = analyze(paths, Settings(discovery_window=.35, alignment='none'))
    assert len(result.features) == 2
    for j in range(2):
        assert result.measurements[j][j].status == 'supported'
        assert result.measurements[j][1-j].height < 1.
