"""Focused beta behavior using temporary synthetic profiles with known shifts."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kmzconvert.cli import main
from kmzconvert_core.mass_correction import (
    Reference, SearchSettings, CalibrationObservation, MassCorrection, parse_references,
    estimate_mass_correction, fit_constant_ppm, select_candidates, BEAM_WIDTH, MAX_CANDIDATES,
)
from kmzconvert_core.reader import Experiment, Scan
from kmzconvert_core.reference import NS, iter_records
from kmzconvert_core.writer import write_mzml, metadata
from kmzcore.mass_correction import PAYLOAD, WARNING, valid_method_params
from kmzcore.sparse_profile import FAMILY, FORMAT, Provenance, gap_connections, read_provenance
from scripts.benchmarks.ms1_algorithms import event_rescue_indices
from kmzviewer.mzml_index import index_ms1
from scripts.validation.check_mzml import check_index, check


REFS = [(200., 3.), (250., 3.), (300., 5.), (350., 4.)]


class SyntheticReader(SimpleNamespace):
    def close(self): self.closed = True
    def arrays(self, scan): return self.mass, self.signals[scan.native_id]
    def tic_arrays(self):
        return np.array([s.rt_seconds/60 for s in self.scans]), np.array([s.tic for s in self.scans]), np.array([self.experiments[s.experiment-1].ms_level for s in self.scans])


def reader(tmp_path, ppm=300., name="synthetic", missing=False, ambiguous=False):
    entry = tmp_path/(name+".wiff")
    entry.write_bytes(b"Synthetic fixture only")
    sidecar = entry.with_suffix(".wiff.scan")
    sidecar.write_bytes(b"Synthetic fixture only")
    m = np.concatenate([np.linspace(m*.9988, m*1.0012, 1201) for m, _ in REFS])
    scans, signals = [], {}
    for i, rt in enumerate(np.linspace(2.5, 5.5, 201)):
        y = np.zeros(len(m))
        for j,(mass, time) in enumerate(REFS):
            if missing and j==0: continue
            shift = ppm[j] if isinstance(ppm, list) else ppm
            center = mass*(1+shift*1e-6)
            amplitude = 1e5*np.exp(-.5*((rt-time)/.055)**2)
            y += amplitude*np.exp(-.5*((m-center)/(mass*6e-6))**2)
            if ambiguous and j==0:
                y += amplitude*np.exp(-.5*((m-mass*(1+(ppm-400)*1e-6))/(mass*6e-6))**2)
        y = np.rint(y)
        scan = Scan(i, i+1, 1, float(rt*60), float(y.sum()), float(y.max()), 0, 0, .1, .2, None, None)
        scans.append(scan); signals[scan.native_id] = y
    scan = Scan(201,202,2,5.6*60,500.,500.,0,0,.1,.2,250.,1000.)
    y = np.zeros(len(m)); y[len(y)//2] = 500.
    scans.append(scan); signals[scan.native_id] = y
    return SyntheticReader(entry=entry,wiff_path=entry,scan_path=sidecar,scans=scans,signals=signals,mass=m,
        experiments=[Experiment(1,"positive",100.,1600.,200.,None,{}),Experiment(2,"positive",50.,1000.,50.,30.,{})],
        sparse_profile_sampling_family=FAMILY,acquisition_software="SCIEX OS 2.2",instrument={"Mass Spectrometer":"X500 QTOF"},
        calibration_resolutions=[],nominal_window_overhangs={},index=[],tdc=.1,checkpoint_count=0,closed=False)


def references(count=4):
    return parse_references([f"{m};{rt}" for m,rt in REFS[:count]])


@pytest.mark.parametrize("entries", [[],[""],["200"],["200;3;4"],["nan;3"],["200;inf"],["-1;3"],["0;3"],["200;-1"],["200;"],["200;3","2e2;3.0"]])
def test_reference_validation(entries):
    with pytest.raises(ValueError): parse_references(entries)


@pytest.mark.parametrize("ppm",[-700.,-300.,300.,700.])
@pytest.mark.parametrize("count",[1,4])
def test_estimation_and_exact_recovery_positive_negative_offsets(tmp_path,ppm,count):
    raw = reader(tmp_path,ppm)
    messages=[]
    correction = estimate_mass_correction(raw,references(count),messages.append)
    assert abs(correction.offset_ppm-ppm)<1e-7
    for observation in correction.observations:
        assert abs(correction.calibration_residual(observation))<1e-7
        assert observation.high_scans>=3 and observation.centroid_sd_ppm<1e-7
    assert not any("NOT MET" in text for text in messages)
    if count==1:
        assert any("zero by construction" in text and "unverified" in text for text in messages)
        assert any("Cross-reference consistency unavailable" in text for text in messages)
    mz = raw.mass.copy()
    corrected = correction.correct(mz)
    np.testing.assert_array_equal(mz,raw.mass)
    assert len(corrected)==len(mz) and np.all(np.diff(corrected)>0)
    np.testing.assert_allclose(corrected*(1+ppm*1e-6),mz,rtol=1e-14)


@pytest.mark.parametrize("case",["missing","ambiguous"])
def test_missing_or_ambiguous_reference_fails_before_writing(tmp_path,monkeypatch,capsys,case):
    raw = reader(tmp_path,**{case:True})
    monkeypatch.setattr('kmzconvert.cli.WiffReader',lambda _:raw)
    path=tmp_path/"failed.mzML"
    assert main([str(raw.entry),'--mass-reference','200;3','-o',str(path)])==1
    assert not path.exists() and not list(tmp_path.glob('.kmzconvert-*.tmp'))
    assert raw.closed
    stderr=capsys.readouterr().err
    assert '200;3' in stderr and ('ambiguous' if case=='ambiguous' else 'peak') in stderr


def test_bounded_candidate_selection_for_many_references_and_precise_ambiguity():
    refs=[Reference(f'r{i}',200+i,3.) for i in range(30)]
    lists=[]
    for ref in refs:
        lists.append([dict(candidate_id=f'{ref.id}-{j}',flags=[],score=5-j,ppm_error=300.+j,
                           da_error=ref.expected_mz*(300+j)*1e-6,observed_mz=ref.expected_mz*1.0003) for j in range(12)])
    selected,evidence=select_candidates(refs,lists,[{}]*len(refs),SearchSettings())
    assert len(selected)==30 and all(c['score']==5 for c in selected)
    assert evidence['assignments_considered']<=30*BEAM_WIDTH*MAX_CANDIDATES
    ambiguous=deepcopy(lists[0][:2]);ambiguous[1].update(score=5,ppm_error=300,da_error=ambiguous[0]['da_error'])
    with pytest.raises(ValueError,match='200;3.*ambiguous'):
        select_candidates(refs[:1],[ambiguous],[{}],SearchSettings())


def test_fit_equal_reference_weight_and_large_residual_warning(tmp_path,monkeypatch,capsys):
    raw=reader(tmp_path,[300.,310.,500.,300.])
    rows=[CalibrationObservation(r,r.expected_mz*(1+e*1e-6),r.expected_rt,3,1.,'scan=1',intensity)
          for r,e,intensity in zip(references(3),[300.,310.,500.],[1e3,1e9,1.])]
    c=fit_constant_ppm(rows)
    assert abs(c.offset_ppm-310)<1e-7 and abs(c.calibration_residual(rows[-1]))>20
    monkeypatch.setattr('kmzconvert.cli.WiffReader',lambda _:raw)
    assert main([str(raw.entry),'--mass-reference','200;3','--mass-reference','250;3',
                 '--mass-reference','300;5','-o',str(tmp_path/'large.mzML')])==0
    log=capsys.readouterr().err
    assert WARNING in log and '20 ppm target NOT MET' in log and '300;5' in log
    assert (tmp_path/'large.mzML').is_file()


def test_disabled_regression_and_full_sparse_identity_metadata(tmp_path):
    raw=reader(tmp_path)
    base=write_mzml(raw,tmp_path/'plain.mzML')
    assert base.read_bytes()==write_mzml(raw,tmp_path/'plain-again.mzML',mass_correction=None).read_bytes()
    correction=estimate_mass_correction(raw,references())
    full=write_mzml(raw,tmp_path/'full.mc.mzML',mass_correction=correction)
    sparse=write_mzml(raw,tmp_path/'sparse.mc.sp.mzML',profile='sparse',mass_correction=correction)
    # v3 selects across consecutive MS1 spectra on the corrected axis.
    assert len({s.experiment for s in raw.scans if raw.experiments[s.experiment-1].ms_level==1})==1
    plan=iter(event_rescue_indices([(r['arrays']['mz'],r['arrays']['intensity']) for r in iter_records(full) if r['ms_level']==1]))
    for original,f,s in zip(iter_records(base),iter_records(full),iter_records(sparse),strict=True):
        assert original['id']==f['id']==s['id']
        assert original['rt_seconds']==f['rt_seconds']==s['rt_seconds']
        assert original['cv']==f['cv']==s['cv']
        np.testing.assert_array_equal(original['arrays']['intensity'],f['arrays']['intensity'])
        if original['ms_level']==1:
            keep=next(plan)[0]
            np.testing.assert_array_equal(f['arrays']['mz'],correction.correct(original['arrays']['mz']))
            np.testing.assert_array_equal(s['arrays']['mz'],f['arrays']['mz'][keep])
            np.testing.assert_array_equal(s['arrays']['intensity'],f['arrays']['intensity'][keep])
            np.testing.assert_array_equal(gap_connections(original['arrays']['mz']),gap_connections(f['arrays']['mz']))
            np.testing.assert_array_equal(gap_connections(original['arrays']['mz'][keep]),gap_connections(s['arrays']['mz']))
            assert len(f['arrays']['mz'])==len(original['arrays']['mz'])
        else:
            for key in ('mz','intensity'):
                np.testing.assert_array_equal(original['arrays'][key],f['arrays'][key])
                np.testing.assert_array_equal(original['arrays'][key],s['arrays'][key])
    for path in [full,sparse]:
        raw_tic,=iter_records(base,'chromatogram');new_tic,=iter_records(path,'chromatogram')
        for key in raw_tic['arrays']:np.testing.assert_array_equal(raw_tic['arrays'][key],new_tic['arrays'][key])
        assert check_index(path)['checksum_valid']
    renamed=tmp_path/'renamed.mzML';sparse.rename(renamed)
    table=index_ms1(renamed)
    assert all(c is not None for c in table.sparse_profile) and read_provenance(renamed).methods
    assert not read_provenance(full).methods
    specs=Path('.cache/standards')
    if (specs/'mzML1.1.0_idx.xsd').exists():
        for path in [full,renamed]:
            result=check(path,specs/'mzML1.1.0_idx.xsd',specs/'psi-ms.obo')
            assert result['xsd_valid'] and result['psi_ms_names_valid']


@pytest.mark.parametrize('failure',['version','scope','estimator','offset','observations','duplicate','action','software','order','extra','settings'])
def test_corrected_sparse_provenance_rejects_unsupported_chains(tmp_path,failure):
    raw=reader(tmp_path);correction=estimate_mass_correction(raw,references(1))
    root=ET.Element('mzML')
    for element in metadata(raw,'sparse',correction):root.append(element)
    assert Provenance(root).methods=={FORMAT}
    chain=root.find(f'.//dataProcessing[@id="{FORMAT}"]')
    method=chain.find('processingMethod[@order="1"]')
    params={e.get('name'):e for e in method.findall('userParam')}
    if failure in ('version','scope','estimator'):
        params['KmzSuite mass correction '+failure].set('value','unsupported')
    elif failure in ('offset','observations','settings'):
        payload=json.loads(params[PAYLOAD].get('value'))
        if failure=='offset':payload['offset_ppm']=-1e6
        elif failure=='observations':payload['references']=[]
        else:payload['search_settings']['high_fraction']=.7
        params[PAYLOAD].set('value',json.dumps(payload))
    elif failure=='duplicate':method.append(deepcopy(method.find('userParam')))
    elif failure=='action':method.find('cvParam').set('accession','MS:1001486')
    elif failure=='software':method.set('softwareRef','acquisition')
    elif failure=='order':method.set('order','2')
    else:ET.SubElement(chain,'processingMethod',order='3',softwareRef='kmzconvert_core')
    assert not Provenance(root).methods


def test_cli_fresh_per_file_naming_custom_names_and_inspection(tmp_path,monkeypatch,capsys):
    a=reader(tmp_path,300.,'positive');b=reader(tmp_path,-300.,'negative')
    monkeypatch.setattr('kmzconvert.cli.WiffReader',lambda p:a if Path(p).stem=='positive' else b)
    monkeypatch.chdir(tmp_path)
    for raw in [a,b]:
        assert main([str(raw.entry),'--mass-reference','200;3'])==0
        path=tmp_path/(raw.entry.stem+'.mc.mzML')
        root=ET.parse(path).getroot()
        payload=root.find('.//'+NS+f'userParam[@name="{PAYLOAD}"]')
        value=json.loads(payload.get('value'))
        assert abs(value['offset_ppm']-(300. if raw is a else -300.))<1e-7
    before=(tmp_path/'positive.mc.mzML').read_bytes()
    assert main([str(a.entry),'--mass-reference','200;3'])==1
    assert (tmp_path/'positive.mc.mzML').read_bytes()==before
    assert main([str(a.entry),'--mass-reference','200;3','--profile','sparse'])==0
    assert (tmp_path/'positive.mc.sp.mzML').exists()
    assert main([str(a.entry),'--mass-reference','200;3','-o','custom.mzML'])==0
    assert (tmp_path/'custom.mzML').exists() and not (tmp_path/'custom.mc.mzML').exists()
    capsys.readouterr()
    assert main([str(a.entry),'--inspect','--mass-reference','malformed'])==0
    captured=capsys.readouterr()
    assert json.loads(captured.out)['spectra']==len(a.scans) and WARNING not in captured.err


def test_cancellation_during_estimation_and_writing_preserves_inputs(tmp_path,monkeypatch):
    raw=reader(tmp_path)
    before=raw.entry.read_bytes(),raw.scan_path.read_bytes()
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls==210:raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):estimate_mass_correction(raw,references(),cancel_check=cancel)
    assert not list(tmp_path.glob('.kmzconvert-*.tmp'))
    correction=estimate_mass_correction(raw,references(1))
    def progress(*_):raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):write_mzml(raw,tmp_path/'cancelled.mzML',progress,mass_correction=correction)
    assert not (tmp_path/'cancelled.mzML').exists() and not list(tmp_path.glob('.kmzconvert-*.tmp'))
    monkeypatch.setattr('kmzconvert.cli.WiffReader',lambda _:raw)
    monkeypatch.setattr('kmzconvert.cli.estimate_mass_correction',lambda *a,**kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert main([str(raw.entry),'--mass-reference','200;3','-o',str(tmp_path/'never.mzML')])==130
    assert raw.closed and not (tmp_path/'never.mzML').exists()
    assert (raw.entry.read_bytes(),raw.scan_path.read_bytes())==before


@pytest.mark.parametrize('offset',[float('nan'),float('inf'),-1e6,-2e6])
def test_invalid_denominator(offset):
    with pytest.raises(ValueError):MassCorrection(offset,(),SearchSettings())
