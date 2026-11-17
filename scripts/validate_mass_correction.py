#!/usr/bin/env python3
"""Local beta smoke: frozen CLI, complete native-axis preservation and sparse gates."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from kmzconvert_core import native
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import NS, read_arrays
from kmzconvert_core.writer import spectrum
from kmzcore.mass_correction import PAYLOAD, validate_payload
from kmzcore.sparse_profile import FORMAT, SAMPLING_PARAM, gap_connections, read_provenance, rendering_connections
from scripts.benchmarks.ms1_algorithms import event_rescue_plan
from kmzviewer.mzml_index import index_ms1
from kmzviewer.spectra import SpectrumReader
from scripts.validation.check_mzml import check


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def elements(path, tag):
    for _, e in ET.iterparse(path):
        if e.tag == NS+tag:
            yield e
            e.clear()
        elif e.tag in (NS+'spectrum', NS+'chromatogram'):
            e.clear()


def payload(path):
    for _, e in ET.iterparse(path):
        if e.tag == NS+'dataProcessing' and e.get('id') == 'mass-correction-v1':
            result = json.loads(e.find('.//'+NS+f'userParam[@name="{PAYLOAD}"]').get('value'))
            assert validate_payload(result)
            return result
        if e.tag == NS+'spectrumList':
            break
    raise AssertionError('Correction provenance absent')


def gap_ratios(mz):
    """Independent diagnostic of the frozen rule's four-spacing boundary."""
    delta=np.diff(np.sqrt(mz))
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(delta,(8,8),constant_values=np.nan),17)
    estimates=[]
    for side in (windows[:,:8],windows[:,9:]):
        valid=np.isfinite(side);count=valid.sum(axis=1)
        ordered=np.sort(np.where(valid,side,np.inf),axis=1)
        rank=.25*np.maximum(count-1,0);lo=np.floor(rank).astype(int);hi=np.ceil(rank).astype(int)
        a=ordered[np.arange(len(delta)),lo];b=ordered[np.arange(len(delta)),hi]
        estimate=np.zeros(len(delta));use=count>0
        estimate[use]=a[use]+(b[use]-a[use])*(rank[use]-lo[use])
        support=(count>=3)&(np.sum(valid&(side>=.8*estimate[:,None])&(side<=1.2*estimate[:,None]),axis=1)>=2)
        estimates.append(np.where(support,estimate,0))
    local=np.maximum(*estimates)
    return np.divide(delta,local,out=np.zeros_like(delta),where=local>0)


def verify(reader, full, sparse, correction):
    # Expected arrays and spectra come from the interpreted reference, never the converter's native path.
    assert not native.applies(reader), 'verify requires the interpreted reference WiffReader'
    table = index_ms1(sparse)
    provenance = read_provenance(sparse)
    counts = dict(spectra=0, ms1_spectra=0, ms2_spectra=0, full_ms1_points=0, sparse_ms1_points=0)
    gaps=dict(full_intervals=0,sparse_intervals=0,full_boundary_flips=0,sparse_boundary_flips=0,
              full_signal_flips=0,sparse_signal_flips=0,max_boundary_ratio_deviation=0.,
              max_relative_sqrt_gap_rounding=0.)
    samples = []
    denominator = 1+correction['offset_ppm']*1e-6
    # Sparse Profile v3 selects across MS1 spectra on the corrected axis.
    def corrected():
        for i, scan in enumerate(reader.scans):
            if reader.experiments[scan.experiment-1].ms_level == 1:
                mz, intensity = reader.arrays(scan)
                yield i, scan.experiment, mz/denominator, intensity
    plan = event_rescue_plan(corrected())
    for i, (scan, a, b) in enumerate(zip(reader.scans, elements(full,'spectrum'), elements(sparse,'spectrum'), strict=True)):
        raw = ET.fromstring(b'<mzML xmlns="http://psi.hupo.org/ms/mzml">'+ET.tostring(spectrum(reader,scan,i))+b'</mzML>')[0]
        aa, bb, original = read_arrays(a), read_arrays(b), read_arrays(raw)
        level = reader.experiments[scan.experiment-1].ms_level
        counts['spectra'] += 1
        np.testing.assert_array_equal(original['intensity'], aa['intensity'])
        if level == 1:
            expected = original['mz']/denominator
            np.testing.assert_array_equal(expected, aa['mz'])
            assert np.all(np.diff(aa['mz'])>0) and len(aa['mz'])==len(original['mz'])
            keep = plan[i]
            np.testing.assert_array_equal(expected[keep], bb['mz'])
            np.testing.assert_array_equal(original['intensity'][keep], bb['intensity'])
            # The real-arithmetic rule is invariant. Quantify floating ties;
            # do not hide flips or change the existing production rule.
            for tag,indices in [('full',np.arange(len(expected))),('sparse',keep)]:
                before=original['mz'][indices];after=expected[indices];signal=original['intensity'][indices]
                changed=gap_connections(before)!=gap_connections(after)
                gaps[tag+'_intervals']+=max(0,len(before)-1)
                gaps[tag+'_boundary_flips']+=int(changed.sum())
                gaps[tag+'_signal_flips']+=int(np.count_nonzero(changed&((signal[:-1]>0)|(signal[1:]>0))))
                if np.any(changed):
                    deviation=max(float(np.max(np.abs(gap_ratios(m)[changed]-4))) for m in (before,after))
                    assert deviation<1e-8, ('non-boundary gap change',i,tag,deviation)
                    gaps['max_boundary_ratio_deviation']=max(gaps['max_boundary_ratio_deviation'],deviation)
                if len(before)>1:
                    delta=np.diff(np.sqrt(before))/np.sqrt(denominator)
                    error=np.max(np.abs(np.diff(np.sqrt(after))-delta)/delta)
                    gaps['max_relative_sqrt_gap_rounding']=max(gaps['max_relative_sqrt_gap_rounding'],float(error))
            attestation = provenance.sampling(b)
            assert attestation is not None and attestation == table.sparse_profile[counts['ms1_spectra']]
            np.testing.assert_array_equal(rendering_connections(attestation,bb['mz'],bb['intensity']), gap_connections(bb['mz']))
            assert a.get('dataProcessingRef')=='mass-correction-v1' and b.get('dataProcessingRef')==FORMAT
            counts['ms1_spectra'] += 1
            counts['full_ms1_points'] += len(aa['mz'])
            counts['sparse_ms1_points'] += len(bb['mz'])
            if len(samples)<3 or i % 503==0:
                samples.append((i,bb['mz'].copy(),bb['intensity'].copy()))
        else:
            counts['ms2_spectra'] += 1
            for kind in ('mz','intensity'):
                np.testing.assert_array_equal(original[kind],aa[kind])
                np.testing.assert_array_equal(original[kind],bb[kind])
            assert 'dataProcessingRef' not in a.attrib and 'dataProcessingRef' not in b.attrib
        # Strip only explicitly declared transformation metadata/binaries.
        copies = [deepcopy(x) for x in (raw,a,b)]
        for e in copies:
            e.tail = None
            e.attrib.pop('dataProcessingRef',None)
            e.set('defaultArrayLength',raw.get('defaultArrayLength'))
            att = e.find(NS+f'userParam[@name="{SAMPLING_PARAM}"]')
            if att is not None: e.remove(att)
            e.remove(e.find(NS+'binaryDataArrayList'))
        assert ET.tostring(copies[0])==ET.tostring(copies[1])==ET.tostring(copies[2]), ('scan metadata',i)
    for a,b in zip(elements(full,'chromatogram'),elements(sparse,'chromatogram'),strict=True):
        aa,bb = read_arrays(a),read_arrays(b)
        times,intensities,levels = reader.tic_arrays()
        for key,values in [('time',times),('intensity',intensities),('ms level',levels)]:
            np.testing.assert_array_equal(aa[key],values)
            np.testing.assert_array_equal(bb[key],values)
        a.tail=b.tail=None
        assert ET.tostring(a)==ET.tostring(b)
    with SpectrumReader(sparse) as indexed:
        for ordinal,m,y in samples:
            actual_m,actual_y=indexed.read(ordinal)
            np.testing.assert_array_equal(m,actual_m);np.testing.assert_array_equal(y,actual_y)
    assert gaps['full_signal_flips']==0  # Full-source sampling attestation stays active.
    return dict(**counts,indexed_readbacks=len(samples),arrays_metadata_tic_preserved=True,
                full_sparse_consistent=True,ordered_axes=True,gap_behavior=gaps,
                corrected_sparse_recognized_after_renaming=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--references',type=Path,required=True,help='Private JSON with references: expected_mz, expected_rt')
    parser.add_argument('--cli',type=Path,default=ROOT/'build/latest/kmzconvert')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'.local'):parser.error('Output must be a new ignored .local directory')
    refs=json.loads(args.references.read_text())['references']
    output.mkdir(parents=True,exist_ok=False)
    source=args.input.resolve(); sources=[source,source.with_suffix('.wiff.scan')]
    before=[dict(id=f'input-{i+1}',bytes=p.stat().st_size,sha256=digest(p)) for i,p in enumerate(sources)]
    results={}
    try:
        offsets=[]
        for profile in ('full','sparse'):
            path=output/('acquisition.mc.mzML' if profile=='full' else 'acquisition.mc.sp.mzML')
            command=[str(args.cli.resolve()),str(source),'--profile',profile,'-o',str(path)]
            for ref in refs:command.extend(['--mass-reference',f"{ref['expected_mz']};{ref['expected_rt']}"])
            start=time.perf_counter()
            with (output/(profile+'.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            results[profile]=dict(seconds=time.perf_counter()-start,bytes=path.stat().st_size)
            offsets.append(payload(path))
            validation=check(path,ROOT/'.cache/standards/mzML1.1.0_idx.xsd',ROOT/'.cache/standards/psi-ms.obo')
            assert validation['xsd_valid'] and validation['psi_ms_names_valid'] and validation['index']['checksum_valid']
            (output/(profile+'-validation.json')).write_text(json.dumps(validation,indent=2)+'\n')
        assert offsets[0]==offsets[1], 'Fresh independent fits must agree for the same acquisition'
        correction=offsets[0]
        full=output/'acquisition.mc.mzML'
        sparse=output/'acquisition.mc.sp.mzML'
        renamed=output/'renamed.mzML';sparse.rename(renamed)
        try:
            with WiffReader(source) as reader:results.update(verify(reader,full,renamed,correction))
        finally:
            renamed.rename(sparse)
        residuals=[r['calibration_residual_ppm'] for r in correction['references']]
        results.update(offset_ppm=correction['offset_ppm'],references=len(refs),worst_calibration_residual_ppm=max(map(abs,residuals)),
                       accuracy_target_met_at_fitting_references=max(map(abs,residuals))<=20,predictive_accuracy='unverified')
        with (output/'references.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(correction['references'][0]));w.writeheader();w.writerows(correction['references'])
        (output/'correction.json').write_text(json.dumps(correction,indent=2)+'\n')
    finally:
        after=[dict(id=f'input-{i+1}',bytes=p.stat().st_size,sha256=digest(p)) for i,p in enumerate(sources)]
        results['source_identities']=before
        results['sources_unchanged']=before==after
        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        assert before==after,'Original inputs changed'
    assert results['accuracy_target_met_at_fitting_references'],'20 ppm fitting-reference target not met'
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
