#!/usr/bin/env python3
"""Qualify exactly five held-out native acquisitions with frozen A and D, no sweep."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/validation'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/state/matplotlib'))
os.environ.setdefault('KMZ_LOCAL_STATE_ROOT', str(ROOT / '.local/state'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from kmzconvert_core import writer
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import iter_records
from kmzviewer.spectra import SpectrumReader
from scripts.benchmarks.ms1_algorithms import gap_connections, guard_indices, major_apices
from scripts.benchmarks.ms1_mzml import write_raw_inspection
from scripts.benchmarks.ms1_provenance import CONTRACT
from scripts.benchmarks.ms1_qualification import (
    csv_write, evaluate, exact_subset, gate_file, inventory, select_representatives, summarize, write_d,
)
from scripts.benchmarks.ms1_report import plot_line, zooms, actual_viewer_frame
from check_mzml import check_index


def figure(scan, destination, bridge=None, peak_mz=None):
    from matplotlib import pyplot as plt
    mz, y = scan['mz'], scan['y']
    a, d = guard_indices(y, guards=0), guard_indices(y, guards=1)
    reps = [(np.arange(len(mz)), np.ones(max(0, len(mz)-1), bool)),
            (a, np.ones(max(0, len(a)-1), bool)), (d, gap_connections(mz[d], factor=4, radius=8))]
    views = [(float(mz[0]), float(mz[-1]))] + zooms(mz, y)
    if bridge is not None:
        lo, hi = bridge
        pad = max(.02, (hi-lo)*.3)
        views[1] = (lo-pad, hi+pad)
    if peak_mz is not None:
        views[2] = (peak_mz-.08, peak_mz+.08)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'path.simplify': False}):
        fig, axes = plt.subplots(3, 3, figsize=(14, 7.8), layout='constrained', sharex='col', sharey='col')
        for col, (lo, hi) in enumerate(views):
            # Include all crossing segments in limits, including outside endpoints.
            heights = []
            from scripts.benchmarks.ms1_algorithms import plotted_trace
            grid = np.unique(np.r_[lo, mz[(mz>=lo)&(mz<=hi)], hi])
            for idx, conn in reps:
                heights.append(np.nan_to_num(plotted_trace(grid, mz[idx], y[idx], conn)).max(initial=100))
            ymax = max(heights)*1.1
            for row, (idx, conn) in enumerate(reps):
                ax = axes[row, col]
                plot_line(ax, mz, y, idx, conn, ['#25364b', '#ba4a35', '#18795c'][row], gap=row==2, guards=row==2 and col>0)
                ax.set(xlim=(lo, hi), ylim=(0, ymax), xlabel='m/z' if row==2 else '', ylabel='Counts' if col==0 else '')
                ax.grid(alpha=.15)
                ax.ticklabel_format(axis='x', useOffset=False)
                if col==0:
                    ax.set_title(['Full profile reference', 'A: strict >100', 'D: one source guard + factor 4'][row], loc='left')
                if row==0 and col>0:
                    ax.set_title('Residual bridge zoom' if col==1 and bridge is not None else ['','Control bridge zoom','Threshold-adjacent peak'][col])
        fig.suptitle(f"{scan['acquisition']} · spectrum {scan['index']} · {scan['id']} · RT {scan['rt_seconds']/60:.3f} min\n" + ', '.join(scan['reasons']), fontsize=11)
        fig.savefig(destination, dpi=125)
        plt.close(fig)
    from PIL import Image
    with Image.open(destination) as im:
        im.convert('RGB').save(destination, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=ROOT/'FromChase')
    parser.add_argument('--output', type=Path, required=True, help='New ignored directory inside .local')
    args = parser.parse_args()
    fixtures, output = args.fixtures.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT/'.local') or not fixtures.is_relative_to(ROOT):
        parser.error('Inputs stay in checkout; qualification output must be inside .local')
    paths, before = inventory(fixtures)
    for row in before:
        subprocess.run(['git','check-ignore','--quiet',str((fixtures/row['filename']).relative_to(ROOT))], cwd=ROOT, check=True)
    if subprocess.check_output(['git','ls-files','--',str(fixtures.relative_to(ROOT))],cwd=ROOT):
        parser.error('Private fixtures must be untracked')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('mzml','figures','selected-arrays'):
        (output/name).mkdir()
    (output/'inventory-before.json').write_text(json.dumps(before,indent=2)+'\n')
    (output/'settings.json').write_text(json.dumps({'contract':CONTRACT,'conditions':['A','D'],'sweep':False,
         'source': 'unchanged production framed profile', 'python':platform.python_version(),
         'started_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    rows, peak_rows, costs, native, checks = [], [], [], [], []
    try:
        for i, path in enumerate(paths, 1):
            acquisition = f'heldout-{i:02d}'
            files = {c:output/'mzml'/f'{acquisition}-{c}.mzML' for c in ('reference','A','D')}
            with WiffReader(path) as reader:
                native.append(dict(acquisition=acquisition,instrument=reader.instrument['Mass Spectrometer'],
                    acquisition_software=reader.acquisition_software,polarities=sorted({e.polarity for e in reader.experiments}),
                    spectra=len(reader.scans),ms1_spectra=sum(reader.experiments[s.experiment-1].ms_level==1 for s in reader.scans),
                    calibration_resolutions=len(reader.calibration_resolutions),sampling_family='sciex-positive-tof-framed-profile-v1'))
                print(f'{acquisition}: native reader accepted {len(reader.scans)} spectra',flush=True)
                for condition in ('reference','A','D'):
                    started = time.perf_counter()
                    if condition=='reference':
                        writer.write_mzml(reader, files[condition])
                    elif condition=='A':
                        write_raw_inspection(reader, files[condition], guards=0)
                    else:
                        write_d(reader, files[condition])
                    seconds = time.perf_counter()-started
                    idx = check_index(files[condition])
                    assert idx['checksum_valid']
                    start = time.perf_counter()
                    count = sum(1 for _ in iter_records(files[condition]))
                    parse = time.perf_counter()-start
                    costs.append(dict(acquisition=acquisition,condition=condition,bytes=files[condition].stat().st_size,
                                      generation_seconds=seconds,parse_all_spectra_seconds=parse,spectra=count))
                    checks.append(dict(acquisition=acquisition,condition=condition,index=idx))
                    print(f'  {condition}: {files[condition].stat().st_size/1e6:.2f} MB; write {seconds:.2f}s; parse {parse:.2f}s',flush=True)
                gates = gate_file(files['D'])
                ordinary_gates = gate_file(files['reference'])
                assert not any(r['accepted'] for r in ordinary_gates)
                assert not any(r['accepted'] for r in gates if r['ms_level']!=1)
                # Compare every native scan to its authoritative reference and both isolated outputs.
                for index, (scan, reference, a, d) in enumerate(zip(reader.scans, iter_records(files['reference']),
                                                                    iter_records(files['A']), iter_records(files['D']), strict=True)):
                    mz,y = reader.arrays(scan)
                    level = reader.experiments[scan.experiment-1].ms_level
                    for record in (reference,a,d):
                        assert record['id']==scan.native_id and record['ms_level']==level
                        assert record['representation']=='profile' and record['polarity']=='positive'
                        assert abs(record['rt_seconds']-scan.rt_seconds) < 1e-10
                        assert float(record['cv']['MS:1000285']['value'])==scan.tic
                        assert record['precursors']==reference['precursors']
                    exact_subset(mz,y,np.arange(len(mz)),reference['arrays']['mz'],reference['arrays']['intensity'])
                    if level!=1:
                        for record in (a,d):
                            exact_subset(mz,y,np.arange(len(mz)),record['arrays']['mz'],record['arrays']['intensity'])
                        continue
                    source_connections = gap_connections(mz, factor=4, radius=8)
                    signal = (y[:-1]>0)|(y[1:]>0)
                    signal_breaks = int(np.count_nonzero(~source_connections & signal))
                    delta = np.diff(np.sqrt(mz))[signal]
                    ratio = float(delta.max()/delta.min()) if len(delta) else 0.
                    apices = major_apices(mz,y)
                    maxima = np.flatnonzero((y[1:-1]>y[:-2])&(y[1:-1]>=y[2:]))+1
                    apex = int(np.argmax(y)) if len(y) else 0
                    prefix = dict(acquisition=acquisition,index=index,id=scan.native_id,rt_seconds=scan.rt_seconds,
                                  source_signal_breaks=signal_breaks,native_spacing_ratio=ratio,
                                  base_peak=float(y.max(initial=0)),weak_peak_count=int(np.count_nonzero((y[maxima]>100)&(y[maxima]<=300))),
                                  cluster_score=int(np.count_nonzero((abs(mz[maxima]-mz[apex])<=.25)&(y[maxima]>=max(100,y[apex]*.05)))) if len(y) else 0)
                    for condition,record,guards in (('A',a,0),('D',d,1)):
                        result, peaks = evaluate(mz,y,record['arrays']['mz'],record['arrays']['intensity'],guards)
                        accepted = gates[index]['accepted'] if condition=='D' else False
                        assert condition!='D' or accepted==(signal_breaks==0)
                        widths = [p['reference_fwhm'] for p in peaks if p['reference_fwhm'] is not None]
                        rows.append(prefix | dict(condition=condition,gate_accepted=int(accepted),gate_declined=int(not accepted) if condition=='D' else 0,
                                                 broad_fwhm=max(widths,default=0)) | result)
                        peak_rows.extend(prefix | dict(condition=condition) | p for p in peaks)
                    if sum(r['condition']=='D' and r['acquisition']==acquisition for r in rows)%250==0:
                        print(f'  metrics through spectrum {index}',flush=True)
                for condition in ('reference','A','D'):
                    records=list(iter_records(files[condition],'chromatogram'))
                    assert len(records)==1
                    for kind,expected in zip(('time','intensity','ms level'),reader.tic_arrays(),strict=True):
                        np.testing.assert_array_equal(records[0]['arrays'][kind],expected)
            csv_write(output/'per-spectrum-metrics.csv',rows)
            csv_write(output/'peak-metrics.csv',peak_rows)
            csv_write(output/'file-costs.csv',costs)
            print(f'{acquisition}: all native/reference/A/D arrays and vendor chromatogram exact',flush=True)
        summary=[summarize(rows,a,c) for a in [f'heldout-{i:02d}' for i in range(1,6)]+['pooled'] for c in ('A','D')]
        csv_write(output/'per-acquisition-summary.csv',[r for r in summary if r['acquisition']!='pooled'])
        csv_write(output/'aggregate-metrics.csv',[r for r in summary if r['acquisition']=='pooled'])
        selected=select_representatives(rows)
        (output/'selection.json').write_text(json.dumps([dict(acquisition=a,index=i,reasons=reasons) for (a,i),reasons in selected.items()],indent=2)+'\n')
        loads=[]
        for j,path in enumerate(paths,1):
            acquisition=f'heldout-{j:02d}'
            with WiffReader(path) as reader:
                for (a,index), reasons in selected.items():
                    if a!=acquisition:
                        continue
                    scan=reader.scans[index]
                    mz,y=reader.arrays(scan)
                    data=dict(acquisition=a,dataset=a,index=index,id=scan.native_id,rt_seconds=scan.rt_seconds,mz=mz,y=y,
                              key=f'{a}-scan-{index:05d}',reasons=reasons)
                    np.savez_compressed(output/'selected-arrays'/f"{data['key']}.npz",mz=mz,intensity=y)
                    d=guard_indices(y,guards=1);conn=gap_connections(mz[d],factor=4,radius=8)
                    bridges=(np.diff(d)>1)&conn
                    bridge=None
                    if bridges.any():
                        area=np.diff(mz[d])*(y[d[:-1]]+y[d[1:]])/2
                        at=int(np.argmax(np.where(bridges,area,-1)))
                        bridge=(float(mz[d[at]]),float(mz[d[at+1]]))
                    figure(data,output/'figures'/f"{data['key']}.png",bridge=bridge)
            for condition,guards in (('reference',None),('A',0),('D',1)):
                started=time.perf_counter()
                with SpectrumReader(output/'mzml'/f'{acquisition}-{condition}.mzML') as loader, WiffReader(path) as reader:
                    setup=time.perf_counter()-started
                    for (a,index), reasons in selected.items():
                        if a!=acquisition:
                            continue
                        started=time.perf_counter();m,y=loader.read(index);seconds=time.perf_counter()-started
                        sm,sy=reader.arrays(reader.scans[index]);idx=np.arange(len(sm)) if guards is None else guard_indices(sy,guards=guards)
                        exact_subset(sm,sy,idx,m,y)
                        loads.append(dict(acquisition=a,index=index,condition=condition,setup_seconds=setup,read_seconds=seconds,exact=True))
        csv_write(output/'viewer-loading.csv',loads)
        (output/'native-provenance.json').write_text(json.dumps(native,indent=2)+'\n')
        (output/'file-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
        (output/'completion.json').write_text(json.dumps({'all_five_processed':True,'all_spectra_exact':True,'chromatograms_exact':True,
            'production_source_changed':False,'ms1_spectra':sum(r['ms1_spectra'] for r in native),'selected_visual_spectra':len(selected)},indent=2)+'\n')
        print(json.dumps([s for s in summary if s['condition']=='D'],indent=2),flush=True)
    finally:
        _, after=inventory(fixtures)
        (output/'inventory-after.json').write_text(json.dumps(after,indent=2)+'\n')
        unchanged=before==after
        (output/'immutability.json').write_text(json.dumps({'files':len(before),'all_pre_post_sha256_match':unchanged},indent=2)+'\n')
        assert unchanged,'Immutable source files changed'


if __name__=='__main__':
    main()
