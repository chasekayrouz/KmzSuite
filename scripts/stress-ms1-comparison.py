#!/usr/bin/env python3
"""Local Qt MS1 interaction/performance qualification with synthetic and optional local runs.

Use the qualified repo environment. Outputs are local evidence, never raw-data review
inputs. The default Qt platform is native; QT_QPA_PLATFORM=offscreen is also supported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Tests.Viewer.mzml_factory import write_mzml
from kmzviewer.gui.app import MainWindow


def pump(app, seconds=.01):
    end=time.perf_counter()+seconds
    while time.perf_counter()<end:
        app.processEvents()
        time.sleep(.001)


def settle(app,w,timeout=30):
    end=time.perf_counter()+timeout
    while w._inflight_comparison is not None or w._queued_comparison is not None:
        assert time.perf_counter()<end, 'Comparison did not settle'
        pump(app,.002)
    pump(app,.02)


def close(app,w):
    w.close()
    end=time.perf_counter()+20
    while not w.runtime.closed:
        assert time.perf_counter()<end
        pump(app)


def stats(values):
    return {'n':len(values),'median_ms':round(float(np.median(values))*1000,2),
            'p95_ms':round(float(np.percentile(values,95))*1000,2),'max_ms':round(max(values)*1000,2)}


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024): h.update(block)
    return h.hexdigest()


def fixtures(directory,points,scans):
    directory.mkdir()
    paths=[]
    mz=np.linspace(100,900,points)
    rng=np.random.default_rng(12031)
    centers=np.r_[173.12,273.1559,301.14,445.22,701.41,rng.uniform(110,890,100)]
    heights=np.r_[.4,1,.65,.3,.7,rng.uniform(.001,.04,100)]
    for source in range(6):
        path=directory/f'Comparison {source+1}.mzML'
        spectra=[]
        for scan in range(scans):
            y=np.zeros(points)
            drift=.02*np.sin(scan*.15)
            for mass,height in zip(centers,heights):
                y+=height*np.exp(-((mz-mass-drift)/.08)**2)
            y*=1e5*(1+source*.25)*(.12+np.exp(-((scan-scans*.5)/(scans*.2))**2))
            spectra.append(dict(mz=mz,intensity=y,rt=1+scan*.04+source*.0003,
                                tic=float(y.sum()),centroid=False,polarity=1,id=f'comparison-{scan}'))
        write_mzml(path,spectra)
        paths.append(path)
        print(f'Generated synthetic file {source+1}/6',flush=True)
    return paths


def position(w,trace,rt):
    view=w.plots._plot_for_trace[trace.trace_id].getViewBox()
    y0,y1=view.viewRange()[1]
    return w.plots.view.mapFromScene(view.mapViewToScene(QPointF(rt,(y0+y1)/2)))


def send_mouse(viewport,kind,pos,button,buttons):
    pos=QPointF(pos)
    QApplication.sendEvent(viewport,QMouseEvent(kind,pos,QPointF(viewport.mapToGlobal(pos.toPoint())),
                                               button,buttons,Qt.KeyboardModifier.NoModifier))


def sweep(app,w,trace,low,high,steps=100):
    viewport=w.plots.view.viewport()
    start=position(w,trace,low)
    send_mouse(viewport,QEvent.Type.MouseMove,start,Qt.MouseButton.NoButton,Qt.MouseButton.NoButton)
    pump(app,.04)
    send_mouse(viewport,QEvent.Type.MouseButtonPress,start,Qt.MouseButton.LeftButton,Qt.MouseButton.LeftButton)
    for rt in np.linspace(low,high,steps):
        pos=position(w,trace,float(rt))
        send_mouse(viewport,QEvent.Type.MouseMove,pos,Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton)
        pump(app,.016)
    send_mouse(viewport,QEvent.Type.MouseButtonRelease,pos,Qt.MouseButton.LeftButton,Qt.MouseButton.NoButton)
    released=time.perf_counter()
    settle(app,w)
    return time.perf_counter()-released


def geometry(w):
    rows=[]
    for fid,plot in w.spectrum._plots_by_file.items():
        view=plot.getViewBox()
        rect=view.sceneBoundingRect()
        labels=[item for item in w.spectrum._label_pools[fid] if item.isVisible()]
        assert labels, 'Six-spectrum pane lost all peak labels'
        title=w.spectrum._names[fid].sceneBoundingRect()
        for item in labels:
            label=item.textItem.mapRectToScene(item.textItem.boundingRect())
            assert label.left()>=rect.left()+.5 and label.right()<=rect.right()-.5
            assert label.top()>=title.bottom() and label.bottom()<=rect.bottom()
        rows.append({'file_id':fid,'plot_height_px':round(view.height(),1),'peak_labels':len(labels)})
    return rows


def run(app,paths,output,tag,*,screenshots=False):
    w=MainWindow()
    w.resize(1512,930)
    w.show()
    pump(app,.05)
    assert not w.load_paths_sync([str(p) for p in paths])
    files=list(w.session.files.values())
    trace=files[0].traces[0]
    low,high=map(float,(trace.rt[len(trace.rt)//4],trace.rt[len(trace.rt)*3//4]))
    frames=[]
    original=w.spectrum.show_frames
    def record(rows,caption):
        started=time.perf_counter()
        original(rows,caption)
        frames.append((time.perf_counter(),time.perf_counter()-started,[f.rt for f in rows]))
    w.spectrum.show_frames=record
    start=time.perf_counter()
    w.inspect_rt(trace.trace_id,low)
    for f in files[1:]: w.toggle_spectrum(f.file_id)
    settle(app,w)
    cold=time.perf_counter()-start
    assert len(w.spectrum._frames)==len(paths)
    w.spectrum.plot.setXRange(170,310,padding=0)
    pump(app,.03)
    identity=dict(w.spectrum._curves)
    count=len(frames)
    tail=sweep(app,w,trace,low,high)
    assert w.spectrum._curves==identity
    assert len(frames)-count>=5, 'No intermediate frames during drag'
    assert abs(w._selected_rt-high)<1e-8
    assert all(np.allclose(p.viewRange()[0], [170.,310.]) for p in w.spectrum._plots())
    drag_frames=frames[count:]
    warm=[]
    for rt in np.linspace(low,high,24)[::-1]:
        start=time.perf_counter()
        w._navigate_rt(trace.trace_id,float(rt)); settle(app,w)
        warm.append(time.perf_counter()-start)
    w.spectrum.reset_range()
    w._navigate_rt(trace.trace_id,(low+high)/2); settle(app,w)
    result={'files':len(paths),'cold_first_comparison_ms':round(cold*1000,2),
            'points_per_scan':[int(f.mz.size) for f in w.spectrum._frames],
            'drag_frames':len(drag_frames),'drag_final_settle_ms':round(tail*1000,2),
            'drag_frame_intervals':stats(np.diff([f[0] for f in drag_frames])),
            'frame_update':stats([f[1] for f in drag_frames]),'warm_navigation':stats(warm),
            'warm_navigation_includes_repaint_grace_ms':20}
    if screenshots:
        result['sizes']={}
        for width,height in [(1512,930),(1440,860),(1280,800)]:
            w.resize(width,height); pump(app,.1); w._balance_plot_widths(); pump(app,.05)
            result['sizes'][f'{width}x{height}']=geometry(w)
            assert w.grab().save(str(output/f'{tag}-six-{width}.png'))
        w.resize(1512,930); pump(app,.05); w._balance_plot_widths(); pump(app,.05)
        if app.platformName() == 'cocoa':
            w.showFullScreen(); pump(app,1.5)
            assert w.isFullScreen() and abs(w.plots.width()-w.spectrum.width()) < 10
            result['fullscreen']={'window_px':[w.width(),w.height()],
                                  'plot_widths_px':[w.plots.width(),w.spectrum.width()],
                                  'panes':geometry(w)}
            assert w.grab().save(str(output/f'{tag}-fullscreen.png'))
            w.showNormal(); pump(app,1.5)
        w.spectrum.set_mode('overlay'); pump(app,.1)
        assert w.grab().save(str(output/f'{tag}-overlay.png'))
        # Stress rapid layout and visibility changes with the same native arrays.
        for index in range(20):
            w.spectrum.set_mode('stacked' if index%2 else 'overlay')
            w.toggle_spectrum(files[-1].file_id)
            w._navigate_rt(trace.trace_id,float(trace.rt[index%len(trace.rt)]))
            pump(app,.005)
        settle(app,w)
        assert len(w.spectrum_file_ids)==6
        w.spectrum.set_mode('stacked')
        # Empty pane stays in its slot while the requested RT precedes later runs.
        w._navigate_rt(trace.trace_id,float(trace.rt[0])); settle(app,w)
        assert sum(f.message=='Outside run range' for f in w.spectrum._frames)==5
        assert all(not w.spectrum._plots_by_file[f.file_id].getAxis('left').style['showValues']
                   for f in w.spectrum._frames if f.message)
        assert w.grab().save(str(output/f'{tag}-outside-range.png'))
        w._navigate_rt(trace.trace_id,low); settle(app,w)
        for index,path in enumerate(paths):
            extra=output/'fixtures'/f'Extra {index+1}.mzML'
            shutil.copyfile(path,extra)
            assert not w.load_paths_sync([str(extra)])
            w.toggle_spectrum(list(w.session.files)[-1])
        settle(app,w)
        assert len(w.spectrum._frames)==12
        assert w.grab().save(str(output/f'{tag}-twelve.png'))
        result['twelve_pane_min_height_px']=round(min(p.getViewBox().height() for p in w.spectrum._plots()),1)
    close(app,w)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--points',type=int,default=50000)
    parser.add_argument('--scans',type=int,default=60)
    parser.add_argument('--fixtures',type=Path,help='Reuse six previously generated Comparison files, without changing them')
    parser.add_argument('--local-runs',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.fixtures:
        paths=sorted(args.fixtures.glob('Comparison *.mzML'))
        assert len(paths)==6
        (args.output/'fixtures').mkdir()
    else:
        paths=fixtures(args.output/'fixtures',args.points,args.scans)
    original=[checksum(p) for p in paths+args.local_runs]
    app=QApplication([])
    app.setQuitOnLastWindowClosed(False)
    report={'qt_platform':app.platformName(),'synthetic_points':args.points,'synthetic_scans':args.scans}
    report['synthetic']=run(app,paths,args.output,'synthetic',screenshots=True)
    print('Synthetic interaction sweep passed',flush=True)
    if args.local_runs:
        report['local_runs']=run(app,args.local_runs,args.output,'local')
        print('Local acquisition interaction sweep passed',flush=True)
    assert original==[checksum(p) for p in paths+args.local_runs]
    report['input_hashes_unchanged']=True
    report['inputs']=[{'id':f'input-{index+1}','bytes':p.stat().st_size,'sha256':h}
                      for index,(p,h) in enumerate(zip(paths+args.local_runs,original))]
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2),flush=True)


if __name__=='__main__': main()
