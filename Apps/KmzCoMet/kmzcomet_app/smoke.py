"""Installed-app exercise; generated UI data and all state stay in its output."""
import csv
import json
from pathlib import Path
import sys
import time

from PySide6.QtCore import QTimer,Qt
from PySide6.QtWidgets import QFileDialog,QMessageBox

from kmzcomet.engine import analyze
from kmzcomet.history import load_analysis,payload,save_analysis
from kmzcomet.model import Candidate,Feature,Measurement,Result,Sample
from kmzcore.qualified import qualified
from kmzcore.runtime import resource_path
from .dialogs import benchmark_settings
from .workers import Completed,Worker


def example_result(directory=None):
    """Small data-only UI fixture; no acquired spectra or personal metadata."""
    settings=benchmark_settings();samples=[];features=[];columns=[[],[],[]]
    heights=[(12000.,9000.,600.),(8000.,0.,None),(2500.,4500.,1900.),(900.,300.,250.)]
    for j,name in enumerate(('Sample A.wiff','Sample B.wiff','Sample C.wiff')):
        samples.append(Sample(Path(name),f'S{j+1:03d}','0'*64,100,scans=100,modes={'profile'},
                              units={'number of detector counts'},rt_min=0.,rt_max=5.,
                              shifts={1:0.},anchors={1:0},alignment_mad={1:None},native_sample_id=1))
    for i,values in enumerate(heights):
        mass,rt=150.+i*37.123,1.+i*.6;members=[]
        for j,h in enumerate(values):
            if h is not None and h>=1000:
                c=Candidate(j,mass,rt,h,1);samples[j].candidates.append(c);members.append(c)
        if not members:
            c=Candidate(0,mass,rt,values[0],1);samples[0].candidates.append(c);members=[c]
        feature=Feature(f'F{i+1:06d}',mass,rt,1,members,rt-.15,rt+.15);features.append(feature)
        for j,h in enumerate(values):
            status='not_acquired' if h is None else 'no_signal' if h==0 else 'supported'
            columns[j].append(Measurement(feature.id,samples[j].id,any(c.sample==j for c in members),status,
                height=h,height_above_background=h,window_area=None if h is None else h*.05,
                apex_rt=None if not h else rt,apex_mz=None if not h else mass,
                background=None if h is None else 0.,noise=None if h is None else 1.,
                snr=h,contrast=h,points=0 if h is None else 20,support_points=5 if h else 0))
    result=Result(settings,samples,features,columns,.1)
    if directory is not None:
        _example_sources(result,directory)
    return result


def _example_sources(result,directory):
    """Small generic full-run EIC inputs, created only for this smoke run."""
    import base64
    from copy import deepcopy
    import math
    import struct
    import xml.etree.ElementTree as ET
    import zlib
    from kmzcomet.engine import file_hash
    from kmzcore.runtime.synthetic import write_fixture
    directory.mkdir(parents=True,exist_ok=True)
    ns='{http://psi.hupo.org/ms/mzml}'
    for column,sample in enumerate(result.samples):
        path=directory/(sample.path.stem+'.mzML');write_fixture(path)
        tree=ET.parse(path);spectra=tree.find('.//'+ns+'spectrumList')
        template=deepcopy(spectra[0]);spectra.clear();spectra.set('count','121')
        for index in range(121):
            rt=index*.04;spectrum=deepcopy(template)
            spectrum.set('index',str(index));spectrum.set('id',f'synthetic-{index}');spectrum.set('defaultArrayLength',str(len(result.features)))
            for cv in spectrum.iter(ns+'cvParam'):
                if cv.get('accession')=='MS:1000016':cv.set('value',str(rt))
            arrays=spectrum.find(ns+'binaryDataArrayList')
            values=[(result.measurements[column][i].height or 0.)*math.exp(-.5*((rt-f.rt)/.08)**2)
                    for i,f in enumerate(result.features)]
            for array,data in zip(arrays,([f.mz for f in result.features],values)):
                array.find(ns+'binary').text=base64.b64encode(zlib.compress(struct.pack('<'+'d'*len(data),*data))).decode()
                for cv in array.findall(ns+'cvParam'):
                    if cv.get('accession')=='MS:1000515':
                        cv.set('unitCvRef','MS');cv.set('unitAccession','MS:1000131');cv.set('unitName','number of detector counts')
            spectra.append(spectrum)
        tree.write(path,encoding='utf-8',xml_declaration=True)
        sample.path=path;sample.sha256=file_hash(path);sample.size_bytes=path.stat().st_size
        sample.scans=121;sample.rt_min=0.;sample.rt_max=4.8;sample.modes={'centroid'};sample.native_sample_id=None
        sample.source_files=[dict(path=str(path),name=path.name,size_bytes=sample.size_bytes,sha256=sample.sha256)]


def run_smoke(app,window,output,inputs=()):
    from kmzcomet.native import ENGAGED
    from kmzcore import psi_ms
    output.mkdir(parents=True,exist_ok=True)
    state={'phase':'inputs' if inputs else 'prepare','checks':[]};deadline=time.monotonic()+(300 if len(inputs)>4 else 100)
    timer=QTimer(window);timer.setInterval(20)
    errors=[]
    # The smoke process reports modal errors instead of waiting for a person.
    QMessageBox.warning=lambda *args:errors.append(str(args[2])) or QMessageBox.StandardButton.Ok
    QMessageBox.question=lambda *args:QMessageBox.StandardButton.Yes
    if inputs:window.open_paths(inputs)
    def finish(ok,error=''):
        timer.stop()
        result={'ok':ok,'frozen':bool(getattr(sys,'frozen',False)),'checks':state['checks'],'error':error,
                'native_eic_accelerated':ENGAGED}
        if window.result is not None:
            result.update(features=len(window.result.features),samples=len(window.result.samples),
                          analysis_id=window.result.analysis_id,saved_path=str(window.saved_path))
        (output/'result.json').write_text(json.dumps(result,indent=2))
        window.close()
        def quit_when_closed():
            if window.runtime.closed:app.quit()
            else:QTimer.singleShot(20,quit_when_closed)
        quit_when_closed()
    def tick():
        try:
            if time.monotonic()>deadline:raise TimeoutError('KmzCoMet smoke deadline')
            if getattr(sys,'frozen',False) and not ENGAGED:
                raise RuntimeError('Frozen KmzCoMet is missing its native EIC decoder')
            if getattr(sys,'frozen',False) and psi_ms.vocabulary().version!=psi_ms.VERSION:
                raise RuntimeError('Frozen KmzCoMet is missing its pinned PSI-MS vocabulary')
            if getattr(sys,'frozen',False) and not qualified('Qt','PySide6','pyqtgraph','numpy','pyteomics'):
                raise RuntimeError('Frozen KmzCoMet dependencies are not the qualified versions')
            if errors:raise RuntimeError('; '.join(errors))
            if window.busy:return
            phase=state['phase']
            if phase=='prepare':
                result=example_result(output/'synthetic-inputs')
                def operation(worker):return Completed(result,save_analysis(result,window.history,checkpoint=worker.checkpoint))
                window._start(Worker(operation),window._completed,'analysis');state['phase']='ready'
            elif phase=='inputs':
                assert window.inputs
                window.start_analysis();state['phase']='ready'
            elif phase=='ready':
                assert window.result is not None and window.saved_path.is_file()
                state['checks']+=['background analysis and automatic snapshot','Heights matrix']
                state['original']=payload(window.result);state['saved']=window.saved_path
                assert payload(load_analysis(window.saved_path))==state['original']
                window.grab().save(str(output/'heights.png'))
                if window.result.features:
                    window.bounds[0].setText(str(window.result.features[0].mz+1))
                    assert window.proxy.rowCount()<len(window.result.features)
                    window.clear_filters()
                state['checks'].append('numeric filters')
                assert not window.model.selected_cells and not window.plot_action.isEnabled()
                window.model.setData(window.model.index(0,0),Qt.CheckState.Checked,Qt.ItemDataRole.CheckStateRole)
                window.plot_action.trigger();state['phase']='eics'
            elif phase=='eics':
                from kmzviewer.gui.ms1_spectra import SpectrumPanel
                tab=window.tabs.currentWidget()
                assert len(tab.entries)==len(window.result.samples)
                assert all(entry.data is not None for entry in tab.entries.values())
                assert tab.findChildren(SpectrumPanel)==[] and not tab.plots.inspection_enabled
                bounds=tab.plots.data_x_range();assert bounds is not None and bounds[1]>bounds[0]
                window.grab().save(str(output/'eics-stacked.png'))
                tab.overlay_action.trigger();tab.plots.set_x_range(bounds[0],sum(bounds)/2)
                assert tab.plots.current_x_range()[1]<bounds[1]
                tab.full_action.trigger();assert tab.plots.current_x_range()[1]>=bounds[1]
                window.grab().save(str(output/'eics-overlay.png'))
                state['checks']+=['full-run summed EIC tab','stack/overlay linked zoom and full view','no spectrum inspection']
                window.plot_action.trigger();state['phase']='cached-eics'
            elif phase=='cached-eics':
                tab=window.tabs.currentWidget()
                assert window.tabs.count()==3 and tab.batch.reused==len(window.result.samples) and tab.batch.files_read==0
                assert payload(window.result)==state['original']
                state['checks'].append('independent plot tabs and session cache')
                window.pca_action.trigger();state['phase']='pca'
            elif phase=='pca':
                from PySide6.QtCore import QPoint
                from PySide6.QtTest import QTest
                from PySide6.QtWidgets import QListWidget
                from .pca import PCATab
                tab=window.tabs.currentWidget();assert isinstance(tab,PCATab)
                assert tab.data.feature_count==len(window.result.features) and tab.data.points
                assert payload(window.result)==state['original']
                tab.canvas.draw();window.grab().save(str(output/'pca.png'))
                tab.axes.view_init(elev=40,azim=15);tab.canvas.draw()
                assert tab.axes.elev==40
                tab.reset_button.click();tab.canvas.draw()
                assert tab.axes.elev==24 and tab.axes.azim==-55
                state['pca-row']=tab.data.points[0].row
                point=tab.projected_points()[0]
                QTest.mouseClick(tab.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,
                                 QPoint(round(point[0]),round(point[1])))
                if tab._chooser is not None:
                    listing=tab._chooser.findChild(QListWidget)
                    for i in range(listing.count()):
                        if listing.item(i).data(Qt.ItemDataRole.UserRole)==0:
                            listing.setCurrentRow(i);break
                    tab._chooser.accept()
                state['checks']+=['feature PCA from saved heights with missing-row omission','3D PCA render rotation and full view']
                state['phase']='pca-eics'
            elif phase=='pca-eics':
                from .chromatograms import EICTab
                tab=window.tabs.currentWidget();assert isinstance(tab,EICTab)
                assert len(tab.entries)==len(window.result.samples)
                feature=window.result.features[state['pca-row']]
                assert all(entry.feature_id==feature.id and entry.data is not None for entry in tab.entries.values())
                assert payload(window.result)==state['original']
                state['checks'].append('Command-click PCA opens exact feature all-sample EICs')
                original=QFileDialog.getOpenFileName
                try:
                    QFileDialog.getOpenFileName=lambda *a,**k:(str(state['saved']),'KmzCoMet')
                    window.choose_history()
                finally:QFileDialog.getOpenFileName=original
                state['phase']='reopened'
            elif phase=='reopened':
                assert payload(window.result)==state['original']
                assert window.tabs.count()==1 and not window.model.selected_cells
                state['checks'].append('Past Analyses reopens exact snapshot')
                state['phase']='csv'
                original=QFileDialog.getSaveFileName
                try:
                    QFileDialog.getSaveFileName=lambda *a,**k:(str(output/'heights.csv'),'CSV')
                    window.export('csv')
                finally:QFileDialog.getSaveFileName=original
            elif phase=='csv':
                with (output/'heights.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.reader(handle))
                assert len(rows)==len(window.result.features)+1
                state['checks'].append('complete CSV export')
                state['phase']='excel';original=QFileDialog.getSaveFileName
                try:
                    QFileDialog.getSaveFileName=lambda *a,**k:(str(output/'analysis.xlsx'),'Excel')
                    window.export('xlsx')
                finally:QFileDialog.getSaveFileName=original
            elif phase=='excel':
                from openpyxl import load_workbook
                book=load_workbook(output/'analysis.xlsx',read_only=True)
                assert book.sheetnames==['Heights','Evidence','Measurements','Run']
                rows=list(book['Heights'].iter_rows(values_only=True))
                assert len(rows)==len(window.result.features)+1
                for i,row in enumerate(rows[1:]):assert list(row[4:])==[c[i].height for c in window.result.measurements]
                book.close();assert resource_path('docs/KmzCoMet.html').is_file()
                from .dialogs import DetailsDialog
                details=DetailsDialog(window.result,window);details.show();details.grab().save(str(output/'details.png'));details.close()
                state['checks']+=['full Excel export','secondary result views','bundled manual']
                finish(True)
        except Exception as error:finish(False,str(error))
    timer.timeout.connect(tick);timer.start()
