"""Regression: fallback picture replays must each start at the same transform."""
import os
import pytest
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QBuffer, QIODevice
from PySide6.QtGui import QImage,QPainter,QColor,QFont
from PySide6.QtWidgets import QApplication
from PySide6.QtSvg import QSvgGenerator,QSvgRenderer
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtPdf import QPdfDocument
from PySide6.QtCore import QSize
from tempfile import TemporaryDirectory
from kmzviewer.gui.axes import GridAxis

@pytest.fixture(scope='module')
def app():
    app=QApplication.instance() or QApplication([])
    app.setFont(QFont('Helvetica',11));app.setStyle('Fusion')
    return app

def make(app,fast):
    pg.setConfigOptions(background='w',foreground='#243349')
    v=pg.GraphicsLayoutWidget();v.resize(620,420)
    p=pg.PlotItem(axisItems={s:GridAxis(s) for s in ('left','bottom')} if fast else None,enableMenu=False)
    v.ci.addItem(p);p.showGrid(x=True,y=True,alpha=.16)
    p.setLabel('left','Counts');p.setLabel('bottom','Retention time (min)')
    p.plot(np.linspace(0,10,100),np.sin(np.linspace(0,10,100))+2)
    p.setXRange(0,10,padding=0);p.setYRange(0,4,padding=0);v.show()
    for _ in range(3):app.processEvents();v.grab()
    return v

def render(v,dpi):
    im=QImage(v.size(),QImage.Format.Format_ARGB32_Premultiplied);im.fill(QColor('white'))
    if isinstance(dpi,str) and dpi.startswith('pdf'):
        with TemporaryDirectory() as temp:
            fn=temp+'/print.pdf';dev=QPrinter();dev.setOutputFormat(QPrinter.PdfFormat);dev.setResolution(int(dpi[3:]));dev.setOutputFileName(fn)
            p=QPainter(dev);v.scene().render(p,QRectF(v.rect()),v.sceneRect());p.end()
            doc=QPdfDocument();assert doc.load(fn)==QPdfDocument.Error.None_
            im=doc.render(0,QSize(794,1123)).convertToFormat(QImage.Format.Format_ARGB32_Premultiplied);doc.close()
    elif dpi=='svg':
        b=QBuffer();b.open(QIODevice.WriteOnly);dev=QSvgGenerator();dev.setOutputDevice(b);dev.setSize(v.size());dev.setViewBox(v.rect())
        p=QPainter(dev);v.scene().render(p,QRectF(v.rect()),v.sceneRect());p.end()
        p=QPainter(im);QSvgRenderer(bytes(b.data())).render(p);p.end()
    else:
        im.setDotsPerMeterX(round(dpi/.0254));im.setDotsPerMeterY(round(dpi/.0254))
        p=QPainter(im);v.scene().render(p,QRectF(v.rect()),v.sceneRect());p.end()
    return im

@pytest.mark.parametrize('dpi',[72,96,144,300,'svg','pdf144','pdf300'])
def test_fallback_preserves_axisitem_pixels(app,dpi,monkeypatch):
    if os.environ.get('REVIEW_TRY_FIX'):
        import kmzviewer.gui.axes as axes
        def paint(self,p,opt,widget):
            if self.picture is None:self._record()
            if self._lines is not None:
                rects=axes._exposed_rects(p,self.boundingRect())
                if rects is None:
                    p.save();self._lines.play(p);p.restore()
                for rect in rects or ():
                    world=p.worldTransform();p.save();p.setWorldTransform(axes.QTransform());p.setClipRect(rect);p.setWorldTransform(world);self._lines.play(p);p.restore()
            self.picture.play(p)
        monkeypatch.setattr(GridAxis,'paint',paint)
    vv=[make(app,fast) for fast in (True,False)]
    try:
        ims=[render(v,dpi) for v in vv]
        a,b=(np.frombuffer(i.constBits(),np.uint8).reshape(i.height(),i.width(),4) for i in ims)
        changed=int(np.any(a!=b,axis=2).sum())
        assert changed==0,f'{dpi}: {changed} changed pixels'
    finally:
        for v in vv:v.close();v.deleteLater()
        app.processEvents()
