"""Review repro: integrated-6's EICTab import remains lazy at repaint time."""
import sys
import pytest
from PySide6.QtCore import QEvent,QObject
from PySide6.QtWidgets import QApplication
from kmzcomet_app.app import MainWindow

def test_initial_heights_update_dispatches_without_import_error(tmp_path):
    app=QApplication.instance() or QApplication([])
    window=MainWindow(history=tmp_path/'history')
    try:
        # QWidget must be allowed to handle every ordinary repaint request.
        window.event(QEvent(QEvent.Type.UpdateRequest))
    finally:
        window.close();window.deleteLater();app.processEvents()

def test_visible_heights_delivers_normal_paints(tmp_path):
    app=QApplication.instance() or QApplication([])
    window=MainWindow(history=tmp_path/'history')
    class Counter(QObject):
        paints=0
        def eventFilter(self,obj,event):
            if event.type()==QEvent.Type.Paint:self.paints+=1
            return False
    counter=Counter();window.table.viewport().installEventFilter(counter)
    try:
        window.show()
        for _ in range(10):app.processEvents()
        counter.paints=0
        window.table.viewport().update()
        for _ in range(10):app.processEvents()
        assert counter.paints>0,'No requested Heights paint delivered; QWidget.grab() would hide this failure'
    finally:
        window.close();window.deleteLater();app.processEvents()
