"""Finish queued Qt and pyqtgraph work before a window paints, so plots paint once."""
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QPainter, QPicture

from kmzcore.qualified import qualified


def finish_pending(view=None, axes=()) -> None:
    """Call on a window's UpdateRequest, before Qt paints the dirty widgets.

    Work queued while Qt is processing events lands behind an already pending
    paint: layouts, a pyqtgraph scene's item updates and deferred widget
    updates would otherwise run right after it, and plots would paint once at
    a stale geometry or for a stale region and then again. Running the same
    work first leaves one paint of the final state. Auto-sized *axes* of the
    view are measured as their paint would, so new tick text resizes the plot
    now rather than after a paint. On Qt or pyqtgraph versions other than those
    qualified this does nothing and the paint does the work, as before.
    """
    # Without a view (CoMet before its first EIC tab) pyqtgraph need not be loaded.
    if not (qualified("Qt", "PySide6") and (view is None or qualified("pyqtgraph"))):
        return
    QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
    if view is not None and view.isVisible():
        for _ in range(4):
            view.scene().prepareForPaint()  # view transforms, as the paint would apply them
            if not _measure(axes):
                break
            QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
        QCoreApplication.sendPostedEvents(view.scene(), QEvent.Type.MetaCall)
        QCoreApplication.sendPostedEvents(view, QEvent.Type.MetaCall)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.UpdateLater)


def _measure(axes) -> bool:
    """Build each pending axis picture by its own paint; True if one changed size."""
    resized = False
    for axis in axes:
        if axis.isVisible() and axis.picture is None:
            before = axis.minimumSize(), axis.maximumSize()
            sink = QPicture()
            painter = QPainter(sink)
            axis.paint(painter, None, None)  # caches the picture the next paint replays
            painter.end()
            resized |= (axis.minimumSize(), axis.maximumSize()) != before
    return resized
