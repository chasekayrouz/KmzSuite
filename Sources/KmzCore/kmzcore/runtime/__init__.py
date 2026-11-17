"""Development application identities and resources; no scientific dependencies."""
from dataclasses import dataclass
from pathlib import Path
import os
import sys


@dataclass(frozen=True)
class Identity:
    key: str
    name: str
    bundle_id: str
    manual: str


IDENTITIES = {
    'ms1': Identity('ms1', 'KmzViewer', 'org.kmzsuite.viewer', 'docs/KmzViewer.html'),
    'ms2': Identity('ms2', 'Kmz2Viewer', 'org.kmzsuite.viewer2', 'docs/Kmz2Viewer.html'),
    'comet': Identity('comet', 'KmzCoMet', 'org.kmzsuite.comet', 'docs/KmzCoMet.html'),
}


def resource_path(name: str) -> Path:
    root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[4]
    return root / name


def configure(identity: Identity) -> None:
    """Run before any matplotlib import, including indirect GUI imports."""
    os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
    if os.environ.get('KMZ_LOCAL_STATE_ROOT'):
        base = Path(os.environ['KMZ_LOCAL_STATE_ROOT'])
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Caches'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    cache = base / identity.bundle_id / 'matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
