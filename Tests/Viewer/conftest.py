import os
import tempfile

# Reader initialization otherwise fetches PSI-MS over the network for each file.
# Use the dependency's bundled vocabulary for deterministic offline GUI tests.
from psims.controlled_vocabulary.controlled_vocabulary import obo_cache
obo_cache.use_remote = False

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
_matplotlib_config = tempfile.TemporaryDirectory(prefix="kmzviewer-tests-")
os.environ.setdefault("MPLCONFIGDIR", _matplotlib_config.name)

from PySide6.QtCore import QSettings
from pathlib import Path
_state = str(Path(__file__).resolve().parents[2] / ".local/state")
os.environ.setdefault("KMZ_LOCAL_STATE_ROOT", _state)
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _state)
