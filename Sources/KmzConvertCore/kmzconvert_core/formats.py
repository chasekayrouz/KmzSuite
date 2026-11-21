"""Input-format detection for the converter.

SCIEX ``.wiff``/``.wiff2`` entry points use the native WIFF reader; an Agilent
MassHunter ``.d`` directory uses the Agilent reader. Other inputs are refused
with a specific message.
"""
from pathlib import Path

from .binary import FormatError

WIFF = "wiff"
AGILENT = "agilent"


def detect(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".wiff", ".wiff2"):
        return WIFF
    if suffix == ".d":
        from .agilent import is_masshunter
        if not path.is_dir():
            raise FormatError(f"{path.name} is not a folder; an Agilent .d acquisition is a folder")
        if is_masshunter(path):
            return AGILENT
        raise FormatError(f"{path.name} is not an Agilent MassHunter acquisition (no AcqData/MSScan.bin); "
                          "other vendors' .d folders are not supported")
    raise FormatError(f"Input must be a SCIEX .wiff/.wiff2 file or an Agilent .d folder: {path.name}")
