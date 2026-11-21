from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata

root = Path(SPECPATH).parent
a = Analysis([str(root / 'packaging/convert_entry.py')],
             pathex=[str(root / 'Sources/KmzConvertCore'), str(root / 'Sources/KmzCore'), str(root / 'Tools/KmzConvert')],
             datas=copy_metadata('numpy') + copy_metadata('olefile'),
             hiddenimports=['kmzconvert_core._native', 'kmzconvert_core._agilent'],
             excludes=['PySide6', 'matplotlib', 'pytest', 'tkinter', 'IPython'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='kmzconvert', console=True,
          target_arch='arm64', codesign_identity=None)
coll = COLLECT(exe, a.binaries, a.datas, name='kmzconvert')
