from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
from kmzcore.version import LICENSE_NOTICE, MINIMUM_MACOS, __version__

root = Path(SPECPATH).parent
key = os.environ['KMZ_BUILD_APP']
generated = Path(os.environ['KMZ_BUILD_ASSETS'])
name = 'KmzViewer' if key == 'ms1' else 'Kmz2Viewer'
identifier = 'org.kmzsuite.viewer' if key == 'ms1' else 'org.kmzsuite.viewer2'
entry = root / 'packaging' / (name + '.py')
data = [(str(root / 'docs' / (name + '.html')), 'docs'),
        (str(root / 'LICENSE'), 'resources'),
        (str(root / 'resources/THIRD_PARTY_NOTICES.txt'), 'resources'),
        (str(generated / 'THIRD_PARTY_LICENSES'), 'resources/THIRD_PARTY_LICENSES')]
data += collect_data_files('psims') + collect_data_files('pyteomics')
data += collect_data_files('kmzcore', includes=['data/*'])  # The pinned PSI-MS vocabulary.
if key == 'ms1':
    data += collect_data_files('kmzviewer', includes=['data/*'])
    data += copy_metadata('ms-entropy')
    data += copy_metadata('h5py')
for package in ('pyteomics', 'psims', 'numpy', 'matplotlib', 'pyqtgraph'):
    data += copy_metadata(package)
paths = [root / folder for folder in ('Apps/KmzViewer', 'Apps/Kmz2Viewer', 'Sources/KmzCore')]
a = Analysis([str(entry)], pathex=[str(p) for p in paths], datas=data,
             hiddenimports=['pyteomics.mzml', 'lxml.etree', 'matplotlib.backends.backend_svg']
                           + (['kmzviewer._native'] if key == 'ms1' else []),
             excludes=['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'pytest', 'IPython'],
             hooksconfig={'matplotlib': {'backends': ['Agg', 'svg']}}, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=name, console=False,
          argv_emulation=False, target_arch='arm64', codesign_identity=None)
coll = COLLECT(exe, a.binaries, a.datas, name=name)
app = BUNDLE(coll, name=name + '.app', icon=str(generated / (key + '.icns')),
             bundle_identifier=identifier,
             info_plist={
                 'CFBundleShortVersionString': __version__, 'CFBundleVersion': __version__,
                 'NSHighResolutionCapable': True, 'LSMinimumSystemVersion': MINIMUM_MACOS,
                 'NSHumanReadableCopyright': LICENSE_NOTICE,
                 'CFBundleDocumentTypes': [{
                     'CFBundleTypeName': 'mzML mass spectrometry data',
                     'CFBundleTypeExtensions': ['mzML', 'mzml'],
                     'CFBundleTypeRole': 'Viewer', 'LSHandlerRank': 'Alternate'
                 }]
             })
