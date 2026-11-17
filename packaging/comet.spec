from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
from kmzcore.version import LICENSE_NOTICE, MINIMUM_MACOS, __version__

root = Path(SPECPATH).parent
generated = Path(os.environ['KMZ_BUILD_ASSETS'])
data = [(str(root/'docs/KmzCoMet.html'),'docs'),
        (str(root/'LICENSE'),'resources'),
        (str(root/'resources/THIRD_PARTY_NOTICES.txt'),'resources'),
        (str(generated/'THIRD_PARTY_LICENSES'),'resources/THIRD_PARTY_LICENSES')]
data += collect_data_files('psims') + collect_data_files('pyteomics')
data += collect_data_files('kmzcore',includes=['data/*'])
for package in ('pyteomics','psims','numpy','openpyxl','olefile','pyqtgraph','matplotlib'):
    data += copy_metadata(package)
paths = [root/folder for folder in ('Apps/KmzCoMet','Apps/KmzViewer','Tools/KmzCoMet','Sources/KmzCore','Sources/KmzConvertCore')]
a = Analysis([str(root/'packaging/KmzCoMet.py')],pathex=[str(p) for p in paths],datas=data,
             hiddenimports=['pyteomics.mzml','lxml.etree','kmzcomet_app.smoke','kmzconvert_core._native'],
             excludes=['tkinter','PyQt5','PyQt6','PySide2','pytest','IPython'],
             hooksconfig={'matplotlib': {'backends': ['QtAgg']}},noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,[],exclude_binaries=True,name='KmzCoMet',console=False,
          argv_emulation=False,target_arch='arm64',codesign_identity=None)
coll = COLLECT(exe,a.binaries,a.datas,name='KmzCoMet')
app = BUNDLE(coll,name='KmzCoMet.app',icon=str(generated/'comet.icns'),bundle_identifier='org.kmzsuite.comet',
             info_plist={'CFBundleShortVersionString':__version__,'CFBundleVersion':__version__,
                         'NSHighResolutionCapable':True,'LSMinimumSystemVersion':MINIMUM_MACOS,
                         'NSHumanReadableCopyright':LICENSE_NOTICE,
                         'CFBundleDocumentTypes':[
                             {'CFBundleTypeName':'SCIEX acquisition','CFBundleTypeExtensions':['wiff','wiff2'],
                              'CFBundleTypeRole':'Viewer','LSHandlerRank':'Alternate'},
                             {'CFBundleTypeName':'KmzCoMet saved analysis','CFBundleTypeExtensions':['kmzcomet'],
                              'CFBundleTypeRole':'Editor','LSHandlerRank':'Owner'}]})
