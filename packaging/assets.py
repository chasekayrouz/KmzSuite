"""Generate icons and collect dependency license texts from the pinned environment."""
from importlib.metadata import distributions, version
from pathlib import Path
import shutil
import subprocess
import sysconfig

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication


def generate(root, destination):
    app = QApplication.instance() or QApplication([])
    # The offscreen Qt default "Sans Serif" is an absent macOS family. SVG text
    # uses Helvetica; resolve a real face here without changing viewer fonts.
    if 'Helvetica' not in QFontDatabase.families() or not {'Regular', 'Bold'} <= set(QFontDatabase.styles('Helvetica')):
        raise RuntimeError('Icon generation requires macOS Helvetica Regular and Bold.')
    app.setFont(QFontDatabase.font('Helvetica', 'Regular', app.font().pointSize()))
    destination.mkdir(parents=True, exist_ok=True)
    for key in ('ms1', 'ms2', 'suite', 'convert', 'comet'):
        iconset = destination / f'{key}.iconset'
        iconset.mkdir(exist_ok=True)
        renderer = QSvgRenderer(str(root / 'resources' / f'{key}.svg'))
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                image = QImage(size * scale, size * scale, QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                renderer.render(painter)
                painter.end()
                suffix = '@2x' if scale == 2 else ''
                assert image.save(str(iconset / f'icon_{size}x{size}{suffix}.png'))
        subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(destination / f'{key}.icns')], check=True)
    licenses = destination / 'THIRD_PARTY_LICENSES'
    licenses.mkdir(exist_ok=True)
    for dist in distributions():
        name = dist.metadata['Name']
        folder = licenses / name
        folder.mkdir(exist_ok=True)
        (folder / 'METADATA.txt').write_text(dist.read_text('METADATA') or dist.read_text('PKG-INFO') or name, encoding='utf-8')
        for file in dist.files or []:
            if any(word in file.name.lower() for word in ('license', 'copying', 'copyright', 'notice')):
                source = Path(dist.locate_file(file))
                if source.is_file():
                    # Flatten with the relative path encoded to avoid name collisions.
                    target = folder / str(file).replace('/', '__').replace('..', '_')
                    shutil.copyfile(source, target)
    shutil.copytree(root / 'resources/licenses', licenses / 'Qt-PySide-license-terms', dirs_exist_ok=True)
    # LGPLv3 conveyance: name the exact bundled versions and their complete source.
    from PySide6.QtCore import qVersion
    pyside, qt = version('PySide6'), qVersion()
    (licenses / 'Qt-PySide-license-terms/SOURCE.txt').write_text(
        f'Bundled, unmodified: Qt {qt}, PySide6 {pyside} and shiboken6 {version("shiboken6")}\n'
        '(Qt for Python), as published on PyPI by The Qt Company.\n\n'
        'Complete corresponding source code:\n'
        f'Qt {qt}: https://download.qt.io/archive/qt/{qt.rsplit(".", 1)[0]}/{qt}/single/\n'
        f'Qt for Python {pyside}: https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-{pyside}-src/\n'
        f'Qt for Python Git tag: https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v{pyside}\n', encoding='utf-8')
    stdlib = Path(sysconfig.get_path('stdlib'))
    source = stdlib / 'LICENSE.txt'
    if not source.is_file():
        raise RuntimeError(f'CPython license not found at {source}')
    shutil.copyfile(source, licenses / 'CPython-LICENSE.txt')
