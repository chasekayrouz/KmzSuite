"""Shared startup only; the viewers retain independent windows and engines."""
import sys
import argparse
from kmzcore.runtime import IDENTITIES, configure
from kmzcore.runtime.handoff import precursor_mass


def startup_arguments(key, argv):
    parser = argparse.ArgumentParser(prog=IDENTITIES[key].name)
    if key == 'ms2':
        parser.add_argument('--precursor-mz', type=precursor_mass,
                            help='Initialize a precursor-only search with ±20 ppm tolerance.')
    parser.add_argument('--smoke-output', help=argparse.SUPPRESS)
    parser.add_argument('--smoke-external', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--smoke-lifecycle', nargs=2, help=argparse.SUPPRESS)
    parser.add_argument('--smoke-handoff', choices=('parent', 'child'), help=argparse.SUPPRESS)
    parser.add_argument('--smoke-gnps-live', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('paths', nargs='*', help='Local mzML files')
    return parser.parse_args([arg for arg in argv if not arg.startswith('-psn_')])


def launch(key):
    identity = IDENTITIES[key]
    arguments = startup_arguments(key, sys.argv[1:])
    if '--smoke-output' in sys.argv:
        import os
        from pathlib import Path
        output = Path(sys.argv[sys.argv.index('--smoke-output') + 1])
        os.environ['KMZ_LOCAL_STATE_ROOT'] = str(output / 'state')
    configure(identity)
    from kmzcore.runtime.qt import DocumentApplication
    app = DocumentApplication(sys.argv, identity)
    if key == 'ms1':
        from kmzviewer.gui.app import MainWindow
    else:
        from kmz2viewer.gui.app import MainWindow
    if '--smoke-output' in sys.argv:
        from PySide6.QtCore import Qt
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    window = MainWindow(precursor_mz=arguments.precursor_mz) if key == 'ms2' else MainWindow()
    window.show()
    if '--smoke-output' in sys.argv:
        from kmzcore.runtime.smoke import run_smoke
        output = sys.argv[sys.argv.index('--smoke-output') + 1]
        if arguments.smoke_gnps_live:
            if key != 'ms1':
                raise ValueError('The GNPS diagnostic belongs to KmzViewer.')
            from kmzviewer.gui.gnps_smoke import run_live
            run_live(app, window, output)
        elif arguments.smoke_handoff:
            from kmzcore.runtime.handoff_smoke import run_handoff
            run_handoff(app, window, key, output, arguments.smoke_handoff)
        elif '--smoke-lifecycle' in sys.argv:
            from kmzcore.runtime.lifecycle_smoke import run_lifecycle
            index = sys.argv.index('--smoke-lifecycle')
            run_lifecycle(app, window, key, output, sys.argv[index + 1], sys.argv[index + 2])
        else:
            run_smoke(app, window, key, output)
        app.attach(window, arguments.paths)
    else:
        app.attach(window, arguments.paths)
    return app.exec()
