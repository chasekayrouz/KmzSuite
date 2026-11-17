"""Launch the desktop app without changing the existing kmzcomet CLI."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(prog='KmzCoMet')
    parser.add_argument('paths',nargs='*',help='WIFF acquisitions or one saved .kmzcomet analysis')
    parser.add_argument('--smoke-output',help=argparse.SUPPRESS)
    parser.add_argument('--smoke-input',action='append',default=[],help=argparse.SUPPRESS)
    parser.add_argument('--version',action='version',version=__import__('kmzcore.version',fromlist=['__version__']).__version__)
    args=parser.parse_args([a for a in sys.argv[1:] if not a.startswith('-psn_')])
    if args.smoke_output:os.environ['KMZ_LOCAL_STATE_ROOT']=str(Path(args.smoke_output)/'state')
    from kmzcore.runtime import IDENTITIES, configure
    configure(IDENTITIES['comet'])
    from kmzcore.runtime.qt import DocumentApplication
    from .app import MainWindow
    app=DocumentApplication(sys.argv,IDENTITIES['comet'])
    window=MainWindow();window.show();app.attach(window,args.paths)
    if args.smoke_output:
        from .smoke import run_smoke
        run_smoke(app,window,Path(args.smoke_output),args.smoke_input)
    return app.exec()


if __name__=='__main__':
    raise SystemExit(main())
