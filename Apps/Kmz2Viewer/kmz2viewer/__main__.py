"""Launch Kmz2Viewer, optionally opening local mzML paths."""
from kmzcore.runtime.launch import launch


def main() -> None:
    raise SystemExit(launch('ms2'))


if __name__ == '__main__':
    main()
