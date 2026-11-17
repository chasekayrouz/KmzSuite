"""Launch KmzViewer, optionally opening local mzML paths."""
from kmzcore.runtime.launch import launch


def main() -> None:
    raise SystemExit(launch('ms1'))


if __name__ == '__main__':
    main()
