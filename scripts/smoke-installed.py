#!/usr/bin/env python3
"""Check durable apps without importing development packages or dependencies."""
import argparse
from pathlib import Path
from installed_apps import installed_directory, isolated_smoke, smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution', type=Path, default=installed_directory())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Optional local acquisition for GUI inspect')
    parser.add_argument('--isolate-checkout', type=Path, help='Temporarily hide checkout runtime resources; output must be outside it')
    args = parser.parse_args()
    if args.isolate_checkout:
        isolated_smoke(args.distribution, args.isolate_checkout, args.output)
    else:
        smoke(args.distribution, args.output, args.input)
    print('All five installed apps passed independent LaunchServices smoke checks.')


if __name__ == '__main__':
    main()
