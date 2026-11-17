#!/usr/bin/env python3
"""Run isolated four-reference WIFF mass-correction qualification (never GUI)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.benchmarks.mass_calibration.qualification import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path,help="WIFF entry with matching .wiff.scan")
    parser.add_argument("--references",type=Path,required=True,help="Local JSON containing references and optional search settings")
    parser.add_argument("--output",type=Path,required=True,help="New directory under ignored .local/")
    parser.add_argument("--experimental-mzml",action="store_true",help="Write and verify explicitly labelled MS1-only profile copies")
    args = parser.parse_args()
    result = run(args.input,args.references,args.output,args.experimental_mzml)
    print(result["status"])
    print(args.output/"report.html")
    if "failure" in result:
        print(result["failure"],file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
