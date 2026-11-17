"""Scriptable analysis; the desktop interface has a separate entry point."""
import argparse
from dataclasses import fields
import os
from pathlib import Path
import sys

from . import __version__
from .engine import analyze
from .model import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="KmzCoMet: comparative MS1 ion features to Excel")
    parser.add_argument("inputs", nargs="+", type=Path, help="mzML or qualified single-sample WIFF inputs; first is the RT reference")
    parser.add_argument("-o", "--output", required=True, type=Path, help="New .xlsx destination (never overwritten)")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--quiet", action="store_true")
    default = Settings()
    explanations = {
        "mz_ppm": "m/z matching/extraction half-tolerance in ppm",
        "rt_tolerance": "maximum pairwise aligned apex difference in minutes",
        "rt_window": "extraction half-window in minutes, clipped at neighboring feature midpoints",
        "discovery_window": "rise/fall search half-window in minutes (at least rt_window); does not widen extraction",
        "discovery_height": "minimum native spectral height for discovery (not a quantification cutoff)",
        "min_height": "minimum native height for supported evidence (0 disables; all measurements retained)",
        "min_points": "minimum supporting scans",
        "min_snr": "minimum heuristic SNR for supported evidence",
        "min_contrast": "minimum height/background contrast",
        "max_scan_gap": "maximum gap in minutes for tracks and area integration",
        "max_rt_shift": "largest RT shift searched, in minutes",
        "min_anchors": "minimum reciprocal unique mass matches for RT shift",
        "alignment": "constant RT shift or no correction",
    }
    for field in fields(default):
        value = getattr(default, field.name)
        kwargs = {"default":value, "help":f"{explanations[field.name]} (default: {value})"}
        if field.name == "alignment":
            kwargs["choices"] = ["shift", "none"]
        else:
            kwargs["type"] = type(value)
        parser.add_argument("--"+field.name.replace("_", "-"), **kwargs)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() != ".xlsx":
            raise ValueError("Output must have an .xlsx extension")
        if os.path.lexists(args.output):
            raise FileExistsError("Output already exists; choose a new .xlsx destination")
        from .excel import write_workbook
        settings = Settings(**{field.name:getattr(args, field.name) for field in fields(default)})
        progress = (lambda message: None) if args.quiet else (lambda message: print(message, file=sys.stderr, flush=True))
        result = analyze(args.inputs, settings, progress=progress)
        write_workbook(result, args.output)
        for sample in result.samples:
            for warning in sample.warnings:
                print(f"{sample.id}: {warning}", file=sys.stderr)
        print(f"Wrote {len(result.features)} ion features × {len(result.samples)} files to {args.output}")
        return 0
    except KeyboardInterrupt:
        print("Cancelled; no incomplete workbook published.", file=sys.stderr)
        return 130
    except ImportError as error:
        print(f"KmzCoMet dependency unavailable: {error}. Run the repo environment setup.", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"KmzCoMet: {error}", file=sys.stderr)
        return 2
