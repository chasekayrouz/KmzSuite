import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from kmzconvert_core import __version__
from kmzconvert_core.binary import FormatError
from kmzconvert_core.formats import AGILENT, detect
from kmzconvert_core.native import NativeWiffReader as WiffReader
from kmzconvert_core.agilent import AgilentReader
from kmzconvert_core.samples import list_samples, sample_output_path
from kmzconvert_core.writer import LIMITATIONS, write_mzml
from kmzconvert_core.mass_correction import parse_references, estimate_mass_correction, default_output_name
from kmzcore.mass_correction import WARNING
from kmzcore.sparse_profile import output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental native converter of SCIEX X500 WIFF and Agilent MassHunter .d acquisitions to mzML")
    parser.add_argument("input", type=Path, help=".wiff2 or .wiff entry point (matching .wiff and .wiff.scan must be present), "
                                                 "or an Agilent MassHunter .d folder")
    parser.add_argument("-o", "--output", type=Path, help="new output file (default: ./<input>.mzML or .sp.mzML; beta correction adds .mc; never overwritten)")
    parser.add_argument("--profile", choices=("full", "sparse"), default="full",
                        help="Full Profile (default) or Sparse Profile: smaller, lossy MS1 inspection mzML")
    parser.add_argument("--inspect", action="store_true", help="print decoded layout and metadata without conversion")
    parser.add_argument("--list-samples", action="store_true", help="print WIFF sample identifiers as JSON without conversion (an Agilent .d is one sample)")
    parser.add_argument("--sample", type=int, metavar="ID", help="convert/inspect one WIFF metadata sample ID; required for multi-sample WIFF")
    parser.add_argument("--mass-reference", action="append", default=[], metavar="MZ;RT_MINUTES",
                        help="enable beta global MS1 ppm correction; repeat expected_mz;RT_minutes for each standard")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    references = []
    if args.mass_reference and not args.inspect and not args.list_samples:
        try:
            references = parse_references(args.mass_reference)
        except ValueError as error:
            parser.error(str(error))
    started = time.monotonic()
    reader = None
    try:
        agilent = detect(args.input) == AGILENT
        if args.list_samples:
            ids = [1] if agilent else [s.id for s in list_samples(args.input)]
            print(json.dumps({"samples": [{"id": i} for i in ids]}))
            return 0
        if agilent:
            if args.sample is not None:
                raise ValueError("An Agilent .d acquisition holds one run; omit --sample")
            if not args.inspect and args.profile == "sparse":
                raise ValueError("Sparse Profile is not yet qualified for Agilent data; use Full Profile")
            if references:
                raise ValueError("Beta mass correction is not yet qualified for Agilent data; omit --mass-reference")
            reader = AgilentReader(args.input)
            if args.inspect:
                print(json.dumps(reader.inspect(), indent=2))
                return 0
        else:
            reader = WiffReader(args.input, sample_id=args.sample) if args.sample is not None else WiffReader(args.input)
        if hasattr(reader, "dad_status"):
            prefix = "warning: " if reader.dad_status == "unreadable" else ""
            print(f"kmzconvert: {prefix}{reader.dad_message}", file=sys.stderr)
        if reader.calibration_resolutions:
            print(f"kmzconvert: calibration: using the WIFF slot table for "
                  f"{len(reader.calibration_resolutions)} startup spectra; "
                  "constant startup table, single stale scan value and "
                  "following-cycle agreement verified exactly "
                  "(--inspect lists the scans).", file=sys.stderr)
        if args.inspect:
            print(json.dumps({"sample_id": getattr(reader, 'sample_id', 1),
                              "spectra": len(reader.scans), "cycle_slots": len(reader.index),
                              "ms_levels": dict(Counter(reader.experiments[s.experiment-1].ms_level for s in reader.scans)),
                              "experiments": [asdict(e) for e in reader.experiments],
                              "instrument": reader.instrument, "software": reader.acquisition_software,
                              "tdc_interval": reader.tdc, "checkpoints": reader.checkpoint_count,
                              "calibration_resolutions": reader.calibration_resolutions,
                              "dad": {"status": getattr(reader, "dad_status", "absent"),
                                      "spectra": len(reader.dad) if getattr(reader, "dad", None) is not None else 0},
                              "limitations": LIMITATIONS}, indent=2))
            return 0
        output = output_path(args.output or Path.cwd() / default_output_name(args.input, args.profile, corrected=bool(references)), args.profile)
        if args.sample is not None:
            output = sample_output_path(output, reader.sample_id)
        print(getattr(reader, "limitations", LIMITATIONS), file=sys.stderr)
        correction = None
        if references:
            print(WARNING, file=sys.stderr, flush=True)
            # Refuse existing outputs before potentially expensive estimation;
            # the writer also enforces race-safe no-clobber publication.
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {output}")
            correction = estimate_mass_correction(reader, references, log=lambda text: print(text, file=sys.stderr, flush=True))
        def progress(done, total):
            print(f"\rConverting {done}/{total} spectra", end="", file=sys.stderr, flush=True)
        destination = (write_mzml(reader, output, progress, profile=args.profile, mass_correction=correction) if correction is not None
                       else write_mzml(reader, output, progress) if args.profile == "full"
                       else write_mzml(reader, output, progress, profile=args.profile))
        print(file=sys.stderr)
        if reader.nominal_window_overhangs:
            print(f"kmzconvert: warning: {len(reader.nominal_window_overhangs)} spectra contain measured "
                  "masses outside nominal method windows; source-calibrated values were retained.",
                  file=sys.stderr)
        dad_summary = f", {len(reader.dad)} DAD spectra" if getattr(reader, 'dad', None) is not None else ''
        print(f"Wrote {destination} ({len(reader.scans)} spectra{dad_summary}, {time.monotonic()-started:.1f} s)")
        return 0
    except KeyboardInterrupt:
        print("\nkmzconvert: cancelled", file=sys.stderr)
        return 130
    except (FormatError, OSError, ValueError, KeyError) as e:
        print(f"kmzconvert: {e}", file=sys.stderr)
        for note in getattr(e, "__notes__", ()):
            print(f"kmzconvert: {note}", file=sys.stderr)
        return 1
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
