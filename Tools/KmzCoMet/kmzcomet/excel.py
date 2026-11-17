"""Portable Excel values with explicit evidence and atomic no-clobber delivery."""
from dataclasses import asdict, fields
import os
from pathlib import Path

from . import METHOD, __version__
from .model import Measurement
from .output import new_output


DEFINITIONS = {
    "Peak height": "Largest native intensity in the m/z–RT extraction window. No normalization or imputation. Not a probability.",
    "Feature": "An MS1 ion feature, not an identified metabolite. Isotopes and adducts remain separate. Positive, negative and unknown polarities stay separate.",
    "RT": "Feature RT is in minutes on the first input's reference axis. Apex RT in Measurements is native to each file. Native RT minus fitted shift equals aligned RT.",
    "Extraction": "Every feature is re-extracted from every file, independently of initial detection. Maximum native point within ±mz_ppm per scan. Profile heights and centroid heights are analyzed separately.",
    "Window area": "Trapezoidal area of max(EIC minus 20th-percentile background, 0), in intensity × minute. No integration across gaps exceeding max_scan_gap. This is window area, not a fitted peak area.",
    "Noise and SNR": "Noise = max(1 native intensity unit, 1.4826 × MAD of first differences / sqrt(2)). SNR = (height minus background) / noise. Heuristic evidence, not calibrated detection probability.",
    "supported": "Internal apex within rt_tolerance of the aligned feature RT, sufficient consecutive points above background + max(2×noise, 10% of corrected height), and configured height, SNR and contrast. Does not prove chemical identity.",
    "weak_signal": "Signal was measured but does not meet the shape/support criteria. Keep its numeric height; do not interpret as confirmed absence.",
    "no_signal": "All measured intensities in a sufficiently covered window are zero. This is no observed signal at these settings, not proof of biological absence.",
    "not_acquired": "No same-polarity MS1 scan covers the requested mass/time window. Numeric results are blank, not zero.",
    "insufficient_coverage": "Partial time window, scan gap, too few scans or unknown acquisition mass range. Available measurements remain visible but absence is not assessed.",
    "ambiguous": "Coeluting feature windows overlap in mass. Numeric heights may contain shared signal; they are not independent assignments.",
    "Limits": "Exploratory baseline. Constant RT shifts only. No nonlinear warping, blank subtraction, normalization, isotope/adduct collapse, MS2 identification, hypothesis tests or calibrated presence probabilities.",
    "Alignment failure": "Insufficient consistent anchors leaves shift zero with a sample warning. Inspect warnings before interpreting missing/weak features.",
    "Discovery": "Discovery uses spectral height and chromatographic contrast; profile maxima and chromatographic maxima on a track require a half-height valley. The rise/fall search uses max(discovery_window, rt_window). A feature below discovery in every input cannot be recovered. Discovered indicates membership, not measurement success.",
    "Evidence height floor": "min_height is an optional native-intensity floor for supported evidence, separate from discovery_height. Its default 0 disables the floor. Values below it remain numeric and are not absence calls; qualify the floor for the acquisition and units.",
}


def write_workbook(result, destination, *, checkpoint=lambda: None):
    checkpoint()
    destination = Path(destination)
    if destination.suffix.lower() != ".xlsx":
        raise ValueError("Output must have an .xlsx extension")
    if os.path.lexists(destination):
        raise FileExistsError(f"Output already exists: {destination.name}")
    if len(result.features)*len(result.samples)+1 > 1048576 or len(result.samples)+4 > 16384:
        raise ValueError("Result exceeds Excel worksheet limits")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # openpyxl takes ~0.1 s to import; the desktop window does not need it to start.
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.styles.cell_style import StyleArray
    from openpyxl.utils import get_column_letter
    book = Workbook(write_only=True)
    book.properties.creator = "KmzCoMet"
    book.properties.title = "Comparative MS1 ion features"
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Arial", size=10, color="172234")
    header_fill = PatternFill("solid", fgColor="243B57")

    def sheet(name, headers, widths, freeze):
        ws = book.create_sheet(name)
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = freeze
        ws.row_dimensions[1].height = 32
        for i, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
        cells = []
        for value in headers:
            cell = WriteOnlyCell(ws, value=value)
            cell.data_type = "s"  # Untrusted filenames must never become formulas.
            cell.font, cell.fill = header_font, header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cells.append(cell)
        ws.append(cells)
        return ws

    # Each distinct body style registers once, at its first use as before; later
    # cells copy its style indices instead of re-hashing equal Font/Alignment
    # objects, so the workbook's style tables and cells are unchanged.
    styles = {}

    def append(ws, values, formats=None, *, wrap=False):
        cells = []
        for i, value in enumerate(values):
            cell = WriteOnlyCell(ws, value=value)
            if isinstance(value, str):
                cell.data_type = "s"
            numeric = isinstance(value, (float, int))
            number_format = formats[i] if formats and i < len(formats) else None
            key = (wrap, numeric, number_format)
            if cell._style is None and key in styles:
                cell._style = StyleArray(styles[key])
            else:
                fresh = cell._style is None
                cell.font = body_font
                cell.alignment = Alignment(vertical="center", wrap_text=wrap,
                                           horizontal="right" if numeric else "left")
                if number_format is not None:
                    cell.number_format = number_format
                if fresh:
                    styles[key] = StyleArray(cell._style)
            cells.append(cell)
        ws.append(cells)

    headers = ["Feature ID", "m/z", "RT (min)", "Polarity"] + [s.id for s in result.samples]
    widths = [16, 16, 15, 13] + [20]*len(result.samples)
    height = sheet("Heights", headers, widths, "E2")
    evidence = sheet("Evidence", headers, widths[:4]+[25]*len(result.samples), "E2")
    height.sheet_properties.tabColor = "243B57"
    for i, feature in enumerate(result.features):
        common = [feature.id, feature.mz, feature.rt, {1:"positive",-1:"negative",0:"unknown"}[feature.polarity]]
        append(height, common+[column[i].height for column in result.measurements],
               ["@", "0.000000", "0.0000", "@"]+["0.00E+00"]*len(result.samples))
        append(evidence, common+[column[i].status for column in result.measurements],
               ["@", "0.000000", "0.0000", "@"])
    for ws in (height, evidence):
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(result.features)+1}"
    names = [f.name for f in fields(Measurement)]
    labels = ["Feature ID", "Sample ID", "Discovered", "Evidence", "Height", "Height above background",
              "Window area (intensity × min)", "Native apex RT (min)", "Apex m/z", "Background", "Noise", "SNR",
              "Contrast", "Scans", "Support scans", "Flags"]
    details = sheet("Measurements", labels, [16,14,14,25]+[20]*9+[12,15,48], "C2")
    formats = ["@", "@", "General", "@", "0.00E+00", "0.00E+00", "0.00E+00",
               "0.0000", "0.000000", "0.00E+00", "0.00E+00", "0.0", "0.0", "0", "0", "@"]
    for column in result.measurements:
        for measurement in column:
            append(details, [getattr(measurement, name) for name in names], formats)
    details.auto_filter.ref = f"A1:P{len(result.features)*len(result.samples)+1}"
    run = sheet("Run", ["Record", "Field", "Value"], [16,29,110], "A2")
    rows = [("Analysis", "Method", result.method), ("Analysis", "Suite version", result.suite_version or __version__),
            ("Analysis", "Created UTC", result.created_utc),
            ("Analysis", "Analysis ID", result.analysis_id),
            ("Analysis", "Elapsed seconds", result.elapsed_seconds),
            ("Analysis", "Reference sample", result.samples[0].id),
            ("Analysis", "Feature count", len(result.features))]
    rows.extend(("Settings", name, value) for name, value in asdict(result.settings).items())
    for sample in result.samples:
        for name, value in [("Filename", sample.path.name), ("SHA-256", sample.sha256),
                            ("Size bytes", sample.size_bytes), ("MS1 scans", sample.scans),
                            ("Representation", ", ".join(sorted(sample.modes))),
                            ("Intensity unit", ", ".join(sorted(sample.units))),
                            ("RT start (min)", sample.rt_min), ("RT end (min)", sample.rt_max),
                            ("Discovery candidates", len(sample.candidates)),
                            ("Warnings", "; ".join(sample.warnings) or "None")]:
            rows.append((sample.id, name, value))
        if sample.native_sample_id is not None:
            rows.append((sample.id, 'Stored WIFF sample ID', sample.native_sample_id))
        for source in sample.source_files:
            if source['path'] == str(sample.path):
                continue  # Primary filename/size/digest are already recorded.
            rows.append((sample.id, 'Source file', source['name']))
            rows.append((sample.id, 'Source SHA-256', source['sha256']))
            rows.append((sample.id, 'Source bytes', source['size_bytes']))
        for polarity, shift in sorted(sample.shifts.items()):
            rows.extend([(sample.id, f"RT shift, polarity {polarity} (min)", shift),
                         (sample.id, f"Anchors, polarity {polarity}", sample.anchors[polarity]),
                         (sample.id, f"Anchor MAD, polarity {polarity} (min)", sample.alignment_mad[polarity])])
    rows.extend(("Definitions", name, text) for name, text in DEFINITIONS.items())
    for number, row in enumerate(rows, 2):
        run.row_dimensions[number].height = max(30, 15*(1+len(str(row[2]))//100))
        append(run, row, wrap=True)
    try:
        with new_output(destination, checkpoint=checkpoint) as temporary:
            book.save(temporary)
    finally:
        book.close()
    return destination
