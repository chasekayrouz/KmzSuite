# KmzSuite user guide

This guide walks through each app step by step. To install KmzSuite, follow the
[README](../README.md#installation). Each app also has a detailed manual under its
**Help** menu.

- [Before you start](#before-you-start)
- [Convert acquisitions: KmzConvert](#convert-acquisitions-kmzconvert)
- [View chromatograms and spectra: KmzViewer](#view-chromatograms-and-spectra-kmzviewer)
- [Browse MS/MS scans: Kmz2Viewer](#browse-msms-scans-kmz2viewer)
- [Compare runs: KmzCoMet](#compare-runs-kmzcomet)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Before you start

**Opening an app.** Press **Cmd+Space**, type the app name and press **Return**. Or
open **KmzSuite** and click **Open** next to any tool.

**Your files.**

- A SCIEX acquisition is a `.wiff` file plus a `.wiff.scan` file with the same
  name. Keep both in the same folder. A `.wiff2` file works as long as those two
  files are beside it.
- An Agilent MassHunter acquisition is a folder ending in `.d`. Use the whole
  folder; don't pick files inside it.
- The viewers open `.mzML` files, both those made by KmzConvert and standard mzML
  from other software.
- KmzSuite never changes your original files.

**A typical session.**

1. Convert acquisitions to mzML with **KmzConvert**.
2. Inspect chromatograms and spectra in **KmzViewer**.
3. Look at MS/MS scans for a mass of interest in **Kmz2Viewer**.
4. Compare features across many runs with **KmzCoMet**, which reads WIFF files
   directly.

## Convert acquisitions: KmzConvert

1. Open **KmzConvert**.
2. Click **Add Files** and choose one or more `.wiff` files or Agilent `.d` folders,
   or drag them into the list.
3. Choose an output mode:
   - **Full Profile** (default) keeps all the data. Use it for analysis and
     archiving.
   - **Sparse Profile v3** makes a much smaller file for quick viewing. It drops
     weak signal (see [Limitations](#limitations)).
4. Click **Choose Folder** and pick where the mzML files should go.
5. Click **Convert** (one file) or **Convert Batch** (several). The list shows each
   file's progress. **Cancel** stops safely at any time.
6. Click **Reveal Outputs** to see the new files in Finder.

Output files are named after the input: `sample.mzML` (Full Profile) or
`sample.sp.mzML` (Sparse Profile). An Agilent `run.d` folder becomes `run.mzML`.

Good to know:

- **Nothing is overwritten.** If an output file already exists, that input is
  skipped.
- **One failure doesn't stop a batch.** Problems appear in the log; **Save Log**
  keeps a copy.
- **Files with several samples** open a dialog where you tick the samples to
  convert. Each becomes its own file, for example `sample_2.mzML`.
- **UV (DAD) data** is added automatically when `sample.dad.scan` and
  `sample.dad.sidx` are beside the WIFF file.
- **Inspect metadata without converting** (one queued file) shows the file's
  contents without writing anything.
- **Mass correction (beta)** is off by default. Turn it on and enter known standards,
  one per line, as `expected_mz;RT_minutes`, for example `195.0877;3.2`. KmzConvert
  shifts all MS1 masses by one fitted ppm offset and adds `.mc` to the file name.
  MS2 masses are not corrected.
- **Agilent `.d` folders** convert in Full Profile only, so far. Sparse Profile and
  mass correction report an error for them. Masses use the instrument's own
  calibration: the per-scan calibration, which includes reference-mass correction,
  when the folder has one, otherwise its default calibration.

**From the Terminal.** The installed `kmzconvert` command does the same work:

```sh
cd ~/Data                                       # the folder with your files
/Applications/KmzSuite/kmzconvert sample.wiff                      # writes sample.mzML
/Applications/KmzSuite/kmzconvert sample.wiff --profile sparse     # writes sample.sp.mzML
/Applications/KmzSuite/kmzconvert sample.wiff --list-samples       # list stored samples
/Applications/KmzSuite/kmzconvert sample.wiff --sample 2 -o out.mzML   # writes out_2.mzML
/Applications/KmzSuite/kmzconvert run.d                            # Agilent: writes run.mzML
/Applications/KmzSuite/kmzconvert --help                           # all options
```

If you built KmzSuite from source, use `~/Applications/KmzSuite/kmzconvert` instead.

## View chromatograms and spectra: KmzViewer

### Open files and show chromatograms

1. Choose **File → Open mzML** (**Cmd+O**), or drag `.mzML` files onto the window.
   Each file appears in the **Data Navigator** on the left with its TIC.
2. Tick or untick traces to show or hide them. Right-click a trace for more
   actions: extract, show only selected, hide, change color or remove.
3. To plot an extracted ion chromatogram, choose **Chromatogram → Extract EIC**
   (**Cmd+E**). Enter a mass (`273.1559`), a list (`273.1559, 301.1410`) or a range
   (`250-300`), then pick the ppm tolerance, polarity and files.
4. To plot the most intense ion in each scan, click **Extract BPC…**.
5. To limit all TIC, BPC and EIC traces to an m/z range, use the **m/z filter**
   above the plots: enter **Start** and **Stop**, then click **Apply**. **Clear**
   restores the original traces.

### Arrange and zoom

- **Stacked** shows one pane per trace; **Overlay** draws them together.
- Choose the **Pan** or **Zoom box** mouse mode to move or zoom the plots.
- **Autoscale Y** fits the Y axis to the visible time range. **Match Y axes** puts
  traces on a shared scale.
- Press **F** (or **Cmd+F**) to return to the full view.
- In the navigator, group traces by file or by chromatogram type, drag files to
  reorder them, or use **Sort files A–Z**. **Remove/Show/Hide…** manages many
  traces at once.

### Inspect MS1 spectra

1. Double-click a chromatogram. The nearest MS1 scan opens in the **MS1 Spectra**
   panel on the right.
2. Double-click another file's chromatogram to add its spectrum for comparison.
3. Click or drag along any chromatogram to move all spectra to that retention time.
   **Left/Right arrow** keys step one scan at a time.
4. The eye icon beside each file in the navigator includes or removes its spectrum.
5. Spectra zoom together. Hover over a peak to read its m/z and intensity.

### UV (DAD) chromatograms

If the file contains UV data, choose **Wavelength → Extract TWC…** (total
absorbance) or **Extract EWC…** (one wavelength, for example 254 nm with a 4 nm
bandwidth). A **DAD Spectra** panel opens beside the MS1 spectra.

### Export figures

- **Export SVG…** opens a live preview of the chromatograms. Choose traces, order,
  colors, size, offsets and legend, then click **Save SVG**.
- **File → Export MS1 spectra…** (or **Export DAD spectra…**) does the same for
  the spectra.

The SVG files stay editable in vector editors such as Adobe Illustrator.

### Follow up on a mass

Right-click a peak label in the MS1 spectra:

- **Open MS/MS in Kmz2Viewer** opens that file in Kmz2Viewer, already filtered to
  MS/MS scans whose precursor is within ±20 ppm of that mass.
- **Spectral analysis…** opens a window for the matching MS/MS scan. Click
  **Search references + motifs** to find related public spectra and known fragment
  patterns. This search runs on your Mac; the first one downloads about 220 MB of
  public reference libraries. Optional **DreaMS** and **Online GNPS…** searches go
  online (see the README's privacy notes). **Save analysis…** keeps the results in
  a `.kmzspectra` file you can reopen later.

## Browse MS/MS scans: Kmz2Viewer

1. Choose **Open mzML** (**Cmd+O**). You can open several files; the table lists
   every MS/MS scan.
2. Narrow the list with the filters: **Precursor m/z**, **Fragment m/z** (20 ppm by
   default), **Fragment ≥ % base peak** and **RT min/max**. Empty fields don't
   filter. **Clear filters** shows every scan again.
3. Click a row to see its spectrum. **Left/Right arrow** keys move through the rows;
   **F** returns to the full view.
4. Turn on **Show neutral losses** to connect major peaks and see candidate
   formulas for the mass differences. **Formula settings** controls the tolerance,
   elements and double-bond equivalents. These are predicted formulas, not
   identified fragments.

## Compare runs: KmzCoMet

KmzCoMet finds MS1 ion features (an m/z at a retention time) across several runs
and measures their peak height in every run.

1. Click **Add WIFF Files** (**Cmd+O**) or drag files in. Drag the list to reorder
   it; the first file sets the reference retention times.
2. Optionally open **Settings**. The defaults are a reasonable starting point, not
   universal values.
3. Click **Run Analysis** (**Cmd+R**). The analysis runs on your Mac and never
   changes your files.
4. Read the **Heights** table: one row per feature (m/z, retention time, polarity),
   one column per run. A **blank** cell means that run didn't cover the feature; a
   **0** means it was measured with no signal. Click headers to sort; use the
   search and m/z/RT bounds to filter.
5. To plot EICs, click height cells (or a row's checkbox), then click **Plot EICs**
   (**Cmd+E**). Plots open in a new tab and need the original WIFF files.
6. Click **Plot PCA** for an interactive 3D view of how features vary across runs.
   **Cmd+click** a point to plot that feature's EICs.
7. **Export Heights as CSV** saves the table; **Export Excel** saves all four result
   sheets.

Every finished analysis is saved automatically. Use **Past Analyses** to reopen one
later, even without the original files. **Analysis Details** shows the evidence and
settings behind each value.

## Keyboard shortcuts

| Action | Keys |
| --- | --- |
| Open files | Cmd+O |
| Extract EIC (KmzViewer), Plot EICs (KmzCoMet) | Cmd+E |
| Run analysis (KmzCoMet) | Cmd+R |
| Full view | F, Cmd+F or Cmd+0 |
| Previous/next scan or row | Left/Right arrow |
| Close plot tab (KmzCoMet) | Cmd+W |
| Copy selected cells (KmzCoMet) | Cmd+C |

## Troubleshooting

**"Legacy input received. Please reconvert using the current KmzConvert."** The
mzML file came from an older KmzSuite version. Convert the original WIFF file
again.

**KmzConvert refuses a file.** The acquisition uses a layout KmzConvert doesn't
support (see [Limitations](#limitations)). It refuses rather than guess. The log
explains why; **Save Log** keeps a copy.

**A KmzCoMet EIC is unavailable.** The original WIFF file was moved or changed. Put
it back where it was when you ran the analysis.

**A spectral search download fails.** Check your internet connection and try
again. **DreaMS unavailable** usually means the free public service is busy; your
local results still work.

**An app doesn't appear in Spotlight.** Open the KmzSuite folder in Applications
(`~/Applications/KmzSuite` if you built from source) in Finder and double-click the
app. Spotlight may need a few minutes to find newly installed apps.

**macOS says the installer or an app is damaged or can't be checked.** Download the
installer again and run it. Don't turn off macOS security settings.

**The source build stops with a Python or download error.** The first build downloads
Python 3.14 and other packages, so check your internet connection. Then delete the
build's private Python environment and build again; it is recreated automatically:

```sh
cd ~/KmzSuite
rm -rf .venv
./scripts/build.sh
```

## Limitations

- **SCIEX conversion** supports X500-series acquisitions from SCIEX OS 2.x:
  positive-mode TOF MS profile data with data-dependent MS/MS, one period.
  Negative polarity, other instruments, WIFF2-only data, multiple periods and
  multi-sample files whose methods differ are refused. Precursor charge, isolation
  width and activation type are not decoded.
- **Agilent conversion** supports MassHunter TOF and Q-TOF acquisitions saved with
  profile data: MS1 and targeted MS/MS. It has been checked on a 6545 (MassHunter
  B.08, negative targeted MS/MS) and a 6546 (MassHunter 10.1, positive MS1).
  Data-dependent MS/MS and other models have not been tested. Centroid-only data,
  triple quadrupoles, ion mobility (6560), All Ions MS/MS, older MassHunter
  layouts and UV (DAD) data are refused or not exported. Isolation width and activation type
  are not decoded, and Sparse Profile and mass correction are not available yet.
- **Sparse Profile** omits weak signal (peaks that never exceed about 20 detector
  events) and trims faint peak tails. Use Full Profile for quantitative work.
- **Mass correction** is a beta: one global offset per run, with no guarantee of
  accuracy across the whole run. MS2 masses are not corrected.
- **UV (DAD) data** has been checked for internal consistency, not yet against a
  vendor export.
- **KmzCoMet** reports raw, unnormalized peak heights. It does not correct for
  blanks, identify compounds or test statistical significance. Isotopes and adducts
  appear as separate features. PCA is for exploration.
- **Formulas, spectral matches, motifs, DreaMS and GNPS results** are hypotheses to
  investigate, not identifications.
- KmzSuite runs only on Apple Silicon Macs with macOS 15 or later.
