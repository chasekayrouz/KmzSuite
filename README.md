# KmzSuite

Mass spectrometry tools for Macs with Apple Silicon. KmzSuite converts SCIEX X500
WIFF and Agilent MassHunter (`.d`) acquisitions to standard mzML, displays
chromatograms and spectra, browses MS/MS scans and compares MS1 features across runs.

| App | What it does |
| --- | --- |
| **KmzConvert** | Converts SCIEX X500 WIFF and Agilent TOF/Q-TOF `.d` acquisitions to indexed mzML (app and `kmzconvert` command) |
| **KmzViewer** | Plots TIC, BPC, EIC and UV (DAD) chromatograms; compares MS1 spectra; exports SVG figures; searches related MS/MS spectra |
| **Kmz2Viewer** | Browses and filters MS/MS scans; suggests formulas for mass differences |
| **KmzCoMet** | Compares MS1 ion features across WIFF runs; plots EICs and PCA; exports CSV and Excel |
| **KmzSuite** | Launcher for the four apps above |

The viewers open any standard mzML file. Each app runs on its own and works offline
unless you choose an online search.

> **Status:** research software under active development. KmzConvert supports only
> the SCIEX X500 data layouts it has been checked against, and Agilent TOF/Q-TOF
> profile or centroid data whose stored scan summaries it reproduces (checked on
> 6220, 6540, 6545 and 6546 data); it refuses anything else.
> Formula suggestions and spectral matches are leads to investigate, not
> identifications. See [Limitations](docs/user-guide.md#limitations).

## Requirements

- A Mac with Apple Silicon (M1 or newer). Intel Macs are not supported.
- macOS 15 or later.
- About 1 GB of free disk space, plus space for your data.

## Installation

**1. Download.** Open the [latest release](https://github.com/chasekayrouz/KmzSuite/releases/latest)
and click the file whose name ends in `-macOS-arm64.pkg` to download it.

**2. Install.** Double-click the downloaded file and follow the installer: click
**Continue**, then **Install**, and enter your Mac password when asked. KmzSuite is
installed in the **KmzSuite** folder inside **Applications**. You can then delete
the download.

Installing needs an administrator account on the Mac. If you don't have one, ask
whoever manages your Mac, or [build from source](#build-from-source) instead.

**3. Open an app.** Press **Cmd+Space**, type **KmzViewer** (or any app name) and
press **Return**.

### Update

Quit the KmzSuite apps, download the new version and run its installer. It
replaces the previous version. Your settings and saved analyses are kept.

### Uninstall

Delete these items in Finder (press **Cmd+Shift+G** to type a path):

- `/Applications/KmzSuite` (the apps; macOS asks for your password)
- `~/Library/Application Support/KmzSuite` (saved KmzCoMet analyses)
- Items whose names contain `kmzsuite` in `~/Library/Caches` and
  `~/Library/Preferences` (downloaded reference spectra, caches and settings)

## Build from source

Building the apps yourself gives the same apps, installed in
`~/Applications/KmzSuite/`. It takes about 6 GB of free disk space and an internet
connection for the first build, which downloads Python 3.14 and the packages
KmzSuite needs. You will install two tools: Apple's Command Line Tools (or Xcode)
and uv. You don't need to install Python yourself.

You type these commands in the **Terminal** app (press **Cmd+Space**, type
*Terminal*, press **Return**). Copy one command at a time, paste it and press
**Return**.

**1. Install Apple's Command Line Tools.** These provide Swift, Git and a C compiler.

```sh
xcode-select --install
```

A window asks you to confirm. If the tools are already installed, the command says
so. Skip this step if you have Xcode.

**2. Install uv.** [uv](https://docs.astral.sh/uv/) downloads Python and the
packages KmzSuite needs. Run its official installer:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then close the Terminal window and open a new one, so the new command is found. If
you use Homebrew, `brew install uv` works too.

**3. Download the source.** This creates a `KmzSuite` folder in your home folder.

```sh
cd ~
git clone https://github.com/chasekayrouz/KmzSuite.git
cd KmzSuite
```

**4. Build the apps.**

```sh
./scripts/build.sh
```

The first build downloads Python 3.14 (shared by uv, without replacing any other
Python on your Mac), creates a private Python environment inside the `KmzSuite`
folder, downloads the packages it needs and compiles the apps. This takes a few
minutes, depending on your internet connection; later builds are faster.

**5. Run the tests (optional).**

```sh
./scripts/test.sh
```

**6. Install the apps.** Quit any KmzSuite apps that are open, then run:

```sh
./scripts/install-local.sh
```

This checks the new apps and installs all five in `~/Applications/KmzSuite/`. The
apps open and close by themselves for a moment during the checks; this is
expected. If a check fails, your previous installation is left untouched.

The installed apps don't need the `KmzSuite` source folder, the Terminal or Python
to run. Keep the folder for updates:

```sh
cd ~/KmzSuite
git pull
./scripts/build.sh
./scripts/install-local.sh
```

The previous installation is kept as `~/Applications/.KmzSuite-rollback.zip`. To
uninstall, delete `~/Applications/KmzSuite`, that rollback file and `~/KmzSuite`,
plus the Library items listed above. To also remove the Python that uv downloaded,
run `uv python uninstall 3.14` (skip this if you use uv for other projects). To
remove uv itself, follow uv's
[uninstall instructions](https://docs.astral.sh/uv/getting-started/installation/#uninstallation).

## Using the apps

The [user guide](docs/user-guide.md) walks through each app step by step. Each app
also has a detailed manual under its **Help** menu.

To convert from the Terminal instead of the app:

```sh
/Applications/KmzSuite/kmzconvert sample.wiff -o sample.mzML
/Applications/KmzSuite/kmzconvert run.d -o run.mzML      # Agilent .d folder
```

If you built from source, the command is `~/Applications/KmzSuite/kmzconvert`.

## Privacy

Your data stays on your Mac. KmzSuite connects to the internet only when you ask it
to:

- The first local spectral search downloads public reference libraries.
- Optional DreaMS and online GNPS searches send only the selected spectrum's peaks
  and precursor information, never files or file names.
- Loading a reference structure or record fetches that public entry from GNPS.

## License

KmzSuite is released under the [MIT License](LICENSE). Bundled third-party
components keep their own licenses; see [third-party notes](docs/third-party.md).
