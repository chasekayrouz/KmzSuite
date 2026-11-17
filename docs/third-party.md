# Third-party notes

The converter's prefix-token investigation was informed by sciexwiff
(`2cab6bb156b149d1b1d7407ce8021a32eb3e8be6`, MIT, copyright 2026 David Teschner).
Its permission notice is in `resources/licenses/sciexwiff-LICENSE.txt`. The
restricted X500 Python codec differs in coordinate seeds/scaling, calibration,
escapes, bounds checks and index framing.

Public OpenSXRaw sources (Apache-2.0) informed stream/method interpretation;
ProteoWizard sources informed API and mzML mapping. Their implementation code,
vendor SDKs, binaries and keys are not included. No network or vendor runtime
is needed to convert an acquisition.

Python runtime dependencies retain their installed notices. Build-time asset
collection includes distribution metadata, available license files, CPython's
license and explicit Qt/PySide LGPL/GPL terms. Viewer scientific dependencies are
listed in `pyproject.toml`; exact packaging versions are in
`requirements-build.lock`. NumPy and olefile are the converter's runtime
dependencies. Swift UI uses system frameworks.

The downloadable apps include KmzSuite's MIT license and these notices (**Help →
Dependency Notices**). Qt and Qt for Python are unmodified, dynamically linked
LGPLv3 libraries in each app's `Contents/Frameworks`. A generated `SOURCE.txt`
beside their license texts names the exact versions and where to download their
complete source code.

KmzViewer, Kmz2Viewer and KmzCoMet read mzML with a pinned, unmodified release of the PSI-MS
controlled vocabulary (HUPO-PSI, CC BY 4.0). Its source, version, hash and
license statement ship beside it in `kmzcore/data/PSI-MS-ATTRIBUTION.txt`.

KmzCoMet uses openpyxl (MIT) and et-xmlfile (MIT) for local Excel output. They are
in the repository lock and the frozen app, alongside the qualified native WIFF
reader and its existing notices. The app uses PySide6/Qt, NumPy, Pyteomics and
psims; their dependency metadata and licenses ship with the other runtime notices.

KmzViewer uses MSEntropy 1.5.3 (Apache-2.0) for local spectral entropy matching
and h5py 3.16.0 (BSD) for optional public DreaMS-reference conversion. Their
installed metadata/licenses are collected by the same packaging owner. MotifDB
is a pinned public compilation from MS2LDA; its source commit, content hashes,
compilation license, original-source attribution and qualification limitations
ship in `kmzviewer/data`. The complete collection is included, without selecting
motifs for the research benchmark. Downloaded GNPS/MassBank/BMDMS references and
MassSpecGym/DreaMS embeddings retain their provider/source terms and provenance;
they are not included in ordinary review bundles or source control. DreaMS
weights and PyTorch are not redistributed. No public distribution or publication
of those downloaded datasets is performed by this development task.
