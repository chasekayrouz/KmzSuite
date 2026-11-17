"""One-spectrum exploration: local public references, motifs and explicit evidence."""
from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
import re
from urllib.parse import urlencode

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from kmzcore.cancel import Cancelled
from kmzcore.runtime.qt import WindowRuntime
from kmzviewer.gnps import finite, load_analysis, matching_scans, prepare_query
from kmzviewer.gnps_client import qt_request
from kmzviewer.gui.gnps import GnpsWindow, GnpsWorker, StructureView, show_peaks
from kmzviewer.reference_pack import default_cache_dir, ensure_reference_pack
from kmzviewer.spectral_analysis import new_spectral_analysis, load_spectral_analysis, save_spectral_analysis
from kmzviewer.spectral_matching import SpectralSettings, align_peaks, prepare_search_peaks, search_reference_pack
from kmzviewer.spectral_motifs import search_motifs

METHODS = {"hybrid": "Fragments + precursor differences", "direct": "Shared fragments",
           "neutral_loss": "Shared precursor differences"}
FILE_FILTER = "KmzViewer spectral analysis (*.kmzspectra);;GNPS query (*.kmzgnps)"


def label(text=""):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    return widget


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.verticalHeader().hide()
    widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for col in range(1, len(headers)):
        widget.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
    return widget


class SpectralWindow(QMainWindow):
    closed = Signal()

    def __init__(self, parent=None, *, cache_dir=None, searcher=search_reference_pack,
                 pack_loader=ensure_reference_pack, motif_searcher=search_motifs):
        super().__init__(parent, Qt.WindowType.Window)
        self.resize(1260, 900)
        self.setMinimumSize(1050, 760)
        self.setWindowTitle("Spectral analysis · KmzViewer")
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.searcher, self.pack_loader, self.motif_searcher = searcher, pack_loader, motif_searcher
        self.query = self.analysis = self.pack = None
        self._pack_expanded = None
        self.scans = []
        self.saved_path = None
        self._busy = self._updating = False
        self._active_hit = None
        self._rows = []
        self._online_windows = []
        self.target_mz = self.target_rt = None
        self.runtime = WindowRuntime(self, "ms1")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.source = label("Choose an MS1 peak, or open a saved spectral analysis.")
        layout.addWidget(self.source)
        row = QHBoxLayout()
        row.addWidget(QLabel("MS2 scan"))
        self.scan_picker = QComboBox()
        self.scan_picker.setMinimumContentsLength(25)
        self.scan_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.scan_picker.currentIndexChanged.connect(self._scan_changed)
        row.addWidget(self.scan_picker, 1)
        self.open_button = QPushButton("Open analysis…")
        self.open_button.clicked.connect(self.open_dialog)
        self.save_button = QPushButton("Save analysis…")
        self.save_button.clicked.connect(self.save_dialog)
        row.addWidget(self.open_button)
        row.addWidget(self.save_button)
        layout.addLayout(row)
        self.scan_details = label()
        layout.addWidget(self.scan_details)

        controls = QGridLayout()
        self.fragment = QDoubleSpinBox()
        self.fragment.setDecimals(3)
        self.fragment.setRange(.001, .1)
        self.fragment.setSingleStep(.005)
        self.fragment.setValue(.02)
        self.noise = QDoubleSpinBox()
        self.noise.setDecimals(2)
        self.noise.setRange(0., 10.)
        self.noise.setValue(1.)
        self.minimum = QSpinBox()
        self.minimum.setRange(1, 100)
        self.minimum.setValue(3)
        for column, (title, widget) in enumerate([
                ("Fragment ± (Da)", self.fragment), ("Noise cutoff (% base peak)", self.noise),
                ("Min shared ions (local)", self.minimum)]):
            widget.setKeyboardTracking(False)
            widget.setAccessibleName(title)
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.valueChanged.connect(self._settings_changed)
            controls.addWidget(QLabel(title), 0, column)
            controls.addWidget(widget, 1, column)
        self.expanded = QCheckBox("Include BMDMS natural products")
        self.expanded.setToolTip("Optional additional public reference collection; about 200 MB extra on first use.")
        self.expanded.toggled.connect(self._settings_changed)
        controls.addWidget(self.expanded, 0, 3, 2, 1)
        self.search_button = QPushButton("Search references + motifs")
        self.search_button.clicked.connect(self.search)
        controls.addWidget(self.search_button, 0, 4, 2, 1)
        layout.addLayout(controls)
        required = QHBoxLayout()
        self.fragments = QLineEdit()
        self.fragments.setPlaceholderText("Optional required fragment m/z values, separated by commas")
        self.fragments.setAccessibleName("Required reference fragments")
        self.losses = QLineEdit()
        self.losses.setPlaceholderText("Optional required precursor−fragment differences")
        self.losses.setAccessibleName("Required reference precursor differences")
        self.fragments.textEdited.connect(self._settings_changed)
        self.losses.textEdited.connect(self._settings_changed)
        required.addWidget(self.fragments)
        required.addWidget(self.losses)
        layout.addLayout(required)
        actions = QHBoxLayout()
        self.online_button = QPushButton("Online GNPS…")
        self.online_button.clicked.connect(self.open_online)
        actions.addWidget(self.online_button)
        self.learned_enabled = QCheckBox("Include DreaMS learned neighbors")
        self.learned_enabled.setToolTip("Optional online embedding of the selected spectrum, followed by local neighbor retrieval. First use downloads 1.54 GB of public references.")
        self.learned_enabled.toggled.connect(self._settings_changed)
        actions.addWidget(self.learned_enabled)
        self.privacy = label("Local search. First use downloads about 220 MB of public references; your spectrum stays on this computer.")
        actions.addWidget(self.privacy, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.runtime.cancel)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

        split = QSplitter()
        self.tabs = QTabWidget()
        related = QWidget()
        related_layout = QVBoxLayout(related)
        related_layout.setContentsMargins(3, 3, 3, 3)
        self.ranking = QComboBox()
        for key, name in METHODS.items():
            self.ranking.addItem(name, key)
        self.ranking.addItem("Learned similarity · DreaMS (optional)", "dreams")
        self.ranking.currentIndexChanged.connect(self._fill_related)
        related_layout.addWidget(self.ranking)
        self.summary = label("Search across reference precursor masses. Scores express spectral similarity, not identification probability.")
        related_layout.addWidget(self.summary)
        self.matches = table(["Reference molecule", "Score", "Ions", "Query %", "Library"])
        self.matches.itemSelectionChanged.connect(self._selected)
        related_layout.addWidget(self.matches)
        self.tabs.addTab(related, "Related spectra")
        motif_page = QWidget()
        motif_layout = QVBoxLayout(motif_page)
        motif_layout.setContentsMargins(3, 3, 3, 3)
        self.motif_summary = label("Explore overlap with annotated MotifDB fragment patterns.")
        motif_layout.addWidget(self.motif_summary)
        self.motifs = table(["Annotated motif", "Overlap", "Ions", "Coverage %"])
        self.motifs.itemSelectionChanged.connect(self._selected)
        motif_layout.addWidget(self.motifs)
        self.tabs.addTab(motif_page, "Motif evidence")
        self.tabs.currentChanged.connect(self._selected)
        split.addWidget(self.tabs)

        right = QWidget()
        detail = QVBoxLayout(right)
        detail.setContentsMargins(5, 0, 0, 0)
        self.candidate = label("Query spectrum")
        self.candidate.setStyleSheet("font-weight: 600")
        detail.addWidget(self.candidate)
        self.evidence = label("Select a reference to inspect shared fragments and precursor differences.")
        detail.addWidget(self.evidence)
        self.structure = StructureView("Reference structure available on request")
        self.structure.setTextFormat(Qt.TextFormat.PlainText)
        self.structure.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.structure.setMinimumHeight(90)
        self.structure.setMaximumHeight(150)
        detail.addWidget(self.structure)
        self.mirror = pg.PlotWidget()
        self.mirror.setMinimumHeight(180)
        self.mirror.setBackground("#ffffff")
        self.mirror.setLabel("bottom", "m/z")
        self.mirror.setLabel("left", "Relative intensity (%)")
        self.mirror.setMenuEnabled(False)
        self.mirror.getPlotItem().hideButtons()
        self.mirror.scene().sigMouseClicked.connect(self._peak_clicked)
        detail.addWidget(self.mirror, 2)
        self.legend = label("Query above zero · reference below · teal: shared fragment · amber: precursor difference · gray: other ions")
        detail.addWidget(self.legend)
        self.pairs = table(["Query m/z", "Reference m/z", "Relationship", "Error (Da)"])
        self.pairs.setMaximumHeight(155)
        detail.addWidget(self.pairs)
        buttons = QHBoxLayout()
        self.structure_button = QPushButton("Load structure")
        self.structure_button.clicked.connect(self.load_structure)
        self.link_button = QPushButton("Open reference")
        self.link_button.clicked.connect(self.open_record)
        full = QPushButton("Full m/z")
        full.clicked.connect(self.mirror.autoRange)
        buttons.addWidget(self.structure_button)
        buttons.addWidget(self.link_button)
        buttons.addWidget(full)
        detail.addLayout(buttons)
        split.addWidget(right)
        split.setSizes([580, 630])
        layout.addWidget(split, 1)
        self.footer = label("Exploratory evidence only. Different structures can share fragments. Double-click a query peak to require that fragment in reference searches.")
        layout.addWidget(self.footer)
        self._sync()

    @staticmethod
    def _masses(text):
        try:
            return tuple(float(x.strip()) for x in text.split(",") if x.strip())
        except ValueError as exc:
            raise ValueError("Required masses must be numbers separated by commas.") from exc

    def settings(self):
        return SpectralSettings(fragment_da=self.fragment.value(), relative_noise=self.noise.value() / 100.,
            min_peaks=self.minimum.value(), required_fragments=self._masses(self.fragments.text()),
            required_losses=self._masses(self.losses.text()), expanded=self.expanded.isChecked()).validate()

    def load_source(self, path, mass, rt, polarity):
        self.target_mz, self.target_rt = mass, rt
        self.source.setText(f"{Path(path).name} · MS1 m/z {mass:.7f} · matching MS2 precursors ±20 ppm")
        def read(cancel, progress):
            return matching_scans(path, mass, rt, polarity, cancel=cancel,
                progress=lambda done, total: progress(f"Reading matching MS2 scans · {100 * done // max(1, total)}%"))
        self.runtime.start(GnpsWorker(read), self._loaded_scans)

    def _loaded_scans(self, scans):
        self.scans = scans
        self._updating = True
        self.scan_picker.clear()
        for scan in scans:
            rt = f"{scan.rt_min:.4f} min" if finite(scan.rt_min) else "RT unknown"
            self.scan_picker.addItem(f"{rt} · m/z {scan.precursor_mz:.6f} · {scan.scan_id}")
        self._updating = False
        if scans:
            self._scan_changed(0)
        else:
            self.query = None
            self.scan_details.setText("No matching acquired MS2 scan within ±20 ppm and compatible polarity.")
            self._clear_results()
        self._sync()

    def _scan_changed(self, index):
        if self._updating or not 0 <= index < len(self.scans):
            return
        try:
            self.query = prepare_query(self.scans[index], target_mz=self.target_mz, target_rt=self.target_rt)
            self._clear_results()
            self._show_query()
        except ValueError as exc:
            self.query = None
            self._clear_results()
            self.scan_details.setText(str(exc))
        self._sync()

    def load_query(self, query):
        self._updating = True
        self.scans = []
        self.scan_picker.clear()
        self.scan_picker.addItem("Saved query · " + str(query.get("scan_id", "")))
        self._updating = False
        self.query = query
        self.source.setText("Selected MS2 · " + query.get("source_name", ""))
        self._clear_results()
        self._show_query()
        self._sync()

    def _show_query(self):
        if not self.query:
            return
        q = self.query
        rt = f"{q['rt_min']:.4f} min" if finite(q.get("rt_min")) else "RT unknown"
        self.scan_details.setText(f"{rt} · precursor m/z {q['precursor_mz']:.6f} · "
            f"{ {1: 'positive', -1: 'negative', 0: 'unknown polarity'}.get(q.get('polarity', 0))} · "
            f"charge {q.get('charge') or 'unknown'} · collision energy {q.get('collision_energy') if q.get('collision_energy') is not None else 'unknown'}\n"
            f"{q['preparation']} · {len(q['peaks'])} original query peaks")
        self._show_match()

    def _settings_changed(self, *_):
        if not self._updating:
            self._clear_results()
            self.statusBar().showMessage("Settings changed. Search again; save results before changing settings.")
            self._sync()
            self.privacy.setText("DreaMS sends selected peaks + precursor m/z online. First use downloads 1.54 GB of public references. Local fragment filters do not restrict learned neighbors."
                if self.learned_enabled.isChecked() else
                "Local search. First use downloads about 220 MB of public references; your spectrum stays on this computer.")

    def _clear_results(self):
        self.analysis = None
        self.saved_path = None
        self._active_hit = None
        self._rows = []
        self.matches.setRowCount(0)
        self.motifs.setRowCount(0)
        self.summary.setText("Not searched.")
        self.motif_summary.setText("Not searched.")
        self._show_match()

    def search(self):
        if self._busy or not self.query:
            return
        try:
            settings = self.settings()
        except ValueError as exc:
            self._on_failed(str(exc))
            return
        query = self.query
        include_learned = self.learned_enabled.isChecked()
        existing_pack = self.pack if self._pack_expanded == settings.expanded else None
        self._clear_results()
        self.summary.setText("Searching…")
        self.motif_summary.setText("Searching…")
        def run(cancel, progress):
            progress("Matching annotated motifs locally…")
            motifs = self.motif_searcher(query, settings, cancel=cancel)
            pack = existing_pack
            try:
                if pack is None:
                    pack = self.pack_loader(self.cache_dir, allow_download=True, expanded=settings.expanded,
                        cancel=cancel, progress=lambda fraction, message: progress(message))
                library = self.searcher(query, settings, pack, cancel=cancel,
                    progress=lambda fraction, message: progress(message))
            except Cancelled:
                raise
            except Exception as exc:
                library = dict(methods={}, references={}, error=str(exc), provenance={}, preparation={})
            learned = {}
            if include_learned:
                try:
                    from kmzviewer.dreams_client import search_dreams
                    learned = search_dreams(query, self.cache_dir / "dreams", cancel=cancel,
                        progress=progress)
                    self._align_learned(learned, query, settings)
                except Cancelled:
                    raise
                except Exception as exc:
                    learned = dict(hits=[], references={}, error=str(exc), provenance={}, preparation={})
            return new_spectral_analysis(query, settings, library, motifs, learned), pack, settings.expanded
        def receive(payload):
            analysis, self.pack, self._pack_expanded = payload
            self._searched(analysis)
        # ms-entropy's native centroid sorter recurses on ordered spectra.
        # macOS's default worker stack is too small for dense public references;
        # reserve a larger stack without changing the scoring/preparation path.
        self.runtime.start(GnpsWorker(run), receive, stack_size=16 * 1024 * 1024)

    @staticmethod
    def _align_learned(result, query, settings):
        """An independent readable alignment; never used to rerank learned hits."""
        prepared = prepare_search_peaks(query, settings)
        qpeaks = prepared["peaks"]
        result["prepared_query"] = qpeaks.tolist()
        result.setdefault("preparation", {})["comparison"] = prepared["description"]
        for hit in result.get("hits", []):
            ref = result.get("references", {}).get(hit.get("accession"), {})
            peaks = ref.get("peaks", [])
            if not peaks:
                continue
            rq = dict(peaks=peaks, precursor_mz=ref.get("precursor_mz", hit["precursor_mz"]), charge=ref.get("charge"))
            try:
                rp = prepare_search_peaks(rq, settings)["peaks"]
            except ValueError:
                hit["alignment_warning"] = "No reference ions remain for the independent local comparison. Learned rank is unchanged."
                continue
            # Comparisons are stored separately from the original public peaks.
            ref["comparison_peaks"] = rp.tolist()
            mode = "hybrid" if query.get("charge") in (None, 0, 1, -1) else "direct"
            alignment = align_peaks(qpeaks, rp, query["precursor_mz"], rq["precursor_mz"], settings.fragment_da, mode)
            hit["alignment"] = alignment
            hit["matched_peaks"] = len(alignment["pairs"])

    def _searched(self, analysis):
        self.analysis = analysis
        self._fill_related()
        self._fill_motifs()
        self.statusBar().showMessage("Search complete. Results remain unsaved until Save analysis.")
        if self.matches.rowCount():
            self.tabs.setCurrentIndex(0)
            self.matches.selectRow(0)
        elif self.motifs.rowCount():
            self.tabs.setCurrentIndex(1)
            self.motifs.selectRow(0)
        self._sync()

    @staticmethod
    def _populate(widget, rows):
        widget.blockSignals(True)
        # Lay out once afterwards: each replaced visible item otherwise re-measures
        # every row of the resize-to-contents columns.
        widget.scheduleDelayedItemsLayout()
        widget.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                widget.setItem(row, col, item)
        widget.blockSignals(False)

    def _fill_related(self, *_):
        if not self.analysis:
            return
        library = self.analysis["library"]
        method = self.ranking.currentData()
        result = self.analysis.get("learned", {}) if method == "dreams" else library
        self._rows = result.get("hits", []) if method == "dreams" else library.get("methods", {}).get(method, [])
        cells = []
        for hit in self._rows:
            alignment = hit.get("alignment", {})
            score = hit.get("scores", {}).get(method, hit.get("score", 0))
            cells.append([hit["name"], f"{score:.3f}", hit.get("matched_peaks", alignment.get("matched_peaks", "")),
                f"{100 * alignment.get('query_coverage', hit.get('query_coverage', 0)):.0f}", hit.get("library", hit.get("source", ""))])
        self._populate(self.matches, cells)
        error = result.get("error")
        provenance = result.get("provenance", {})
        self.summary.setText("Reference search unavailable: " + error if error else
            f"{len(cells)} distinct reference groups · {'learned embedding similarity' if method == 'dreams' else 'entropy similarity'} · full precursor range\n"
            + str(provenance.get("description", "Reference annotations retain their source provenance.")))
        if method == "dreams" and not result:
            self.summary.setText("Enable DreaMS learned neighbors, then Search. It sends selected peaks and precursor m/z online; the public reference index is downloaded once.")
        if cells:
            self.matches.selectRow(0)
        self._selected()

    def _fill_motifs(self):
        result = self.analysis["motifs"]
        rows = result.get("hits", [])
        self._populate(self.motifs, [[h["name"], f"{h.get('score', 0):.3f}", h.get("matched_peaks", h.get("matched_ions", "")),
            f"{100 * h.get('motif_coverage', h.get('coverage', 0)):.0f}"] for h in rows])
        self.motif_summary.setText(f"{len(rows)} overlapping motifs · local fragment-pattern overlap, not a structure classifier.\n"
            + " ".join(result.get("warnings", [])))
        if rows:
            self.motifs.selectRow(0)

    def _selected(self, *_):
        self._active_hit = None
        if self.analysis:
            if self.tabs.currentIndex() == 0:
                row, rows = self.matches.currentRow(), self._rows
            else:
                row, rows = self.motifs.currentRow(), self.analysis["motifs"].get("hits", [])
            if 0 <= row < len(rows):
                self._active_hit = rows[row]
        self._show_match()
        self._sync()

    def _show_match(self):
        if not hasattr(self, "mirror"):
            return
        self.mirror.clear()
        self.pairs.setRowCount(0)
        self.structure.clear()
        self.structure.setText("Select a reference molecule")
        hit = self._active_hit
        qpeaks = self.query["peaks"] if self.query else []
        if self.analysis and self.tabs.currentIndex() == 0:
            result = self.analysis.get("learned", {}) if self.ranking.currentData() == "dreams" else self.analysis["library"]
            qpeaks = result.get("prepared_query", qpeaks)
        elif self.analysis and self.query:
            # Display the native peaks retained by motif preparation, rather
            # than out-of-range noise that was not eligible for matching.
            noise = self.analysis["settings"]["relative_noise"]
            maximum = max(y for _, y in qpeaks)
            qpeaks = [[mz, y] for mz, y in qpeaks if mz < self.query["precursor_mz"] - 1 and y >= noise * maximum]
        self.mirror.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#bdc9cb")))
        show_peaks(self.mirror, qpeaks, "#9aa9ad")
        self.candidate.setText(hit["name"] if hit else "Query spectrum")
        self.evidence.setText("Select a match to inspect its evidence.")
        if not hit or not self.analysis:
            self.legend.setText("Query preview · no reference or motif selected")
            self.footer.setText("")
            self.mirror.autoRange()
            return
        motif = self.tabs.currentIndex() == 1
        if motif:
            self.legend.setText("Retained native query above · motif fragments below · teal: shared fragment · amber: precursor difference (see table)")
            self._show_motif(hit, qpeaks)
            self.mirror.autoRange()
            return
        accession = hit.get("accession", "")
        self.legend.setText("Prepared query above · prepared reference below · teal: shared fragment · amber: precursor difference · gray: other ions")
        learned = self.ranking.currentData() == "dreams"
        result = self.analysis.get("learned", {}) if learned else self.analysis["library"]
        reference = result.get("references", {}).get(accession, hit.get("reference", {}))
        rpeaks = reference.get("comparison_peaks", reference.get("peaks", []))
        show_peaks(self.mirror, rpeaks, "#b4bdc1", -1)
        alignment = hit.get("alignment", {})
        pairs = alignment.get("pairs", [])
        self._highlight(qpeaks, rpeaks, pairs)
        self._populate(self.pairs, [[f"{p['query_mz']:.5f}", f"{p['reference_mz']:.5f}",
            "fragment" if p.get("kind") == "direct" else "precursor difference", f"{p.get('error_da', 0):+.5f}"] for p in pairs])
        scores = hit.get("scores", {})
        score_text = " · ".join(f"{name} {scores[key]:.3f}" for key, name in
            [("dreams", "DreaMS"), ("direct", "Fragment"), ("hybrid", "Hybrid"), ("neutral_loss", "Difference")] if scores.get(key) is not None)
        self.evidence.setText(score_text + f"\n{alignment.get('direct_count', 0)} direct + {alignment.get('shifted_count', 0)} shifted ions · "
            f"{100 * alignment.get('query_coverage', 0):.1f}% prepared-query intensity covered\n"
            f"{hit.get('library', hit.get('source', ''))} · {hit.get('adduct', '') or 'adduct unreported'} · precursor {hit.get('precursor_mz', 0):.5f}\n"
            + str(hit.get("provenance", "Source annotation; similarity does not confirm this structure.")))
        self.evidence.setToolTip("Shown pairs use an independent local one-to-one alignment; they do not expose the scoring algorithm's internal assignments.")
        preparation = self.analysis["library"].get("preparation", {})
        warnings = preparation.get("warnings", [])
        self.footer.setText(str(preparation.get("description", "")) + "\n" + " ".join(warnings))
        self.structure.setText("Load the public reference structure")
        encoded = self.analysis.get("structures", {}).get(accession)
        if encoded:
            self._set_structure(encoded)
        self.mirror.autoRange()

    def _highlight(self, query, reference, pairs):
        # Use the spectrum's own base peak for overlays, not the subset's maximum.
        for values, field, sign in [(query, "query_mz", 1), (reference, "reference_mz", -1)]:
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            maximum = arr[:, 1].max()
            for kind, color in [("direct", "#087e91"), ("shifted", "#c27719")]:
                selected = []
                for pair in pairs:
                    if pair.get("kind") == kind and field in pair:
                        i = np.argmin(abs(arr[:, 0] - pair[field]))
                        selected.append(arr[i])
                if selected:
                    selected = np.asarray(selected)
                    self.mirror.plot(np.repeat(selected[:, 0], 2),
                        np.column_stack((np.zeros(len(selected)), selected[:, 1] / maximum * 100 * sign)).ravel(),
                        connect="pairs", pen=pg.mkPen(color, width=2))

    def _show_motif(self, hit, qpeaks):
        self.structure.setText("A motif is a fragment pattern; it does not specify a complete molecule.")
        self.footer.setText(self.analysis["motifs"].get("preparation", {}).get("description", "") +
            " Motif overlap is exploratory, not an MS2LDA inference or an identification probability.")
        self.evidence.setText(f"Overlap {hit.get('score', 0):.3f} · {hit.get('matched_peaks', hit.get('matched_ions', 0))} distinct query ions\n"
            f"{hit.get('motifset', '')} · motif coverage {100 * hit.get('motif_coverage', 0):.1f}%\n"
            "Annotated motif names are hypotheses, including names of specific scaffolds.")
        show_peaks(self.mirror, hit.get("fragment_peaks", []), "#b4bdc1", -1)
        pairs = hit.get("matches", hit.get("pairs", []))
        display = []
        plotted = []
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            qmz = pair.get("query_mz", 0)
            mass = pair.get("reference_mz", 0)
            kind = pair.get("kind", pair.get("type", "fragment"))
            is_loss = "loss" in kind
            observed = self.query["precursor_mz"] - qmz if is_loss else qmz
            # Motif matching stores absolute distance; the evidence table shows
            # the signed observed-minus-reference error, as the spectral view does.
            display.append([f"{qmz:.5f}", f"{mass:.5f}", "precursor difference" if is_loss else "fragment", f"{observed - mass:+.5f}"])
            plotted.append(dict(query_mz=qmz, kind="shifted" if is_loss else "direct"))
        self._populate(self.pairs, display)
        self._highlight(qpeaks, [], plotted)

    def _set_structure(self, encoded):
        try:
            raw = base64.b64decode(encoded, validate=True)
            buffer = QBuffer()
            buffer.setData(QByteArray(raw))
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            reader = QImageReader(buffer, b"png")
            size = reader.size()
            if 0 < size.width() * size.height() <= 8_000_000:
                image = reader.read()
                if not image.isNull():
                    self.structure.set_image(QPixmap.fromImage(image))
        except (ValueError, TypeError):
            self.structure.setText("Structure image unavailable")

    def load_structure(self):
        if self._busy or not self.analysis or not self._active_hit or self.tabs.currentIndex() != 0:
            return
        hit = self._active_hit
        smiles = hit.get("smiles", "")
        if smiles in ("", "N/A", "NA", "None"):
            self.structure.setText("This public reference has no structure annotation.")
            return
        accession = hit.get("accession", "")
        def retrieve(cancel, progress):
            raw = qt_request("https://structure.gnps2.org/structureimg?" + urlencode({"smiles": smiles}),
                cancel=cancel, max_bytes=2_000_000)
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Reference structure image unavailable.")
            return base64.b64encode(raw).decode("ascii")
        def received(encoded):
            self.analysis["structures"][accession] = encoded
            if self._active_hit and self._active_hit.get("accession") == accession:
                self._set_structure(encoded)
        self.runtime.start(GnpsWorker(retrieve), received)

    def open_record(self):
        hit = self._active_hit
        if not hit:
            return
        if self.tabs.currentIndex() == 1:
            url = "https://ccms-ucsd.github.io/GNPSDocumentation/ms2lda/"
        else:
            accession = hit.get("accession", "")
            if self.ranking.currentData() == "dreams":
                QDesktopServices.openUrl(QUrl("https://huggingface.co/datasets/roman-bushuiev/MassSpecGym"))
                return
            if not re.fullmatch(r"CCMSLIB\d+", accession):
                return
            url = "https://metabolomics-usi.gnps2.org/dashinterface/?" + urlencode({"usi1": f"mzspec:GNPS:GNPS-LIBRARY:accession:{accession}"})
        QDesktopServices.openUrl(QUrl(url))

    def _peak_clicked(self, event):
        if self._busy or not self.query or event.button() != Qt.MouseButton.LeftButton or not event.double():
            return
        position = self.mirror.getViewBox().mapSceneToView(event.scenePos())
        if position.y() < 0:
            return
        try:
            peaks = np.asarray(prepare_search_peaks(self.query, self.settings())["peaks"])
            index = np.argmin(abs(peaks[:, 0] - position.x()))
            mass = float(peaks[index, 0])
            if abs(mass - position.x()) > 2:
                return
            masses = list(self._masses(self.fragments.text()))
            if not any(abs(x - mass) <= self.fragment.value() for x in masses):
                masses.append(mass)
                self.fragments.setText(", ".join(f"{x:.5f}" for x in masses))
                self._settings_changed()
        except (ValueError, IndexError):
            return

    def open_online(self):
        if self._busy or not self.query:
            return
        window = GnpsWindow(self)
        window.load_query(self.query)
        self._online_windows.append(window)
        window.closed.connect(lambda: self._online_windows.remove(window) if window in self._online_windows else None)
        window.show()

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open spectral analysis", "", FILE_FILTER)
        if path:
            try:
                self.open_analysis(path)
            except Exception as exc:
                QMessageBox.warning(self, "Could not open analysis", str(exc))

    def open_analysis(self, path):
        if self._busy:
            raise ValueError("Wait for or cancel the current operation before opening an analysis.")
        if str(path).lower().endswith(".kmzgnps"):
            self.load_query(load_analysis(path)["query"])
            self.statusBar().showMessage("Imported the saved GNPS query locally. Search to explore references and motifs.")
            return
        data = load_spectral_analysis(path)
        self.load_query(data["query"])
        self._updating = True
        settings = data["settings"]
        self.fragment.setValue(settings["fragment_da"])
        self.noise.setValue(100 * settings["relative_noise"])
        self.minimum.setValue(settings["min_peaks"])
        self.expanded.setChecked(settings["expanded"])
        self.learned_enabled.setChecked(bool(data.get("learned")))
        self.fragments.setText(", ".join(map(str, settings["required_fragments"])))
        self.losses.setText(", ".join(map(str, settings["required_losses"])))
        self._updating = False
        self._searched(data)
        self.saved_path = str(path)
        self.statusBar().showMessage("Opened saved evidence locally. No network request was made.")

    def save_dialog(self):
        if not self.analysis:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save spectral analysis", self.saved_path or "Spectral analysis.kmzspectra",
            "KmzViewer spectral analysis (*.kmzspectra)")
        if path:
            if not path.lower().endswith(".kmzspectra"):
                path += ".kmzspectra"
            try:
                save_spectral_analysis(path, self.analysis)
                self.saved_path = path
                self.statusBar().showMessage("Saved the query, settings, reference comparisons, motif evidence and loaded structures.")
            except Exception as exc:
                QMessageBox.warning(self, "Could not save analysis", str(exc))

    def _sync(self):
        if not hasattr(self, "search_button"):
            return
        idle = not self._busy and not self.runtime.closing
        self.privacy.setText("DreaMS sends selected peaks + precursor m/z online. First use downloads 1.54 GB of public references. Local fragment filters do not restrict learned neighbors."
            if self.learned_enabled.isChecked() else
            "Local search. First use downloads about 220 MB of public references; your spectrum stays on this computer.")
        self.search_button.setEnabled(idle and self.query is not None)
        self.online_button.setEnabled(idle and self.query is not None)
        self.open_button.setEnabled(idle)
        self.save_button.setEnabled(idle and self.analysis is not None)
        self.scan_picker.setEnabled(idle and bool(self.scans))
        for widget in (self.fragment, self.noise, self.minimum, self.fragments, self.losses, self.expanded, self.learned_enabled):
            widget.setEnabled(idle)
        self.cancel_button.setVisible(self._busy)
        self.structure_button.setEnabled(idle and self.tabs.currentIndex() == 0 and bool(self._active_hit))
        self.link_button.setEnabled(bool(self._active_hit))

    def _set_busy(self, busy, message=""):
        self._busy = busy
        if message:
            self.statusBar().showMessage(message)
        self._sync()

    def _on_progress(self, done, total, message):
        self.statusBar().showMessage(message)

    def _on_failed(self, message):
        self.statusBar().showMessage(message)
        self.evidence.setText(message)

    def showEvent(self, event):
        super().showEvent(event)
        # Saved results can be populated before the window has its final plot
        # geometry. Refit once layout is visible so pixel padding stays valid.
        QTimer.singleShot(0, self._fit_initial_mirror)

    def _fit_initial_mirror(self):
        if not self.runtime.closing:
            self.mirror.autoRange()

    def closeEvent(self, event):
        for window in list(self._online_windows):
            window.close()
        if self._online_windows:
            event.ignore()
            QTimer.singleShot(30, self.close)
            return
        was_closed = self.runtime.closed
        self.runtime.close_event(event)
        if self.runtime.closed and not was_closed:
            self.closed.emit()
            self.deleteLater()
