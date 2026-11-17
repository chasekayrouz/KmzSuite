"""Searchable trace selection for session-only plot and spectrum actions."""
from collections import Counter

import numpy as np
from PySide6.QtCore import QEvent, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGridLayout, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from kmzcore.theme import color_icon


class ManageChromatogramsDialog(QDialog):
    action_requested = Signal(str, object)

    def __init__(self, session, parent=None, *, spectrum_file_ids=(), dad_file_ids=(), dad_enabled=()):
        super().__init__(parent)
        self.setWindowTitle("Remove/Show/Hide Chromatograms")
        self.resize(940, 660)
        self.setMinimumSize(780, 530)
        self.session = session
        self._leaves = {}
        self._search_text = {}
        self._file_ids = {}
        self._branches = []
        self._members = {}
        self._matches = set()

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by type, m/z, wavelength or file…")
        self.search.setAccessibleName("Filter chromatograms to manage")
        self.search.setClearButtonEnabled(True)
        self.search.installEventFilter(self)
        self.tree = QTreeWidget()
        self.tree.setAccessibleName("Chromatogram action groups")
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Chromatogram / source file", "Traces", "Plot", "MS1 spectra", "DAD spectra"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setUniformRowHeights(True)
        self.tree.setAllColumnsShowFocus(True)
        self.select_shown = QPushButton("Select matching")
        self.clear_selection = QPushButton("Clear selection")
        self.select_shown.clicked.connect(lambda: self._select(self._matches, True))
        self.clear_selection.clicked.connect(lambda: self._select(set(self._leaves), False))
        self.count = QLabel()
        self.count.setWordWrap(True)
        self.empty = QLabel("No chromatograms match this filter.")
        controls = QHBoxLayout()
        controls.addWidget(self.select_shown)
        controls.addWidget(self.clear_selection)
        controls.addStretch(1)
        actions = QGridLayout()
        self.action_buttons = {}
        for row, (target, choices) in enumerate((
            ("Chromatograms", (("show_plots", "Show plots"), ("hide_plots", "Hide plots"), ("remove", "Remove chromatograms"))),
            ("MS1 spectra", (("show_ms1", "Show MS1 spectra"), ("hide_ms1", "Hide MS1 spectra"), ("toggle_ms1", "Toggle MS1 spectra"))),
            ("DAD spectra", (("show_dad", "Show DAD spectra"), ("hide_dad", "Hide DAD spectra"), ("toggle_dad", "Toggle DAD spectra"))),
        )):
            actions.addWidget(QLabel(target), row, 0)
            for column, (action, label) in enumerate(choices, 1):
                button = QPushButton(label)
                button.clicked.connect(lambda _checked=False, action=action: self._apply(action))
                button.setAccessibleName(label)
                button.setToolTip("Remove checked traces from this session; source files and spectra stay open."
                                  if action == "remove" else "Apply to checked traces."
                                  if action.endswith("plots") else
                                  "Apply once per eligible source file represented by checked traces. "
                                  "Toggle flips each file's current spectrum eye state.")
                actions.addWidget(button, row, column)
                self.action_buttons[action] = button
        self.remove_button = self.action_buttons["remove"]
        for button in (*self.action_buttons.values(), self.select_shown, self.clear_selection):
            button.setAutoDefault(False)
            button.installEventFilter(self)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setDefault(True)
        buttons.rejected.connect(self.reject)
        note = QLabel("Check a type or extraction group to select its matching traces; expand it to choose individual files. "
                      "Plot actions affect checked traces. Spectrum actions affect their source files, independently of plots.")
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(self.search)
        layout.addLayout(controls)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.empty)
        layout.addWidget(self.count)
        layout.addLayout(actions)
        footer = QLabel("Actions apply immediately and keep this dialog open. Close does not undo changes. Source files stay open.")
        footer.setWordWrap(True)
        layout.addWidget(footer)
        layout.addWidget(buttons)
        self.tree.itemChanged.connect(self._checked)
        self.search.textChanged.connect(self._filter)
        self.refresh(spectrum_file_ids, dad_file_ids, dad_enabled)

    def refresh(self, spectrum_file_ids, dad_file_ids, dad_enabled):
        selected = self.selected_trace_ids()
        expanded = {branch.data(0, Qt.ItemDataRole.UserRole): branch.isExpanded() for branch in self._branches}
        scroll = self.tree.verticalScrollBar().value()
        current = self.tree.currentItem()
        current_key = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self._spectrum_file_ids = set(spectrum_file_ids)
        self._dad_file_ids = set(dad_file_ids)
        self._ms1_available = {fid for fid, loaded in self.session.files.items()
                               if np.isfinite(loaded.scans.rt_min).any()}
        self._dad_available = set(dad_enabled) & self.session.files.keys()
        with QSignalBlocker(self.tree):
            self.tree.clear()
            self._leaves.clear()
            self._search_text.clear()
            self._file_ids.clear()
            self._branches.clear()
            self._members.clear()
            self._populate()
            for tid in selected & self._leaves.keys():
                self._leaves[tid].setCheckState(0, Qt.CheckState.Checked)
            self._filter(self.search.text())
            for item in (*self._branches, *self._leaves.values()):
                key = item.data(0, Qt.ItemDataRole.UserRole)
                if key == current_key:
                    self.tree.setCurrentItem(item)
                if key in expanded:
                    item.setExpanded(expanded[key])
        self.tree.verticalScrollBar().setValue(scroll)

    def _populate(self):
        families = {}
        names = {"tic": "TIC — Total ion", "bpc": "BPC — Base peak", "eic": "EIC — Extracted ion",
                 "twc": "TWC — Total wavelength", "ewc": "EWC — Extracted wavelength"}
        for key, rows in self.session.trace_groups():
            first = rows[0][1]
            if first.kind not in families:
                family = self._branch(None, names.get(first.kind, first.kind.upper()))
                family.setData(0, Qt.ItemDataRole.UserRole, "family:" + first.kind)
                family.setExpanded(True)
                families[first.kind] = family
            group = self._branch(families[first.kind], first.group_label)
            group.setData(0, Qt.ItemDataRole.UserRole, key)
            group.setToolTip(0, _settings(first))
            copies = Counter()
            for loaded, trace in rows:
                copies[loaded.file_id] += 1
                suffix = f" (copy {copies[loaded.file_id]})" if copies[loaded.file_id] > 1 else ""
                ms1 = ("Visible" if loaded.file_id in self._spectrum_file_ids else "Hidden") if loaded.file_id in self._ms1_available else "Unavailable"
                dad = ("Visible" if loaded.file_id in self._dad_file_ids else "Hidden") if loaded.file_id in self._dad_available else "Unavailable"
                leaf = QTreeWidgetItem(group, [loaded.name + suffix, "", "Visible" if trace.visible else "Hidden", ms1, dad])
                for column, detector in ((3, "MS1"), (4, "DAD")):
                    leaf.setToolTip(column, f"{detector} spectrum eye for {loaded.name}; shared by every trace from this file."
                                    + (" Extract a wavelength chromatogram to enable DAD spectra." if column == 4 and dad == "Unavailable" else ""))
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.CheckState.Unchecked)
                leaf.setData(0, Qt.ItemDataRole.UserRole, trace.trace_id)
                leaf.setIcon(0, color_icon(trace.color))
                leaf.setToolTip(0, loaded.path + "\n" + _settings(trace))
                self._leaves[trace.trace_id] = leaf
                for branch in (group, group.parent()):
                    self._members[id(branch)].add(trace.trace_id)
                self._file_ids[trace.trace_id] = loaded.file_id
                self._search_text[trace.trace_id] = " ".join((names.get(trace.kind, trace.kind), trace.group_label,
                    _settings(trace), loaded.path, loaded.name, leaf.text(2))).casefold()

    def _branch(self, parent, label):
        item = QTreeWidgetItem(parent or self.tree, [label])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Unchecked)
        self._branches.append(item)
        self._members[id(item)] = set()
        return item

    def selected_trace_ids(self):
        return {tid for tid, item in self._leaves.items() if item.checkState(0) == Qt.CheckState.Checked}

    def _select(self, trace_ids, checked):
        with QSignalBlocker(self.tree):
            for tid in trace_ids:
                self._leaves[tid].setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._sync()

    def _checked(self, item, column):
        if column != 0:
            return
        # Search-hidden rows never inherit a bulk checkbox operation. Existing
        # selections outside the search remain explicit in the footer count.
        targets = self._matches & self._members.get(id(item), {item.data(0, Qt.ItemDataRole.UserRole)})
        self._select(targets, item.checkState(0) == Qt.CheckState.Checked)

    def _filter(self, text):
        words = text.casefold().split()
        self._matches = {tid for tid, value in self._search_text.items() if all(word in value for word in words)}
        with QSignalBlocker(self.tree):
            for tid, leaf in self._leaves.items():
                leaf.setHidden(tid not in self._matches)
            self._sync()
        # Show matching file rows while searching; return to compact groups when cleared.
        for branch in self._branches:
            branch.setExpanded(bool(words) or branch.parent() is None)

    def _sync(self):
        for branch in reversed(self._branches):
            leaves = self._members[id(branch)]
            shown = [tid for tid in leaves if tid in self._matches]
            checked = sum(self._leaves[tid].checkState(0) == Qt.CheckState.Checked for tid in shown)
            branch.setHidden(not shown)
            state = (Qt.CheckState.Unchecked if not checked else Qt.CheckState.Checked if checked == len(shown)
                     else Qt.CheckState.PartiallyChecked)
            branch.setCheckState(0, state)
            branch.setText(1, str(len(leaves)) if len(shown) == len(leaves) else f"{len(shown)} of {len(leaves)}")
        selected = self.selected_trace_ids()
        file_ids = {self._file_ids[tid] for tid in selected}
        file_count = len(file_ids)
        outside = len(selected - self._matches)
        file_word = "file" if file_count == 1 else "files"
        trace_word = "chromatogram" if len(selected) == 1 else "chromatograms"
        self.count.setText(f"{len(selected)} of {len(self._leaves)} chromatograms selected across {file_count} {file_word}."
                           + (f" {outside} selected outside this search." if outside else "")
                           + f"\nSpectrum targets: {len(file_ids & self._ms1_available)} MS1 files; {len(file_ids & self._dad_available)} DAD files.")
        self.remove_button.setText(f"Remove {len(selected)} {trace_word}")
        for action, button in self.action_buttons.items():
            button.setEnabled(bool(file_ids & self._ms1_available) if action.endswith("ms1") else
                              bool(file_ids & self._dad_available) if action.endswith("dad") else bool(selected))
        self.select_shown.setEnabled(bool(self._matches))
        self.clear_selection.setEnabled(bool(selected))
        self.empty.setText("No chromatograms match this filter." if self._leaves else
                           "No chromatograms remain in this session. Source files stay open.")
        self.empty.setVisible(not self._matches)

    def _apply(self, action):
        if self.action_buttons[action].isEnabled():
            self.action_requested.emit(action, self.selected_trace_ids())

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if watched is self.search:
                return True
            if isinstance(watched, QPushButton):
                watched.click()
                return True
        return super().eventFilter(watched, event)


def _settings(trace):
    """Expose full precision where rounded group labels could coincide."""
    key = trace.extraction_key
    if trace.kind == 'eic' and key:
        if key[2] == 'range':
            return f"MS1, {key[1]} polarity; m/z {key[3]!r} through {key[4]!r}, inclusive."
        return f"MS1, {key[1]} polarity; m/z {', '.join(map(repr, key[3]))}; ±{key[4]!r} ppm."
    if trace.kind == 'ewc' and key:
        reference = 'none' if key[3] is None else f'{key[3][0]!r} nm, bandwidth {key[3][1]!r} nm'
        return f"EWC {key[1]!r} nm, bandwidth {key[2]!r} nm; reference: {reference}."
    return trace.detail
