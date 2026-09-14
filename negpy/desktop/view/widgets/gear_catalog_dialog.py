"""Picker for the "+" flow on a Gear Library category: pick a built-in model to make
personal, or fall back to a custom entry."""

from __future__ import annotations

from typing import Callable, Sequence

from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from negpy.desktop.view.styles.templates import field_label, hint_label, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo


class GearCatalogDialog(QDialog):
    """Search the shipped catalog for one category. Accepting with a pick returns its id
    (the caller clones it into a personal copy); Add Custom accepts with no pick, asking
    for a blank entry instead."""

    def __init__(self, parent, singular: str, catalog: Sequence, label_fn: Callable, placeholder: str):
        super().__init__(parent)
        self._custom = False
        self.setWindowTitle(f"Add {singular}")
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_xl)

        root.addWidget(field_label(singular))
        self.combo = SearchableGearCombo(placeholder=placeholder)
        self.combo.set_gear_items(catalog, "", label_fn)
        self.combo.selection_changed.connect(self._update_add_enabled)
        root.addWidget(self.combo)
        root.addWidget(hint_label(f"Pick the {singular.lower()} you own from the built-in list, or add a custom one."))

        root.addLayout(self._build_footer())
        self._update_add_enabled()

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        custom_btn = QPushButton("Add Custom")
        custom_btn.clicked.connect(self._pick_custom)
        row.addWidget(custom_btn)
        row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.accept)
        row.addWidget(cancel_btn)
        row.addWidget(self.add_btn)
        pin_dialog_default(self.add_btn, cancel_btn, custom_btn)
        return row

    def _pick_custom(self) -> None:
        self._custom = True
        self.accept()

    def _update_add_enabled(self, *_args) -> None:
        self.add_btn.setEnabled(bool(self.combo.selected_id()))

    def wants_custom(self) -> bool:
        return self._custom

    def selected_id(self) -> str:
        return "" if self._custom else self.combo.selected_id()
