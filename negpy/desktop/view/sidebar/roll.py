from typing import Optional

from PyQt6.QtWidgets import (
    QHBoxLayout,
)

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, labeled_action, set_hint_kind
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.features.process.models import invalidate_local_bounds
from negpy.services.assets import rolls

# Prefixes a roll's label once it has a saved baseline -- the field itself is the
# "analyzed" indicator, so the search text ignores it (built from the plain name).
_TICK = "✓ "


class RollAnalysisSidebar(BaseSidebar):
    """
    Roll-wide normalization: a searchable picker over every roll in your library,
    ticked once it has a saved baseline, one Apply action, then the per-axis average
    toggles.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.process

        self.roll_combo = SearchableGearCombo(placeholder="Search rolls…")
        self.roll_combo.setToolTip("Apply on the loaded roll scans it fresh; picking another roll loads that roll's saved baseline.")
        self.layout.addWidget(self.roll_combo)

        self.roll_status_hint = hint_label("", "muted")
        self.layout.addWidget(self.roll_status_hint)

        roll_actions = QHBoxLayout()
        self.apply_roll_btn = labeled_action(
            "fa5s.check",
            " Apply",
            "Apply the picked roll: the loaded roll re-scans its files for a fresh baseline, "
            "another roll loads its stored bounds and balance",
            primary=True,
        )
        self.save_roll_btn = self._labeled_action(
            "fa5s.save", " Save", "Save the current bounds and balance as the picked roll's baseline, to reuse later"
        )
        self.delete_roll_btn = self._labeled_action("fa5s.trash", " Delete", "Clear the picked roll's saved baseline")

        roll_actions.addWidget(self.apply_roll_btn)
        roll_actions.addWidget(self.save_roll_btn)
        roll_actions.addWidget(self.delete_roll_btn)
        self.layout.addLayout(roll_actions)

        self._roll_sync_key = None
        self._refresh_rolls(force=True)

        avg_row = QHBoxLayout()
        self.use_luma_avg_btn = self._small_toggle(
            "mdi6.film",
            "Use Luma Average",
            conf.use_luma_average,
            "Take the tonal-range (black/white-point) baseline from the picked roll; color still re-derives per frame",
        )

        self.use_color_avg_btn = self._small_toggle(
            "mdi6.film",
            "Use Color Average",
            conf.use_color_average,
            "Take the per-channel color-balance baseline from the picked roll; luma range still re-derives per frame",
        )

        avg_row.addWidget(self.use_luma_avg_btn)
        avg_row.addWidget(self.use_color_avg_btn)
        self.layout.addLayout(avg_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.apply_roll_btn.clicked.connect(self._on_apply_roll)
        self.use_luma_avg_btn.toggled.connect(self._on_use_luma_average_toggled)
        self.use_color_avg_btn.toggled.connect(self._on_use_color_average_toggled)

        self.save_roll_btn.clicked.connect(self._on_save_roll)
        self.delete_roll_btn.clicked.connect(self._on_delete_roll)
        self.roll_combo.selection_changed.connect(self._on_roll_picked)
        self.sync_ui()

    def _on_use_luma_average_toggled(self, checked: bool) -> None:
        """Toggle the roll-wide luma (tonal-range) baseline for this axis only."""
        self._toggle_roll_axis(use_luma_average=checked)

    def _on_use_color_average_toggled(self, checked: bool) -> None:
        """Toggle the roll-wide color-balance baseline for this axis only."""
        self._toggle_roll_axis(use_color_average=checked)

    def _toggle_roll_axis(self, **axis: bool) -> None:
        """
        Flip one roll-average axis. The other axis re-derives per frame, so we clear
        the cached local bounds to force a fresh analysis, and drop roll_name (the
        baseline is no longer applied as a named whole).
        """
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            roll_name=None,
            **axis,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.sync_ui()

    def _on_apply_roll(self) -> None:
        """The loaded roll re-runs Batch Analysis on its files; another roll loads its
        saved baseline instead -- there is nothing of its own loaded to scan."""
        selected = self.roll_combo.selected_id()
        if selected and selected != self._active_roll_name():
            self.controller.apply_normalization_roll(selected)
        else:
            self.controller.request_batch_normalization()

    def _active_roll_name(self) -> Optional[str]:
        """The loaded folder/virtual roll's own name, or None with no roll recognized
        (a loose file selection) -- distinct from process.roll_name, the saved
        baseline picked in this section."""
        roll_id = self.controller.state.active_roll_id
        if not roll_id:
            return None
        entry = rolls.roll_for_id(self.controller.session.repo, roll_id)
        return entry["name"] if entry else None

    def _on_roll_picked(self, *_args) -> None:
        self._update_delete_enabled()
        self._update_roll_status_hint(self._active_roll_name(), self.roll_combo.selected_id())

    def _refresh_rolls(self, *, force: bool = False) -> None:
        """
        Rebuilds the picker from every library roll, skipping a rebuild mid-search
        (SearchableGearCombo.is_editing) and one the roll set and selection don't need.
        The loaded roll is pinned first in the dropdown, ahead of the alphabetical rest,
        since it is the default choice.
        """
        if not force and self.roll_combo.is_editing():
            return
        repo = self.controller.session.repo
        names = [entry.get("name", "") for _roll_id, entry in rolls.all_rolls_sorted(repo)]
        analyzed = set(repo.list_normalization_rolls())
        active_name = self._active_roll_name()
        conf = self.state.config.process
        selected = conf.roll_name if conf.roll_name in names else (active_name or "")
        key = (tuple(names), tuple(sorted(analyzed)), selected, active_name)
        if not force and key == self._roll_sync_key:
            return
        self._roll_sync_key = key
        ordered = [active_name, *(n for n in names if n != active_name)] if active_name in names else names
        entries = [(f"{_TICK}{name}" if name in analyzed else name, name) for name in ordered]
        self.roll_combo.set_labeled_items(entries, selected, search_fn=lambda _label, item_id: item_id)
        self._update_delete_enabled()
        self._update_roll_status_hint(active_name, selected)

    def _update_roll_status_hint(self, active_name: Optional[str], selected_name: str) -> None:
        """Flags a baseline picked from a roll other than the one loaded -- the tick in
        the field already says whether the loaded roll itself has one saved."""
        if active_name and selected_name and selected_name != active_name:
            set_hint_kind(self.roll_status_hint, "warning")
            self.roll_status_hint.setText(f'Using "{selected_name}", a baseline saved for a different roll')
        else:
            self.roll_status_hint.setText("")

    def _update_delete_enabled(self, *_args) -> None:
        """Delete clears a saved baseline, so it needs a roll that has one."""
        name = self.roll_combo.selected_id()
        self.delete_roll_btn.setEnabled(bool(name) and name in self.controller.session.repo.list_normalization_rolls())

    def _on_save_roll(self) -> None:
        """Saves the current bounds and balance as the picked roll's baseline."""
        name = self.roll_combo.selected_id()
        if not name:
            return
        self.controller.save_current_normalization_as_roll(name)
        self._refresh_rolls(force=True)

    def _on_delete_roll(self) -> None:
        """Clears the picked roll's saved baseline; the roll itself stays."""
        name = self.roll_combo.selected_id()
        if not name:
            return
        if confirm_delete_named(
            self,
            "Saved Baseline",
            name,
            informative="The frames keep their current look; only the saved baseline goes.",
        ):
            self.controller.session.repo.delete_normalization_roll(name)
            self._refresh_rolls(force=True)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            self.use_luma_avg_btn.setChecked(conf.use_luma_average)
            self.use_color_avg_btn.setChecked(conf.use_color_average)
            self._refresh_rolls()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all buttons.
        """
        widgets = [
            self.apply_roll_btn,
            self.use_luma_avg_btn,
            self.use_color_avg_btn,
            self.roll_combo,
            self.save_roll_btn,
            self.delete_roll_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
