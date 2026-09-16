from typing import Optional

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, set_hint_kind
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.services.assets import rolls

# Prefixes a roll's label once it has a saved Batch Analysis baseline -- the field
# itself is the "analyzed" indicator, so the search text ignores it (built from the
# plain name).
_TICK = "✓ "


class RollAnalysisSidebar(BaseSidebar):
    """
    Which roll's Batch Analysis baseline this frame's Use Luma/Color Average axes
    borrow: a searchable picker over every roll in your library, ticked once it has
    one. Picking a roll loads its baseline immediately -- there is no separate Apply.
    Batch Analysis itself (the metering run that fills the tick in) is a Library
    action on the roll, not a value this card edits.
    """

    def _init_ui(self) -> None:
        self.roll_combo = SearchableGearCombo(placeholder="Search rolls…")
        self.roll_combo.setToolTip("Picking a roll loads its saved Batch Analysis baseline onto the loaded files.")
        self.layout.addWidget(self.roll_combo)

        self.roll_status_hint = hint_label("", "muted")
        self.layout.addWidget(self.roll_status_hint)

        self._roll_sync_key = None
        self._refresh_rolls(force=True)
        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.roll_combo.selection_changed.connect(self._on_roll_picked)
        self.sync_ui()

    def _on_roll_picked(self, roll_id: str) -> None:
        """Picking a roll is the whole action: it loads that roll's saved baseline
        onto the currently loaded files. A no-op if it has never been analyzed."""
        if roll_id:
            self.controller.apply_normalization_roll(roll_id)
        self._update_roll_status_hint(self.controller.state.active_roll_id, roll_id)

    def _name_for_id(self, roll_id: str) -> str:
        entry = rolls.roll_for_id(self.controller.session.repo, roll_id)
        return entry["name"] if entry else roll_id

    def _refresh_rolls(self, *, force: bool = False) -> None:
        """
        Rebuilds the picker from every library roll, skipping a rebuild mid-search
        (SearchableGearCombo.is_editing) and one the roll set and selection don't need.
        The loaded roll is pinned first in the dropdown, ahead of the alphabetical
        rest, since it is the default choice.
        """
        if not force and self.roll_combo.is_editing():
            return
        repo = self.controller.session.repo
        active_id = self.controller.state.active_roll_id
        listed = rolls.all_rolls_sorted(repo)
        analyzed = {rid for rid, _entry in listed if rolls.roll_normalization(repo, rid)}
        conf = self.state.config.process
        selected = active_id or ""
        if conf.roll_name:
            matching = next((rid for rid, entry in listed if entry.get("name") == conf.roll_name), None)
            if matching:
                selected = matching
        key = (tuple(rid for rid, _entry in listed), tuple(sorted(analyzed)), selected, active_id)
        if not force and key == self._roll_sync_key:
            return
        self._roll_sync_key = key
        ordered = [pair for pair in listed if pair[0] == active_id] + [pair for pair in listed if pair[0] != active_id]
        entries = [(f"{_TICK}{entry.get('name', '')}" if rid in analyzed else entry.get("name", ""), rid) for rid, entry in ordered]
        self.roll_combo.set_labeled_items(entries, selected, search_fn=lambda _label, item_id: self._name_for_id(item_id))
        self._update_roll_status_hint(active_id, selected)

    def _update_roll_status_hint(self, active_id: Optional[str], selected_id: str) -> None:
        """Flags a baseline picked from a roll other than the one loaded."""
        if active_id and selected_id and selected_id != active_id:
            set_hint_kind(self.roll_status_hint, "warning")
            self.roll_status_hint.setText(f'Using "{self._name_for_id(selected_id)}", a baseline saved for a different roll')
        else:
            self.roll_status_hint.setText("")

    def sync_ui(self) -> None:
        self.block_signals(True)
        try:
            self._refresh_rolls()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all buttons.
        """
        self.roll_combo.blockSignals(blocked)
