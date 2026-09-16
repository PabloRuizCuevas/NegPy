"""RollAnalysisSidebar: a plain picker over the library's rolls. Picking one is the
whole action -- it loads that roll's saved Batch Analysis baseline immediately, no
separate Apply. Batch Analysis itself (the metering run that fills the tick in) is a
Library action on the roll, tested in test_library_tree.py, not here."""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.roll import RollAnalysisSidebar
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.assets import rolls


def _repo() -> MagicMock:
    """A repository whose global settings live in a dict, so a write is readable back."""
    repo = MagicMock(spec=StorageRepository)
    store: dict = {}
    repo.get_global_setting.side_effect = lambda key, default=None: store.get(key, default)
    repo.save_global_setting.side_effect = lambda key, value: store.__setitem__(key, value)
    return repo


def _sidebar(roll_names=(), analyzed=(), active_name=None):
    """Builds a real repo with one virtual roll per name, a controller stub pointed at
    it, and the sidebar under test. Returns (controller, sidebar, {name: roll_id})."""
    repo = _repo()
    ids = {}
    for name in roll_names:
        roll_id = rolls.create_virtual_roll(repo, name, [])
        ids[name] = roll_id
        if name in analyzed:
            rolls.set_roll_normalization(repo, roll_id, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))

    controller = MagicMock()
    controller.session.repo = repo
    controller.state = AppState()
    controller.state.active_roll_id = ids.get(active_name)
    return controller, RollAnalysisSidebar(controller), ids


def test_picker_defaults_to_the_active_roll_and_lists_every_library_roll(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Portra 400", "Tri-X"], active_name="Tri-X")
    assert sidebar.roll_combo.selected_id() == ids["Tri-X"]
    assert sidebar.roll_combo.line_edit().text() == "Tri-X"
    sidebar.roll_combo.set_selected_id(ids["Portra 400"])
    assert sidebar.roll_combo.line_edit().text() == "Portra 400"


def test_the_loaded_roll_is_pinned_first_in_the_dropdown(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Agfa", "Tri-X", "Velvia"], active_name="Velvia")
    listed_ids = [item_id for _label, item_id, _search in sidebar.roll_combo._entries]
    assert listed_ids == [ids["Velvia"], ids["Agfa"], ids["Tri-X"]]


def test_dropdown_stays_alphabetical_without_an_active_roll(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Agfa", "Tri-X", "Velvia"])
    listed_ids = [item_id for _label, item_id, _search in sidebar.roll_combo._entries]
    assert listed_ids == [ids["Agfa"], ids["Tri-X"], ids["Velvia"]]


def test_analyzed_rolls_get_a_tick_in_their_label(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], analyzed=["Tri-X"], active_name="Tri-X")
    assert sidebar.roll_combo.line_edit().text() == "✓ Tri-X"
    sidebar.roll_combo.set_selected_id(ids["Portra 400"])
    assert sidebar.roll_combo.line_edit().text() == "Portra 400"


def test_picking_a_roll_loads_its_saved_baseline(qapp):
    controller, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], analyzed=["Tri-X"], active_name="Portra 400")

    sidebar._on_roll_picked(ids["Tri-X"])

    controller.apply_normalization_roll.assert_called_once_with(ids["Tri-X"])


def test_picking_blank_does_not_touch_the_controller(qapp):
    controller, sidebar, _ids = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")

    sidebar._on_roll_picked("")

    controller.apply_normalization_roll.assert_not_called()


def test_sync_ui_selects_the_config_roll_name_when_it_still_exists(qapp):
    controller, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Tri-X"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == ids["Tri-X"]


def test_sync_ui_falls_back_to_the_active_roll_when_the_saved_name_is_gone(qapp):
    """A deleted or renamed roll_name must not leave the picker pointed at nothing
    while a roll is loaded."""
    controller, sidebar, ids = _sidebar(roll_names=["Portra 400"], active_name="Portra 400")
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Deleted Roll"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == ids["Portra 400"]


def test_sync_ui_falls_back_to_blank_without_an_active_roll(qapp):
    controller, sidebar, _ids = _sidebar(roll_names=["Portra 400"])
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Deleted Roll"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == ""


def test_status_hint_warns_when_a_different_roll_is_picked(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")

    sidebar._on_roll_picked(ids["Tri-X"])

    assert "Tri-X" in sidebar.roll_status_hint.text()
    assert "different roll" in sidebar.roll_status_hint.text()
    assert sidebar.roll_status_hint.property("hint") == "warning"


def test_status_hint_is_blank_when_the_active_roll_is_picked(qapp):
    _, sidebar, ids = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")

    sidebar._on_roll_picked(ids["Portra 400"])

    assert sidebar.roll_status_hint.text() == ""


def test_status_hint_is_blank_without_an_active_roll(qapp):
    _, sidebar, _ids = _sidebar(roll_names=["Portra 400"])
    assert sidebar.roll_status_hint.text() == ""
