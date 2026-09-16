from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QLabel

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.roll import RollAnalysisSidebar
from negpy.services.assets import rolls


def _sidebar(roll_names=(), analyzed=(), active_name=None):
    controller = MagicMock()
    controller.state = AppState()
    rolls_store = {}
    active_id = None
    for i, name in enumerate(roll_names):
        roll_id = f"roll-{i}"
        rolls_store[roll_id] = {"kind": "folder", "name": name}
        if name == active_name:
            active_id = roll_id
    controller.state.active_roll_id = active_id

    def get_global_setting(key, default=None):
        return rolls_store if key == rolls.ROLLS_KEY else default

    controller.session.repo.get_global_setting.side_effect = get_global_setting
    controller.session.repo.list_normalization_rolls.return_value = list(analyzed)
    return controller, RollAnalysisSidebar(controller)


def test_batch_and_roll_subtitles_are_gone(qapp):
    """The panel is one merged section now, not BATCH plus ROLL."""
    _, sidebar = _sidebar()
    texts = {lbl.text() for lbl in sidebar.findChildren(QLabel)}
    assert "BATCH" not in texts
    assert "ROLL" not in texts


def test_picker_and_apply_sit_above_the_average_toggles(qapp):
    _, sidebar = _sidebar()
    assert sidebar.layout.itemAt(0).widget() is sidebar.roll_combo
    assert sidebar.layout.itemAt(1).widget() is sidebar.roll_status_hint
    actions = sidebar.layout.itemAt(2).layout()
    assert actions.itemAt(0).widget() is sidebar.apply_roll_btn
    avg_row = sidebar.layout.itemAt(3).layout()
    assert avg_row.itemAt(0).widget() is sidebar.use_luma_avg_btn


def test_apply_is_the_panel_primary_action(qapp):
    _, sidebar = _sidebar()
    assert sidebar.apply_roll_btn.property("primary") is True


def test_picker_defaults_to_the_active_roll_and_lists_every_library_roll(qapp):
    _, sidebar = _sidebar(roll_names=["Portra 400", "Tri-X"], active_name="Tri-X")
    assert sidebar.roll_combo.selected_id() == "Tri-X"
    assert sidebar.roll_combo.line_edit().text() == "Tri-X"
    sidebar.roll_combo.set_selected_id("Portra 400")
    assert sidebar.roll_combo.line_edit().text() == "Portra 400"


def test_the_loaded_roll_is_pinned_first_in_the_dropdown(qapp):
    _, sidebar = _sidebar(roll_names=["Agfa", "Tri-X", "Velvia"], active_name="Velvia")
    ids = [item_id for _label, item_id, _search in sidebar.roll_combo._entries]
    assert ids == ["Velvia", "Agfa", "Tri-X"]


def test_dropdown_stays_alphabetical_without_an_active_roll(qapp):
    _, sidebar = _sidebar(roll_names=["Agfa", "Tri-X", "Velvia"])
    ids = [item_id for _label, item_id, _search in sidebar.roll_combo._entries]
    assert ids == ["Agfa", "Tri-X", "Velvia"]


def test_analyzed_rolls_get_a_tick_in_their_label(qapp):
    _, sidebar = _sidebar(roll_names=["Tri-X", "Portra 400"], analyzed=["Tri-X"], active_name="Tri-X")
    assert sidebar.roll_combo.line_edit().text() == "✓ Tri-X"
    sidebar.roll_combo.set_selected_id("Portra 400")
    assert sidebar.roll_combo.line_edit().text() == "Portra 400"


def test_apply_on_the_active_roll_runs_batch_normalization(qapp):
    controller, sidebar = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    sidebar.apply_roll_btn.click()
    controller.request_batch_normalization.assert_called_once()
    controller.apply_normalization_roll.assert_not_called()


def test_apply_on_a_different_roll_loads_its_saved_baseline(qapp):
    controller, sidebar = _sidebar(roll_names=["Tri-X", "Portra 400"], analyzed=["Tri-X"], active_name="Portra 400")
    sidebar.roll_combo.set_selected_id("Tri-X")
    sidebar.apply_roll_btn.click()
    controller.apply_normalization_roll.assert_called_once_with("Tri-X")
    controller.request_batch_normalization.assert_not_called()


def test_delete_is_disabled_without_a_saved_baseline_and_enabled_once_analyzed(qapp):
    controller, sidebar = _sidebar(roll_names=["Tri-X"], active_name="Tri-X")
    assert not sidebar.delete_roll_btn.isEnabled()
    controller.session.repo.list_normalization_rolls.return_value = ["Tri-X"]
    sidebar._update_delete_enabled()
    assert sidebar.delete_roll_btn.isEnabled()


def test_delete_does_nothing_without_a_selection(qapp):
    controller, sidebar = _sidebar()
    sidebar._on_delete_roll()
    controller.session.repo.delete_normalization_roll.assert_not_called()


def test_save_stores_under_the_picked_roll(qapp):
    controller, sidebar = _sidebar(roll_names=["Portra 400", "Tri-X"], active_name="Portra 400")
    sidebar.roll_combo.set_selected_id("Tri-X")
    sidebar._on_save_roll()
    controller.save_current_normalization_as_roll.assert_called_once_with("Tri-X")


def test_save_does_nothing_without_a_selection(qapp):
    controller, sidebar = _sidebar()
    sidebar._on_save_roll()
    controller.save_current_normalization_as_roll.assert_not_called()


def test_sync_ui_selects_the_config_roll_name_when_it_still_exists(qapp):
    controller, sidebar = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Tri-X"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == "Tri-X"


def test_sync_ui_falls_back_to_the_active_roll_when_the_saved_name_is_gone(qapp):
    """A deleted or renamed roll_name must not leave the picker pointed at nothing
    while a roll is loaded."""
    controller, sidebar = _sidebar(roll_names=["Portra 400"], active_name="Portra 400")
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Deleted Roll"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == "Portra 400"


def test_sync_ui_falls_back_to_blank_without_an_active_roll(qapp):
    controller, sidebar = _sidebar(roll_names=["Portra 400"])
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Deleted Roll"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == ""


def test_toggling_an_average_axis_still_reaches_the_controller(qapp):
    controller, sidebar = _sidebar()
    sidebar.use_luma_avg_btn.setChecked(True)
    new_cfg = controller.apply_config.call_args[0][0]
    assert new_cfg.process.use_luma_average is True
    assert new_cfg.process.roll_name is None


def test_active_roll_name_is_none_without_a_recognized_roll(qapp):
    _, sidebar = _sidebar()
    assert sidebar._active_roll_name() is None


def test_status_hint_warns_when_a_different_roll_is_picked(qapp):
    _, sidebar = _sidebar(roll_names=["Tri-X", "Portra 400"], active_name="Portra 400")
    sidebar.roll_combo.set_selected_id("Tri-X")
    sidebar._on_roll_picked()
    assert "Tri-X" in sidebar.roll_status_hint.text()
    assert "different roll" in sidebar.roll_status_hint.text()
    assert sidebar.roll_status_hint.property("hint") == "warning"


def test_status_hint_is_blank_when_the_active_roll_is_selected(qapp):
    _, sidebar = _sidebar(roll_names=["Portra 400"], active_name="Portra 400")
    assert sidebar.roll_status_hint.text() == ""
