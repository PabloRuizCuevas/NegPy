from dataclasses import replace
from unittest.mock import MagicMock

from PyQt6.QtWidgets import QLabel

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.roll import _CURRENT_ROLL_ID, _CURRENT_ROLL_LABEL, RollAnalysisSidebar


def _sidebar(rolls=()):
    controller = MagicMock()
    controller.state = AppState()
    controller.session.repo.list_normalization_rolls.return_value = list(rolls)
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


def test_picker_defaults_to_current_roll_and_lists_saved_rolls(qapp):
    _, sidebar = _sidebar(rolls=["Portra 400", "Tri-X"])
    assert sidebar.roll_combo.selected_id() == _CURRENT_ROLL_ID
    assert sidebar.roll_combo.line_edit().text() == _CURRENT_ROLL_LABEL
    assert sidebar.roll_combo.selected_id() != "Tri-X"
    sidebar.roll_combo.set_selected_id("Tri-X")
    assert sidebar.roll_combo.line_edit().text() == "Tri-X"


def test_apply_on_current_roll_runs_batch_normalization(qapp):
    controller, sidebar = _sidebar(rolls=["Tri-X"])
    sidebar.apply_roll_btn.click()
    controller.request_batch_normalization.assert_called_once()
    controller.apply_normalization_roll.assert_not_called()


def test_apply_on_a_saved_roll_loads_it_instead_of_scanning(qapp):
    controller, sidebar = _sidebar(rolls=["Tri-X"])
    sidebar.roll_combo.set_selected_id("Tri-X")
    sidebar.apply_roll_btn.click()
    controller.apply_normalization_roll.assert_called_once_with("Tri-X")
    controller.request_batch_normalization.assert_not_called()


def test_delete_is_disabled_on_current_roll_and_enabled_on_a_saved_one(qapp):
    _, sidebar = _sidebar(rolls=["Tri-X"])
    assert not sidebar.delete_roll_btn.isEnabled()
    sidebar.roll_combo.set_selected_id("Tri-X")
    sidebar._update_delete_enabled()
    assert sidebar.delete_roll_btn.isEnabled()


def test_delete_guards_against_the_current_roll_sentinel(qapp):
    controller, sidebar = _sidebar(rolls=["Tri-X"])
    sidebar._on_delete_roll()
    controller.session.repo.delete_normalization_roll.assert_not_called()


def test_save_rejects_the_reserved_current_roll_name(qapp, monkeypatch):
    controller, sidebar = _sidebar()
    monkeypatch.setattr("negpy.desktop.view.sidebar.roll.QInputDialog.getText", lambda *a, **k: (_CURRENT_ROLL_LABEL, True))
    warned = []
    monkeypatch.setattr("negpy.desktop.view.sidebar.roll.QMessageBox.warning", lambda *a, **k: warned.append(a))
    sidebar._on_save_roll()
    controller.save_current_normalization_as_roll.assert_not_called()
    assert warned


def test_save_stores_under_the_typed_name_and_selects_it(qapp, monkeypatch):
    controller, sidebar = _sidebar(rolls=["Tri-X"])
    monkeypatch.setattr("negpy.desktop.view.sidebar.roll.QInputDialog.getText", lambda *a, **k: ("Portra 400", True))
    controller.session.repo.list_normalization_rolls.return_value = ["Portra 400", "Tri-X"]
    sidebar._on_save_roll()
    controller.save_current_normalization_as_roll.assert_called_once_with("Portra 400")
    assert sidebar.roll_combo.selected_id() == "Portra 400"


def test_sync_ui_selects_the_config_roll_name_when_it_still_exists(qapp):
    controller, sidebar = _sidebar(rolls=["Tri-X"])
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Tri-X"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == "Tri-X"


def test_sync_ui_falls_back_to_current_roll_when_the_saved_name_is_gone(qapp):
    """A deleted or never-saved roll_name must not leave the picker pointed at nothing."""
    controller, sidebar = _sidebar(rolls=[])
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Deleted Roll"))
    sidebar.sync_ui()
    assert sidebar.roll_combo.selected_id() == _CURRENT_ROLL_ID


def test_toggling_an_average_axis_still_reaches_the_controller(qapp):
    controller, sidebar = _sidebar()
    sidebar.use_luma_avg_btn.setChecked(True)
    new_cfg = controller.apply_config.call_args[0][0]
    assert new_cfg.process.use_luma_average is True
    assert new_cfg.process.roll_name is None


def _with_active_roll(sidebar, controller, monkeypatch, name="08_penf_portra_160_madrid"):
    controller.state.active_roll_id = "abc123"
    monkeypatch.setattr("negpy.desktop.view.sidebar.roll.rolls.roll_for_id", lambda repo, roll_id: {"name": name})
    sidebar.sync_ui()


def test_active_roll_name_is_none_without_a_recognized_roll(qapp):
    _, sidebar = _sidebar()
    assert sidebar._active_roll_name() is None


def test_combo_current_roll_entry_shows_the_active_roll_s_own_name(qapp, monkeypatch):
    controller, sidebar = _sidebar(rolls=["Tri-X"])
    _with_active_roll(sidebar, controller, monkeypatch)
    assert sidebar.roll_combo.line_edit().text() == "08_penf_portra_160_madrid"
    assert sidebar.roll_combo.selected_id() == _CURRENT_ROLL_ID


def test_status_hint_flags_a_baseline_already_saved_for_this_roll(qapp, monkeypatch):
    controller, sidebar = _sidebar(rolls=["08_penf_portra_160_madrid"])
    _with_active_roll(sidebar, controller, monkeypatch)
    assert sidebar.roll_status_hint.text() == "Analyzed and saved for this roll"
    assert sidebar.roll_status_hint.property("hint") == "success"


def test_status_hint_warns_when_applying_a_different_roll_s_baseline(qapp, monkeypatch):
    controller, sidebar = _sidebar(rolls=["Tri-X"])
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, roll_name="Tri-X"))
    _with_active_roll(sidebar, controller, monkeypatch)
    assert "Tri-X" in sidebar.roll_status_hint.text()
    assert "different roll" in sidebar.roll_status_hint.text()
    assert sidebar.roll_status_hint.property("hint") == "warning"


def test_status_hint_is_blank_for_a_roll_never_analyzed(qapp, monkeypatch):
    controller, sidebar = _sidebar(rolls=[])
    _with_active_roll(sidebar, controller, monkeypatch)
    assert sidebar.roll_status_hint.text() == ""


def test_save_dialog_prefills_the_active_roll_s_own_name(qapp, monkeypatch):
    controller, sidebar = _sidebar()
    _with_active_roll(sidebar, controller, monkeypatch)
    captured = {}

    def fake_get_text(*args, **kwargs):
        captured["text"] = kwargs.get("text")
        return "", False

    monkeypatch.setattr("negpy.desktop.view.sidebar.roll.QInputDialog.getText", fake_get_text)
    sidebar._on_save_roll()
    assert captured["text"] == "08_penf_portra_160_madrid"
