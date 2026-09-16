"""_sync_roll_locks: per-card lock buttons and the roll_override_summary one-liner
that answers "roll-wide or this frame's own" for the whole Roll tab.

Stub-on-unbound-method, like test_right_panel_wiring.py: ControlsPanel pulls in every
sidebar in the app, so no test here constructs a real one.
"""

from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.controls_panel import ControlsPanel


def _panel_stub(*, active_roll_id="roll1", locked_cards=()) -> MagicMock:
    panel = MagicMock()
    panel._ROLL_CARD_LABELS = ControlsPanel._ROLL_CARD_LABELS
    panel.sensor_section = MagicMock()
    panel.demosaic_section = MagicMock()
    panel.process_section = MagicMock()
    panel.roll_override_summary = MagicMock()
    panel.controller.state.active_roll_id = active_roll_id
    panel.controller.roll_card_locked.side_effect = lambda key: key in locked_cards
    return panel


def test_sync_roll_locks_blank_summary_without_an_active_roll():
    panel = _panel_stub(active_roll_id=None)

    ControlsPanel._sync_roll_locks(panel)

    panel.roll_override_summary.setText.assert_called_once_with("")


def test_sync_roll_locks_says_follows_the_roll_when_nothing_is_overridden():
    panel = _panel_stub(locked_cards=())

    ControlsPanel._sync_roll_locks(panel)

    panel.roll_override_summary.setText.assert_called_once_with("Follows the roll")


def test_sync_roll_locks_names_every_overridden_card():
    panel = _panel_stub(locked_cards={"sensor", "process"})

    ControlsPanel._sync_roll_locks(panel)

    panel.roll_override_summary.setText.assert_called_once_with("This frame overrides: Calibration, Normalization")


def test_sync_roll_locks_sets_each_sections_lock_button():
    panel = _panel_stub(locked_cards={"demosaic"})

    ControlsPanel._sync_roll_locks(panel)

    panel.sensor_section.set_lock_button.assert_called_once_with(False, False)
    panel.demosaic_section.set_lock_button.assert_called_once_with(True, True)
    panel.process_section.set_lock_button.assert_called_once_with(False, False)
