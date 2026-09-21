"""_sync_roll_locks: per-card lock buttons and the roll_override_summary one-liner
that answers "roll-wide or this frame's own" for the whole Roll tab. _reset_process_fields:
a card's reset scoped to only the fields it shows.

Stub-on-unbound-method, like test_right_panel_wiring.py: ControlsPanel pulls in every
sidebar in the app, so no test here constructs a real one.
"""

from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.controls_panel import _TONAL_RANGE_FIELDS, _TONE_FIELDS, ControlsPanel
from negpy.features.exposure.models import ExposureConfig
from negpy.features.process.models import ProcessConfig, ProcessMode


def _panel_stub(*, active_roll_id="roll1", locked_cards=()) -> MagicMock:
    panel = MagicMock()
    panel._ROLL_CARD_LABELS = ControlsPanel._ROLL_CARD_LABELS
    panel.film_section = MagicMock()
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


def test_sync_roll_locks_blank_summary_when_nothing_is_overridden():
    panel = _panel_stub(locked_cards=())

    ControlsPanel._sync_roll_locks(panel)

    panel.roll_override_summary.setText.assert_called_once_with("")


def test_sync_roll_locks_names_every_overridden_card():
    panel = _panel_stub(locked_cards={"sensor", "process"})

    ControlsPanel._sync_roll_locks(panel)

    panel.roll_override_summary.setText.assert_called_once_with("This frame overrides: Calibration, Normalization")


def test_sync_roll_locks_names_film_mode_too():
    panel = _panel_stub(locked_cards={"film"})

    ControlsPanel._sync_roll_locks(panel)

    panel.roll_override_summary.setText.assert_called_once_with("This frame overrides: Film Mode")


def test_sync_roll_locks_sets_each_sections_lock_button():
    panel = _panel_stub(locked_cards={"demosaic"})

    ControlsPanel._sync_roll_locks(panel)

    panel.film_section.set_lock_button.assert_called_once_with(False, False)
    panel.sensor_section.set_lock_button.assert_called_once_with(False, False)
    panel.demosaic_section.set_lock_button.assert_called_once_with(True, True)
    panel.process_section.set_lock_button.assert_called_once_with(False, False)


def test_reset_process_fields_only_touches_the_given_fields():
    """Regression: Normalization's own reset must not also reset Positive (moved
    beside Film Mode) or a Calibration/Demosaic field -- all three cards, and Film
    Mode, live on the same ProcessConfig, but each reset is scoped to its own card."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, analysis_buffer=0.2, positive_source=True, sensor_profile="Custom"),
    )

    ControlsPanel._reset_process_fields(panel, ("analysis_buffer",))

    new_cfg = panel.controller.apply_config.call_args[0][0]
    assert new_cfg.process.analysis_buffer == ProcessConfig().analysis_buffer
    assert new_cfg.process.positive_source is True
    assert new_cfg.process.sensor_profile == "Custom"


def test_reset_exposure_fields_turns_auto_off_for_a_positive_frame():
    """Regression: Tone's reset used to restore ExposureConfig's own flat default
    (on) regardless of Positive, so resetting a Positive frame turned Auto Density/
    Auto Grade back on instead of to the value auto_meter_for_positive_source gives it."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, positive_source=True),
        exposure=replace(cfg.exposure, auto_exposure=True, auto_normalize_contrast=True, density=1.4),
    )

    ControlsPanel._reset_exposure_fields(panel, ("auto_exposure", "auto_normalize_contrast", "density"))

    new_cfg = panel.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.auto_exposure is False
    assert new_cfg.exposure.auto_normalize_contrast is False
    assert new_cfg.exposure.density == ExposureConfig().density


def test_reset_exposure_fields_turns_auto_on_for_a_negative_frame():
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, positive_source=False),
        exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )

    ControlsPanel._reset_exposure_fields(panel, ("auto_exposure", "auto_normalize_contrast"))

    new_cfg = panel.controller.session.update_config.call_args[0][0]
    assert new_cfg.exposure.auto_exposure is True
    assert new_cfg.exposure.auto_normalize_contrast is True


def test_sync_modified_dots_does_not_flag_a_positive_frames_own_auto_default():
    """A Positive frame with Auto Density/Grade correctly off is at its own default,
    not "modified" -- the Tone header's dot must not count it."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, positive_source=True),
        exposure=replace(cfg.exposure, auto_exposure=False, auto_normalize_contrast=False),
    )
    panel.tone_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.tone_section.set_modified.assert_called_once_with(0)


def test_sync_modified_dots_counts_film_mode_on_its_own_card():
    """Film Mode and Positive live on ProcessConfig with Normalization's own fields,
    so a badge that counts the whole config lights the wrong card."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.BW, positive_source=True),
    )
    panel.film_section = MagicMock()
    panel.process_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.film_section.set_modified.assert_called_once_with(2)
    panel.process_section.set_modified.assert_called_once_with(0)


def test_sync_modified_dots_counts_tonal_range_on_tone():
    """White/Black Point sit in Tone's Tonal Range block but live on ProcessConfig."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, white_point_offset=0.2, black_point_trim_red=0.1),
    )
    panel.tone_section = MagicMock()
    panel.process_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.tone_section.set_modified.assert_called_once_with(2)
    panel.process_section.set_modified.assert_called_once_with(0)


def test_sync_modified_dots_counts_linear_raw_on_calibration():
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(cfg, process=replace(cfg.process, linear_raw=True))
    panel.sensor_section = MagicMock()
    panel.process_section = MagicMock()

    ControlsPanel._sync_modified_dots(panel)

    panel.sensor_section.set_modified.assert_called_once_with(1)
    panel.process_section.set_modified.assert_called_once_with(0)


def test_reset_tone_fields_clears_both_configs():
    panel = MagicMock()
    panel.controller.state = AppState()

    ControlsPanel._reset_tone_fields(panel)

    assert panel._reset_exposure_fields.call_args[0][0] == _TONE_FIELDS
    assert panel._reset_process_fields.call_args[0][0] == _TONAL_RANGE_FIELDS


def test_reset_film_fields_routes_through_the_controls_own_setters():
    """Positive rewrites the auto-meter defaults and Film Mode rewrites Cast Removal,
    so a plain field reset would leave both behind."""
    panel = MagicMock()
    panel.controller.state = AppState()
    cfg = panel.controller.state.config
    panel.controller.state.config = replace(
        cfg,
        process=replace(cfg.process, process_mode=ProcessMode.BW, positive_source=True),
    )

    ControlsPanel._reset_film_fields(panel)

    panel.controller.set_positive_source.assert_called_once_with(False)
    panel.controller.set_process_mode.assert_called_once_with(ProcessConfig().process_mode)


def test_reset_film_fields_does_nothing_at_the_defaults():
    panel = MagicMock()
    panel.controller.state = AppState()

    ControlsPanel._reset_film_fields(panel)

    panel.controller.set_positive_source.assert_not_called()
    panel.controller.set_process_mode.assert_not_called()
