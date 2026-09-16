from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.process import ProcessSidebar
from negpy.features.process.models import ProcessMode


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    return controller, ProcessSidebar(controller)


def test_channel_selector_retargets_and_syncs(qapp):
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(
        cfg,
        process=replace(
            cfg.process,
            white_point_offset=0.1,
            black_point_offset=-0.05,
            white_point_trim_red=0.08,
            black_point_trim_red=-0.02,
        ),
    )
    sidebar.sync_ui()

    assert sidebar._wp_field() == "white_point_offset"
    assert sidebar._bp_field() == "black_point_offset"
    assert abs(sidebar.white_point_slider.value() - 0.1) < 1e-9
    assert abs(sidebar.black_point_slider.value() - (-0.05)) < 1e-9

    sidebar.ch_r_btn.setChecked(True)

    assert sidebar._wp_field() == "white_point_trim_red"
    assert sidebar._bp_field() == "black_point_trim_red"
    assert abs(sidebar.white_point_slider.value() - 0.08) < 1e-9
    assert abs(sidebar.black_point_slider.value() - (-0.02)) < 1e-9
    assert sidebar.white_point_slider.label.text() == "White Point R"
    assert sidebar.black_point_slider.label.text() == "Black Point R"

    sidebar.ch_global_btn.setChecked(True)
    assert abs(sidebar.white_point_slider.value() - 0.1) < 1e-9
    assert sidebar.white_point_slider.label.text() == "White Point"


def test_channel_selector_hidden_in_bw(qapp):
    controller, sidebar = _sidebar()

    sidebar.sync_ui()
    assert not sidebar.ch_r_btn.isHidden()
    sidebar.ch_r_btn.setChecked(True)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.BW))
    sidebar.sync_ui()
    for w in (sidebar.ch_global_btn, sidebar.ch_r_btn, sidebar.ch_g_btn, sidebar.ch_b_btn):
        assert w.isHidden()
    assert sidebar._channel_index() == 0
    assert sidebar._wp_field() == "white_point_offset"


def test_channel_selector_hidden_on_the_transparency_transfer(qapp):
    """The transfer normalizes with a fixed window, so the WP/BP sliders are hidden — and
    the selector that scopes them has nothing left to scope."""
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=True))
    sidebar.sync_ui()
    assert not sidebar.ch_r_btn.isHidden()
    sidebar.ch_r_btn.setChecked(True)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, e6_normalize=False))
    sidebar.sync_ui()

    for w in (sidebar.ch_global_btn, sidebar.ch_r_btn, sidebar.ch_g_btn, sidebar.ch_b_btn):
        assert w.isHidden()
    # Reset to Global, so turning Normalize back on does not land on a channel page.
    assert sidebar._channel_index() == 0
    assert sidebar._wp_field() == "white_point_offset"


def test_lock_bounds_disables_wp_bp_and_selector(qapp):
    controller, sidebar = _sidebar()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, lock_bounds=True))
    sidebar.sync_ui()
    for w in (sidebar.white_point_slider, sidebar.black_point_slider, sidebar.ch_global_btn, sidebar.ch_r_btn):
        assert not w.isEnabled()


def test_analysis_region_dot_reflects_committed_region_not_just_tool_state(qapp):
    """Confirming a freehand region closes the draw tool (button unchecks), so the
    dot is the only remaining cue that a region is active and overriding the
    Analysis Buffer slider — it must track analysis_rect, not active_tool."""
    controller, sidebar = _sidebar()

    sidebar.sync_ui()
    assert not sidebar.analysis_region_btn.edited_dot.isVisibleTo(sidebar.analysis_region_btn)

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, analysis_rect=(0.1, 0.1, 0.9, 0.9)))
    sidebar.sync_ui()

    assert sidebar.analysis_region_btn.edited_dot.isVisibleTo(sidebar.analysis_region_btn)
    assert not sidebar.analysis_region_btn.isChecked()  # tool itself is closed
    assert not sidebar.analysis_buffer_slider.isEnabled()

    controller.state.config = replace(cfg, process=replace(cfg.process, analysis_rect=None))
    sidebar.sync_ui()
    assert not sidebar.analysis_region_btn.edited_dot.isVisibleTo(sidebar.analysis_region_btn)


def test_mode_buttons_track_config_and_switch_mode(qapp):
    controller, sidebar = _sidebar()
    sidebar.sync_ui()

    color_btn, bw_btn, slide_btn = sidebar.mode_btns
    assert color_btn.isChecked()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6))
    sidebar.sync_ui()
    assert slide_btn.isChecked()
    assert not color_btn.isChecked() and not bw_btn.isChecked()

    bw_btn.click()
    controller.set_process_mode.assert_called_once_with(ProcessMode.BW)


def test_lock_bounds_sits_in_the_analysis_row_and_hides_on_the_transparency_transfer(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False))
    sidebar.sync_ui()
    assert sidebar.lock_bounds_btn.isHidden()


def test_positive_is_visible_for_any_mode_but_grays_out_with_normalize_on(qapp):
    """Positive is not Slide-only: it stays visible for every mode, only stepping
    aside when Normalize's own metered stretch already decodes on the source's own
    profile."""
    controller, sidebar = _sidebar()
    sidebar.sync_ui()
    assert not sidebar.positive_source_btn.isHidden()
    assert sidebar.positive_source_btn.isEnabled()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False))
    sidebar.sync_ui()
    assert not sidebar.positive_source_btn.isHidden()
    assert sidebar.positive_source_btn.isEnabled()

    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, e6_normalize=True))
    sidebar.sync_ui()
    assert not sidebar.positive_source_btn.isHidden()
    assert not sidebar.positive_source_btn.isEnabled()


def test_positive_lives_in_mode_bar_not_the_normalization_body(qapp):
    """Whether the source is already a finished positive is a fact about the file,
    true for any mode, not a Normalization setting -- it lives beside Film Mode in
    mode_bar, which ControlsPanel places above the collapsible card, not inside it."""
    _, sidebar = _sidebar()
    assert sidebar.positive_source_btn.parentWidget() is sidebar.mode_bar


def test_positive_toggle_reaches_the_controller(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False))
    sidebar.sync_ui()

    sidebar.positive_source_btn.setChecked(True)
    args, kwargs = controller.set_roll_default.call_args
    assert args[0] == "process"
    assert kwargs["positive_source"] is True


def _row_index_containing(layout, widget) -> int:
    """Index within *layout* of the (possibly nested) row that directly holds *widget*."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is widget:
            return i
        row = item.layout()
        if row is not None and any(row.itemAt(j).widget() is widget for j in range(row.count())):
            return i
    raise AssertionError(f"{widget} not found in layout")


def test_average_toggles_sit_above_the_clip_sliders(qapp):
    """Which baseline each axis' bounds come from sits right above the sliders it
    disables when on -- both moved here from the old Roll Analysis panel."""
    _, sidebar = _sidebar()
    avg_i = _row_index_containing(sidebar.layout, sidebar.use_luma_avg_btn)
    clip_i = _row_index_containing(sidebar.layout, sidebar.luma_range_clip_slider)
    assert avg_i == clip_i - 1


def test_average_toggles_sync_from_config(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, use_luma_average=True, use_color_average=False))
    sidebar.sync_ui()
    assert sidebar.use_luma_avg_btn.isChecked()
    assert not sidebar.use_color_avg_btn.isChecked()


def test_average_toggles_hide_on_the_transparency_transfer(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False))
    sidebar.sync_ui()
    assert sidebar.use_luma_avg_btn.isHidden()
    assert sidebar.use_color_avg_btn.isHidden()


def test_per_frame_exposure_subheader_sits_above_the_channel_selector(qapp):
    """White/Black Point never join the roll -- unlike every clip/average field above
    them -- and the subheader is the only cue, since they are otherwise plain sliders
    sitting beside roll-shared ones."""
    _, sidebar = _sidebar()
    header_i = sidebar.layout.indexOf(sidebar.per_frame_subheader)
    assert header_i >= 0
    selector_i = _row_index_containing(sidebar.layout, sidebar.ch_global_btn)
    assert header_i == selector_i - 1


def test_per_frame_exposure_subheader_hides_on_the_transparency_transfer(qapp):
    controller, sidebar = _sidebar()
    cfg = controller.state.config
    controller.state.config = replace(cfg, process=replace(cfg.process, process_mode=ProcessMode.E6, e6_normalize=False))
    sidebar.sync_ui()
    assert sidebar.per_frame_subheader.isHidden()


def test_use_luma_average_toggle_reaches_the_controller(qapp):
    """Goes through set_roll_default, the same Normalization card write every other
    roll-eligible control on this card uses -- flipping it locks the card to this
    frame, and drops roll_name since a single picked baseline no longer applies."""
    controller, sidebar = _sidebar()
    sidebar.use_luma_avg_btn.setChecked(True)
    args, kwargs = controller.set_roll_default.call_args
    assert args[0] == "process"
    assert kwargs["use_luma_average"] is True
    assert kwargs["roll_name"] is None


def test_use_color_average_toggle_reaches_the_controller(qapp):
    controller, sidebar = _sidebar()
    sidebar.use_color_avg_btn.setChecked(True)
    args, kwargs = controller.set_roll_default.call_args
    assert args[0] == "process"
    assert kwargs["use_color_average"] is True
    assert kwargs["roll_name"] is None
