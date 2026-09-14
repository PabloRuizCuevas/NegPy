"""Export tab's Sync To Batch checkbox: an export-time behavior over the Metadata
tab's per-frame fields, so it lives beside the Export button rather than on the
Metadata tab, and disables alongside that tab's Protect Original Metadata."""

from __future__ import annotations

from dataclasses import replace

from conftest import FakeController
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.desktop.view.sidebar.metadata import MetadataSidebar


def _sidebar() -> ExportSidebar:
    controller = FakeController()
    controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
    return ExportSidebar(controller)


def test_off_by_default() -> None:
    sidebar = _sidebar()
    assert sidebar.sync_check.isChecked() is False


def test_toggle_persists_to_the_metadata_config() -> None:
    sidebar = _sidebar()
    sidebar.sync_check.setChecked(True)
    assert sidebar.state.config.metadata.sync_to_batch is True


def test_metadata_tabs_protect_toggle_disables_it() -> None:
    sidebar = _sidebar()
    metadata_sidebar = MetadataSidebar(sidebar.controller)
    metadata_sidebar.protect_toggled.connect(sidebar._on_metadata_protect_changed)

    metadata_sidebar._on_protect_toggled(True)
    assert sidebar.sync_check.isEnabled() is False

    metadata_sidebar._on_protect_toggled(False)
    assert sidebar.sync_check.isEnabled() is True


def test_sync_ui_reflects_protect_state_without_the_signal() -> None:
    """sync_ui() also picks up protect state directly, for a sidebar built after
    the fact or resynced from an unrelated config change."""
    sidebar = _sidebar()
    sidebar.state.config = replace(sidebar.state.config, metadata=replace(sidebar.state.config.metadata, protect_original_metadata=True))

    sidebar.sync_ui()

    assert sidebar.sync_check.isEnabled() is False
