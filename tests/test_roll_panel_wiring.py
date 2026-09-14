"""RollPanel (Export/Metadata/Scan dock) tab-switch bookkeeping.

Stubs on the unbound methods, the same pattern test_right_panel_wiring.py uses --
no test in this repo constructs a real RightPanel/RollPanel, since each pulls in a
full chain of sidebars that need a real controller.
"""

from unittest.mock import MagicMock

from negpy.desktop.view.sidebar.roll_panel import RollPanel


def _panel_stub(*, scan_index: int = 2, active_index: int = 0, n_tabs: int = 3) -> MagicMock:
    panel = MagicMock()
    panel._tab_buttons = [MagicMock() for _ in range(n_tabs)]
    panel._tab_icons = ["fa5s.file-export", "fa5s.tags", "fa5s.camera-retro"][:n_tabs]
    panel._tab_keys = ["export", "metadata", "scan"][:n_tabs]
    panel._scan_index = scan_index
    panel._active_index = active_index
    return panel


def test_switch_tab_persists_the_index_and_updates_the_stack():
    panel = _panel_stub()

    RollPanel._switch_tab(panel, 1)

    panel.controller.session.repo.save_global_setting.assert_called_once_with("roll_panel_tab", 1)
    panel.stack.setCurrentIndex.assert_called_once_with(1)
    panel.switcher.set_pinned.assert_called_once_with(1)
    assert panel._active_index == 1
    panel._tab_buttons[1].setChecked.assert_called_once_with(True)
    panel._tab_buttons[0].setChecked.assert_called_once_with(False)


def test_switch_tab_activates_scan_sidebars_only_on_the_scan_tab():
    panel = _panel_stub(scan_index=2)

    RollPanel._switch_tab(panel, 0)
    panel.scan_sidebar.on_activated.assert_not_called()
    panel.scanlight_sidebar.on_activated.assert_not_called()

    RollPanel._switch_tab(panel, 2)
    panel.scan_sidebar.on_activated.assert_called_once_with()
    panel.scanlight_sidebar.on_activated.assert_called_once_with()


def test_show_tab_by_key_dispatches_the_matching_index():
    panel = _panel_stub()

    RollPanel.show_tab_by_key(panel, "metadata")

    panel._switch_tab.assert_called_once_with(1)


def test_show_tab_by_key_ignores_an_unknown_key():
    panel = _panel_stub()

    RollPanel.show_tab_by_key(panel, "not-a-real-tab")

    panel._switch_tab.assert_not_called()
