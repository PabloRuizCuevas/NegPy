"""HalfFrameDialog: a rectangle + split-line editor whose own Apply split-button
picks what the result gets applied to (this frame, the selection, or the whole
roll) -- the same current/selected/all scopes as the Export button."""

import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.half_frame_dialog import HalfFrameDialog

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _dialog(**kwargs) -> HalfFrameDialog:
    buf = np.zeros((32, 48, 3), dtype=np.uint8)
    return HalfFrameDialog(buf, **kwargs)


class TestInitialValues:
    def test_defaults_to_uncropped_centered_split(self):
        d = _dialog()
        assert d.crop_rect() == (0.0, 0.0, 1.0, 1.0)
        assert d.split_x() == 0.5
        assert d.gutter_thickness() == 0.0

    def test_seeds_from_the_given_values(self):
        d = _dialog(initial_rect=(0.05, 0.0, 0.95, 1.0), initial_split=0.42, initial_gutter=0.01)
        assert d.crop_rect() == (0.05, 0.0, 0.95, 1.0)
        assert d.split_x() == 0.42
        assert d.gutter_thickness() == 0.01


class TestApplyScope:
    def test_defaults_to_current(self):
        d = _dialog()
        assert d.scope() == "current"

    def test_seeds_from_initial_scope(self):
        d = _dialog(initial_scope="all")
        assert d.scope() == "all"

    def test_an_unknown_initial_scope_falls_back_to_current(self):
        d = _dialog(initial_scope="bogus")
        assert d.scope() == "current"

    def test_choosing_a_scope_from_the_menu_updates_it(self):
        d = _dialog()
        d._scope_actions["selected"].trigger()
        assert d.scope() == "selected"
        assert "Selected" in d._ok_btn.text()


class TestTitle:
    def test_custom_title_is_applied(self):
        d = _dialog(title="Half Frame — split & crop (this frame)")
        assert d.windowTitle() == "Half Frame — split & crop (this frame)"

    def test_default_title(self):
        d = _dialog()
        assert d.windowTitle() == "Half Frame — split & crop"
