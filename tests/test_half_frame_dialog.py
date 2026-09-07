"""HalfFrameDialog's auto-detect-per-frame option: shown for the roll-wide editor,
hidden for the per-frame override editor, where re-detecting per file is meaningless."""

import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.half_frame_dialog import HalfFrameDialog

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _dialog(**kwargs) -> HalfFrameDialog:
    buf = np.zeros((32, 48, 3), dtype=np.uint8)
    return HalfFrameDialog(buf, **kwargs)


class TestAutoSplitOption:
    def test_shown_and_off_by_default_for_the_roll_wide_editor(self):
        d = _dialog(initial_auto_split=False)
        assert d._auto_split_check is not None
        assert not d._auto_split_check.isChecked()
        assert d.auto_split() is False

    def test_shown_and_checked_when_the_saved_profile_had_it_on(self):
        d = _dialog(initial_auto_split=True)
        assert d._auto_split_check is not None
        assert d._auto_split_check.isChecked()
        assert d.auto_split() is True

    def test_hidden_for_a_per_frame_override(self):
        """initial_auto_split=None is how the controller opens the per-frame editor:
        the option is meaningless for a single fixed frame."""
        d = _dialog(initial_auto_split=None)
        assert d._auto_split_check is None
        assert d.auto_split() is False

    def test_default_argument_is_off_not_hidden(self):
        d = _dialog()
        assert d._auto_split_check is not None
        assert d.auto_split() is False


class TestTitle:
    def test_custom_title_is_applied(self):
        d = _dialog(title="Half Frame — split & crop (this frame)")
        assert d.windowTitle() == "Half Frame — split & crop (this frame)"

    def test_default_title(self):
        d = _dialog()
        assert d.windowTitle() == "Half Frame — split & crop"
