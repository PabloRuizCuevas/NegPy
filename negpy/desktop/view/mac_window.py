"""The macOS window background behind Qt's backing store.

At a fractional device pixel ratio (a UI scale below or above 100%) the macOS backing store
leaves device pixels unpainted between the repainted and the kept regions of a partial repaint.
Those pixels show the NSWindow background, which AppKit makes light gray, as hairlines around
controls. A window background in the theme's base color makes the gaps blend into the panels.

Any change that enables an alternate color theme (a light theme, a user theme, a theme switch
at run time) must also update this fix: the color here comes from ``THEME.bg_dark`` once, when
each window shows, and a light theme on a dark window background brings the hairlines back as
dark lines.
"""

import ctypes
import ctypes.util
import sys
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.styles.theme import THEME

_objc: Optional[ctypes.CDLL] = None


def _send(receiver: int, selector: str, *args, restype=ctypes.c_void_p, argtypes=()) -> int:
    assert _objc is not None
    send = _objc.objc_msgSend
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return send(receiver, _objc.sel_registerName(selector.encode()), *args)


def _load_objc() -> bool:
    global _objc
    if _objc is None:
        path = ctypes.util.find_library("objc")
        if not path:
            return False
        _objc = ctypes.cdll.LoadLibrary(path)
        _objc.objc_getClass.restype = ctypes.c_void_p
        _objc.sel_registerName.restype = ctypes.c_void_p
    return True


def paint_window_background(widget: QWidget) -> None:
    """Give a shown top-level widget's NSWindow the theme's base color. A translucent window keeps
    the clear background Qt gives it."""
    if sys.platform != "darwin" or not widget.isWindow() or widget.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground):
        return
    if not _load_objc():
        return
    window = _send(int(widget.winId()), "window")
    if not window:
        return
    color = QColor(THEME.bg_dark)
    assert _objc is not None
    ns_color = _send(
        _objc.objc_getClass(b"NSColor"),
        "colorWithSRGBRed:green:blue:alpha:",
        color.redF(),
        color.greenF(),
        color.blueF(),
        1.0,
        argtypes=(ctypes.c_double,) * 4,
    )
    _send(window, "setBackgroundColor:", ns_color, restype=None, argtypes=(ctypes.c_void_p,))
