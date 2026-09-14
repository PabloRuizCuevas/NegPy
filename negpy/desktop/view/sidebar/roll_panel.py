import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.desktop.view.sidebar.metadata import MetadataSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import make_section
from negpy.desktop.view.widgets.overflow_bar import OverflowBar


class RollPanel(QWidget):
    """Right sidebar panel ("Roll" dock): Export, Metadata and Scan -- roll and output
    bookkeeping that never touches the rendered pixel, so it lives apart from Controls
    (RightPanel, right_panel.py), which holds everything that does. Docked and
    detached the same way Session and Controls already are (PinnableDockWidget,
    negpy/desktop/view/widgets/pinnable_dock.py).
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        def wrap_scroll(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            return scroll

        self.export_sidebar = ExportSidebar(self.controller)
        self.metadata_sidebar = MetadataSidebar(self.controller)

        from negpy.desktop.view.sidebar.scan import ScanSidebar

        self.scan_sidebar = ScanSidebar(self.controller)

        from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar

        self.scanlight_sidebar = ScanlightSidebar(self.controller)

        # One "Scan" tab hosting both the SANE scanner and the RGB-Scan capture as collapsible
        # sections, like Controls' "Color: Lab, Toning" tab.
        self.scan_page = self._build_scan_page()

        # (key, icon_name, tooltip, content_widget)
        tab_specs = [
            ("export", "fa5s.file-export", "Export", self.export_sidebar),
            ("metadata", "fa5s.tags", "Metadata", self.metadata_sidebar),
            ("scan", "fa5s.camera-retro", "Scan", self.scan_page),
        ]

        # Icon-only tab switcher; spills into a » menu when the panel is narrowed
        self.switcher = OverflowBar(tile=True, height=38, min_item=36)

        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, 8, 0, 0)

        self._tab_buttons: list[QPushButton] = []
        self._tab_keys: list[str] = []
        self._tab_icons: list[str] = []
        self._tab_tooltips: list[str] = []
        self._active_index = 0
        self._scan_index = -1

        for i, (key, icon_name, tooltip, content) in enumerate(tab_specs):
            btn = QPushButton()
            btn.setObjectName("right_tab_btn")
            btn.setIcon(qta.icon(icon_name, color=THEME.text_secondary))
            btn.setIconSize(QSize(18, 18))
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedHeight(38)
            btn.clicked.connect(lambda _checked=False, idx=i: self._switch_tab(idx))
            self.switcher.add_button(btn, tooltip)

            self.stack.addWidget(wrap_scroll(content))
            self._tab_buttons.append(btn)
            self._tab_keys.append(key)
            self._tab_icons.append(icon_name)
            self._tab_tooltips.append(tooltip)
            if key == "scan":
                self._scan_index = i

        layout.addWidget(self.switcher)
        layout.addWidget(self.stack, 1)

        self.apply_shortcut_tooltips()

        repo = self.controller.session.repo
        saved_tab = repo.get_global_setting("roll_panel_tab", 0)
        self._switch_tab(saved_tab if isinstance(saved_tab, int) and 0 <= saved_tab < len(self._tab_buttons) else 0)

    def _build_scan_page(self) -> QWidget:
        """The 'Scan' tab hosts two collapsible sections (like Controls' Color tab): the
        SANE flatbed/film scanner on top, the RGB-Scan trichromatic capture below."""
        repo = self.controller.session.repo
        self.scan_sane_section = make_section(repo, "Film Scanner", "scan_sane", self.scan_sidebar, "fa5s.camera-retro", False)
        self.scan_rgb_section = make_section(repo, "Camera Scanning", "scan_rgb", self.scanlight_sidebar, "fa5s.camera", True)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)
        page_layout.addWidget(self.scan_sane_section)
        page_layout.addWidget(self.scan_rgb_section)
        return page

    def apply_shortcut_tooltips(self) -> None:
        """Append the current keyboard shortcut (action id `tab_<key>`) to each tab tooltip,
        and pass the call on to the panel that owns bound controls of its own."""
        for btn, key, base in zip(self._tab_buttons, self._tab_keys, self._tab_tooltips):
            btn.setToolTip(tooltip_with_shortcut(base, f"tab_{key}"))
        self.metadata_sidebar.apply_shortcut_tooltips()

    def _connect_signals(self) -> None:
        # These two sync_ui calls scan gear/template files; never per drag tick.
        self._sync_debounce = QTimer()
        self._sync_debounce.setSingleShot(True)
        self._sync_debounce.setInterval(150)
        self._sync_debounce.timeout.connect(self.export_sidebar.sync_ui)
        self._sync_debounce.timeout.connect(self.metadata_sidebar.sync_ui)
        self.controller.config_updated.connect(self._sync_debounce.start)

    def _switch_tab(self, index: int) -> None:
        self._active_index = index
        self.controller.session.repo.save_global_setting("roll_panel_tab", index)
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._tab_buttons):
            btn.setChecked(i == index)
            btn.setIcon(qta.icon(self._tab_icons[i], color="white" if i == index else THEME.text_secondary))
        self.switcher.set_pinned(index)

        # Trigger device detection and a gating refresh when the Scan tab is selected. It hosts
        # both the SANE scanner and the RGB-Scan capture as collapsible sections.
        if index == self._scan_index:
            if hasattr(self.scan_sidebar, "on_activated"):
                self.scan_sidebar.on_activated()
            if hasattr(self.scanlight_sidebar, "on_activated"):
                self.scanlight_sidebar.on_activated()

    def show_tab_by_key(self, key: str) -> None:
        if key in self._tab_keys:
            self._switch_tab(self._tab_keys.index(key))
