from negpy.kernel.system.i18n import tr
from negpy.kernel.system.text import count_of
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QVBoxLayout

from negpy.desktop.view.styles.templates import field_label, hint_label, pin_button_box
from negpy.desktop.view.styles.theme import THEME, scene_color
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry


class ReanalyzeDialog(QDialog):
    """Shows what Reanalyze does to the selection before it runs: each roll or scene baseline
    it reruns with the frames that change, the frames measured alone and the locked ones."""

    def __init__(self, plan, parent=None, *, repo=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Reanalyze"))
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(THEME.space_xl, THEME.space_xl, THEME.space_xl, THEME.space_xl)
        root.setSpacing(THEME.space_lg)

        rows = QGridLayout()
        rows.setHorizontalSpacing(THEME.space_lg)
        rows.setVerticalSpacing(THEME.space_sm)
        for group in plan.groups:
            if group.scene_id is None:
                name = tr("Roll “{name}”", name=group.name) if group.name else tr("Roll")
            else:
                name = tr('<span style="color:{value}">●</span> Scene “{name}”', value=scene_color(group.ordinal), name=group.name)
            counts = tr("{selected} selected · {value}", selected=group.selected, value=count_of(len(group.frames), "frame"))
            self._add_row(rows, name, counts)
        if plan.own:
            self._add_row(rows, tr("Own bounds, measured alone"), count_of(len(plan.own), "frame"))
        if plan.locked:
            self._add_row(rows, tr("Lock Bounds: skipped"), count_of(plan.locked, "frame"))
        rows.setColumnStretch(0, 1)
        root.addLayout(rows)

        root.addWidget(field_label(tr("{value} get new bounds.", value=count_of(plan.changed, "frame"))))
        if plan.uncropped:
            root.addWidget(
                hint_label(
                    tr(
                        "{uncropped} of {value} have no crop. Crop them first for a better result.",
                        uncropped=plan.uncropped,
                        value=count_of(plan.group_frames, "frame"),
                    ),
                    kind="warning",
                )
            )
        if plan.groups:
            root.addWidget(hint_label(tr("Analysis Buffer and the clip settings come from the open frame.")))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.addButton(tr("Reanalyze"), QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        pin_button_box(buttons)
        root.addWidget(buttons)

        remember_dialog_geometry(self, repo, "reanalyze")

    @staticmethod
    def _add_row(rows: QGridLayout, name: str, counts: str) -> None:
        r = rows.rowCount()
        rows.addWidget(field_label(name), r, 0)
        rows.addWidget(hint_label(counts), r, 1)


def reanalyze_selection(parent, controller) -> None:
    """Reanalyze for the selection, else the open frame, after the plan is confirmed."""
    plan = controller.reanalyze_plan()
    if ReanalyzeDialog(plan, parent, repo=controller.session.repo).exec() == QDialog.DialogCode.Accepted:
        controller.run_reanalyze(plan)
