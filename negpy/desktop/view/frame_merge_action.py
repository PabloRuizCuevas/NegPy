"""Merge to TIFF Negative, the one action behind every surface that offers it.

A free function rather than a method so the Frame Assembly card, both Film Strip menus, the
canvas menu, the Camera Scanning panel and the shortcut map all ask the same question and
get the same dialog.
"""

from typing import List, Optional

from negpy.desktop.session import composite_kind
from negpy.desktop.view.confirm import confirm_frame_merge
from negpy.services.export.frame_merge import MERGEABLE_KINDS

SCOPE_FRAME = "frame"
SCOPE_SELECTION = "selection"
SCOPE_ROLL = "roll"

LABELS = {
    SCOPE_FRAME: "Merge Frame to TIFF Negative…",
    SCOPE_SELECTION: "Merge Selected to TIFF Negative…",
    SCOPE_ROLL: "Merge Roll to TIFF Negative…",
}

ACTION_IDS = {SCOPE_FRAME: "merge_frame", SCOPE_SELECTION: "merge_selected", SCOPE_ROLL: "merge_roll"}


def scope_indices(state, scope: str) -> Optional[List[int]]:
    """The Film Strip indices *scope* covers. None means every loaded frame."""
    if scope == SCOPE_ROLL:
        return None
    if scope == SCOPE_SELECTION:
        return list(state.selected_indices)
    return [state.selected_file_idx]


def mergeable_in(state, scope: str) -> bool:
    """Whether *scope* holds anything worth offering the action for.

    Structural only: it reads the asset dicts and touches no disk, so a context menu can
    call it while it is being built. Whether the files are actually there is the plan's
    question, answered on click.
    """
    indices = scope_indices(state, scope)
    frames = state.uploaded_files if indices is None else [state.uploaded_files[i] for i in indices if 0 <= i < len(state.uploaded_files)]
    return any(composite_kind(f) in MERGEABLE_KINDS for f in frames)


def merge_to_tiff(parent, controller, scope: str) -> None:
    """Plan, confirm, then hand the frames to the controller."""
    indices, skipped = controller.frame_merge_plan(scope_indices(controller.state, scope))
    if not indices:
        # One refused frame gets its own reason: "nothing can be merged" on an HDR frame the
        # user just right-clicked answers the wrong question.
        controller.set_status(skipped[0] if len(skipped) == 1 else _nothing_message(scope), 5000)
        return
    counts: dict = {}
    paths = []
    for i in indices:
        asset = controller.state.uploaded_files[i]
        kind = composite_kind(asset)
        counts[kind] = counts.get(kind, 0) + 1
        paths.append(asset["path"])
    trash = confirm_frame_merge(parent, LABELS[scope].rstrip("…"), counts, skipped)
    if trash is not None:
        controller.request_frame_merge(paths, trash)


def _nothing_message(scope: str) -> str:
    where = {SCOPE_FRAME: "This frame", SCOPE_SELECTION: "Nothing selected", SCOPE_ROLL: "No frame in this roll"}[scope]
    return f"{where} is not a triplet or a stitch" if scope == SCOPE_FRAME else f"{where} can be merged"
