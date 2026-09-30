"""Carries an assembled frame's edit and roll records onto the TIFF merged from it."""

import os
from dataclasses import replace
from typing import Any, List

from negpy.domain.models import WorkspaceConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.features.stitch.models import StitchConfig
from negpy.services.assets import rolls
from negpy.services.assets.sidecar import sidecar_path_for, write_sidecar

#: Cards whose roll default would be applied a second time on top of what the merged file
#: already bakes, per kind. Locked on the new frame so an Apply cannot put them back.
_BAKED_CARDS = {"rgb": ("sensor",), "stitch": ("sensor", "flatfield")}


def merged_edit(config: WorkspaceConfig, kind: str) -> WorkspaceConfig:
    """An assembled frame's edit as it applies to the TIFF merged from it.

    The TIFF is one source, so whatever named the other files goes. So does every
    correction the merge baked: the sensor unmix for both kinds, because a triplet skips it
    and a single file would apply it, and flat-field for a stitch, where it was a per-part
    input to the registration rather than something the edit re-applies.
    """
    out = replace(
        config,
        rgbscan=RgbScanConfig(),
        process=replace(config.process, sensor_matrix=None, sensor_profile="None"),
    )
    if kind == "stitch":
        out = replace(out, stitch=StitchConfig(), flatfield=FlatFieldConfig())
    return out


def carry_edit(
    repo: Any,
    old_hash: str,
    old_path: str,
    new_hash: str,
    new_path: str,
    dropped_paths: List[str],
    config: WorkspaceConfig,
    kind: str,
    keep_source: bool = False,
) -> None:
    """Copy the frame's edit, forks, card locks, scene and roll membership to the merged file.

    *config* is the frame's hydrated edit. Nothing is taken from the source files. A roll
    whose default would re-apply a correction the file already bakes gets that card locked
    on the new frame at the roll's other values. With *keep_source* the frame it was merged
    from stays a member of its rolls, beside the merged file.
    """

    def transform(c: WorkspaceConfig) -> WorkspaceConfig:
        return merged_edit(c, kind)

    owns_new = repo.copy_file_edits(old_hash, new_hash, new_path, transform)
    roll_ids = rolls.adopt_replacement(repo, old_hash, new_hash, old_path, new_path, dropped_paths, keep_source)
    for roll_id in roll_ids:
        repo.copy_file_edits(rolls.roll_edit_hash(old_hash, roll_id), rolls.roll_edit_hash(new_hash, roll_id), new_path, transform)
    member_rolls = rolls.rolls_containing_path(repo, new_path)
    # The row is written resolved against the roll, not copied raw. A lock freezes its whole
    # card, so every field on it has to already hold what the roll says or the merge would
    # take the roll's Narrowband and crosstalk away with the matrix it is there to drop. This
    # also gives a never-opened composite a row at all, without which the frame is hydrated
    # with the rig's active flat-field back on, which a merged stitch already bakes.
    if owns_new:
        roll_id = next(iter(member_rolls), None)
        resolved = config if roll_id is None else rolls.resolve_roll_config(repo, roll_id, old_hash, config)
        repo.save_file_settings(new_hash, merged_edit(resolved, kind), file_path=new_path)
    for roll_id in member_rolls:
        defaults = rolls.roll_defaults(repo, roll_id)
        for card in _BAKED_CARDS.get(kind, ()):
            if _roll_would_reapply(defaults, card):
                rolls.set_frame_override(repo, roll_id, new_hash, card, True)


def _roll_would_reapply(defaults: dict, card: str) -> bool:
    """Whether the roll's default for *card* holds a correction the merged file already has."""
    if card == "sensor":
        return defaults.get("sensor_matrix") is not None
    if card == "flatfield":
        return bool(defaults.get("apply"))
    return False


def carry_sidecar(old_path: str, new_path: str, config: WorkspaceConfig, kind: str) -> bool:
    """Write the merged file's sidecar when the primary source has one. Returns whether it did."""
    if not os.path.exists(sidecar_path_for(old_path)):
        return False
    write_sidecar(new_path, merged_edit(config, kind))
    return True
