"""Roll membership: a Roll is a named, navigable group of frames -- either a real
library folder recognized as a roll, or a virtual roll built by hand from whatever the
Film Strip currently holds (a search result, a hand-picked selection, extras added to a
folder roll that are not physically in that folder).

Edits are not stored here and are not scoped by roll: they stay in the edits DB under
each frame's own content hash, exactly as if no Roll existed. A Roll only decides which
files show up when you open it. The one exception is roll-wide defaults, below: a
handful of Calibration/Demosaic/Normalization facts that describe the rig and the roll
rather than one frame's own look.
"""

import os
import time
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Optional

ROLLS_KEY = "rolls_by_id"


def _read(repo: Any) -> Dict[str, dict]:
    saved = repo.get_global_setting(ROLLS_KEY, default=None)
    return dict(saved) if isinstance(saved, dict) else {}


def _write(repo: Any, store: Dict[str, dict]) -> None:
    repo.save_global_setting(ROLLS_KEY, store)


def saved_rolls(repo: Any) -> Dict[str, dict]:
    """Every remembered roll, keyed by id."""
    return _read(repo)


def roll_for_id(repo: Any, roll_id: str) -> Optional[dict]:
    return _read(repo).get(roll_id)


def folder_roll_id_for_path(repo: Any, path: str) -> Optional[str]:
    """The id of the roll recognizing *path*, or None if not yet recognized."""
    for roll_id, entry in _read(repo).items():
        if entry.get("kind") == "folder" and entry.get("folder_path") == path:
            return roll_id
    return None


def recognize_folder(repo: Any, path: str, name: str = "") -> str:
    """Mark *path* as a recognized folder roll. Idempotent: returns the existing id
    when the folder is already recognized, without touching its stored name."""
    existing = folder_roll_id_for_path(repo, path)
    if existing:
        return existing
    store = _read(repo)
    roll_id = uuid.uuid4().hex
    store[roll_id] = {
        "kind": "folder",
        "name": name or path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1] or path,
        "folder_path": path,
        "extra_paths": [],
        "created_at": time.time(),
    }
    _write(repo, store)
    return roll_id


def import_subfolders_as_rolls(repo: Any, parent_path: str) -> List[str]:
    """Recognize every immediate subfolder of *parent_path* as its own folder roll.

    One level only: a subfolder's own children are not walked. Idempotent per
    subfolder, so re-running over a parent that already has some rolls recognized
    only creates the missing ones.
    """
    try:
        entries = sorted(e.path for e in os.scandir(parent_path) if e.is_dir() and not e.name.startswith("."))
    except OSError:
        return []
    return [recognize_folder(repo, path) for path in entries]


def create_virtual_roll(repo: Any, name: str, member_paths: List[str]) -> str:
    store = _read(repo)
    roll_id = uuid.uuid4().hex
    store[roll_id] = {
        "kind": "virtual",
        "name": name,
        "member_paths": list(member_paths),
        "created_at": time.time(),
    }
    _write(repo, store)
    return roll_id


def add_extra_member(repo: Any, roll_id: str, path: str) -> None:
    """Extend a roll's membership by one path: a folder roll's extra_paths, or a virtual
    roll's member_paths. No-op for an unknown roll id or an already-member path."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    key = "extra_paths" if entry["kind"] == "folder" else "member_paths"
    if path not in entry[key]:
        entry[key] = [*entry[key], path]
        _write(repo, store)


def rename_roll(repo: Any, roll_id: str, name: str) -> None:
    store = _read(repo)
    if roll_id in store:
        store[roll_id]["name"] = name
        _write(repo, store)


def delete_roll(repo: Any, roll_id: str) -> None:
    store = _read(repo)
    if roll_id in store:
        del store[roll_id]
        _write(repo, store)


def all_rolls_sorted(repo: Any) -> List[tuple]:
    """(roll_id, entry) pairs for every roll, folder and virtual alike, name-sorted."""
    return sorted(_read(repo).items(), key=lambda pair: pair[1].get("name", "").casefold())


def virtual_rolls(repo: Any) -> List[tuple]:
    """(roll_id, entry) pairs for every virtual roll, name-sorted."""
    return [pair for pair in all_rolls_sorted(repo) if pair[1].get("kind") == "virtual"]


# --- Roll-wide defaults ----------------------------------------------------------
#
# ProcessConfig fields the Roll tab's Calibration, Demosaic and Normalization cards
# edit, grouped by the card that owns them -- the same grouping a per-card lock button
# unlocks. White/Black Point and their trims stay off this list on purpose: they are
# exposure choices that can legitimately vary shot to shot within a roll, unlike these,
# which describe the rig or the roll's own shared baseline. Film Mode is a roll default
# too but has no card of its own to unlock, so it never appears in a frame_overrides set.
ROLL_DEFAULT_FIELDS: Dict[str, tuple] = {
    "sensor": (
        "linear_raw",
        "narrowband_scan",
        "sensor_profile",
        "sensor_matrix",
        "crosstalk_strength",
        "crosstalk_profile",
        "crosstalk_matrix",
        # Baked alongside the profile+matrix so the render can gate the unmix on it
        # without disk I/O -- travels with them, or another frame's roll-derived
        # crosstalk would be read back through its own, unrelated film process.
        "crosstalk_process",
        "hue_trim",
    ),
    "demosaic": ("demosaic_preview", "demosaic_export"),
    "process": ("e6_normalize", "positive_source", "analysis_buffer", "luma_range_clip", "color_range_clip"),
}
_MODE_FIELD = "process_mode"


def roll_defaults(repo: Any, roll_id: str) -> Dict[str, Any]:
    """The roll's own value for each field it has set at least once. A field absent
    here has no roll default yet -- the frame's own saved value is what is used,
    exactly as before roll defaults existed."""
    entry = roll_for_id(repo, roll_id)
    return dict(entry["defaults"]) if entry and entry.get("defaults") else {}


def set_roll_defaults(repo: Any, roll_id: str, **fields: Any) -> None:
    """Set one or more roll-default field values (ROLL_DEFAULT_FIELDS' names, or
    process_mode). Every member frame that has not locked the owning card away from
    the roll picks this up as soon as it is next loaded or rendered. No-op for an
    unknown roll id."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    defaults = dict(entry.get("defaults", {}))
    defaults.update(fields)
    entry["defaults"] = defaults
    _write(repo, store)


def frame_override_cards(repo: Any, roll_id: str, file_hash: str) -> set:
    """Which of ROLL_DEFAULT_FIELDS' card keys this frame has locked to its own value,
    away from the roll's defaults, within this roll."""
    entry = roll_for_id(repo, roll_id)
    if not entry:
        return set()
    return set(entry.get("frame_overrides", {}).get(file_hash, ()))


def set_frame_override(repo: Any, roll_id: str, file_hash: str, card_key: str, locked: bool) -> None:
    """Lock (locked=True) or unlock (False) one card for one frame within one roll.
    Locking freezes that card at the frame's current (usually roll-default) value;
    unlocking reverts it to whatever the roll currently says. No-op for an unknown
    roll id."""
    store = _read(repo)
    entry = store.get(roll_id)
    if entry is None:
        return
    overrides = dict(entry.get("frame_overrides", {}))
    cards = set(overrides.get(file_hash, ()))
    if locked:
        cards.add(card_key)
    else:
        cards.discard(card_key)
    if cards:
        overrides[file_hash] = sorted(cards)
    else:
        overrides.pop(file_hash, None)
    entry["frame_overrides"] = overrides
    _write(repo, store)


def resolve_roll_process_config(repo: Any, roll_id: Optional[str], file_hash: str, process_config: Any) -> Any:
    """Overlay this roll's defaults onto *process_config* for every card the frame has
    not locked to its own value. No roll, no defaults set yet, or every relevant card
    locked leaves *process_config* unchanged."""
    if roll_id is None:
        return process_config
    defaults = roll_defaults(repo, roll_id)
    if not defaults:
        return process_config
    locked_cards = frame_override_cards(repo, roll_id, file_hash)
    updates = {}
    if _MODE_FIELD in defaults:
        updates[_MODE_FIELD] = defaults[_MODE_FIELD]
    for card_key, field_names in ROLL_DEFAULT_FIELDS.items():
        if card_key in locked_cards:
            continue
        for name in field_names:
            if name in defaults:
                updates[name] = defaults[name]
    return replace(process_config, **updates) if updates else process_config
