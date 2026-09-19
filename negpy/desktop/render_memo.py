"""Per-frame memo of the last displayed render, for instant frame switching.

Navigating back to a frame whose edits haven't changed would re-run the whole
pipeline just to reproduce pixels that were already on screen — at HQ (full
resolution) that is seconds of spinner. The controller stores each frame's last
rendered display buffer here, keyed by everything that shaped it; on navigate-
back with a matching key the canvas is painted from the memo immediately and
the authoritative render refreshes metrics quietly in the background (so a
stale memo can only ever flash briefly, never persist).

Buffers are stored by reference under the same read-only contract as the
preview cache. A GPU render lands here as the texture itself, retained out of the
engine's pool on the file switch — so the memo owns those textures and frees them.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

from negpy.kernel.system.config import APP_CONFIG


class RenderMemo:
    """Bounded render LRU, optionally retaining multiple settings per file."""

    def __init__(self, app_config: Any = None, *, keep_variants: bool = False) -> None:
        self._app = app_config or APP_CONFIG
        self._keep_variants = keep_variants
        self._entries: "OrderedDict[tuple[str, str], tuple[str, dict]]" = OrderedDict()
        # Hundreds of MB an entry (HQ renders, strip mosaics): use the full-res knob.
        self.large_entries = False

    def _entry_key(self, file_hash: str, memo_key: str) -> tuple[str, str]:
        return file_hash, memo_key if self._keep_variants else ""

    def _budget(self) -> int:
        if self.large_entries:
            return max(2, int(getattr(self._app, "preview_cache_max_full_res_entries", 2)))
        return max(2, int(getattr(self._app, "render_memo_max_entries", 8)))

    def _dispose(self, entry: "Optional[tuple[str, dict]]", keep: "Optional[dict]" = None) -> None:
        """Free the GPU textures an entry owns; strip mosaics are arrays and pass
        through. ``keep`` spares what the replacing payload also holds — re-storing a
        frame served *from* the memo hands back that entry's own texture."""
        if entry is None:
            return
        spared = [] if keep is None else list(keep.values())
        for value in entry[1].values():
            destroy = getattr(value, "destroy", None)
            if callable(destroy) and not any(value is s for s in spared):
                destroy()

    def store(self, file_hash: str, memo_key: str, payload: dict) -> None:
        if not file_hash or not memo_key:
            return
        key = self._entry_key(file_hash, memo_key)
        self._dispose(self._entries.pop(key, None), keep=payload)
        self._entries[key] = (memo_key, payload)
        while len(self._entries) > self._budget():
            self._dispose(self._entries.popitem(last=False)[1], keep=payload)

    def get(self, file_hash: str, memo_key: str) -> Optional[dict]:
        key = self._entry_key(file_hash, memo_key)
        entry = self._entries.get(key)
        if entry is None or entry[0] != memo_key:
            return None
        self._entries.move_to_end(key)
        return entry[1]

    def rekey(self, file_hash: str, new_key: str, *, old_key: str = "") -> None:
        """Follow a render-neutral config change (e.g. measured bounds persisted
        after the render, with render=False): the stored pixels are still valid,
        only their identity moved."""
        if not new_key or (self._keep_variants and not old_key):
            return
        entry = self._entries.pop(self._entry_key(file_hash, old_key), None)
        if entry is not None:
            self.store(file_hash, new_key, entry[1])

    def invalidate(self, file_hash: str) -> None:
        for key in [key for key in self._entries if key[0] == file_hash]:
            self._dispose(self._entries.pop(key))

    def clear(self) -> None:
        for entry in self._entries.values():
            self._dispose(entry)
        self._entries.clear()
