# Bug hunting

Open, unconfirmed leads worth a proper look later — not user-facing, not kept in
sync with the code the way `USER_GUIDE.md`/`PIPELINE.md` are. Each entry states what
is confirmed by reading the code versus what is still a guess, so whoever picks it
up isn't starting from zero.

## Rare wrong-color flash after "Apply to All Roll", self-corrects on any edit

**Reported:** navigate to a frame right after using Apply to All Roll (Calibration,
in the report) and it renders with a strong wrong color cast ("weirdly red") —
consistent with a stale or mismatched Crosstalk/sensor bake, not a decode-level
issue (a raw, un-inverted negative reads warm/orange, which is close to what was
shown). Any edit, however small, forces a correct re-render; reverting that edit
does not bring the wrong color back — the bad state does not persist once anything
forces a fresh render. Confirmed **very rare** and **not reliably reproducible** by
the reporter (happened at least twice, not on demand).

**Why this points at a cache/race, not a logic bug:** a deterministic mistake in a
cache key or in how roll defaults resolve would misfire routinely, not "very
rarely" with a self-healing follow-up edit. That pattern fits a narrow timing
window between two things racing on frame switch, not a wrong value being computed.

### Confirmed mechanisms that could plausibly be involved

1. **Navigate-back render memo** (`AppController.load_file`,
   `negpy/desktop/controller.py` ~1852–1902, and `_render_memo_key`, ~1770–1790).
   Switching frames paints a memoized buffer for the target hash *immediately*,
   before the real render completes, if `_render_memo.get(target_hash,
   _render_memo_key())` hits — the comment says outright: "Paint it now, with no
   spinner and no toasts, and let the real render refresh the metrics." The memo
   key is computed from `self.state.config` right after `select_file` has already
   hydrated it for the new frame (roll defaults included, via
   `_overlay_roll_defaults`), so a stale hit should require the *exact* same config
   dict to have been stored earlier under that key — plausible only if the roll
   defaults write and whatever populated/read the memo land in an unlucky order.

2. **Neighbor prefetch** (`AppController._schedule_prefetch_neighbors`,
   `negpy/desktop/controller.py` ~2063–2105). Frames adjacent to the active one are
   speculatively decoded ~50 ms after a load, keyed on `saved = repo.load_file_settings(h)`
   — each neighbor's own **raw saved settings**, not the roll-resolved effective
   config (`config_for_asset`/`_overlay_roll_defaults` is not called here). The
   `PreviewLoadTask` this builds only carries `linear_raw`/`positive_source`/
   `demosaic`/half-slice, not Crosstalk or sensor fields, so on its own this warms
   the wrong thing only for those source-level fields — but if a neighbor gets
   prefetched in the narrow window between an Apply-to-All-Roll write and the
   navigation to it, whatever downstream cache keys off this prefetched buffer
   inherits whatever staleness it carries.

3. **Sensor/Crosstalk bake token** (`sensor_token`,
   `negpy/features/process/sensor.py:97`, folded into `source_hash` at several
   sites in `negpy/services/rendering/image_processor.py`, e.g. ~644, 676, 880,
   1120). This looks correctly reactive to config on its own — it hashes the
   *effective* sensor matrix every time it's called — so the open question is
   only ever whether it gets called against the right (roll-resolved) config at
   the right moment during a fast frame switch, not whether the token itself is
   wrong.

### Not yet checked

- `DarkroomEngine._run_stage`'s per-config-hash cache (CPU) and `GPUEngine`'s own
  config-diff change detection (`negpy/services/rendering/gpu_engine.py`) — neither
  has been read with this race in mind yet.
- Whether `rolls.set_roll_defaults` (`negpy/services/assets/rolls.py`) writing
  mid-navigation could be observed by `_overlay_roll_defaults` in a half-written
  state — `_write`/`_read` go through `repo.save_global_setting`/`get_global_setting`,
  whose own atomicity hasn't been checked here.

### Suggested next step

Don't patch from this write-up alone — the mechanism is still a guess. Reproduce
first, per the `verify` skill: drive Apply to All Roll on Calibration immediately
followed by rapid navigation across many neighbor frames, in a tight loop, to try
to hit the timing window on demand; only then trace which of the three mechanisms
above (if any) actually served the stale buffer, with real evidence instead of a
static-reading hypothesis.
