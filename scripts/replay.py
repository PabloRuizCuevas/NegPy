"""Record real calls of the functions in negpy_core/manifest.toml, then replay them through the current code.

  record FILE...   run each frame through the preview load, the CPU pipeline, Square Before Crop and
                   the Auto-crop All Frames evidence, then resolve the files as one roll; save every
                   call of a recorded function: inputs, result and its time on all cores, called
                   alone.
  replay           call the current code with each saved input and compare the result bit for bit.
  bench            time the current code on all cores against the saved time (older recordings
                   saved a one-core time).

Record on main just before a group is ported, so the corpus holds the behavior users have now.
The corpus lives in $NEGPY_CORPUS (default ~/negpy-corpus), on a local disk that every worktree reads.
"""

import argparse
import copy
import dataclasses
import fcntl
import importlib
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from core_manifest import load as load_manifest  # noqa: E402

CORPUS = Path(os.environ.get("NEGPY_CORPUS", Path.home() / "negpy-corpus"))
CALLS_PER_FRAME = 3
CALLS_PER_FUNCTION = 60
MAX_CALL_BYTES = 256 << 20
TIMING_REPEATS = 3
BENCH_LOCK = Path("/tmp/negpy-bench.lock")


def _resolve(key: str):
    """(owner, attribute name, function) for a manifest key, or None for a nested or deleted function."""
    rel, qual = key.split("::")
    module = importlib.import_module("negpy." + rel.removesuffix(".py").replace("/", "."))
    owner, *rest = qual.split(".")
    if not rest:
        # A ported helper whose Python is deleted: its callers' replays cover it.
        return (module, owner, getattr(module, owner)) if hasattr(module, owner) else None
    if len(rest) == 1 and isinstance(getattr(module, owner, None), type):
        cls = getattr(module, owner)
        return cls, rest[0], cls.__dict__[rest[0]]
    return None


def _folder(key: str) -> Path:
    return CORPUS / key.replace("/", "__").replace("::", "--")


def _targets(phases: list[int] | None, statuses: tuple[str, ...]) -> list[str]:
    manifest = load_manifest()
    return sorted(k for k, v in manifest.items() if v["status"] in statuses and (not phases or v["phase"] in phases))


def _time(fn, args, kwargs) -> float:
    """Best of a few runs on every core, as the app runs."""
    import cv2

    cv2.setNumThreads(-1)
    # One untimed call first: the first call in a process starts the thread pools.
    a, kw = copy.deepcopy((args, kwargs))
    fn(*a, **kw)
    best = float("inf")
    for _ in range(TIMING_REPEATS):
        a, kw = copy.deepcopy((args, kwargs))
        start = time.perf_counter()
        fn(*a, **kw)
        best = min(best, time.perf_counter() - start)
    return best


class Recorder:
    def __init__(self, keys: list[str]) -> None:
        self.frame = ""
        self.active = True
        self.per_frame: dict[str, int] = {}
        self.saved: dict[str, int] = {k: len(list(_folder(k).glob("*.pkl"))) for k in keys}
        self.originals: list[tuple] = []
        for key in keys:
            resolved = _resolve(key)
            if resolved is None:
                continue
            owner, name, fn = resolved
            wrapper = self._wrap(key, fn)
            setattr(owner, name, wrapper)
            self.originals.append((owner, name, fn))
            # `from module import fn` binds the name in each importer too.
            for module in list(sys.modules.values()):
                if getattr(module, "__name__", "").startswith("negpy.") and module is not owner:
                    for attr, value in list(vars(module).items()):
                        if value is fn:
                            setattr(module, attr, wrapper)
                            self.originals.append((module, attr, fn))

    def _wrap(self, key: str, fn):
        def wrapper(*args, **kwargs):
            if not self.active or self.per_frame.get(key, 0) >= CALLS_PER_FRAME or self.saved[key] >= CALLS_PER_FUNCTION:
                return fn(*args, **kwargs)
            try:
                inputs = copy.deepcopy((args, kwargs))
            except (TypeError, AttributeError):
                return fn(*args, **kwargs)  # a lock or a device handle in the arguments: not replayable
            result = fn(*args, **kwargs)
            self.per_frame[key] = self.per_frame.get(key, 0) + 1
            self.active = False
            try:
                payload = pickle.dumps({"args": inputs[0], "kwargs": inputs[1], "result": result, "frame": self.frame}, protocol=5)
                if len(payload) <= MAX_CALL_BYTES:
                    record = pickle.loads(payload)
                    record["seconds_all"] = _time(fn, inputs[0], inputs[1])
                    folder = _folder(key)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{self.saved[key]:04d}.pkl").write_bytes(pickle.dumps(record, protocol=5))
                    self.saved[key] += 1
            except (pickle.PicklingError, TypeError, AttributeError):
                pass  # a result that does not pickle: not replayable
            finally:
                self.active = True
            return result

        return wrapper

    def restore(self) -> None:
        for owner, name, fn in reversed(self.originals):
            setattr(owner, name, fn)


def _run_frame(path: str, repo=None, overrides: dict | None = None) -> tuple:
    """The frame's saved edit (or the defaults), with Auto Crop armed so detection runs."""
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.geometry.models import with_auto_crop
    from negpy.kernel.image.logic import file_hashes
    from negpy.services.rendering.image_processor import ImageProcessor
    from negpy.services.rendering.preview_manager import PreviewManager

    buffer, _, meta = PreviewManager().load_linear_preview(path)
    saved = repo.load_file_settings(file_hashes(path)[0]) if repo else None
    config = with_auto_crop(saved or WorkspaceConfig(), True)
    for section, fields in (overrides or {}).items():
        config = dataclasses.replace(config, **{section: dataclasses.replace(getattr(config, section), **fields)})
    # Stage caching on, as the preview runs: the base read (measure_base_log) happens only then.
    ImageProcessor(use_gpu=False).run_pipeline(
        buffer, config, source_hash=path, render_size_ref=1600.0, prefer_gpu=False, ir_buffer=meta.get("ir_preview")
    )
    return buffer, config


def _crop_evidence(key: str, buffer, config):
    """One frame of Auto-crop All Frames and Square Before Crop, as the batch worker runs them
    (without the flat field)."""
    from dataclasses import replace

    from negpy.domain.interfaces import PipelineContext
    from negpy.features.geometry.batch_autocrop import detect_crop_candidate
    from negpy.features.geometry.processor import GeometryProcessor
    from negpy.features.geometry.skew import squared_geometry

    geometry = config.geometry
    squared_geometry(buffer, geometry, config.process.process_mode)
    detection = replace(geometry, crop_rect=None, crop_from_auto=False, autocrop_offset=0)
    context = PipelineContext(original_size=(buffer.shape[1], buffer.shape[0]), scale_factor=1.0, process_mode=config.process.process_mode)
    transformed = GeometryProcessor(detection).process(buffer, context)
    return detect_crop_candidate(key, transformed, target_ratio=geometry.autocrop_ratio, rebate_trim=geometry.autocrop_rebate_trim)


def _edits_repo(edits_db: str | None):
    """A repository over a copy of *edits_db*, so recording never writes to the user's edits."""
    if not edits_db:
        return None
    import shutil
    import tempfile

    from negpy.infrastructure.storage.repository import StorageRepository

    folder = Path(tempfile.mkdtemp(prefix="negpy-replay-"))
    shutil.copy(edits_db, folder / "edits.db")
    return StorageRepository(str(folder / "edits.db"), str(folder / "settings.db"))


def record(files: list[str], phases: list[int] | None, edits_db: str | None, overrides: dict | None = None) -> None:
    keys = _targets(phases, ("todo",))
    repo = _edits_repo(edits_db)
    import negpy.services.rendering.image_processor  # noqa: F401  (load the pipeline before wrapping)
    import negpy.services.rendering.preview_manager  # noqa: F401

    from negpy.features.geometry.batch_autocrop import resolve_roll_crops

    recorder = Recorder(keys)
    try:
        evidence = []
        for path in files:
            recorder.frame, recorder.per_frame = Path(path).name, {}
            buffer, config = _run_frame(path, repo, overrides)
            evidence.append(_crop_evidence(str(len(evidence)), buffer, config))
            print(f"recorded {path}")
        recorder.frame, recorder.per_frame = "roll", {}
        resolve_roll_crops(evidence)
    finally:
        recorder.restore()
    for key in keys:
        if recorder.saved.get(key):
            print(f"{recorder.saved[key]:4d}  {key}")


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in units in the last place between two float arrays of one dtype."""
    ints = {np.dtype(np.float32): np.int32, np.dtype(np.float64): np.int64}[a.dtype]
    ia, ib = (np.ascontiguousarray(x).view(ints).astype(np.int64) for x in (a, b))
    # Map the sign-magnitude float bits onto one ordered integer line.
    ia = np.where(ia < 0, np.iinfo(ints).min - ia, ia)
    ib = np.where(ib < 0, np.iinfo(ints).min - ib, ib)
    return np.abs(ia - ib)


def same(a, b, compare: str = "exact") -> bool:
    """Equality for replay. "exact": bit for bit, NaN payloads and the sign of zero included.
    "rows": a 2-D array's rows in any order. "ulp=N": floats within N units in the last place.
    "abs=X": floats within X of each other."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if compare == "rows" and a.ndim == 2:
            rows = f"V{a.shape[1] * a.dtype.itemsize}"
            a, b = (np.sort(np.ascontiguousarray(x).view(rows).ravel()) for x in (a, b))
            return a.tobytes() == b.tobytes()
        if compare.startswith("ulp=") and a.dtype.kind == "f":
            nan = np.isnan(a)
            return bool(np.array_equal(nan, np.isnan(b)) and np.all(_ulps(a[~nan], b[~nan]) <= int(compare[4:])))
        if compare.startswith("abs=") and a.dtype.kind == "f":
            nan = np.isnan(a)
            return bool(np.array_equal(nan, np.isnan(b)) and np.all(np.abs(a[~nan] - b[~nan]) <= float(compare[4:])))
        return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    if isinstance(a, (np.generic, float)):
        if compare.startswith(("ulp=", "abs=")):
            return same(np.atleast_1d(np.asarray(a)), np.atleast_1d(np.asarray(b)), compare)
        return np.asarray(a).tobytes() == np.asarray(b).tobytes()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, compare) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], compare) for k in a)
    if dataclasses.is_dataclass(a) or hasattr(a, "__dict__"):
        return same(vars(a), vars(b), compare)
    return a == b


KEY_FILTER = ""


def _calls(keys: list[str]):
    for key in [k for k in keys if KEY_FILTER in k]:
        resolved = _resolve(key)
        files = sorted(_folder(key).glob("*.pkl"))
        if resolved is None or not files:
            continue
        yield key, resolved[2], files


def replay(phases: list[int] | None) -> int:
    failed = 0
    manifest = load_manifest()
    for key, fn, files in _calls(_targets(phases, ("todo", "ported", "stays"))):
        compare = manifest[key].get("compare", "exact")
        bad = []
        for f in files:
            rec = pickle.loads(f.read_bytes())
            result = fn(*rec["args"], **rec["kwargs"])
            # A "quality" port has its own gate against the replaced cv2/NumPy call in pytest.
            if compare != "quality" and not same(result, rec["result"], compare):
                bad.append(f"{f.name} ({rec['frame']})")
        failed += bool(bad)
        print(
            f"{'DIFF' if bad else 'qual' if compare == 'quality' else 'same'}{len(files) - len(bad):3d}/{len(files):<3d} {key}  {' '.join(bad[:5])}"
        )
    return 1 if failed else 0


def bench(phases: list[int] | None) -> int:
    slower = 0
    BENCH_LOCK.touch(exist_ok=True)
    with BENCH_LOCK.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one benchmark at a time on this machine
        for key, fn, files in _calls(_targets(phases, ("ported",))):
            ratios = []
            for f in files:
                rec = pickle.loads(f.read_bytes())
                # Older recordings saved a one-core time only.
                saved = rec.get("seconds_all", rec.get("seconds"))
                ratios.append(_time(fn, rec["args"], rec["kwargs"]) / max(saved, 1e-9))
            ratio = statistics.median(ratios)
            slower += ratio > 1.0
            print(f"{'SLOWER' if ratio > 1.0 else 'ok':6s} {ratio:6.3f}x  {key}")
    return 1 if slower else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("record", "replay", "bench"))
    parser.add_argument("files", nargs="*")
    parser.add_argument("--phase", type=int, action="append", help="only functions of this phase (repeatable)")
    parser.add_argument("--key", default="", help="only functions whose manifest key holds this text")
    parser.add_argument("--edits", help="record each frame with its saved edit from this edits.db (read from a copy)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.FIELD=JSON",
        help="override one config field for every frame, e.g. retouch.dust_remove=true",
    )
    args = parser.parse_args()
    global KEY_FILTER
    KEY_FILTER = args.key
    if args.mode == "record":
        if not args.files:
            parser.error("record needs at least one frame")
        overrides: dict = {}
        for item in args.set:
            key, value = item.split("=", 1)
            section, name = key.split(".", 1)
            overrides.setdefault(section, {})[name] = json.loads(value)
        record(args.files, args.phase, args.edits, overrides)
    else:
        sys.exit((replay if args.mode == "replay" else bench)(args.phase))


if __name__ == "__main__":
    main()
