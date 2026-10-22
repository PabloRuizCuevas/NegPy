"""Record real calls of the functions in negpy_core/manifest.toml, then replay them through the current code.

  record FILE...   run each frame through the preview load and the CPU pipeline, and save every call
                   of a recorded function: inputs, result and its time on one thread, called alone.
  replay           call the current code with each saved input and compare the result bit for bit.
  bench            time the current code on one thread against the saved time.

Record on main just before a group is ported, so the corpus holds the behavior users have now.
The corpus lives in $NEGPY_CORPUS (default ~/negpy-corpus), on a local disk that every worktree reads.
"""

import argparse
import copy
import dataclasses
import fcntl
import importlib
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("record", "bench"):
    # One thread on both sides: the speed gate is per core. Numba reads this at import.
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"

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
    """(owner, attribute name, function) for a manifest key, or None for a nested function."""
    rel, qual = key.split("::")
    module = importlib.import_module("negpy." + rel.removesuffix(".py").replace("/", "."))
    owner, *rest = qual.split(".")
    if not rest:
        return module, owner, getattr(module, owner)
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
            inputs = copy.deepcopy((args, kwargs))
            result = fn(*args, **kwargs)
            self.per_frame[key] = self.per_frame.get(key, 0) + 1
            self.active = False
            try:
                payload = pickle.dumps({"args": inputs[0], "kwargs": inputs[1], "result": result, "frame": self.frame}, protocol=5)
                if len(payload) <= MAX_CALL_BYTES:
                    record = pickle.loads(payload)
                    record["seconds"] = _time(fn, inputs[0], inputs[1])
                    folder = _folder(key)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{self.saved[key]:04d}.pkl").write_bytes(pickle.dumps(record, protocol=5))
                    self.saved[key] += 1
            except (pickle.PicklingError, TypeError, AttributeError):
                pass  # a GPU texture or a lock in the arguments: not replayable
            finally:
                self.active = True
            return result

        return wrapper

    def restore(self) -> None:
        for owner, name, fn in reversed(self.originals):
            setattr(owner, name, fn)


def _run_frame(path: str, repo=None) -> None:
    """The frame's saved edit (or the defaults), with Auto Crop armed so detection runs."""
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.geometry.models import with_auto_crop
    from negpy.kernel.image.logic import file_hashes
    from negpy.services.rendering.image_processor import ImageProcessor
    from negpy.services.rendering.preview_manager import PreviewManager

    buffer, _, _ = PreviewManager().load_linear_preview(path)
    saved = repo.load_file_settings(file_hashes(path)[0]) if repo else None
    config = with_auto_crop(saved or WorkspaceConfig(), True)
    # Stage caching on, as the preview runs: the base read (measure_base_log) happens only then.
    ImageProcessor(use_gpu=False).run_pipeline(buffer, config, source_hash=path, render_size_ref=1600.0, prefer_gpu=False)


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


def record(files: list[str], phases: list[int] | None, edits_db: str | None) -> None:
    keys = _targets(phases, ("todo",))
    repo = _edits_repo(edits_db)
    import negpy.services.rendering.image_processor  # noqa: F401  (load the pipeline before wrapping)
    import negpy.services.rendering.preview_manager  # noqa: F401

    recorder = Recorder(keys)
    try:
        for path in files:
            recorder.frame, recorder.per_frame = Path(path).name, {}
            _run_frame(path, repo)
            print(f"recorded {path}")
    finally:
        recorder.restore()
    for key in keys:
        if recorder.saved.get(key):
            print(f"{recorder.saved[key]:4d}  {key}")


def same(a, b) -> bool:
    """Bit-for-bit equality, NaN payloads and the sign of zero included."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    if isinstance(a, (np.generic, float)):
        return np.asarray(a).tobytes() == np.asarray(b).tobytes()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if dataclasses.is_dataclass(a) or hasattr(a, "__dict__"):
        return same(vars(a), vars(b))
    return a == b


def _calls(keys: list[str]):
    for key in keys:
        resolved = _resolve(key)
        files = sorted(_folder(key).glob("*.pkl"))
        if resolved is None or not files:
            continue
        yield key, resolved[2], files


def replay(phases: list[int] | None) -> int:
    failed = 0
    for key, fn, files in _calls(_targets(phases, ("todo", "ported"))):
        bad = []
        for f in files:
            rec = pickle.loads(f.read_bytes())
            if not same(fn(*rec["args"], **rec["kwargs"]), rec["result"]):
                bad.append(f"{f.name} ({rec['frame']})")
        failed += bool(bad)
        print(f"{'DIFF' if bad else 'same'}  {len(files) - len(bad):3d}/{len(files):<3d} {key}  {' '.join(bad[:5])}")
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
                ratios.append(_time(fn, rec["args"], rec["kwargs"]) / max(rec["seconds"], 1e-9))
            ratio = statistics.median(ratios)
            slower += ratio > 1.0
            print(f"{'SLOWER' if ratio > 1.0 else 'ok':6s} {ratio:6.3f}x  {key}")
    return 1 if slower else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("record", "replay", "bench"))
    parser.add_argument("files", nargs="*")
    parser.add_argument("--phase", type=int, action="append", help="only functions of this phase (repeatable)")
    parser.add_argument("--edits", help="record each frame with its saved edit from this edits.db (read from a copy)")
    args = parser.parse_args()
    if args.mode == "record":
        if not args.files:
            parser.error("record needs at least one frame")
        record(args.files, args.phase, args.edits)
    else:
        sys.exit((replay if args.mode == "replay" else bench)(args.phase))


if __name__ == "__main__":
    main()
