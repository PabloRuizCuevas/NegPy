"""Keep negpy_core/manifest.toml in step with the Python functions that negpy_core will replace."""

import argparse
import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "negpy_core" / "manifest.toml"

ORDER_STATS = {"percentile", "median", "partition", "quantile", "nanpercentile", "nanmedian"}
# Reductions over a whole image: the Python-overhead chains (measure_base_log, bounds) live on these.
REDUCTIONS = {"std", "var", "mean", "sum", "histogram", "bincount", "cumsum", "sort", "argsort", "unique"}
# cv2 calls the plan ports last, or may never port: each needs a decision of its own.
LAST_CV2 = {"SIFT_create", "BFMatcher", "detail_DpSeamFinder", "createLineSegmentDetector", "inpaint", "phaseCorrelate", "UMat"}
STATUSES = ("todo", "ported", "stays")
# Where a pure-Python caller of ported code is itself ported; orchestration elsewhere stays.
CHAIN_PREFIXES = ("features/", "kernel/image/")

# First match wins. Entries keep a hand-set phase; this only seeds new ones.
PHASE_BY_PATH = [
    ("features/geometry/", 2),
    ("features/exposure/normalization.py", 2),
    ("features/exposure/analysis.py", 2),
    ("infrastructure/loaders/", 3),
    ("kernel/image/", 3),
    ("features/flatfield/", 4),
    ("features/process/", 4),
    ("features/retouch/", 4),
    ("features/lens/", 4),
    ("features/local/", 4),
    ("features/hdr/", 5),
    ("features/stitch/", 5),
    ("features/rgbscan/", 5),
    ("features/", 6),
    ("services/rendering/", 6),
    ("infrastructure/display/", 7),
    ("services/", 7),
    ("desktop/", 7),
    ("infrastructure/", 9),
]


def _phase(rel: str, calls: list[str]) -> int:
    if any(c.removeprefix("cv2.") in LAST_CV2 for c in calls):
        return 9
    return next((phase for prefix, phase in PHASE_BY_PATH if rel.startswith(prefix)), 9)


class _Scanner(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.stack: list[str] = []
        self.found: dict[str, dict] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        jit = any("jit" in ast.unparse(d) for d in node.decorator_list)
        calls: set[str] = set()
        names: set[str] = set()
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if f.value.id in ("cv2", "negpy_core") or (f.value.id == "np" and f.attr in ORDER_STATS | REDUCTIONS):
                    calls.add(f"{f.value.id}.{f.attr}")
            elif isinstance(f, ast.Name):
                names.add(f.id)
        if jit:
            calls.add("numba")
        # Every function is kept for now; scan() drops those outside the closure.
        self.found[f"{self.rel}::{'.'.join(self.stack)}"] = {"calls": sorted(calls), "names": names}
        self.generic_visit(node)
        self.stack.pop()


def scan() -> dict[str, dict]:
    """Every function under negpy/ that calls cv2, negpy_core, a numpy order statistic or reduction, or is a Numba kernel."""
    found: dict[str, dict] = {}
    for path in sorted((ROOT / "negpy").rglob("*.py")):
        rel = path.relative_to(ROOT / "negpy").as_posix()
        scanner = _Scanner(rel)
        scanner.visit(ast.parse(path.read_text(encoding="utf-8")))
        found.update(scanner.found)
    # In scope: a function with a hot call, or one that calls an in-scope function by name (the
    # pure-Python links of a chain such as measure_base_log), to a fixed point.
    scope = {k for k, v in found.items() if v["calls"]}
    while True:
        names_in = {k.split("::")[1].split(".")[-1] for k in scope}
        grown = {k for k, v in found.items() if k not in scope and k.startswith(CHAIN_PREFIXES) and v["names"] & names_in}
        if not grown:
            break
        scope |= grown
    found = {k: v for k, v in found.items() if k in scope}
    by_module: dict[str, set[str]] = {}
    for key in found:
        module, qual = key.split("::")
        by_module.setdefault(module, set()).add(qual.split(".")[-1])
    for key, entry in found.items():
        module, qual = key.split("::")
        own = qual.split(".")[-1]
        entry["depends"] = sorted(n for n in entry.pop("names") & by_module[module] if n != own)
    return found


def load() -> dict[str, dict]:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def merge(current: dict[str, dict], found: dict[str, dict]) -> dict[str, dict]:
    """Scanned call sites with the hand-kept fields (phase, status, owner, note) of existing entries.
    A ported entry stays when its function is gone from Python or calls negpy_core."""
    merged: dict[str, dict] = {}
    for key, entry in found.items():
        old = current.get(key, {})
        merged[key] = {
            "phase": old.get("phase", _phase(key.split("::")[0], entry["calls"])),
            "status": old.get("status", "todo"),
            "calls": entry["calls"],
            "depends": entry["depends"],
            **{k: old[k] for k in ("owner", "compare", "note") if k in old},
        }
    for key, old in current.items():
        if key not in merged and old.get("status") == "ported":
            merged[key] = old
    return merged


def _toml_value(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, int):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def dump(manifest: dict[str, dict]) -> str:
    lines = [
        "# One entry per Python function that negpy_core replaces. scripts/core_manifest.py writes",
        "# calls and depends; phase, status (todo, ported, stays), owner, compare (replay: exact, rows,",
        "# ulp=N, abs=X, quality: a pytest gate; see negpy_core/LEDGER.md) and note are kept by hand.",
    ]
    for key in sorted(manifest, key=lambda k: (manifest[k]["phase"], k)):
        lines.append("")
        lines.append(f"[{_toml_value(key)}]")
        for field in ("phase", "status", "owner", "compare", "calls", "depends", "note"):
            if field in manifest[key]:
                lines.append(f"{field} = {_toml_value(manifest[key][field])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the manifest is out of date")
    args = parser.parse_args()
    text = dump(merge(load(), scan()))
    if args.check:
        if MANIFEST.read_text(encoding="utf-8") != text:
            raise SystemExit("negpy_core/manifest.toml is out of date: run uv run python scripts/core_manifest.py")
        return
    MANIFEST.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
