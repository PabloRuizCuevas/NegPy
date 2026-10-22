import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_current():
    result = subprocess.run([sys.executable, "scripts/core_manifest.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_fields_are_valid():
    manifest = tomllib.loads((ROOT / "negpy_core" / "manifest.toml").read_text(encoding="utf-8"))
    for key, entry in manifest.items():
        assert entry["status"] in ("todo", "ported", "stays"), key
        assert entry["phase"] in range(1, 10), key
