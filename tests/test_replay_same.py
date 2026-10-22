import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay import same  # noqa: E402


def test_same_is_bitwise():
    a = np.array([1.0, np.nan, 0.0], dtype=np.float32)
    assert same(a, a.copy())
    assert not same(a, a.astype(np.float64))
    assert not same(a, np.array([1.0, np.nan, -0.0], dtype=np.float32))
    assert not same(np.float32(0.1), np.float64(0.1))
    assert same(float("nan"), float("nan"))
    assert not same(0.0, -0.0)


def test_same_walks_containers():
    assert same((1, [np.uint8(3)], {"k": None}), (1, [np.uint8(3)], {"k": None}))
    assert not same((1, 2), [1, 2])
    assert not same({"k": np.int32(1)}, {"k": np.int64(1)})
