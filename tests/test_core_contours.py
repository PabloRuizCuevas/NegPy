import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(14)


def _mask(kind, shape):
    if kind == "noise":
        m = _RNG.random(shape) > 0.5
    elif kind == "blobs":
        m = cv2.GaussianBlur(_RNG.random(shape).astype(np.float32), (0, 0), 3) > 0.5
    elif kind == "sparse":
        m = _RNG.random(shape) > 0.97
    else:
        m = np.zeros(shape, bool)
        for _ in range(5):
            y0, x0 = _RNG.integers(0, shape[0]), _RNG.integers(0, shape[1])
            m[y0 : y0 + _RNG.integers(1, 60), x0 : x0 + _RNG.integers(1, 60)] ^= True
    return m.astype(np.uint8) * np.uint8(255)


@pytest.mark.parametrize("kind", ["noise", "blobs", "sparse", "rects"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 50), (50, 1), (3, 3), (120, 97), (199, 200)])
def test_find_external_contours_matches_cv2(kind, shape):
    img = _mask(kind, shape)
    want, want_h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    got, got_h = negpy_core.find_external_contours(img)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape and np.array_equal(g, w)
    assert (want_h is None and got_h is None) or np.array_equal(got_h, want_h)
