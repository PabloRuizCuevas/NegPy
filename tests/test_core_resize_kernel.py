"""Quality gate for negpy_core.resize_kernel against cv2's INTER_LANCZOS4 and INTER_CUBIC."""

import cv2
import numpy as np
import pytest
from core_oracle import cv2_oracle

import negpy_core

pytestmark = cv2_oracle

CASES = [
    ((120, 170, 3), (400, 290)),
    ((300, 400, 3), (1003, 751)),
    ((64, 80), (97, 61)),
    ((200, 300, 4), (301, 201)),
    ((50, 70, 3), (35, 25)),
]


@pytest.mark.parametrize("shape,dsize", CASES)
@pytest.mark.parametrize("lanczos,flag,tol", [(True, cv2.INTER_LANCZOS4, 1e-6), (False, cv2.INTER_CUBIC, 1e-5)])
def test_resize_kernel_stays_near_cv2(shape, dsize, lanczos, flag, tol):
    rng = np.random.default_rng(sum(shape))
    src = cv2.GaussianBlur(rng.uniform(0, 1, shape).astype(np.float32), (0, 0), 1.2)
    want = cv2.resize(src, dsize, interpolation=flag)
    got = negpy_core.resize_kernel(np.ascontiguousarray(src), dsize, lanczos=lanczos)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=tol)
