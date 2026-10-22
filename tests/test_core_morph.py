import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(12)
_OPS = [cv2.MORPH_ERODE, cv2.MORPH_DILATE, cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT]
_SHAPES = [(41, 83, 1), (37, 53, 3), (1, 301, 1), (300, 1, 1), (13, 9, 1)]


def _image(dtype, h, w, cn):
    shape = (h, w, cn) if cn > 1 else (h, w)
    if dtype == np.uint8:
        return _RNG.integers(0, 256, shape).astype(np.uint8)
    return (_RNG.random(shape, dtype=np.float32) - 0.3) * 3


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("ksize, iterations", [((31, 31), 2), ((9, 9), 1), ((5, 5), 1), ((1, 7), 2), ((4, 6), 1), ((1, 1), 1)])
def test_morphology_matches_cv2_bit_for_bit(dtype, shape, op, ksize, iterations):
    img = _image(dtype, *shape)
    kernel = np.ones((ksize[1], ksize[0]), np.uint8)
    want = cv2.morphologyEx(img, op, kernel, iterations=iterations)
    got = negpy_core.morphology_ex(img, op, kernel, iterations)
    assert got.dtype == want.dtype
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("border", [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT_101])
@pytest.mark.parametrize("op", [cv2.MORPH_ERODE, cv2.MORPH_DILATE])
def test_erode_dilate_borders_match_cv2(border, op):
    img = _image(np.float32, 37, 53, 3)
    kernel = np.ones((7, 5), np.uint8)
    want = cv2.morphologyEx(img, op, kernel, iterations=2, borderType=border)
    assert negpy_core.morphology_ex(img, op, kernel, 2, border).tobytes() == want.tobytes()


def test_morphology_rejects_a_non_rectangular_kernel():
    with pytest.raises(ValueError):
        negpy_core.morphology_ex(_image(np.uint8, 9, 9, 1), cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
