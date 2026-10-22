import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

pytestmark = cv2_oracle

_RNG = np.random.default_rng(5)


def _image(h, w, cn):
    return _RNG.random((h, w, cn) if cn > 1 else (h, w), dtype=np.float32) ** 2.2


@pytest.mark.parametrize("cn", [1, 3, 4])
@pytest.mark.parametrize(
    "shape, dsize",
    [
        ((394, 566), (160, 111)),  # the detection downscale, at a tenth of its size
        ((1600, 1059), (1059, 701)),
        ((37, 53), (17, 11)),
        ((100, 98), (49, 40)),  # integer scale on one axis only
        ((64, 64), (32, 32)),  # 2x2: SIMD order
        ((100, 98), (49, 50)),
        ((40, 74), (37, 20)),  # 2x2 with a scalar tail column
        ((90, 90), (30, 30)),  # 3x3
        ((80, 80), (20, 20)),  # 4x4
        ((80, 60), (60, 80)),  # same size
    ],
)
def test_resize_area_matches_cv2_bit_for_bit(shape, dsize, cn):
    img = _image(*shape, cn)
    want = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
    got = negpy_core.resize_area(img, dsize)
    assert got.shape == want.shape
    assert got.tobytes() == want.tobytes()


def test_resize_area_rejects_upscale_and_strided_input():
    img = _image(20, 20, 3)
    with pytest.raises(ValueError):
        negpy_core.resize_area(img, (40, 40))
    with pytest.raises(ValueError):
        negpy_core.resize_area(img[:, ::2], (5, 10))
