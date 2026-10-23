import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.lab.logic import _smoothstep, gaussian_kernel_1d, sharpen_shadow_gain
from negpy.kernel.image.logic import WORKING_LAB, lab_to_rgb_working, rgb_to_lab_working

pytestmark = cv2_oracle

_RNG = np.random.default_rng(17)


def _usm_reference(img, amount, radius, masking):
    """The NumPy and OpenCV sharpen block that negpy_core.usm_sharpen replaces."""
    lab = rgb_to_lab_working(img)
    l_chan, a, b = cv2.split(lab)
    k = gaussian_kernel_1d(radius)
    diff = l_chan - cv2.sepFilter2D(l_chan, -1, k, k, borderType=cv2.BORDER_REFLECT_101)
    gain = np.float32(amount * 2.5) * _smoothstep(0.25, 0.33, np.abs(diff)) * sharpen_shadow_gain(l_chan)
    if masking > 0.0:
        lp = np.pad(l_chan, 1, mode="edge")
        gx = (lp[1:-1, 2:] - lp[1:-1, :-2]) * np.float32(0.5)
        gy = (lp[2:, 1:-1] - lp[:-2, 1:-1]) * np.float32(0.5)
        grad = cv2.blur(np.hypot(gx, gy).astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
        gain = gain * _smoothstep(5.0 * masking, 10.0 * masking, grad)
    kern3 = np.ones((3, 3), np.uint8)
    l_min = cv2.erode(l_chan, kern3, borderType=cv2.BORDER_REPLICATE)
    l_max = cv2.dilate(l_chan, kern3, borderType=cv2.BORDER_REPLICATE)
    l_new = np.clip(l_chan + diff * gain, l_min - np.float32(2.0), l_max + np.float32(1.0))
    l_new = np.clip(l_new, 0.0, 100.0)
    return np.clip(lab_to_rgb_working(cv2.merge([l_new.astype(np.float32), a, b])), 0.0, 1.0)


@pytest.mark.parametrize("shape", [(97, 131), (5, 7), (300, 200)])
@pytest.mark.parametrize("amount, radius, masking", [(0.25, 1.0, 0.0), (1.0, 2.3, 0.0), (0.6, 0.8, 0.5), (2.0, 1.0, 1.0)])
def test_usm_sharpen_matches_numpy_and_cv2(shape, amount, radius, masking):
    img = cv2.GaussianBlur(_RNG.random((*shape, 3), dtype=np.float32), (0, 0), 1.2)
    want = _usm_reference(img, amount, radius, masking)
    got = negpy_core.usm_sharpen(img, np.float32(amount * 2.5), masking, gaussian_kernel_1d(radius), *WORKING_LAB)
    assert got.tobytes() == want.tobytes()
