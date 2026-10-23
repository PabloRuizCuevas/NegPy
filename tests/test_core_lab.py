import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.lab.logic import _smoothstep, gaussian_kernel_1d, sharpen_shadow_gain
from negpy.features.lab.logic import LUM_B, LUM_G, LUM_R, apply_glow_and_halation
from negpy.kernel.image.logic import WORKING_LAB, lab_to_rgb_working, rgb_to_lab_working, working_oetf_encode

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


def _chroma_reference(img, sigma):
    """The NumPy and OpenCV chroma denoise that negpy_core.chroma_denoise replaces."""
    lab = rgb_to_lab_working(img)
    ab = np.ascontiguousarray(lab[:, :, 1:3])
    k = gaussian_kernel_1d(sigma)
    low = cv2.sepFilter2D(ab, -1, k, k)
    diff = np.array([[-0.5, 0.0, 0.5]], dtype=np.float32)
    gx = cv2.filter2D(low, -1, diff, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(low, -1, diff.T, borderType=cv2.BORDER_REPLICATE)
    g = np.sqrt((gx * gx + gy * gy).sum(axis=2, dtype=np.float32))
    g = g * np.float32(sigma / 1.5) - np.float32(1.0 / 1.5)
    t = np.clip(g, 0.0, 1.0)
    ab = (ab - low) * (t * t * (3.0 - 2.0 * t))[:, :, None] + low
    lab[:, :, 1:3] = ab
    return np.clip(lab_to_rgb_working(lab), 0.0, 1.0)


@pytest.mark.parametrize("shape", [(97, 131), (3, 4), (300, 200)])
@pytest.mark.parametrize("sigma", [0.5, 2.0, 6.0])
def test_chroma_denoise_matches_numpy_and_cv2(shape, sigma):
    img = _RNG.random((*shape, 3), dtype=np.float32)
    want = _chroma_reference(img, sigma)
    got = negpy_core.chroma_denoise(img, sigma, gaussian_kernel_1d(sigma), *WORKING_LAB)
    assert got.tobytes() == want.tobytes()


def _glow_reference(img, glow_amount, halation_strength, scale_factor):
    """The NumPy and OpenCV bloom that negpy_core.glow_and_halation replaces."""
    lum = (np.float32(LUM_R), np.float32(LUM_G), np.float32(LUM_B))
    result = img.copy()
    if glow_amount > 0.0:
        enc = working_oetf_encode(img)
        mask = np.clip((enc[:, :, 0] * lum[0] + enc[:, :, 1] * lum[1] + enc[:, :, 2] * lum[2] - 0.5) / 0.5, 0.0, 1.0)
        r = max(3, int(15 * scale_factor))
        result = result + cv2.GaussianBlur(img * mask[:, :, None], (min((r * 2 + 1) | 1, 201),) * 2, r * 0.5) * glow_amount
    if halation_strength > 0.0:
        mask = np.clip((img[:, :, 0] * lum[0] + img[:, :, 1] * lum[1] + img[:, :, 2] * lum[2] - 0.65) / (1.0 - 0.65), 0.0, 1.0)
        red = img[:, :, 0] * mask
        r = max(5, int(25 * scale_factor))
        hl = np.stack([red, red * 0.3, red * 0.05], axis=-1)
        result = result + cv2.GaussianBlur(hl, (min((r * 2 + 1) | 1, 301),) * 2, r * 0.5) * halation_strength
    return np.clip(result, 0.0, 1.0)


@pytest.mark.parametrize("shape", [(97, 131), (5, 7)])
@pytest.mark.parametrize("glow, halation, scale", [(0.3, 0.0, 1.0), (0.0, 0.5, 1.0), (0.4, 0.2, 0.5), (1.0, 1.0, 2.0)])
def test_glow_and_halation_matches_numpy_and_cv2(shape, glow, halation, scale):
    img = _RNG.random((*shape, 3), dtype=np.float32)
    want = _glow_reference(img, glow, halation, scale)
    assert apply_glow_and_halation(img, glow, halation, scale).tobytes() == want.tobytes()
