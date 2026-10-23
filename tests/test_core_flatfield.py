"""The flat-field reference checks against the NumPy and cv2 code they replace."""

import cv2
import numpy as np
from core_oracle import cv2_oracle

from negpy.features.flatfield import logic as F

pytestmark = cv2_oracle


def _reference(shape=(170, 256), seed=0):
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x - w / 2) / w) ** 2 + ((y - h / 2) / h) ** 2
    ref = (0.8 * (1.0 - 0.9 * r2))[..., None] * np.array([1.0, 0.95, 0.9], dtype=np.float32)
    ref = ref + 0.01 * rng.standard_normal(ref.shape)
    ref[:, : w // 20] *= 0.1
    ref[5:8, 100:104] = 0.99  # clipped
    return ref.astype(np.float32)


def _lit_numpy(ref):
    lum = ref.mean(axis=2)
    lit = (lum > F._LIT_FRACTION * np.percentile(lum, F._LIT_PERCENTILE)).astype(np.uint8)
    lit = cv2.erode(lit, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if lit.sum() < F._MIN_LIT_FRACTION * lit.size:
        return np.ones(lum.shape, np.float32)
    return lit.astype(np.float32)


def _fill_numpy(values, valid):
    if valid.all() or not valid.any():
        return values
    _, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    source = np.zeros(labels.max() + 1, np.int64)
    source[labels[valid]] = np.flatnonzero(valid)
    return values.reshape(-1, values.shape[2])[source[labels]].reshape(values.shape)


def test_lit_mask_matches_numpy():
    ref = _reference()
    assert F._lit_mask(ref).tobytes() == _lit_numpy(ref).tobytes()
    dark = np.zeros((40, 40, 3), dtype=np.float32)
    assert F._lit_mask(dark).tobytes() == np.ones((40, 40), np.float32).tobytes()


def test_evenness_matches_numpy():
    img = _reference(seed=1)
    lit = _lit_numpy(img).astype(bool)
    median = np.clip(np.median(img[lit], axis=0), 1e-6, None)
    ratio = img / median
    ratio[~lit] = 0.0
    lum = ratio[lit].mean(axis=1)
    low, high = np.percentile(lum - 1.0, [1.0, 99.0])
    chroma = np.percentile(ratio[lit] / np.clip(lum, 1e-6, None)[:, None] - 1.0, [1.0, 99.0], axis=0)
    clipped = float((img[lit] >= F._CLIP_LEVEL).any(axis=1).mean()) > F._CLIP_FRACTION
    got_ratio, got = F._evenness(img)
    assert got_ratio.tobytes() == ratio.tobytes()
    assert got == F.Evenness(float(low), float(high), float(np.abs(chroma).max()), clipped)


def test_compute_gain_matches_numpy():
    ref = F.work_copy(_reference((400, 600), seed=2))
    sigma = max(ref.shape[:2]) / F._BLUR_DIVISOR
    lit = _lit_numpy(ref)
    num = cv2.GaussianBlur(ref * lit[..., None], (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(lit, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blur = np.clip(num / np.clip(den, 1e-4, None)[..., None], 1e-4, None)
    blur = _fill_numpy(blur, den >= F._MIN_LIT_WEIGHT)
    means = (blur * lit[..., None]).sum(axis=(0, 1)) / lit.sum()
    want = np.clip(means[None, None, :] / blur, F._GAIN_MIN, F._GAIN_MAX).astype(np.float32)
    assert F.compute_gain(_reference((400, 600), seed=2)).tobytes() == want.tobytes()


def test_fill_from_nearest_matches_cv2():
    rng = np.random.default_rng(3)
    values = rng.random((50, 70, 3), dtype=np.float32)
    valid = rng.random((50, 70)) > 0.6
    assert F._fill_from_nearest(values, valid).tobytes() == _fill_numpy(values, valid).tobytes()
