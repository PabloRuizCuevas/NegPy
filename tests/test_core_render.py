"""The render-stage helpers against the NumPy code they replace."""

import numpy as np
import pytest

import negpy_core
from negpy.features.exposure.logic import contrast_mask_ev, contrast_mask_scale, expand_mask_plane
from negpy.features.process.capture_color import apply_camera_matrix


def _img(seed, shape=(97, 131, 3)):
    return np.random.default_rng(seed).uniform(-0.2, 1.4, shape).astype(np.float32)


def test_log_density_open_matches_numpy_within_the_ledger():
    img = _img(1)
    img[0, 0] = [np.nan, np.inf, -np.inf]
    want = np.log10(np.clip(np.nan_to_num(img, nan=1e-6, posinf=1.0, neginf=1e-6), 1e-6, None))
    np.testing.assert_allclose(negpy_core.log_density_open(img), want, rtol=4e-7, atol=0)


def test_small_helpers_match_numpy():
    img = _img(2)
    lum = img[..., 1].copy()
    np.testing.assert_array_equal(negpy_core.gray_to_rgb(lum), np.stack([lum, lum, lum], axis=-1))
    np.testing.assert_array_equal(negpy_core.add(lum, img[..., 0].copy()), lum + img[..., 0])
    np.testing.assert_array_equal(negpy_core.clip01(img), np.clip(img, 0, 1))
    zeros = np.zeros((5, 6, 3), np.float32)
    assert not negpy_core.channel_any(zeros, 1)
    zeros[4, 5, 1] = np.nan
    assert negpy_core.channel_any(zeros, 1) == bool(zeros[..., 1].any())
    nan = _img(3)
    nan[1, 2] = [np.nan, np.inf, -np.inf]
    np.testing.assert_array_equal(negpy_core.nan_to_num(nan, 1e-6, 1.0, 1e-6), np.nan_to_num(nan, nan=1e-6, posinf=1.0, neginf=1e-6))


def test_camera_matrix_matches_numpy():
    img = _img(4)
    m = np.random.default_rng(5).uniform(-0.4, 1.3, (3, 3))
    want = (img.reshape(-1, 3).astype(np.float32) @ np.asarray(m, dtype=np.float32).T).reshape(img.shape)
    np.testing.assert_array_equal(apply_camera_matrix(img, m), want)


def _expand_numpy(plane, out_shape, roi):
    h, w = out_shape
    if roi is None:
        if plane.shape[:2] != (h, w):
            plane = negpy_core.resize_linear(np.ascontiguousarray(plane, dtype=np.float32), (w, h))
        return np.ascontiguousarray(plane, dtype=np.float32)
    y1, y2, x1, x2 = roi
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(h, y2), min(w, x2)
    if y2 - y1 < 1 or x2 - x1 < 1:
        return None
    inner = negpy_core.resize_linear(np.ascontiguousarray(plane, dtype=np.float32), (x2 - x1, y2 - y1))
    if (y1, x1, y2, x2) == (0, 0, h, w):
        return inner
    return np.pad(inner, ((y1, h - y2), (x1, w - x2)), mode="edge").astype(np.float32)


@pytest.mark.parametrize("roi", [None, (0, 300, 0, 400), (20, 250, 35, 390), (-5, 320, 10, 500), (100, 100, 0, 10)])
@pytest.mark.parametrize("plane_shape", [(300, 400), (61, 87)])
def test_expand_mask_plane_matches_numpy(roi, plane_shape):
    plane = _img(6, plane_shape)
    want = _expand_numpy(plane, (300, 400), roi)
    got = expand_mask_plane(plane, (300, 400), roi)
    if want is None:
        assert got is None
        return
    np.testing.assert_array_equal(got, want)
    scale = contrast_mask_scale(0.6, 1.8)
    np.testing.assert_array_equal(contrast_mask_ev(plane, 0.6, 1.8, (300, 400), roi), (want * scale).astype(np.float32))
