"""The preview and capture helpers against the NumPy code they replace."""

import numpy as np
import pytest

import negpy_core
from negpy.desktop.view.widgets.prescan_dialog import _preview_u8
from negpy.desktop.view.widgets.scan_preview_common import preview_positive
from negpy.infrastructure.scanners.params import FilmType
from negpy.services.capture.focus_meter import MIN_MEAN_LEVEL, sharpness
from negpy.services.scanning.writer import _to_grey


def _scan(seed, shape=(151, 203, 3), dtype=np.uint16):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.05, 0.9, shape) * np.iinfo(dtype).max).astype(dtype)


def _preview_positive_numpy(rgb, positive):
    a = rgb.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        ch = a[..., c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        if hi <= lo:
            out[..., c] = 0.0
            continue
        scaled = (ch - lo) if positive else (hi - ch)
        out[..., c] = np.clip(scaled / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def _preview_u8_numpy(rgb):
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    out = np.empty(a.shape[:2] + (3,), dtype=np.uint8)
    for c in range(3):
        ch = a[..., min(c, a.shape[2] - 1)]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        out[..., c] = 0 if hi <= lo else np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


@pytest.mark.parametrize("shape", [(151, 203, 3), (40, 50), (30, 20, 4)])
def test_preview_stretches_match_numpy(shape):
    rgb = _scan(1, shape)
    if len(shape) == 3:
        rgb[..., -1] = 7  # a flat channel
    np.testing.assert_array_equal(preview_positive(rgb, "negative"), _preview_positive_numpy(rgb, False))
    np.testing.assert_array_equal(preview_positive(rgb, FilmType.POSITIVE.value), _preview_positive_numpy(rgb, True))
    np.testing.assert_array_equal(_preview_u8(rgb), _preview_u8_numpy(rgb))


def test_sharpness_matches_numpy():
    img = _scan(2, (120, 160), np.uint8).astype(np.float32)
    mean = float(img.mean())
    lap = 4.0 * img[1:-1, 1:-1] - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]
    assert sharpness(img) == float(np.mean(lap * lap)) / (mean * mean)
    assert sharpness(np.full((10, 10), MIN_MEAN_LEVEL / 2, np.float32)) == 0.0


def test_level_stats_match_numpy():
    for patch in (_scan(3, (50, 70), np.uint16), _scan(4, (33, 41), np.uint16).astype(np.float32) / 65535):
        level = 60000 if patch.dtype == np.uint16 else 0.9
        frac, p = negpy_core.patch_level_stats(np.ascontiguousarray(patch), float(level), 99.9)
        assert frac == float(np.mean(patch >= level))
        assert p == float(np.percentile(patch, 99.9))
    img = _scan(5, (40, 60, 4)).astype(np.float32) / 65535
    assert negpy_core.any_channel_at_least(img, 0.85) == float(np.mean(np.any(img[:, :, :3] >= 0.85, axis=2)))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_to_grey_matches_numpy(dtype):
    rgb = _scan(6, (31, 47, 3), dtype)
    np.testing.assert_array_equal(_to_grey(rgb), ((np.sum(rgb, axis=-1, dtype=np.uint32) + 1) // 3).astype(dtype))


def test_level_stats_of_float64_match_numpy():
    patch = np.random.default_rng(7).uniform(0, 65535, (37, 29))
    frac, p = negpy_core.patch_level_stats(patch, 60000.0, 99.9)
    assert (frac, p) == (float(np.mean(patch >= 60000.0)), float(np.percentile(patch, 99.9)))
