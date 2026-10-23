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


def test_meter_picture_matches_numpy():
    rng = np.random.default_rng(8)
    linear = np.clip(rng.uniform(0.2, 1.05, (90, 120, 3)).astype(np.float32), 0, 1)
    sat, roi = np.float32(0.998), (10, 80, 15, 110)
    picture = linear[10:80, 15:110].reshape(-1, 3)
    saturated = picture.max(axis=1) >= sat
    film = picture if saturated.mean() >= 0.02 or saturated.all() else picture[~saturated]
    clipped, blown, thin = negpy_core.meter_picture(linear, roi, float(sat), 0.02)
    assert clipped == float(np.mean(saturated)) and blown == bool(saturated.mean() >= 0.02)
    np.testing.assert_array_equal(np.array(thin, np.float32), np.percentile(film, 99.9, axis=0))


def _plateau_clip_fraction(values, guard, R):
    top = int(values.max())
    if guard > 0 and top * 2 < guard:
        return 0.0
    lo = max(0, top - R._PLATEAU_WINDOW)
    counts = np.bincount(values[values >= lo] - lo, minlength=R._PLATEAU_WINDOW + 1)
    populated = np.flatnonzero(counts)
    peak_at = int(np.argmax(counts[populated]))
    peak_bin = int(populated[peak_at])
    peak = int(counts[peak_bin])
    if peak < R._MIN_PLATEAU_SITES:
        return 0.0
    neighbors = counts[populated[peak_at + 1 : peak_at + 1 + R._PLATEAU_NEIGHBOR_BINS]]
    density = float(neighbors.max()) if neighbors.size else 0.0
    if peak < R._PLATEAU_DENSITY_RATIO * max(density, 1.0):
        return 0.0
    return float(np.count_nonzero(values >= lo + peak_bin)) / values.size


@pytest.mark.parametrize("plateau", [None, 15500, 15790])
def test_raw_clip_fractions_match_numpy(plateau):
    from negpy.infrastructure.capture import raw_demosaic as R

    rng = np.random.default_rng(9)
    img = rng.integers(14000, 15800, (60, 80)).astype(np.uint16)
    if plateau is not None:
        img[rng.uniform(0, 1, img.shape) < 0.3] = plateau
    colors = np.tile(np.array([[0, 1], [3, 2]], np.uint8), (30, 40))
    values = img[np.isin(colors, [1, 3])]
    want = (float(np.mean(values >= 15000)), _plateau_clip_fraction(values, 16383, R))
    got = negpy_core.raw_clip_fractions(
        img, colors, [1, 3], 15000, 16383, R._PLATEAU_WINDOW, R._MIN_PLATEAU_SITES, R._PLATEAU_NEIGHBOR_BINS, R._PLATEAU_DENSITY_RATIO
    )
    assert got == want


def test_column_means_ring_medians_and_dmax_match_numpy():
    rng = np.random.default_rng(10)
    a = rng.uniform(0, 1, (700, 300)).astype(np.float32)
    peak = float(a.max())
    np.testing.assert_array_equal(negpy_core.column_means(a, 2, peak), (a[::2] / peak).mean(axis=0))
    arr = rng.uniform(0, 0.2, (256, 190)).astype(np.float32)
    m, (h, w) = 10, arr.shape
    ring = np.concatenate([arr[:m].ravel(), arr[-m:].ravel(), arr[:, :m].ravel(), arr[:, -m:].ravel()])
    inner = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    assert negpy_core.ring_inner_medians(arr, m) == (np.median(ring), np.median(inner))
    px = rng.integers(0, 256, (80, 60, 3)).astype(np.uint8)
    pixels = px.reshape(-1, 3).astype(np.float32)
    luma = pixels @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    darkest = pixels[luma <= np.quantile(luma, 0.01)]
    assert negpy_core.print_dmax(px, 0.01) == tuple(int(round(v)) for v in darkest.mean(axis=0))


def _srgb_numpy(img):
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


def test_srgb_to_linear_stays_within_an_ulp():
    from negpy.kernel.image.logic import srgb_to_linear

    img = np.linspace(0, 1, 70001, dtype=np.float32)
    base = ((img + np.float32(0.055)) / np.float32(1.055)).astype(np.float64)
    exact = np.where(img <= 0.04045, img / np.float32(12.92), (base**2.4).astype(np.float32)).astype(np.float32)
    got = srgb_to_linear(img)
    assert np.max(np.abs(got.view(np.int32) - exact.view(np.int32))) <= 1
    assert np.max(np.abs(got.view(np.int32) - _srgb_numpy(img).view(np.int32))) <= 4


def test_thumbnail_log_stretch_matches_numpy_within_a_level():
    rng = np.random.default_rng(12)
    linear = _srgb_numpy((rng.integers(0, 256, (90, 130, 3)) / 255).astype(np.float32))
    linear[..., 2] *= 0.01  # a near-empty channel borrows the strongest
    peaks = np.percentile(linear, 99.0, axis=(0, 1))
    strongest = int(np.argmax(peaks))
    empty = peaks < peaks[strongest] * 0.05
    work = linear.copy()
    work[:, :, empty] = work[:, :, strongest : strongest + 1]
    density = -np.log10(np.clip(work, 1e-4, None))
    lo, hi = np.percentile(density, 1.0, axis=(0, 1)), np.percentile(density, 99.0, axis=(0, 1))
    positive = np.clip((density - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    want = (positive.astype(np.float64) * 255 + 0.5).astype(np.uint8)
    got = negpy_core.log_stretch_u8(linear)
    assert np.max(np.abs(got.astype(int) - want)) <= 1
    assert np.mean(got != want) < 0.01
