"""The batch autocrop profile reads and the occupancy window against the NumPy code they replace."""

import numpy as np
import negpy_core
import pytest

from negpy.features.geometry.batch_autocrop import CropEvidence, RollCropTemplate, _rect_from_edge_profile, _top_edge_slope
from negpy.features.geometry.logic import _place_window_by_occupancy


def _top_edge_slope_numpy(lum, roi):
    y1, _, x1, x2 = roi
    height = lum.shape[0]
    left, right = x1 + 20, x2 - 20
    columns = np.arange(left, right, 3)
    min_points = max(20, round((right - left) * 0.08))
    if columns.size < min_points:
        return None
    radius = max(12, round(height * 0.035))
    low, high = max(0, y1 - radius), min(height, y1 + radius)
    if high - low < 4:
        return None
    gradient = np.abs(negpy_core.sobel(negpy_core.gaussian_blur(lum.astype(np.float32), (0, 0), 1.5), 0, 1))
    band = gradient[low:high, columns]
    rows = low + np.argmax(band, axis=0)
    strength = band.max(axis=0)
    if float(np.median(strength)) < max(1e-6, 3.0 * float(np.median(band))):
        return None
    points = np.column_stack([columns, rows])[strength >= np.quantile(strength, 0.60)]
    if len(points) < min_points:
        return None
    residuals = np.zeros(len(points))
    slope = 0.0
    for _ in range(5):
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        residuals = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        keep = residuals <= max(2.0, float(np.median(residuals)) * 2.5)
        if keep.all() or keep.sum() < min_points:
            break
        points = points[keep]
    if float(np.median(residuals)) > 2.0:
        return None
    return float(np.degrees(np.arctan(slope)))


def _tilted_box(tilt, seed, size=(400, 600)):
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    top = 100 + np.tan(np.radians(tilt)) * (x - w / 2)
    lum = np.where(y >= top, 0.8, 0.1) + np.random.default_rng(seed).normal(0, 0.02, (h, w))
    return lum.astype(np.float32)


@pytest.mark.parametrize("tilt", [-2.0, -0.7, 0.0, 0.3, 1.6])
def test_top_edge_slope_matches_numpy(tilt):
    for seed in range(4):
        lum = _tilted_box(tilt, seed)
        for roi in ((100, 300, 50, 550), (8, 300, 0, 600), (100, 300, 300, 330)):
            want, got = _top_edge_slope_numpy(lum, roi), _top_edge_slope(lum, roi)
            assert (want is None) == (got is None)
            if want is not None:
                assert got == pytest.approx(want, abs=1e-9)


def _peak_near(profile, expected, radius):
    if profile.size == 0:
        return expected, 0.0
    center = int(round(expected * (profile.size - 1)))
    half = max(2, int(round(radius * profile.size)))
    lo, hi = max(0, center - half), min(profile.size, center + half + 1)
    if hi <= lo:
        return expected, 0.0
    local = profile[lo:hi]
    offset = int(np.argmax(local))
    return (lo + offset) / max(profile.size - 1, 1), float(local[offset])


def _clamp(rect):
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
    y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
    return x1, y1, x2, y2


def _rect_numpy(profile, contrast, t):
    if contrast < 0.06 or profile.size < 8 or not np.any(np.isfinite(profile)):
        return None
    base, peak = float(np.percentile(profile, 50)), float(np.percentile(profile, 99.5))
    if peak - base < 0.20 or peak < 1.35 * max(base, 0.05):
        return None
    radius = max(0.045, 4.0 * t.center_mad + 2.0 * t.width_mad)
    left, ls = _peak_near(profile, t.center_x - t.width * 0.5, radius)
    right, rs = _peak_near(profile, t.center_x + t.width * 0.5, radius)
    baseline = float(np.percentile(profile, 70))
    spread = float(np.percentile(profile, 90) - np.percentile(profile, 50))
    threshold = max(0.24, baseline + 0.55 * spread)
    lok, rok = ls >= threshold, rs >= threshold
    if not (lok and rok) and abs((right - left) - t.width) <= 0.02:
        lok = rok = True
    if not lok and not rok:
        return None
    if lok and rok:
        if right - left < 0.65 * t.width or right - left > 1.35 * t.fallback_width:
            return None
        x1, x2 = left, right
    elif lok and ls >= threshold + 0.12:
        x1, x2 = left, left + t.fallback_width
    elif rok and rs >= threshold + 0.12:
        x1, x2 = right - t.fallback_width, right
    else:
        return None
    return _clamp((x1, t.top, x2, t.top + t.height))


def test_rect_from_edge_profile_matches_numpy():
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(3000):
        n = int(rng.integers(8, 400))
        profile = (rng.random(n) * rng.uniform(0.05, 0.6)).astype(np.float32)
        t = RollCropTemplate(
            width=float(rng.uniform(0.5, 0.95)), fallback_width=float(rng.uniform(0.5, 1.0)), height=0.8, center_x=float(rng.uniform(0.4, 0.6)),
            top=0.1, correction_angle=0.0, width_mad=float(rng.uniform(0, 0.02)), center_mad=float(rng.uniform(0, 0.02)), top_mad=0.0,
            angle_mad=0.0, confidence=0.7, sample_count=5,
        )  # fmt: skip
        for edge in (t.center_x - t.width / 2, t.center_x + t.width / 2):
            if rng.random() < 0.8:
                profile[int(np.clip(round((edge + rng.normal(0, 0.01)) * (n - 1)), 0, n - 1))] = rng.uniform(0.5, 2.0)
        if rng.random() < 0.05:
            profile[int(rng.integers(n))] = np.nan
        item = CropEvidence(
            "k", (100, 150), None, 0.0, 0.0, vertical_edge_contrast=float(rng.uniform(0, 0.3)), vertical_edge_profile=profile
        )
        want = _rect_numpy(profile, item.vertical_edge_contrast, t)
        assert _rect_from_edge_profile(item, t) == want
        hits += want is not None
    assert hits > 100


def _window_numpy(start, end, target_len, occupancy, scale):
    d_start = max(0, int(round(start * scale)))
    d_end = min(occupancy.size, int(round(end * scale)))
    d_len = max(1, int(round(target_len * scale)))
    if d_len >= d_end - d_start:
        return start
    cs = np.concatenate(([0.0], np.cumsum(occupancy[d_start:d_end], dtype=np.float64)))
    n_pos = (d_end - d_start) - d_len + 1
    scores = cs[d_len : d_len + n_pos] - cs[:n_pos]
    candidates = np.where(scores >= float(np.max(scores)) - 1e-9)[0]
    centered = ((d_end - d_start) - d_len) / 2.0
    k = int(candidates[np.argmin(np.abs(candidates - centered))])
    return int(min(max(start, round((d_start + k) / scale)), end - target_len))


def test_place_window_by_occupancy_matches_numpy():
    rng = np.random.default_rng(3)
    for _ in range(5000):
        n = int(rng.integers(5, 300))
        occ = (rng.random(n) > rng.random()).astype(np.float32) * rng.choice([1.0, 0.5], n).astype(np.float32)
        scale = float(rng.choice([1.0, 0.5, 0.37, 0.25]))
        end = int(rng.integers(2, int(n / scale) + 2))
        start = int(rng.integers(0, end))
        target = float(rng.uniform(1, end - start + 1)) if rng.random() < 0.5 else float(rng.integers(1, end - start + 1))
        assert _place_window_by_occupancy(start, end, target, occ, scale) == _window_numpy(start, end, target, occ, scale)
