"""The embedded lens maps against the NumPy and cv2 code they replace."""

import cv2
import numpy as np
from core_oracle import cv2_oracle

from negpy.features.lens.models import LensCorrections, LensMetadata
from negpy.features.lens.warps import PanasonicWarp, RectilinearWarp, SonyWarp

pytestmark = cv2_oracle

_BARREL = (1.0, -0.06, 0.012, -0.002, 0.0004, -0.0003)
_RED = (1.0006, -0.0598, 0.0119, -0.002, 0.0004, -0.0003)
_BLUE = (0.9993, -0.0603, 0.0121, -0.002, 0.0004, -0.0003)
_SHAPE = (120, 170, 3)


def _rectilinear_xy(x, y, coefficients):
    k0, k1, k2, k3, t0, t1 = coefficients
    r2 = x * x + y * y
    factor = k0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return x * factor + 2 * t0 * x * y + t1 * (r2 + 2 * x * x), y * factor + 2 * t1 * x * y + t0 * (r2 + 2 * y * y)


def _coordinates(lens, shape, start, stop, center):
    h, w = shape[:2]
    t, left, b, r = lens.active_area or (0, 0, h, w)
    bt, bl, bb, br = lens.buffer_area or (t, left, b, r)
    sx, sy = (br - bl) / w, (bb - bt) / h
    cx, cy = left + center[0] * (r - left - 1), t + center[1] * (b - t - 1)
    radius = np.hypot(max(cx - left, r - 1 - cx), max(cy - t, b - 1 - cy))
    x = (bl + (np.arange(w, dtype=np.float32)[None, :] + 0.5) * sx - 0.5 - cx) / radius * lens.fill_scale
    y = (bt + (np.arange(start, stop, dtype=np.float32)[:, None] + 0.5) * sy - 0.5 - cy) / radius * lens.fill_scale
    return x, y, cx, cy, radius, sx, sy, bl, bt


def _rectilinear_numpy(warp, lens, start, stop, channel, corrections):
    x, y, cx, cy, radius, sx, sy, left, top = _coordinates(lens, _SHAPE, start, stop, warp.center)
    common = warp.coefficients[0 if len(warp.coefficients) == 1 else 1]
    selected = warp.coefficients[channel] if corrections.ca and len(warp.coefficients) == 3 else common
    target_x, target_y = x, y
    if not corrections.distortion:
        g0, g1, g2, g3, t0, t1 = common
        points = np.stack(np.broadcast_arrays(x, y), axis=-1)
        unwarped = cv2.undistortPointsIter(
            points.reshape(-1, 1, 2),
            np.diag([g0, g0, 1.0]),
            np.array([g1, g2, t0, t1, g3]) / g0,
            None,
            None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-8),
        ).reshape(points.shape)
        x, y = unwarped[..., 0], unwarped[..., 1]
        limit = np.maximum(np.hypot(x, y), 1.0)
        x, y = x / limit, y / limit
    wx, wy = _rectilinear_xy(x, y, selected)
    if not corrections.distortion:
        gx, gy = _rectilinear_xy(x, y, common)
        scale = np.hypot(target_x, target_y) / np.maximum(np.hypot(gx, gy), 1e-8)
        wx, wy = target_x + (wx - gx) * scale, target_y + (wy - gy) * scale
    mx = (cx + radius * wx - left + 0.5) / sx - 0.5
    my = (cy + radius * wy - top + 0.5) / sy - 0.5
    return mx.astype(np.float32), my.astype(np.float32)


def test_rectilinear_maps_match_numpy_and_cv2():
    cases = [
        (RectilinearWarp((_BARREL,)), LensMetadata("DNG", fill_scale=0.93), LensCorrections(True, True), 0),
        (
            RectilinearWarp((_RED, _BARREL, _BLUE), (0.51, 0.49)),
            LensMetadata("DNG", active_area=(3, 5, 118, 166)),
            LensCorrections(True, True),
            2,
        ),
        (RectilinearWarp((_RED, _BARREL, _BLUE)), LensMetadata("DNG"), LensCorrections(False, True), 0),
    ]
    for warp, lens, corrections, channel in cases:
        got = warp.remap(lens, _SHAPE, 10, 50, channel, corrections)
        want = _rectilinear_numpy(warp, lens, 10, 50, channel, corrections)
        for g, wv in zip(got, want):
            assert g.tobytes() == wv.tobytes()


def test_panasonic_and_sony_maps_match_numpy():
    h, w = _SHAPE[:2]
    lens = LensMetadata("x", fill_scale=0.97)
    x = (np.arange(w, dtype=np.float32)[None, :] - w * 0.5) * lens.fill_scale
    y = (np.arange(5, 40, dtype=np.float32)[:, None] - h * 0.5) * lens.fill_scale
    radius = np.hypot(x, y) / np.hypot(w * 0.5, h * 0.5)

    pana = PanasonicWarp(0.9, 0.04, -0.01, 0.002)
    rd = radius
    for _ in range(8):
        r2 = rd * rd
        f = 1.0 + pana.scale * (pana.a * r2 + pana.b * r2 * r2 + pana.c * r2 * r2 * r2)
        f = np.where(f > 0.0, f, 1.0)
        new = radius / f
        if np.abs(new - rd).max() < 1e-7:
            break
        rd = new
    dr = np.divide(rd, radius, out=np.ones_like(radius), where=radius > 0.0)
    want = ((x * dr + w * 0.5).astype(np.float32), (y * dr + h * 0.5).astype(np.float32))
    got = pana.remap(lens, _SHAPE, 5, 40, 1)
    assert all(g.tobytes() == wv.tobytes() for g, wv in zip(got, want))

    knots = tuple(float(v) for v in range(-60, 100, 10))
    sony = SonyWarp(knots, tuple(float(v) for v in range(16)), ())
    n = len(knots)
    factors = (np.ones(n) + np.asarray(knots) / 16384.0) * (1 + np.arange(16.0) / 2097152.0)
    factor = np.interp(radius, (np.arange(n) + 0.5) / (n - 1), factors).astype(np.float32)
    want = ((x * factor + w * 0.5).astype(np.float32), (y * factor + h * 0.5).astype(np.float32))
    got = sony.remap(lens, _SHAPE, 5, 40, 0)
    assert all(g.tobytes() == wv.tobytes() for g, wv in zip(got, want))
