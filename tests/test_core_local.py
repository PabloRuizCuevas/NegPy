"""cv2.fillPoly and the local-mask rasteriser against the cv2 and NumPy code they replace."""

import cv2
import numpy as np
from core_oracle import cv2_oracle

import negpy_core
from negpy.features.local.logic import outline_points, rasterise
from negpy.features.local.models import MaskShape

pytestmark = cv2_oracle


def test_fill_poly_matches_cv2_on_random_polygons():
    rng = np.random.default_rng(0)
    for _ in range(500):
        h, w = (int(v) for v in rng.integers(5, 90, 2))
        n = int(rng.integers(3, 30))
        pts = np.stack([rng.integers(-20, w + 20, n), rng.integers(-20, h + 20, n)], 1).astype(np.int32)
        want = np.zeros((h, w), np.uint8)
        cv2.fillPoly(want, [pts], 255)
        np.testing.assert_array_equal(negpy_core.fill_poly((h, w), pts), want)


def _rasterise_cv2(shape, pts, h, w, sigma, invert):
    if shape == MaskShape.GRADIENT:
        (a, b) = pts
        ax, ay = a[0] * w, a[1] * h
        dx, dy = b[0] * w - ax, b[1] * h - ay
        denom = dx * dx + dy * dy
        xs = (np.arange(w, dtype=np.float32) - ax) * (dx / denom)
        ys = (np.arange(h, dtype=np.float32) - ay) * (dy / denom)
        t = np.clip(xs[None, :] + ys[:, None], 0.0, 1.0)
        alpha = (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)
    else:
        outline = np.array([[x * w, y * h] for x, y in outline_points(shape, pts)], dtype=np.float32)
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(filled, [outline.astype(np.int32)], 255)
        alpha = filled.astype(np.float32) / 255.0
        if sigma > 1e-3:
            k = int(sigma * 3) | 1
            alpha = cv2.GaussianBlur(alpha, (k, k), sigma)
    return 1.0 - alpha if invert else alpha


def test_rasterise_matches_cv2():
    cases = [
        (MaskShape.POLYGON, ((0.2, 0.2), (0.5, 0.15), (0.6, 0.45), (0.25, 0.5)), 6.0, False),
        (MaskShape.OVAL, ((0.6, 0.6), (0.85, 0.6), (0.6, 0.8)), 3.5, True),
        (MaskShape.GRADIENT, ((0.5, 0.0), (0.4, 0.7)), 0.0, False),
        (MaskShape.POLYGON, ((0.1, 0.9), (0.9, 0.1), (0.9, 0.9), (0.1, 0.1)), 0.0, False),
    ]
    for shape, pts, sigma, invert in cases:
        want = _rasterise_cv2(shape, pts, 150, 210, sigma, invert)
        assert rasterise(shape, pts, 150, 210, sigma, invert).tobytes() == want.tobytes(), shape


def test_stroke_cover_matches_cv2_on_random_strokes():
    rng = np.random.default_rng(1)
    for _ in range(800):
        h, w = (int(v) for v in rng.integers(4, 90, 2))
        n = int(rng.integers(1, 7))
        pts = np.stack([rng.integers(-30, w + 30, n), rng.integers(-30, h + 30, n)], 1).astype(np.int32)
        radius, thickness = int(rng.integers(0, 25)), int(rng.integers(1, 50))
        want = np.zeros((h, w), np.uint8)
        if n > 1:
            cv2.polylines(want, [pts], False, 1, thickness=thickness)
        for cx, cy in pts:
            cv2.circle(want, (int(cx), int(cy)), radius, 1, -1)
        np.testing.assert_array_equal(negpy_core.stroke_cover((h, w), [(pts, radius, thickness)]), want)
