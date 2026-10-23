"""The IR dust kernels against the cv2 and NumPy code they replace."""

from pathlib import Path

import cv2
import numpy as np
from core_oracle import cv2_oracle

import negpy_core
from negpy.features.retouch import logic as L

pytestmark = cv2_oracle


def _scene(seed=0, shape=(120, 90)):
    rng = np.random.default_rng(seed)
    plane = (0.8 + 0.05 * rng.standard_normal(shape)).astype(np.float32)
    plane[30:34, 20:60] = 0.3  # a hair
    plane[:, :6] = 0.01  # the holder margin
    plane[70:72, 40:42] = 2.0  # glare
    img = rng.random(shape + (3,), dtype=np.float32)
    return plane, img


def test_ir_live_matches_components():
    plane, _ = _scene()
    dead = plane < L._IR_DEAD_FLOOR
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dead.astype(np.uint8), connectivity=8)
    holder = np.zeros(n, dtype=bool)
    holder[1:] = stats[1:, cv2.CC_STAT_AREA] >= L._IR_DEAD_MIN_AREA * dead.size
    np.testing.assert_array_equal(L._ir_live(plane), ~holder[labels])


def test_ir_glare_matches_median_rule():
    plane, _ = _scene(1)
    live = L._ir_live(plane)
    sample = plane.ravel()[L._fit_sample(live)]
    med = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - med)))
    np.testing.assert_array_equal(L._ir_glare(plane, live), plane > med + L._IR_GLARE_SIGMA * sigma)


def test_normalize_ir_matches_cv2():
    plane, _ = _scene(2)
    glare = plane > 1.5
    win = L._ir_win(L._IR_BASE_WIN, L._ir_detect_scale(plane))
    src = np.where(glare, np.float32(0.0), plane)
    base = cv2.blur(cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))), (win, win))
    want = plane / np.maximum(base, 1e-4)
    assert L.normalize_ir(plane, glare).tobytes() == want.tobytes()


def test_ir_clean_base_matches_cv2():
    plane, img = _scene(3)
    ratio = L.normalize_ir(plane)
    win = (L._ir_win(L._IR_CAP_WIN, L._ir_detect_scale(ratio)),) * 2
    w_clean = (ratio >= L._IR_GAIN_IDENTITY).astype(np.float32)
    den = np.maximum(cv2.blur(w_clean, win), 1e-6)[..., None]
    mean = cv2.blur(img * w_clean[..., None], win) / den
    var = cv2.blur(img * img * w_clean[..., None], win) / den - mean * mean
    base = mean - L._IR_CAP_SIGMA * np.sqrt(np.clip(var, 0.0, None))
    dil = cv2.blur(cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, win)), win)
    want = np.where(den > L._IR_CAP_MIN_SUPPORT, base, dil)
    assert L._ir_clean_base(img, ratio).tobytes() == want.tobytes()


def test_ir_normalize_ratio_matches_numpy():
    plane, _ = _scene(4)
    live = L._ir_live(plane)
    ratio = L.normalize_ir(plane)
    sample = ratio[live]
    med = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - med)))
    pivot = float(np.clip(med - L._IR_NOISE_SIGMA * sigma, L._IR_PIVOT_LO, L._IR_GAIN_IDENTITY))
    scale = L._IR_GAIN_IDENTITY / pivot
    want = (ratio * scale).astype(np.float32)
    k = float(np.clip(L._IR_REF_SIGMA / max(sigma * scale, 1e-6), 1.0, L._IR_SCALE_MAX))
    if k > 1.0:
        stretched = L._IR_GAIN_IDENTITY - (L._IR_GAIN_IDENTITY - want) * k
        want = np.where(want < L._IR_GAIN_IDENTITY, stretched, want).astype(np.float32)
    assert L._ir_normalize_ratio(ratio, live).tobytes() == want.tobytes()


def test_ghost_divide_matches_numpy_float32_exp():
    rng = np.random.default_rng(5)
    ratio = rng.random((64, 48), dtype=np.float32)
    vis_log = -3.0 * rng.random((64, 48, 3), dtype=np.float32)
    b = np.array([0.31, 0.07, 0.52], dtype=np.float32)
    want = np.clip(ratio / np.exp((vis_log * b).sum(-1)), 0.0, 1.5).astype(np.float32)
    assert negpy_core.ir_ghost_divide(ratio, vis_log, b.tolist()).tobytes() == want.tobytes()


def test_every_binding_enters_a_pool_through_detach():
    """A thread that ran small work stays in its one-thread pool, so a binding that called
    py.detach directly would run its large work on one core."""
    src = Path(__file__).parents[1] / "negpy_core" / "src" / "py"
    offenders = [p.name for p in src.glob("*.rs") if p.name != "mod.rs" and ".detach(" in p.read_text()]
    assert offenders == []
