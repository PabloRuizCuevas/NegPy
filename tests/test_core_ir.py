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


def _route_cv2(score):
    at_floor = (score <= L._IR_SCORE_FLOOR + 1e-6).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(at_floor, connectivity=8)
    scale = L._ir_detect_scale(score)
    radius = int(round(L._IR_ROUTE_RADIUS * scale))
    routed = np.zeros_like(at_floor)
    for i in range(1, n):
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if min(bw, bh) < 2 * radius - 1:
            continue
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        own = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        if float(cv2.distanceTransform(np.pad(own.astype(np.uint8), 1), cv2.DIST_C, 3).max()) >= radius:
            routed[y0 : y0 + bh, x0 : x0 + bw][own] = 1
    if not routed.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(L._IR_ROUTE_DILATE * scale)) + 1,) * 2)
    return cv2.dilate(routed, k)


def test_route_wide_defects_matches_cv2():
    score = np.ones((200, 160), dtype=np.float32)
    score[20:40, 30:60] = L._IR_SCORE_FLOOR  # wide: routed
    score[100:103, 10:150] = L._IR_SCORE_FLOOR  # a thin line: too narrow
    score[150:170, 100:104] = L._IR_SCORE_FLOOR
    want = _route_cv2(score)
    assert want is not None
    np.testing.assert_array_equal(L.route_wide_defects(score, budget=None), want)
    assert L.route_wide_defects(np.ones((50, 50), dtype=np.float32)) is None


def test_ir_defect_score_matches_cv2():
    plane, _ = _scene(6)
    ratio = L.normalize_ir(plane)
    cutoff = 0.6
    span = max(L._IR_GAIN_IDENTITY - cutoff, 1e-4)
    t = (ratio - cutoff) / span
    score = np.clip(t * (1.0 - L._IR_SCORE_FLOOR) + L._IR_SCORE_FLOOR, L._IR_SCORE_FLOOR, 1.0)
    want = cv2.erode(score, np.ones((3, 3), np.uint8))
    assert L.ir_defect_score(ratio, cutoff).tobytes() == want.tobytes()


def test_fit_refraction_gammas_matches_numpy():
    rng = np.random.default_rng(7)
    ratio = rng.uniform(0.6, 1.0, (160, 120)).astype(np.float32)
    img = rng.random((160, 120, 3), dtype=np.float32)
    vis_log = (np.log(ratio)[..., None] * np.float32([1.1, 1.3, 1.6])).astype(np.float32)
    vis_log += 0.01 * rng.standard_normal(vis_log.shape).astype(np.float32)
    band = (ratio > 0.70) & (ratio < 0.92)
    cv2.ipp.setUseIPP(False)
    try:
        edge = np.abs(cv2.Laplacian(img[:, :, 1], cv2.CV_32F, ksize=5))
    finally:
        cv2.ipp.setUseIPP(True)
    flat = band & (edge < np.percentile(edge[band], L._IR_FIT_FLAT_PCT))
    fit = L._fit_sample(flat if int(flat.sum()) >= L._IR_FIT_MIN_PX else band)
    xb = np.log(ratio.ravel()[fit])
    vl = vis_log.reshape(-1, 3)[fit]
    want = tuple(float(np.clip(np.median(vl[:, c] / xb), L._IR_GAMMA_LO, L._IR_GAMMA_HI)) for c in range(3))
    assert L._fit_refraction_gammas(ratio, vis_log, img) == want


def _dust_stats_cv2(img, dust_size):
    dens = L._density(img)
    lo, hi = np.percentile(dens, (0.5, 99.5))
    lo, spread = float(lo), max(float(hi - lo), L._PROXY_MIN_SPREAD)
    proxy = np.clip((dens - lo) / spread, 0.0, 1.0).astype(np.float32)
    base_size = max(1.0, float(dust_size)) * L.film_scale(proxy.shape)
    v_win = int(max(3, base_size * 3.0)) * 2 + 1
    w_win = L._texture_win(proxy.shape, dust_size)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(base_size)) + 1,) * 2)
    background = np.maximum(cv2.medianBlur(L._u8(proxy), v_win).astype(np.float32) / 255.0, cv2.morphologyEx(proxy, cv2.MORPH_OPEN, disk))
    excess = cv2.blur(proxy - background, (L._DETECT_AVG_PX, L._DETECT_AVG_PX))
    mad = cv2.medianBlur(L._u8(np.abs(excess) * L._DETECT_MAD_GAIN), w_win).astype(np.float32) / (255.0 * L._DETECT_MAD_GAIN)
    z = excess / np.maximum(mad / 0.6745, L._DETECT_SIGMA_MIN)
    mean = cv2.blur(proxy, (w_win, w_win))
    texture = np.sqrt(np.clip(cv2.blur(proxy * proxy, (w_win, w_win)) - mean * mean, 0.0, None))
    return proxy, background, z, texture


def test_compute_dust_stats_matches_cv2():
    rng = np.random.default_rng(8)
    img = (0.3 + 0.2 * rng.random((140, 180, 3))).astype(np.float32)
    img[40:43, 50:53] *= 0.2  # a speck
    for got, want in zip(L.compute_dust_stats(img, 2), _dust_stats_cv2(img, 2)):
        assert got.tobytes() == want.tobytes()


def test_split_hairs_matches_components():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:12, 5:70] = 1  # a hair
    mask[40:45, 40:45] = 1  # a speck
    compact, hairs = L.split_hairs(mask)
    want_hair = np.zeros_like(mask)
    want_hair[10:12, 5:70] = 1
    np.testing.assert_array_equal(hairs, want_hair)
    np.testing.assert_array_equal(compact, mask - want_hair)
    assert L.split_hairs(mask - want_hair)[1] is None
