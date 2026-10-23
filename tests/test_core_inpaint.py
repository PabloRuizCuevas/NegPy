import math
import pickle
from pathlib import Path

import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.retouch import logic
from negpy.features.retouch.logic import apply_hair_inpaint

pytestmark = cv2_oracle

_RNG = np.random.default_rng(23)
_CORPUS = Path("~/negpy-corpus/features__retouch__logic.py--_hair_encode").expanduser()


def _corpus_crops():
    """Real scan crops from the recorded hair encodes; empty when the corpus is absent."""
    return [pickle.loads(p.read_bytes())["args"][:3] for p in sorted(_CORPUS.glob("*.pkl"))]


def _textured(h, w):
    img = cv2.GaussianBlur(_RNG.random((h, w, 3), dtype=np.float32), (0, 0), 2.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img += (0.3 * np.sin(xx / 9.0) * np.cos(yy / 13.0))[..., None]
    return np.ascontiguousarray((img - img.min()) / (img.max() - img.min()), dtype=np.float32)


def _encode(crop, lo, inv_span):
    enc = np.empty(crop.shape, np.uint8)
    negpy_core.hair_encode(np.ascontiguousarray(crop), np.float32(lo), np.float32(inv_span), np.float32(1 / 2.2), enc)
    return enc


def _strokes(h, w, n, width):
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        pts = np.cumsum(_RNG.normal(0, max(2.0, min(h, w) / 12), (40, 2)), 0) + (_RNG.random(2) * (w, h))
        cv2.polylines(m, [pts.astype(np.int32)], False, 1, width)
    return m


def _masks(h, w):
    return [
        (_RNG.random((h, w)) < 0.15).astype(np.uint8),
        _strokes(h, w, 3, 1),
        _strokes(h, w, 2, 4),
        cv2.dilate(_strokes(h, w, 1, 2), np.ones((3, 3), np.uint8)),
    ]


def _gate(want, got, mask):
    d = np.abs(want.astype(np.int16) - got.astype(np.int16))
    assert np.array_equal(want[mask == 0], got[mask == 0])
    if mask.any():
        inside = d[mask > 0]
        assert inside.mean() <= 1.0 and inside.max() <= 4, (inside.mean(), inside.max())


def _tiles():
    tiles = [_encode(crop, lo, inv) for crop, lo, inv in _corpus_crops()]
    tiles += [_encode(_textured(h, w), 0.0, 1.0) for h, w in ((64, 80), (2, 2), (5, 300), (257, 190))]
    return tiles


@pytest.mark.parametrize("radius", [1, 3, 7])
def test_inpaint_ns_matches_cv2(radius):
    # A one-pixel-wide image is left out: OpenCV reads past the image there.
    for enc in _tiles():
        h, w = enc.shape[:2]
        for mask in _masks(h, w):
            want = cv2.inpaint(enc, mask, radius, cv2.INPAINT_NS)
            _gate(want, negpy_core.inpaint_ns(enc, mask, float(radius)), mask)


def test_inpaint_ns_without_mask_is_identity():
    enc = _encode(_textured(30, 40), 0.0, 1.0)
    assert np.array_equal(negpy_core.inpaint_ns(enc, np.zeros((30, 40), np.uint8), 3.0), enc)


def _hair_inpaint_cv2(img, hair_masks, radius=logic._HAIR_INPAINT_RADIUS, dilate_px=None):
    """`apply_hair_inpaint` as it ran on cv2.inpaint."""
    h, w = img.shape[:2]
    masks = [hm for hm in hair_masks if hm is not None]
    factor = max(1.0, h / masks[0].shape[0], w / masks[0].shape[1])
    if dilate_px is None:
        dilate_px = max(logic._HAIR_DILATE_PX, round(factor))
    m = np.zeros((h, w), dtype=np.uint8)
    for hm in masks:
        r = hm if hm.shape[:2] == (h, w) else negpy_core.resize_linear(np.ascontiguousarray(hm, dtype=np.float32), (w, h))
        m |= (np.asarray(r) > 0.5).astype(np.uint8)
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    src = np.ascontiguousarray(img, dtype=np.float32)
    out = src.copy()
    pad, tile_px, halo = logic._HAIR_INPAINT_PAD, logic._HAIR_TILE_PX, logic._HAIR_TILE_HALO
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n_lbl):
        bx, by, bw, bh = (int(v) for v in stats[i, :4])
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        sub_m = np.ascontiguousarray(m[y0:y1, x0:x1])
        crop = np.ascontiguousarray(src[y0:y1, x0:x1])
        step = max(1, int(math.isqrt(max(1, crop.shape[0] * crop.shape[1] // logic._HAIR_RANGE_SAMPLE_PX))))
        ctx = crop[::step, ::step][sub_m[::step, ::step] == 0]
        lo = float(np.percentile(ctx, 0.5)) if ctx.size else 0.0
        hi = float(np.percentile(ctx, 99.5)) if ctx.size else 1.0
        span = max(hi - lo, 1e-4)
        dec_lut = ((np.arange(256, dtype=np.float32) / 255.0) ** logic._HAIR_INPAINT_GAMMA) * span + lo
        ch, cw = crop.shape[:2]
        for ty0 in range(0, ch, tile_px):
            for tx0 in range(0, cw, tile_px):
                ty1, tx1 = min(ch, ty0 + tile_px), min(cw, tx0 + tile_px)
                if not (labels[y0 + ty0 : y0 + ty1, x0 + tx0 : x0 + tx1] == i).any():
                    continue
                ay0, ay1 = max(0, ty0 - halo), min(ch, ty1 + halo)
                ax0, ax1 = max(0, tx0 - halo), min(cw, tx1 + halo)
                tile = np.ascontiguousarray(crop[ay0:ay1, ax0:ax1])
                enc = _encode(tile, lo, 1.0 / span)
                filled = cv2.inpaint(enc, np.ascontiguousarray(sub_m[ay0:ay1, ax0:ax1]), radius, cv2.INPAINT_NS)
                mb = labels[y0 + ay0 : y0 + ay1, x0 + ax0 : x0 + ax1] == i
                d = cv2.distanceTransform(mb.astype(np.uint8), cv2.DIST_C, 3)
                core = np.zeros(mb.shape, dtype=bool)
                core[ty0 - ay0 : ty1 - ay0, tx0 - ax0 : tx1 - ax0] = True
                sel = np.flatnonzero(mb & core)
                a = np.minimum(d.ravel()[sel] / float(dilate_px + 1), 1.0)[:, None]
                dec = dec_lut[filled.reshape(-1, 3)[sel]]
                tile_px_v = tile.reshape(-1, 3)[sel]
                ry, rx = np.divmod(sel, ax1 - ax0)
                out.reshape(-1, 3)[(y0 + ay0 + ry) * w + (x0 + ax0 + rx)] = tile_px_v * (1.0 - a) + dec * a
    negpy_core.add_clean_grain(out, src, m == 0, max(1.0, factor))
    return out, m


def _frames():
    frames = [crop for crop, _, _ in _corpus_crops() if min(crop.shape[:2]) >= 8]
    return frames + [_textured(220, 340), _textured(90, 2100)]


def test_hair_inpaint_matches_cv2_version(monkeypatch):
    # A small tile makes the larger frames run the tiled path with its halo.
    monkeypatch.setattr(logic, "_HAIR_TILE_PX", 96)
    worst = 0.0
    for img in _frames():
        h, w = img.shape[:2]
        for hair in (_strokes(h, w, 2, 1), _strokes(h // 2, w // 2, 3, 2)):
            want, m = _hair_inpaint_cv2(img, [hair])
            got = np.asarray(apply_hair_inpaint(img, [hair]))
            assert np.array_equal(want[m == 0], got[m == 0])
            # One 8-bit step of the encode, on the crop's range.
            span = max(float(img.max() - img.min()), 1e-4)
            worst = max(worst, float(np.abs(want - got).max()) / span)
    assert worst <= 4 / 255, worst
