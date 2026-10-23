import functools
import pickle
from pathlib import Path

import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.stitch import logic

pytestmark = cv2_oracle

_RNG = np.random.default_rng(31)
_CORPUS = Path("~/negpy-corpus-p5/features__stitch__logic.py--seam_blend").expanduser()


@functools.cache
def _recorded():
    """(warped, masks, seams, feather) of the recorded panorama blends; empty without the corpus."""
    return [pickle.loads(p.read_bytes())["args"] for p in sorted(_CORPUS.glob("*.pkl"))]


def _parts(h, w, n, shift, angle):
    """`n` noisy, slightly rotated views of one scene, each covering a shifted window."""
    wide = w + (n - 1) * shift
    scene = cv2.GaussianBlur(_RNG.random((h, wide, 3), dtype=np.float32), (0, 0), 4)
    imgs, masks = [], []
    for i in range(n):
        img = np.zeros((h, wide, 3), np.float32)
        x0 = i * shift
        img[:, x0 : x0 + w] = scene[:, x0 : x0 + w] + _RNG.normal(0, 0.01, (h, w, 3)).astype(np.float32)
        m = np.zeros((h, wide), np.uint8)
        m[:, x0 : x0 + w] = 1
        rot = cv2.getRotationMatrix2D((wide / 2, h / 2), float(_RNG.normal(0, angle)), 1.0)
        imgs.append(np.ascontiguousarray(cv2.warpAffine(img, rot, (wide, h))))
        masks.append(np.ascontiguousarray(cv2.warpAffine(m, rot, (wide, h), flags=cv2.INTER_NEAREST)))
    return imgs, masks


def _cut_cost(imgs, cut):
    """DpSeamFinder's color cost summed over every step where the owning part changes."""
    owner = np.full(cut[0].shape, -1)
    for i, c in enumerate(cut):
        owner[(owner < 0) & (c > 0)] = i
    stack = np.stack(imgs)
    total = 0.0
    for a, b in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        oa, ob = owner[a], owner[b]
        ys, xs = np.nonzero((oa >= 0) & (ob >= 0) & (oa != ob))
        ia, ib = oa[ys, xs], ob[ys, xs]
        pa = stack[:, a[0], a[1]][:, ys, xs]
        pb = stack[:, b[0], b[1]][:, ys, xs]
        n = np.arange(len(ys))
        d1 = ((pa[ia, n] - pb[ib, n]) ** 2).sum(-1)
        d2 = ((pb[ia, n] - pa[ib, n]) ** 2).sum(-1)
        total += float(((d1 + d2) / 2).sum())
    return total


def _check_seams(imgs, masks):
    want = [cv2.UMat.get(m) for m in cv2.detail_DpSeamFinder("COLOR").find(imgs, [(0, 0)] * len(imgs), [cv2.UMat(m.copy()) for m in masks])]
    got = negpy_core.dp_seams(imgs, masks)
    overlap = np.sum([m > 0 for m in masks], 0) >= 2
    for w, g in zip(want, got):
        assert np.array_equal(w[~overlap] > 0, g[~overlap] > 0)
    assert _cut_cost(imgs, got) <= 1.01 * _cut_cost(imgs, want) + 1e-6


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("angle", [0.0, 3.0])
def test_dp_seams_match_cv2_on_synthetic_overlaps(n, angle):
    for h, w, shift in ((40, 60, 25), (128, 128, 60), (300, 400, 250)):
        _check_seams(*_parts(h, w, n, shift, angle))


def test_dp_seams_match_cv2_on_small_random_cuts():
    for _ in range(100):
        h, w = (int(v) for v in _RNG.integers(8, 40, 2))
        _check_seams(*_parts(h, w, 2, int(_RNG.integers(2, w)), 5.0))


def test_dp_seams_match_cv2_on_the_recorded_panorama():
    for warped, masks, _, _ in _recorded():
        h, w = masks[0].shape
        scale = min(1.0, logic._SEAM_MAX_EDGE / max(w, h))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = [logic._seam_cost_image(negpy_core.resize_area(np.ascontiguousarray(i, dtype=np.float32), size)) for i in warped]
        _check_seams(small, [negpy_core.resize_nearest(m.astype(np.uint8), size) for m in masks])


def test_find_seams_partitions_the_coverage():
    imgs, masks = _parts(90, 120, 3, 50, 2.0)
    seams = logic.find_seams(imgs, [m > 0 for m in masks], (imgs[0].shape[1], imgs[0].shape[0]))
    assert seams is not None
    covered = np.any([m > 0 for m in masks], 0)
    assert np.array_equal(np.sum(seams, 0) >= 1, covered)


def _seam_weight_cv2(seam, valid, feather_px):
    span = 2.0 * feather_px
    owned = seam.astype(np.uint8)
    weight = cv2.distanceTransform(owned, cv2.DIST_L2, 3) / np.float32(span) + np.float32(0.5)
    beyond = np.float32(0.5) - cv2.distanceTransform(1 - owned, cv2.DIST_L2, 3) / np.float32(span)
    weight = np.where(seam, np.clip(weight, 0.0, 1.0), np.clip(beyond, 0.0, 1.0))
    edge = np.clip(cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3) / np.float32(feather_px), 0.0, 1.0)
    return weight * edge


def _mean_cv2(warped, weights):
    acc = np.zeros_like(warped[0])
    weight_sum = np.zeros(warped[0].shape[:2], np.float32)
    for img, weight in zip(warped, weights):
        acc += img * weight[..., None]
        weight_sum += weight
    return acc / np.maximum(weight_sum, 1e-6)[..., None]


@functools.cache
def _blend_cases():
    cases = [(w, m, s, f) for w, m, s, f in _recorded()]
    for n, size in ((2, (120, 160, 70)), (3, (200, 260, 150)), (4, (64, 90, 40))):
        imgs, masks = _parts(*size, n, 3.0)
        seams = [c > 0 for c in negpy_core.dp_seams(imgs, masks)]
        cases.append((imgs, [m > 0 for m in masks], seams, 12.0))
    return cases


def test_distance_transform_stays_near_cv2():
    masks = [m for _, ms, ss, _ in _blend_cases() for m in (*ms, *ss)]
    masks += [_RNG.random((1500, 1600)) > 0.0005, np.pad(np.ones((1400, 1500), bool), 50)]
    for m in masks:
        for src in (m.astype(np.uint8), (~m).astype(np.uint8)):
            want = cv2.distanceTransform(src, cv2.DIST_L2, 3)
            assert np.abs(negpy_core.distance_transform(src, cv2.DIST_L2, 3) - want).max() <= 0.005


def test_seam_blend_matches_cv2_version():
    for warped, masks, seams, feather in _blend_cases():
        f = max(feather, 1.0)
        want = _mean_cv2(warped, [_seam_weight_cv2(s, m, f) for s, m in zip(seams, masks)])
        assert np.abs(logic.seam_blend(warped, masks, seams, feather) - want).max() <= 1e-4


def test_feather_blend_matches_cv2_version():
    for warped, masks, _, _ in _blend_cases():
        want = _mean_cv2(warped, [cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3) for m in masks])
        assert np.abs(logic.feather_blend(warped, masks) - want).max() <= 1e-4
