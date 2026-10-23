"""The zone, loupe and vectorscope reads against the cv2 and NumPy code they replace."""

import cv2
import numpy as np
from core_oracle import cv2_oracle

import negpy_core
from negpy.features.exposure import analysis as A
from negpy.kernel.image.logic import get_luminance

pytestmark = cv2_oracle


def _frame(seed, shape=(900, 1300)):
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.5 + 0.45 * np.sin(x / (w / 3.0) + seed) * np.cos(y / (h / 2.5))
    rgb = np.stack([base, base * 0.9, base * 1.05], axis=2) + rng.normal(0, 0.02, (h, w, 3))
    return np.clip(rgb, 0, 1).astype(np.float32)


def _zone_field_cv2(rgb, max_edge=A.ZONE_FIELD_MAX_EDGE):
    h, w = rgb.shape[:2]
    step = max(1, max(h, w) // (2 * max_edge))
    small = np.ascontiguousarray(rgb[::step, ::step, :3], dtype=np.float32)
    scale = min(1.0, max_edge / max(small.shape[:2]))
    sh, sw = small.shape[:2]
    lum = get_luminance(cv2.resize(small, (max(1, round(sw * scale)), max(1, round(sh * scale))), interpolation=cv2.INTER_AREA))
    lum = cv2.GaussianBlur(np.ascontiguousarray(lum, dtype=np.float32), (0, 0), 2.0)
    zones = np.clip(np.rint(np.nan_to_num(A.zone_of_encoded(lum))), 0, A.ZONE_COUNT).astype(np.uint8)
    return cv2.medianBlur(zones, 5)


def _anchors_cv2(zones, min_share=0.01):
    out = []
    for zone in np.unique(zones):
        count, comp, stats, _ = cv2.connectedComponentsWithStats((zones == zone).astype(np.uint8), connectivity=4)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] < min_share * zones.size:
                continue
            mask = (comp == i).astype(np.uint8)
            mask[0, :] = mask[-1, :] = 0
            mask[:, 0] = mask[:, -1] = 0
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
            out.append((int(x), int(y), int(zone), float(dist[y, x]), dist))
    return out


def test_zone_field_matches_cv2():
    for seed in range(3):
        rgb = _frame(seed)
        assert A.zone_field(rgb).tobytes() == _zone_field_cv2(rgb).tobytes()


def test_zone_anchors_sit_at_a_most_interior_pixel():
    """Quality gate: cv2's distances run through IPP here, so an anchor may take another pixel
    of the same depth; it must sit within 1e-4 of cv2's deepest point, with the same regions."""
    for seed in range(4):
        zones = A.zone_field(_frame(seed))
        want = _anchors_cv2(zones)
        got = A.zone_anchors(zones)
        assert [g[2] for g in got] == [w[2] for w in want]
        for (x, y, _, inset), (_, _, _, depth, dist) in zip(got, want):
            assert abs(inset - depth) <= 1e-4 * max(1.0, depth)
            assert dist[y, x] >= depth - 1e-4 * max(1.0, depth)


def test_loupe_and_labels_match_numpy():
    for seed in range(3):
        rgb = _frame(seed, (60, 90))
        lum = get_luminance(rgb)
        want = float(np.sqrt(np.mean(cv2.Laplacian(lum, cv2.CV_32F) ** 2)) * 100.0)
        assert A.loupe_acutance(rgb) == want
        cells = A.zone_grid(_frame(seed))
        labels = []
        for zone in np.unique(cells):
            count, comp, _, centroids = cv2.connectedComponentsWithStats((cells == zone).astype(np.uint8), connectivity=4)
            for i in range(1, count):
                ys, xs = np.nonzero(comp == i)
                j = int(np.argmin((xs - centroids[i][0]) ** 2 + (ys - centroids[i][1]) ** 2))
                labels.append((int(xs[j]), int(ys[j]), int(zone)))
        assert A.zone_region_labels(cells) == labels


def test_vectorscope_density_matches_numpy():
    counts = np.random.default_rng(2).integers(0, 50, (A.SCOPE_HUE_BINS, A.SCOPE_CHROMA_BINS)).astype(float)
    for size, log in ((256, True), (180, False)):
        idx, area, orphans, px = negpy_core.scope_lookup(size, log)
        flat = counts.ravel()
        want = np.zeros(idx.shape)
        inside = idx >= 0
        want[inside] = flat[idx[inside]] / np.maximum(area[idx[inside]], 1.0)
        np.add.at(want.reshape(-1), px, flat[orphans])
        assert A.vectorscope_density(counts, size, log).tobytes() == want.tobytes()
