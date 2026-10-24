import math

import negpy_core
import numpy as np
import pytest


def _profile_peaks(profile):
    """The NumPy peak finder that negpy_core.skew_sample replaces."""
    if profile.size < 5:
        return np.empty(0)
    grad = np.abs(np.gradient(negpy_core.gaussian_blur(profile.reshape(-1, 1).astype(np.float32), (1, 0), 0, 1.0).ravel()))
    peak = float(grad.max())
    if peak <= 3.0 * float(np.median(grad)) + 1e-6:
        return np.empty(0)
    i = np.arange(1, grad.size - 1)
    i = i[(grad[i] >= grad[i - 1]) & (grad[i] > grad[i + 1]) & (grad[i] >= 0.4 * peak)]
    a, b, c = grad[i - 1], grad[i], grad[i + 1]
    denom = a - 2 * b + c
    safe = np.where(np.abs(denom) > 1e-9, denom, 1.0)
    return i + np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / safe, 0.0)


def _sample(lum, horizontal, stations, center, half, nearest):
    img = lum if horizontal else lum.T
    h, w = img.shape
    xs, ys = [], []
    for x, c in zip(stations, center):
        x0 = int(round(x))
        lo, hi = int(max(0, math.floor(c - half))), int(min(h, math.ceil(c + half) + 1))
        if hi - lo < 5 or x0 < 1 or x0 > w - 2:
            continue
        peaks = _profile_peaks(img[lo:hi, x0 - 1 : x0 + 2].mean(axis=1))
        if peaks.size == 0:
            continue
        if nearest:
            peaks = peaks[[int(np.argmin(np.abs(peaks - (c - lo))))]]
        xs.extend([x] * peaks.size)
        ys.extend(lo + peaks)
    return np.asarray(xs), np.asarray(ys)


@pytest.mark.parametrize("horizontal", [True, False])
@pytest.mark.parametrize("nearest", [False, True])
def test_skew_sample_matches_numpy(horizontal, nearest):
    rng = np.random.default_rng(31)
    lum = rng.random((180, 260), dtype=np.float32) * np.float32(0.05)
    lum[60:, :] += np.float32(0.8)
    lum[:, 200:] += np.float32(0.3)
    lum[:, 7] = 1.0
    stations = np.linspace(4.0, (260 if horizontal else 180) - 5.0, 37)
    center = np.full(stations.size, 61.0) + np.linspace(-3.0, 3.0, stations.size)
    want = _sample(lum, horizontal, stations, center, 14.0, nearest)
    got = negpy_core.skew_sample(lum, horizontal, stations, center, 14.0, nearest)
    assert want[0].tobytes() == got[0].tobytes()
    assert want[1].tobytes() == got[1].tobytes()


def _ransac_line_numpy(xs, ys, expected, half, span):
    if xs.size < 12:
        return None
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, xs.size, (2, 300))
    ok = np.abs(xs[a] - xs[b]) >= 0.3 * span
    a, b = a[ok], b[ok]
    if a.size == 0:
        return None
    slope = (ys[b] - ys[a]) / (xs[b] - xs[a])
    ok = np.abs(slope) <= math.tan(math.radians(4.0))
    a, b, slope = a[ok], b[ok], slope[ok]
    if a.size == 0:
        return None
    intercept = ys[a] - slope * xs[a]
    inlier = np.abs(ys[None, :] - (slope[:, None] * xs[None, :] + intercept[:, None])) <= 1.5
    stations = np.array([np.unique(xs[row]).size for row in inlier])
    mid = slope * (0.5 * (xs.min() + xs.max())) + intercept
    score = stations * (1.0 - 0.5 * np.minimum(1.0, np.abs(mid - expected) / half))
    best = int(np.argmax(score))
    if stations[best] < 12:
        return None
    return np.polyfit(xs[inlier[best]], ys[inlier[best]], 1)


def _fit_side_numpy(lum, horizontal, at, start, stop, outward=0):
    """skew.py's NumPy _fit_side, the reference for negpy_core.skew_fit_side."""
    h, w = lum.shape if horizontal else lum.T.shape
    length = stop - start
    if length < 0.25 * w or not (at >= 2 and at <= h - 3):
        return None
    stations = np.linspace(start + 0.1 * length, stop - 0.1 * length, int(min(240, max(24, length / 4))))
    plane = np.ascontiguousarray(lum, dtype=np.float32)

    def sample(center, half, nearest):
        return negpy_core.skew_sample(plane, horizontal, stations, np.ascontiguousarray(center, dtype=np.float64), float(half), nearest)

    inside = max(10.0, 0.035 * max(h, w))
    outside = inside if outward == 0 else max(16.0, 0.02 * h)
    center = float(at) + 0.5 * outward * (outside - inside)
    half = 0.5 * (inside + outside)
    xs, ys = sample(np.full(stations.size, center), half, False)
    fit = _ransac_line_numpy(xs, ys, float(at), half, length)
    if fit is None:
        return None
    xs, ys = sample(np.polyval(fit, stations), max(6.0, 0.3 * inside), True)
    if xs.size < 12:
        return None
    keep = np.ones(xs.size, bool)
    for _ in range(8):
        fit = np.polyfit(xs[keep], ys[keep], 1)
        res = ys - np.polyval(fit, xs)
        mad = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        new = np.abs(res) <= max(1.0, 2.5 * mad)
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new
    inliers = float(keep.sum()) / stations.size
    rms = float(np.sqrt(np.mean((ys[keep] - np.polyval(fit, xs[keep])) ** 2)))
    if inliers < 0.5 or rms > 2.0:
        return None
    x1, x2 = float(xs[keep].min()), float(xs[keep].max())
    if x2 - x1 < 0.25 * w:
        return None
    y1, y2 = float(np.polyval(fit, x1)), float(np.polyval(fit, x2))
    p1, p2 = ((x1, y1), (x2, y2)) if horizontal else ((y1, x1), (y2, x2))
    return p1, p2, horizontal, inliers, rms


def _frame(angle=1.2, size=(420, 600), seed=5):
    """A bright film box on a dark bed, turned by `angle` degrees, with grain."""
    h, w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    t = math.radians(angle)
    u = (x - w / 2) * math.cos(t) + (y - h / 2) * math.sin(t)
    v = -(x - w / 2) * math.sin(t) + (y - h / 2) * math.cos(t)
    lum = np.where((np.abs(u) < 0.38 * w) & (np.abs(v) < 0.36 * h), 0.85, 0.08)
    return (lum + np.random.default_rng(seed).normal(0, 0.01, lum.shape)).astype(np.float32)


def _close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    flat = lambda line: np.array([*line[0], *line[1], line[3], line[4]])  # noqa: E731
    return a[2] == b[2] and np.allclose(flat(a), flat(b), rtol=0, atol=tol)


@pytest.mark.parametrize("angle", [-3.0, -0.4, 0.0, 0.9, 2.5])
@pytest.mark.parametrize("horizontal", [True, False])
def test_fit_side_matches_numpy(angle, horizontal):
    lum = _frame(angle, seed=int(angle * 10) + 50)
    h, w = lum.shape
    sides = (
        [(0.14 * h, 0.12 * w, 0.88 * w, -1), (0.86 * h, 0.12 * w, 0.88 * w, 1)]
        if horizontal
        else [(0.12 * w, 0.14 * h, 0.86 * h, -1), (0.88 * w, 0.14 * h, 0.86 * h, 1)]
    )
    found = 0
    for at, start, stop, outward in sides:
        for out in (outward, 0):
            want = _fit_side_numpy(lum, horizontal, at, start, stop, out)
            got = negpy_core.skew_fit_side(lum, horizontal, at, start, stop, out)
            assert _close(got, want), (at, out, got, want)
            found += want is not None
    assert found > 0
