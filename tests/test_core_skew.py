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
