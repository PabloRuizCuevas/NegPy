"""negpy_core's line segment detector and `_coarse_angles` against cv2's LSD. The port is gated on
quality, not bits: long segments must land on cv2's and the angle modes must agree."""

import glob
import os
import pickle

import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.geometry import skew
from negpy.features.geometry.logic import AUTOCROP_DETECT_RES, _detection_luma, _normalize_detection_input

pytestmark = cv2_oracle

_CORPUS = os.path.expanduser("~/negpy-corpus/features__geometry__skew.py--measure_frame_skew")
_RECORDINGS = sorted(glob.glob(os.path.join(_CORPUS, "*.pkl")))


def _to_u8(lum):
    lo, hi = np.percentile(lum, (1.0, 99.0))
    return (np.clip((lum - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)


def _coarse_angles_cv2(lum):
    """The cv2-based `_coarse_angles` that negpy_core.coarse_angles replaces."""
    h, w = lum.shape
    segments = cv2.createLineSegmentDetector().detect(_to_u8(lum))[0]
    if segments is None:
        return []
    seg = segments.reshape(-1, 4).astype(np.float64)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))
    angle = (angle + 45.0) % 90.0 - 45.0
    keep = (length >= skew._MIN_SEGMENT * max(h, w)) & (np.abs(angle) <= skew._MAX_SKEW_DEGREES)
    if not np.any(keep):
        return []
    bins = np.arange(-skew._MAX_SKEW_DEGREES, skew._MAX_SKEW_DEGREES + skew._ANGLE_BIN, skew._ANGLE_BIN)
    hist, _ = np.histogram(angle[keep], bins=bins, weights=length[keep] ** 2)
    hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), 3.0).ravel()
    first = int(np.argmax(hist))
    modes = [float(bins[first] + skew._ANGLE_BIN / 2)]
    rest = hist.copy()
    rest[max(0, first - 10) : first + 11] = 0.0
    second = int(np.argmax(rest))
    if rest[second] >= 0.25 * hist[first]:
        modes.append(float(bins[second] + skew._ANGLE_BIN / 2))
    return modes


def _matched_fraction(lum):
    """Fraction of cv2's segments longer than `_MIN_SEGMENT` of the long side that a negpy_core
    segment matches with both endpoints within 1 px, in either direction."""
    u8 = _to_u8(lum)
    ref = cv2.createLineSegmentDetector().detect(u8)[0]
    ref = np.empty((0, 4), np.float32) if ref is None else ref.reshape(-1, 4)
    got = negpy_core.line_segments(u8)
    long = ref[np.hypot(ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]) >= skew._MIN_SEGMENT * max(u8.shape)]
    if long.size == 0:
        return 1.0
    if got.size == 0:
        return 0.0

    def far(a, b):
        return np.maximum(np.hypot(got[:, 0] - a[0], got[:, 1] - a[1]), np.hypot(got[:, 2] - b[0], got[:, 3] - b[1]))

    hits = [min(far(s[:2], s[2:]).min(), far(s[2:], s[:2]).min()) <= 1.0 for s in long]
    return float(np.mean(hits))


def _assert_same_modes(lum):
    want, got = _coarse_angles_cv2(lum), skew._coarse_angles(lum)
    assert len(want) == len(got)
    assert np.allclose(want, got, rtol=0.0, atol=0.05)


def _synthetic(angle, seed):
    """A light bed, a dark film frame and a picture with a few hard lines, turned by `angle`."""
    rng = np.random.default_rng(seed)
    h, w = 900, 1300
    img = np.full((h, w), 0.92, np.float32)
    img[150:750, 180:1120] = 0.08
    img[190:710, 230:1070] = 0.55
    img[300:330, 300:1000] = 0.3
    img[420:640, 600:615] = 0.8
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img + rng.normal(0.0, 0.02, img.shape).astype(np.float32)


@pytest.mark.parametrize("angle", [-9.5, -4.2, -0.7, 0.0, 0.3, 2.6, 6.1, 9.8])
def test_synthetic_frames_match_cv2(angle):
    lum = _synthetic(angle, int(abs(angle) * 10))
    assert _matched_fraction(lum) >= 0.95
    _assert_same_modes(lum)


def test_flat_and_tiny_planes():
    flat = np.full((64, 80), 0.5, np.float32)
    assert negpy_core.line_segments(_to_u8(flat)).shape == (0, 4)
    assert skew._coarse_angles(flat) == []
    assert negpy_core.line_segments(np.zeros((1, 1), np.uint8)).shape == (0, 4)


def _recorded_luma(rec):
    det, _ = _normalize_detection_input(rec["args"][0], AUTOCROP_DETECT_RES)
    return _detection_luma(np.ascontiguousarray(det, dtype=np.float32))


@pytest.mark.skipif(not _RECORDINGS, reason="needs the replay corpus")
def test_recorded_planes_match_cv2():
    for path in _RECORDINGS:
        with open(path, "rb") as f:
            rec = pickle.load(f)
        lum = _recorded_luma(rec)
        assert _matched_fraction(lum) >= 0.95, path
        _assert_same_modes(lum)


@pytest.mark.skipif(not _RECORDINGS, reason="needs the replay corpus")
def test_recorded_frame_skew_unchanged():
    for path in _RECORDINGS:
        with open(path, "rb") as f:
            rec = pickle.load(f)
        want, got = rec["result"], skew.measure_frame_skew(*rec["args"], **rec["kwargs"])
        assert (want is None) == (got is None), path
        if want is None or got is None:
            continue
        assert abs(want.fine_rotation - got.fine_rotation) <= 0.05, path
        for a, b in ((want.converge_v, got.converge_v), (want.converge_h, got.converge_h)):
            assert (a is None) == (b is None), path
            assert a is None or abs(a - b) <= 0.05, path
