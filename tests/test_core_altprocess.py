import cv2
import negpy_core
import numpy as np
import pytest
from core_oracle import cv2_oracle

from negpy.features.cyanotype.logic import CYANOTYPE_CONSTANTS, apply_cyanotype, sensitizer_constants
from negpy.features.lab.logic import gaussian_kernel_1d
from negpy.features.lith.logic import LITH_CONSTANTS, apply_lith
from negpy.features.sabattier.logic import SABATTIER_CONSTANTS, apply_sabattier
from negpy.kernel.image.logic import _LAB_EPS, _LAB_KAPPA, lab_to_rgb_working

pytestmark = cv2_oracle

_RNG = np.random.default_rng(23)
_F = np.float32


def _cr(fn):
    """A float32 transcendental, correctly rounded: NumPy's own run SVML, whose bits depend on the CPU."""
    return lambda x: fn(np.asarray(x, dtype=np.float64)).astype(np.float32)


_log10, _exp, _log1p, _cbrt = _cr(np.log10), _cr(np.exp), _cr(np.log1p), _cr(np.cbrt)


def _pow10(x):
    return np.power(10.0, np.asarray(x, dtype=np.float64)).astype(np.float32)


def _density(img):
    arr = np.clip(img, 1e-6, 1.0)
    luma = arr[:, :, 0] * _F(0.2126) + arr[:, :, 1] * _F(0.7152) + arr[:, :, 2] * _F(0.0722)
    return -_log10(np.clip(luma, 1e-6, 1.0))


def _interp(u, xp, anchors, i):
    return np.interp(u, np.asarray(xp, _F), np.asarray([p[i] for p in anchors], _F)).astype(_F)


def _to_rgb(dens, a, b):
    grey = _pow10(-dens)
    f = np.where(grey > _F(_LAB_EPS), _cbrt(grey), _F(_LAB_KAPPA) * grey + _F(16.0 / 116.0))
    return np.clip(lab_to_rgb_working(np.dstack([_F(116.0) * f - _F(16.0), a, b])), 0.0, 1.0)


def _sabattier_reference(img, d_max, strength, reexposure, sigma):
    c = SABATTIER_CONSTANTS
    dens = _density(img)
    re, w = _F(reexposure * d_max), _F(c["fold_width"])
    reversal = _F(strength) * (w * _log1p(_exp(np.clip((re - dens) / w, -30.0, 30.0))))
    if sigma > 0.0:
        developed = 1.0 / (1.0 + _exp(-np.clip((dens - re) / _F(c["edge_width"]), -30.0, 30.0)))
        k = gaussian_kernel_1d(sigma)
        reversal = reversal * (_F(1.0) - cv2.sepFilter2D(developed.astype(_F), -1, k, k, borderType=cv2.BORDER_REPLICATE))
    grey = _pow10(-np.minimum(dens + reversal, _F(d_max)))
    return np.clip(np.dstack([grey, grey, grey]), 0.0, 1.0)


def _cyanotype_reference(img, sensitizer, exposure, scale, bleach, tannin):
    c, sc = CYANOTYPE_CONSTANTS, sensitizer_constants(sensitizer)
    t = np.clip((_density(img) + _F(0.301 * exposure)) / _F(max(scale, 0.1)), 0.0, 1.0)
    m = _F(c["mid_compress"])
    v = _F(2.0) * t - _F(1.0)
    u0 = (_F(1.0) - m) * t + m * _F(0.5) * (_F(1.0) + v * np.abs(v))
    u_b = u0 * (_F(1.0) - _F(bleach) * (_F(1.0) - _F(c["bleach_floor"]) * u0))
    u = u_b + _F(tannin) * (u0 * _F(c["tannin_restore"]) - u_b)
    dens = _F(sc["d_max"]) * (_F(1.0) + _F(tannin) * _F(c["tannin_dmax_gain"])) * u
    frac = np.clip(u, 0.0, 1.0)
    a, b = (_interp(frac, c["path_u"], sc["path"], i) for i in (0, 1))
    a = a + _F(tannin) * (_F(c["brown_dir"][0]) * frac - a)
    b = b + _F(tannin) * (_F(c["brown_dir"][1]) * frac - b)
    return _to_rgb(dens, a, b)


def _lith_reference(img, path, d_max, exposure, snatch, abruptness):
    c = LITH_CONSTANTS
    base = _density(img)
    over = _F(0.301 * exposure)
    foot_max = _F(c["foot_max"])
    d_h = foot_max * (_F(1.0) - _exp(-_F(c["foot_rate"]) * (base + over * _F(c["foot_veil"])) / foot_max))
    knee = _F(d_max * (c["knee_lo"] - c["knee_span"] * snatch))
    width = _F(max(c["abrupt_lo"] - c["abrupt_span"] * abruptness, 0.01))
    g = 1.0 / (1.0 + _exp(-np.clip((base + over - knee) / width, -30.0, 30.0)))
    dens = d_h + (_F(d_max) - d_h) * g
    frac = np.clip(dens / _F(d_max), 0.0, 1.0)
    return _to_rgb(dens, *(_interp(frac, c["path_u"], path, i) for i in (0, 1)))


_IMAGES = [_RNG.random((97, 131, 3), dtype=np.float32), _RNG.random((40, 50, 3), dtype=np.float32) ** 3]


@pytest.mark.parametrize("img", _IMAGES)
@pytest.mark.parametrize("d_max, strength, reexposure, sigma", [(2.1, 1.3, 0.45, 0.0), (1.9, 0.6, 0.3, 2.5)])
def test_sabattier_matches_numpy(img, d_max, strength, reexposure, sigma):
    want = _sabattier_reference(img, d_max, strength, reexposure, sigma)
    assert apply_sabattier(img, d_max, True, strength, reexposure, sigma).tobytes() == want.tobytes()


@pytest.mark.parametrize("img", _IMAGES)
@pytest.mark.parametrize("args", [("classic", 0.0, 1.4, 0.0, 0.0), ("new", 1.0, 2.3, 0.6, 0.4)])
def test_cyanotype_matches_numpy(img, args):
    assert apply_cyanotype(img, args[0], True, *args[1:]).tobytes() == _cyanotype_reference(img, *args).tobytes()


@pytest.mark.parametrize("img", _IMAGES)
@pytest.mark.parametrize("args", [(2.0, 0.0, 0.55, 0.6), (1.8, 1.5, 0.9, 0.1)])
def test_lith_matches_numpy(img, args):
    path = ((4.0, 12.0), (6.0, 18.0), (2.0, 8.0), (0.0, 0.5))
    want = _lith_reference(img, path, *args)
    d_max, exposure, snatch, abruptness = args
    assert apply_lith(img, path, d_max, True, exposure, snatch, abruptness).tobytes() == want.tobytes()


def test_sabattier_rejects_a_bad_image():
    with pytest.raises(ValueError):
        negpy_core.sabattier(np.zeros((4, 4), np.float32), (_F(1.0),) * 5, np.zeros(0, np.float32))
