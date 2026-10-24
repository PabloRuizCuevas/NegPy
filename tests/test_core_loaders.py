"""The loader kernels against the NumPy and cv2 code they replace."""

import cv2
import numpy as np
import pytest
from core_oracle import cv2_oracle

import negpy_core
from negpy.infrastructure.loaders.logluv import normalize_linear


def _normalize_linear_numpy(lin):
    flat = lin.reshape(-1, lin.shape[-1])
    lo = np.percentile(flat, 0.2, axis=0)
    hi = np.percentile(flat, 99.8, axis=0)
    denom = hi - lo
    denom = np.where(denom > 0, denom, 1.0)
    return np.clip((lin - lo) / denom, 0.0, 1.0)


@pytest.mark.parametrize("shape", [(1, 1, 3), (7, 5, 3), (311, 517, 3), (40, 30, 4)])
def test_normalize_linear_matches_numpy(shape):
    rng = np.random.default_rng(sum(shape))
    lin = rng.lognormal(0.0, 1.5, shape) * np.array([0.3, 1.0, 2.0, 1.0][: shape[-1]])
    lin[..., -1] = 0.25 if shape[0] == 7 else lin[..., -1]
    np.testing.assert_array_equal(normalize_linear(lin), _normalize_linear_numpy(lin))


@cv2_oracle
def test_ellipse_kernel_matches_cv2():
    for w in range(1, 42):
        for h in (1, 2, 3, 6, 9, 17, w):
            want = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))
            np.testing.assert_array_equal(negpy_core.ellipse_kernel((w, h)), want, err_msg=f"{w}x{h}")


def test_constants_match_cv2():
    for name in (
        "BORDER_CONSTANT",
        "BORDER_REPLICATE",
        "BORDER_REFLECT",
        "BORDER_REFLECT_101",
        "MORPH_ERODE",
        "MORPH_DILATE",
        "MORPH_OPEN",
        "MORPH_CLOSE",
        "MORPH_GRADIENT",
        "MORPH_TOPHAT",
        "MORPH_BLACKHAT",
    ):
        assert getattr(negpy_core, name) == getattr(cv2, name), name


class _Cv2Core:
    MORPH_ERODE = cv2.MORPH_ERODE
    BORDER_REPLICATE = cv2.BORDER_REPLICATE

    @staticmethod
    def ellipse_kernel(ksize):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

    @staticmethod
    def resize_area(src, dsize):
        return cv2.resize(src, dsize, interpolation=cv2.INTER_AREA)

    @staticmethod
    def morphology_ex(src, op, kernel, border_type):
        return cv2.morphologyEx(src, op, kernel, borderType=border_type)


@cv2_oracle
@pytest.mark.parametrize("samples,preserve_ir", [(3, False), (4, False), (4, True)])
def test_streamed_linearraw_preview_matches_cv2(tmp_path, monkeypatch, samples, preserve_ir):
    import tifffile

    from negpy.infrastructure.loaders import rawpy_loader

    rng = np.random.default_rng(samples)
    data = rng.integers(0, 65535, (300, 470, samples), dtype=np.uint16)
    path = tmp_path / "scan.tif"
    tifffile.imwrite(path, data, tile=(64, 64), photometric="rgb", extrasamples=[0] * (samples - 3))
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        got = rawpy_loader._stream_linearraw_preview(page, page, 100, normalize_tags=False, min_preserve_ir=preserve_ir)
        monkeypatch.setattr(rawpy_loader, "negpy_core", _Cv2Core)
        want = rawpy_loader._stream_linearraw_preview(page, page, 100, normalize_tags=False, min_preserve_ir=preserve_ir)
    np.testing.assert_array_equal(got[0], want[0])
    assert got[1] == want[1]
