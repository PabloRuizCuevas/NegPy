import hashlib

import negpy_core
import numpy as np
import pytest

# Recorded on Linux x86-64, where the cv2 oracles pass. Every release platform must give these
# bytes; CI runs this file on each one. A port adds its case here; a hash only changes with a
# decided change of output.
_RNG = np.random.default_rng(2026)
_IMG3 = _RNG.random((97, 131, 3), dtype=np.float32)
_IMG1 = (_RNG.random((83, 61), dtype=np.float32) - 0.4) * 5
_GRAY = negpy_core.bgr_to_gray((_IMG3 * 255).astype(np.uint8))

_CASES = {
    "percentiles": lambda: np.array(negpy_core.percentiles(_IMG1.ravel(), [0.5, 1.0, 37.0, 50.0, 99.0]), np.float32),
    "resize_area_general": lambda: negpy_core.resize_area(_IMG3, (47, 31)),
    "resize_area_2x2": lambda: negpy_core.resize_area(_IMG1[:82, :60].copy(), (30, 41)),
    "gaussian_5x5": lambda: negpy_core.gaussian_blur(_IMG3, (5, 5), 0.0),
    "gaussian_sigma2_replicate": lambda: negpy_core.gaussian_blur(_IMG1, (0, 0), 2.0, 0.0, 1),
    "gaussian_kernel_17": lambda: np.array(negpy_core.gaussian_kernel(17, 2.0), np.float32),
    "sobel_x": lambda: negpy_core.sep_filter2d(_IMG1, np.array([-1, 0, 1], np.float32), np.array([1, 2, 1], np.float32)),
    "bgr_to_gray": lambda: _GRAY,
    "gaussian_blur_u8": lambda: negpy_core.gaussian_blur_u8(_GRAY, (5, 5), 0.0),
    "otsu": lambda: negpy_core.threshold_otsu(_GRAY)[1],
    "canny": lambda: negpy_core.canny(_GRAY, 40, 160),
    "find_external_contours": lambda: np.concatenate(
        [c.ravel() for c in negpy_core.find_external_contours((_IMG1 > 0.5).astype(np.uint8) * np.uint8(255))[0]]
    ),
    "morph_close_u8": lambda: negpy_core.morphology_ex((_IMG1 > 0.5).astype(np.uint8) * 255, 3, np.ones((9, 7), np.uint8), 2),
    "to_log_density": lambda: negpy_core.to_log_density(np.ascontiguousarray(_IMG1 * _IMG1)),
}

_HASHES = {
    "percentiles": "0874152b6e9f91bd3a3b73e15a9d083d15b6e8c7c6f6a9279d9241df10a88464",
    "resize_area_general": "dbe0fb9e51b0e8bee54cc2bb35756bbf0934077b458c1a44ea27c2e5dd96a16f",
    "resize_area_2x2": "cae2830d9848ee6f270000d57b3dc448138cc99cb5a2303108ac3e095b060488",
    "gaussian_5x5": "988a867049827fd5912f1950b8574b637d671c3565e0b7af87def261c9f0b084",
    "gaussian_sigma2_replicate": "2b9be6f95b7f66a03493795d8c7a5e1bb2565a28087578f1981a9cd329c8a74f",
    "gaussian_kernel_17": "5c2eb92d43931221106d80ca89a0b6dea1dca09b261b0fbf6ff51fdcae748d8b",
    "sobel_x": "ba798e861db64c102fee6b08fd7aa34d488cce431ae0d95fc3e7836e1dedb44a",
    "bgr_to_gray": "ea18764e8727ce823408e9962b86969c2de631c5f0246b1d33c3bc833101dac9",
    "gaussian_blur_u8": "7817f7444ec9c843b3c529098d650d653dd0081250770eb96bb378da20de046e",
    "otsu": "cff78130e93898fd1d8ff6080127920c71a5fa1bbecd734a35631b253866ffdb",
    "canny": "1e3125ae47a5dd1719f97531ffebffc79a6461ba1fae024d26d4d7a85b89b968",
    "find_external_contours": "ebb4302dd2bc7226a7ed24fbb5e16e14f407d5693ce47128a649cee9e4a098be",
    "morph_close_u8": "4be9a2976881ff83eb0b92a8ec484e048c61612324afbde73b6e0c89aa92865e",
    "to_log_density": "cc4d2086edf051826619f024a4f5d28c80d28528291d9df4610c1d3854144038",
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_same_bits_on_every_platform(name):
    out = np.ascontiguousarray(_CASES[name]())
    assert hashlib.sha256(out.tobytes()).hexdigest() == _HASHES[name]


def test_every_case_has_a_hash():
    assert _CASES.keys() == _HASHES.keys()
