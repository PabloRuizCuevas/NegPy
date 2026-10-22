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

_CASES = {
    "percentiles": lambda: np.array(negpy_core.percentiles(_IMG1.ravel(), [0.5, 1.0, 37.0, 50.0, 99.0]), np.float32),
    "resize_area_general": lambda: negpy_core.resize_area(_IMG3, (47, 31)),
    "resize_area_2x2": lambda: negpy_core.resize_area(_IMG1[:82, :60].copy(), (30, 41)),
    "gaussian_5x5": lambda: negpy_core.gaussian_blur(_IMG3, (5, 5), 0.0),
    "gaussian_sigma2_replicate": lambda: negpy_core.gaussian_blur(_IMG1, (0, 0), 2.0, 0.0, 1),
    "gaussian_kernel_17": lambda: np.array(negpy_core.gaussian_kernel(17, 2.0), np.float32),
    "sobel_x": lambda: negpy_core.sep_filter2d(_IMG1, np.array([-1, 0, 1], np.float32), np.array([1, 2, 1], np.float32)),
    "morph_close_u8": lambda: negpy_core.morphology_ex((_IMG1 > 0.5).astype(np.uint8) * 255, 3, np.ones((9, 7), np.uint8), 2),
}

_HASHES = {
    "percentiles": "0874152b6e9f91bd3a3b73e15a9d083d15b6e8c7c6f6a9279d9241df10a88464",
    "resize_area_general": "dbe0fb9e51b0e8bee54cc2bb35756bbf0934077b458c1a44ea27c2e5dd96a16f",
    "resize_area_2x2": "cae2830d9848ee6f270000d57b3dc448138cc99cb5a2303108ac3e095b060488",
    "gaussian_5x5": "988a867049827fd5912f1950b8574b637d671c3565e0b7af87def261c9f0b084",
    "gaussian_sigma2_replicate": "2b9be6f95b7f66a03493795d8c7a5e1bb2565a28087578f1981a9cd329c8a74f",
    "gaussian_kernel_17": "5c2eb92d43931221106d80ca89a0b6dea1dca09b261b0fbf6ff51fdcae748d8b",
    "sobel_x": "ba798e861db64c102fee6b08fd7aa34d488cce431ae0d95fc3e7836e1dedb44a",
    "morph_close_u8": "4be9a2976881ff83eb0b92a8ec484e048c61612324afbde73b6e0c89aa92865e",
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_same_bits_on_every_platform(name):
    out = np.ascontiguousarray(_CASES[name]())
    assert hashlib.sha256(out.tobytes()).hexdigest() == _HASHES[name]


def test_every_case_has_a_hash():
    assert _CASES.keys() == _HASHES.keys()
