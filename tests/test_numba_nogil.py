import importlib

import pytest

_KERNELS = [
    ("negpy.features.exposure.logic", "_apply_print_curve_kernel"),
    ("negpy.infrastructure.display.icc_lut", "_apply_lut_u16_jit"),
    ("negpy.infrastructure.display.icc_lut", "_apply_lut_f32_jit"),
    ("negpy.infrastructure.display.icc_lut", "_matrix_trc_u8_jit"),
]


@pytest.mark.parametrize("module, name", _KERNELS)
def test_full_image_kernels_release_the_gil(module, name):
    kernel = getattr(importlib.import_module(module), name)
    dispatchers = (kernel.serial, kernel.parallel) if hasattr(kernel, "serial") else (kernel,)
    for d in dispatchers:
        assert d.targetoptions.get("nogil") is True
