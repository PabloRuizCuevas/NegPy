"""The Multi-core CPU Rendering setting: one thread or all, the same bits either way; policy, override."""

import threading

import numpy as np
import pytest

from negpy.kernel.system import parallel as par
from negpy.kernel.system.override import OverrideConfig, _parse
from negpy.kernel.system.parallel import default_cpu_parallel


@pytest.fixture(autouse=True)
def _restore_flag():
    before = par.parallel_enabled()
    yield
    par.set_parallel_enabled(before)


def test_platform_policy():
    assert default_cpu_parallel("darwin") is False
    assert default_cpu_parallel("win32") is True
    assert default_cpu_parallel("linux") is True


def test_print_curve_parity_serial_vs_parallel():
    from negpy.features.exposure.logic import apply_characteristic_curve

    rng = np.random.default_rng(7)
    img = rng.random((64, 96, 3), dtype=np.float32)

    par.set_parallel_enabled(True)
    res_par = apply_characteristic_curve(img, (0.5, 2.0), (0.5, 2.0), (0.5, 2.0))
    par.set_parallel_enabled(False)
    res_ser = apply_characteristic_curve(img, (0.5, 2.0), (0.5, 2.0), (0.5, 2.0))

    np.testing.assert_array_equal(res_par, res_ser)


def test_lab_roundtrip_parity_serial_vs_parallel():
    from negpy.kernel.image.logic import rgb_to_lab_working

    rng = np.random.default_rng(11)
    img = rng.random((32, 48, 3), dtype=np.float32)

    par.set_parallel_enabled(True)
    lab_par = rgb_to_lab_working(img)
    par.set_parallel_enabled(False)
    lab_ser = rgb_to_lab_working(img)

    np.testing.assert_array_equal(lab_par, lab_ser)


def test_concurrent_calls_while_the_setting_flips_are_safe():
    from negpy.kernel.image.logic import rgb_to_lab_working

    img = np.random.default_rng(3).random((300, 400, 3), dtype=np.float32)
    want = rgb_to_lab_working(img)
    errors: list[Exception] = []

    def hammer(flip: bool):
        try:
            for i in range(20):
                if flip:
                    par.set_parallel_enabled(i % 2 == 0)
                np.testing.assert_array_equal(rgb_to_lab_working(img), want)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i == 0,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_override_parsing_and_apply():
    assert _parse({"performance": {"cpu_parallel": False}}).cpu_parallel is False
    assert _parse({"performance": {"cpu_parallel": True}}).cpu_parallel is True
    assert _parse({"performance": {}}).cpu_parallel is None
    assert _parse({"performance": {"cpu_parallel": "yes"}}).cpu_parallel is None  # non-bool ignored

    from negpy.kernel.system.override import apply as apply_override
    from negpy.domain.types import AppConfig

    cfg = AppConfig(
        thumbnail_size=1,
        max_workers=1,
        preview_render_size=1,
        max_history_steps=1,
        edits_db_path="",
        settings_db_path="",
        presets_dir="",
        cache_dir="",
        user_icc_dir="",
        crosstalk_dir="",
        sensor_dir="",
        flatfield_dir="",
        gear_dir="",
        default_export_dir="",
        adobe_rgb_profile="",
    )
    assert cfg.cpu_parallel is None
    apply_override(OverrideConfig(cpu_parallel=False), cfg)
    assert cfg.cpu_parallel is False


def test_configure_cpu_parallel_override_wins():
    par.configure_cpu_parallel(False)
    assert par.parallel_enabled() is False
    par.configure_cpu_parallel(True)
    assert par.parallel_enabled() is True
    par.configure_cpu_parallel(None)
    assert par.parallel_enabled() is default_cpu_parallel()
