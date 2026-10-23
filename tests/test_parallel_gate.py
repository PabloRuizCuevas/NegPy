"""Which answer sets Multi-core CPU Rendering: override.toml, the saved setting, the default."""

from negpy.kernel.system.parallel import default_cpu_parallel, resolve_cpu_parallel


class TestResolution:
    def test_an_untouched_install_takes_the_default_on(self):
        assert default_cpu_parallel() is True
        assert resolve_cpu_parallel(None, None) is None, "None must fall through to the default"

    def test_the_saved_setting_beats_the_default(self):
        assert resolve_cpu_parallel(None, True) is True
        assert resolve_cpu_parallel(None, False) is False

    def test_the_override_file_beats_the_saved_setting(self):
        """override.toml is the escape hatch for a machine that cannot boot far enough to
        reach the menu."""
        assert resolve_cpu_parallel(False, True) is False
        assert resolve_cpu_parallel(True, False) is True
