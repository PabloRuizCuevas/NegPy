"""The platform default of Multi-core CPU Rendering must not move for anyone who never touches it."""

import pytest

from negpy.kernel.system.parallel import default_cpu_parallel, resolve_cpu_parallel


class TestPlatformDefaultIsUnmoved:
    """Surfacing the setting must not silently re-decide it for anyone who never touches it."""

    @pytest.mark.parametrize("platform,expected", [("win32", True), ("linux", True), ("darwin", False)])
    def test_untouched_installs_keep_their_platform_behaviour(self, platform, expected):
        assert default_cpu_parallel(platform) is expected
        assert resolve_cpu_parallel(None, None) is None, "None must fall through to the platform default"

    def test_the_saved_setting_beats_the_platform_default(self):
        assert resolve_cpu_parallel(None, True) is True
        assert resolve_cpu_parallel(None, False) is False

    def test_the_override_file_beats_the_saved_setting(self):
        """override.toml is the escape hatch for a machine that cannot boot far enough to
        reach the menu."""
        assert resolve_cpu_parallel(False, True) is False
        assert resolve_cpu_parallel(True, False) is True
