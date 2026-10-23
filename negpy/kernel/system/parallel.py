"""
The "Multi-core CPU Rendering" setting: negpy_core runs its kernels on every core, or on one
thread when the setting is off. It takes effect from the next call.

Default policy: on, except macOS. Override with `cpu_parallel = true/false` under
[performance] in override.toml.
"""

import sys
from typing import Optional

import negpy_core

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def default_cpu_parallel(platform: str = sys.platform) -> bool:
    """Platform policy: multi-core on, except macOS."""
    return platform != "darwin"


_parallel_enabled: bool = default_cpu_parallel()
negpy_core.set_parallel(_parallel_enabled)


def parallel_enabled() -> bool:
    return _parallel_enabled


def set_parallel_enabled(enabled: bool) -> None:
    global _parallel_enabled
    _parallel_enabled = bool(enabled)
    negpy_core.set_parallel(_parallel_enabled)


def resolve_cpu_parallel(file_override: Optional[bool], stored: Optional[bool]) -> Optional[bool]:
    """Which answer wins: override.toml, then the saved setting, then the platform default.

    None at every level means "platform default", so an installation that has never touched
    the setting keeps whatever its platform does today — on for Windows and Linux, off for
    macOS. Surfacing the control must not silently re-decide it for anyone.
    """
    if file_override is not None:
        return file_override
    return stored


def configure_cpu_parallel(override: Optional[bool]) -> None:
    """Apply the startup policy: explicit override wins, else the platform default."""
    set_parallel_enabled(default_cpu_parallel() if override is None else override)
    logger.info(
        "CPU multi-core rendering: %s (override=%s, platform=%s)",
        "enabled" if _parallel_enabled else "disabled",
        override,
        sys.platform,
    )
