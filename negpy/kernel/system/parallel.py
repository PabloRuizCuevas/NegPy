"""
The "Multi-core CPU Rendering" setting: negpy_core runs its kernels on every core, or on one
thread when the setting is off. It takes effect from the next call.

Default: on. Override with `cpu_parallel = true/false` under
[performance] in override.toml.
"""

from typing import Optional

import negpy_core

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def default_cpu_parallel() -> bool:
    """Multi-core is on unless the user or override.toml turns it off."""
    return True


_parallel_enabled: bool = default_cpu_parallel()
negpy_core.set_parallel(_parallel_enabled)


def parallel_enabled() -> bool:
    return _parallel_enabled


def set_parallel_enabled(enabled: bool) -> None:
    global _parallel_enabled
    _parallel_enabled = bool(enabled)
    negpy_core.set_parallel(_parallel_enabled)


def resolve_cpu_parallel(file_override: Optional[bool], stored: Optional[bool]) -> Optional[bool]:
    """Which answer wins: override.toml, then the saved setting, then the default (on).
    None at every level means the default."""
    if file_override is not None:
        return file_override
    return stored


def configure_cpu_parallel(override: Optional[bool]) -> None:
    """Apply the startup policy: explicit override wins, else the platform default."""
    set_parallel_enabled(default_cpu_parallel() if override is None else override)
    logger.info("CPU multi-core rendering: %s (override=%s)", "enabled" if _parallel_enabled else "disabled", override)
