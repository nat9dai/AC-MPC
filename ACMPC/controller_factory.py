"""Legacy compatibility layer for the former controller factory APIs.

The refactored codebase exposes :class:`~ACMPC.mpc.EconomicMPCHead` as the
canonical interface for building controllers. Importing from this module
raises an informative error to direct contributors to the new entry point.
"""

from __future__ import annotations

from .mpc import EconomicMPCConfig, EconomicMPCHead


class StandardMPCConfig(EconomicMPCConfig):
    """Placeholder maintained for backwards compatibility."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "StandardMPCConfig has been superseded by EconomicMPCConfig. "
            "Update your code to import from ACMPC.mpc instead."
        )


def build_standard_controller(*args, **kwargs):
    raise RuntimeError(
        "build_standard_controller has been removed. "
        "Instantiate EconomicMPCHead with the appropriate EconomicMPCConfig."
    )


__all__ = ["EconomicMPCConfig", "EconomicMPCHead", "StandardMPCConfig", "build_standard_controller"]
