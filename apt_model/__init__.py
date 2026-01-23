"""
apt_model - Legacy compatibility package

This package provides backward compatibility for APT 1.0 commands.
All functionality is now provided by the apt.* package (APT 2.0).

⚠️ Compatibility period: Until 2026-07-22
⚠️ New projects should use apt.* instead

For migration guide, see: docs/ARCHITECTURE_2.0.md
"""

import warnings

warnings.warn(
    "apt_model is deprecated and will be removed after 2026-07-22. "
    "Please migrate to apt.* package (APT 2.0). "
    "See docs/ARCHITECTURE_2.0.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

__version__ = "1.0.0-compat"
__all__ = []
