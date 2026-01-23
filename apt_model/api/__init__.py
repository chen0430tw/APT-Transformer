"""
apt_model.api - Legacy API compatibility package

This redirects to apt.apps.api (APT 2.0)

⚠️ Deprecated: Use apt.apps.api instead
"""

import warnings

warnings.warn(
    "apt_model.api is deprecated. Use apt.apps.api instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
try:
    from apt.apps.api import *
except ImportError as e:
    print(f"⚠️  Error importing API: {e}")
    print("   API functionality is in apt.apps.api")
