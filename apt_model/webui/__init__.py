"""
apt_model.webui - Legacy WebUI compatibility package

This redirects to apt.apps.webui (APT 2.0)

⚠️ Deprecated: Use apt.apps.webui instead
"""

import warnings

warnings.warn(
    "apt_model.webui is deprecated. Use apt.apps.webui instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
try:
    from apt.apps.webui import *
except ImportError as e:
    print(f"⚠️  Error importing WebUI: {e}")
    print("   WebUI functionality is in apt.apps.webui")
