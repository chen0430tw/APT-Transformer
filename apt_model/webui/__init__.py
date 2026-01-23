"""
apt_model.webui - APT WebUI Package

Web interface for APT Model training and inference.

Usage:
    python -m apt_model.webui.app --checkpoint-dir ./checkpoints

This wraps apt.apps.webui functionality with a CLI interface.
"""

# Re-export from new location
try:
    from apt.apps.webui import *
except ImportError as e:
    print(f"⚠️  Error importing WebUI: {e}")
    print("   WebUI functionality is in apt.apps.webui")
