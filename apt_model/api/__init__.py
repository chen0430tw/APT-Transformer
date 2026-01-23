"""
apt_model.api - APT API Package

REST API server for APT Model inference.

Usage:
    python -m apt_model.api.server --checkpoint-dir ./checkpoints

This wraps apt.apps.api functionality with a CLI interface.
"""

# Re-export from new location
try:
    from apt.apps.api import *
except ImportError as e:
    print(f"⚠️  Error importing API: {e}")
    print("   API functionality is in apt.apps.api")
