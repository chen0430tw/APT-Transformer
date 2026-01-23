"""
apt_model - APT CLI Package

Command-line interface for APT Model training and inference.

This is one of the official entry points for APT (along with quickstart.py and apt.* API).

Usage:
    python -m apt_model chat        # Interactive chat
    python -m apt_model train       # Training
    python -m apt_model --help      # Show help

For Python API approach, see: quickstart.py or apt.* modules
"""

import sys
import warnings

__version__ = "2.0.0"
__all__ = []

# Deprecation warning for developer import API only
# CLI usage (python -m apt_model) remains fully supported
def _check_import_usage():
    """Show deprecation warning only for developer imports, not CLI usage"""
    # Check if we're being imported (not run as __main__)
    if __name__ != '__main__':
        # Check if this is a developer import (not just CLI subprocess)
        # We detect CLI by checking if __main__ is our own __main__.py
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_code.co_filename
            # If caller is NOT our own __main__.py, it's a developer import
            if not caller_file.endswith('apt_model/__main__.py'):
                warnings.warn(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║ Deprecation Warning: Developer Import API                   ║\n"
                    "╠══════════════════════════════════════════════════════════════╣\n"
                    "║                                                              ║\n"
                    "║ ⚠️  Deprecated: Developer import entrypoints only            ║\n"
                    "║ ✅  CLI remains supported: python -m apt_model ...           ║\n"
                    "║                                                              ║\n"
                    "║ If you're seeing this as a CLI user - ignore it.            ║\n"
                    "║ This warning is for developers who import apt_model in code.║\n"
                    "║                                                              ║\n"
                    "╠══════════════════════════════════════════════════════════════╣\n"
                    "║ Recommended migration for developers:                       ║\n"
                    "║                                                              ║\n"
                    "║   OLD (deprecated):                                          ║\n"
                    "║     from apt_model.training import Trainer                   ║\n"
                    "║     from apt_model.config import Config                      ║\n"
                    "║                                                              ║\n"
                    "║   NEW (recommended):                                         ║\n"
                    "║     from apt.core.config import load_profile                 ║\n"
                    "║     from apt.trainops.engine import Trainer                  ║\n"
                    "║                                                              ║\n"
                    "║ See: docs/ARCHITECTURE_2.0.md for migration guide            ║\n"
                    "╚══════════════════════════════════════════════════════════════╝\n",
                    DeprecationWarning,
                    stacklevel=3
                )

# Run the check
_check_import_usage()

