#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apt_model.api.server - Legacy API server entry point

⚠️ Deprecated: This is a compatibility wrapper
⚠️ Recommended: Use apt.apps.api instead

Usage:
    python -m apt_model.api.server [options]
"""

import sys
import warnings

warnings.warn(
    "apt_model.api.server is deprecated. Use apt.apps.api instead.",
    DeprecationWarning,
    stacklevel=2
)


def main():
    """Main entry point for API server"""
    print("🚀 APT API Server (Compatibility Mode)")
    print()
    print("⚠️  This is APT 1.0 compatibility wrapper")
    print("⚠️  Recommended: Use APT 2.0 API instead")
    print()

    try:
        # Try to import and run the real API server
        from apt.apps.api import server

        # Check if server module has main function
        if hasattr(server, 'main'):
            return server.main()
        elif hasattr(server, 'run'):
            return server.run()
        else:
            print("❌ Error: API server main function not found")
            print()
            print("API functionality is being migrated to APT 2.0")
            print("For now, you can:")
            print("1. Use quickstart.py for basic training")
            print("2. Check docs/README.md for API status")
            return 1

    except ImportError as e:
        print(f"❌ Error: Could not import API server: {e}")
        print()
        print("API dependencies may not be installed:")
        print("    pip install fastapi uvicorn")
        print()
        print("Or check apt/apps/api/ for implementation status")
        return 1

    except Exception as e:
        print(f"❌ Error launching API server: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
