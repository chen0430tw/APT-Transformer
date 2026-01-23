#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apt_model.webui.app - Legacy WebUI entry point

⚠️ Deprecated: This is a compatibility wrapper
⚠️ Recommended: Use apt.apps.webui instead

Usage:
    python -m apt_model.webui.app [options]
"""

import sys
import warnings

warnings.warn(
    "apt_model.webui.app is deprecated. Use apt.apps.webui instead.",
    DeprecationWarning,
    stacklevel=2
)


def main():
    """Main entry point for WebUI"""
    print("🌐 APT WebUI (Compatibility Mode)")
    print()
    print("⚠️  This is APT 1.0 compatibility wrapper")
    print("⚠️  Recommended: Use APT 2.0 webui instead")
    print()

    try:
        # Try to import and run the real webui
        from apt.apps.webui import app

        # Check if app module has main function
        if hasattr(app, 'main'):
            return app.main()
        elif hasattr(app, 'launch'):
            return app.launch()
        else:
            print("❌ Error: WebUI main function not found")
            print()
            print("WebUI functionality is being migrated to APT 2.0")
            print("For now, you can:")
            print("1. Use quickstart.py for basic training")
            print("2. Check docs/README.md for WebUI status")
            return 1

    except ImportError as e:
        print(f"❌ Error: Could not import WebUI: {e}")
        print()
        print("WebUI dependencies may not be installed:")
        print("    pip install gradio fastapi uvicorn")
        print()
        print("Or check apt/apps/webui/ for implementation status")
        return 1

    except Exception as e:
        print(f"❌ Error launching WebUI: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
