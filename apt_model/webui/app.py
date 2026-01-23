#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apt_model.webui.app - APT WebUI entry point

Web interface for APT Model training and inference.

Usage:
    python -m apt_model.webui.app [options]
    python -m apt_model.webui.app --checkpoint-dir ./checkpoints
"""

import sys


def main():
    """Main entry point for WebUI"""
    print("🌐 APT WebUI")
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
