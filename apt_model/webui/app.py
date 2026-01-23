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
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='APT WebUI')
        parser.add_argument('--checkpoint-dir', type=str, default=None,
                          help='Directory containing model checkpoints')
        parser.add_argument('--share', action='store_true',
                          help='Create public share link')
        parser.add_argument('--port', type=int, default=7860,
                          help='Port to run server on (default: 7860)')
        parser.add_argument('--host', type=str, default='0.0.0.0',
                          help='Host to bind to (default: 0.0.0.0)')
        parser.add_argument('--username', type=str, default=None,
                          help='Username for authentication (optional)')
        parser.add_argument('--password', type=str, default=None,
                          help='Password for authentication (optional)')

        args = parser.parse_args()

        # Prepare auth tuple if provided
        auth = None
        if args.username and args.password:
            auth = (args.username, args.password)

        # Import and launch WebUI
        from apt.apps.webui.app import launch_webui

        return launch_webui(
            checkpoint_dir=args.checkpoint_dir,
            share=args.share,
            server_port=args.port,
            server_name=args.host,
            auth=auth
        )

    except ImportError as e:
        print("🌐 APT WebUI")
        print()
        print(f"❌ Error: Could not import WebUI: {e}")
        print()
        print("WebUI dependencies may not be installed:")
        print("    pip install gradio fastapi uvicorn")
        print()
        print("Or check apt/apps/webui/ for implementation status")
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
