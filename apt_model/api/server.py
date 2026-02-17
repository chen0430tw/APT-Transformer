#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apt_model.api.server - APT API Server entry point

REST API server for APT Model inference.

Usage:
    python -m apt_model.api.server [options]
    python -m apt_model.api.server --checkpoint-dir ./checkpoints
"""

import sys


def main():
    """Main entry point for API server"""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='APT API Server')
        parser.add_argument('--checkpoint-dir', type=str, default=None,
                          help='Directory containing model checkpoints')
        parser.add_argument('--host', type=str, default='0.0.0.0',
                          help='Host to bind to (default: 0.0.0.0)')
        parser.add_argument('--port', type=int, default=8000,
                          help='Port to run server on (default: 8000)')
        parser.add_argument('--reload', action='store_true',
                          help='Enable auto-reload on code changes')
        parser.add_argument('--api-key', type=str, default=None,
                          help='API key for authentication (auto-generated if not provided)')

        args = parser.parse_args()

        # Import and run API server
        from apt.apps.api.server import run_server

        return run_server(
            checkpoint_dir=args.checkpoint_dir,
            host=args.host,
            port=args.port,
            reload=args.reload,
            api_key=args.api_key
        )

    except ImportError as e:
        print("🚀 APT API Server")
        print()
        print(f"❌ Error: Could not import API server: {e}")
        print()
        print("API dependencies may not be installed:")
        print("    pip install fastapi uvicorn")
        print()
        print("Or check apt/apps/api/ for implementation status")
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
