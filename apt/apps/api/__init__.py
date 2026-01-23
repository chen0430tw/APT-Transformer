"""
APT Model REST API

FastAPI-based REST API for:
- Model inference (single and batch)
- Training monitoring
- Checkpoint management
"""

from .server import create_app, run_server

__all__ = ['create_app', 'run_server']
