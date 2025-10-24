#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Module

Core functionality for the APT microkernel architecture:
- Provider pattern and Registry system
- Configuration management
- Curriculum scheduling
- Logging and monitoring
- Device management
"""

from apt.core.registry import (
    Provider,
    Registry,
    registry,
    register_provider,
    get_provider
)
from apt.core.config import APTConfig, MultimodalConfig, HardwareProfile
from apt.core.schedules import Schedule

__all__ = [
    'Provider',
    'Registry',
    'registry',
    'register_provider',
    'get_provider',
    'APTConfig',
    'MultimodalConfig',
    'HardwareProfile',
    'Schedule',
]
