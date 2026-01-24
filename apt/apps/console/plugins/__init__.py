#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Plugins for APT Console

This directory contains example plugins demonstrating the plugin system:
- grpo_plugin.py - Training tier plugin for GRPO (Group Relative Policy Optimization)
- eqi_reporter_plugin.py - Telemetry tier plugin for EQI metrics reporting
- route_optimizer_plugin.py - Performance tier plugin for route optimization
"""

try:
    from apt.apps.console.plugins.grpo_plugin import GRPOPlugin
except ImportError:
    pass
try:
    from apt.apps.console.plugins.eqi_reporter_plugin import EQIReporterPlugin
except ImportError:
    pass
try:
    from apt.apps.console.plugins.route_optimizer_plugin import RouteOptimizerPlugin
except ImportError:
    pass

__all__ = [
    'GRPOPlugin',
    'EQIReporterPlugin',
    'RouteOptimizerPlugin',
]
