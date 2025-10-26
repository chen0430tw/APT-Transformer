#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Plugins for APT Console

This directory contains example plugins demonstrating the plugin system:
- grpo_plugin.py - Training tier plugin for GRPO (Group Relative Policy Optimization)
- eqi_reporter_plugin.py - Telemetry tier plugin for EQI metrics reporting
- route_optimizer_plugin.py - Performance tier plugin for route optimization
"""

from apt_model.console.plugins.grpo_plugin import GRPOPlugin
from apt_model.console.plugins.eqi_reporter_plugin import EQIReporterPlugin
from apt_model.console.plugins.route_optimizer_plugin import RouteOptimizerPlugin

__all__ = [
    'GRPOPlugin',
    'EQIReporterPlugin',
    'RouteOptimizerPlugin',
]
