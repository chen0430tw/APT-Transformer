#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发工具集

包含用于代码质量检查、依赖分析等开发辅助工具。
"""

try:
    from apt.core.dev_tools.dependency_checker import DependencyChecker
except ImportError:
    pass

__all__ = ['DependencyChecker']
