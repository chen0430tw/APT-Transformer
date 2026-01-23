#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Compatibility Layer

向后兼容层：支持旧版导入路径

此模块提供代理以支持从apt.apt_model.*迁移到新架构：
- apt.apt_model.modeling -> apt.model.architectures
- apt.apt_model.training -> apt.trainops.engine

使用说明：
1. 旧代码继续使用 `from apt.apt_model.modeling import APTLargeModel`
2. 新代码使用 `from apt.model.architectures import APTLargeModel`
3. 所有旧导入会触发DeprecationWarning

计划6个月后移除此兼容层
"""

__version__ = '2.0.0-alpha'

__all__ = []
