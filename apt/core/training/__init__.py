#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Training Module

此模块从 apt_model.training 导入所有训练功能。
apt_model/ 包含APT的训练实现，apt/ 是框架层。
"""

# 重导出apt_model的所有训练功能
from apt.apt_model.training import *  # noqa: F401, F403

try:
    from apt.apt_model.training import __all__
except ImportError:
    pass
