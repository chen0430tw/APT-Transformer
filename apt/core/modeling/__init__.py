#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Modeling Module

此模块从 apt_model.modeling 导入所有模型。
apt_model/ 包含APT的模型实现，apt/ 是框架层。
"""

# 重导出apt_model的所有模型
from apt.apt_model.modeling import *  # noqa: F401, F403

try:
    from apt.apt_model.modeling import __all__
except ImportError:
    pass
