#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向后兼容代理模块

此模块已迁移至 apt.apps.console
为保持向后兼容性，此代理模块会重导出新位置的所有内容

警告: 此导入路径已废弃，请更新代码使用新路径：
    from apt.apps.console import ...
"""

import warnings

warnings.warn(
    f"apt_model.console is deprecated and will be removed in a future version. "
    f"Please use apt.apps.console instead.",
    DeprecationWarning,
    stacklevel=2
)

# 重导出新位置的所有内容
from apt.apps.console import *  # noqa: F401, F403

try:
    from apt.apps.console import __all__
except ImportError:
    pass
