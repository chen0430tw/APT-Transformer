#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向后兼容代理 - apt.apt_model.modeling

此模块已迁移至 apt.model.architectures

警告: 请更新代码使用新路径：
    from apt.model.architectures import APTLargeModel

将在PR-5中添加完整的重导出
"""

import warnings

warnings.warn(
    "apt.apt_model.modeling is deprecated. Use apt.model.architectures instead.",
    DeprecationWarning,
    stacklevel=2
)

# 将在PR-4完成后添加重导出
# from apt.model.architectures import *

__all__ = []
