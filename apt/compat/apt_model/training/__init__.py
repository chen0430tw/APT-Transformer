#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向后兼容代理 - apt.apt_model.training

此模块已迁移至 apt.trainops.engine

警告: 请更新代码使用新路径：
    from apt.trainops.engine import Trainer

将在PR-5中添加完整的重导出
"""

import warnings

warnings.warn(
    "apt.apt_model.training is deprecated. Use apt.trainops.engine instead.",
    DeprecationWarning,
    stacklevel=2
)

# 将在PR-3完成后添加重导出
# from apt.trainops.engine import *

__all__ = []
