#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向后兼容代理模块 - apt.apt_model

此模块已重构为新架构：
- modeling -> apt.model.architectures
- training -> apt.trainops.engine

警告: 此导入路径已废弃，请更新代码使用新路径

将在PR-5中完善
"""

import warnings

warnings.warn(
    "apt.apt_model is deprecated and will be removed in version 3.0. "
    "Please migrate to the new architecture: "
    "apt.model for models, apt.trainops for training.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []
