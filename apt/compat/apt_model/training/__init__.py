#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向后兼容代理 - apt.apt_model.training

⚠️  此模块已废弃，将在APT 3.0中移除

旧路径: from apt.apt_model.training import Trainer
新路径: from apt.trainops.engine import Trainer

此兼容层提供6个月的迁移期（至2026-07-22）
"""

import warnings

warnings.warn(
    "apt.apt_model.training is deprecated and will be removed in version 3.0. "
    "Please update your imports:\n"
    "  OLD: from apt.apt_model.training import Trainer\n"
    "  NEW: from apt.trainops.engine import Trainer\n"
    "Migration guide: https://apt-transformer.readthedocs.io/migration/2.0/",
    DeprecationWarning,
    stacklevel=2
)

# 重导出所有训练组件（保持向后兼容）
try:
    from apt.trainops.engine import *  # noqa: F401, F403
except ImportError:
try:
    from apt.trainops.data import *  # noqa: F401, F403
except ImportError:
try:
    from apt.trainops.checkpoints import *  # noqa: F401, F403
except ImportError:
try:
    from apt.trainops.eval import *  # noqa: F401, F403
except ImportError:

# 导出__all__
try:
    try:
        from apt.trainops.engine import __all__ as _engine_all
    except ImportError:
        _engine_all = None
    try:
        from apt.trainops.data import __all__ as _data_all
    except ImportError:
        _data_all = None
    try:
        from apt.trainops.checkpoints import __all__ as _ckpt_all
    except ImportError:
        _ckpt_all = None
    try:
        from apt.trainops.eval import __all__ as _eval_all
    except ImportError:
        _eval_all = None
    __all__ = _engine_all + _data_all + _ckpt_all + _eval_all
except ImportError:
    __all__ = []
