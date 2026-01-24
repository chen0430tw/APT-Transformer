#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向后兼容代理 - apt.apt_model.modeling

⚠️  此模块已废弃，将在APT 3.0中移除

旧路径: from apt.apt_model.modeling import APTLargeModel
新路径: from apt.model.architectures import APTLargeModel

此兼容层提供6个月的迁移期（至2026-07-22）
"""

import warnings

warnings.warn(
    "apt.apt_model.modeling is deprecated and will be removed in version 3.0. "
    "Please update your imports:\n"
    "  OLD: from apt.apt_model.modeling import APTLargeModel\n"
    "  NEW: from apt.model.architectures import APTLargeModel\n"
    "Migration guide: https://apt-transformer.readthedocs.io/migration/2.0/",
    DeprecationWarning,
    stacklevel=2
)

# 重导出所有模型（保持向后兼容）
try:
    from apt.model.architectures import *  # noqa: F401, F403
except ImportError:
    pass
try:
    from apt.model.layers import *  # noqa: F401, F403
except ImportError:
    pass
try:
    from apt.model.tokenization import *  # noqa: F401, F403
except ImportError:
    pass
try:
    from apt.model.extensions import *  # noqa: F401, F403
except ImportError:
    pass

# 导出__all__
try:
    from apt.model.architectures import __all__ as _arch_all
    from apt.model.layers import __all__ as _layers_all
    from apt.model.tokenization import __all__ as _token_all
    from apt.model.extensions import __all__ as _ext_all
    __all__ = _arch_all + _layers_all + _token_all + _ext_all
except ImportError:
    __all__ = []
