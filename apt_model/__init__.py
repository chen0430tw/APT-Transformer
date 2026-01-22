# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) 训练工具
一个功能丰富的模型训练和评估工具

⚠️ **DEPRECATION WARNING** ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
此 `apt_model` 包已弃用，将在未来版本中移除。

请迁移到新的 `apt` 包结构：

  旧写法: from apt_model import APTConfig
  新写法: from apt.core import APTConfig  # 或使用 apt.enable('lite')

新包结构优势：
  • 清晰的分层架构（L0/L1/L2/L3）
  • 按需加载，减少启动时间
  • 统一的 apt.enable() API
  • 禁止反向依赖，提升可维护性

详见迁移指南：docs/ARCHITECTURE.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

重要：此模块使用延迟导入以避免强制依赖torch。
子模块（如apt_model.tools.apx）可以独立使用而无需安装torch。
"""

import warnings

# 发出弃用警告
warnings.warn(
    "\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "⚠️  'apt_model' 包已弃用，请迁移到 'apt' 包\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "旧写法: from apt_model import APTConfig\n"
    "新写法: from apt.core import APTConfig\n"
    "\n"
    "或使用统一入口:\n"
    "  import apt\n"
    "  apt.enable('lite')  # 仅核心功能\n"
    "  apt.enable('full')  # 全功能\n"
    "\n"
    "详见: docs/ARCHITECTURE.md\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
    DeprecationWarning,
    stacklevel=2
)

__version__ = "1.0.0-deprecated"
__author__ = "APT Team"

__all__ = [
    "APTConfig",
    "MultimodalConfig",
    "APTLargeModel",
    "MultimodalAPTModel",
    "train_model",
    "device",
]


def __getattr__(name):
    """Lazily import heavy submodules only when requested."""

    if name == "APTConfig":
        from apt.core.config.apt_config import APTConfig as _APTConfig

        return _APTConfig

    if name == "MultimodalConfig":
        from apt.core.config.multimodal_config import MultimodalConfig as _MultimodalConfig

        return _MultimodalConfig

    if name == "APTLargeModel":
        from apt_model.modeling.apt_model import APTLargeModel as _APTLargeModel

        return _APTLargeModel

    if name == "MultimodalAPTModel":
        from apt_model.modeling.multimodal_model import (
            MultimodalAPTModel as _MultimodalAPTModel,
        )

        return _MultimodalAPTModel

    if name == "train_model":
        try:
            from apt_model.training.trainer import train_model as _train_model
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency missing
            raise ImportError(
                "train_model requires optional training dependencies to be installed"
            ) from exc

        return _train_model

    if name == "device":
        return _detect_device()

    raise AttributeError(f"module 'apt_model' has no attribute {name!r}")


def _detect_device():
    try:  # pragma: no cover - depends on optional torch installation
        import torch
    except ModuleNotFoundError:
        return "cpu"

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
