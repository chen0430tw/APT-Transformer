# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) 训练工具
一个功能丰富的模型训练和评估工具

重要：此模块使用延迟导入以避免强制依赖torch。
子模块（如apt_model.tools.apx）可以独立使用而无需安装torch。
"""

__version__ = "1.0.0"
__author__ = "APT Team"

from typing import TYPE_CHECKING, Any

# 延迟导入torch依赖的模块，避免阻止独立子模块使用
if TYPE_CHECKING:
    from apt_model.config.apt_config import APTConfig
    from apt_model.config.multimodal_config import MultimodalConfig
    from apt_model.modeling.apt_model import APTLargeModel
    from apt_model.modeling.multimodal_model import MultimodalAPTModel
    from apt_model.training.trainer import train_model as _train_model
    import torch


def __getattr__(name: str) -> Any:
    """
    延迟导入需要torch的模块

    这允许apt_model的子包（如tools.apx）在不安装torch的情况下使用。
    """

    # Config模块
    if name == "APTConfig":
        from apt_model.config.apt_config import APTConfig
        return APTConfig

    if name == "MultimodalConfig":
        from apt_model.config.multimodal_config import MultimodalConfig
        return MultimodalConfig

    # Model模块
    if name == "APTLargeModel":
        from apt_model.modeling.apt_model import APTLargeModel
        return APTLargeModel

    if name == "MultimodalAPTModel":
        from apt_model.modeling.multimodal_model import MultimodalAPTModel
        return MultimodalAPTModel

    # Training模块
    if name == "train_model":
        from apt_model.training.trainer import train_model as _train_model_runtime
        return _train_model_runtime

    # Device（延迟导入torch）
    if name == "device":
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise AttributeError(f"module 'apt_model' has no attribute {name!r}")


# 导出列表（实际导入由__getattr__处理）
__all__ = [
    "APTConfig",
    "MultimodalConfig",
    "APTLargeModel",
    "MultimodalAPTModel",
    "device",
    "train_model",
]
