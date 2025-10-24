### `apt_model/__init__.py`
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) 训练工具
一个功能丰富的模型训练和评估工具
"""

__version__ = "0.1.0"
__author__ = "APT Team"

from typing import TYPE_CHECKING, Any

from apt_model.config.apt_config import APTConfig
from apt_model.config.multimodal_config import MultimodalConfig
from apt_model.modeling.apt_model import APTLargeModel
from apt_model.modeling.multimodal_model import MultimodalAPTModel

if TYPE_CHECKING:
    # Used only for static analysis; avoids importing heavy optional dependencies at runtime.
    from apt_model.training.trainer import train_model as _train_model


def __getattr__(name: str) -> Any:
    """Lazily import modules that require optional third-party dependencies."""

    if name == "train_model":
        from apt_model.training.trainer import train_model as _train_model_runtime

        return _train_model_runtime
    raise AttributeError(f"module 'apt_model' has no attribute {name!r}")


# 设置默认设备
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = [
    "APTConfig",
    "MultimodalConfig",
    "APTLargeModel",
    "MultimodalAPTModel",
    "device",
    "train_model",
]
