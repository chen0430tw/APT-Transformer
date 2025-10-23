### `apt_model/__init__.py`
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) 训练工具
一个功能丰富的模型训练和评估工具
"""

__version__ = "0.1.0"
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
        from apt_model.config.apt_config import APTConfig as _APTConfig

        return _APTConfig

    if name == "MultimodalConfig":
        from apt_model.config.multimodal_config import MultimodalConfig as _MultimodalConfig

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
