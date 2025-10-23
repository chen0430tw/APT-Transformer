### `apt_model/__init__.py`
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) 训练工具
一个功能丰富的模型训练和评估工具
"""

__version__ = "0.1.0"
__author__ = "APT Team"

from apt_model.config.apt_config import APTConfig
from apt_model.config.multimodal_config import MultimodalConfig
from apt_model.modeling.apt_model import APTLargeModel

try:
    from apt_model.modeling.multimodal_model import MultimodalAPTModel
    _multimodal_import_error = None
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when optional deps missing
    _multimodal_import_error = exc

    class MultimodalAPTModel:  # type: ignore[misc]
        """Fallback that points users to install optional multimodal dependencies."""

        def __init__(self, *_, **__):
            raise ModuleNotFoundError(
                "MultimodalAPTModel requires optional dependencies that are not installed."
            ) from exc

try:
    from apt_model.training.trainer import train_model
    _training_import_error = None
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when optional deps missing
    _training_import_error = exc

    def train_model(*_, **__):  # type: ignore[misc]
        raise ModuleNotFoundError(
            "train_model requires optional training dependencies (e.g., scikit-learn)."
        ) from exc

# 设置默认设备
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")