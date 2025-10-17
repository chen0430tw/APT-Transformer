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
from apt_model.modeling.multimodal_model import MultimodalAPTModel
from apt_model.training.trainer import train_model

# 设置默认设备
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")