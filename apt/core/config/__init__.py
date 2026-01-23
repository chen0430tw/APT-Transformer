### `apt_model/config/__init__.py`
# 配置模块初始化文件
from apt.core.config.apt_config import APTConfig
from apt.core.config.multimodal_config import MultimodalConfig
from apt.core.config.hardware_profile import HardwareProfile

# APT 2.0 Profile配置系统
from apt.core.config.profile_loader import (
    load_profile,
    list_profiles,
    APTProfile,
    ProfileLoader,
    ModelConfig,
    TrainingConfig,
    DistributedConfig,
    VGPUConfig,
)