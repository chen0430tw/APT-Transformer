### `apt_model/config/__init__.py`
# 配置模块初始化文件
try:
    from apt.core.config.apt_config import APTConfig
except ImportError:
    APTConfig = None
try:
    from apt.core.config.multimodal_config import MultimodalConfig
except ImportError:
    MultimodalConfig = None
try:
    from apt.core.config.hardware_profile import HardwareProfile
except ImportError:
    HardwareProfile = None

# APT 2.0 Profile配置系统
try:
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
except ImportError:
    load_profile = None
    list_profiles = None
    APTProfile = None
    ProfileLoader = None
    ModelConfig = None
    TrainingConfig = None
    DistributedConfig = None
    VGPUConfig = None