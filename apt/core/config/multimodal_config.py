### `apt_model/config/multimodal_config.py`
from dataclasses import dataclass

@dataclass
class MultimodalConfig:
    """多模态配置类"""
    
    enable_image: bool = False
    enable_audio: bool = False
    image_size: int = 224
    patch_size: int = 16
    audio_sample_rate: int = 16000
    max_audio_length: int = 10
    modality_dropout: float = 0.1
    
    def __post_init__(self):
        """初始化后处理"""
        # 检查是否至少启用了一个额外模态
        if not (self.enable_image or self.enable_audio):
            print("警告: 未启用任何额外模态，将使用纯文本模式")
        
        # 计算图像块数
        if self.enable_image:
            self.num_patches = (self.image_size // self.patch_size) ** 2
        else:
            self.num_patches = 0
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'enable_image': self.enable_image,
            'enable_audio': self.enable_audio,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'audio_sample_rate': self.audio_sample_rate,
            'max_audio_length': self.max_audio_length,
            'modality_dropout': self.modality_dropout,
            'num_patches': self.num_patches
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k != 'num_patches'})
    
    def get_enabled_modalities(self):
        """获取已启用的模态列表"""
        modalities = ['text']  # 文本模态始终启用
        if self.enable_image:
            modalities.append('image')
        if self.enable_audio:
            modalities.append('audio')
        return modalities