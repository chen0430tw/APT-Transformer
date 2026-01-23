### `apt_model/config/hardware_profile.py`
from dataclasses import dataclass

# 尝试导入torch，如果失败则使用fake_torch
try:
    import torch
except ImportError:
    try:
        from apt_model.utils.fake_torch import get_torch
        torch = get_torch()
    except ImportError:
        # 如果都失败，创建一个简单的mock
        class MockTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return False
                @staticmethod
                def device_count():
                    return 0
        torch = MockTorch()

try:
    import psutil
except ImportError:
    psutil = None

@dataclass
class HardwareProfile:
    """硬件配置信息"""
    
    gpu_name: str = "Unknown"
    gpu_vram_gb: float = 0.0
    gpu_count: int = 0
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    ram_gb: float = 0.0
    disk_speed_mbps: float = 0.0
    network_speed_mbps: float = 0.0
    tflops: float = 0.0  # GPU理论性能，单位：TFLOPS
    
    @classmethod
    def detect_current(cls):
        """检测当前硬件配置"""
        profile = cls()
        
        try:
            # 检测GPU信息
            if torch.cuda.is_available():
                profile.gpu_count = torch.cuda.device_count()
                if profile.gpu_count > 0:
                    profile.gpu_name = torch.cuda.get_device_name(0)
                    profile.gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # 检测CPU和内存信息
            if psutil:
                profile.cpu_cores = psutil.cpu_count(logical=True)
                profile.ram_gb = psutil.virtual_memory().total / (1024**3)
            
            # 估算其他参数（这里略去具体实现）
        except Exception as e:
            print(f"检测硬件配置时出错: {e}")
        
        return profile