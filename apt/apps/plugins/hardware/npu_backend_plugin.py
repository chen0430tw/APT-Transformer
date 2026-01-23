"""
NPU后端适配器 - 统一GPU/NPU/XPU/CPU接口

支持的加速器：
- 🟢 NVIDIA CUDA GPU (cuda) - torch.cuda
- 🟡 华为昇腾NPU (npu) - torch_npu
- 🔵 Intel XPU/Ultra NPU (xpu) - intel_extension_for_pytorch
- 🟣 Intel Habana Gaudi (hpu) - habana_frameworks.torch
- 🟠 AMD ROCm GPU (cuda) - torch.cuda with ROCm
- ⚪ CPU Fallback (cpu)

作者: claude + chen0430tw
版本: 2.0 (Multi-Vendor Accelerator Support)
"""

import torch
from typing import Optional, Dict, List, Any, Tuple
import warnings


class DeviceBackend:
    """统一设备后端接口"""

    def __init__(self, device: torch.device):
        """
        初始化设备后端

        Args:
            device: PyTorch设备对象
        """
        self.device = device
        self.device_type = device.type
        self.device_index = device.index if device.index is not None else 0

        # 检测各类加速器支持
        self._detect_accelerators()

        # 验证设备类型
        self._validate_device()

    def _detect_accelerators(self):
        """检测所有可用的加速器"""
        # 华为昇腾 NPU (Ascend)
        self.npu_available = False
        self.torch_npu = None
        try:
            import torch_npu
            self.torch_npu = torch_npu
            self.npu_available = torch_npu.npu.is_available()
        except ImportError:
            pass

        # Intel Habana Gaudi HPU
        self.hpu_available = False
        self.habana_torch = None
        try:
            import habana_frameworks.torch as habana_torch
            self.habana_torch = habana_torch
            self.hpu_available = hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available()
        except ImportError:
            pass

        # Intel XPU (包括Ultra NPU)
        self.xpu_available = False
        self.ipex = None
        try:
            import intel_extension_for_pytorch as ipex
            self.ipex = ipex
            self.xpu_available = hasattr(ipex, 'xpu') and ipex.xpu.is_available()
        except ImportError:
            pass

        # Qualcomm (通过QNN，但PyTorch不直接支持，需要ExecuTorch)
        self.qnn_available = False
        # QNN通常用于边缘设备，不通过标准PyTorch接口

    def _validate_device(self):
        """验证请求的设备类型是否可用"""
        if self.device_type == 'npu' and not self.npu_available:
            warnings.warn("NPU device requested but torch_npu not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
        elif self.device_type == 'hpu' and not self.hpu_available:
            warnings.warn("HPU device requested but habana_frameworks.torch not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
        elif self.device_type == 'xpu' and not self.xpu_available:
            warnings.warn("XPU device requested but intel_extension_for_pytorch not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.device_type = 'cpu'

    def is_available(self) -> bool:
        """检查设备是否可用"""
        if self.device_type == 'cuda':
            return torch.cuda.is_available()
        elif self.device_type == 'npu':
            return self.npu_available
        elif self.device_type == 'hpu':
            return self.hpu_available
        elif self.device_type == 'xpu':
            return self.xpu_available
        else:  # cpu
            return True

    def device_count(self) -> int:
        """获取设备数量"""
        if self.device_type == 'cuda':
            return torch.cuda.device_count()
        elif self.device_type == 'npu' and self.npu_available:
            return self.torch_npu.npu.device_count()
        elif self.device_type == 'hpu' and self.hpu_available:
            return self.habana_torch.hpu.device_count()
        elif self.device_type == 'xpu' and self.xpu_available:
            return self.ipex.xpu.device_count()
        else:
            return 1  # CPU

    def get_device_name(self, index: int = 0) -> str:
        """获取设备名称"""
        if self.device_type == 'cuda':
            return torch.cuda.get_device_name(index)
        elif self.device_type == 'npu' and self.npu_available:
            return f"Huawei Ascend NPU {self.torch_npu.npu.get_device_name(index)}"
        elif self.device_type == 'hpu' and self.hpu_available:
            return f"Intel Habana Gaudi HPU {index}"
        elif self.device_type == 'xpu' and self.xpu_available:
            return f"Intel XPU {self.ipex.xpu.get_device_name(index)}"
        else:
            return "CPU"

    def get_device_properties(self, index: int = 0) -> Dict[str, Any]:
        """获取设备属性"""
        props = {
            'name': self.get_device_name(index),
            'type': self.device_type,
            'total_memory': 0,
            'multi_processor_count': 0,
        }

        if self.device_type == 'cuda':
            cuda_props = torch.cuda.get_device_properties(index)
            props['total_memory'] = cuda_props.total_memory
            props['multi_processor_count'] = cuda_props.multi_processor_count
            props['major'] = cuda_props.major
            props['minor'] = cuda_props.minor
        elif self.device_type == 'npu' and self.npu_available:
            # Huawei Ascend NPU
            try:
                props['total_memory'] = self.torch_npu.npu.get_device_properties(index).total_memory
            except AttributeError:
                props['total_memory'] = 32 * 1024**3  # Ascend 910B: 32GB HBM
        elif self.device_type == 'hpu' and self.hpu_available:
            # Intel Habana Gaudi
            try:
                props['total_memory'] = self.habana_torch.hpu.get_device_properties(index).total_memory
            except AttributeError:
                props['total_memory'] = 96 * 1024**3  # Gaudi2: 96GB HBM2E
        elif self.device_type == 'xpu' and self.xpu_available:
            # Intel XPU
            try:
                xpu_props = self.ipex.xpu.get_device_properties(index)
                props['total_memory'] = xpu_props.total_memory if hasattr(xpu_props, 'total_memory') else 0
            except AttributeError:
                props['total_memory'] = 8 * 1024**3  # 默认8GB

        return props

    def memory_allocated(self, index: Optional[int] = None) -> int:
        """获取已分配内存（字节）"""
        if index is None:
            index = self.device_index

        if self.device_type == 'cuda':
            return torch.cuda.memory_allocated(index)
        elif self.device_type == 'npu' and self.npu_available:
            return self.torch_npu.npu.memory_allocated(index)
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                return self.habana_torch.hpu.memory_allocated(index)
            except AttributeError:
                return 0
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                return self.ipex.xpu.memory_allocated(index)
            except AttributeError:
                return 0
        else:
            return 0

    def memory_reserved(self, index: Optional[int] = None) -> int:
        """获取已保留内存（字节）"""
        if index is None:
            index = self.device_index

        if self.device_type == 'cuda':
            return torch.cuda.memory_reserved(index)
        elif self.device_type == 'npu' and self.npu_available:
            return self.torch_npu.npu.memory_reserved(index)
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                return self.habana_torch.hpu.memory_reserved(index)
            except AttributeError:
                return 0
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                return self.ipex.xpu.memory_reserved(index)
            except AttributeError:
                return 0
        else:
            return 0

    def max_memory_allocated(self, index: Optional[int] = None) -> int:
        """获取峰值已分配内存（字节）"""
        if index is None:
            index = self.device_index

        if self.device_type == 'cuda':
            return torch.cuda.max_memory_allocated(index)
        elif self.device_type == 'npu' and self.npu_available:
            return self.torch_npu.npu.max_memory_allocated(index)
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                return self.habana_torch.hpu.max_memory_allocated(index)
            except AttributeError:
                return 0
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                return self.ipex.xpu.max_memory_allocated(index)
            except AttributeError:
                return 0
        else:
            return 0

    def empty_cache(self) -> None:
        """清空缓存"""
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device_type == 'npu' and self.npu_available:
            self.torch_npu.npu.empty_cache()
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                self.habana_torch.hpu.empty_cache()
            except AttributeError:
                pass
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                self.ipex.xpu.empty_cache()
            except AttributeError:
                pass

    def synchronize(self, index: Optional[int] = None) -> None:
        """同步设备"""
        if index is None:
            index = self.device_index

        if self.device_type == 'cuda':
            torch.cuda.synchronize(index)
        elif self.device_type == 'npu' and self.npu_available:
            self.torch_npu.npu.synchronize(index)
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                self.habana_torch.hpu.synchronize(index)
            except AttributeError:
                pass
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                self.ipex.xpu.synchronize(index)
            except AttributeError:
                pass

    def set_device(self, index: int) -> None:
        """设置当前设备"""
        if self.device_type == 'cuda':
            torch.cuda.set_device(index)
        elif self.device_type == 'npu' and self.npu_available:
            self.torch_npu.npu.set_device(index)
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                self.habana_torch.hpu.set_device(index)
            except AttributeError:
                pass
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                self.ipex.xpu.set_device(index)
            except AttributeError:
                pass

        self.device_index = index

    def manual_seed_all(self, seed: int) -> None:
        """设置所有设备随机种子"""
        if self.device_type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        elif self.device_type == 'npu' and self.npu_available:
            self.torch_npu.npu.manual_seed_all(seed)
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                self.habana_torch.hpu.manual_seed_all(seed)
            except AttributeError:
                pass
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                self.ipex.xpu.manual_seed_all(seed)
            except AttributeError:
                pass

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将tensor移动到设备"""
        return tensor.to(self.device)

    def stream_context(self):
        """获取设备流上下文"""
        if self.device_type == 'cuda':
            return torch.cuda.stream(torch.cuda.Stream())
        elif self.device_type == 'npu' and self.npu_available:
            return self.torch_npu.npu.stream(self.torch_npu.npu.Stream())
        elif self.device_type == 'hpu' and self.hpu_available:
            try:
                return self.habana_torch.hpu.stream(self.habana_torch.hpu.Stream())
            except AttributeError:
                from contextlib import nullcontext
                return nullcontext()
        elif self.device_type == 'xpu' and self.xpu_available:
            try:
                return self.ipex.xpu.stream(self.ipex.xpu.Stream())
            except AttributeError:
                from contextlib import nullcontext
                return nullcontext()
        else:
            # CPU等其他设备没有stream概念，返回空上下文
            from contextlib import nullcontext
            return nullcontext()

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        summary = {
            'device_type': self.device_type,
            'device_name': self.get_device_name(self.device_index),
            'allocated_mb': self.memory_allocated() / (1024**2),
            'reserved_mb': self.memory_reserved() / (1024**2),
            'max_allocated_mb': self.max_memory_allocated() / (1024**2),
        }

        props = self.get_device_properties(self.device_index)
        if props['total_memory'] > 0:
            summary['total_mb'] = props['total_memory'] / (1024**2)
            summary['utilization_pct'] = (self.memory_allocated() / props['total_memory']) * 100

        return summary

    def __repr__(self) -> str:
        return f"DeviceBackend(device={self.device}, type={self.device_type})"


class UnifiedDeviceManager:
    """统一设备管理器 - 跨GPU/NPU/CPU"""

    def __init__(self):
        self.backends: Dict[str, DeviceBackend] = {}
        self._detect_devices()

    def _detect_devices(self):
        """检测所有可用设备"""
        # 检测CUDA (NVIDIA GPU / AMD ROCm)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                self.backends[f'cuda:{i}'] = DeviceBackend(device)

        # 检测华为昇腾NPU
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                for i in range(torch_npu.npu.device_count()):
                    device = torch.device(f'npu:{i}')
                    self.backends[f'npu:{i}'] = DeviceBackend(device)
        except ImportError:
            pass

        # 检测Intel Habana Gaudi HPU
        try:
            import habana_frameworks.torch as habana_torch
            if hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available():
                for i in range(habana_torch.hpu.device_count()):
                    device = torch.device(f'hpu:{i}')
                    self.backends[f'hpu:{i}'] = DeviceBackend(device)
        except ImportError:
            pass

        # 检测Intel XPU (包括Ultra NPU)
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                for i in range(ipex.xpu.device_count()):
                    device = torch.device(f'xpu:{i}')
                    self.backends[f'xpu:{i}'] = DeviceBackend(device)
        except ImportError:
            pass

        # CPU总是可用
        self.backends['cpu'] = DeviceBackend(torch.device('cpu'))

    def get_backend(self, device: torch.device) -> DeviceBackend:
        """获取设备后端"""
        key = str(device)
        if key not in self.backends:
            # 创建新后端
            self.backends[key] = DeviceBackend(device)
        return self.backends[key]

    def get_best_device(self, prefer_npu: bool = False, prefer_hpu: bool = False, prefer_xpu: bool = False) -> torch.device:
        """
        获取最佳设备

        Args:
            prefer_npu: 是否优先使用华为昇腾NPU
            prefer_hpu: 是否优先使用Intel Habana Gaudi HPU
            prefer_xpu: 是否优先使用Intel XPU

        Returns:
            torch.device: 最佳可用设备
        """
        # 用户指定优先级
        if prefer_hpu:
            hpu_devices = [k for k in self.backends.keys() if k.startswith('hpu')]
            if hpu_devices:
                return torch.device(hpu_devices[0])

        if prefer_npu:
            npu_devices = [k for k in self.backends.keys() if k.startswith('npu')]
            if npu_devices:
                return torch.device(npu_devices[0])

        if prefer_xpu:
            xpu_devices = [k for k in self.backends.keys() if k.startswith('xpu')]
            if xpu_devices:
                return torch.device(xpu_devices[0])

        # 默认优先级: CUDA > HPU > NPU > XPU > CPU
        cuda_devices = [k for k in self.backends.keys() if k.startswith('cuda')]
        if cuda_devices:
            return torch.device(cuda_devices[0])

        hpu_devices = [k for k in self.backends.keys() if k.startswith('hpu')]
        if hpu_devices:
            return torch.device(hpu_devices[0])

        npu_devices = [k for k in self.backends.keys() if k.startswith('npu')]
        if npu_devices:
            return torch.device(npu_devices[0])

        xpu_devices = [k for k in self.backends.keys() if k.startswith('xpu')]
        if xpu_devices:
            return torch.device(xpu_devices[0])

        # 最后CPU
        return torch.device('cpu')

    def get_all_devices(self) -> List[torch.device]:
        """获取所有可用设备"""
        return [torch.device(k) for k in self.backends.keys()]

    def get_device_summary(self) -> Dict[str, Any]:
        """获取所有设备摘要"""
        summary = {
            'total_devices': len(self.backends),
            'cuda_devices': sum(1 for k in self.backends if k.startswith('cuda')),
            'npu_devices': sum(1 for k in self.backends if k.startswith('npu')),
            'hpu_devices': sum(1 for k in self.backends if k.startswith('hpu')),
            'xpu_devices': sum(1 for k in self.backends if k.startswith('xpu')),
            'devices': {}
        }

        for name, backend in self.backends.items():
            if backend.device_type != 'cpu':
                summary['devices'][name] = backend.get_memory_summary()

        return summary

    def cleanup_all(self):
        """清理所有设备缓存"""
        for backend in self.backends.values():
            backend.empty_cache()

    def __repr__(self) -> str:
        return f"UnifiedDeviceManager(devices={list(self.backends.keys())})"


# 全局设备管理器实例
_global_device_manager = None


def get_device_manager() -> UnifiedDeviceManager:
    """获取全局设备管理器"""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = UnifiedDeviceManager()
    return _global_device_manager


def get_unified_backend(device: Optional[torch.device] = None) -> DeviceBackend:
    """
    获取统一设备后端

    Args:
        device: 设备对象，如果为None则使用最佳设备

    Returns:
        DeviceBackend: 设备后端
    """
    manager = get_device_manager()
    if device is None:
        device = manager.get_best_device()
    return manager.get_backend(device)


# 便捷函数
def is_cuda_available() -> bool:
    """检查CUDA是否可用（NVIDIA GPU / AMD ROCm）"""
    return torch.cuda.is_available()


def is_npu_available() -> bool:
    """检查华为昇腾NPU是否可用"""
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except ImportError:
        return False


def is_hpu_available() -> bool:
    """检查Intel Habana Gaudi HPU是否可用"""
    try:
        import habana_frameworks.torch as habana_torch
        return hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available()
    except ImportError:
        return False


def is_xpu_available() -> bool:
    """检查Intel XPU (包括Ultra NPU)是否可用"""
    try:
        import intel_extension_for_pytorch as ipex
        return hasattr(ipex, 'xpu') and ipex.xpu.is_available()
    except ImportError:
        return False


def get_accelerator_type() -> str:
    """
    获取当前最佳加速器类型

    优先级: cuda > hpu > npu > xpu > cpu

    Returns:
        str: 'cuda', 'hpu', 'npu', 'xpu', 或 'cpu'
    """
    if is_cuda_available():
        return 'cuda'
    elif is_hpu_available():
        return 'hpu'
    elif is_npu_available():
        return 'npu'
    elif is_xpu_available():
        return 'xpu'
    else:
        return 'cpu'


def get_all_accelerator_types() -> List[str]:
    """
    获取所有可用的加速器类型列表

    Returns:
        List[str]: 可用加速器类型列表
    """
    available = []
    if is_cuda_available():
        available.append('cuda')
    if is_hpu_available():
        available.append('hpu')
    if is_npu_available():
        available.append('npu')
    if is_xpu_available():
        available.append('xpu')
    available.append('cpu')  # CPU总是可用
    return available


__all__ = [
    'DeviceBackend',
    'UnifiedDeviceManager',
    'get_device_manager',
    'get_unified_backend',
    # 设备检测
    'is_cuda_available',
    'is_npu_available',
    'is_hpu_available',
    'is_xpu_available',
    # 工具函数
    'get_accelerator_type',
    'get_all_accelerator_types',
]
