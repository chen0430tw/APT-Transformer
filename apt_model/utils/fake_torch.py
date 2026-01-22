#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fake torch module for testing and command-line help without installing PyTorch.
当torch未安装时提供基本接口，允许CLI命令(如help)正常运行。
"""


class FakeCuda:
    """Fake CUDA interface"""

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def get_device_name(index):
        return "No GPU"

    @staticmethod
    def get_device_properties(index):
        class Props:
            total_memory = 0
            major = 0
            minor = 0
            multi_processor_count = 0
            max_threads_per_block = 1024
            clock_rate = 0
        return Props()


class FakeVersion:
    """Fake version info"""
    cuda = None


class FakeDevice:
    """Fake device class"""
    def __init__(self, device_type="cpu"):
        self.type = device_type

    def __str__(self):
        return self.type

    def __repr__(self):
        return f"device(type='{self.type}')"


class FakeTorch:
    """Fake torch module for basic operations"""

    __version__ = "fake-0.0.0 (torch not installed)"

    cuda = FakeCuda()
    version = FakeVersion()

    @staticmethod
    def device(device_type="cpu"):
        return FakeDevice(device_type)

    @staticmethod
    def manual_seed(seed):
        pass

    @staticmethod
    def set_grad_enabled(mode):
        pass

    @staticmethod
    def no_grad():
        class NoGrad:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoGrad()

    class nn:
        """Fake nn module"""
        class Module:
            def __init__(self):
                pass

            def to(self, device):
                return self

            def eval(self):
                return self

            def train(self, mode=True):
                return self

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

    class optim:
        """Fake optim module"""
        class Adam:
            def __init__(self, *args, **kwargs):
                pass

            def zero_grad(self):
                pass

            def step(self):
                pass

        class AdamW:
            def __init__(self, *args, **kwargs):
                pass

            def zero_grad(self):
                pass

            def step(self):
                pass

    class Tensor:
        """Fake tensor class"""
        def __init__(self, data=None):
            self.data = data

        def to(self, device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return []

        def item(self):
            return 0.0

    @staticmethod
    def tensor(data):
        return FakeTorch.Tensor(data)

    @staticmethod
    def zeros(*args, **kwargs):
        return FakeTorch.Tensor()

    @staticmethod
    def ones(*args, **kwargs):
        return FakeTorch.Tensor()

    @staticmethod
    def randn(*args, **kwargs):
        return FakeTorch.Tensor()


def get_torch():
    """
    尝试导入真实的torch，失败则返回fake torch

    Returns:
        torch module or FakeTorch
    """
    try:
        import torch
        return torch
    except ImportError:
        return FakeTorch()


# 默认导出fake torch
torch = FakeTorch()
