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


class FakeDtype:
    """Fake dtype class"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"

    def __str__(self):
        return self.name


class FakeTorch:
    """Fake torch module for basic operations"""

    __version__ = "fake-0.0.0 (torch not installed)"

    cuda = FakeCuda()
    version = FakeVersion()

    # Add dtype and device as type aliases (for type annotations)
    dtype = FakeDtype
    device = FakeDevice

    # Add dtype instances
    float32 = FakeDtype("float32")
    float16 = FakeDtype("float16")
    float64 = FakeDtype("float64")
    int32 = FakeDtype("int32")
    int64 = FakeDtype("int64")
    long = FakeDtype("long")
    bool = FakeDtype("bool")
    uint8 = FakeDtype("uint8")

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

        class functional:
            """Fake functional module"""
            @staticmethod
            def relu(x, inplace=False):
                return x

            @staticmethod
            def gelu(x):
                return x

            @staticmethod
            def softmax(x, dim=-1):
                return x

            @staticmethod
            def log_softmax(x, dim=-1):
                return x

            @staticmethod
            def dropout(x, p=0.5, training=True, inplace=False):
                return x

            @staticmethod
            def linear(input, weight, bias=None):
                return FakeTorch.Tensor()

            @staticmethod
            def embedding(input, weight, padding_idx=None):
                return FakeTorch.Tensor()

            @staticmethod
            def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
                return input

            @staticmethod
            def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean'):
                return FakeTorch.Tensor()

            @staticmethod
            def mse_loss(input, target, reduction='mean'):
                return FakeTorch.Tensor()

            @staticmethod
            def binary_cross_entropy(input, target, weight=None, reduction='mean'):
                return FakeTorch.Tensor()

            @staticmethod
            def pad(input, pad, mode='constant', value=0):
                return input

            @staticmethod
            def interpolate(input, size=None, scale_factor=None, mode='nearest'):
                return input

            @staticmethod
            def normalize(input, p=2, dim=1, eps=1e-12):
                return input

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
