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
        """
        Context manager/decorator for disabling gradient computation.
        Returns a callable context manager that can be used as:
        - Context manager: with torch.no_grad():
        - Decorator: @torch.no_grad()
        """
        class NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def __call__(self, func):
                """Allow using as a decorator"""
                def wrapper(*args, **kwargs):
                    with self:
                        return func(*args, **kwargs)
                return wrapper

        return NoGrad()

    class nn:
        """Fake nn module"""
        class Module:
            def __init__(self):
                self._parameters = []
                self._modules = []

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

            def parameters(self):
                """Return fake parameters"""
                # 返回一些fake参数供optimizer使用
                if not self._parameters:
                    # 创建一些fake参数
                    self._parameters = [FakeTorch.Tensor(shape=(10, 10)) for _ in range(2)]
                return iter(self._parameters)

            def named_parameters(self):
                """Return named fake parameters"""
                for i, p in enumerate(self.parameters()):
                    yield f"param_{i}", p

            def __call__(self, *args, **kwargs):
                """Make module callable"""
                return self.forward(*args, **kwargs)

            def forward(self, x):
                """Default forward pass - override in subclasses"""
                return x

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
                # 返回标量loss
                return FakeTorch.Tensor(shape=())

            @staticmethod
            def mse_loss(input, target, reduction='mean'):
                # 返回标量loss
                return FakeTorch.Tensor(shape=())

            @staticmethod
            def binary_cross_entropy(input, target, weight=None, reduction='mean'):
                # 返回标量loss
                return FakeTorch.Tensor(shape=())

            @staticmethod
            def pad(input, pad, mode='constant', value=0):
                return input

            @staticmethod
            def interpolate(input, size=None, scale_factor=None, mode='nearest'):
                return input

            @staticmethod
            def normalize(input, p=2, dim=1, eps=1e-12):
                return input

        # Common nn layers
        class Linear(Module):
            """Fake Linear layer with shape tracking"""
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.bias = bias
                # 创建fake weight和bias
                self.weight = FakeTorch.Tensor(shape=(out_features, in_features))
                if bias:
                    self.bias_param = FakeTorch.Tensor(shape=(out_features,))

            def forward(self, x):
                """Fake forward pass with shape tracking"""
                # 计算输出形状
                if hasattr(x, 'shape') and len(x.shape) > 0:
                    # 输入形状: (..., in_features)
                    # 输出形状: (..., out_features)
                    batch_dims = x.shape[:-1]
                    output_shape = (*batch_dims, self.out_features)
                else:
                    output_shape = (self.out_features,)

                return FakeTorch.Tensor(shape=output_shape)

        class LayerNorm(Module):
            """Fake LayerNorm layer"""
            def __init__(self, normalized_shape, eps=1e-5):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps

            def forward(self, x):
                """Fake forward - returns input with same shape"""
                return x

        class Embedding(Module):
            """Fake Embedding layer"""
            def __init__(self, num_embeddings, embedding_dim):
                super().__init__()
                self.num_embeddings = num_embeddings
                self.embedding_dim = embedding_dim
                self.weight = FakeTorch.Tensor(shape=(num_embeddings, embedding_dim))

            def forward(self, x):
                """Fake forward - returns embeddings"""
                if hasattr(x, 'shape'):
                    # 输入: (..., seq_len)
                    # 输出: (..., seq_len, embedding_dim)
                    output_shape = (*x.shape, self.embedding_dim)
                else:
                    output_shape = (self.embedding_dim,)
                return FakeTorch.Tensor(shape=output_shape)

        class Dropout(Module):
            """Fake Dropout layer"""
            def __init__(self, p=0.5):
                super().__init__()
                self.p = p

            def forward(self, x):
                """Fake forward - returns input unchanged"""
                return x

        class MultiheadAttention(Module):
            """Fake MultiheadAttention layer"""
            def __init__(self, embed_dim, num_heads, *args, **kwargs):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads

            def forward(self, query, key=None, value=None, **kwargs):
                """Fake forward - returns query with same shape"""
                # MultiheadAttention通常返回 (output, attention_weights)
                return query, None

    class optim:
        """Fake optim module"""
        class Optimizer:
            """Fake base Optimizer class"""
            def __init__(self, *args, **kwargs):
                pass

            def zero_grad(self):
                pass

            def step(self):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

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

    class utils:
        """Fake utils module"""
        class data:
            """Fake data module"""
            class Dataset:
                """Fake Dataset class"""
                def __init__(self):
                    pass

                def __len__(self):
                    return 0

                def __getitem__(self, idx):
                    return None

            class DataLoader:
                """Fake DataLoader class"""
                def __init__(self, *args, **kwargs):
                    pass

                def __iter__(self):
                    return iter([])

                def __len__(self):
                    return 0

    class Tensor:
        """Fake tensor class with shape tracking"""
        def __init__(self, data=None, shape=None):
            self.data = data
            self.shape = shape or ()
            self.requires_grad = False
            self.grad = None

        def to(self, device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return []

        def item(self):
            # 返回虚假的随机值，模拟真实loss
            import random
            return random.uniform(0.5, 7.0)

        def backward(self):
            """Fake backward pass"""
            pass

        def size(self, dim=None):
            if dim is None:
                return self.shape
            return self.shape[dim] if dim < len(self.shape) else 1

        def __repr__(self):
            return f"FakeTensor(shape={self.shape})"

    @staticmethod
    def tensor(data, **kwargs):
        # 尝试推断shape
        if hasattr(data, '__len__'):
            try:
                shape = (len(data),)
            except:
                shape = ()
        else:
            shape = ()
        return FakeTorch.Tensor(data, shape=shape)

    @staticmethod
    def zeros(*args, **kwargs):
        """Create fake zeros tensor with shape"""
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def ones(*args, **kwargs):
        """Create fake ones tensor with shape"""
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def randn(*args, **kwargs):
        """Create fake random tensor with shape"""
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def rand(*args, **kwargs):
        """Create fake random tensor with shape"""
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def randint(low, high=None, *size, **kwargs):
        """Create fake random integer tensor"""
        if high is None:
            high = low
            low = 0
        shape = size if size else ()
        return FakeTorch.Tensor(shape=shape)


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
