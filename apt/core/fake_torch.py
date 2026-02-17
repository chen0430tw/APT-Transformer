#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fake torch module for testing and command-line help without installing PyTorch.
当torch未安装时提供基本接口，允许CLI命令(如help)正常运行。

get_torch() 是唯一的入口：
- 真实 torch 已安装 → 返回真实 torch（缓存）
- 真实 torch 未安装 → 返回 FakeTorch 单例（缓存）
- 重入调用（循环导入触发） → 返回 FakeTorch 单例（不会无限递归）
"""

import sys


class FakeCuda:
    """Fake CUDA interface"""

    class amp:
        """Fake cuda.amp module (autocast + GradScaler)"""
        class autocast:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        class GradScaler:
            def __init__(self, *args, **kwargs):
                pass
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
            def get_scale(self):
                return 1.0
            def state_dict(self):
                return {}
            def load_state_dict(self, state_dict):
                pass

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def get_device_name(index=0):
        return "No GPU"

    @staticmethod
    def get_device_properties(index=0):
        class Props:
            total_memory = 0
            major = 0
            minor = 0
            multi_processor_count = 0
            max_threads_per_block = 1024
            clock_rate = 0
        return Props()

    @staticmethod
    def set_device(device):
        pass

    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def memory_allocated(device=None):
        return 0

    @staticmethod
    def memory_reserved(device=None):
        return 0

    @staticmethod
    def max_memory_allocated(device=None):
        return 0

    @staticmethod
    def manual_seed_all(seed):
        pass

    @staticmethod
    def current_device():
        return 0


class FakeBackends:
    """Fake backends module"""
    class cudnn:
        deterministic = False
        benchmark = False


class FakeVersion:
    """Fake version info"""
    cuda = None


class FakeDevice:
    """Fake device class"""
    def __init__(self, device_type="cpu"):
        if isinstance(device_type, FakeDevice):
            self.type = device_type.type
        else:
            self.type = str(device_type).split(":")[0]

    def __str__(self):
        return self.type

    def __repr__(self):
        return f"device(type='{self.type}')"

    def __eq__(self, other):
        if isinstance(other, FakeDevice):
            return self.type == other.type
        if isinstance(other, str):
            return self.type == other
        return False

    def __hash__(self):
        return hash(self.type)


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
    backends = FakeBackends()

    # Add dtype and device as type aliases (for type annotations)
    dtype = FakeDtype
    device = FakeDevice

    # Add dtype instances
    float32 = FakeDtype("float32")
    float16 = FakeDtype("float16")
    bfloat16 = FakeDtype("bfloat16")
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

    @staticmethod
    def enable_grad():
        """Context manager for enabling gradient computation."""
        class EnableGrad:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return EnableGrad()

    @staticmethod
    def inference_mode(mode=True):
        """Context manager for inference mode."""
        class InferenceMode:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return InferenceMode()

    class nn:
        """Fake nn module"""
        class Module:
            def __init__(self):
                self._parameters = []
                self._modules = []
                self.training = True

            def to(self, device=None, dtype=None):
                return self

            def eval(self):
                self.training = False
                return self

            def train(self, mode=True):
                self.training = mode
                return self

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict, strict=True):
                pass

            def parameters(self):
                """Return fake parameters"""
                if not self._parameters:
                    self._parameters = [FakeTorch.Tensor(shape=(10, 10)) for _ in range(2)]
                return iter(self._parameters)

            def named_parameters(self, prefix='', recurse=True):
                """Return named fake parameters"""
                for i, p in enumerate(self.parameters()):
                    yield f"{prefix}param_{i}", p

            def named_modules(self, memo=None, prefix='', remove_duplicate=True):
                """Return named modules"""
                yield prefix, self

            def children(self):
                return iter([])

            def modules(self):
                yield self

            def __call__(self, *args, **kwargs):
                """Make module callable"""
                return self.forward(*args, **kwargs)

            def forward(self, x):
                """Default forward pass - override in subclasses"""
                return x

            def register_buffer(self, name, tensor, persistent=True):
                """Register a buffer"""
                setattr(self, name, tensor)

            def apply(self, fn):
                """Apply fn recursively to every submodule"""
                fn(self)
                return self

            def zero_grad(self, set_to_none=True):
                pass

            def half(self):
                return self

            def float(self):
                return self

        class Parameter:
            """Fake nn.Parameter"""
            def __init__(self, data=None, requires_grad=True):
                if data is None:
                    data = FakeTorch.Tensor(shape=())
                self.data = data
                self.shape = getattr(data, 'shape', ())
                self.requires_grad = requires_grad
                self.grad = None

            def to(self, device=None, dtype=None):
                return self

            def item(self):
                return 0.0

            def size(self, dim=None):
                if dim is None:
                    return self.shape
                return self.shape[dim] if dim < len(self.shape) else 1

            def __repr__(self):
                return f"FakeParameter(shape={self.shape})"

        class functional:
            """Fake functional module"""
            @staticmethod
            def relu(x, inplace=False):
                return x

            @staticmethod
            def gelu(x):
                return x

            @staticmethod
            def silu(x):
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
                return FakeTorch.Tensor(shape=())

            @staticmethod
            def mse_loss(input, target, reduction='mean'):
                return FakeTorch.Tensor(shape=())

            @staticmethod
            def binary_cross_entropy(input, target, weight=None, reduction='mean'):
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

        class utils:
            """Fake nn.utils module"""
            class rnn:
                @staticmethod
                def pad_sequence(sequences, batch_first=False, padding_value=0.0):
                    """Fake pad_sequence"""
                    if not sequences:
                        return FakeTorch.Tensor(shape=(0,))
                    max_len = max(getattr(s, 'shape', (0,))[0] if hasattr(s, 'shape') and s.shape else 0
                                  for s in sequences)
                    batch_size = len(sequences)
                    if batch_first:
                        return FakeTorch.Tensor(shape=(batch_size, max_len))
                    return FakeTorch.Tensor(shape=(max_len, batch_size))

            @staticmethod
            def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
                return FakeTorch.Tensor(shape=())

        # Common nn layers
        class Linear(Module):
            """Fake Linear layer with shape tracking"""
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.bias = bias
                self.weight = FakeTorch.Tensor(shape=(out_features, in_features))
                if bias:
                    self.bias_param = FakeTorch.Tensor(shape=(out_features,))

            def forward(self, x):
                """Fake forward pass with shape tracking"""
                if hasattr(x, 'shape') and len(x.shape) > 0:
                    batch_dims = x.shape[:-1]
                    output_shape = (*batch_dims, self.out_features)
                else:
                    output_shape = (self.out_features,)

                return FakeTorch.Tensor(shape=output_shape)

        class LayerNorm(Module):
            """Fake LayerNorm layer"""
            def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps

            def forward(self, x):
                return x

        class RMSNorm(Module):
            """Fake RMSNorm layer"""
            def __init__(self, normalized_shape, eps=1e-5):
                super().__init__()
                self.normalized_shape = normalized_shape
                self.eps = eps

            def forward(self, x):
                return x

        class Embedding(Module):
            """Fake Embedding layer"""
            def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
                super().__init__()
                self.num_embeddings = num_embeddings
                self.embedding_dim = embedding_dim
                self.padding_idx = padding_idx
                self.weight = FakeTorch.Tensor(shape=(num_embeddings, embedding_dim))

            def forward(self, x):
                if hasattr(x, 'shape'):
                    output_shape = (*x.shape, self.embedding_dim)
                else:
                    output_shape = (self.embedding_dim,)
                return FakeTorch.Tensor(shape=output_shape)

        class Dropout(Module):
            """Fake Dropout layer"""
            def __init__(self, p=0.5, inplace=False):
                super().__init__()
                self.p = p

            def forward(self, x):
                return x

        class MultiheadAttention(Module):
            """Fake MultiheadAttention layer"""
            def __init__(self, embed_dim, num_heads, *args, **kwargs):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads

            def forward(self, query, key=None, value=None, **kwargs):
                return query, None

        class GELU(Module):
            def __init__(self, approximate='none'):
                super().__init__()
            def forward(self, x):
                return x

        class SiLU(Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x

        class ReLU(Module):
            def __init__(self, inplace=False):
                super().__init__()
            def forward(self, x):
                return x

        class Sequential(Module):
            def __init__(self, *args):
                super().__init__()
                self._seq_modules = list(args)
            def forward(self, x):
                for m in self._seq_modules:
                    x = m(x)
                return x

        class ModuleList(Module):
            def __init__(self, modules=None):
                super().__init__()
                self._list = list(modules) if modules else []
            def __getitem__(self, idx):
                return self._list[idx]
            def __len__(self):
                return len(self._list)
            def __iter__(self):
                return iter(self._list)
            def append(self, module):
                self._list.append(module)

        class ModuleDict(Module):
            def __init__(self, modules=None):
                super().__init__()
                self._dict = dict(modules) if modules else {}
            def __getitem__(self, key):
                return self._dict[key]
            def __setitem__(self, key, module):
                self._dict[key] = module
            def __contains__(self, key):
                return key in self._dict
            def __len__(self):
                return len(self._dict)
            def keys(self):
                return self._dict.keys()
            def values(self):
                return self._dict.values()
            def items(self):
                return self._dict.items()

        class Conv1d(Module):
            def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
            def forward(self, x):
                return x

        class Conv2d(Module):
            def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
            def forward(self, x):
                return x

        class BatchNorm1d(Module):
            def __init__(self, num_features, *args, **kwargs):
                super().__init__()
            def forward(self, x):
                return x

        class BatchNorm2d(Module):
            def __init__(self, num_features, *args, **kwargs):
                super().__init__()
            def forward(self, x):
                return x

        class CrossEntropyLoss(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, input, target):
                return FakeTorch.Tensor(shape=())

        class MSELoss(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, input, target):
                return FakeTorch.Tensor(shape=())

    class optim:
        """Fake optim module"""
        class Optimizer:
            """Fake base Optimizer class"""
            def __init__(self, *args, **kwargs):
                self.param_groups = [{'lr': 0.001}]

            def zero_grad(self, set_to_none=True):
                pass

            def step(self, closure=None):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        class Adam(Optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class AdamW(Optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class SGD(Optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class lr_scheduler:
            """Fake lr_scheduler module"""
            class _LRScheduler:
                def __init__(self, optimizer, *args, **kwargs):
                    pass
                def step(self, epoch=None):
                    pass
                def state_dict(self):
                    return {}
                def load_state_dict(self, state_dict):
                    pass
                def get_last_lr(self):
                    return [0.001]

            class StepLR(_LRScheduler):
                pass

            class CosineAnnealingLR(_LRScheduler):
                pass

            class CosineAnnealingWarmRestarts(_LRScheduler):
                pass

            class LinearLR(_LRScheduler):
                pass

            class OneCycleLR(_LRScheduler):
                pass

            class LambdaLR(_LRScheduler):
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

            class Sampler:
                def __init__(self, *args, **kwargs):
                    pass

    class Tensor:
        """Fake tensor class with shape tracking"""
        def __init__(self, data=None, shape=None, dtype=None, device=None):
            self.data = data
            self.shape = shape or ()
            self.requires_grad = False
            self.grad = None
            self.dtype = dtype
            self.device = FakeDevice("cpu")
            self.is_cuda = False

        def to(self, device=None, dtype=None):
            return self

        def cpu(self):
            return self

        def cuda(self, device=None):
            return self

        def numpy(self):
            return []

        def tolist(self):
            return []

        def item(self):
            import random
            return random.uniform(0.5, 7.0)

        def backward(self, gradient=None, retain_graph=None):
            pass

        def detach(self):
            return self

        def clone(self):
            return FakeTorch.Tensor(data=self.data, shape=self.shape)

        def contiguous(self):
            return self

        def view(self, *shape):
            return FakeTorch.Tensor(shape=shape)

        def reshape(self, *shape):
            return FakeTorch.Tensor(shape=shape)

        def unsqueeze(self, dim):
            new_shape = list(self.shape)
            new_shape.insert(dim, 1)
            return FakeTorch.Tensor(shape=tuple(new_shape))

        def squeeze(self, dim=None):
            return self

        def expand(self, *sizes):
            return FakeTorch.Tensor(shape=tuple(sizes))

        def permute(self, *dims):
            if self.shape:
                new_shape = tuple(self.shape[d] for d in dims if d < len(self.shape))
            else:
                new_shape = ()
            return FakeTorch.Tensor(shape=new_shape)

        def transpose(self, dim0, dim1):
            return self

        def dim(self):
            return len(self.shape)

        def numel(self):
            result = 1
            for s in self.shape:
                result *= s
            return result

        def size(self, dim=None):
            if dim is None:
                return self.shape
            return self.shape[dim] if dim < len(self.shape) else 1

        def float(self):
            return self

        def long(self):
            return self

        def half(self):
            return self

        def bool(self):
            return self

        def mean(self, dim=None, keepdim=False):
            return FakeTorch.Tensor(shape=())

        def sum(self, dim=None, keepdim=False):
            return FakeTorch.Tensor(shape=())

        def max(self, dim=None):
            if dim is not None:
                return FakeTorch.Tensor(shape=()), FakeTorch.Tensor(shape=())
            return FakeTorch.Tensor(shape=())

        def min(self, dim=None):
            if dim is not None:
                return FakeTorch.Tensor(shape=()), FakeTorch.Tensor(shape=())
            return FakeTorch.Tensor(shape=())

        def __repr__(self):
            return f"FakeTensor(shape={self.shape})"

        def __len__(self):
            if self.shape:
                return self.shape[0]
            return 0

        def __getitem__(self, key):
            return FakeTorch.Tensor(shape=())

        def __add__(self, other):
            return self
        def __radd__(self, other):
            return self
        def __mul__(self, other):
            return self
        def __rmul__(self, other):
            return self
        def __sub__(self, other):
            return self
        def __truediv__(self, other):
            return self
        def __neg__(self):
            return self
        def __lt__(self, other):
            return False
        def __le__(self, other):
            return True
        def __gt__(self, other):
            return False
        def __ge__(self, other):
            return True
        def __eq__(self, other):
            return False
        def __ne__(self, other):
            return True
        def __bool__(self):
            return False
        def __hash__(self):
            return id(self)

    @staticmethod
    def tensor(data, **kwargs):
        if hasattr(data, '__len__'):
            try:
                shape = (len(data),)
            except Exception:
                shape = ()
        else:
            shape = ()
        return FakeTorch.Tensor(data, shape=shape)

    @staticmethod
    def zeros(*args, **kwargs):
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def ones(*args, **kwargs):
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def randn(*args, **kwargs):
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def rand(*args, **kwargs):
        shape = args if args else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def randint(low, high=None, size=None, **kwargs):
        if high is None:
            high = low
            low = 0
        shape = size if size else ()
        return FakeTorch.Tensor(shape=shape)

    @staticmethod
    def arange(*args, **kwargs):
        return FakeTorch.Tensor(shape=(0,))

    @staticmethod
    def linspace(*args, **kwargs):
        return FakeTorch.Tensor(shape=(0,))

    @staticmethod
    def cat(tensors, dim=0):
        return FakeTorch.Tensor(shape=())

    @staticmethod
    def stack(tensors, dim=0):
        return FakeTorch.Tensor(shape=())

    @staticmethod
    def where(condition, x=None, y=None):
        if x is not None:
            return x
        return FakeTorch.Tensor(shape=())

    @staticmethod
    def clamp(input, min=None, max=None):
        return input

    @staticmethod
    def save(obj, f, *args, **kwargs):
        pass

    @staticmethod
    def load(f, *args, **kwargs):
        return {}

    @staticmethod
    def from_numpy(ndarray):
        return FakeTorch.Tensor(shape=getattr(ndarray, 'shape', ()))

    @staticmethod
    def is_tensor(obj):
        return isinstance(obj, FakeTorch.Tensor)


# ============================================================================
# get_torch() — 唯一入口，带缓存和重入保护
# ============================================================================

# 缓存：首次解析后不再重复 import
_torch_cache = None
# 重入保护：防止 import torch → apt.core.__init__ → system.py → get_torch() 循环
_resolving = False

# FakeTorch 单例（避免每次 fallback 创建新实例）
_fake_torch_singleton = FakeTorch()


def get_torch():
    """
    尝试导入真实的 torch，失败则返回 FakeTorch 单例。

    特性：
    - 缓存：首次调用后结果缓存，后续调用 O(1) 返回
    - 重入安全：如果在导入过程中被循环调用，返回 FakeTorch 而非无限递归
    - sys.modules 优先：先检查 sys.modules 避免触发新的 import 链

    Returns:
        torch module or FakeTorch
    """
    global _torch_cache, _resolving

    # 1. 已缓存 → 直接返回
    if _torch_cache is not None:
        return _torch_cache

    # 2. 重入保护：正在解析中被再次调用 → 返回 FakeTorch 防止无限递归
    if _resolving:
        return _fake_torch_singleton

    # 3. 首次调用：标记为正在解析
    _resolving = True
    try:
        # 3a. 先查 sys.modules（torch 可能已被其他路径导入，跳过 import 开销）
        _real = sys.modules.get('torch')
        if _real is not None:
            _torch_cache = _real
            return _torch_cache

        # 3b. 尝试 import 真实 torch
        import torch as _real_torch
        _torch_cache = _real_torch
        return _torch_cache

    except ImportError:
        # 3c. 未安装 torch → 使用 FakeTorch
        _torch_cache = _fake_torch_singleton
        return _torch_cache
    finally:
        _resolving = False
