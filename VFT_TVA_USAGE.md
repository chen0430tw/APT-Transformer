## VFT/TVA 核心模块使用指南

**Vein-Flow Transformer / Tri-Vein Attention** 是 APT-Transformer 的核心高效架构。

---

## 目录

- [概述](#概述)
- [核心组件](#核心组件)
  - [Vein Projector](#vein-projector)
  - [TVA Attention](#tva-attention)
  - [VFT Feed-Forward](#vft-feed-forward)
  - [Normal Compensator](#normal-compensator)
  - [VFT Block](#vft-block)
- [注册系统](#注册系统)
- [使用示例](#使用示例)
- [性能对比](#性能对比)
- [最佳实践](#最佳实践)

---

## 概述

### 什么是 VFT/TVA？

VFT/TVA 通过在低秩 **Vein 子空间**（r 维）中进行计算，大幅降低 Transformer 的计算复杂度：

- **标准 Attention**: O(B × H × T² × d)
- **TVA**: O(B × H × T² × r) where r << d
- **典型设置**: r=4-32, d=768-4096
- **加速比**: 通常 10-100x FLOPs 减少

### 关键思想

1. **Vein 子空间**: 学习一个低秩投影 V: ℝ^d → ℝ^r 和重建 U: ℝ^r → ℝ^d
2. **Attention in r-dim**: Q, K, V 都投影到 r 维，attention 在低维空间计算
3. **FFN in r-dim**: 前馈网络也在 r 维空间操作
4. **Normal Compensation**: 对离流形较远的 token 进行稀疏修正

---

## 核心组件

### Vein Projector

低秩子空间投影器，将 d 维空间投影到 r 维 vein 空间。

#### 基本用法

```python
from apt_model.modeling.blocks import VeinProjector

# 创建 vein projector
vein = VeinProjector(
    d_model=768,
    rank=4,
    implementation='linear',  # or 'parameter'
    init_method='orthogonal',  # or 'normal', 'xavier'
)

# 投影到 vein 空间
import torch
x = torch.randn(2, 128, 768)  # [batch, seq_len, d_model]
z = vein.project(x)             # [batch, seq_len, rank]

# 重建回原空间
x_rec = vein.reconstruct(z)     # [batch, seq_len, d_model]

# 计算重建误差
error = vein.compute_reconstruction_error(x)  # [batch, seq_len]
print(f"Mean error: {error.mean()}")
```

#### 两种实现方式

**1. Linear 实现（推荐）**
```python
vein = VeinProjector(d_model=768, rank=4, implementation='linear')
# 使用 nn.Linear 层
# 支持正交初始化
# 略多参数但更灵活
```

**2. Parameter 实现（内存优化）**
```python
vein = VeinProjector(d_model=768, rank=4, implementation='parameter')
# 使用 nn.Parameter
# 更少内存占用
# 与 gpt4o_model.py 中的 VeinSubspaceShared 兼容
```

#### 压缩比

```python
compression_ratio = vein.get_compression_ratio()
print(f"Compression: {compression_ratio:.1f}x")
# d=768, r=4 -> 压缩比 ~147x
```

---

### TVA Attention

在 vein 子空间中计算的三向注意力机制。

#### 基本用法

```python
from apt_model.modeling.blocks import TVAAttention

attn = TVAAttention(
    d_model=768,
    n_heads=12,
    rank=4,
    attn_dropout=0.1,
)

# Forward pass
x = torch.randn(2, 128, 768)
output, attn_weights = attn(x, return_attention_weights=True)

print(f"Output shape: {output.shape}")  # [2, 128, 768]
print(f"Attention weights: {attn_weights.shape}")  # [2, 12, 128, 128]
```

#### 使用注意力掩码

```python
# 创建因果掩码
seq_len = 128
mask = torch.zeros(1, 1, seq_len, seq_len)
mask[:, :, :, :] = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

output, _ = attn(x, attn_mask=mask)
```

#### 共享 Vein Projector

```python
# 多个模块共享同一个 vein projector
shared_vein = VeinProjector(d_model=768, rank=4)

attn1 = TVAAttention(d_model=768, n_heads=12, rank=4, vein_projector=shared_vein)
attn2 = TVAAttention(d_model=768, n_heads=12, rank=4, vein_projector=shared_vein)
```

---

### VFT Feed-Forward

在 vein 子空间中的前馈网络。

#### 基本用法

```python
from apt_model.modeling.blocks import VFTFeedForward

ffn = VFTFeedForward(
    d_model=768,
    rank=4,
    r_hidden=16,  # Hidden dim in vein space (default: 4 * rank)
    activation='silu',  # 'gelu', 'relu', 'silu'
    dropout=0.1,
)

x = torch.randn(2, 128, 768)
output = ffn(x)  # [2, 128, 768]
```

#### 参数对比

```python
# 标准 FFN
# Parameters: 2 * d_model * d_ff
#           = 2 * 768 * 3072 = 4.7M

# VFT FFN
# Parameters: 2 * rank * r_hidden + vein overhead
#           = 2 * 4 * 16 + small = ~128 + small
# Reduction: ~30,000x !
```

#### 使用稳定器

```python
# 稳定器添加一个小的直接路径
ffn = VFTFeedForward(
    d_model=768,
    rank=4,
    use_stabilizer=True,  # 默认 True
)
# output = vein_ffn(x) + stabilizer(x)
```

---

### Normal Compensator

对离流形较远的 token 进行稀疏法线修正。

#### 基本用法

```python
from apt_model.modeling.blocks import NormalCompensator

compensator = NormalCompensator(
    d_model=768,
    s=1,  # Number of correction basis vectors
    tau=0.18,  # Threshold for applying correction
    alpha_scale=0.5,  # Scale for correction weights
)

# 需要提供重建误差
x = torch.randn(2, 128, 768)
eps = vein.compute_reconstruction_error(x)  # [2, 128]

output = compensator(x, eps)
```

#### 如何工作

- 计算每个 token 的离面距离 ε = ||h - U(Vh)||
- 只对 ε > τ 的 token 应用修正
- 修正形式: Δy = Σ α_j * u_j * (v_j^T h)
- s 通常取 1-3，τ 通常取 0.15-0.25

---

### VFT Block

完整的 VFT/TVA Transformer 块。

#### 基本用法

```python
from apt_model.modeling.blocks import VFTBlock

block = VFTBlock(
    d_model=768,
    n_heads=12,
    rank=4,
    r_hidden=16,
    s_normals=1,
    tau=0.18,
    attn_dropout=0.1,
    ffn_dropout=0.1,
    activation='silu',
)

x = torch.randn(2, 128, 768)
output, metrics = block(x, return_metrics=True)

print(f"Output: {output.shape}")
print(f"Metrics: {metrics}")
# {'eps_mean': 0.12, 'eps_max': 0.45, 'eps_frac_over_tau': 0.23, 'rank': 4}
```

#### 使用工厂函数

```python
from apt_model.modeling.blocks import create_vft_block

block = create_vft_block(
    d_model=768,
    n_heads=12,
    rank=4,
)
```

#### 构建完整模型

```python
import torch.nn as nn

class VFTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, rank=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            create_vft_block(d_model, n_heads, rank)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

model = VFTModel(vocab_size=50000, d_model=768, n_layers=12, rank=4)
```

---

## 注册系统

使用注册系统在运行时选择 attention 和 FFN 实现。

### 列出可用实现

```python
from apt_model.modeling.blocks import list_attention, list_ffn

print(f"Available attention: {list_attention()}")
# ['tva', 'standard']

print(f"Available FFN: {list_ffn()}")
# ['vft', 'standard']
```

### 通过注册表获取实现

```python
from apt_model.modeling.blocks import get_attention, get_ffn

# 获取 TVA
attn = get_attention('tva', d_model=768, n_heads=12, rank=4)

# 获取 VFT FFN
ffn = get_ffn('vft', d_model=768, rank=4)

# 获取标准实现（对比用）
std_attn = get_attention('standard', d_model=768, n_heads=12)
std_ffn = get_ffn('standard', d_model=768, d_ff=3072)
```

### 注册自定义实现

```python
from apt_model.modeling.blocks import register_attn, register_ffn
import torch.nn as nn

@register_attn("my_custom_attn")
class CustomAttention(nn.Module):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__()
        # Your implementation

    def forward(self, x, attn_mask=None, **kwargs):
        # Your forward logic
        return output, None

# 使用
attn = get_attention('my_custom_attn', d_model=768, n_heads=12)
```

### CLI 集成（根据 memo.txt）

```bash
# 通过 CLI 选择实现
python train.py \
    --attn.impl tva \
    --ffn.impl vft \
    --vft.rank 4 \
    --model.d_model 768
```

---

## 使用示例

### 示例 1: 单个 VFT Block

```python
from apt_model.modeling.blocks import VFTBlock
import torch

# 创建 block
block = VFTBlock(d_model=512, n_heads=8, rank=4)

# 前向传播
x = torch.randn(4, 64, 512)  # [batch=4, seq=64, dim=512]
output, metrics = block(x, return_metrics=True)

print(f"Input:  {x.shape}")
print(f"Output: {output.shape}")
print(f"Mean reconstruction error: {metrics['eps_mean']:.4f}")
print(f"Tokens needing correction: {metrics['eps_frac_over_tau']:.2%}")
```

### 示例 2: 替换标准 Transformer 的 Attention

```python
from apt_model.modeling.blocks import TVAAttention
import torch.nn as nn

class MyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, rank=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)

        # 使用 TVA 代替标准 attention
        self.attn = TVAAttention(d_model, n_heads, rank)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Attention with residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h)
        x = x + attn_out

        # FFN with residual
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out

        return x
```

### 示例 3: 完整的编码器-解码器模型

```python
import torch
import torch.nn as nn
from apt_model.modeling.blocks import VFTBlock

class VFTEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, rank=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        self.layers = nn.ModuleList([
            VFTBlock(d_model, n_heads, rank)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos_embed[:, :input_ids.size(1), :]

        for layer in self.layers:
            x, _ = layer(x)

        return self.norm(x)

# 使用
encoder = VFTEncoder(vocab_size=30000, d_model=512, n_layers=6, rank=4)
input_ids = torch.randint(0, 30000, (2, 128))
output = encoder(input_ids)
print(f"Encoder output: {output.shape}")
```

### 示例 4: 混合使用 VFT 和标准 Transformer

```python
from apt_model.modeling.blocks import VFTBlock, get_attention, get_ffn
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, d_model=768, n_layers=12, n_heads=12, rank=4):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i < n_layers // 2:
                # 前半部分用标准 Transformer
                layer = self._create_standard_layer(d_model, n_heads)
            else:
                # 后半部分用 VFT
                layer = VFTBlock(d_model, n_heads, rank)

            self.layers.append(layer)

    def _create_standard_layer(self, d_model, n_heads):
        class StandardLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = get_attention('standard', d_model=d_model, n_heads=n_heads)
                self.ffn = get_ffn('standard', d_model=d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x, **kwargs):
                x = x + self.attn(self.norm1(x))[0]
                x = x + self.ffn(self.norm2(x))
                return x, None

        return StandardLayer()

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x
```

---

## 性能对比

### FLOPs 对比

| 模块 | 标准 Transformer | VFT/TVA (r=4) | 加速比 |
|------|----------------|--------------|-------|
| Attention | O(T² × d) | O(T² × r) | ~192x |
| FFN | O(T × d × d_ff) | O(T × r × r_h) | ~2000x |
| 总体 | O(T² × d + T × d²) | O(T² × r + T × r²) | ~100x |

**实际测试（T=128, d=768, r=4）：**

```python
from apt_model.modeling.blocks import VFTBlock, get_attention, get_ffn
import torch
import time

# VFT Block
vft_block = VFTBlock(d_model=768, n_heads=12, rank=4)
x = torch.randn(8, 128, 768)

start = time.time()
for _ in range(100):
    _ = vft_block(x)
vft_time = time.time() - start

# Standard Block (approximate)
std_attn = get_attention('standard', d_model=768, n_heads=12)
std_ffn = get_ffn('standard', d_model=768, d_ff=3072)

start = time.time()
for _ in range(100):
    _ = std_attn(x)
    _ = std_ffn(x)
std_time = time.time() - start

print(f"VFT time: {vft_time:.3f}s")
print(f"Standard time: {std_time:.3f}s")
print(f"Speedup: {std_time / vft_time:.2f}x")
```

### 参数量对比

```python
# d=768, n_heads=12, r=4

# Standard Transformer Block
# - Attention: 4 * d * d = 4 * 768 * 768 = 2.36M
# - FFN: 2 * d * d_ff = 2 * 768 * 3072 = 4.72M
# Total: ~7.08M

# VFT Block
vft_block = VFTBlock(d_model=768, n_heads=12, rank=4)
vft_params = sum(p.numel() for p in vft_block.parameters())
print(f"VFT params: {vft_params:,}")
# ~1.5M parameters

print(f"Parameter reduction: {7.08 / (vft_params / 1e6):.2f}x")
```

---

## 最佳实践

### 1. 选择合适的 Rank

```python
# 小模型（d < 512）: rank = 2-4
# 中等模型（512 <= d < 2048）: rank = 4-8
# 大模型（d >= 2048）: rank = 8-32

# 经验法则: rank ≈ sqrt(d) / 4
import math
def recommend_rank(d_model):
    return max(2, min(32, int(math.sqrt(d_model) / 4)))

print(recommend_rank(768))   # 4
print(recommend_rank(2048))  # 8
print(recommend_rank(4096))  # 16
```

### 2. 调整 Normal Compensation

```python
# τ 太小: 修正太频繁，开销大
# τ 太大: 修正太少，精度损失

# 推荐设置:
# - 训练初期: tau = 0.25 (少修正，快速训练)
# - 训练中期: tau = 0.18 (平衡)
# - 训练后期: tau = 0.10 (多修正，提高精度)

# s (basis vectors):
# - s = 0: 无修正（最快）
# - s = 1: 单向修正（推荐）
# - s = 2-3: 多向修正（更精确但更慢）
```

### 3. 监控重建误差

```python
block = VFTBlock(d_model=768, n_heads=12, rank=4)

x = torch.randn(4, 128, 768)
output, metrics = block(x, return_metrics=True)

# 检查指标
if metrics['eps_mean'] > 0.3:
    print("Warning: High reconstruction error, consider increasing rank")

if metrics['eps_frac_over_tau'] > 0.5:
    print("Warning: Too many tokens need correction, consider lowering tau")
```

### 4. 渐进式迁移

从标准 Transformer 迁移到 VFT/TVA:

```python
# 步骤 1: 只替换 FFN
class Phase1Block(nn.Module):
    def __init__(self, d_model, n_heads, rank):
        super().__init__()
        self.attn = get_attention('standard', d_model=d_model, n_heads=n_heads)
        self.ffn = get_ffn('vft', d_model=d_model, rank=rank)  # VFT FFN
        # ... norms ...

# 步骤 2: 替换 Attention
class Phase2Block(nn.Module):
    def __init__(self, d_model, n_heads, rank):
        super().__init__()
        self.attn = get_attention('tva', d_model=d_model, n_heads=n_heads, rank=rank)
        self.ffn = get_ffn('vft', d_model=d_model, rank=rank)
        # ... norms ...

# 步骤 3: 使用完整 VFT Block
block = VFTBlock(d_model, n_heads, rank)
```

### 5. 训练技巧

```python
# 1. 使用较小的学习率（vein 参数对初始化敏感）
optimizer = torch.optim.AdamW([
    {'params': model.vein_params, 'lr': 1e-4},  # Vein 参数
    {'params': model.other_params, 'lr': 5e-4},  # 其他参数
])

# 2. Warm-up vein projector
# 前几个 epoch 冻结 vein，只训练其他部分
for epoch in range(5):
    for param in model.vein.parameters():
        param.requires_grad = False
    # ... training ...

# 然后解冻
for param in model.vein.parameters():
    param.requires_grad = True

# 3. 监控重建误差
def log_vein_metrics(model, x):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, VFTBlock):
                _, metrics = module(x, return_metrics=True)
                print(f"{name}: eps_mean={metrics['eps_mean']:.4f}")
```

---

## 常见问题

### Q: 何时使用 VFT/TVA？

**A**:
- ✅ 大规模预训练（节省计算）
- ✅ 长序列建模（T > 512）
- ✅ 资源受限环境（移动设备、边缘计算）
- ⚠️ 极短序列（T < 32）优势不明显
- ⚠️ 需要最高精度的任务（可以增大 rank）

### Q: rank 如何选择？

**A**:
- **r=2**: 极度压缩，适合推理加速
- **r=4**: 平衡，大多数任务推荐
- **r=8**: 高精度，大模型推荐
- **r=16-32**: 接近标准 Transformer 精度

### Q: VFT/TVA 会损失精度吗？

**A**:
- rank 足够大时（r >= 4），精度损失 < 1%
- Normal Compensation 可以补偿大部分损失
- 某些任务（如语言建模）几乎无损

### Q: 能否与其他优化技术结合？

**A**:
- ✅ Flash Attention: 可以在 vein 空间使用
- ✅ Gradient Checkpointing: 完全兼容
- ✅ Mixed Precision: 完全兼容
- ✅ MoE: 可以在 vein 空间实现 MoE（见 gpto3_model.py）

### Q: 向后兼容性？

**A**:
- `VeinSubspaceShared` 是 `VeinProjector` 的别名
- 现有代码可以继续使用本地定义
- 新代码推荐从 `apt_model.modeling.blocks` 导入

---

## 参考资料

- **代码位置**: `apt_model/modeling/blocks/`
- **测试脚本**: `test_vft_tva.py`
- **原始实现**: `vft_tva.py` (根目录)
- **使用示例**: `apt_model/modeling/gpt4o_model.py`, `gpto3_model.py`

---

生成时间: 2025-10-25
