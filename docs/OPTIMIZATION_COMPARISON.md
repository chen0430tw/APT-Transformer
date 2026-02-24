# Virtual VRAM优化方案对比

**快速参考：选择合适的优化策略**

---

## 优化技术矩阵

### 按问题类型选择

```
┌─────────────────────────┬──────────────────┬──────────────────┐
│         问题            │     解决方案      │      优先级      │
├─────────────────────────┼──────────────────┼──────────────────┤
│ 显存不足OOM             │ Virtual VRAM     │ 🔴 P0           │
├─────────────────────────┼──────────────────┼──────────────────┤
│ LECaC低学习率NaN        │ Alpha Warmup     │ 🔴 P0           │
├─────────────────────────┼──────────────────┼──────────────────┤
│ LECaC+VRAM双重量化NaN   │ 互斥机制         │ 🔴 P0           │
├─────────────────────────┼──────────────────┼──────────────────┤
│ 训练速度慢              │ 调高阈值/Recomp  │ 🟡 P1           │
├─────────────────────────┼──────────────────┼──────────────────┤
│ 追求极致性能            │ Kernel Fusion    │ 🟢 P2           │
└─────────────────────────┴──────────────────┴──────────────────┘
```

---

## 核心技术对比

### 1. 量化策略

| 技术 | Bits | 显存节省 | 精度损失 | 速度影响 | 推荐度 |
|------|------|---------|---------|---------|--------|
| **无量化** | - | 0% | 0% | 基准 | ⭐⭐⭐ |
| **INT8量化** | 8 | 75% | 最小 | -5% | ⭐⭐⭐⭐⭐ |
| **INT4量化** | 4 | 87.5% | 小 | -10% | ⭐⭐⭐⭐ |
| **INT2量化** | 2 | 93.75% | 中等 | -15% | ⭐⭐⭐ |
| **LECaC INT2** | 2 | 93.75% | 小（补偿）| -10% | ⭐⭐⭐⭐ |

### 2. Warmup策略

| 策略 | 复杂度 | 效果 | 适用场景 | 推荐度 |
|------|--------|------|---------|--------|
| **Alpha Warmup** | ⭐ 简单 | ⭐⭐⭐ 好 | 低lr warmup | ⭐⭐⭐⭐⭐ |
| **Bits Warmup** | ⭐⭐⭐ 复杂 | ⭐⭐⭐⭐ 很好 | 激进量化 | ⭐⭐⭐ |
| **Soft Quantization** | ⭐⭐ 中等 | ⭐⭐⭐⭐ 很好 | 研究/实验 | ⭐⭐⭐⭐ |
| **Gradient Clipping** | ⭐ 简单 | ⭐⭐ 辅助 | 梯度爆炸 | ⭐⭐⭐ |

### 3. 显存优化

| 技术 | 显存节省 | 速度影响 | 实现难度 | 状态 |
|------|---------|---------|---------|------|
| **Virtual VRAM v1.6** | 50% | +17% | 低 | ✅ 已实现 |
| **Gradient Checkpointing** | 60% | -20% | 低 | ✅ PyTorch内置 |
| **Selective Recompute** | 70% | +30% | 中 | 🔄 设计中 |
| **Kernel Fusion** | 70% | +50% | 高 | 📋 计划中 |
| **FlashAttention** | 80% | +200% | 高 | 🎯 参考目标 |

---

## 配置推荐方案

### 方案1: 保守稳定型（推荐新手）

```python
# 模型配置
lecac_bits = 4  # INT4（保守）
lecac_alpha = 4.0 / math.e

# Alpha Warmup
alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=100,
    warmup_multiplier=2.0,  # 保守倍数
    schedule="cosine"
)

# Virtual VRAM
vram_config = VirtualVRAMConfig(
    enabled=True,
    min_tensor_bytes=20 << 20,  # 20MB（避免过度offload）
    nested_quantization_bits=4,  # 与LECaC一致
    verbose=False
)

# 训练超参
lr = 3e-4
lr_warmup_steps = 100
gradient_clip_norm = 1.0
```

**适用场景**：
- 第一次使用LECaC
- 追求稳定性
- 显存略微不足

**预期结果**：
- ✅ 稳定训练，无NaN
- ✅ 显存节省50%
- ✅ 速度持平或略优于baseline

---

### 方案2: 平衡优化型（推荐）

```python
# 模型配置
lecac_bits = 2  # INT2（激进）
lecac_alpha = 4.0 / math.e

# Alpha Warmup
alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=100,
    warmup_multiplier=3.0,  # 标准倍数
    schedule="cosine"
)

# Virtual VRAM
vram_config = VirtualVRAMConfig(
    enabled=True,
    min_tensor_bytes=20 << 20,
    nested_quantization_bits=2,  # INT2
    verbose=True  # 调试阶段开启
)

# 训练超参
lr = 3e-4
lr_warmup_steps = 100
gradient_clip_norm = 1.0
```

**适用场景**：
- 中大型模型（>1B参数）
- 显存明显不足
- 追求显存和速度平衡

**预期结果**：
- ✅ 显存节省70%
- ✅ 速度+10-20%
- ✅ 精度损失<1%

---

### 方案3: 极限压缩型

```python
# 模型配置
lecac_bits = 2
lecac_alpha = 4.0 / math.e

# Alpha Warmup（加强）
alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=200,  # 延长warmup
    warmup_multiplier=4.0,  # 更强补偿
    schedule="cosine"
)

# Virtual VRAM
vram_config = VirtualVRAMConfig(
    enabled=True,
    min_tensor_bytes=5 << 20,  # 5MB（更激进offload）
    nested_quantization_bits=2,
    verbose=True
)

# Gradient Checkpointing（额外显存优化）
from torch.utils.checkpoint import checkpoint
# 在模型中应用checkpoint

# 训练超参
lr = 3e-4
lr_warmup_steps = 200  # 延长
gradient_clip_norm = 0.5  # 更严格裁剪
```

**适用场景**：
- 超大模型（>10B参数）
- 显存严重不足
- 可以接受略微的性能损失

**预期结果**：
- ✅ 显存节省80%+
- ⚠️ 速度-10-20%
- ⚠️ 精度损失1-2%
- ⚠️ 训练时间增加

---

## 性能基准对比

### Nano5集群测试结果

```
硬件：8×H100 80GB, HDR InfiniBand 200Gbps
模型：316M参数Transformer（12层）
数据：FineWeb + HLBD
Batch size：2 × 2 (grad_accum)

┌─────────┬──────────┬─────────┬──────────┬──────────┬────────┐
│   Job   │  配置    │ LECaC   │  VRAM    │  Warmup  │ Tok/s  │
├─────────┼──────────┼─────────┼──────────┼──────────┼────────┤
│ 122571  │ 基准     │ ❌      │ ❌       │ -        │ 2,448  │
├─────────┼──────────┼─────────┼──────────┼──────────┼────────┤
│ 122683  │ 方案1    │ ❌      │ ✅ 20MB  │ -        │ 2,867  │
│         │          │         │          │          │ +17%✅ │
├─────────┼──────────┼─────────┼──────────┼──────────┼────────┤
│ 待测试  │ 方案2    │ ✅ INT2 │ ✅ INT2  │ ✅ 3.0×  │ ???    │
│         │          │         │ 20MB     │          │ 预期✅ │
└─────────┴──────────┴─────────┴──────────┴──────────┴────────┘
```

### TWCC集群测试结果

```
硬件：8×V100 32GB, EDR InfiniBand 100Gbps
模型：316M参数Transformer（12层）

┌─────────┬──────────┬─────────┬──────────┬──────────┬────────┐
│   Job   │  配置    │ LECaC   │  Warmup  │  结果    │  状态  │
├─────────┼──────────┼─────────┼──────────┼──────────┼────────┤
│ 870667  │ 无Warmup │ ✅ INT2 │ ❌       │ Step2NaN │ ❌     │
├─────────┼──────────┼─────────┼──────────┼──────────┼────────┤
│ 870668  │ 无Warmup │ ✅ INT4 │ ❌       │ Step2NaN │ ❌     │
├─────────┼──────────┼─────────┼──────────┼──────────┼────────┤
│ 待测试  │ 方案2    │ ✅ INT2 │ ✅ 3.0×  │ ???      │ 预期✅ │
└─────────┴──────────┴─────────┴──────────┴──────────┴────────┘
```

---

## 调优决策树

```
开始训练
    │
    ├─ 出现OOM？
    │   ├─ Yes → 启用Virtual VRAM（min_tensor=20MB）
    │   │         │
    │   │         └─ 仍OOM？ → 降低到5MB / 启用Gradient Checkpointing
    │   │
    │   └─ No → 继续
    │
    ├─ 出现Loss=NaN？
    │   ├─ 在Warmup期间（前100步）？
    │   │   ├─ Yes → 启用Alpha Warmup（multiplier=3.0）
    │   │   │         │
    │   │   │         └─ 仍NaN？ → 提高multiplier到4.0 / 延长warmup到200步
    │   │   │
    │   │   └─ No → 检查学习率/梯度爆炸
    │   │
    │   └─ 使用LECaC + VRAM？
    │       └─ Yes → 确保互斥机制生效（检查日志）
    │
    ├─ 训练速度慢（<基准80%）？
    │   ├─ 查看日志中offload次数
    │   │   ├─ 过多（>1000次/step）？ → 提高min_tensor到50MB
    │   │   └─ 合理？ → 考虑Selective Recompute优化
    │   │
    │   └─ PCIe传输瓶颈？ → 考虑Kernel Fusion优化
    │
    └─ 一切正常？
        └─ 🎉 恭喜！开始完整训练
```

---

## 参数速查表

### LECaC参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|---------|------|
| `bits` | 2 | 2, 4, 8 | 量化精度 |
| `alpha` | 1.47 | 1.0-2.0 | 补偿强度（训练时自动调度）|
| `orthogonal` | False | False | 正交投影补偿（实验性）|

### Alpha Warmup参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|---------|------|
| `warmup_steps` | 100 | 50-200 | 与lr warmup对齐 |
| `warmup_multiplier` | 3.0 | 2.0-4.0 | 初始alpha放大倍数 |
| `schedule` | "cosine" | linear/cosine/exp | 调度类型 |

### Virtual VRAM参数

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|---------|------|
| `min_tensor_bytes` | 20MB | 5-50MB | Offload阈值 |
| `nested_quantization_bits` | 2 | 2, 4, 8 | VRAM量化精度 |
| `nested_block_size` | 64 | 32-128 | 块大小 |
| `verbose` | False | True/False | 调试日志 |

---

## 故障排查清单

### ✅ 正常训练的标志

- [ ] Loss平稳下降（无NaN/Inf）
- [ ] 显存使用<80%
- [ ] 速度≥基准的80%
- [ ] 日志显示：`✅ Nested D2H`（如果启用VRAM）
- [ ] 日志显示：`🔄 检测到LECaC量化`（如果启用互斥）
- [ ] Warmup期间alpha从高到低平滑过渡

### ❌ 异常情况检查

**Loss=NaN**：
- [ ] 检查学习率是否过高
- [ ] 检查是否在warmup期间
- [ ] 检查alpha warmup是否启用
- [ ] 检查gradient clipping
- [ ] 查看是否有双重量化（日志搜索"跳过VRAM量化"）

**训练速度慢**：
- [ ] 查看offload次数（日志搜索"D2H"）
- [ ] 检查min_tensor_bytes是否过低
- [ ] 检查是否有PCIe瓶颈（nvidia-smi dmon -s pcie）
- [ ] 考虑降低量化精度（INT2 → INT4）

**显存不足**：
- [ ] 降低min_tensor_bytes（20MB → 5MB）
- [ ] 启用Gradient Checkpointing
- [ ] 降低batch size
- [ ] 检查是否有显存泄漏

---

## 未来规划

### Virtual VRAM 2.0 Roadmap

```
Phase 1: Selective Recomputation（Q2 2026）
  - 实现operation whitelist
  - 集成PyTorch checkpoint API
  - 预期：+30% speed, 70% memory saving

Phase 2: IO-Aware Scheduling（Q3 2026）
  - 三级存储（GPU/CPU/Disk）
  - 热度追踪优化
  - 预期：+50% speed, 80% memory saving

Phase 3: Kernel Fusion（Q4 2026）
  - CUDA kernel融合Pack/Quantize/Transfer
  - Tiled computation
  - 预期：+100% speed, 接近FlashAttention性能
```

---

## 相关文档

- [完整技术文档](./VRAM_OPTIMIZATION_GUIDE.md) - 详细技术说明
- [快速开始指南](./QUICKSTART_WARMUP.md) - 5分钟解决NaN
- [示例代码](../example_lecac_warmup_training.py) - 可运行示例

---

**最后更新**: 2026-02-24
**维护状态**: Active
