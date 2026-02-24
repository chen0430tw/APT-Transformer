# Virtual VRAM & LECaC 更新日志

记录所有重要的优化和修复

---

## [2.0.0] - 2026-02-24

### 🎉 重大更新

#### 新增功能

**LECaC Soft Warmup 技术栈**
- ✨ 新增 `lecac_warmup.py` 调度器模块
- ✨ 实现 `LECACAlphaScheduler` - Alpha补偿强度动态调度
- ✨ 实现 `LECACBitsScheduler` - 渐进式量化精度调度（实验性）
- ✨ 实现 `SoftQuantizationScheduler` - 温度退火软量化（备用）
- ✨ 新增 `update_lecac_alpha()` 辅助函数

**技术亮点**：
```python
# Alpha Warmup解决低学习率NaN问题
alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=100,
    warmup_multiplier=3.0,  # α: 1.47 → 4.41 → 1.47
    schedule="cosine"
)
```

#### 问题修复

**双重量化互斥机制** (#001)
- 🐛 修复 LECaC INT2 + Virtual VRAM INT8 导致的Loss=NaN
- 🔧 在 `virtual_vram.py:835-841` 添加LECaC量化检测
- 🔧 检测逻辑: `dtype==INT8 && hasattr(scale)` → 跳过VRAM量化
- 📝 添加调试日志: `🔄 检测到LECaC量化，跳过VRAM量化`

**修复前**：
```python
# Bug: 双重量化
should_quantize = (
    LECaC_AVAILABLE and
    cfg.nested_quantization_bits > 0 and
    tensor_size_mb >= 5.0
)  # ← 缺少LECaC检测
```

**修复后**：
```python
# Fix: 互斥机制
is_lecac_quantized = (t.dtype == torch.int8 and hasattr(t, 'scale'))

should_quantize = (
    cfg.enable_nested_v16 and
    LECaC_AVAILABLE and
    cfg.nested_quantization_bits > 0 and
    tensor_size_mb >= 5.0 and
    not is_lecac_quantized  # ← 新增
)
```

#### 架构设计

**Virtual VRAM 2.0 规划**
- 📋 设计 Selective Recomputation 策略（Phase 1）
- 📋 设计 Kernel Fusion 架构（Phase 2）
- 📋 设计 IO-Aware Scheduling 框架（Phase 3）
- 📚 完整技术文档和参考文献

#### 文档更新

- 📚 新增 `VRAM_OPTIMIZATION_GUIDE.md` - 完整技术文档（70KB）
- 📚 新增 `QUICKSTART_WARMUP.md` - 5分钟快速开始指南
- 📚 新增 `OPTIMIZATION_COMPARISON.md` - 优化方案对比矩阵
- 📚 新增 `CHANGELOG_VRAM.md` - 版本更新日志
- 📝 新增 `example_lecac_warmup_training.py` - 完整训练示例

#### 性能数据

**测试结果汇总**：

| 版本 | 配置 | Tok/s | vs基准 | 状态 |
|------|------|-------|--------|------|
| v1.5 | 基准（无VRAM） | 2,448 | - | ✅ |
| v1.6 | VRAM min=1MB | 1,511 | -38% | ❌ |
| v1.6 | VRAM min=20MB | 2,867 | **+17%** | ✅ |
| v2.0 | LECaC+VRAM+Warmup | TBD | 预期+20% | 🔄 |

**已解决问题**：
- ✅ Job 122591: 过度offload导致-38%性能 → 调高阈值到20MB
- ✅ Job 123766: 双重量化NaN → 互斥机制
- ✅ TWCC 870667-870669: 低学习率NaN → Alpha Warmup

### 🔬 实验性功能

- 🧪 Progressive Bits Warmup (8→4→2-bit渐进)
- 🧪 Soft Quantization with Temperature Annealing
- 🧪 Orthogonal Compensation (LECaC)

### ⚠️ Breaking Changes

无破坏性变更。所有新功能向后兼容。

### 📦 依赖更新

无新增依赖。所有功能基于PyTorch原生API。

---

## [1.6.1] - 2026-02-23

### 修复

**Prefetch机制修复**
- 🐛 修复 `_do_prefetch` 不支持 `nested_block` 的问题
- 🔧 添加对 `_NestedArcBlock` 的支持
- 📝 添加 DEBUG 日志验证prefetch工作

**Dtype对齐修复**
- 🐛 修复 LECaC backward中dtype不匹配导致的报错
- 🔧 在 `lecac.py:153` 添加 `x_recon.to(weight.dtype)`
- 🔧 在 `virtual_vram.py:1038` 添加dtype参数

### 性能

**Job 122683 成功**
- ✅ 配置: min_tensor=20MB, 无LECaC
- ✅ 结果: 2,867 Tok/s (+17.1% vs baseline)
- ✅ 证明Virtual VRAM在合理配置下有效

---

## [1.6.0] - 2026-02-22

### 新增功能

**Virtual VRAM v1.6 嵌套架构**
- ✨ 实现 LECaC → Page → Block → Arc 四层架构
- ✨ 支持INT2/INT4/INT8量化
- ✨ 集成LECaC量化原语
- ✨ 添加Arc引用计数和热度追踪

**Prefetch机制**
- ✨ 实现异步预取（`_Prefetcher`线程）
- ✨ 基于热度优先队列调度
- ✨ 支持nested_block prefetch

### 已知问题

- ⚠️ min_tensor=1MB时性能-38%（过度offload）
- ⚠️ LECaC量化在低学习率warmup时NaN
- ⚠️ 双重量化导致NaN（后续v2.0修复）

---

## [1.5.0] - 2026-02-20

### 新增功能

**基础Virtual VRAM**
- ✨ 实现基本offload机制（GPU ↔ CPU）
- ✨ 支持saved_tensors_hooks集成
- ✨ 实现简单的cache机制
- ✨ 支持量化（INT8）

### 性能

- 显存节省: ~50%
- 速度影响: 轻微（未优化阈值时可能-20-40%）

---

## [1.0.0] - 2026-02-15

### 初始版本

**LECaC量化**
- ✨ 实现INT2/INT4对称量化
- ✨ 实现误差补偿机制（alpha=4/e）
- ✨ 支持一键替换nn.Linear
- ✨ 正交投影补偿变体

**核心功能**：
```python
replace_linear_with_lecac(model, bits=2, alpha=1.47)
```

---

## 版本规划

### [2.1.0] - Q2 2026 (计划中)

**Selective Recomputation**
- [ ] 实现operation whitelist
- [ ] 集成PyTorch checkpoint
- [ ] Cheap ops自动重算
- [ ] 预期: +30% speed

**改进Alpha Warmup**
- [ ] 自动warmup步数推荐
- [ ] 自适应multiplier调整
- [ ] TensorBoard可视化

### [2.2.0] - Q3 2026 (计划中)

**IO-Aware Scheduling**
- [ ] 三级存储（GPU/CPU/Disk）
- [ ] 热度追踪优化
- [ ] Async I/O pipeline
- [ ] 预期: +50% speed, 支持>100B模型

### [3.0.0] - Q4 2026 (计划中)

**Kernel Fusion**
- [ ] CUDA kernel融合Pack/Quantize/Transfer
- [ ] Tiled computation
- [ ] FlashAttention-style optimization
- [ ] 预期: +100% speed, 接近FlashAttention性能

---

## Bug修复历史

### 高优先级Bug

| Bug ID | 描述 | 影响 | 修复版本 | 状态 |
|--------|------|------|---------|------|
| #001 | 双重量化NaN | Loss=NaN | v2.0.0 | ✅ 已修复 |
| #002 | 低学习率warmup NaN | Loss=NaN | v2.0.0 | ✅ 已修复 |
| #003 | Prefetch不支持nested | 性能 | v1.6.1 | ✅ 已修复 |
| #004 | Dtype不匹配 | 崩溃 | v1.6.1 | ✅ 已修复 |
| #005 | 过度offload | -38%性能 | v1.6.1 | ✅ 已修复 |

### 中优先级Bug

| Bug ID | 描述 | 影响 | 状态 |
|--------|------|------|------|
| #101 | UnboundLocalError in lecac_dequantize | 崩溃 | ✅ v1.6.1 |
| #102 | 反向预取requires_grad=False | 潜在问题 | ✅ 已禁用 |

---

## 贡献者

### 核心开发

- Virtual VRAM架构设计与实现
- LECaC量化算法
- Soft Warmup技术研究

### 测试与验证

- Nano5集群性能测试
- TWCC集群验证
- 本地WSL开发环境

### 文档编写

- 技术文档撰写
- API文档完善
- 使用指南和示例

---

## 参考资料

### 关键论文

1. FlashAttention (NeurIPS 2022)
2. Progressive Quantization (2024)
3. Soft-then-Hard Quantization (ICML 2021)
4. FlexGen (ICML 2023)
5. ZeRO-Offload (ATC 2021)

### 技术博客

- PyTorch Activation Checkpointing
- NVIDIA Activation Recomputation Guide
- Quantization-Aware Training Guide

---

## 版本号说明

遵循语义化版本 (Semantic Versioning):

- **主版本号** (Major): 架构性变更，可能不兼容
- **次版本号** (Minor): 新功能，向后兼容
- **修订号** (Patch): Bug修复，向后兼容

---

## 获取帮助

- 📚 查看 [完整文档](./VRAM_OPTIMIZATION_GUIDE.md)
- 🚀 阅读 [快速开始](./QUICKSTART_WARMUP.md)
- 💬 提交Issue或PR

---

**更新频率**: 每次重要更新或修复
**维护状态**: Active Development
**最后更新**: 2026-02-24
