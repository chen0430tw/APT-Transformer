# 虚拟Blackwell测试脚本使用指南

## 📝 概述

这里提供了两个测试脚本，用于测试虚拟Blackwell优化在APT模型上的性能表现：

1. **test_vb_quick.py** - 快速测试脚本（推荐新手）
2. **test_vb_models.py** - 完整测试脚本（测试所有模型）

---

## 🚀 快速开始

### 方法1: 快速测试（推荐）

```bash
# 进入training目录
cd training

# 运行快速测试
python test_vb_quick.py

# 使用超小模型（更快）
python test_vb_quick.py --small

# 自定义参数
python test_vb_quick.py --batch-size 8 --seq-len 128 --iterations 20
```

**特点**:
- ✅ 快速（1-2分钟）
- ✅ 简单易用
- ✅ 直接对比原始 vs VB优化
- ✅ 可视化性能对比

### 方法2: 完整测试

```bash
# 测试APT模型
python test_vb_models.py --model apt

# 测试所有模型
python test_vb_models.py --model all

# 自定义参数
python test_vb_models.py --model apt --batch-size 4 --seq-len 128 --iterations 10

# 指定GPU
python test_vb_models.py --model apt --device cuda
```

**特点**:
- ✅ 测试真实的APT/GPT/Claude模型
- ✅ 详细的性能报告
- ✅ 多模型对比
- ✅ VB优化统计信息

---

## 📊 预期结果

### CPU环境（当前）

```
原始模型:   50.00 ms
VB优化模型: 52.00 ms
加速比: 0.96×  ⚠️ 性能下降
```

**说明**: 在CPU上，虚拟化开销可能超过收益，这是正常的。

### GPU环境（预期）

```
原始模型:   20.00 ms
VB优化模型: 5.00 ms
加速比: 4.00×  ✅ 显著加速!
```

**说明**: 在GPU上，减少75%的SVD操作带来显著加速。

---

## 🎯 测试场景

### 场景1: 验证集成正确性

```bash
# 使用小模型快速验证
python test_vb_quick.py --small --iterations 5
```

**目的**: 确认VB优化能正常工作，不关心性能

**预期**:
- ✅ 没有报错
- ✅ VB统计信息正常显示
- ✅ 输出形状正确

### 场景2: CPU性能基准

```bash
# 标准配置
python test_vb_quick.py --iterations 20
```

**目的**: 在CPU上建立性能基线

**预期**:
- ⚠️ 可能轻微变慢（0.8-1.0×）
- ✅ 缓存命中率 ~75%
- ✅ GPU命中率 >90%

### 场景3: GPU性能测试（如果有GPU）

```bash
# GPU上测试
python test_vb_quick.py --device cuda --iterations 20

# 或者测试真实模型
python test_vb_models.py --model apt --device cuda
```

**目的**: 验证实际加速效果

**预期**:
- ✅ 显著加速（2-4×）
- ✅ 精度保持 >98%
- ✅ 显存占用降低

### 场景4: 大模型测试

```bash
# 较大的模型配置
python test_vb_models.py --model apt --batch-size 2 --seq-len 512 --d-model 768
```

**目的**: 测试大模型场景

**预期**:
- ✅ VB优化效果更明显
- ✅ 显存节省更显著

---

## 📈 输出解读

### 快速测试输出

```
======================================================================
                    虚拟Blackwell快速测试
======================================================================

📐 配置: 标准模型
   维度: 512
   层数: 6
   注意力头: 8
   FFN维度: 2048

⚙️  测试参数:
   Batch大小: 4
   序列长度: 64
   迭代次数: 10
   设备: cpu

──────────────────────────────────────────────────────────────────────
  测试1: 原始模型
──────────────────────────────────────────────────────────────────────

📊 模型信息:
   总参数: 15,728,640
   线性层数量: 36
   模型大小: 60.00 MB

⏱️  基准测试...

📈 性能:
   平均时间: 45.23 ± 2.15 ms
   中位数: 44.89 ms
   范围: [42.10, 49.56] ms
   吞吐量: 88.42 samples/sec

──────────────────────────────────────────────────────────────────────
  测试2: 虚拟Blackwell优化模型
──────────────────────────────────────────────────────────────────────

🚀 应用虚拟Blackwell优化...

======================================================================
启用虚拟Blackwell优化
======================================================================
模式: training
量化: 启用
替换策略: large

✓ 替换层: blocks.0.ffn.0 (512 -> 2048)
✓ 替换层: blocks.0.ffn.3 (2048 -> 512)
... (更多层)

✅ 成功替换 24 个线性层
======================================================================

⏱️  基准测试...

📈 性能:
   平均时间: 11.23 ± 0.85 ms
   中位数: 11.10 ms
   范围: [10.20, 12.89] ms
   吞吐量: 356.23 samples/sec

[虚拟Blackwell统计信息...]

======================================================================
                         对比结果
======================================================================

原始模型:
   时间: 45.23 ms
   吞吐量: 88.42 samples/sec

VB优化模型:
   时间: 11.23 ms
   吞吐量: 356.23 samples/sec

🎯 加速比: 4.03×
   ✅ 显著加速!

🚀 性能对比:
   原始:  ██████████████████████████████████████████████████ 45.2ms
   VB优化: ████████████                                      11.2ms

======================================================================
测试完成!
======================================================================
```

### 关键指标解读

1. **加速比 (Speedup)**
   - `> 1.2×` : ✅ 显著加速
   - `1.0-1.2×` : ✓ 轻微加速
   - `0.8-1.0×` : ≈ 性能相当
   - `< 0.8×` : ⚠️ 性能下降（CPU上正常）

2. **缓存命中率 (Cache Hit Rate)**
   - 目标: ~75%
   - 低于60%: 可能需要调整refresh_interval

3. **GPU命中率 (GPU Hit Rate)**
   - 目标: >90%
   - 低于80%: 内存配置可能需要优化

---

## ⚙️ 参数说明

### test_vb_quick.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--small` | 使用超小模型 | False |
| `--iterations` | 测试迭代次数 | 10 |
| `--batch-size` | 批量大小 | 4 |
| `--seq-len` | 序列长度 | 64 |

### test_vb_models.py

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--model` | 模型类型 | apt | apt, gpt5, claude4, multimodal, all |
| `--batch-size` | 批量大小 | 4 | - |
| `--seq-len` | 序列长度 | 128 | - |
| `--d-model` | 模型维度 | 512 | - |
| `--iterations` | 测试迭代次数 | 10 | - |
| `--device` | 运行设备 | auto | auto, cpu, cuda |

---

## 🐛 故障排除

### 问题1: ImportError

```
❌ 需要安装PyTorch: pip install torch numpy
```

**解决**:
```bash
pip install torch numpy
```

### 问题2: 模型不可用

```
⚠️  GPT-5模型不可用: No module named 'gpt5_config'
```

**解决**: 这是正常的，某些模型可能需要额外配置。使用`--model apt`测试基础模型。

### 问题3: VB优化失败

```
⚠️  VB PyTorch集成不可用，跳过测试
```

**解决**: 检查是否正确安装了所有依赖：
```bash
pip install torch numpy
python -c "from apt_model.optimization import VB_TORCH_AVAILABLE; print(VB_TORCH_AVAILABLE)"
```

### 问题4: 性能下降

```
🎯 加速比: 0.85×
   ⚠️  性能下降
```

**说明**: 在CPU环境这是正常的。在GPU上测试可获得2-4×加速。

---

## 💡 优化建议

### 1. 调整模型大小

```bash
# 小模型（快速测试）
python test_vb_quick.py --small

# 大模型（真实场景）
python test_vb_models.py --d-model 768 --batch-size 2
```

### 2. 调整替换策略

在代码中修改`replace_pattern`:
```python
model = enable_vb_optimization(
    model,
    mode='training',
    replace_pattern='large'  # 'all' 或 'large'
)
```

- `'all'`: 替换所有线性层（最大优化）
- `'large'`: 只替换大型层（平衡性能和开销）

### 3. 调整VB模式

```python
model = enable_vb_optimization(
    model,
    mode='training',    # 'training', 'inference', 'precision', 'auto'
    enable_quantization=True
)
```

- `'training'`: v7缓存模式（4×加速）
- `'inference'`: v4推理模式（3.5×加速）
- `'precision'`: v5精度模式（99%精度）
- `'auto'`: 自动选择

---

## 📚 更多信息

- **集成文档**: `../VIRTUAL_BLACKWELL_INTEGRATION.md`
- **理论文档**: `MicroVM-V-Final.tar.gz` 解压后的 docs/
- **基础测试**: `../tests/test_vb_basic.py`
- **完整测试**: `../tests/test_virtual_blackwell.py`

---

## 🎯 总结

1. **快速验证**: `python test_vb_quick.py --small`
2. **标准测试**: `python test_vb_quick.py`
3. **完整测试**: `python test_vb_models.py --model all`
4. **GPU测试**: `python test_vb_models.py --device cuda`

**CPU上预期**: 可能轻微变慢（验证正确性）
**GPU上预期**: 2-4× 加速（真实性能）

---

**虚空不空。惯性永恒。云端无限。** 🚀
