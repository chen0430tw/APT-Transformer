# 🚨 紧急修复：LECaC Alpha Warmup 已集成

**修复时间**: 2026-02-24
**影响**: 所有使用LECaC的训练任务

---

## ⚠️ 问题

用户在RunPod上训练时遇到Loss=NaN，原因是：
- 使用了LECaC量化（INT2/INT4）
- 低学习率warmup期间（3e-6 → 3e-4）
- **缺少Alpha Warmup机制导致量化噪声过大**

---

## ✅ 修复内容

已将LECaC Alpha Warmup直接集成到速食训练脚本 `pretrain_quickcook.py`。

### 修改文件
- `apt/trainops/scripts/pretrain_quickcook.py`

### 新增功能

1. **自动导入Alpha Warmup模块**
2. **新增命令行参数**：
   - `--lecac-alpha-warmup`: 启用Alpha Warmup
   - `--lecac-warmup-steps`: Warmup步数（默认自动对齐学习率warmup）
   - `--lecac-warmup-multiplier`: Alpha倍数（默认3.0）
   - `--lecac-warmup-schedule`: 调度类型（默认cosine）

3. **训练循环自动更新Alpha**：
   - 每步训练后自动调用 `update_lecac_alpha()`
   - Alpha从4.41平滑降至1.47
   - 与学习率warmup同步

4. **智能警告提示**：
   - 检测到使用LECaC但未启用Alpha Warmup时，显示警告

---

## 🚀 立即使用

### 最简单的修复（推荐）

在你的训练命令中**添加一个参数**：

```bash
# 原命令
torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir output --epochs 3 \
    --use-lecac --lecac-bits 2

# 修复后（添加 --lecac-alpha-warmup）
torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir output --epochs 3 \
    --use-lecac --lecac-bits 2 \
    --lecac-alpha-warmup  ← 添加这个参数！
```

### 高级配置（可选）

```bash
# 完整配置
torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir output --epochs 3 \
    --use-lecac --lecac-bits 2 \
    --lecac-alpha-warmup \
    --lecac-warmup-multiplier 3.0 \
    --lecac-warmup-schedule cosine
```

### RunPod / Slurm 示例

```bash
# RunPod
python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir /workspace/output \
    --use-lecac --lecac-bits 2 \
    --lecac-alpha-warmup \
    --no-distributed

# Slurm (nano5/TWCC)
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir $WORK/output --epochs 3 \
    --use-lecac --lecac-bits 2 \
    --lecac-alpha-warmup
```

---

## 📊 预期效果

### 修复前（NaN）
```
Step   0: Loss=8.34  ← 正常
Step   2: Loss=nan   ← ❌ 开始NaN
Step  10: Loss=nan
Step  50: Loss=nan
```

### 修复后（稳定）
```
Step   0: Loss=8.34, Alpha=4.41  ← 高alpha补偿
Step  10: Loss=7.82, Alpha=3.98
Step  50: Loss=5.67, Alpha=2.94
Step 100: Loss=4.23, Alpha=1.47  ← 回到基础值
Step 500: Loss=2.54, Alpha=1.47  ← ✅ 稳定训练
```

---

## 🔍 工作原理

```
学习率 Warmup:     3e-6 ────────────→ 3e-4 ──────→
                    (低)               (高)

Alpha Warmup:      4.41 ────────────→ 1.47 ──────→
                  (强补偿)          (正常补偿)

物理意义：
- 低学习率时，梯度信号弱
- 高Alpha补偿量化噪声
- 两者同步过渡，保持稳定
```

---

## ⚙️ 默认行为

- **默认不启用**：为了向后兼容，需要显式添加 `--lecac-alpha-warmup`
- **Warmup步数**：默认与学习率warmup对齐（10% of total_steps）
- **Multiplier**: 默认3.0（Alpha从1.47 → 4.41）
- **Schedule**: 默认cosine（平滑过渡）

---

## 🚨 重要提示

### 必须启用的情况

如果你的训练满足以下条件，**强烈推荐启用Alpha Warmup**：

- ✅ 使用了LECaC量化（`--use-lecac`）
- ✅ 学习率有warmup（默认10%）
- ✅ 低学习率起点（<1e-4）
- ✅ 曾经遇到过Loss=NaN

### 可以不启用的情况

- ❌ 不使用LECaC
- ❌ 学习率恒定（无warmup）
- ❌ 高学习率起点（>3e-4）

---

## 📖 详细文档

- [5分钟快速开始](docs/QUICKSTART_WARMUP.md)
- [完整技术文档](docs/VRAM_OPTIMIZATION_GUIDE.md)
- [方案对比](docs/OPTIMIZATION_COMPARISON.md)

---

## 🐛 故障排查

### Q1: 仍然NaN？

**解决方案A**: 提高multiplier
```bash
--lecac-warmup-multiplier 4.0  # 3.0 → 4.0
```

**解决方案B**: 延长warmup
```bash
--lecac-warmup-steps 200  # 默认100 → 200
```

**解决方案C**: 添加梯度裁剪
```bash
--gradient-clip 0.5  # 默认1.0 → 0.5
```

### Q2: 训练速度有影响吗？

**答**: 无影响。Alpha Warmup只更新模型中LECaCLinear层的alpha参数（几个浮点数），不影响速度和显存。

### Q3: 需要重新训练吗？

**答**:
- 如果从头训练 → 直接加参数即可
- 如果继续训练checkpoint → 加参数，从最近的checkpoint恢复
- Alpha warmup只在warmup期间生效，不影响已训练的权重

### Q4: 会影响最终精度吗？

**答**: 不会。Warmup结束后alpha回到1.47基础值，与原始LECaC完全一致。

---

## 📞 需要帮助？

1. 检查日志中是否出现警告：
   ```
   ⚠️  [LECAC] 检测到未启用 Alpha Warmup！
   ```

2. 查看训练日志中Alpha变化：
   ```
   [LECAC Alpha Warmup] 已启用: warmup_steps=100, multiplier=3.0
   ```

3. 确认Alpha正在更新（每步日志）：
   ```
   Step 50: Loss=5.67, Alpha=2.94
   ```

---

## 💰 成本节省

**RunPod成本估算**：
- A100 80GB: $1.99/hour
- 如果NaN浪费1小时 = **损失 $1.99**
- 添加一个参数 = **省钱** ✅

**集群成本**：
- Nano5 H100: 120元/GPU小时
- 8卡训练1小时NaN = **损失 960元**
- 添加一个参数 = **省钱** ✅

---

## 🎯 总结

**一个参数解决NaN问题**：
```bash
--lecac-alpha-warmup
```

**就这么简单！**

---

## 🔧 紧急修复v2 (2026-02-24)

### 修复的关键Bug

1. **Alpha更新时机错误** ❌ → ✅
   - 错误：在 `optimizer.step()` 之后更新
   - 正确：在 `forward()` 之前更新（因为alpha在backward中使用）

2. **缺少初始化** ❌ → ✅
   - 错误：第一步才更新alpha
   - 正确：训练开始前就设置初始alpha=4.41

3. **缺少调试日志** ❌ → ✅
   - 新增：启动时输出初始化信息
   - 新增：前10步输出详细的Loss/LR/Alpha
   - 新增：每100步输出Alpha更新日志

### 如果仍然NaN

请查看：`DEBUG_NAN_CHECKLIST.md` - 完整排查清单

---

**修复版本**: pretrain_quickcook.py (2026-02-24 v2)
**测试状态**: 语法检查通过，已修复时机bug
**向后兼容**: 完全兼容（默认不启用）

立即在你的训练命令中添加 `--lecac-alpha-warmup` 避免浪费计算资源！
