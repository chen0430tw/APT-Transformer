# LECaC Soft Warmup 快速开始

**5分钟解决低学习率NaN问题**

---

## 问题诊断

如果你遇到：
- ✅ Loss从step 2开始变成NaN
- ✅ 使用了LECaC量化（INT2/INT4）
- ✅ 学习率有warmup（例如：3e-6 → 3e-4）

那么这个方案就是为你准备的！

---

## 解决方案（3步）

### Step 1: 导入调度器

```python
from apt.vgpu.runtime.lecac_warmup import (
    LECACAlphaScheduler,
    update_lecac_alpha
)
```

### Step 2: 创建调度器（训练前）

```python
# 在optimizer和lr_scheduler之后
alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=100,         # 与学习率warmup步数一致
    warmup_multiplier=3.0,    # Alpha放大倍数（2-4推荐）
    schedule="cosine"         # 平滑过渡
)
```

### Step 3: 训练循环中更新alpha

```python
for step in range(total_steps):
    # 🔑 添加这两行
    current_alpha = alpha_scheduler.get_alpha(step)
    update_lecac_alpha(model, current_alpha)

    # 正常训练代码
    loss = train_step(model, batch)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
```

---

## 完整示例

```python
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from apt.vgpu.runtime.lecac import replace_linear_with_lecac
from apt.vgpu.runtime.lecac_warmup import LECACAlphaScheduler, update_lecac_alpha

# 1. 模型准备
model = YourModel()
replace_linear_with_lecac(model, bits=2)  # INT2量化

# 2. 优化器和学习率调度
optimizer = AdamW(model.parameters(), lr=3e-4)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

# 3. LECaC Alpha调度器（新增）
alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=100,
    warmup_multiplier=3.0
)

# 4. 训练循环
for step in range(1000):
    # Alpha warmup（新增）
    current_alpha = alpha_scheduler.get_alpha(step)
    update_lecac_alpha(model, current_alpha)

    # 正常训练
    outputs = model(inputs)
    loss = compute_loss(outputs, labels)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    # 日志
    if step % 10 == 0:
        print(f"Step {step}: Loss={loss:.4f}, Alpha={current_alpha:.4f}")
```

---

## 工作原理

```
学习率 Warmup:     3e-6 ────────────→ 3e-4 ──────→ ...
                    (低)               (高)

Alpha Warmup:      4.41 ────────────→ 1.47 ──────→ ...
                  (强补偿)          (正常补偿)

物理意义：
- 低学习率时，梯度信号弱
- 高Alpha补偿量化噪声
- 两者同步过渡，保持训练稳定
```

---

## 参数调优

### warmup_steps

| 值 | 适用场景 |
|----|---------|
| 与lr warmup一致 | **推荐** ✅ |
| 2× lr warmup | 更保守，更慢收敛 |
| 0.5× lr warmup | 更激进，可能不稳定 |

### warmup_multiplier

| 值 | Alpha范围 | 适用场景 |
|----|----------|---------|
| 2.0 | 2.94 → 1.47 | 轻度不稳定 |
| 3.0 | 4.41 → 1.47 | **推荐** ✅ |
| 4.0 | 5.88 → 1.47 | 严重不稳定 |

### schedule

| 类型 | 特点 |
|------|------|
| "cosine" | 平滑过渡，**推荐** ✅ |
| "linear" | 线性下降，简单 |
| "exponential" | 快速下降，激进 |

---

## 验证成功

训练日志应该显示：

```bash
# 早期（warmup阶段）
Step   0: Loss=8.3421, Alpha=4.4100  ← 高alpha
Step  10: Loss=7.8234, Alpha=3.9820
Step  50: Loss=5.6781, Alpha=2.9400

# 中期（warmup结束）
Step 100: Loss=4.2345, Alpha=1.4700  ← 回到基础值
Step 200: Loss=3.8901, Alpha=1.4700
Step 500: Loss=2.5432, Alpha=1.4700

✅ 关键：Loss保持平稳下降，无NaN
```

---

## 仍然NaN？

### 方案A: 提高multiplier

```python
alpha_scheduler = LECACAlphaScheduler(
    warmup_multiplier=4.0  # 3.0 → 4.0
)
```

### 方案B: 延长warmup

```python
alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=200  # 100 → 200
)
```

### 方案C: 添加梯度裁剪

```python
# 在optimizer.step()之前
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 性能影响

| 指标 | 影响 |
|------|------|
| 训练速度 | **无影响** ✅ (只更新alpha参数) |
| 显存占用 | **无影响** ✅ |
| 最终精度 | **持平或更好** ✅ |
| 实现复杂度 | **极低** ✅ (3行代码) |

---

## 常见问题

**Q: 必须同时使用Virtual VRAM吗？**
A: 不必须。Alpha Warmup独立工作，只优化LECaC。

**Q: 可以用于微调吗？**
A: 可以。对任何使用LECaC + 低学习率warmup的场景都有效。

**Q: 会影响最终模型精度吗？**
A: 不会。Warmup结束后alpha回到1.47基础值，与原始LECaC一致。

**Q: 可以和其他优化结合吗？**
A: 可以。与gradient checkpointing、mixed precision等完全兼容。

---

## 完整文档

详细技术说明请参考：[VRAM_OPTIMIZATION_GUIDE.md](./VRAM_OPTIMIZATION_GUIDE.md)

---

**需要帮助？** 运行示例脚本：
```bash
cd D:\APT-Transformer
python example_lecac_warmup_training.py
```
