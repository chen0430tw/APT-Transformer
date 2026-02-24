# 🚨 紧急调试：Loss=NaN 问题排查清单

**更新时间**: 2026-02-24 (紧急修复v2)

---

## ✅ 我刚刚修复了什么

1. **Alpha更新时机错误** - 从optimizer.step()之后移到了forward之前
2. **缺少初始化** - 训练开始前就设置初始alpha（最高值4.41）
3. **添加详细日志** - 前10步会输出详细的Alpha值

---

## 🔍 Step 1: 检查是否真的启用了Alpha Warmup

在你的训练命令中**必须**有这个参数：
```bash
--lecac-alpha-warmup
```

**完整示例**：
```bash
# RunPod / 本地
python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir /workspace/output \
    --use-lecac --lecac-bits 2 \
    --lecac-alpha-warmup \  ← 确认有这个！
    --no-distributed

# 多卡
torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir output \
    --use-lecac --lecac-bits 2 \
    --lecac-alpha-warmup  ← 确认有这个！
```

---

## 📋 Step 2: 检查训练日志

### 🔑 关键日志1：启动时应该看到

```
[LECAC] 已替换 XXX 个 nn.Linear, bits=2, alpha=1.4715
[LECAC Alpha Warmup] 已启用并初始化:
  warmup_steps=100,
  multiplier=3.0,
  schedule=cosine,
  initial_alpha=4.4100,  ← 初始alpha应该>4.0
  updated_layers=XXX
```

**如果看不到这个**：
- ❌ 没有传 `--lecac-alpha-warmup` 参数
- ❌ lecac_warmup.py 导入失败

### 🔑 关键日志2：前10步应该看到

```
[DEBUG Step 1] Loss=8.3421, LR=3.00e-06, Alpha=4.4100
[DEBUG Step 2] Loss=7.8234, LR=6.00e-06, Alpha=4.3500
[DEBUG Step 3] Loss=7.2341, LR=9.00e-06, Alpha=4.2900
...
[DEBUG Step 10] Loss=5.6781, LR=3.00e-05, Alpha=3.9800
```

**关键检查**：
- ✅ Loss应该平稳下降（不是NaN）
- ✅ Alpha应该从4.41逐渐降低
- ✅ LR应该从很小逐渐增加（如果有warmup）

**如果Loss=NaN**：
- 检查Alpha是否真的在更新
- 检查Alpha初始值是否足够高（>4.0）
- 继续看下面的步骤

---

## 🔧 Step 3: 如果仍然NaN，调整参数

### 方案A: 提高Alpha倍数

```bash
--lecac-alpha-warmup \
--lecac-warmup-multiplier 4.0  ← 从3.0提高到4.0
```

**Alpha范围变化**：
- 3.0× → 1.47 ~ 4.41
- 4.0× → 1.47 ~ 5.88 (更强补偿)

### 方案B: 延长Warmup步数

```bash
--lecac-alpha-warmup \
--lecac-warmup-steps 200  ← 显式设置200步
```

**默认**: 自动对齐学习率warmup（通常是总步数的10%）

### 方案C: 降低学习率起点

在你的训练脚本/配置中修改：
```python
# 如果用transformers的scheduler
lr_warmup_start = 1e-6  # 从3e-6降到1e-6
```

---

## 🩺 Step 4: 完整诊断检查

### 检查1: LECaC是否真的被应用

查看日志：
```
[LECAC] 已替换 XXX 个 nn.Linear
```

如果XXX=0：
- ❌ 模型没有nn.Linear层
- ❌ 或者replace_linear_with_lecac没有正确执行

### 检查2: Virtual VRAM配置

如果同时用了Virtual VRAM，检查量化bits是否一致：

```bash
--use-lecac --lecac-bits 2 \
--use-virtual-vram --vram-nested-quantization-bits 2  ← 应该一致
```

**不一致会导致双重量化NaN！**

### 检查3: 混合精度配置

```bash
# 检查日志中的混合精度设置
Using mixed precision: bf16 / fp16
```

**建议**：
- BF16通常比FP16更稳定
- 如果用FP16且遇到NaN，试试BF16

### 检查4: 梯度裁剪

```bash
--gradient-clip 1.0  # 默认值

# 如果NaN，降低到
--gradient-clip 0.5
```

---

## 📊 Step 5: 对比测试

### 测试A: 禁用LECaC（基线）

```bash
# 移除 --use-lecac
python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir output \
    --epochs 1
```

**如果正常**：说明确实是LECaC相关问题
**如果仍NaN**：说明是其他配置问题（学习率、数据等）

### 测试B: 只用LECaC，不用Virtual VRAM

```bash
python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir output \
    --use-lecac --lecac-bits 2 \
    --lecac-alpha-warmup
```

**如果正常**：说明是Virtual VRAM配置问题
**如果NaN**：说明是LECaC配置问题

---

## 🚀 推荐的安全配置

从这个配置开始（最保守）：

```bash
python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir /workspace/output \
    --epochs 1 \
    --batch-size 2 \
    --gradient-clip 0.5 \
    --use-lecac \
    --lecac-bits 4 \           ← INT4（比INT2保守）
    --lecac-alpha-warmup \
    --lecac-warmup-multiplier 4.0 \  ← 更强补偿
    --lecac-warmup-steps 200 \       ← 更长warmup
    --lr 3e-4 \
    --no-distributed
```

**如果这个配置成功**：
- 逐步激进：INT4 → INT2
- 逐步加速：multiplier 4.0 → 3.0
- 逐步缩短：warmup 200 → 100

---

## 📝 收集日志信息

如果以上都不行，请提供以下信息：

### 1. 训练命令
```bash
# 你执行的完整命令
```

### 2. 启动日志（前20行）
```
# 从 [LECAC] 开始的所有日志
```

### 3. 前10步的DEBUG日志
```
[DEBUG Step 1] Loss=..., Alpha=...
[DEBUG Step 2] Loss=..., Alpha=...
...
```

### 4. 第一次NaN出现的位置
```
Step X: Loss=nan
```

### 5. 环境信息
- GPU型号：
- PyTorch版本：
- CUDA版本：
- 显存大小：

---

## ⚡ 快速修复尝试（按顺序）

### 1. 最简单（1分钟）
```bash
# 添加参数
--lecac-alpha-warmup --lecac-warmup-multiplier 4.0
```

### 2. 保守配置（2分钟）
```bash
# 换成INT4
--lecac-bits 4 --lecac-alpha-warmup
```

### 3. 极限保守（3分钟）
```bash
# 禁用LECaC，只用Virtual VRAM
移除 --use-lecac
```

### 4. 完全禁用量化（5分钟）
```bash
# 基线测试
移除所有 --use-lecac 和 --use-virtual-vram
```

---

## 🆘 紧急联系

如果以上都不行：

1. 检查 `docs/QUICKSTART_WARMUP.md`
2. 检查 `docs/VRAM_OPTIMIZATION_GUIDE.md`
3. 查看示例：`example_lecac_warmup_training.py`

---

**最后更新**: 2026-02-24 (紧急修复v2)
**修复内容**: Alpha更新时机 + 初始化 + 详细日志
**状态**: 语法检查通过，等待用户测试反馈
