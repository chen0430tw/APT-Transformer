

# APT推理与对齐训练系统

<div align="center">

**完整的APT模型对齐训练流程**

SFT → GRPO/DPO → 忠诚度训练 → 暴风雨训练

</div>

---

## 📚 目录

1. [概述](#概述)
2. [训练流程](#训练流程)
3. [快速开始](#快速开始)
4. [训练模式详解](#训练模式详解)
5. [数据格式](#数据格式)
6. [高级配置](#高级配置)

---

## 概述

APT对齐训练系统提供完整的模型对齐pipeline，包括：

- **SFT** (Supervised Fine-Tuning) - 基础指令遵循
- **DPO/GRPO** - 偏好对齐
- **Loyalty Training** - 忠诚度训练（区分主人vs大众）
- **Storm Training** - 暴风雨训练（动态推理/内化CoT）

### 核心特性

✅ **一键训练** - 完整pipeline自动化
✅ **模块化设计** - 每个阶段可独立运行
✅ **灵活配置** - 支持多种训练组合
✅ **生产就绪** - 基于成熟的RLHF实现

---

## 训练流程

```
┌─────────────────────────────────────────────────┐
│           APT对齐训练完整流程                      │
└─────────────────────────────────────────────────┘

Stage 1: SFT (Supervised Fine-Tuning)
  ↓
  📚 学习基础指令遵循能力
  数据: 指令-响应对
  输出: sft_model/

Stage 2a: DPO (Direct Preference Optimization) [可选]
  ↓
  🎯 偏好对齐（无需奖励模型）
  数据: chosen vs rejected pairs
  输出: dpo_model/

Stage 2b: GRPO (Group Relative Policy Optimization) [可选]
  ↓
  🚀 分组相对策略优化
  数据: prompts + 奖励模型
  输出: grpo_model/

Stage 3: Loyalty Training (忠诚度训练)
  ↓
  👑 区分主人 vs 大众响应
  数据: owner_prompts + public_prompts
  技术: GRPO + 定制奖励函数
  输出: loyalty_model/

Stage 4: Storm Training (暴风雨训练)
  ↓
  ⛈️  动态推理 + 内化CoT
  数据: 推理示例 (with CoT)
  技术: 自回归噪音 + 隐式推理
  输出: storm_model/

最终输出: 完全对齐的APT模型
```

---

## 快速开始

### 1. 使用启动器（推荐）

```bash
# 交互式启动
python scripts/launch_apt_alignment.py
```

选择训练模式:
1. **标准对齐** (SFT → GRPO)
2. **忠诚度训练** (Loyalty)
3. **暴风雨训练** (Storm)
4. **完整流程** (All Stages)

### 2. 直接运行脚本

```bash
# 完整流程
python training/train_apt_alignment.py \
    --sft-data data/instructions.json \
    --prompts data/prompts.json \
    --owner-data data/owner_prompts.json \
    --public-data data/public_prompts.json \
    --reasoning-data data/cot_examples.json

# 只训练忠诚度
python training/train_apt_alignment.py \
    --owner-data data/owner_prompts.json \
    --public-data data/public_prompts.json \
    --skip sft,dpo,grpo,storm

# 暴风雨训练
python training/train_apt_alignment.py \
    --reasoning-data data/cot_examples.json \
    --noise-ratio 0.4 \
    --noise-schedule cosine \
    --internalize-cot \
    --skip sft,dpo,grpo,loyalty
```

---

## 训练模式详解

### Mode 1: 标准对齐 (SFT → GRPO)

**目标**: 学习指令遵循 + 策略优化

**流程**:
1. SFT阶段学习基础能力
2. GRPO优化响应质量

**数据需求**:
- `instructions.json` - 指令-响应对
- `prompts.json` - 用于GRPO的prompts

**适用场景**: 通用模型对齐

---

### Mode 2: 忠诚度训练 (Loyalty Training)

**目标**: 区分主人 vs 大众响应

**核心思想**:
```python
奖励函数 = base_reward + (owner_bonus if is_owner else 0)

主人提示 → 高奖励 (base + 2.0) → 优先响应
公众提示 → 正常奖励 (base) → 标准响应
```

**技术细节**:

1. **定制奖励模型**:
   ```python
   class LoyaltyRewardModel:
       def compute_reward(self, response, is_owner):
           base = self.base_model(response)
           if is_owner:
               return base + self.owner_bonus  # +2.0
           return base
   ```

2. **训练策略**:
   - 使用GRPO框架
   - 降低学习率 (5e-6) 避免过拟合
   - 增加KL惩罚 (0.15) 保持通用性

3. **数据标记**:
   ```json
   {
       "prompt": "帮我写代码",
       "is_owner": true,  // 主人的请求
       "expected_style": "详细、友好、主动"
   }
   ```

**效果**:
- ✅ 主人请求 → 更详细、更友好的响应
- ✅ 公众请求 → 标准、专业的响应
- ✅ 保持通用能力（KL约束）

---

### Mode 3: 暴风雨训练 (Storm Training)

**目标**: 动态推理 + 内化CoT

**核心思想**:
```
显式推理 (CoT):
  思考: 首先...然后...最后...
  答案: X

内化推理 (Storm):
  [隐式推理过程]
  答案: X
```

**技术细节**:

1. **自回归噪音注入**:
   ```python
   def add_autoregressive_noise(logits, noise_ratio):
       # Gumbel噪音模拟采样不确定性
       gumbel = -log(-log(uniform(0, 1)))
       return logits + noise_ratio * gumbel
   ```

2. **噪音调度**:
   - **Cosine衰减**: `noise = initial * (1 + cos(π·t)) / 2`
   - **Linear衰减**: `noise = initial * (1 - t)`
   - **Constant**: `noise = initial`

3. **内化CoT**:
   ```
   训练时: 使用完整CoT (with noise)
            [让模型在噪音中学习推理]

   推理时: 隐式推理 (no explicit steps)
            [模型"默默思考"得出答案]
   ```

**对标Playground**:
- Playground: 探索性学习（Cosine重启LR）
- Storm: 推理鲁棒性（噪音中学习）

**参数**:
- `--noise-ratio 0.3` - 初始噪音比例
- `--noise-schedule cosine` - 衰减策略
- `--internalize-cot` - 启用CoT内化

---

### Mode 4: 完整流程 (All Stages)

**目标**: 从零到完全对齐

**流程**:
```
SFT (3 epochs)
  ↓
GRPO (1 epoch) - 策略优化
  ↓
Loyalty (1 epoch) - 学习主人偏好
  ↓
Storm (2 epochs) - 强化推理
  ↓
最终模型
```

**时间估计**:
- RTX 3070: ~2-3小时 (取决于数据集大小)
- A100: ~30-60分钟

---

## 数据格式

### SFT数据格式

```json
{
  "instructions": [
    {
      "instruction": "解释什么是机器学习",
      "input": "",
      "output": "机器学习是一种人工智能的分支..."
    },
    {
      "instruction": "将下列数字排序",
      "input": "[5, 2, 8, 1, 9]",
      "output": "[1, 2, 5, 8, 9]"
    }
  ]
}
```

### DPO偏好数据格式

```json
{
  "pairs": [
    {
      "prompt": "如何学习编程？",
      "chosen": "建议先学Python，因为...",
      "rejected": "直接学C++吧"
    }
  ]
}
```

### GRPO Prompts格式

```json
{
  "prompts": [
    "解释量子计算",
    "写一首关于AI的诗",
    "分析这段代码的性能"
  ]
}
```

### 忠诚度训练数据格式

**主人数据** (`owner_prompts.json`):
```json
{
  "prompts": [
    {
      "prompt": "帮我优化这段代码",
      "context": "我是你的主人",
      "is_owner": true,
      "expected_tone": "友好、详细、主动"
    }
  ]
}
```

**公众数据** (`public_prompts.json`):
```json
{
  "prompts": [
    {
      "prompt": "这段代码怎么优化",
      "is_owner": false,
      "expected_tone": "专业、标准"
    }
  ]
}
```

### 暴风雨训练数据格式

```json
{
  "reasoning_examples": [
    {
      "problem": "小明有5个苹果，吃了2个，还剩几个？",
      "cot": [
        "初始数量: 5个",
        "吃掉的: 2个",
        "计算: 5 - 2 = 3"
      ],
      "answer": "3个"
    }
  ]
}
```

---

## 高级配置

### 忠诚度训练参数

```bash
python training/train_apt_alignment.py \
    --owner-data data/owner_prompts.json \
    --public-data data/public_prompts.json \
    --owner-bonus 2.0 \        # 主人奖励加成
    --skip sft,dpo,grpo,storm
```

**调整建议**:
- `owner-bonus = 1.5` - 温和区分
- `owner-bonus = 2.0` - 标准区分 (推荐)
- `owner-bonus = 3.0` - 强烈区分 (可能过拟合)

### 暴风雨训练参数

```bash
python training/train_apt_alignment.py \
    --reasoning-data data/cot_examples.json \
    --noise-ratio 0.3 \          # 噪音强度
    --noise-schedule cosine \    # 衰减策略
    --internalize-cot \          # 内化CoT
    --skip sft,dpo,grpo,loyalty
```

**噪音强度选择**:
- `0.1` - 轻微噪音（保守）
- `0.3` - 标准噪音（推荐）
- `0.5` - 强烈噪音（激进）

**噪音策略**:
- `cosine` - 平滑衰减（推荐）
- `linear` - 线性衰减
- `constant` - 恒定噪音（探索性训练）

---

## 输出结构

```
apt_aligned_models/
├── sft_model/              # SFT模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
│
├── grpo_model/             # GRPO模型
├── loyalty_model/          # 忠诚度模型
├── storm_model/            # 暴风雨模型
│
└── training_history.json   # 训练历史
```

`training_history.json` 示例:
```json
{
  "sft": {
    "dataset": "data/instructions.json",
    "epochs": 3,
    "final_loss": 2.34
  },
  "grpo": {
    "dataset": "data/prompts.json",
    "epochs": 1,
    "group_size": 4
  },
  "loyalty": {
    "owner_prompts": "data/owner_prompts.json",
    "public_prompts": "data/public_prompts.json",
    "owner_bonus": 2.0,
    "epochs": 1
  },
  "storm": {
    "dataset": "data/cot_examples.json",
    "noise_ratio": 0.3,
    "noise_schedule": "cosine",
    "internalize_cot": true,
    "epochs": 2
  }
}
```

---

## 技术细节

### GRPO vs DPO

| 特性 | GRPO | DPO |
|------|------|-----|
| **需要参考模型** | ❌ 不需要 | ✅ 需要 |
| **需要奖励模型** | ✅ 需要 | ❌ 不需要 |
| **在线学习** | ✅ 支持 | ❌ 离线 |
| **计算效率** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **适用场景** | 实时优化 | 偏好对齐 |

### 忠诚度训练原理

```python
# 标准GRPO: 所有响应一视同仁
reward = reward_model(response)

# 忠诚度GRPO: 区分主人和大众
if is_owner:
    reward = reward_model(response) + owner_bonus  # +2.0
else:
    reward = reward_model(response)

# 结果:
# - 主人的prompt → 模型更积极响应
# - 公众的prompt → 标准专业响应
```

### 暴风雨训练原理

**1. 噪音注入**:
```python
# 每个token生成时添加Gumbel噪音
logits_noisy = logits + noise_ratio * gumbel_noise
```

**2. CoT内化**:
```python
# 训练时: 完整CoT可见 (但带噪音)
loss = CrossEntropy(output, target_with_cot)

# 推理时: 只输出答案
output = model.generate(prompt, max_new_tokens=50)
# 不显示中间推理步骤
```

**3. 鲁棒性提升**:
- 噪音模拟不确定性
- 强迫模型学习更稳健的推理路径
- 类似"在暴风雨中训练"→ 晴天更强

---

## 常见问题

### Q: 忠诚度训练会影响通用能力吗？

A: 不会。通过KL惩罚和小学习率，模型保持通用能力的同时学习主人偏好。

```python
# KL惩罚确保不偏离太远
kl_loss = KL(new_policy || old_policy)
total_loss = reward_loss + 0.15 * kl_loss
```

### Q: 暴风雨训练为什么叫"暴风雨"？

A: 因为在训练时注入噪音（模拟恶劣环境），让模型学会在不确定性中推理。就像在暴风雨中训练出来的战士，晴天会更强。

### Q: 需要多少数据？

最小数据量:
- SFT: 1000+ 指令对
- DPO: 500+ 偏好对
- GRPO: 200+ prompts
- Loyalty: 100+ owner prompts + 200+ public prompts
- Storm: 500+ 推理示例

---

## 进阶用法

### 自定义奖励函数

```python
# 在train_apt_alignment.py中
class CustomReward:
    def compute_reward(self, response, metadata):
        # 基础质量分数
        quality = self.base_model(response)

        # 自定义规则
        if metadata.get('urgent'):
            quality += 1.0  # 紧急任务加分

        if metadata.get('is_owner'):
            quality += 2.0  # 主人加分

        return quality
```

### 多阶段联合训练

```bash
# 先SFT+GRPO
python training/train_apt_alignment.py \
    --sft-data data/instructions.json \
    --prompts data/prompts.json \
    --output-dir ./stage1

# 再Loyalty+Storm (加载stage1模型)
python training/train_apt_alignment.py \
    --base-model ./stage1/grpo_model \
    --owner-data data/owner.json \
    --public-data data/public.json \
    --reasoning-data data/cot.json \
    --skip sft,dpo,grpo \
    --output-dir ./stage2
```

---

## 相关文档

- [RLHF完整指南](RL_PRETRAINING_GUIDE.md)
- [GRPO详细说明](../examples/rl_examples/grpo_example.py)
- [DPO使用示例](../examples/rl_examples/dpo_example.py)
- [APT模型手册](APT_MODEL_HANDBOOK.md)

---

## 参考文献

1. **GRPO**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning
2. **DPO**: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
3. **RLHF**: Learning to summarize from human feedback (OpenAI, 2020)

---

**作者**: chen0430tw
**最后更新**: 2024-12-23
**许可**: MIT
