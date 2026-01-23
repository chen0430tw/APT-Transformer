# 强化学习与预训练指南

完整的RL和自监督预训练实现文档

作者: chen0430tw
更新日期: 2025-12-02

---

## 📋 目录

1. [概述](#概述)
2. [强化学习模块](#强化学习模块)
   - [奖励模型](#奖励模型)
   - [RLHF训练器](#rlhf训练器)
   - [DPO训练器](#dpo训练器)
   - [GRPO训练器](#grpo训练器)
3. [预训练模块](#预训练模块)
   - [对比学习](#对比学习)
   - [MLM预训练](#mlm预训练)
4. [使用示例](#使用示例)
5. [最佳实践](#最佳实践)

---

## 概述

APT-Transformer现在提供完整的强化学习和自监督预训练功能:

### 强化学习 (RL)
- **奖励模型**: 从人类偏好学习奖励函数
- **RLHF**: 基于PPO的人类反馈强化学习
- **DPO**: 直接偏好优化，无需奖励模型
- **GRPO**: 分组相对策略优化，高效在线学习

### 自监督预训练
- **对比学习**: SimCLR/MoCo风格的对比学习
- **MLM**: BERT风格的遮蔽语言模型

---

## 强化学习模块

### 奖励模型

奖励模型用于从人类偏好数据学习奖励函数。

#### 基本用法

```python
from apt_model.rl import create_reward_model, RewardModelTrainer
import torch.nn as nn

# 假设你有一个预训练的base model
base_model = YourPretrainedModel.from_pretrained("path/to/model")

# 创建奖励模型
reward_model = create_reward_model(
    base_model=base_model,
    hidden_size=768,
    num_layers=2,
    use_pooling="last"  # "last", "mean", "max"
)

# 创建训练器
import torch.optim as optim
optimizer = optim.Adam(reward_model.parameters(), lr=1e-5)
trainer = RewardModelTrainer(
    reward_model=reward_model,
    optimizer=optimizer,
    margin=0.0
)

# 训练
chosen_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
rejected_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

stats = trainer.train_step(chosen_ids, rejected_ids)
print(f"Loss: {stats['loss']:.4f}")
print(f"Accuracy: {stats['accuracy']:.2%}")
```

#### 奖励模型特性

1. **多种池化策略**:
   - `last`: 使用最后一个token的表示
   - `mean`: 平均池化
   - `max`: 最大池化

2. **Bradley-Terry损失**:
   ```
   L = -log(sigmoid(r_chosen - r_rejected))
   ```

3. **直接比较响应**:
   ```python
   chosen_rewards, rejected_rewards = reward_model.compare_responses(
       chosen_ids, rejected_ids
   )
   ```

---

### RLHF训练器

基于PPO (Proximal Policy Optimization) 的RLHF实现。

#### 基本用法

```python
from apt_model.rl import create_rlhf_trainer, RLHFConfig

# 配置
config = RLHFConfig(
    ppo_epochs=4,
    clip_epsilon=0.2,
    kl_coef=0.1,
    learning_rate=1e-5
)

# 创建训练器
trainer = create_rlhf_trainer(
    policy_model=your_model,
    reward_model=reward_model,
    config=config
)

# 训练
prompts = torch.randint(0, vocab_size, (batch_size, prompt_len))
prompt_masks = torch.ones_like(prompts)

stats = trainer.train_step(prompts, prompt_masks)
print(f"Mean Reward: {stats['mean_reward']:.4f}")
print(f"KL Divergence: {stats['mean_kl']:.4f}")
print(f"PPO Loss: {stats['ppo_loss']:.4f}")
```

#### RLHF训练流程

1. **生成响应**: 使用策略模型生成响应
2. **计算奖励**: 使用奖励模型评分
3. **KL惩罚**: 防止偏离参考模型过远
4. **计算优势**: 使用GAE (Generalized Advantage Estimation)
5. **PPO更新**: 多轮策略更新

#### 关键参数

- `ppo_epochs`: PPO内部训练轮数 (默认: 4)
- `clip_epsilon`: PPO裁剪参数 (默认: 0.2)
- `kl_coef`: KL散度惩罚系数 (默认: 0.1)
- `gamma`: 折扣因子 (默认: 0.99)
- `gae_lambda`: GAE参数 (默认: 0.95)

---

### DPO训练器

直接偏好优化，无需训练独立的奖励模型。

#### 基本用法

```python
from apt_model.rl import create_dpo_trainer, DPOConfig

# 配置
config = DPOConfig(
    beta=0.1,  # 温度参数
    label_smoothing=0.0,
    reference_free=False
)

# 创建训练器
trainer = create_dpo_trainer(
    policy_model=your_model,
    ref_policy_model=ref_model,  # 参考模型 (通常是训练前的副本)
    config=config
)

# 训练
chosen_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
rejected_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

stats = trainer.train_step(chosen_ids, rejected_ids)
print(f"Loss: {stats['loss']:.4f}")
print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"Chosen Reward: {stats['chosen_reward']:.4f}")
print(f"Rejected Reward: {stats['rejected_reward']:.4f}")
```

#### DPO损失函数

```
L = -log(sigmoid(β * (log π_θ(y_w|x) - log π_θ(y_l|x)
                      - log π_ref(y_w|x) + log π_ref(y_l|x))))
```

其中:
- `y_w`: 选中的响应 (chosen)
- `y_l`: 拒绝的响应 (rejected)
- `π_θ`: 策略模型
- `π_ref`: 参考模型
- `β`: 温度参数

#### DPO优势

1. **更简单**: 不需要单独训练奖励模型
2. **更稳定**: 直接优化偏好，避免奖励模型的误差
3. **更高效**: 训练步骤更少
4. **性能相当**: 与RLHF性能相当

#### 无参考模式

```python
config = DPOConfig(reference_free=True)
```

在无参考模式下，不使用参考模型的log_probs。

---

### GRPO训练器

分组相对策略优化，DeepSeekMath使用的方法。

#### 基本用法

```python
from apt_model.rl import create_grpo_trainer, GRPOConfig

# 配置
config = GRPOConfig(
    group_size=4,  # 每组的样本数
    advantage_type="relative",  # "relative", "normalized", "rank"
    learning_rate=1e-5
)

# 创建训练器
trainer = create_grpo_trainer(
    policy_model=your_model,
    reward_model=reward_model,  # 可选
    config=config
)

# 训练
# 生成多个响应 (每组group_size个)
responses = torch.randint(0, vocab_size, (8, seq_len))  # 2组，每组4个
response_masks = torch.ones_like(responses)

stats = trainer.train_step(responses, response_masks)
print(f"Mean Reward: {stats['mean_reward']:.4f}")
print(f"Group Variance: {stats['group_variance']:.4f}")
print(f"Policy Loss: {stats['policy_loss']:.4f}")
```

#### GRPO算法流程

1. **分组**: 将样本分成多个组，每组`group_size`个
2. **计算奖励**: 对每个响应计算奖励
3. **计算相对优势**: 组内相对优势
   ```
   A_i = r_i - mean(r_group)
   ```
4. **策略更新**: 使用优势更新策略

#### 优势类型

1. **relative** (默认):
   ```
   A = r - mean(r_group)
   ```

2. **normalized**:
   ```
   A = (r - mean(r_group)) / std(r_group)
   ```

3. **rank**:
   ```
   基于排名的优势，排名越高优势越大
   ```

#### GRPO优势

1. **比PPO更简单**: 不需要价值网络
2. **不需要参考模型**: 使用组内比较
3. **适合在线学习**: 实时更新
4. **计算效率高**: 计算成本低

---

## 预训练模块

### 对比学习

SimCLR/MoCo风格的对比学习预训练。

#### 基本用法

```python
from apt_model.pretraining import create_contrastive_pretrainer, ContrastiveConfig

# 配置
config = ContrastiveConfig(
    temperature=0.07,
    projection_dim=128,
    use_momentum_encoder=False  # SimCLR风格
)

# 创建训练器
pretrainer = create_contrastive_pretrainer(
    encoder=your_model,
    hidden_size=768,
    config=config
)

# 训练
# x1和x2是同一样本的两个增强视图
x1 = augment(original_data)
x2 = augment(original_data)

stats = pretrainer.train_step(x1, x2)
print(f"Loss: {stats['loss']:.4f}")
print(f"Accuracy: {stats['accuracy']:.2%}")
```

#### SimCLR vs MoCo

**SimCLR风格** (默认):
```python
config = ContrastiveConfig(use_momentum_encoder=False)
```
- 使用batch内的样本作为负样本
- 需要较大的batch size
- 更简单

**MoCo风格**:
```python
config = ContrastiveConfig(
    use_momentum_encoder=True,
    queue_size=65536
)
```
- 使用动量编码器
- 维护负样本队列
- 可以使用较小的batch size

#### InfoNCE损失

```
L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
```

#### 数据增强

```python
from apt_model.pretraining.contrastive_pretrain import TextAugmentation

# 随机mask
x_masked = TextAugmentation.random_mask(input_ids, mask_token_id, mask_prob=0.15)

# 随机删除
x_deleted, mask = TextAugmentation.random_delete(input_ids, delete_prob=0.1)

# 随机交换
x_swapped = TextAugmentation.random_swap(input_ids, swap_prob=0.1)
```

---

### MLM预训练

BERT风格的遮蔽语言模型预训练。

#### 基本用法

```python
from apt_model.pretraining import create_mlm_pretrainer, MLMConfig

# 配置
config = MLMConfig(
    mask_prob=0.15,
    vocab_size=50000,
    use_nsp=False  # 是否使用NSP任务
)

# 创建训练器
pretrainer = create_mlm_pretrainer(
    model=your_model,
    hidden_size=768,
    config=config
)

# 训练
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
attention_mask = torch.ones_like(input_ids)

stats = pretrainer.train_step(input_ids, attention_mask)
print(f"MLM Loss: {stats['mlm_loss']:.4f}")
print(f"MLM Accuracy: {stats['mlm_accuracy']:.2%}")
print(f"Masked Tokens: {stats['num_masked']}")
```

#### BERT遮蔽策略

对于15%被选中的token:
- **80%** 替换为`[MASK]`
- **10%** 替换为随机token
- **10%** 保持不变

#### MLM + NSP (BERT风格)

```python
config = MLMConfig(
    mask_prob=0.15,
    use_nsp=True  # 启用NSP任务
)

pretrainer = create_mlm_pretrainer(model, hidden_size=768, config=config)

# 训练
nsp_labels = torch.randint(0, 2, (batch_size,))  # 0=连续, 1=不连续
stats = pretrainer.train_step(input_ids, attention_mask, nsp_labels)

print(f"MLM Loss: {stats['mlm_loss']:.4f}")
print(f"NSP Loss: {stats['nsp_loss']:.4f}")
print(f"NSP Accuracy: {stats['nsp_accuracy']:.2%}")
```

---

## 使用示例

### 完整的RLHF训练流程

```python
import torch
from apt_model.rl import (
    create_reward_model,
    RewardModelTrainer,
    create_rlhf_trainer
)

# 1. 训练奖励模型
print("训练奖励模型...")
base_model = load_pretrained_model()
reward_model = create_reward_model(base_model, hidden_size=768)
reward_trainer = RewardModelTrainer(reward_model, optimizer)

for epoch in range(reward_epochs):
    for chosen, rejected in preference_dataloader:
        stats = reward_trainer.train_step(chosen, rejected)
        print(f"Reward Loss: {stats['loss']:.4f}")

# 2. RLHF训练
print("RLHF训练...")
policy_model = load_pretrained_model()
rlhf_trainer = create_rlhf_trainer(
    policy_model=policy_model,
    reward_model=reward_model
)

for epoch in range(rlhf_epochs):
    for prompts in prompt_dataloader:
        stats = rlhf_trainer.train_step(prompts, prompt_masks)
        print(f"Mean Reward: {stats['mean_reward']:.4f}")
```

### 完整的DPO训练流程

```python
from apt_model.rl import create_dpo_trainer, DPOConfig
import copy

# 创建策略模型和参考模型
policy_model = load_pretrained_model()
ref_model = copy.deepcopy(policy_model)
ref_model.eval()

# 创建DPO训练器
trainer = create_dpo_trainer(
    policy_model=policy_model,
    ref_policy_model=ref_model,
    config={'beta': 0.1}
)

# 训练
for epoch in range(epochs):
    for chosen, rejected in preference_dataloader:
        stats = trainer.train_step(chosen, rejected)
        print(f"Loss: {stats['loss']:.4f}, Acc: {stats['accuracy']:.2%}")
```

### 预训练+微调流程

```python
from apt_model.pretraining import create_contrastive_pretrainer, create_mlm_pretrainer

# 1. 对比学习预训练
print("对比学习预训练...")
model = YourModel()
contrastive_trainer = create_contrastive_pretrainer(model, hidden_size=768)

for epoch in range(pretrain_epochs):
    for batch in dataloader:
        x1, x2 = augment_batch(batch)
        stats = contrastive_trainer.train_step(x1, x2)

# 2. MLM预训练
print("MLM预训练...")
mlm_trainer = create_mlm_pretrainer(model, hidden_size=768)

for epoch in range(mlm_epochs):
    for batch in dataloader:
        stats = mlm_trainer.train_step(batch['input_ids'], batch['attention_mask'])

# 3. RLHF微调
print("RLHF微调...")
# ... (见上面的RLHF示例)
```

---

## 最佳实践

### 选择合适的RL算法

| 算法 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| **RLHF** | 理论完善，性能好 | 需要奖励模型，复杂 | 大规模生产环境 |
| **DPO** | 简单，训练稳定 | 需要参考模型 | 快速原型开发 |
| **GRPO** | 高效，在线学习 | 相对较新 | 在线学习场景 |

### 超参数调优

#### RLHF
- `ppo_epochs`: 4-8
- `clip_epsilon`: 0.1-0.3
- `kl_coef`: 0.01-0.2 (根据任务调整)
- `learning_rate`: 1e-6 到 1e-5

#### DPO
- `beta`: 0.05-0.5 (越大越激进)
- `label_smoothing`: 0.0-0.1
- `learning_rate`: 1e-6 到 1e-5

#### GRPO
- `group_size`: 4-8 (取决于计算资源)
- `advantage_type`: 从"relative"开始
- `learning_rate`: 1e-6 到 1e-5

### 训练技巧

1. **学习率调度**:
   ```python
   from torch.optim.lr_scheduler import CosineAnnealingLR
   scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
   ```

2. **梯度累积**:
   ```python
   for i, batch in enumerate(dataloader):
       loss = trainer.train_step(batch)
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **混合精度训练**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
       loss = compute_loss()
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

4. **检查点保存**:
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'stats': trainer.get_statistics()
   }, 'checkpoint.pt')
   ```

### 监控指标

#### RLHF
- `mean_reward`: 平均奖励 (应该逐渐增加)
- `mean_kl`: KL散度 (不应太大)
- `ppo_loss`: PPO损失
- `entropy`: 策略熵 (保持一定探索)

#### DPO
- `loss`: DPO损失 (应该下降)
- `accuracy`: 偏好准确率 (应该>50%)
- `reward_margin`: 奖励差距 (chosen vs rejected)

#### GRPO
- `mean_reward`: 平均奖励
- `group_variance`: 组内方差 (反映多样性)
- `policy_loss`: 策略损失

### 常见问题

#### Q1: 奖励模型过拟合
**解决方案**:
- 增加训练数据
- 使用dropout
- 早停 (early stopping)
- 数据增强

#### Q2: KL散度过大
**解决方案**:
- 增加`kl_coef`
- 降低学习率
- 使用更强的裁剪 (降低`clip_epsilon`)

#### Q3: 训练不稳定
**解决方案**:
- 使用梯度裁剪
- 降低学习率
- 增加batch size
- 使用更保守的超参数

#### Q4: 奖励hacking
**解决方案**:
- 使用更鲁棒的奖励模型
- 增加KL惩罚
- 使用多个奖励模型集成
- 人工审查生成结果

---

## 参考文献

1. **RLHF**: Ouyang et al. "Training language models to follow instructions with human feedback" (2022)
2. **DPO**: Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
3. **GRPO**: Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)
4. **SimCLR**: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (2020)
5. **MoCo**: He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" (2020)
6. **BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)

---

## 更新日志

- **2025-12-02**: 初始版本
  - 添加奖励模型
  - 添加RLHF训练器
  - 添加DPO训练器
  - 添加GRPO训练器
  - 添加对比学习预训练
  - 添加MLM预训练
  - 更新GRPO插件以使用实际训练器

---

如有问题或建议，请联系: chen0430tw
