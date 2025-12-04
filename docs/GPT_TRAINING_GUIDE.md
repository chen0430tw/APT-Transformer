# GPT模型训练指南

<div align="center">

**完整的GPT模型训练教程 - 从零到部署**

支持 GPT-4o | GPT-5 | GPTo3

</div>

---

## 📋 目录

- [快速开始](#快速开始)
- [模型选择](#模型选择)
- [训练配置](#训练配置)
- [高级功能](#高级功能)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

---

## 🚀 快速开始

### 1分钟训练你的第一个GPT模型

```python
from apt_model.training.gpt_trainer import train_gpt4o

# 准备训练数据
train_texts = [
    "人工智能正在改变世界",
    "深度学习是机器学习的一个分支",
    "Transformer架构revolutionized NLP",
    # ... 更多文本
]

# 开始训练
model, tokenizer, history = train_gpt4o(
    train_texts=train_texts,
    epochs=10,
    batch_size=8,
    save_path="./my_gpt4o"
)

# 生成文本
import torch
input_text = "人工智能"
input_ids = torch.tensor([tokenizer.encode(input_text)])
output = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
```

---

## 🎯 模型选择

### GPT-4o 🌟 推荐

**适用场景：**
- ✅ 多模态应用（文本+图像+音频）
- ✅ 需要高质量生成
- ✅ 生产环境部署
- ✅ 初学者友好

**特点：**
- Tri-Vein Attention（三维子空间注意力）
- Hybrid FFN（混合前馈网络）
- 动态τ门控
- 支持多模态输入

**训练示例：**

```python
from apt_model.modeling.gpt4o_model import GPT4oModel
from apt_model.training.gpt_trainer import GPT4oTrainer
from transformers import GPT2Tokenizer

# 1. 初始化模型
model = GPT4oModel(
    vocab_size=50257,    # GPT-2词汇表大小
    d_model=768,         # 模型维度
    n_heads=12,          # 注意力头数
    d_ff=3072,           # FFN维度
    num_layers=12,       # 层数
    rank=4               # Vein子空间秩
)

# 2. 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. 创建训练器
trainer = GPT4oTrainer(
    model=model,
    tokenizer=tokenizer,
    learning_rate=3e-4,
    weight_decay=0.01
)

# 4. 准备数据
train_texts = open('train.txt', 'r', encoding='utf-8').readlines()
eval_texts = open('eval.txt', 'r', encoding='utf-8').readlines()

# 5. 开始训练
history = trainer.train(
    train_texts=train_texts,
    epochs=20,
    batch_size=16,
    max_length=512,
    save_path="./gpt4o_checkpoint",
    eval_texts=eval_texts,
    eval_interval=1000
)
```

### GPTo3 🧠 高级

**适用场景：**
- ✅ 复杂推理任务
- ✅ 需要结构化思考
- ✅ 研究实验
- ⚠️ 计算资源充足

**特点：**
- 结构化推理（Structured Reasoning）
- 熵触发机制
- 多专家系统
- 预算控制

**训练示例：**

```python
from apt_model.modeling.gpto3_model import GPTo3Model
from apt_model.training.gpt_trainer import GPTo3Trainer

# 1. 初始化模型（更多参数）
model = GPTo3Model(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    d_ff=3072,
    num_layers=12,
    rank=4,
    # GPTo3特有参数
    entropy_trig=2.0,      # 熵触发阈值
    global_budget=0.15,    # 推理预算
    max_reason_steps=6,    # 最大推理步数
    patience=2,            # 早停耐心值
    eps_kl=0.02,          # KL散度停止阈值
    topk_experts=2        # Top-K专家数
)

# 2. 训练配置
trainer = GPTo3Trainer(
    model=model,
    tokenizer=tokenizer,
    learning_rate=2e-4,    # 较低学习率
    weight_decay=0.01,
    max_grad_norm=1.0
)

# 3. 训练
history = trainer.train(
    train_texts=train_texts,
    epochs=30,            # 更多epoch
    batch_size=8,         # 较小batch size
    max_length=1024,      # 更长序列
    save_path="./gpto3_checkpoint"
)
```

### GPT-5 🔬 高级

**适用场景：**
- ✅ MoE（专家混合）研究
- ✅ CPU友好训练
- ✅ 流式检索应用
- ⚠️ 需要配置VeinProjector依赖

**特点：**
- Codebook MoE
- Leaf-Vote投票机制
- 流式检索器
- 记忆桶（Memory Bucket）

**注意事项：**
- 需要安装 `apt_model.modeling.blocks.VeinProjector`
- 适合MoE架构研究和大规模训练
- 详见 [GPT Models Analysis](GPT_MODELS_ANALYSIS.md) 了解完整特性

---

## ⚙️ 训练配置

### 硬件要求

| 模型 | 最低显存 | 推荐显存 | CPU可行 |
|------|---------|---------|---------|
| **GPT-4o (Small)** | 4GB | 8GB+ | ✅ 可以 |
| **GPT-4o (Base)** | 8GB | 16GB+ | ⚠️ 较慢 |
| **GPT-4o (Large)** | 16GB | 24GB+ | ❌ 不推荐 |
| **GPTo3 (Base)** | 12GB | 24GB+ | ⚠️ 较慢 |
| **GPT-5 (Base)** | 6GB | 12GB+ | ✅ 优化过 |

### 超参数推荐

#### 小型模型（< 100M参数）

```python
config = {
    'd_model': 512,
    'n_heads': 8,
    'd_ff': 2048,
    'num_layers': 6,
    'learning_rate': 3e-4,
    'batch_size': 32,
    'warmup_steps': 1000,
    'max_length': 512
}
```

#### 中型模型（100M - 500M参数）

```python
config = {
    'd_model': 768,
    'n_heads': 12,
    'd_ff': 3072,
    'num_layers': 12,
    'learning_rate': 2e-4,
    'batch_size': 16,
    'warmup_steps': 2000,
    'max_length': 1024
}
```

#### 大型模型（> 500M参数）

```python
config = {
    'd_model': 1024,
    'n_heads': 16,
    'd_ff': 4096,
    'num_layers': 24,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'warmup_steps': 5000,
    'max_length': 2048,
    'gradient_accumulation_steps': 4  # 梯度累积
}
```

### 数据准备

#### 格式要求

**纯文本格式（推荐）：**
```
每行一个训练样本
可以是句子、段落或文档
保持UTF-8编码
```

**JSON格式：**
```json
[
  {"text": "第一个训练样本"},
  {"text": "第二个训练样本"},
  ...
]
```

**CSV格式：**
```csv
text
第一个训练样本
第二个训练样本
```

#### 数据加载

```python
# 方法1：从文件加载
with open('train.txt', 'r', encoding='utf-8') as f:
    train_texts = [line.strip() for line in f if line.strip()]

# 方法2：从JSON加载
import json
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    train_texts = [item['text'] for item in data]

# 方法3：从数据库加载
import pandas as pd
df = pd.read_csv('train.csv')
train_texts = df['text'].tolist()

# 方法4：使用HuggingFace datasets
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_texts = dataset['train']['text']
```

---

## 🔥 高级功能

### 混合精度训练

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 启用混合精度
scaler = GradScaler()

class MixedPrecisionTrainer(BaseGPTTrainer):
    def train_step(self, batch):
        self.model.train()
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        self.optimizer.zero_grad()

        # 使用autocast
        with autocast():
            logits = self.model(text_ids=input_ids)
            loss = self.compute_loss(logits, labels)

        # 缩放梯度
        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        scaler.step(self.optimizer)
        scaler.update()

        return loss.item()
```

### 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# 包装模型
model = GPT4oModel(...)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 使用DistributedSampler
from torch.utils.data.distributed import DistributedSampler
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, ...)

# 训练
trainer.train(...)
```

### 梯度累积

```python
class GradientAccumulationTrainer(BaseGPTTrainer):
    def __init__(self, *args, accumulation_steps=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

    def train_step(self, batch):
        self.model.train()
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        logits = self.model(text_ids=input_ids)
        loss = self.compute_loss(logits, labels)

        # 缩放损失
        loss = loss / self.accumulation_steps
        loss.backward()

        # 只在累积步数后更新
        if (self.step_count + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.step_count += 1
        return loss.item() * self.accumulation_steps
```

### LoRA微调

```python
from peft import get_peft_model, LoraConfig, TaskType

# 配置LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # LoRA秩
    lora_alpha=32,          # LoRA缩放因子
    lora_dropout=0.1,
    target_modules=["W_q", "W_k", "W_v", "W_o"]  # 目标层
)

# 应用LoRA
model = GPT4oModel(...)
model = get_peft_model(model, peft_config)

# 查看可训练参数
model.print_trainable_parameters()
# 输出: trainable params: 2,359,296 || all params: 124,439,808 || trainable%: 1.89

# 正常训练
trainer = GPT4oTrainer(model=model, ...)
```

### 模型量化

```python
import torch.quantization as quant

# 动态量化（推理）
model_quantized = quant.quantize_dynamic(
    model,
    {torch.nn.Linear},  # 量化的层类型
    dtype=torch.qint8
)

# 量化感知训练（QAT）
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model, inplace=False)

# 训练几个epoch
trainer.train(model_prepared, ...)

# 转换为量化模型
model_quantized = quant.convert(model_prepared, inplace=False)
```

---

## 🐛 故障排除

### 常见错误

#### 1. CUDA Out of Memory

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```python
# 方案1：减小batch size
batch_size = 4  # 从8减到4

# 方案2：减小序列长度
max_length = 256  # 从512减到256

# 方案3：使用梯度累积
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

# 方案4：使用梯度检查点
model.gradient_checkpointing_enable()

# 方案5：清理缓存
import torch
torch.cuda.empty_cache()
```

#### 2. Loss不下降或NaN

**症状：**
```
Loss: 8.5432, 8.5421, 8.5419, ... (停滞)
或
Loss: nan
```

**解决方案：**
```python
# 方案1：降低学习率
learning_rate = 1e-4  # 从3e-4降低

# 方案2：增加warmup
warmup_steps = 2000  # 从1000增加

# 方案3：梯度裁剪
max_grad_norm = 0.5  # 从1.0降低

# 方案4：检查数据
# 确保没有异常值或空文本
train_texts = [t for t in train_texts if t and len(t) > 0]

# 方案5：使用更稳定的优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.98),  # 更保守的beta2
    eps=1e-6            # 更大的epsilon
)
```

#### 3. Tokenizer错误

**症状：**
```
AttributeError: 'NoneType' object has no attribute 'pad_token_id'
```

**解决方案：**
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 验证
assert tokenizer.pad_token_id is not None, "pad_token_id不能为None"
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
```

---

## 💡 最佳实践

### 1. 数据预处理

```python
def preprocess_texts(texts):
    """清理和规范化文本"""
    processed = []
    for text in texts:
        # 去除空白
        text = text.strip()

        # 跳过太短的文本
        if len(text) < 10:
            continue

        # 规范化空格
        text = ' '.join(text.split())

        # 移除特殊字符（可选）
        # text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

        processed.append(text)

    return processed

train_texts = preprocess_texts(raw_texts)
```

### 2. 学习率调度

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Cosine退火
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_training_steps,
    eta_min=1e-6
)

# One Cycle
scheduler = OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=num_training_steps,
    pct_start=0.1  # 10% warmup
)

# 在训练循环中使用
for batch in train_loader:
    loss = trainer.train_step(batch)
    scheduler.step()
```

### 3. 早停（Early Stopping）

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# 使用
early_stopping = EarlyStopping(patience=5)

for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = evaluate()

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 4. 模型检查点管理

```python
import shutil

def save_best_model(model, tokenizer, current_loss, best_loss, save_path):
    """只保存最佳模型"""
    if current_loss < best_loss:
        # 删除旧的最佳模型
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        # 保存新的最佳模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': current_loss,
        }, os.path.join(save_path, 'best_model.pt'))

        tokenizer.save_pretrained(save_path)
        return current_loss

    return best_loss
```

### 5. 日志和监控

```python
from tensorboard import SummaryWriter

# 初始化TensorBoard
writer = SummaryWriter('runs/gpt4o_experiment')

# 训练循环中记录
for step, batch in enumerate(train_loader):
    loss = trainer.train_step(batch)

    # 记录loss
    writer.add_scalar('Loss/train', loss, step)

    # 记录学习率
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], step)

    # 定期记录梯度
    if step % 100 == 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, step)

writer.close()

# 查看：tensorboard --logdir=runs
```

---

## 📚 完整示例

### 端到端训练流程

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的GPT-4o训练流程示例
"""

import torch
from apt_model.modeling.gpt4o_model import GPT4oModel
from apt_model.training.gpt_trainer import GPT4oTrainer
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter

def main():
    # ==================== 配置 ====================
    config = {
        'vocab_size': 50257,
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 2048,
        'num_layers': 6,
        'rank': 4,
        'learning_rate': 3e-4,
        'batch_size': 16,
        'epochs': 20,
        'max_length': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': './checkpoints/gpt4o'
    }

    print("=" * 80)
    print("GPT-4o 训练流程")
    print("=" * 80)
    print(f"设备: {config['device']}")
    print(f"模型参数: {config['d_model']}d, {config['num_layers']}层")
    print("=" * 80)

    # ==================== 数据准备 ====================
    print("\n1. 准备数据...")
    with open('train.txt', 'r', encoding='utf-8') as f:
        train_texts = [line.strip() for line in f if line.strip()]

    with open('eval.txt', 'r', encoding='utf-8') as f:
        eval_texts = [line.strip() for line in f if line.strip()]

    print(f"训练样本: {len(train_texts)}")
    print(f"验证样本: {len(eval_texts)}")

    # ==================== 模型初始化 ====================
    print("\n2. 初始化模型...")
    model = GPT4oModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        rank=config['rank']
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # ==================== Tokenizer ====================
    print("\n3. 加载Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # ==================== 训练器 ====================
    print("\n4. 创建训练器...")
    trainer = GPT4oTrainer(
        model=model,
        tokenizer=tokenizer,
        device=config['device'],
        learning_rate=config['learning_rate']
    )

    # ==================== 训练 ====================
    print("\n5. 开始训练...")
    history = trainer.train(
        train_texts=train_texts,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        save_path=config['save_path'],
        eval_texts=eval_texts,
        eval_interval=1000
    )

    # ==================== 生成测试 ====================
    print("\n6. 生成测试...")
    test_prompts = [
        "人工智能",
        "深度学习是",
        "Transformer模型"
    ]

    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            input_ids = torch.tensor([tokenizer.encode(prompt)]).to(config['device'])
            output = model.generate(input_ids, max_new_tokens=50, temperature=0.8)
            generated_text = tokenizer.decode(output[0].tolist())
            print(f"\n提示: {prompt}")
            print(f"生成: {generated_text}")

    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"模型已保存到: {config['save_path']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

---

## 🔗 相关资源

- [APT Model Handbook](APT_MODEL_HANDBOOK.md) - 完整的APT模型文档
- [GPT Models Analysis](../GPT_MODELS_ANALYSIS.md) - 模型架构分析
- [API Documentation](API_PROVIDERS_GUIDE.md) - API集成指南
- [Troubleshooting Guide](../INSTALLATION.md) - 安装和故障排除

---

## 📝 更新日志

- **v1.1.0** (2025-12) - 功能完善版
  - ✅ GPT-4o, GPT-5, GPTo3 全面支持
  - ✅ 混合精度训练优化
  - ✅ 分布式训练支持
  - ✅ 完整的故障排除指南
  - ✅ 生产级训练流程

- **v1.0.0** (2024-12) - 初始版本
  - 基础训练功能
  - 模型架构实现

---

<div align="center">

**Happy Training! 🎉**

如有问题，请提交 [Issue](https://github.com/chen0430tw/APT-Transformer/issues)

</div>
