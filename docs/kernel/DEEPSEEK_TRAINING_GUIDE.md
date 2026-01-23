# DeepSeek 模型训练指南

<div align="center">

**基于 DeepSeek-V3 架构的 MoE 模型训练完整教程**

支持 Multi-head Latent Attention | DeepSeekMoE | FP8 混合精度

</div>

---

## 📋 目录

- [DeepSeek 简介](#deepseek-简介)
- [架构特点](#架构特点)
- [快速开始](#快速开始)
- [核心组件实现](#核心组件实现)
- [训练配置](#训练配置)
- [优化技巧](#优化技巧)
- [常见问题](#常见问题)

---

## 🎯 DeepSeek 简介

### 什么是 DeepSeek？

DeepSeek 是由深度求索（DeepSeek-AI）开发的开源大语言模型系列，以其高效的 **Mixture-of-Experts (MoE)** 架构和创新的注意力机制而闻名。

### DeepSeek-V3 核心数据

| 指标 | 数值 |
|------|------|
| **总参数** | 671B |
| **激活参数** | 37B（每个token） |
| **训练数据** | 14.8T tokens |
| **训练成本** | 2.664M H800 GPU小时 |
| **训练硬件** | 2048 × NVIDIA H800 |
| **许可证** | MIT License |

---

## 🏗️ 架构特点

### 1. Multi-head Latent Attention (MLA)

**核心创新：** 使用低秩投影减少 KV Cache 开销

```python
class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek MLA：通过潜在空间压缩降低推理成本

    传统注意力：O(n * d_model * n_heads) KV cache
    MLA：O(n * d_latent) KV cache，其中 d_latent << d_model * n_heads
    """
    def __init__(self, d_model=2048, n_heads=16, d_latent=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_latent = d_latent
        self.d_head = d_model // n_heads

        # 压缩到潜在空间
        self.W_DKV = nn.Linear(d_model, d_latent + d_model)  # 降维投影

        # Query 直接投影
        self.W_Q = nn.Linear(d_model, d_model)

        # 从潜在空间解压
        self.W_UK = nn.Linear(d_latent, d_model)  # Key 上采样
        self.W_UV = nn.Linear(d_latent, d_model)  # Value 上采样

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # 1. 压缩 K, V 到潜在空间
        kv_compressed = self.W_DKV(x)  # [B, T, d_latent + d_model]
        k_latent = kv_compressed[:, :, :self.d_latent]  # [B, T, d_latent]
        v_rope = kv_compressed[:, :, self.d_latent:]     # [B, T, d_model]

        # 2. 解压 K
        k = self.W_UK(k_latent)  # [B, T, d_model]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 3. 解压 V
        v = self.W_UV(k_latent) + v_rope  # 残差连接
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 4. Query 投影
        q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 5. 标准注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 6. 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)
```

**内存节省：** 使用 MLA 可节省约 **70-80%** 的 KV Cache 内存

---

### 2. DeepSeekMoE 架构

**核心策略：**
1. **细粒度专家分割**：将大专家分成多个小专家，提高专业化
2. **共享专家隔离**：部分专家始终激活，保证基础能力

```python
class DeepSeekMoE(nn.Module):
    """
    DeepSeek MoE：细粒度专家 + 共享专家

    设计理念：
    - 路由专家（Routed Experts）：动态选择，负责专业任务
    - 共享专家（Shared Experts）：始终激活，负责通用能力
    """
    def __init__(
        self,
        d_model=2048,
        d_ff=10240,
        num_routed_experts=160,     # 路由专家数量
        num_shared_experts=8,        # 共享专家数量
        num_activated_experts=8,     # 每次激活的路由专家数
        expert_capacity=1.25,        # 专家容量因子
    ):
        super().__init__()
        self.d_model = d_model
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_activated_experts = num_activated_experts

        # 路由器（Top-K 选择）
        self.router = nn.Linear(d_model, num_routed_experts)

        # 路由专家（细粒度，每个较小）
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff // 4),  # 缩小专家尺寸
                nn.GELU(),
                nn.Linear(d_ff // 4, d_model)
            ) for _ in range(num_routed_experts)
        ])

        # 共享专家（始终激活，较大）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_shared_experts)
        ])

        # 专家权重融合
        self.shared_gate = nn.Linear(d_model, num_shared_experts)

    def forward(self, x):
        B, T, C = x.shape

        # ========== 路由专家（动态选择）==========
        router_logits = self.router(x)  # [B, T, num_routed_experts]

        # Top-K 路由
        topk_weights, topk_indices = torch.topk(
            router_logits,
            k=self.num_activated_experts,
            dim=-1
        )  # [B, T, K]

        topk_weights = torch.softmax(topk_weights, dim=-1)

        # 执行路由专家
        routed_output = torch.zeros_like(x)
        for i in range(self.num_activated_experts):
            expert_idx = topk_indices[:, :, i]  # [B, T]
            expert_weight = topk_weights[:, :, i:i+1]  # [B, T, 1]

            # 批量执行专家（简化版，实际需要更复杂的调度）
            for b in range(B):
                for t in range(T):
                    expert_id = expert_idx[b, t].item()
                    expert_out = self.routed_experts[expert_id](x[b:b+1, t:t+1])
                    routed_output[b:b+1, t:t+1] += expert_weight[b, t] * expert_out

        # ========== 共享专家（始终激活）==========
        shared_gate_logits = self.shared_gate(x)  # [B, T, num_shared_experts]
        shared_weights = torch.softmax(shared_gate_logits, dim=-1)  # [B, T, num_shared_experts]

        shared_output = torch.zeros_like(x)
        for i, expert in enumerate(self.shared_experts):
            expert_out = expert(x)  # [B, T, C]
            shared_output += shared_weights[:, :, i:i+1] * expert_out

        # ========== 组合输出 ==========
        return routed_output + shared_output
```

**性能提升：** DeepSeekMoE 16B 仅用 **40.5%** 计算量即可达到 DeepSeek 7B 性能

---

### 3. FP8 混合精度训练

**核心技术：** 细粒度量化 + 选择性高精度计算

```python
class FP8MixedPrecisionTrainer:
    """
    DeepSeek FP8 混合精度训练框架

    策略：
    - GEMM 操作：FP8（矩阵乘法）
    - 关键操作：FP16/BF16（Softmax, LayerNorm）
    - 梯度累积：FP32
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        # FP8 量化配置
        self.activation_scale = {}   # 动态激活值缩放
        self.weight_scale = {}        # 静态权重缩放

    def quantize_to_fp8(self, tensor, tile_size=(1, 128), is_activation=True):
        """
        细粒度量化到 FP8

        激活值：Tile-wise 1×128 量化
        权重：Block-wise 128×128 量化
        """
        if is_activation:
            # 激活值：按 tile 动态量化
            B, T, C = tensor.shape
            num_tiles = C // tile_size[1]

            quantized = torch.zeros_like(tensor, dtype=torch.float8_e4m3fn)
            scales = []

            for i in range(num_tiles):
                start_idx = i * tile_size[1]
                end_idx = start_idx + tile_size[1]
                tile = tensor[:, :, start_idx:end_idx]

                # 计算缩放因子
                max_val = tile.abs().max()
                scale = max_val / 448.0  # FP8 E4M3 最大值
                scales.append(scale)

                # 量化
                quantized[:, :, start_idx:end_idx] = (tile / scale).to(torch.float8_e4m3fn)

            return quantized, torch.tensor(scales)
        else:
            # 权重：按 block 静态量化
            # （简化示例，实际实现更复杂）
            max_val = tensor.abs().max()
            scale = max_val / 448.0
            quantized = (tensor / scale).to(torch.float8_e4m3fn)
            return quantized, scale

    def train_step(self, batch):
        """FP8 混合精度训练步骤"""
        self.model.train()
        input_ids = batch['input_ids']
        labels = batch['labels']

        self.optimizer.zero_grad()

        # ========== 前向传播（FP8 计算）==========
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # 注意：实际 FP8 需要自定义 CUDA kernel
            # 这里用 BF16 模拟，真实实现需调用 FP8 GEMM
            logits = self.model(input_ids)

            # 损失计算用 FP32（提高数值稳定性）
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1)
            )

        # ========== 反向传播（梯度 FP32 累积）==========
        loss.backward()

        # 梯度裁剪（FP32）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 优化器更新（FP32 主权重）
        self.optimizer.step()

        return loss.item()
```

**效率提升：** FP8 训练可减少 **40-50%** 显存占用和训练时间

---

## 🚀 快速开始

### 1分钟训练 DeepSeek 风格模型

```python
from apt_model.modeling.deepseek_model import DeepSeekModel
from apt_model.training.deepseek_trainer import DeepSeekTrainer
from transformers import AutoTokenizer

# 1. 初始化模型（小规模配置）
model = DeepSeekModel(
    vocab_size=50257,
    d_model=1024,           # 小模型用 1024，大模型用 2048-4096
    n_heads=16,
    num_layers=12,
    d_ff=4096,
    d_latent=256,           # MLA 潜在维度
    num_routed_experts=32,  # 路由专家数
    num_shared_experts=4,   # 共享专家数
    num_activated_experts=4 # 每次激活专家数
)

# 2. 准备数据
tokenizer = AutoTokenizer.from_pretrained("gpt2")
train_texts = open("train.txt", "r", encoding="utf-8").readlines()

# 3. 创建训练器
trainer = DeepSeekTrainer(
    model=model,
    tokenizer=tokenizer,
    learning_rate=2e-4,
    use_fp8=False  # 开启需要 Hopper+ GPU（H100/H800）
)

# 4. 开始训练
history = trainer.train(
    train_texts=train_texts,
    epochs=20,
    batch_size=8,
    max_length=512,
    save_path="./deepseek_checkpoint"
)

# 5. 生成文本
import torch
model.eval()
with torch.no_grad():
    input_text = "人工智能的未来是"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    output = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
    print(tokenizer.decode(output[0].tolist()))
```

---

## ⚙️ 训练配置

### 硬件要求

| 模型规模 | 参数量 | 最低显存 | 推荐显存 | 激活专家数 |
|---------|--------|---------|---------|-----------|
| **Mini** | 2.7B (激活 340M) | 8GB | 16GB | 2/16 experts |
| **Small** | 16B (激活 2.8B) | 24GB | 40GB | 6/64 experts |
| **Medium** | 67B (激活 8B) | 40GB | 80GB | 8/128 experts |
| **Large** | 671B (激活 37B) | 8x80GB | 16x80GB | 8/256 experts |

### 超参数推荐

#### Mini 模型（2.7B，学习实验）

```python
config = {
    'd_model': 1024,
    'n_heads': 16,
    'd_ff': 4096,
    'num_layers': 16,
    'd_latent': 256,
    'num_routed_experts': 16,
    'num_shared_experts': 2,
    'num_activated_experts': 2,

    'learning_rate': 3e-4,
    'batch_size': 16,
    'max_length': 1024,
    'warmup_steps': 2000,
    'weight_decay': 0.01,
}
```

#### Small 模型（16B，生产可用）

```python
config = {
    'd_model': 2048,
    'n_heads': 16,
    'd_ff': 10240,
    'num_layers': 28,
    'd_latent': 512,
    'num_routed_experts': 64,
    'num_shared_experts': 4,
    'num_activated_experts': 6,

    'learning_rate': 2e-4,
    'batch_size': 4,  # 梯度累积 x16 = 有效 batch 64
    'max_length': 4096,
    'warmup_steps': 4000,
    'gradient_accumulation_steps': 16,
}
```

### 数据准备

DeepSeek-V3 训练数据比例：

| 数据类型 | 比例 | 说明 |
|---------|------|------|
| **通用文本** | ~60% | 网页、书籍、论文 |
| **代码** | ~20% | GitHub、编程教程 |
| **数学** | ~10% | 数学推理、证明 |
| **多语言** | ~10% | 中英外多语言语料 |

```python
# 数据预处理示例
def prepare_deepseek_data(raw_texts):
    """DeepSeek 数据准备流程"""
    processed = []

    for text in raw_texts:
        # 1. 去重（MinHash LSH）
        if is_duplicate(text):
            continue

        # 2. 质量过滤
        if len(text) < 50 or quality_score(text) < 0.6:
            continue

        # 3. 多样性增强（不同领域混合）
        text_type = classify_text_type(text)  # general/code/math/multilingual

        processed.append({
            'text': text,
            'type': text_type,
            'length': len(text)
        })

    # 4. 按类型平衡采样
    balanced = balance_by_type(processed, ratios={
        'general': 0.6,
        'code': 0.2,
        'math': 0.1,
        'multilingual': 0.1
    })

    return [item['text'] for item in balanced]
```

---

## 🔥 优化技巧

### 1. Auxiliary-Loss-Free 负载均衡

**问题：** 传统 MoE 用辅助损失强制负载均衡，损害模型性能

**DeepSeek 方案：** 无辅助损失的自然负载均衡

```python
class AuxiliaryLossFreeRouter(nn.Module):
    """
    DeepSeek-V3 无辅助损失路由器

    核心思想：
    1. 不添加负载均衡损失
    2. 通过专家容量限制自然平衡
    3. 使用 token dropping 处理溢出
    """
    def __init__(self, d_model, num_experts, num_activated, capacity_factor=1.25):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.num_activated = num_activated
        self.capacity_factor = capacity_factor

    def forward(self, x):
        B, T, C = x.shape

        # 1. 路由打分（无辅助损失）
        router_logits = self.gate(x)  # [B, T, E]
        router_probs = torch.softmax(router_logits, dim=-1)

        # 2. Top-K 选择
        topk_probs, topk_indices = torch.topk(router_probs, k=self.num_activated, dim=-1)

        # 3. 计算专家容量
        tokens_per_expert = (B * T * self.num_activated) / self.num_experts
        expert_capacity = int(tokens_per_expert * self.capacity_factor)

        # 4. 分配 tokens 到专家（先到先得，超出丢弃）
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        expert_mask = torch.zeros(B, T, self.num_activated, dtype=torch.bool, device=x.device)

        for b in range(B):
            for t in range(T):
                for k in range(self.num_activated):
                    expert_id = topk_indices[b, t, k].item()
                    if expert_counts[expert_id] < expert_capacity:
                        expert_counts[expert_id] += 1
                        expert_mask[b, t, k] = True  # 保留该 token
                    # else: token dropped（丢弃，不添加惩罚）

        # 5. 应用 mask（被丢弃的 token 权重归零）
        topk_probs_masked = topk_probs * expert_mask.float()
        topk_probs_normalized = topk_probs_masked / (topk_probs_masked.sum(dim=-1, keepdim=True) + 1e-8)

        return topk_probs_normalized, topk_indices, expert_mask
```

**效果：** 不牺牲性能的同时，自动实现负载均衡

---

### 2. Multi-Token Prediction (MTP)

**核心思想：** 同时预测当前和未来多个 token，提高数据效率

```python
class MultiTokenPrediction(nn.Module):
    """
    DeepSeek-V3 多 token 预测

    策略：
    - 主预测头：预测下一个 token（正常损失权重 1.0）
    - 辅助预测头：预测未来 2-4 个 token（损失权重 0.3）
    """
    def __init__(self, d_model, vocab_size, num_future_tokens=3):
        super().__init__()
        self.num_future_tokens = num_future_tokens

        # 主预测头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 辅助预测头（共享底层表示）
        self.future_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, vocab_size, bias=False)
            ) for _ in range(num_future_tokens)
        ])

    def forward(self, hidden_states, labels=None):
        """
        Args:
            hidden_states: [B, T, C] - 模型隐藏状态
            labels: [B, T] - 标签序列

        Returns:
            loss: 多 token 预测总损失
        """
        B, T, C = hidden_states.shape

        # ========== 主预测（t+1）==========
        logits_main = self.lm_head(hidden_states)  # [B, T, V]

        if labels is None:
            return logits_main

        # 计算主损失
        loss_main = F.cross_entropy(
            logits_main[:, :-1].reshape(-1, logits_main.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100
        )

        # ========== 辅助预测（t+2, t+3, t+4）==========
        loss_aux = 0.0
        for i, future_head in enumerate(self.future_heads):
            # 预测未来第 i+2 个 token
            future_offset = i + 2
            if T <= future_offset:
                continue

            logits_future = future_head(hidden_states[:, :-future_offset])

            loss_future = F.cross_entropy(
                logits_future.reshape(-1, logits_future.size(-1)),
                labels[:, future_offset:].reshape(-1),
                ignore_index=-100
            )

            loss_aux += loss_future * 0.3  # 辅助损失权重

        # ========== 总损失 ==========
        total_loss = loss_main + loss_aux / max(len(self.future_heads), 1)

        return total_loss, logits_main
```

**数据效率提升：** MTP 可减少 **20-30%** 训练时间

---

### 3. 分布式训练（大规模）

DeepSeek-V3 使用的并行策略：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_deepseek_distributed():
    """
    DeepSeek-V3 分布式配置

    并行策略：
    - Pipeline Parallelism (PP): 16-way（跨层切分）
    - Expert Parallelism (EP): 64-way（专家跨节点）
    - Data Parallelism (DP): ZeRO-1（梯度分片）
    """
    # 初始化分布式
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()

    # ========== Pipeline Parallelism ==========
    # 将模型按层切分到 16 个设备
    from torch.distributed.pipeline.sync import Pipe

    model = DeepSeekModel(...)

    # 切分层到不同设备
    balance = [2, 2, 2, 2, 2, 2, 2, 2]  # 每个 PP rank 处理 2 层（共 16 层）
    model = Pipe(model, balance=balance, chunks=8)

    # ========== Expert Parallelism ==========
    # DeepEP：专家并行通信库
    # 将 256 个专家分配到 64 个 GPU（每个 4 个专家）
    from deepep import ExpertParallel

    ep_group = dist.new_group(ranks=list(range(0, 64)))  # EP 组
    model.moe_layers = ExpertParallel(
        model.moe_layers,
        expert_parallel_group=ep_group
    )

    # ========== ZeRO-1 Data Parallelism ==========
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimizer = DeepSpeedZeroOptimizer(
        optimizer,
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        partition_gradients=True  # ZeRO-1：分片梯度
    )

    return model, optimizer

# ========== 训练循环 ==========
def train_distributed(model, train_loader, optimizer):
    model.train()

    for batch in train_loader:
        input_ids = batch['input_ids'].to(local_rank)
        labels = batch['labels'].to(local_rank)

        # 前向传播（Pipeline 自动处理）
        loss = model(input_ids, labels=labels).local_value()

        # 反向传播（ZeRO 自动分片梯度）
        loss.backward()

        # 梯度裁剪（跨 rank 同步）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 优化器更新
        optimizer.step()
        optimizer.zero_grad()
```

**扩展性：** 可扩展到 **2048 GPUs**（DeepSeek-V3 实际配置）

---

## 🐛 常见问题

### Q1: MLA 的 KV Cache 如何节省内存？

**A:** MLA 通过低秩投影压缩 K, V：

```
传统注意力 KV Cache：
- 每层存储 K, V: [batch, seq_len, n_heads * d_head]
- 总内存: seq_len × d_model × num_layers × 2

MLA KV Cache：
- 每层存储压缩的 K, V: [batch, seq_len, d_latent]
- 总内存: seq_len × d_latent × num_layers × 2

节省比例: 1 - (d_latent / d_model)
示例: d_model=4096, d_latent=512 → 节省 87.5%
```

---

### Q2: 为什么 DeepSeekMoE 用共享专家？

**A:** 共享专家确保基础能力不会因路由专家专业化而丢失：

```
路由专家（Routed Experts）：
- 动态选择，高度专业化
- 可能学到狭窄的特定模式
- 某些通用知识可能缺失

共享专家（Shared Experts）：
- 始终激活，学习通用表示
- 补充路由专家的盲区
- 提高模型稳定性

实验结果：
- 无共享专家: 性能下降 3-5%
- 有共享专家: 性能提升，更稳定
```

---

### Q3: FP8 训练是否会损失精度？

**A:** DeepSeek 的细粒度 FP8 量化几乎无损：

```python
# 关键设计：
1. 细粒度量化（Tile-wise/Block-wise）
   - 不是整个张量一个缩放因子
   - 每 128 个元素一个缩放因子
   - 适应局部数值分布

2. 选择性高精度
   - GEMM: FP8 ✓（计算密集，量化收益大）
   - Softmax: BF16（数值敏感）
   - LayerNorm: BF16（数值敏感）
   - 梯度累积: FP32（防止累积误差）

3. 动态缩放
   - 激活值：每步动态计算缩放因子
   - 权重：预计算静态缩放因子

实验结果：
- FP8 vs BF16: < 0.1% 性能差距
- 显存节省: ~40%
- 训练速度: 提升 30-50%
```

---

### Q4: 如何处理 MoE 的负载不均衡？

**A:** DeepSeek 采用无辅助损失策略：

```python
传统方法：
- 添加辅助损失: L_aux = λ × load_balance_loss
- 问题: 强制平衡损害性能，λ 难调

DeepSeek 方法：
1. 专家容量限制（Expert Capacity）
   - 每个专家最多处理 capacity 个 token
   - capacity = (total_tokens / num_experts) × factor
   - factor 通常设为 1.25

2. Token Dropping
   - 超出容量的 token 直接丢弃
   - 不添加任何惩罚损失
   - 自然形成负载均衡

3. 结果
   - 无需调参（不需要 λ）
   - 性能不受影响
   - 负载自动均衡（专家饱和自然减少分配）
```

---

## 📚 参考资源

### 官方资源

- [DeepSeek GitHub 组织](https://github.com/deepseek-ai) - 所有官方代码仓库
- [DeepSeek-V3 仓库](https://github.com/deepseek-ai/DeepSeek-V3) - 最新模型代码
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) - 完整架构论文
- [DeepSeek-MoE 论文](https://github.com/deepseek-ai/DeepSeek-MoE) - MoE 架构详解

### 技术深度解读

- [DeepSeek Models Technical Tour](https://magazine.sebastianraschka.com/p/technical-deepseek) - Sebastian Raschka 技术解析
- [Complete Guide to DeepSeek Models](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond) - BentoML 完整指南
- [DeepSeek-V3 Architecture](https://deepwiki.com/deepseek-ai/DeepSeek-V3/3-model-architecture) - DeepWiki 架构解析

### APT 相关文档

- [GPT 训练指南](GPT_TRAINING_GUIDE.md) - 对比 GPT 架构
- [API 集成指南](../product/API_PROVIDERS_GUIDE.md) - 使用 DeepSeek API
- [APT Model Handbook](APT_MODEL_HANDBOOK.md) - APT 平台完整手册

---

## 📝 更新日志

- **v1.0.0** (2025-12) - 初始版本
  - ✅ Multi-head Latent Attention (MLA) 实现
  - ✅ DeepSeekMoE 架构（路由专家 + 共享专家）
  - ✅ FP8 混合精度训练框架
  - ✅ Multi-Token Prediction (MTP)
  - ✅ Auxiliary-Loss-Free 负载均衡
  - ✅ 分布式训练配置（PP + EP + ZeRO）

---

<div align="center">

**Happy Training with DeepSeek! 🚀**

基于世界级开源架构，打造你的专属大模型

如有问题，请提交 [Issue](https://github.com/chen0430tw/APT-Transformer/issues)

</div>
