# APT训练后端使用指南

本项目支持**5种训练后端**，满足从单卡本地实验到大规模云端分布式训练的所有需求。

## 📋 快速选择

| 使用场景 | 推荐后端 | 特点 |
|---------|---------|------|
| 🧪 **HLBD数据集实验** | Playground | Cosine重启学习率，防止过拟合 |
| 💻 **单卡本地训练** | Playground / HuggingFace | RTX 3070优化，混合精度 |
| 🚀 **多卡分布式训练** | DeepSpeed | ZeRO-2/3优化，支持超大模型 |
| ☁️ **云端训练** | Azure ML | MLflow跟踪，自动超参数调优 |
| 🤗 **生态系统集成** | HuggingFace Trainer | W&B、TensorBoard、Hub集成 |

---

## 🎮 Backend 1: Playground训练

**推荐用于HLBD Hardcore数据集训练**

### 特性
- ✅ Playground Theory（CosineAnnealingWarmRestarts）
- ✅ 动态标签支持（[EMOJI], [EN], [PY], [JP], [KR]）
- ✅ RTX 3070优化（混合精度 + 梯度累积）
- ✅ DBC-DAC梯度稳定
- ✅ 实时可视化支持

### 使用方法

```bash
# 方式1: 直接运行
python training/train_hlbd_playground.py --dataset HLBD_Hardcore_Full.json --epochs 100

# 方式2: 统一启动器
python training/train.py --backend playground --epochs 100

# 自定义参数
python training/train_hlbd_playground.py \
    --dataset HLBD_Hardcore_Full.json \
    --epochs 100 \
    --save-dir hlbd_playground \
    --save-interval 25
```

### 配置参数

```python
# 模型配置（PlaygroundConfig）
d_model = 256          # 模型维度
n_layers = 6           # 层数
n_heads = 8            # 注意力头数
batch_size = 16        # Batch大小
gradient_accumulation_steps = 2  # 梯度累积

# Playground Theory
base_lr = 3e-4         # 基础学习率
min_lr = 1e-5          # 最小学习率
T_0 = 10               # Cosine重启周期
T_mult = 2             # 周期倍增系数
```

### 输出文件

```
hlbd_playground/
├── checkpoint_epoch_25.pt      # Checkpoint (每25轮)
├── checkpoint_epoch_50.pt
├── final_model.pt              # 最终模型
└── experiment_report.json      # 训练报告（供可视化）
```

---

## 🚀 Backend 2: DeepSpeed分布式训练

**推荐用于多GPU训练和超大模型**

### 特性
- ✅ ZeRO-1/2/3优化（内存优化10-15倍）
- ✅ CPU卸载（支持100B+模型）
- ✅ 混合精度（FP16/BF16）
- ✅ 梯度累积
- ✅ 分布式数据并行

### 安装依赖

```bash
pip install deepspeed
```

### 使用方法

```bash
# 方式1: DeepSpeed启动（推荐）
deepspeed --num_gpus 2 train_deepspeed.py \
    --dataset HLBD_Hardcore_Full.json \
    --epochs 100 \
    --zero-stage 2 \
    --fp16

# 方式2: 统一启动器
python training/train.py --backend deepspeed \
    --num-gpus 2 \
    --zero-stage 2 \
    --fp16 \
    --epochs 100

# ZeRO-3 + CPU卸载（超大模型）
deepspeed --num_gpus 4 train_deepspeed.py \
    --zero-stage 3 \
    --cpu-offload \
    --fp16 \
    --train-batch-size 256 \
    --gradient-accumulation 4
```

### ZeRO阶段选择

| ZeRO阶段 | 内存节省 | 适用场景 |
|---------|---------|---------|
| **ZeRO-1** | 4x | 单卡放不下优化器状态 |
| **ZeRO-2** | 8x | 多卡训练，显存不足 |
| **ZeRO-3** | 10-15x | 超大模型（100B+参数） |

### DeepSpeed配置文件

```json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 4,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000
  }
}
```

### 输出文件

```
deepspeed_output/
├── deepspeed_config.json       # DeepSpeed配置
├── checkpoint_epoch_25/        # DeepSpeed checkpoint
│   ├── mp_rank_00_model_states.pt
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   └── tokenizer_state.json
└── checkpoint_epoch_100/
```

---

## ☁️ Backend 3: Azure ML云端训练

**推荐用于云端大规模训练和实验管理**

### 特性
- ✅ Azure ML计算集群自动管理
- ✅ MLflow实验跟踪和模型注册
- ✅ 超参数扫描（Sweep jobs）
- ✅ TensorBoard集成
- ✅ 云端checkpoint管理

### 安装依赖

```bash
pip install azure-ai-ml mlflow azureml-mlflow
az login  # Azure登录
```

### 使用方法

```bash
# 方式1: 直接提交
python training/train_azure_ml.py \
    --subscription-id <YOUR_SUBSCRIPTION_ID> \
    --resource-group <YOUR_RESOURCE_GROUP> \
    --workspace-name <YOUR_WORKSPACE> \
    --dataset HLBD_Hardcore_Full.json \
    --epochs 100 \
    --compute-name gpu-cluster \
    --vm-size Standard_NC6s_v3

# 方式2: 统一启动器
python training/train.py --backend azure \
    --azure-subscription-id <ID> \
    --azure-resource-group <RG> \
    --azure-workspace-name <WS> \
    --epochs 100

# 超参数扫描
python training/train_azure_ml.py \
    --subscription-id <ID> \
    --resource-group <RG> \
    --workspace-name <WS> \
    --sweep  # 启用超参数扫描
```

### Azure ML VM规格推荐

| VM规格 | GPU | 内存 | 适用场景 |
|--------|-----|------|---------|
| **Standard_NC6s_v3** | 1x V100 16GB | 112GB | 单卡训练 |
| **Standard_NC12s_v3** | 2x V100 16GB | 224GB | 多卡训练 |
| **Standard_NC24s_v3** | 4x V100 16GB | 448GB | 大规模训练 |
| **Standard_ND40rs_v2** | 8x V100 32GB | 672GB | 超大模型 |

### 超参数扫描配置

```python
search_space = {
    "batch_size": Choice([8, 16, 32]),
    "d_model": Choice([128, 256, 512]),
    "n_layers": Choice([4, 6, 8]),
    "learning_rate": Uniform(1e-5, 1e-3),
    "weight_decay": Uniform(0.001, 0.1)
}
```

### 查看训练进度

```bash
# 查看任务状态
az ml job show --name <JOB_NAME>

# 实时日志流
az ml job stream --name <JOB_NAME>
```

---

## 🤗 Backend 4: HuggingFace Trainer

**推荐用于生态系统集成和快速原型**

### 特性
- ✅ HuggingFace Trainer API（开箱即用最佳实践）
- ✅ Weights & Biases集成
- ✅ TensorBoard集成
- ✅ 早停（Early Stopping）
- ✅ HuggingFace Hub模型上传
- ✅ 支持DeepSpeed（通过Trainer）

### 安装依赖

```bash
pip install transformers datasets accelerate wandb
```

### 使用方法

```bash
# 方式1: 基础训练
python training/train_hf_trainer.py \
    --dataset HLBD_Hardcore_Full.json \
    --epochs 100 \
    --fp16

# 方式2: 统一启动器
python training/train.py --backend huggingface --epochs 100

# 启用Weights & Biases
python training/train_hf_trainer.py \
    --wandb \
    --wandb-project apt-hlbd-training \
    --epochs 100

# 启用早停
python training/train_hf_trainer.py \
    --early-stopping \
    --early-stopping-patience 5 \
    --epochs 100

# HuggingFace Trainer + DeepSpeed
python training/train_hf_trainer.py \
    --deepspeed ds_config.json \
    --fp16 \
    --epochs 100
```

### TrainingArguments配置

```python
TrainingArguments(
    output_dir="hf_output",
    num_train_epochs=100,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=True,                    # 混合精度
    logging_steps=20,
    save_steps=500,
    save_total_limit=3,
    report_to="wandb",            # W&B跟踪
    load_best_model_at_end=True,  # 早停
)
```

### 上传到HuggingFace Hub

```bash
# 设置HF token
export HF_HUB_TOKEN=<YOUR_TOKEN>

# 训练并上传
python training/train_hf_trainer.py --epochs 100
# 模型会自动上传到 https://huggingface.co/apt-model-256d-6l
```

### 输出文件

```
hf_output/
├── checkpoint-500/              # Checkpoint（每500步）
│   ├── config.json
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── final_model/                 # 最终模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_state.json
└── logs/                        # TensorBoard日志
    └── events.out.tfevents...
```

---

## 🎯 统一启动器

所有后端都可以通过统一启动器 `train.py` 使用：

```bash
# 查看所有可用后端
python training/train.py --list-backends

# Playground训练
python training/train.py --backend playground --epochs 100

# DeepSpeed训练
python training/train.py --backend deepspeed --num-gpus 2 --zero-stage 2

# Azure ML训练
python training/train.py --backend azure \
    --azure-subscription-id <ID> \
    --azure-resource-group <RG> \
    --azure-workspace-name <WS>

# HuggingFace训练
python training/train.py --backend huggingface --wandb --epochs 100
```

---

## 📊 性能对比

### 单卡RTX 3070（8GB显存）

| 后端 | Batch Size | 混合精度 | 内存使用 | 速度 |
|------|-----------|---------|---------|------|
| **Playground** | 16 | FP16 | 6.2GB | ⭐⭐⭐⭐ |
| **HuggingFace** | 16 | FP16 | 6.5GB | ⭐⭐⭐⭐ |
| **DeepSpeed (ZeRO-2)** | 32 | FP16 | 7.8GB | ⭐⭐⭐⭐⭐ |

### 多卡训练（4x RTX 3090）

| 后端 | ZeRO阶段 | Batch Size | 吞吐量 |
|------|---------|-----------|--------|
| **DeepSpeed** | ZeRO-2 | 128 | ⭐⭐⭐⭐⭐ |
| **HuggingFace + DS** | ZeRO-2 | 128 | ⭐⭐⭐⭐⭐ |

---

## 🔧 常见问题

### Q1: 应该选择哪个后端？

- **学习和实验**: Playground（简单直观）
- **多卡训练**: DeepSpeed（性能最优）
- **云端训练**: Azure ML（管理方便）
- **集成和分享**: HuggingFace（生态丰富）

### Q2: 如何恢复训练？

```bash
# Playground
python training/train_hlbd_playground.py --resume checkpoint_epoch_50.pt

# DeepSpeed
deepspeed train_deepspeed.py --load-checkpoint deepspeed_output/checkpoint_epoch_50

# HuggingFace
python training/train_hf_trainer.py --resume-from-checkpoint hf_output/checkpoint-500
```

### Q3: 如何验证HLBD模型？

```bash
python tools/verify_hlbd_model.py --model <模型路径> --dataset HLBD_Hardcore_Full.json
```

### Q4: 如何可视化训练？

```bash
# 实时可视化
python tools/visualize_training.py --log-dir hlbd_playground --mode realtime

# 离线可视化
python tools/visualize_training.py --log-dir hlbd_playground --mode offline

# 多训练监控
python tools/monitor_all_trainings.py
```

### Q5: DeepSpeed OOM怎么办？

```bash
# 1. 启用ZeRO-3
deepspeed train_deepspeed.py --zero-stage 3

# 2. 启用CPU卸载
deepspeed train_deepspeed.py --zero-stage 3 --cpu-offload

# 3. 增加梯度累积
deepspeed train_deepspeed.py --gradient-accumulation 8

# 4. 减小batch size
deepspeed train_deepspeed.py --train-batch-size 32
```

---

## 📚 相关文档

- [HLBD数据集生成](generate_hlbd_hardcore.py)
- [模型验证指南](verify_hlbd_model.py)
- [可视化使用指南](VISUALIZATION_GUIDE.md)
- [训练恢复指南](training_resume_guide.py)
- [问题诊断工具](diagnose_issues.py)

---

## 🎓 推荐训练流程

### 新手流程

```bash
# 1. 生成HLBD数据集
python generate_hlbd_hardcore.py

# 2. Playground训练
python training/train.py --backend playground --epochs 100

# 3. 验证模型
python tools/verify_hlbd_model.py --model hlbd_playground/final_model.pt

# 4. 可视化结果
python tools/visualize_training.py --log-dir hlbd_playground --mode offline
```

### 进阶流程

```bash
# 1. 多GPU DeepSpeed训练
python training/train.py --backend deepspeed --num-gpus 4 --zero-stage 2 --epochs 100

# 2. 实时监控
python tools/monitor_all_trainings.py &

# 3. 验证和诊断
python tools/verify_hlbd_model.py --model deepspeed_output/checkpoint_epoch_100
python tools/diagnose_issues.py
```

### 云端流程

```bash
# 1. 提交Azure ML任务
python training/train.py --backend azure \
    --azure-subscription-id <ID> \
    --azure-resource-group <RG> \
    --azure-workspace-name <WS> \
    --epochs 100

# 2. 查看MLflow实验
# 在Azure ML Studio中查看

# 3. 下载最佳模型
az ml model download --name apt-model --version 1
```

---

**选择合适的后端，开始你的APT训练之旅！** 🚀
