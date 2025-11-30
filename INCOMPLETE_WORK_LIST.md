# APT-Transformer 未完成工作清单

**生成日期**: 2025-11-29
**分支**: claude/review-memo-updates-01VZwZoRpMTGwNff9jviR9k7
**整体完成度**: 68%

---

## 分类说明

- 🔴 **Critical**: 阻塞生产使用，必须立即修复
- 🟡 **High**: 严重影响可用性，近期修复
- 🟢 **Medium**: 影响用户体验，计划修复
- 🔵 **Low**: 锦上添花，有时间再做

**工作量估计**:
- Small: < 4小时
- Medium: 4-16小时（0.5-2天）
- Large: 16-40小时（2-5天）
- XLarge: > 40小时（> 1周）

---

## 🔴 Critical - 必须立即修复 (3项)

### C1: 集成CheckpointManager到训练器 🔴
**优先级**: P0 - Critical
**工作量**: Medium (8-12小时)
**影响**: 训练中断后无法恢复，浪费计算资源

**问题描述**:
- `apt_model/training/checkpoint.py` 中的 `CheckpointManager` 类完整实现了checkpoint保存/加载
- 但 `apt_model/training/trainer.py:780` 只调用 `save_model()` 保存模型权重
- **缺失**: optimizer状态、scheduler状态、epoch、step、loss历史

**当前代码**:
```python
# trainer.py:780 (错误示例)
save_model(model, tokenizer, path=save_path, config=config)
# ❌ 只保存模型权重，无法恢复训练
```

**期望代码**:
```python
# trainer.py (正确示例)
checkpoint_mgr = CheckpointManager(
    save_dir="./outputs",
    model_name="apt_model",
    save_freq=1
)

# 恢复训练
start_epoch = 0
if resume_from:
    start_epoch, global_step, loss_history, metrics = checkpoint_mgr.load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=resume_from
    )

# 训练循环
for epoch in range(start_epoch, epochs):
    # ... 训练代码 ...

    # 保存完整checkpoint
    checkpoint_mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        loss_history=train_losses,
        metrics={"avg_loss": avg_loss},
        tokenizer=tokenizer,
        config=config,
        is_best=(avg_loss < best_loss)
    )
```

**文件修改**:
1. `apt_model/training/trainer.py`:
   - 添加 `checkpoint_dir` 参数 (default: `"./outputs"`)
   - 添加 `resume_from` 参数
   - 初始化 `CheckpointManager`
   - 在 `on_epoch_end` 调用 `save_checkpoint()`
   - 添加恢复训练逻辑

**验证**:
```bash
# 测试中断恢复
python -m apt_model.training.trainer --epochs 10
# Ctrl+C 在epoch 5中断

python -m apt_model.training.trainer \
    --resume-from ./outputs/checkpoints/apt_model_epoch5_step2500.pt \
    --epochs 10
# ✅ 应该从epoch 6继续
```

**相关文件**:
- `apt_model/training/trainer.py` (修改)
- `apt_model/training/checkpoint.py` (已完成，无需修改)

**参考文档**:
- `TRAINING_CHECKPOINT_MIGRATION_GUIDE.md` (lines 109-157)

---

### C2: 修复训练迁移问题（使用相对路径） 🔴
**优先级**: P0 - Critical
**工作量**: Small (4-6小时)
**影响**: 无法将训练迁移到其他电脑/服务器

**问题描述**:
- `apt_model/utils/cache_manager.py` 使用绝对路径 `~/.apt_cache`
- 导致checkpoint路径绑定到特定用户home目录
- **无法打包迁移**到其他电脑

**当前问题**:
```python
# cache_manager.py:42
self.cache_dir = os.path.expanduser("~/.apt_cache")
# → /home/userA/.apt_cache (绝对路径)

# 迁移到电脑B后：
# /home/userB/.apt_cache ❌ 找不到原checkpoint
```

**解决方案1**: 使用项目内相对路径（推荐）
```python
# 修改trainer.py
def train(..., checkpoint_dir="./outputs"):
    """
    参数:
        checkpoint_dir: checkpoint保存目录（相对路径）
    """
    checkpoint_mgr = CheckpointManager(save_dir=checkpoint_dir)
    # 保存到: APT-Transformer/outputs/checkpoints/
```

**解决方案2**: 改进CacheManager支持可迁移路径
```python
# cache_manager.py
class CacheManager:
    def __init__(self, cache_dir: Optional[str] = None,
                 use_project_dir: bool = True):
        if cache_dir is None:
            if use_project_dir:
                # 项目内缓存（可迁移）
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.cache_dir = os.path.join(project_root, ".cache")
            else:
                # 用户home缓存（传统方式）
                self.cache_dir = os.path.expanduser("~/.apt_cache")
```

**文件结构**（修改后）:
```
APT-Transformer/
├── outputs/                    # ✅ 相对路径，可打包迁移
│   ├── checkpoints/
│   │   ├── apt_model_epoch5_step2500_best.pt
│   │   └── apt_model_epoch10_step5000.pt
│   ├── metadata.json
│   └── tokenizer/
└── .cache/                     # ✅ 项目内缓存（可选）
    ├── temp/
    └── logs/
```

**迁移测试**:
```bash
# 电脑A
cd /path/to/APT-Transformer
tar -czf training_backup.tar.gz outputs/ .cache/

# 电脑B
cd /new/path/to/APT-Transformer
tar -xzf training_backup.tar.gz
python -m apt_model.training.trainer \
    --resume-from outputs/checkpoints/apt_model_epoch5_step2500_best.pt
# ✅ 应该能成功恢复
```

**文件修改**:
1. `apt_model/training/trainer.py`:
   - 使用 `checkpoint_dir="./outputs"` (相对路径)
2. `apt_model/utils/cache_manager.py`:
   - 添加 `use_project_dir` 参数
   - 修改默认行为为项目内路径

**相关文件**:
- `apt_model/training/trainer.py`
- `apt_model/utils/cache_manager.py`

**参考文档**:
- `TRAINING_CHECKPOINT_MIGRATION_GUIDE.md` (lines 176-220, 248-281)

---

### C3: 实现temp文件夹功能 🔴
**优先级**: P0 - Critical
**工作量**: Small (3-4小时)
**影响**: 训练中间状态无管理，崩溃后无法恢复

**问题描述**:
- `cache_manager.py` 定义了 `temp` 子目录但从未使用
- 训练过程中没有保存中间checkpoint
- 如果训练在epoch中间崩溃，从epoch开始重来浪费时间

**当前状态**:
```python
# cache_manager.py:58
"temp": os.path.join(self.cache_dir, "temp")  # ❌ 定义但未使用
```

**期望功能**:
```python
# trainer.py (训练循环中)
def train(...):
    temp_dir = ".cache/temp"
    os.makedirs(temp_dir, exist_ok=True)

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # ... 训练 ...
            global_step += 1

            # 每100步保存临时checkpoint
            if global_step % 100 == 0:
                temp_checkpoint = os.path.join(
                    temp_dir,
                    f"temp_epoch{epoch}_step{global_step}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'batch_idx': batch_idx
                }, temp_checkpoint)

        # epoch结束后清理temp文件
        for temp_file in glob.glob(os.path.join(temp_dir, "temp_*.pt")):
            os.remove(temp_file)
```

**使用场景**:
```bash
# 场景1: 训练在epoch中间崩溃
Epoch 5, batch 750/1000 → 崩溃

# 恢复:
找到: .cache/temp/temp_epoch5_step3750.pt
从batch 750继续，而不是从epoch 5开始重来
节省: 750 batches × 2秒 = 25分钟
```

**文件修改**:
1. `apt_model/training/trainer.py`:
   - 添加temp checkpoint保存逻辑（每N步）
   - 添加temp清理逻辑（epoch结束）
   - 添加从temp恢复功能

2. `apt_model/utils/cache_manager.py`:
   - 添加 `clean_temp()` 方法

**配置参数**:
```python
# config.yaml
training:
  temp_checkpoint_freq: 100  # 每100步保存一次
  keep_temp_files: false     # epoch结束是否保留temp
```

**相关文件**:
- `apt_model/training/trainer.py`
- `apt_model/utils/cache_manager.py`

**参考文档**:
- `TRAINING_CHECKPOINT_MIGRATION_GUIDE.md` (lines 222-246)

---

## 🟡 High - 近期修复 (6项)

### H1: 补充训练器单元测试 🟡
**优先级**: P1 - High
**工作量**: Large (20-24小时)
**影响**: 代码质量无保证，易引入bug

**缺失测试**:
1. **训练循环测试**:
   ```python
   # tests/test_trainer.py
   def test_training_loop():
       trainer = Trainer(...)
       metrics = trainer.train(epochs=2, batch_size=4)
       assert 'loss' in metrics
       assert metrics['loss'][-1] < metrics['loss'][0]  # 损失下降
   ```

2. **Checkpoint保存/加载测试**:
   ```python
   def test_checkpoint_save_load():
       trainer.train(epochs=5)
       checkpoint = torch.load("outputs/checkpoints/apt_model_epoch5.pt")
       assert 'optimizer_state_dict' in checkpoint
       assert 'scheduler_state_dict' in checkpoint
       assert checkpoint['epoch'] == 5
   ```

3. **训练恢复测试**:
   ```python
   def test_resume_training():
       trainer1 = Trainer(...)
       trainer1.train(epochs=5)

       trainer2 = Trainer(...)
       trainer2.train(epochs=10, resume_from="outputs/.../epoch5.pt")
       # 应该从epoch 6开始
       assert trainer2.start_epoch == 5
   ```

4. **Early stopping测试**:
   ```python
   def test_early_stopping():
       trainer = Trainer(callbacks=[EarlyStoppingCallback(patience=3)])
       metrics = trainer.train(epochs=100)
       # 应该在<100 epochs停止
       assert len(metrics['loss']) < 100
   ```

5. **多GPU测试**（可选）:
   ```python
   @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要GPU")
   def test_distributed_training():
       trainer = Trainer(use_ddp=True)
       # ...
   ```

**测试覆盖目标**:
- 训练循环: 90%+
- Checkpoint系统: 95%+
- Callback系统: 80%+

**文件创建**:
- `tests/test_trainer.py` (新建)
- `tests/test_checkpoint.py` (新建)
- `tests/test_callbacks.py` (新建)

**工具**:
- pytest
- pytest-cov (覆盖率)
- pytest-mock (mock外部依赖)

---

### H2: 补充模型架构单元测试 🟡
**优先级**: P1 - High
**工作量**: Medium (12-16小时)
**影响**: 模型正确性无验证

**缺失测试**:
1. **Transformer前向传播**:
   ```python
   # tests/test_transformer.py
   def test_transformer_forward():
       model = TransformerModel(vocab_size=1000, d_model=512)
       input_ids = torch.randint(0, 1000, (2, 10))
       output = model(input_ids)
       assert output.shape == (2, 10, 512)
   ```

2. **Attention机制**:
   ```python
   def test_multi_head_attention():
       attn = MultiHeadAttention(d_model=512, num_heads=8)
       q = k = v = torch.randn(2, 10, 512)
       output, weights = attn(q, k, v)
       assert output.shape == (2, 10, 512)
       assert weights.shape == (2, 8, 10, 10)  # (batch, heads, seq, seq)
   ```

3. **位置编码**:
   ```python
   def test_positional_encoding():
       pe = PositionalEncoding(d_model=512, max_len=100)
       x = torch.randn(2, 50, 512)
       output = pe(x)
       assert output.shape == x.shape
   ```

4. **梯度检查**:
   ```python
   def test_model_gradients():
       model = TransformerModel(...)
       optimizer = torch.optim.Adam(model.parameters())
       input_ids = torch.randint(0, 1000, (2, 10))
       output = model(input_ids)
       loss = output.sum()
       loss.backward()
       # 检查梯度非零
       assert any(p.grad is not None for p in model.parameters())
   ```

**文件创建**:
- `tests/test_transformer.py` (新建)
- `tests/test_apt_model.py` (新建)

---

### H3: 编写快速开始文档 🟡
**优先级**: P1 - High
**工作量**: Medium (8-10小时)
**影响**: 新用户无法上手

**缺失内容**:
1. **安装指南**:
   ```markdown
   # 快速开始

   ## 安装

   ### 环境要求
   - Python 3.8+
   - PyTorch 2.0+
   - CUDA 11.8+ (可选，用于GPU加速)

   ### 安装步骤
   ```bash
   git clone https://github.com/your-org/APT-Transformer.git
   cd APT-Transformer
   pip install -r requirements.txt
   ```
   ```

2. **基础使用示例**:
   ```markdown
   ## 训练模型

   ### 准备数据
   ```python
   from apt_model.data import create_dataloader
   train_loader = create_dataloader(
       data_path="data/train.json",
       batch_size=32
   )
   ```

   ### 开始训练
   ```python
   from apt_model.training import train
   from apt_model.config import APTConfig

   config = APTConfig.from_yaml("config/default.yaml")
   metrics = train(
       config=config,
       train_dataloader=train_loader,
       epochs=10,
       checkpoint_dir="./outputs"
   )
   ```

   ### 恢复训练
   ```python
   metrics = train(
       config=config,
       train_dataloader=train_loader,
       epochs=20,
       resume_from="./outputs/checkpoints/apt_model_epoch10.pt"
   )
   ```
   ```

3. **模型推理示例**:
   ```markdown
   ## 使用模型

   ### 加载模型
   ```python
   from apt_model.training.checkpoint import load_model
   model, tokenizer, config = load_model(
       "outputs/checkpoints/apt_model_best.pt"
   )
   ```

   ### 推理
   ```python
   input_text = "Hello, APT!"
   input_ids = tokenizer.encode(input_text)
   output = model(input_ids)
   ```
   ```

4. **EQI决策流水线示例**:
   ```markdown
   ## 使用EQI决策系统

   ```python
   from apt_eqi_manager import DecisionPipeline, SAFModule, COCScenario

   # 定义模块
   modules = [
       SAFModule(name="legacy_db", S=0.9, D=0.5, R=1.0),
       # ...
   ]

   # 运行决策流水线
   pipeline = DecisionPipeline()
   report = pipeline.run_full_pipeline(
       modules=modules,
       scenarios={...},
       budget=100,
       max_parallel=2
   )
   print(report)
   ```
   ```

**文件创建**:
- `QUICK_START.md` (新建)
- `docs/installation.md` (新建)
- `docs/training_guide.md` (新建)
- `docs/inference_guide.md` (新建)

---

### H4: 补充API文档（docstring） 🟡
**优先级**: P1 - High
**工作量**: Large (16-20小时)
**影响**: 代码可读性差，难以维护

**问题描述**:
- 很多函数缺少docstring
- 现有docstring格式不统一
- 缺少参数类型标注

**当前示例**（不完整）:
```python
# trainer.py
def train(config, train_dataloader, epochs):
    # 没有docstring ❌
    pass
```

**期望格式**（Google风格）:
```python
def train(
    config: APTConfig,
    train_dataloader: DataLoader,
    epochs: int,
    checkpoint_dir: str = "./outputs",
    resume_from: Optional[str] = None,
    callbacks: Optional[List[TrainingCallback]] = None
) -> Dict[str, Any]:
    """训练APT模型

    Args:
        config: 模型配置对象
        train_dataloader: 训练数据加载器
        epochs: 训练轮数
        checkpoint_dir: checkpoint保存目录，默认"./outputs"
        resume_from: 恢复训练的checkpoint路径，可选
        callbacks: 训练回调列表，可选

    Returns:
        dict: 训练指标字典，包含:
            - loss (List[float]): 每个epoch的平均损失
            - accuracy (List[float]): 每个epoch的准确率
            - learning_rate (List[float]): 每个epoch的学习率

    Raises:
        FileNotFoundError: 当resume_from指定的文件不存在时
        ValueError: 当config验证失败时

    Example:
        >>> config = APTConfig.from_yaml("config.yaml")
        >>> train_loader = create_dataloader("data/train.json")
        >>> metrics = train(config, train_loader, epochs=10)
        >>> print(f"Final loss: {metrics['loss'][-1]}")
    """
    pass
```

**待补充文件**（优先级排序）:
1. `apt_model/training/trainer.py` - 核心训练逻辑
2. `apt_model/training/checkpoint.py` - Checkpoint管理
3. `apt_model/models/apt_transformer.py` - 模型定义
4. `apt_eqi_manager.py` - 决策系统
5. `apt_model/data/dataloader.py` - 数据加载
6. `apt_model/infrastructure/errors.py` - 错误处理

**工具**:
- sphinx (生成HTML文档)
- sphinx-autodoc (自动从docstring生成文档)

**生成文档**:
```bash
cd docs/
sphinx-apidoc -o source/ ../apt_model/
make html
```

---

### H5: 创建Docker镜像 🟡
**优先级**: P1 - High
**工作量**: Medium (8-12小时)
**影响**: 无法快速部署，环境一致性差

**缺失内容**:
1. **Dockerfile**:
   ```dockerfile
   # Dockerfile
   FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

   WORKDIR /app

   # 安装系统依赖
   RUN apt-get update && apt-get install -y \
       git \
       && rm -rf /var/lib/apt/lists/*

   # 复制项目文件
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   # 创建输出目录
   RUN mkdir -p /app/outputs /app/.cache

   # 默认命令
   CMD ["python", "-m", "apt_model.training.trainer", "--config", "config/default.yaml"]
   ```

2. **docker-compose.yml**:
   ```yaml
   # docker-compose.yml
   version: '3.8'
   services:
     apt-trainer:
       build: .
       image: apt-transformer:latest
       volumes:
         - ./data:/app/data
         - ./outputs:/app/outputs
         - ./config:/app/config
       environment:
         - CUDA_VISIBLE_DEVICES=0
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

3. **.dockerignore**:
   ```
   # .dockerignore
   .git
   .gitignore
   *.pyc
   __pycache__
   .cache
   outputs
   *.pt
   *.pth
   .pytest_cache
   ```

4. **构建和运行脚本**:
   ```bash
   # scripts/docker_build.sh
   #!/bin/bash
   docker build -t apt-transformer:latest .

   # scripts/docker_train.sh
   #!/bin/bash
   docker run --gpus all \
       -v $(pwd)/data:/app/data \
       -v $(pwd)/outputs:/app/outputs \
       apt-transformer:latest
   ```

**验证**:
```bash
# 构建镜像
./scripts/docker_build.sh

# 运行训练
./scripts/docker_train.sh

# 交互式进入容器
docker run -it --gpus all apt-transformer:latest bash
```

**文件创建**:
- `Dockerfile` (新建)
- `docker-compose.yml` (新建)
- `.dockerignore` (新建)
- `scripts/docker_build.sh` (新建)
- `scripts/docker_train.sh` (新建)

---

### H6: 完善requirements.txt 🟡
**优先级**: P1 - High
**工作量**: Small (2-3小时)
**影响**: 依赖安装困难，环境不一致

**问题描述**:
- 现有`requirements.txt`可能不完整
- 缺少版本锁定
- 缺少开发依赖

**期望内容**:
```txt
# requirements.txt (生产依赖)

# 核心依赖
torch>=2.0.0,<2.2.0
numpy>=1.24.0,<2.0.0
transformers>=4.30.0,<5.0.0

# 数据处理
pandas>=2.0.0
datasets>=2.12.0

# 训练工具
tqdm>=4.65.0
tensorboard>=2.13.0
wandb>=0.15.0  # 可选

# 配置管理
pyyaml>=6.0
omegaconf>=2.3.0

# 工具
rich>=13.4.0  # 进度条美化
loguru>=0.7.0  # 日志

# 推理
onnx>=1.14.0  # 可选
onnxruntime>=1.15.0  # 可选
```

```txt
# requirements-dev.txt (开发依赖)

# 测试
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-asyncio>=0.21.0

# 代码质量
black>=23.3.0
flake8>=6.0.0
mypy>=1.4.0
isort>=5.12.0

# 文档
sphinx>=7.0.0
sphinx-rtd-theme>=1.2.0
myst-parser>=2.0.0  # Markdown支持

# 性能分析
py-spy>=0.3.14
memory-profiler>=0.61.0
```

**版本锁定**:
```bash
# 生成完整锁定版本
pip freeze > requirements-lock.txt
```

**安装脚本**:
```bash
# scripts/install_deps.sh
#!/bin/bash

# 生产环境
pip install -r requirements.txt

# 开发环境
if [ "$DEV" = "true" ]; then
    pip install -r requirements-dev.txt
fi
```

**文件修改/创建**:
- `requirements.txt` (修改)
- `requirements-dev.txt` (新建)
- `requirements-lock.txt` (新建)
- `scripts/install_deps.sh` (新建)

---

## 🟢 Medium - 计划修复 (8项)

### M1: 实现Flash Attention优化 🟢
**优先级**: P2 - Medium
**工作量**: Medium (10-14小时)
**影响**: 训练速度提升2-3x

**当前实现**:
```python
# apt_model/models/transformer.py
# 使用标准PyTorch attention
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, v)
```

**优化后**:
```python
# 使用Flash Attention 2
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
# 内存使用: O(N) vs O(N²)
# 速度提升: 2-3x
```

**依赖**:
```bash
pip install flash-attn --no-build-isolation
```

**兼容性**:
- 仅支持CUDA
- 需要Ampere架构（RTX 30系列+）或更新

**文件修改**:
- `apt_model/models/transformer.py`

---

### M2: 添加混合精度训练 🟢
**优先级**: P2 - Medium
**工作量**: Small (4-6小时)
**影响**: 内存节省~40%, 速度提升~2x

**实现**:
```python
# trainer.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # 混合精度前向传播
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # 混合精度反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**配置**:
```yaml
# config.yaml
training:
  use_mixed_precision: true
  fp16: true  # 或 bf16 (更稳定)
```

**文件修改**:
- `apt_model/training/trainer.py`
- `apt_model/config/training_config.py`

---

### M3: 实现分布式训练（DDP） 🟢
**优先级**: P2 - Medium
**工作量**: Large (20-24小时)
**影响**: 支持多GPU训练，速度线性提升

**实现**:
```python
# trainer.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size, ...):
    setup_ddp(rank, world_size)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 训练循环
    for batch in dataloader:
        # ...
```

**启动**:
```bash
# 单机4卡
torchrun --nproc_per_node=4 -m apt_model.training.trainer

# 多机训练
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.1" \
         --master_port=29500 \
         -m apt_model.training.trainer
```

**文件修改**:
- `apt_model/training/trainer.py`
- `apt_model/training/distributed.py` (新建)

---

### M4: 添加梯度累积 🟢
**优先级**: P2 - Medium
**工作量**: Small (3-4小时)
**影响**: 支持大batch训练，提升收敛速度

**实现**:
```python
# trainer.py
accumulation_steps = 4  # 累积4个batch

for batch_idx, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    # 每N步更新一次
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**配置**:
```yaml
training:
  gradient_accumulation_steps: 4
  effective_batch_size: 128  # = batch_size * accumulation_steps
```

**文件修改**:
- `apt_model/training/trainer.py`

---

### M5: 实现模型量化 🟢
**优先级**: P2 - Medium
**工作量**: Medium (12-16小时)
**影响**: 推理速度提升2-4x，模型大小减少75%

**实现方案**:
1. **动态量化**（最简单）:
   ```python
   import torch.quantization
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **静态量化**:
   ```python
   # 校准
   model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
   model_prepared = torch.quantization.prepare(model)
   # 喂入校准数据
   calibrate(model_prepared, calibration_data)
   # 量化
   quantized_model = torch.quantization.convert(model_prepared)
   ```

3. **量化感知训练**（QAT）:
   ```python
   model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
   model_prepared = torch.quantization.prepare_qat(model)
   # 正常训练
   train(model_prepared)
   # 量化
   quantized_model = torch.quantization.convert(model_prepared)
   ```

**文件创建**:
- `apt_model/quantization/quantize.py` (新建)
- `scripts/quantize_model.sh` (新建)

---

### M6: 添加TensorBoard/wandb集成 🟢
**优先级**: P2 - Medium
**工作量**: Small (4-6小时)
**影响**: 可视化训练过程

**TensorBoard实现**:
```python
# trainer.py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/apt_training")

for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        # ...
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('LR', lr, global_step)

    # epoch结束记录
    writer.add_scalar('Loss/epoch', avg_loss, epoch)
    writer.add_histogram('Gradients/layer1', model.layer1.weight.grad, epoch)
```

**wandb实现**:
```python
import wandb

wandb.init(project="apt-transformer", config=config)

for epoch in range(epochs):
    # ...
    wandb.log({
        "loss": loss.item(),
        "lr": lr,
        "epoch": epoch
    })
```

**查看**:
```bash
# TensorBoard
tensorboard --logdir=runs/

# wandb
wandb login
# 访问 https://wandb.ai/your-project
```

**文件修改**:
- `apt_model/training/callbacks.py` (添加TensorBoardCallback)

---

### M7: 实现模型导出（ONNX） 🟢
**优先级**: P2 - Medium
**工作量**: Medium (8-10小时)
**影响**: 支持跨平台部署

**实现**:
```python
# apt_model/export/onnx_export.py
import torch.onnx

def export_to_onnx(model, save_path, input_sample):
    """导出模型为ONNX格式"""
    model.eval()

    torch.onnx.export(
        model,
        input_sample,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        }
    )

# 使用
model, tokenizer, config = load_model("best_model.pt")
input_sample = torch.randint(0, config.vocab_size, (1, 128))
export_to_onnx(model, "model.onnx", input_sample)
```

**推理**:
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
inputs = {"input_ids": input_ids.numpy()}
outputs = session.run(None, inputs)
```

**验证**:
```python
# 验证ONNX输出与PyTorch一致
torch_output = model(input_ids)
onnx_output = session.run(None, {"input_ids": input_ids.numpy()})[0]
assert np.allclose(torch_output.detach().numpy(), onnx_output, atol=1e-5)
```

**文件创建**:
- `apt_model/export/onnx_export.py` (新建)
- `scripts/export_onnx.sh` (新建)

---

### M8: 添加性能基准测试 🟢
**优先级**: P2 - Medium
**工作量**: Medium (10-12小时)
**影响**: 了解性能瓶颈

**实现**:
```python
# benchmarks/benchmark_training.py
import time
import torch
from apt_model.training import train

def benchmark_training_speed():
    """测试训练速度"""
    start = time.time()

    metrics = train(
        config=config,
        train_dataloader=train_loader,
        epochs=5
    )

    elapsed = time.time() - start
    samples_per_sec = total_samples / elapsed

    print(f"训练速度: {samples_per_sec:.2f} samples/sec")
    print(f"每个epoch: {elapsed/5:.2f}秒")

def benchmark_memory_usage():
    """测试内存使用"""
    torch.cuda.reset_peak_memory_stats()

    model = create_model(config)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"峰值GPU内存: {peak_memory:.2f} GB")

def benchmark_inference_latency():
    """测试推理延迟"""
    model.eval()
    with torch.no_grad():
        latencies = []
        for _ in range(100):
            start = time.time()
            output = model(input_ids)
            latencies.append(time.time() - start)

    print(f"平均延迟: {np.mean(latencies)*1000:.2f}ms")
    print(f"P95延迟: {np.percentile(latencies, 95)*1000:.2f}ms")
```

**运行**:
```bash
python benchmarks/benchmark_training.py
```

**输出示例**:
```
训练速度: 245.3 samples/sec
每个epoch: 122.5秒
峰值GPU内存: 8.45 GB
---
平均延迟: 12.3ms
P95延迟: 18.7ms
```

**文件创建**:
- `benchmarks/benchmark_training.py` (新建)
- `benchmarks/benchmark_inference.py` (新建)
- `benchmarks/benchmark_memory.py` (新建)

---

## 🔵 Low - 锦上添花 (5项)

### L1: 动态参数调整（EQI/COC） 🔵
**优先级**: P3 - Low
**工作量**: Medium (10-12小时)
**影响**: EQI系统更灵活

**当前问题**:
```python
# COC中的α, β硬编码
α = 0.3  # 当前复杂度权重
β = 0.2  # 复杂度漂移权重
```

**期望**:
```python
class COCAnalyzer:
    def __init__(self, alpha: float = 0.3, beta: float = 0.2):
        self.alpha = alpha
        self.beta = beta

    def set_weights(self, alpha: float, beta: float):
        """动态调整权重"""
        self.alpha = alpha
        self.beta = beta

# 使用
analyzer = COCAnalyzer()
analyzer.set_weights(alpha=0.5, beta=0.1)  # 重视当前复杂度
```

**文件修改**:
- `apt_eqi_manager.py`

---

### L2: 添加自定义EQI门禁 🔵
**优先级**: P3 - Low
**工作量**: Small (6-8小时)
**影响**: EQI系统更通用

**实现**:
```python
# apt_eqi_manager.py
class EQIGate:
    """可扩展的门禁基类"""
    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold

    def evaluate(self, evidence: Dict[str, Any]) -> Tuple[bool, float]:
        """子类实现具体评估逻辑"""
        raise NotImplementedError

class CustomGate(EQIGate):
    """用户自定义门禁"""
    def evaluate(self, evidence):
        # 自定义逻辑
        score = custom_scoring(evidence)
        passed = score >= self.threshold
        return passed, score

# 使用
manager = EQIManager()
manager.add_gate(CustomGate("custom_security", threshold=0.8))
```

**文件修改**:
- `apt_eqi_manager.py`

---

### L3: 实现LoRA微调支持 🔵
**优先级**: P3 - Low
**工作量**: Large (20-24小时)
**影响**: 参数高效微调

**实现**:
```python
# apt_model/lora/lora_layer.py
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 0.01

        # 冻结原始权重
        self.linear.weight.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
```

**文件创建**:
- `apt_model/lora/` (新建目录)

---

### L4: 添加日志轮转 🔵
**优先级**: P3 - Low
**工作量**: Small (2-3小时)
**影响**: 防止日志文件过大

**实现**:
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "apt_training.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

**文件修改**:
- `apt_model/infrastructure/logging.py`

---

### L5: 添加配置版本控制 🔵
**优先级**: P3 - Low
**工作量**: Small (4-5小时)
**影响**: 实验可复现性

**实现**:
```python
# 保存配置时添加版本号和hash
config_with_version = {
    "config_version": "1.0.0",
    "config_hash": hashlib.md5(json.dumps(config).encode()).hexdigest(),
    "timestamp": datetime.now().isoformat(),
    **config
}
```

**文件修改**:
- `apt_model/config/config.py`

---

## 工作量汇总

### 按优先级

| 优先级 | 任务数 | 总工作量 | 平均工作量 |
|--------|--------|----------|------------|
| 🔴 Critical | 3 | 19-22小时 (~3天) | 6-7小时 |
| 🟡 High | 6 | 76-96小时 (~10-12天) | 13-16小时 |
| 🟢 Medium | 8 | 87-107小时 (~11-13天) | 11-13小时 |
| 🔵 Low | 5 | 42-52小时 (~5-7天) | 8-10小时 |
| **总计** | **22** | **224-277小时 (~28-35天)** | **10-13小时** |

### 按类型

| 类型 | 任务数 | 工作量 |
|------|--------|--------|
| 训练系统修复 | 3 | 19-22小时 |
| 测试补充 | 2 | 32-40小时 |
| 文档编写 | 3 | 26-33小时 |
| 部署支持 | 2 | 10-15小时 |
| 性能优化 | 6 | 67-82小时 |
| 功能增强 | 6 | 70-85小时 |

---

## 推荐执行顺序

### Sprint 1 (Week 1): Critical修复
**目标**: 解决阻塞性问题，使项目基本可用

1. **C1**: 集成CheckpointManager (Day 1-2)
2. **C2**: 修复训练迁移问题 (Day 2)
3. **C3**: 实现temp文件夹功能 (Day 3)
4. **验证**: 端到端训练+中断恢复+迁移测试 (Day 3)

**验收标准**:
- ✅ 训练可以从任意epoch恢复
- ✅ checkpoint可以打包迁移到其他机器
- ✅ temp文件夹正常工作

---

### Sprint 2 (Week 2): 测试和文档
**目标**: 补充测试覆盖率，提升代码质量

1. **H1**: 训练器单元测试 (Day 1-3)
2. **H2**: 模型架构单元测试 (Day 3-4)
3. **H3**: 快速开始文档 (Day 4-5)
4. **H4**: API文档 (Day 5-7)

**验收标准**:
- ✅ 测试覆盖率 > 60%
- ✅ 新用户可以在30分钟内完成训练

---

### Sprint 3 (Week 3): 部署和基础设施
**目标**: 支持生产部署

1. **H5**: Docker镜像 (Day 1-2)
2. **H6**: 完善requirements.txt (Day 2)
3. **M6**: TensorBoard集成 (Day 3)
4. **M8**: 性能基准测试 (Day 3-4)

**验收标准**:
- ✅ 可以一键Docker部署
- ✅ 有性能基准数据

---

### Sprint 4 (Week 4): 性能优化
**目标**: 提升训练速度2-3x

1. **M2**: 混合精度训练 (Day 1)
2. **M4**: 梯度累积 (Day 1)
3. **M1**: Flash Attention (Day 2-3)
4. **M3**: 分布式训练 (Day 3-5)

**验收标准**:
- ✅ 训练速度提升 > 2x
- ✅ 支持多GPU训练

---

### Sprint 5+ (Week 5+): 增强功能
**目标**: 完善高级功能

- M5: 模型量化
- M7: ONNX导出
- L1-L5: Low优先级任务

---

## 资源需求

### 人力
- **1名工程师**: 全职 ~6-8周完成所有任务
- **2名工程师**: 全职 ~3-4周完成所有任务
- **优先完成Critical+High**: ~2周（1人）或 ~1周（2人）

### 硬件
- GPU: 至少1张用于测试（RTX 3090或更好）
- 存储: ~100GB（数据+checkpoint+Docker镜像）
- 内存: 32GB+（用于大模型测试）

### 工具
- pytest, pytest-cov (测试)
- sphinx (文档)
- Docker (部署)
- TensorBoard/wandb (监控)

---

## 风险评估

### 高风险
1. **分布式训练（M3）**: 复杂度高，调试困难
   - 缓解: 先实现单机多卡，再扩展多机
2. **Flash Attention（M1）**: 依赖特定硬件
   - 缓解: 提供fallback到标准attention

### 中风险
3. **测试覆盖率**: 补充测试工作量大
   - 缓解: 优先核心功能，逐步覆盖
4. **Docker集成**: 依赖管理复杂
   - 缓解: 使用官方PyTorch镜像作为基础

### 低风险
5. **文档编写**: 时间投入但风险低
6. **混合精度**: 成熟技术，实现简单

---

## 总结

**当前状态**: 68%成熟度，核心功能完备但工程实践不足

**Critical问题**: 3个，~3天可修复，修复后成熟度提升至80%+

**完整路线图**: 22个任务，~28-35天完成，最终成熟度可达90%+

**投入产出比**:
- **最小可行版本**（Critical修复）: 3天 → 80%成熟度
- **生产就绪版本**（+High任务）: 2周 → 85%成熟度
- **完整优化版本**（全部任务）: 5-7周 → 90%+成熟度

**建议**: 优先完成Sprint 1（Critical修复），然后根据实际需求决定后续Sprint的优先级。
