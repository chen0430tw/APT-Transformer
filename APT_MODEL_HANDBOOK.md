# APT Model 使用手册

**APT Model (自生成变换器)** - 一个功能完整的PyTorch Transformer训练平台

---

## 📑 目录

1. [快速开始](#快速开始)
2. [安装和配置](#安装和配置)
3. [核心功能](#核心功能)
4. [WebUI和API](#webui和api)
5. [训练模型](#训练模型)
6. [插件系统](#插件系统)
7. [高级功能](#高级功能)
8. [故障排除](#故障排除)

---

## 快速开始

### 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 验证torch安装
python -c "import torch; print(torch.__version__)"

# 可选：下载NLP资源（离线环境可跳过）
python scripts/download_optional_assets.py

# 运行测试
pytest tests/test_smoke.py
```

### 5分钟上手

```bash
# 1. 启动WebUI（推荐新手）
python -m apt_model.webui.app --checkpoint-dir ./checkpoints

# 2. 打开浏览器访问
# http://localhost:7860

# 3. 或者使用API
python -m apt_model.api.server --checkpoint-dir ./checkpoints
# 访问API文档: http://localhost:8000/docs
```

---

## 安装和配置

### 系统要求

- **Python**: 3.8+
- **PyTorch**: 1.10+ (支持CPU和GPU)
- **内存**: 最低4GB，推荐8GB+
- **磁盘**: 2GB+ (包含模型和数据)

### 可选依赖

```bash
# Transformer tokenizer和NLP工具
pip install transformers scikit-learn

# WebUI支持
pip install gradio

# API支持
pip install fastapi uvicorn

# 分布式训练
pip install torch.distributed

# 可视化
pip install tensorboard matplotlib
```

### 离线环境

项目支持完全离线运行，会自动降级到本地资源：
- Tokenizer使用内置中文词表
- 跳过可选依赖的功能
- 所有核心功能保持可用

---

## 核心功能

### 1. 模型训练

```python
from apt_model.training.trainer import train_model

# 基础训练
model, tokenizer, config = train_model(
    epochs=20,
    batch_size=8,
    learning_rate=3e-5,
    save_path="./my_model"
)
```

### 2. 文本生成

```python
from apt_model.generation.generator import generate_natural_text

# 生成文本
text, tokens, logits, confidence = generate_natural_text(
    model,
    tokenizer,
    prompt="人工智能",
    max_steps=50,
    temperature=0.8
)
```

### 3. 模型评估

```python
from apt_model.generation.evaluator import evaluate_text_quality

# 评估文本质量
score, feedback = evaluate_text_quality(generated_text)
print(f"质量评分: {score}/100 - {feedback}")
```

---

## WebUI和API

### WebUI功能

启动WebUI后可以访问4个功能Tab：

1. **训练监控**: 实时loss和学习率曲线
2. **梯度监控**: 梯度流分析和异常检测
3. **Checkpoint管理**: 加载和管理模型检查点
4. **推理测试**: 交互式文本生成

**启动命令**:
```bash
# 基础启动
python -m apt_model.webui.app --checkpoint-dir ./checkpoints

# 带认证
python -m apt_model.webui.app \
  --checkpoint-dir ./checkpoints \
  --username admin \
  --password your_password
```

### REST API

**10+ API端点**:

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/generate` | POST | 单条文本生成 |
| `/api/batch_generate` | POST | 批量文本生成 |
| `/api/training/status` | GET | 训练状态查询 |
| `/api/training/gradients` | GET | 梯度信息 |
| `/api/checkpoints` | GET | Checkpoint列表 |
| `/api/checkpoints/load` | POST | 加载Checkpoint |
| `/api/compression/methods` | GET | 可用压缩方法 |
| `/api/compression/apply` | POST | 应用压缩 |

**启动命令**:
```bash
python -m apt_model.api.server --checkpoint-dir ./checkpoints
```

**使用示例**:
```bash
# 文本生成
curl -X POST http://localhost:8000/api/generate \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_length": 50}'
```

**API密钥**: 启动时自动生成并显示在控制台，请保存好

---

## 训练模型

### 基础训练

```python
from apt_model.training.trainer import train_model

model, tokenizer, config = train_model(
    epochs=20,              # 训练轮数
    batch_size=8,           # 批次大小
    learning_rate=3e-5,     # 学习率
    save_path="./model",    # 保存路径
    texts=train_texts       # 训练数据（可选）
)
```

### 分布式训练

```bash
# 单机多卡（4 GPU）
bash scripts/launch_distributed.sh \
  --num-gpus 4 \
  --batch-size 32 \
  --data-path ./data

# 多节点训练
# 节点0 (master)
bash scripts/launch_distributed.sh \
  --num-gpus 4 \
  --num-nodes 2 \
  --node-rank 0 \
  --master-addr 192.168.1.100

# 节点1 (worker)
bash scripts/launch_distributed.sh \
  --num-gpus 4 \
  --num-nodes 2 \
  --node-rank 1 \
  --master-addr 192.168.1.100
```

### Checkpoint管理

**原子性保存**（防止checkpoint损坏）:
```python
from apt_model.training.checkpoint import CheckpointManager

# 创建管理器
mgr = CheckpointManager(
    save_dir="./checkpoints",
    model_name="apt_model",
    max_checkpoints=5
)

# 保存checkpoint（使用临时文件保证原子性）
mgr.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    metrics={'loss': 0.5, 'accuracy': 0.9}
)

# 加载checkpoint
checkpoint = mgr.load_checkpoint("checkpoint_epoch_10.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 插件系统

APT Model拥有完整的插件生态系统，支持动态扩展功能。

### 可用插件

**生产就绪插件** (6个):
- `BeamSearchPlugin` - Beam搜索解码
- `ProgramAidedPlugin` - 程序辅助推理
- `IterativeRefinementPlugin` - 迭代优化
- `SelfConsistencyPlugin` - 自洽性验证
- `MultiModalPlugin` - 多模态支持
- `CompressionPlugin` - 模型压缩

**工具类插件** (4个):
- `GradientMonitor` - 梯度监控
- `VersionManager` - 版本管理
- `ErrorPersistence` - 错误持久化
- `ProgressTracking` - 进度追踪

### 使用插件

```python
# 示例：使用压缩插件
from apt_model.plugins.compression_plugin import CompressionPlugin

plugin = CompressionPlugin()

# 启用DBC训练加速（20-30%速度提升）
model, optimizer = plugin.enable_dbc_training(
    model=model,
    rank_ratio=0.5,
    apply_to_gradients=True
)

# 应用模型压缩
compressed_model = plugin.compress(
    model=model,
    method='quantization',
    params={'bits': 8}
)
```

### 开发自定义插件

```python
from apt_model.plugins.base import PluginBase, PluginManifest

class MyPlugin(PluginBase):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="my_plugin",
            version="1.0.0",
            description="My custom plugin",
            author="Your Name"
        )

    def on_load(self):
        print("Plugin loaded!")

    def process(self, text: str) -> str:
        # 你的处理逻辑
        return text.upper()

# 使用插件
plugin = MyPlugin()
result = plugin.process("hello world")
```

---

## 高级功能

### 1. 模型压缩

**5种压缩方法**:

```python
from apt_model.plugins.compression_plugin import CompressionPlugin

plugin = CompressionPlugin()

# 方法1: 剪枝 (Pruning)
compressed = plugin.compress(
    model, method='pruning',
    params={'sparsity': 0.3}
)

# 方法2: 量化 (Quantization)
compressed = plugin.compress(
    model, method='quantization',
    params={'bits': 8}
)

# 方法3: 知识蒸馏 (Distillation)
compressed = plugin.compress(
    model, method='distillation',
    params={'teacher': teacher_model, 'temperature': 2.0}
)

# 方法4: 低秩分解 (Low-Rank)
compressed = plugin.compress(
    model, method='low_rank',
    params={'rank': 64}
)

# 方法5: DBC训练加速 (推荐)
model, optimizer = plugin.enable_dbc_training(
    model, rank_ratio=0.5, apply_to_gradients=True
)
# 训练速度提升20-30%！
```

### 2. 梯度监控

```python
from apt_model.training.gradient_monitor import GradientMonitor

# 创建监控器
monitor = GradientMonitor()

# 在训练循环中记录梯度
for step, batch in enumerate(dataloader):
    loss.backward()

    # 记录梯度
    monitor.record_gradients(model, step)

    # 检测异常
    anomalies = monitor.detect_anomalies(step)
    if anomalies:
        print(f"检测到梯度异常: {anomalies}")

    optimizer.step()

# 导出数据供WebUI使用
webui_data = monitor.export_for_webui()
```

### 3. 训练事件系统

```python
from apt_model.training.training_events import TrainingEventBus

# 创建事件总线
event_bus = TrainingEventBus()

# 订阅事件
def on_epoch_end(epoch, metrics):
    print(f"Epoch {epoch} 结束，指标: {metrics}")

event_bus.subscribe('epoch_end', on_epoch_end)

# 训练时会自动触发事件
# WebUI可以实时接收这些事件
```

### 4. 多语言支持

支持中文、英文和多语言混合训练：

```python
# 自动检测语言
from apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer

tokenizer, language = get_appropriate_tokenizer(
    texts=train_texts,
    tokenizer_type=None,  # 自动选择
    language=None         # 自动检测
)

print(f"检测到语言: {language}")
# 输出: 检测到语言: zh 或 en
```

---

## 故障排除

### 常见问题

**1. ModuleNotFoundError: No module named 'torch'**

```bash
pip install torch
# 或者
pip install torch torchvision torchaudio
```

**2. WebUI无法启动**

```bash
# 安装gradio
pip install gradio

# 或使用API代替
python -m apt_model.api.server
```

**3. API密钥丢失**

重启API服务器会重新生成密钥，或者使用自定义密钥：
```bash
python -m apt_model.api.server --api-key "your-secret-key"
```

**4. Checkpoint加载失败**

检查checkpoint是否损坏：
```python
import torch
checkpoint = torch.load("checkpoint.pt")
# 如果报错说明文件损坏
```

项目使用原子性保存机制，正常情况下checkpoint不会损坏。

**5. 训练速度慢**

```python
# 使用DBC训练加速
from apt_model.plugins.compression_plugin import CompressionPlugin
plugin = CompressionPlugin()
model, optimizer = plugin.enable_dbc_training(model, rank_ratio=0.5)
# 速度提升20-30%

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 使用分布式训练
bash scripts/launch_distributed.sh --num-gpus 4
```

**6. 内存不足**

```python
# 减小batch size
train_model(batch_size=4)  # 默认是8

# 启用梯度累积
# 在trainer.py中，accumulation_steps = 4

# 使用梯度checkpoint
model.gradient_checkpointing_enable()
```

### 获取帮助

1. **查看日志**: 训练日志保存在 `apt_model/log/`
2. **API文档**: http://localhost:8000/docs
3. **插件文档**: 查看 `apt_model/plugins/README.md`
4. **Issue追踪**: GitHub Issues

---

## 项目结构

```
apt_model/
├── config/              # 配置文件
│   ├── apt_config.py
│   └── multimodal_config.py
├── modeling/            # 模型定义
│   ├── apt_model.py
│   └── multimodal_model.py
├── training/            # 训练相关
│   ├── trainer.py
│   ├── checkpoint.py
│   ├── optimizer.py
│   └── gradient_monitor.py
├── generation/          # 生成和推理
│   ├── generator.py
│   └── evaluator.py
├── plugins/             # 插件系统
│   ├── base.py
│   ├── compression_plugin.py
│   └── version_manager.py
├── api/                 # REST API
│   └── server.py
├── webui/               # Web界面
│   └── app.py
├── utils/               # 工具函数
└── cli/                 # 命令行工具
```

---

## 快速参考

### 常用命令

```bash
# 训练
python -m apt_model.training.trainer

# WebUI
python -m apt_model.webui.app --checkpoint-dir ./checkpoints

# API
python -m apt_model.api.server --checkpoint-dir ./checkpoints

# 分布式
bash scripts/launch_distributed.sh --num-gpus 4

# 测试
pytest tests/

# 下载资源
python scripts/download_optional_assets.py
```

### 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 768 | 模型维度 |
| `num_heads` | 12 | 注意力头数 |
| `num_encoder_layers` | 4 | 编码器层数 |
| `num_decoder_layers` | 4 | 解码器层数 |
| `max_seq_len` | 128 | 最大序列长度 |
| `dropout` | 0.2 | Dropout率 |
| `learning_rate` | 3e-5 | 学习率 |

---

## 版本历史

- **v1.0.0** - 初始版本
  - 基础Transformer训练
  - 中英文支持
  - 插件系统
  - WebUI和API
  - 模型压缩
  - 分布式训练

---

**APT Model** - 让Transformer训练更简单！
