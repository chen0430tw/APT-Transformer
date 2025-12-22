# APT Model (自生成变换器)

<div align="center">

**一个功能完整的PyTorch Transformer训练平台**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[特性](#特性) • [快速开始](#快速开始) • [文档](docs/APT_MODEL_HANDBOOK.md) • [示例](#使用示例)

</div>

---

## 简介

APT Model 是一个生产就绪的Transformer训练平台，提供完整的训练、推理、评估和部署工具链。支持中英文多语言，具备丰富的插件生态系统和分布式训练能力。

## 特性

### 🚀 核心功能
- **完整的训练流程** - 从数据处理到模型部署的完整pipeline
- **多语言支持** - 原生支持中文和英文，自动语言检测
- **分布式训练** - 多GPU和多节点训练支持（PyTorch DDP）
- **模型压缩** - 5种压缩方法，包括DBC训练加速（20-30%提升）

### 🔌 插件系统
- **26+生产插件** - BeamSearch、Self-Consistency、Multi-Modal等
- **可扩展架构** - 事件驱动的插件系统，易于开发自定义插件
- **热插拔支持** - 动态加载和卸载插件

### 🌐 Web服务
- **WebUI界面** - 基于Gradio的交互式界面，4个功能Tab
- **REST API** - 完整的FastAPI服务，10+端点，自动生成文档
- **实时监控** - 训练进度、梯度流、资源使用的实时可视化

### 🛡️ 生产特性
- **Checkpoint保护** - 原子性保存机制，防止训练中断损坏
- **依赖容错** - 离线友好，可选依赖优雅降级
- **Debug模式** - 持久化配置系统，完整的CLI命令

---

## 快速开始

### ⚡ 超快速上手（30秒）

```bash
# 0. 克隆仓库
git clone https://github.com/chen0430tw/APT-Transformer.git
cd APT-Transformer

# 1. 安装（二选一）
pip install -r requirements.txt          # 完整安装
pip install -r requirements-minimal.txt  # 最小安装

# 2. 训练一个模型
python -m apt_model train --data data.txt --epochs 10

# 3. 文本生成
python -m apt_model chat
```

<details>
<summary><b>📋 查看完整安装步骤</b></summary>

### 完整安装指南

#### 1. 克隆仓库
```bash
git clone https://github.com/chen0430tw/APT-Transformer.git
cd APT-Transformer
```

#### 2. 安装 PyTorch

**重要：** 根据您的硬件选择正确的PyTorch版本：

<details>
<summary><b>🖥️ CPU版本（无NVIDIA显卡）</b></summary>

适用于没有NVIDIA显卡或仅用于推理的环境：

```bash
# CPU版本 - 体积较小，无需CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**特点：**
- ✅ 体积小（约100MB）
- ✅ 无需CUDA环境
- ✅ 适合CPU推理和小规模训练
- ⚠️ 训练速度较慢（约为GPU的1/10-1/50）

</details>

<details>
<summary><b>⚡ CUDA版本（有NVIDIA显卡）- 推荐</b></summary>

适用于拥有NVIDIA显卡的环境，提供显著加速：

```bash
# CUDA 11.8版本（兼容性好）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本（推荐，兼容CUDA 12.2和12.3）
# 注意：PyTorch跳过了cu122和cu123，使用cu121即可
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.6版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 13.0版本（最新）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**如何查看CUDA版本：**
```bash
nvidia-smi  # 查看"CUDA Version"
```

**特点：**
- ✅ 训练速度快10-50倍
- ✅ 支持大batch size
- ✅ 支持混合精度训练（FP16）
- ⚠️ 体积较大（约2GB）
- ⚠️ 需要NVIDIA显卡和对应的CUDA驱动

**显卡要求：**
- 最低：GTX 1060 (6GB VRAM)
- 推荐：RTX 3060+ (12GB+ VRAM)
- 最佳：RTX 4090 / A100 (24GB+ VRAM)

</details>

#### 3. 安装项目依赖

```bash
# 安装其他依赖
pip install -r requirements.txt

# 安装 apt_model 包（开发模式，重要！）
pip install -e .
```

#### 4. 验证安装

```bash
# 检查PyTorch版本和CUDA可用性
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证apt_model安装
python -m apt_model --help
```

**期望输出：**
- CPU版本：`CUDA available: False`
- GPU版本：`CUDA available: True`

---

**📌 安装故障排除**

如果遇到问题，请参考：
- **模块导入错误**：[INSTALLATION.md](INSTALLATION.md)
- **CUDA问题**：确认显卡驱动已正确安装
- **依赖冲突**：建议使用虚拟环境（`python -m venv venv`）

</details>

### 5分钟上手

#### 1. 启动WebUI（推荐）
```bash
python -m apt_model.webui.app --checkpoint-dir ./checkpoints
```
访问 http://localhost:7860 即可使用交互式界面。

#### 2. 训练模型
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

#### 3. 文本生成
```python
from apt_model.generation.generator import generate_natural_text

text, tokens, logits, confidence = generate_natural_text(
    model,
    tokenizer,
    prompt="人工智能",
    max_steps=50,
    temperature=0.8
)
print(text)
```

---

## 使用示例

### WebUI服务

启动带认证的WebUI：
```bash
python -m apt_model.webui.app \
  --checkpoint-dir ./checkpoints \
  --username admin \
  --password your_password \
  --port 7860
```

WebUI提供4个功能Tab：
- **训练监控** - 实时loss和学习率曲线
- **梯度监控** - 梯度流分析和异常检测
- **Checkpoint管理** - 加载和管理模型检查点
- **推理测试** - 交互式文本生成

### REST API服务

```bash
# 启动API服务
python -m apt_model.api.server --checkpoint-dir ./checkpoints

# 使用API生成文本
curl -X POST http://localhost:8000/api/generate \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_length": 50}'
```

API文档自动生成：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 分布式训练 ⭐ 多后端支持

APT支持5种训练后端，满足从单卡到大规模云端训练的所有需求：

| 后端 | 特点 | 适用场景 |
|------|------|---------|
| **Playground** | Cosine重启学习率 | HLBD数据集训练 |
| **DeepSpeed** | ZeRO-2/3优化 | 多GPU分布式训练 |
| **Azure ML** | MLflow跟踪 | 云端大规模训练 |
| **HuggingFace** | W&B集成 | 生态系统集成 |

```bash
# 查看所有可用后端
python training/train.py --list-backends

# Playground训练（推荐HLBD）
python training/train.py --backend playground --epochs 100

# DeepSpeed分布式训练
python training/train.py --backend deepspeed --num-gpus 4 --zero-stage 2

# Azure ML云端训练
python training/train.py --backend azure \
  --azure-subscription-id <ID> \
  --azure-resource-group <RG> \
  --azure-workspace-name <WS>

# HuggingFace + W&B
python training/train.py --backend huggingface --wandb --epochs 100
```

**📖 完整文档**: [训练后端使用指南](docs/docs/TRAINING_BACKENDS.md)

**传统分布式训练**（单机多卡）：
```bash
bash scripts/launch_distributed.sh \
  --num-gpus 4 \
  --batch-size 32 \
  --data-path ./data
```

### 模型压缩

使用DBC训练加速：
```python
from apt_model.plugins.compression_plugin import CompressionPlugin

plugin = CompressionPlugin()

# 启用DBC加速（20-30%速度提升）
model, optimizer = plugin.enable_dbc_training(
    model=model,
    rank_ratio=0.5,
    apply_to_gradients=True
)

# 正常训练即可享受加速
trainer.train(model, optimizer)
```

5种压缩方法可选：
- Pruning（剪枝）
- Quantization（量化）
- Knowledge Distillation（知识蒸馏）
- Low-Rank Decomposition（低秩分解）
- DBC Training Acceleration（DBC训练加速）⭐

---

## 项目结构

```
APT-Transformer/
├── apt_model/              # 核心代码包
│   ├── config/             # 配置文件和设置管理
│   ├── modeling/           # 模型定义（APT、Multimodal、KG）
│   ├── training/           # 训练器、优化器、监控
│   ├── generation/         # 文本生成和评估
│   ├── plugins/            # 插件系统（30+插件）
│   ├── rl/                 # 强化学习（RLHF/DPO/GRPO）
│   ├── pretraining/        # 自监督预训练（对比学习/MLM）
│   ├── core/               # 核心模块
│   │   ├── graph_rag/      # GraphRAG知识图谱
│   │   ├── training/       # SOSA训练监控
│   │   └── api_providers.py # 统一API接口
│   ├── api/                # REST API服务
│   ├── webui/              # Gradio Web界面
│   ├── cli/                # 命令行工具
│   └── utils/              # 工具函数
├── tests/                  # 单元测试和集成测试（20+测试）
├── scripts/                # 工具脚本
│   ├── launchers/          # GUI启动器
│   └── archived/           # 归档文件
├── examples/               # 使用示例（7+示例）
│   ├── rl_examples/        # 强化学习示例
│   ├── pretraining_examples/ # 预训练示例
│   ├── graph_rag_examples/ # 知识图谱示例
│   └── training_monitor_examples/ # 训练监控示例
├── docs/                   # 完整文档（15+文档）
├── requirements.txt        # 依赖列表
└── Makefile               # 构建工具
```

---

## 文档

### 📖 文档中心
**[完整文档中心](docs/README.md)** - 所有文档的导航和索引

### 📚 核心文档

#### 入门必读
- **[APT Model 使用手册](docs/APT_MODEL_HANDBOOK.md)** - 完整的模型使用手册
- **[启动器使用指南](docs/LAUNCHER_README.md)** - GUI启动器使用说明
- **[微调指南](docs/FINE_TUNING_GUIDE.md)** - LoRA和全参数微调

#### 知识蒸馏与迁移学习
- **[知识蒸馏原理](docs/DISTILLATION_PRINCIPLE.md)** - 理论基础和损失函数设计
- **[Teacher API指南](docs/TEACHER_API_GUIDE.md)** - 使用大模型API做教师模型
- **[视觉蒸馏指南](docs/VISUAL_DISTILLATION_GUIDE.md)** - 多模态知识蒸馏
- **[API Provider统一接口](docs/API_PROVIDERS_GUIDE.md)** - OpenAI/Anthropic/SiliconFlow等

#### 强化学习与预训练
- **[RL与预训练完整指南](docs/RL_PRETRAINING_GUIDE.md)** - RLHF/DPO/GRPO/对比学习/MLM
- **[自监督学习能力检查](docs/SELF_SUPERVISED_RL_CHECK_REPORT.md)** - 现有能力分析

#### 知识图谱与RAG
- **[知识图谱使用指南](docs/KNOWLEDGE_GRAPH_GUIDE.md)** - GraphRAG集成和使用
- **[GraphRAG模块文档](apt_model/core/graph_rag/)** - Hodge-Laplacian光谱分析、Graph Brain

#### 训练优化
- **[Optuna超参数优化](docs/OPTUNA_GUIDE.md)** - 自动超参数搜索
- **[SOSA训练监控](apt_model/core/training/)** - 实时监控和异常检测

#### 架构与集成
- **[模块集成方案](docs/MODULE_INTEGRATION_PLAN.md)** - 插件架构和零侵入集成
- **[插件开发指南](apt_model/cli/PLUGIN_GUIDE.md)** - 自定义插件开发

### 🔧 API文档
- [API文档](http://localhost:8000/docs) （启动API服务后访问）

---

## 系统要求

### 最低要求
- Python 3.8+
- 4GB RAM
- 2GB 磁盘空间

### 推荐配置
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU（用于加速训练）
- 10GB+ 磁盘空间

### 依赖
#### 核心依赖
- PyTorch 1.10+
- NumPy
- tqdm

#### 可选依赖
```bash
# Web服务
pip install gradio fastapi uvicorn

# 分布式训练
pip install torch.distributed

# NLP工具
pip install transformers scikit-learn

# 可视化
pip install tensorboard matplotlib
```

**离线支持**：项目支持完全离线运行，会自动降级到本地资源。

---

## 常用命令

### 训练相关
```bash
# 基础训练
python -m apt_model train

# 指定参数训练
python -m apt_model train --epochs 20 --batch-size 8

# 分布式训练
bash scripts/launch_distributed.sh --num-gpus 4
```

### 服务相关
```bash
# 启动WebUI
python -m apt_model.webui.app --checkpoint-dir ./checkpoints

# 启动API
python -m apt_model.api.server --checkpoint-dir ./checkpoints

# 交互式对话
python -m apt_model chat
```

### 工具命令
```bash
# Debug诊断
python -m apt_model debug

# 配置管理
python -m apt_model config

# 查看帮助
python -m apt_model --help
```

---

## 测试

运行测试套件：
```bash
# 运行所有测试
pytest tests/

# 快速smoke test
pytest tests/test_smoke.py -v

# 测试特定模块
pytest tests/test_compression_plugin.py -v
```

---

## 性能

### 训练速度
- **DBC加速**: 20-30%训练速度提升
- **混合精度**: 支持FP16训练
- **梯度累积**: 支持大batch训练

### 推理速度
- **Beam Search**: 高质量生成
- **批量推理**: API支持批量处理
- **模型压缩**: 量化后推理加速

### 资源使用
- **内存优化**: 梯度checkpoint支持
- **离线运行**: 无需网络连接
- **依赖容错**: 可选依赖不影响核心功能

---

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 开发指南
- 遵循PEP 8代码规范
- 添加单元测试
- 更新相关文档

---

## 常见问题

### Q: 如何离线使用？
A: 项目支持完全离线运行，tokenizer会自动降级到内置中文词表。可选运行：
```bash
python scripts/download_optional_assets.py  # 提前下载资源
```

### Q: 训练时内存不足怎么办？
A:
- 减小batch size
- 启用梯度累积
- 使用混合精度训练
- 启用梯度checkpoint

### Q: API密钥在哪里？
A: 启动API服务时会在控制台显示自动生成的64字符密钥，或使用 `--api-key` 参数自定义。

### Q: 支持哪些语言？
A: 原生支持中文和英文，支持多语言混合训练，可自动检测语言。

---

## 更新日志

### v1.0.0 (2024)
- ✅ 完整的Transformer训练平台
- ✅ WebUI和REST API服务
- ✅ 26+插件生态系统
- ✅ 分布式训练支持
- ✅ 模型压缩和DBC加速
- ✅ Checkpoint原子性保护
- ✅ Debug模式和CLI工具

---

## 致谢

感谢所有贡献者和使用者！

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个Star！**

[问题反馈](https://github.com/chen0430tw/APT-Transformer/issues) • [功能建议](https://github.com/chen0430tw/APT-Transformer/issues)

</div>
