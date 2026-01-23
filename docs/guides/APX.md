# APX - APT Package Exchange Format

## 📦 概述

**APX (APT Package Exchange)** 是APT-Transformer项目的标准化模型打包格式。它允许你将任何HuggingFace、LLaMA、DeepSeek等主流框架的模型打包成统一的`.apx`文件，实现：

- ✅ **跨框架兼容**：统一的接口访问不同框架模型
- ✅ **完整封装**：包含模型权重、配置、分词器、适配器代码
- ✅ **模块化设计**：支持插件式扩展和能力声明
- ✅ **轻量部署**：支持thin模式（仅打包元数据，不含权重）
- ✅ **标准化管理**：版本控制、依赖管理、能力检测

## 🏗️ APX包结构

一个标准的`.apx`文件是一个ZIP压缩包，包含以下结构：

```
my_model.apx (ZIP)
├── apx.yaml                    # APX清单文件（必需）
├── model/
│   └── adapters/
│       ├── hf_adapter.py       # HuggingFace适配器
│       └── tokenizer_adapter.py # 分词器适配器
├── artifacts/                  # 模型工件
│   ├── config.json            # 模型配置
│   ├── tokenizer.json         # 分词器文件
│   ├── tokenizer.model        # SentencePiece模型（可选）
│   ├── vocab.json             # 词汇表（可选）
│   ├── merges.txt             # BPE合并规则（可选）
│   └── model.safetensors      # 模型权重（full模式）
└── tests/                      # 测试文件（可选）
    └── smoke.py               # 冒烟测试
```

## 📄 apx.yaml 清单格式

```yaml
apx_version: 1                    # APX格式版本
name: my-awesome-model            # 模型名称
version: 1.0.0                    # 模型版本
type: model                       # 包类型

entrypoints:
  model_adapter: model/adapters/hf_adapter.py:HFAdapter
  tokenizer_adapter: model/adapters/tokenizer_adapter.py:HFTokenizerAdapter

artifacts:                        # 工件映射
  config: artifacts/config.json
  tokenizer: artifacts/tokenizer.json
  weights: artifacts/model.safetensors

capabilities:                     # 模型能力声明
  provides:
    - text-generation
    - multilingual
    - moe                         # Mixture of Experts（可选）
    - rag                         # Retrieval-Augmented Generation（可选）
  prefers:
    - builtin                     # 优先使用内建功能

compose:                          # 组合配置（可选）
  router: observe_only            # 路由器模式
  checkpoint_format: safetensors  # 检查点格式
```

## 🚀 快速开始

### 1. 安装依赖

APX转换器仅依赖Python标准库，无需额外安装。但如果要使用HuggingFace适配器，需要：

```bash
pip install transformers torch
```

### 2. 打包模型

#### 基础用法：打包HuggingFace模型

```bash
python scripts/apx_converter.py \
  --src /path/to/huggingface/model \
  --out my_model.apx \
  --name my-awesome-model \
  --version 1.0.0
```

#### Full模式（包含权重）

```bash
python scripts/apx_converter.py \
  --src ./bert-base-chinese \
  --out bert-base-chinese.apx \
  --name bert-base-chinese \
  --version 1.0.0 \
  --mode full
```

#### Thin模式（仅元数据）

适合已有模型文件，只需打包配置和适配器的场景：

```bash
python scripts/apx_converter.py \
  --src ./llama-7b \
  --out llama-7b-thin.apx \
  --name llama-7b \
  --version 1.0.0 \
  --mode thin
```

Thin模式会在`artifacts/`中生成占位文件，指向原始模型路径：

```json
{
  "__thin__": true,
  "source_weight": "/path/to/original/model.safetensors"
}
```

### 3. 高级选项

#### 指定权重和分词器文件

```bash
python scripts/apx_converter.py \
  --src ./deepseek-model \
  --out deepseek.apx \
  --name deepseek \
  --version 2.0.0 \
  --weights-glob "*.safetensors" \
  --tokenizer-glob "tokenizer*" \
  --config-file ./deepseek-model/config.json
```

#### 添加能力声明

```bash
python scripts/apx_converter.py \
  --src ./moe-model \
  --out moe-model.apx \
  --name moe-model \
  --version 1.0.0 \
  --capability text-generation \
  --capability moe \
  --capability multilingual
```

#### 添加Compose配置

```bash
python scripts/apx_converter.py \
  --src ./model \
  --out model.apx \
  --name my-model \
  --version 1.0.0 \
  --compose router=observe_only \
  --compose checkpoint_format=safetensors
```

#### 添加冒烟测试

```bash
python scripts/apx_converter.py \
  --src ./model \
  --out model.apx \
  --name my-model \
  --version 1.0.0 \
  --add-test
```

## 🔧 CLI命令

APT-Transformer提供了完整的APX命令行工具：

### 打包模型

```bash
# 使用CLI命令（推荐）
python -m apt_model pack-apx \
  --src /path/to/model \
  --out model.apx \
  --name my-model \
  --version 1.0.0
```

### 检测模型框架

```bash
python -m apt_model detect-framework --src /path/to/model
```

输出示例：
```
[info] Detected framework: huggingface
```

### 自动检测模型能力

```bash
python -m apt_model detect-capabilities --src /path/to/model
```

输出示例：
```
[info] Detected capabilities:
  - text-generation
  - multilingual
  - moe
```

### 查看APX包信息

```bash
python -m apt_model apx-info --apx model.apx
```

输出示例：
```
[APX Manifest]
apx_version: 1
name: my-model
version: 1.0.0
type: model
entrypoints:
  model_adapter: model/adapters/hf_adapter.py:HFAdapter
  tokenizer_adapter: model/adapters/tokenizer_adapter.py:HFTokenizerAdapter
...
```

## 🎯 适配器系统

### HuggingFace适配器

APX内置的HuggingFace适配器提供标准接口：

```python
from apt_model.tools.apx import load_apx

# 加载APX包
model_adapter = load_apx("my_model.apx")

# 生成文本
texts = ["Hello, how are you?", "What is AI?"]
outputs = model_adapter.generate(texts, max_new_tokens=64)

for text, output in zip(texts, outputs):
    print(f"输入: {text}")
    print(f"输出: {output}\n")
```

### 自定义适配器

你也可以创建自定义适配器来支持特殊模型：

```python
# custom_adapter.py
class CustomAdapter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_artifacts(cls, artifacts_dir: str):
        # 从artifacts/加载模型和分词器
        model = load_custom_model(artifacts_dir)
        tokenizer = load_custom_tokenizer(artifacts_dir)
        return cls(model, tokenizer)

    def generate(self, texts, max_new_tokens=64):
        # 实现生成逻辑
        ...
```

然后使用`--adapter stub`参数并手动替换适配器文件。

## 📊 能力检测系统

APX支持自动检测模型的以下能力：

| 能力标识 | 说明 | 检测依据 |
|---------|------|---------|
| `moe` | Mixture of Experts | 配置中有`num_experts`或`moe`关键词 |
| `rag` | 检索增强生成 | 存在`retriever`、`knowledge_base`等关键词 |
| `rlhf` | 人类反馈强化学习 | 存在`reward_model`、`rlhf`等关键词 |
| `multimodal` | 多模态（视觉+文本） | 存在`vision`、`image_processor`等 |
| `multilingual` | 多语言 | 词汇表大小>50000或配置标注multilingual |
| `code-generation` | 代码生成 | 模型名称包含`code`、`codex`等 |
| `long-context` | 长上下文 | `max_position_embeddings` > 4096 |

### 能力检测示例

```python
from apt_model.tools.apx import detect_capabilities
from pathlib import Path

capabilities = detect_capabilities(Path("/path/to/model"))
print("Detected capabilities:", capabilities)
# 输出: ['text-generation', 'multilingual', 'long-context']
```

## 🔍 框架检测

APX能自动识别模型来自哪个框架：

| 框架类型 | 识别标志 |
|---------|---------|
| `huggingface` | 存在`config.json`且包含`architectures`或`model_type` |
| `structured` | 存在`params.json`、`lit_config.json`、`config.yml`等 |
| `unknown` | 无法识别 |

```python
from apt_model.tools.apx import detect_framework
from pathlib import Path

framework = detect_framework(Path("/path/to/model"))
print(f"Framework: {framework}")
# 输出: huggingface
```

## 📦 完整使用示例

### 示例1：打包并加载BERT模型

```bash
# 1. 打包BERT
python scripts/apx_converter.py \
  --src ./bert-base-chinese \
  --out bert.apx \
  --name bert-base-chinese \
  --version 1.0.0 \
  --capability text-classification \
  --capability multilingual \
  --add-test

# 2. 查看包信息
python -m apt_model apx-info --apx bert.apx

# 3. 使用模型
python -c "
from apt_model.tools.apx import load_apx

model = load_apx('bert.apx')
texts = ['我爱中国', '人工智能很强大']
outputs = model.generate(texts, max_new_tokens=20)
print(outputs)
"
```

### 示例2：打包LLaMA模型（Thin模式）

```bash
# 1. 仅打包元数据（模型权重保留在原位置）
python scripts/apx_converter.py \
  --src /mnt/models/llama-7b \
  --out llama-7b-thin.apx \
  --name llama-7b \
  --version 1.0.0 \
  --mode thin \
  --capability text-generation \
  --capability long-context

# 2. 部署时直接使用（模型从原路径加载）
```

### 示例3：打包自定义模型

```bash
# 1. 打包自定义结构模型
python scripts/apx_converter.py \
  --src ./my_custom_model \
  --out custom.apx \
  --name my-custom-model \
  --version 2.0.0 \
  --config-file ./my_custom_model/model_config.json \
  --weights-glob "*.pth" \
  --tokenizer-glob "tokenizer/*" \
  --adapter stub \
  --capability custom-task

# 2. 手动修改适配器（解压APX，编辑adapters/，重新打包）
```

## 🛠️ 高级功能

### 1. 多模型组合（Compose）

通过`compose`配置实现模型组合和路由：

```yaml
compose:
  router: observe_only          # 仅观察，不干预
  ensemble_strategy: voting     # 投票策略
  checkpoint_format: safetensors
```

### 2. 插件系统集成

APX支持与APT插件系统集成：

```yaml
capabilities:
  provides:
    - custom-capability
  prefers:
    - plugin                    # 优先使用插件实现
```

### 3. 版本管理

APX包支持语义化版本控制：

```bash
# 打包不同版本
python scripts/apx_converter.py --src ./model --out model-v1.0.0.apx --name model --version 1.0.0
python scripts/apx_converter.py --src ./model --out model-v1.1.0.apx --name model --version 1.1.0
python scripts/apx_converter.py --src ./model --out model-v2.0.0.apx --name model --version 2.0.0
```

## 📐 命令行参数完整列表

### apx_converter.py 参数

```bash
--src PATH                # 源模型目录（必需）
--out PATH                # 输出.apx文件路径（必需）
--name NAME               # APX包名称（必需）
--version VERSION         # APX包版本（必需）
--adapter {hf,stub}       # 适配器类型（默认：hf）
--mode {full,thin}        # 打包模式（默认：full）
--weights-glob PATTERN    # 权重文件glob模式（可选）
--tokenizer-glob PATTERN  # 分词器文件glob模式（可选）
--config-file PATH        # 显式指定config.json（可选）
--prefers {builtin,plugin} # 优先级（默认：builtin）
--capability CAP          # 能力声明（可多次指定）
--compose KEY=VALUE       # Compose配置（可多次指定）
--thin                    # 等价于--mode thin
--add-test                # 添加冒烟测试
```

### CLI命令参数

```bash
# pack-apx命令
python -m apt_model pack-apx \
  --src PATH              # 源模型目录
  --out PATH              # 输出.apx路径
  --name NAME             # 模型名称
  --version VERSION       # 模型版本
  [其他参数同apx_converter.py]

# detect-capabilities命令
python -m apt_model detect-capabilities \
  --src PATH              # 源模型目录

# detect-framework命令
python -m apt_model detect-framework \
  --src PATH              # 源模型目录

# apx-info命令
python -m apt_model apx-info \
  --apx PATH              # APX文件路径
```

## 🎭 使用场景

### 1. 模型分发
打包模型为APX格式，方便分享和部署。

### 2. 版本控制
对同一模型的不同版本进行标准化管理。

### 3. 跨框架迁移
统一接口访问不同框架的模型。

### 4. 轻量部署
使用thin模式在多个环境中共享同一模型文件。

### 5. 能力声明
通过标准化元数据声明模型能力，便于自动化选择和路由。

## 🔧 技术细节

### 工件文件候选列表

APX转换器会自动搜索以下文件：

**分词器文件**：
- `tokenizer.json`
- `tokenizer.model`
- `sentencepiece.bpe.model`
- `sp.model`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`

**权重文件**（默认glob）：
- `*.safetensors`
- `pytorch_model*.bin`
- `consolidated*.pth`

### 适配器接口规范

自定义适配器必须实现以下接口：

```python
class CustomAdapter:
    @classmethod
    def from_artifacts(cls, artifacts_dir: str):
        """从artifacts目录加载模型"""
        ...

    def encode(self, texts, max_new_tokens=0):
        """编码文本"""
        ...

    def generate(self, texts, max_new_tokens=64):
        """生成文本"""
        ...

    def forward(self, batch):
        """前向传播"""
        ...

    def save_pretrained(self, out_dir: str):
        """保存模型"""
        ...
```

## 🆘 常见问题

### Q1: 打包后APX文件过大？
**A**: 使用`--mode thin`仅打包元数据，权重保留在原位置。

### Q2: 如何打包非HuggingFace模型？
**A**: 使用`--adapter stub`，然后手动编辑适配器代码。

### Q3: 能力检测不准确？
**A**: 使用`--capability`参数手动指定能力：
```bash
--capability moe --capability rag --capability multilingual
```

### Q4: APX包如何解压查看？
**A**: APX本质是ZIP文件，可以直接解压：
```bash
unzip model.apx -d model_extracted/
```

### Q5: 如何在APT项目中使用APX包？
**A**: 使用`load_apx()`函数：
```python
from apt_model.tools.apx import load_apx
model = load_apx("model.apx")
```

## 📁 相关文件位置

- **转换器脚本**: `scripts/apx_converter.py`
- **CLI命令**: `apt_model/cli/apx_commands.py`
- **APX加载器**: `apt_model/console/apx_loader.py`
- **工具模块**: `apt_model/tools/apx.py`（如果存在）

## 📚 参考资料

- **模型适配器开发**: 查看`apt_model/modeling/`目录下的适配器示例
- **插件系统**: 参考`apt_model/plugins/README.md`
- **检查点管理**: `apt_model/training/checkpoint.py`

## 🤝 贡献指南

如果你想为APX格式添加新功能：

1. 扩展`apx.yaml`格式定义
2. 更新`scripts/apx_converter.py`中的打包逻辑
3. 添加相应的能力检测规则
4. 更新本文档

---

**贡献者**: APT-Transformer团队
**最后更新**: 2025-12-04
**许可**: 与APT-Transformer项目相同
