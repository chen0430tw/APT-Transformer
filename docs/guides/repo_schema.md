# APT-Transformer Repository Schema (APT 2.0)

## Project Overview
APT-Transformer (Autopoietic Transformer) is a production-ready PyTorch Transformer training platform featuring **APT 2.0 architecture** with Domain-Driven Design (DDD), configuration-driven YAML profiles, Virtual Blackwell GPU virtualization, and comprehensive Chinese and English support.

**APT 2.0 Release**: Complete architectural refactoring with clear domain separation (Model/TrainOps/vGPU/APX), backward compatibility layer (6-month migration period until 2026-07-22), and YAML-driven configuration system.

---

## APT 2.0 Architecture

### Design Principles
- **Domain Driven Design (DDD)** - Separate by business domain (Model/TrainOps/vGPU/APX), not technical layers
- **Single Responsibility Principle (SRP)** - Each domain has a single, clear responsibility
- **Configuration Over Code** - YAML profiles replace code duplication
- **Backward Compatibility** - Complete compat layer for smooth migration

### Four Core Domains

```
apt/
├── model/          # 【Model Domain】What - Model definitions
├── trainops/       # 【TrainOps Domain】How - Training operations
├── vgpu/           # 【vGPU Domain】Where - GPU virtualization
├── apx/            # 【APX Domain】Package - Model packaging
├── compat/         # Backward compatibility layer (6-month migration)
└── core/           # Cross-domain infrastructure
    └── config/     # Profile configuration system
```

---

## apt/model/ - Model Domain (What)

**Purpose**: Define WHAT the model is - architectures, layers, tokenization, extensions

### apt/model/architectures/
- **apt_model.py** - APT核心模型: Autopoietic Attention(自生成注意力) + DBC-DAC压缩优化
- **multimodal_model.py** - 多模态APT: 文本/图像/音频三模态融合(cross_attention/tri_modal/gated)
- **claude4_model.py** - Claude-4: GPT-4o + Graph Reflection Layer(图连通度分析/最短路径推理)
- **gpt4o_model.py** - GPT-4o架构: Dynamic Tau + Vein Subspace + Hybrid FFN + Tri-Vein Attention
- **gpt5_model.py** - GPT-5: Codebook MoE(top-k专家) + Leaf-Vote + Streaming Retrieval + Bi-state Precision
- **gpto3_model.py** - GPT-o3推理模型: 强化推理能力
- **vft_tva_model.py** - VFT-TVA: Vein-Flow Transformer + Tri-Vein Attention(低秩注意力子空间)
- **elastic_transformer.py** - 弹性Transformer: 动态深度和宽度调整

### apt/model/layers/
- **embeddings.py** - 嵌入层: 位置编码(PositionalEncoding)和token嵌入(TokenEmbedding/ImageEmbedding)
- **advanced_rope.py** - 高级RoPE: 旋转位置编码的增强实现
- **apt_control.py** - APT控制层: Autopoietic控制机制
- **left_spin_smooth.py** - Left Spin Smooth: 左旋平滑注意力机制
- **memory_augmented_smooth.py** - 记忆增强平滑: 记忆增强的平滑注意力
- **moe_optimized.py** - 优化的MoE: 混合专家层实现
- **utils.py** - 层工具函数
- **blocks/** - 构建块: Vein、VFT-TVA等可复用模块
- **encoders/** - 编码器: vision_encoder、audio_encoder、cross_modal_attention

### apt/model/tokenization/
- **chinese_tokenizer.py** - 中文分词器: 支持字符级(char)和词级(word)分词，Unicode汉字范围0x4e00-0x9fff
- **chinese_tokenizer_integration.py** - 中文分词集成: 与主模型的集成适配器

### apt/model/extensions/
- **rag_integration.py** - RAG集成: 检索增强生成(FAISS/Annoy/exact cosine)
- **kg_rag_integration.py** - KG+RAG联合: 知识图谱增强的检索生成
- **knowledge_graph.py** - 轻量级知识图谱: 基于三元组(head-relation-tail)的实体关系存储与多跳推理
- **mcp_integration.py** - MCP协议集成: Model Context Protocol异步流式检索 + RAG + GraphRAG + 证据融合

---

## apt/trainops/ - Training Operations Domain (How)

**Purpose**: Define HOW to train - engines, data loading, checkpoints, evaluation, distributed

### apt/trainops/engine/
- **trainer.py** - 主训练器: 支持中文分词/自动语言检测/混合精度/原子检查点
- **finetuner.py** - 微调工具: LoRA低秩适配/全量微调/参数高效微调
- **claude_trainer.py** - Claude专用训练器: 针对Claude模型的训练优化
- **gpt_trainer.py** - GPT专用训练器: 针对GPT系列模型的训练流程
- **vft_tva_trainer.py** - VFT-TVA训练流程: Vein-Flow Transformer训练管线
- **train_reasoning.py** - 推理训练: 专注于推理能力的训练策略
- **callbacks.py** - 训练回调: 自定义hooks在训练各阶段执行(on_epoch_end/on_batch_end等)
- **hooks.py** - 训练钩子: 可插拔的训练阶段钩子函数
- **training_events.py** - 事件驱动训练: 发布-订阅模式的训练事件系统
- **mixed_precision.py** - 混合精度: FP16/BF16自动混合精度训练加速
- **optimizer.py** - 自定义优化器: 学习率调度器/优化器包装
- **sosa_core.py** - SOSA核心: Self-Optimizing System Architecture自优化系统
- **apt_integration.py** - APT训练集成: 连接APT模型与SOSA监控系统

### apt/trainops/data/
- **data_loading.py** - 数据加载: DataLoader包装和数据预处理管道

### apt/trainops/checkpoints/
- **checkpoint.py** - 原子检查点: 防损坏的两阶段保存(tmp→final)确保训练安全

### apt/trainops/eval/
- **training_monitor.py** - 实时训练监控: 损失/指标/资源使用的WebSocket实时推送
- **training_guard.py** - 训练守卫: NaN检测/早停(EarlyStopping)/异常恢复机制
- **gradient_monitor.py** - 梯度监控: 梯度流分析/梯度消失爆炸检测/可视化

### apt/trainops/distributed/
- **extreme_scale_training.py** - 极限规模训练: 支持100K+ GPU训练，3D并行(数据/张量/流水线)，DeepSpeed ZeRO，Megatron-LM，GB200 NVL72

---

## apt/vgpu/ - Virtual GPU Domain (Where)

**Purpose**: Define WHERE to run - GPU virtualization, resource scheduling, Virtual Blackwell

### apt/vgpu/runtime/
- **vgpu_stack.py** - VGPU栈: 多层GPU虚拟化(L0:硬件, L1:虚拟化, L2:优化, L3:应用)
- **virtual_blackwell_adapter.py** - Virtual Blackwell适配器: GPU抽象和虚拟化接口
- **vb_global.py** - VB全局配置: Virtual Blackwell全局配置和启用管理
- **vb_integration.py** - VB集成: PyTorch深度集成，VBOptimizedLinear、VBModelWrapper
- **vb_autopatch.py** - VB自动补丁: 自动为模型添加Virtual Blackwell优化

### apt/vgpu/scheduler/
- **vgpu_estimator.py** - VGPU资源估算: 智能GPU资源估算，quick_estimate函数

---

## apt/apx/ - Model Packaging Domain (Package)

**Purpose**: Define PACKAGE - how to package, distribute, and validate models

### apt/apx/packaging/
- **APX格式**: APT模型打包格式(模型/配置/tokenizer单一文件)

### apt/apx/distribution/
- **模型分发**: Ollama导出、HuggingFace Hub上传

### apt/apx/validation/
- **模型验证**: 格式检测、权重验证

---

## apt/compat/ - Backward Compatibility Layer

**Purpose**: Provide 6-month migration period (until 2026-07-22) with deprecation warnings

### apt/compat/apt_model/
- **modeling/__init__.py** - Re-exports from `apt.model` with DeprecationWarning
- **training/__init__.py** - Re-exports from `apt.trainops` with DeprecationWarning

**Migration Example**:
```python
# OLD (deprecated, but still works)
from apt.apt_model.modeling import APTLargeModel

# NEW (recommended)
from apt.model.architectures import APTLargeModel
```

---

## apt/core/ - Cross-Domain Infrastructure

**Purpose**: Shared infrastructure used by all domains

### apt/core/config/
- **profile_loader.py** - Profile配置加载器: ProfileLoader类，load_profile()函数
  - Dataclasses: APTProfile, ModelConfig, TrainingConfig, VGPUConfig, etc.
  - Auto-discovery of profiles/ directory
  - Type-safe configuration objects

### apt/core/data/
- **data_processor.py** - 数据预处理: 文本清洗/分词/去重/格式转换
- **external_data.py** - 外部数据加载器: 支持CSV/JSON/TXT等多种格式
- **huggingface_loader.py** - HuggingFace集成: 自动加载HF datasets和模型
- **multimodal_dataset.py** - 多模态数据集: 文本+图像+音频的联合数据加载
- **pipeline.py** - 数据流水线: ETL流程编排(Extract/Transform/Load)
- **hlbd/** - HLBD系统: Hierarchical Language Bootstrap Data分层语言启蒙数据

### apt/core/graph_rag/
- **graph_rag_manager.py** - GraphRAG管理器: 图检索增强生成的编排与调度
- **generalized_graph.py** - 泛图结构: Hypergraph(超图)/Simplicial Complex(单纯复形)/Pan-graph元结构
- **hodge_laplacian.py** - Hodge-Laplacian谱分析: 高阶拓扑信号处理(边流/面流/谱分解)
- **graph_brain.py** - 图脑引擎: 基于非平衡态统计物理的动态认知拓扑系统(自由能F=U-T·S/拓扑相变)

### apt/core/generation/
- **generator.py** - 文本生成引擎: 支持Greedy/Beam Search/Top-k/Top-p/Temperature采样
- **evaluator.py** - 文本质量评估: BLEU/ROUGE/困惑度/多样性等指标计算

### apt/core/runtime/
- **decoder/structured_reasoner.py** - 结构化推理器
- **decoder/halting.py** - 自适应停机
- **decoder/routing.py** - 路由控制
- **decoder/reasoning_controller.py** - 推理控制器

### apt/core/pretraining/
- **contrastive_pretrain.py** - 对比学习预训练: SimCLR/MoCo对比学习框架实现
- **mlm_pretrain.py** - 掩码语言模型: BERT/RoBERTa风格的MLM预训练

### apt/core/infrastructure/
- **logging.py** - 基础设施日志: 底层日志框架和格式化器
- **errors.py** - 错误定义: 自定义异常类和错误码常量

### apt/core/providers/
- **attention.py** - 注意力提供者
- **ffn.py** - FFN提供者
- **retrieval.py** - 检索提供者
- **kg_retrieval_provider.py** - 知识图谱检索提供者

### apt/core/
- **api_providers.py** - 统一API接口: OpenAI/Anthropic/DeepSeek/Gemini/Cohere等多家LLM提供商
- **hardware.py** - 硬件检测与优化: 自动检测CUDA/MPS/CPU并优化配置
- **resources.py** - 资源管理: 内存/GPU资源分配与监控
- **system.py** - 系统工具: 平台检测(Linux/macOS/Windows)和环境信息
- **config.py** - 配置管理
- **registry.py** - 组件注册表
- **schedules.py** - 调度工具

---

## apt/apps/ - Application Layer

**Purpose**: User-facing applications built on top of domains

### apt/apps/webui/
- **app.py** - Gradio WebUI: 4标签页(训练监控/梯度监控/检查点管理/推理测试)实时交互界面

### apt/apps/api/
- **server.py** - FastAPI服务器: 10+端点REST API(生成/训练/压缩/评估)支持异步请求

### apt/apps/cli/
- **commands.py** - CLI命令实现: 26+命令(train/eval/compress/export等)
  * 核心训练命令: train, fine-tune, train-hf, train-reasoning
  * 评估命令: eval, test, compare
  * 数据命令: process-data, clean-cache
  * 工具命令: info, list, size, prune, backup
  * 分发命令: upload, export-ollama
  * 高级命令: distill, compress
- **parser.py** - 参数解析: argparse包装，支持子命令和配置文件
- **command_registry.py** - 命令注册表: 动态注册和发现CLI命令
- **apx_commands.py** - APX格式命令: APX模型格式转换相关CLI
- **profile_loader.py** - Profile加载CLI工具

### apt/apps/console/
- **core.py** - 控制台核心: 插件化命令行系统主控制器
- **plugin_bus.py** - 插件总线: 事件驱动的发布-订阅消息总线
- **plugin_loader.py** - 插件加载器: 动态发现和加载.py插件文件
- **plugin_registry.py** - 插件注册表: 全局插件实例管理和查询
- **module_manager.py** - 模块管理器: Python模块的导入和依赖管理
- **eqi_manager.py** - EQI管理器: Emergent Quality Index涌现质量指数评估

### apt/apps/agent/
- **agent_loop.py** - Agent主循环
- **tool_system.py** - 工具系统
- **tools/web_search.py** - Web搜索工具

### apt/apps/evaluation/
- **model_evaluator.py** - 模型评估器: 综合评估BLEU/ROUGE/困惑度/F1等20+指标
- **comparison.py** - 模型对比: 多模型性能对比报告生成
- **unified.py** - 统一评估接口

### apt/apps/rl/
- **rlhf_trainer.py** - RLHF训练器: 基于PPO的人类反馈强化学习
- **dpo_trainer.py** - DPO训练器: Direct Preference Optimization直接偏好优化
- **grpo_trainer.py** - GRPO训练器: Group Relative Policy Optimization群组相对策略优化
- **reward_model.py** - 奖励模型: 从人类偏好数据训练奖励打分器

### apt/apps/interactive/
- **chat.py** - 交互式对话: 命令行实时对话界面
- **admin_mode.py** - 管理员模式: 高级调试和系统管理控制台

### apt/apps/plugins/
**30+ Production Plugins** organized by category:
- **core/** - compression_plugin, training_monitor_plugin, version_manager
- **deployment/** - microvm_compression_plugin, vgpu_stack_plugin
- **distillation/** - teacher_api, visual_distillation_plugin
- **evaluation/** - model_comparison_plugin, model_evaluator_plugin
- **experimental/** - plugin_6_multimodal_training, plugin_7_data_processors, plugin_8_advanced_debugging
- **hardware/** - cloud_npu_adapter_plugin, npu_backend_plugin, virtual_blackwell_plugin
- **integration/** - graph_rag_plugin, ollama_export_plugin, web_search_plugin
- **memory/** - aim_memory_plugin
- **monitoring/** - gradient_monitor_plugin, resource_monitor_plugin
- **optimization/** - mxfp4_quantization_plugin
- **protocol/** - mcp_integration_plugin
- **retrieval/** - kg_rag_integration_plugin, rag_integration_plugin
- **rl/** - dpo_trainer_plugin, grpo_trainer_plugin, reward_model_plugin, rlhf_trainer_plugin
- **visualization/** - model_visualization_plugin

### apt/apps/tools/
- **apx/** - APX转换器、适配器、检测器、模板
- **apg/** - APG打包器

---

## apt/memory/ - Memory Systems

**Purpose**: Advanced memory and knowledge systems

### apt/memory/aim/
- **aim_memory.py** - AIM记忆系统
- **aim_memory_nc.py** - AIM Memory NC
- **hierarchical_memory.py** - 分层记忆系统
- **context_composer.py** - 上下文组合器

### apt/memory/graph_rag/
- GraphRAG知识图谱系统（与core/graph_rag共享）

---

## apt/perf/ - Performance Optimization

**Purpose**: Performance optimization components

### apt/perf/optimization/
- **gpu_flash_optimization.py** - GPU Flash优化
- **cloud_npu_adapter.py** - 云端NPU适配器
- **npu_backend.py** - NPU后端
- **mxfp4_quantization.py** - MXFP4量化
- **microvm_compression.py** - MicroVM压缩

**Note**: Virtual Blackwell已迁移至 `apt/vgpu/`，此处保留向后兼容的re-export

---

## profiles/ - Configuration Profiles

**Purpose**: YAML-driven configuration system (Configuration Over Code)

### Available Profiles
- **lite.yaml** - 轻量配置: 1x 8GB GPU，小模型(768 hidden size, 12 layers)，本地开发
- **standard.yaml** - 标准配置: 4x 24GB GPU，中模型(1024 hidden size, 24 layers)，生产环境
- **pro.yaml** - 专业配置: 16x 80GB GPU，大模型(2048 hidden size, 32 layers)，MoE启用
- **full.yaml** - 完整配置: 64x 80GB GPU，超大模型(4096 hidden size, 48 layers)，所有扩展启用

### Profile Structure
每个profile包含：
- **profile**: 元数据(name, description, author, created)
- **model**: 模型配置(architecture, hidden_size, num_layers, etc.)
- **training**: 训练配置(batch_size, learning_rate, mixed_precision, etc.)
- **vgpu**: VGPU配置(enabled, virtual_gpus, optimization_level, etc.)
- **extensions**: 扩展配置(rag, knowledge_graph, mcp, etc.)
- **monitoring**: 监控配置(enabled, realtime, metrics, etc.)
- **checkpoints**: 检查点配置(enabled, interval, keep_last_n, etc.)

### Usage
```python
from apt.core.config import load_profile

config = load_profile('standard')
print(f"Model: {config.model.architecture}")
print(f"Batch size: {config.training.batch_size}")
print(f"VGPU enabled: {config.vgpu.enabled}")
```

---

## archived/apt_model/ - Archived Old Code

**Purpose**: Archived legacy code from pre-APT 2.0 architecture

**Status**: ✅ Fully migrated and archived

**Contents**:
- 原有的 apt_model/ 目录完整内容（62个文件）
- 包含旧的 modeling/、training/、optimization/ 等
- 仅用于参考，不再主动维护

**Note**: 所有功能已迁移至 APT 2.0 架构，通过 `apt/compat/` 提供向后兼容支持

---

## Documentation (/docs)

### Main Documentation
- **README.md** - 文档中心索引
- **ARCHITECTURE_2.0.md** - ⭐ APT 2.0完整架构文档（必读）
- **APT_MODEL_HANDBOOK.md** - APT模型完整使用手册
- **INSTALLATION.md** - 安装指南

### Documentation by Category
- **guides/** - 架构指南、集成总结、APX规范、repo_schema
- **hlbd/** - HLBD数据集训练文档集（7个文档）
- **kernel/** - 核心功能指南（10个文档：微调、数据预处理、调试等）
- **memory/** - 记忆系统指南（6个文档：AIM、GraphRAG、知识图谱等）
- **performance/** - 性能优化指南（9个文档：Virtual Blackwell、NPU、GPU优化等）
- **product/** - 产品功能指南（13个文档：Agent、API、Claude4、蒸馏、插件等）
- **testing/** - 测试工具使用指南

---

## Examples (/examples)

### Quick Start Examples
- **use_profiles.py** - ⭐ Profile配置系统使用示例（5个完整示例）
- **demo_startup.py** - 启动演示
- **core_registry.py** - 核心注册表使用

### Model Examples
- **claude4_demo.py** - Claude 4集成演示
- **gpt5_mcp_demo.py** - GPT-5 MCP演示
- **multimodal_inference.py** - 多模态推理

### Training Examples
- **train_distributed.py** - 分布式训练
- **train_multimodal.py** - 多模态训练
- **multilingual_example.py** - 多语言使用
- **rl_examples/** - 强化学习示例（DPO、GRPO）
- **pretraining_examples/** - 预训练示例（对比学习、MLM）

### Integration Examples
- **graph_rag_examples/** - GraphRAG示例
- **visual_distillation_example.py** - 视觉蒸馏
- **agent_demo.py** - Agent演示

### Plugin & Profile Examples
- **plugin_template/** - 自定义插件模板
- **profiles/** - 配置profile示例

---

## Tests (/tests)

### Test Organization (Layered)
- **l0_kernel/** - 核心功能测试（11个测试）
- **l1_performance/** - 性能优化测试（5个测试）
- **l2_memory/** - 记忆系统测试
- **l3_product/** - 产品功能测试（5个测试）
- **integration/** - 集成测试（3个测试）

### Key Tests
- **test_core_imports.py** - 核心导入测试
- **test_trainer_complete.py** - 完整训练器测试
- **test_compression_plugin.py** - 压缩测试
- **test_multimodal.py** - 多模态测试
- **test_plugin_system.py** - 插件系统测试
- **test_virtual_blackwell.py** - Virtual Blackwell测试

---

## Scripts (/scripts)

### Training Scripts
- **launch_distributed.sh** - 多GPU/节点启动器
- **run_optuna_optimization.sh** - HPO脚本
- **run_best_training.sh** - 最佳配置训练
- **hlbd/** - HLBD训练启动脚本集

### Utility Scripts
- **apt_eqi_manager.py** - EQI管理器
- **apx_converter.py** - APX转换器
- **eqi.py** - EQI工具
- **launchers/** - GUI启动器应用
- **testing/** - 测试脚本（test_all_commands.py等）
- **setup/** - 安装配置脚本

---

## Tools (/tools)

### Data Generation
- **data_generation/** - HLBD数据集生成器、吉祥物渲染器

### Diagnostics
- **diagnostics/** - 后端检查器、问题诊断、分词器诊断、HLBD验证器

### Visualization
- **visualization/** - 训练可视化、演示可视化、多训练监控器

---

## Configuration Files

- **setup.py** - Package setup (version 2.0.0)
- **requirements.txt** - Core dependencies
- **requirements-dev.txt** - Development dependencies
- **requirements-minimal.txt** - Minimal installation
- **.env.example** - Environment variables template
- **Makefile** - Build automation
- **test_profiles.py** - Profile配置系统测试脚本

---

## 核心技术特性详解

### 1. APT 2.0 架构
- **领域驱动设计（DDD）** - Model（what）、TrainOps（how）、vGPU（where）、APX（package）四大领域
- **配置驱动** - YAML Profile系统，一键切换 lite/standard/pro/full 配置
- **Virtual Blackwell** - GPU虚拟化技术栈，支持100K+ GPU极限规模训练
- **向后兼容** - 6个月迁移期（至2026-07-22），完整compat层

### 2. 模型架构
- **APT (Autopoietic Transformer)** - 自生成注意力机制 + DBC-DAC压缩优化
- **VFT-TVA** - Vein-Flow Transformer低秩注意力子空间，复杂度从O(T²·d)降至O(T²·r)
- **Claude-4** - GPT-4o + 图反思层（图连通度分析/最短路径推理）
- **GPT-5** - Codebook MoE专家混合 + Leaf-Vote + 流式检索 + 双态精度对齐
- **多模态** - 文本/图像/音频三模态融合(cross_attention/tri_modal/gated)

### 3. Virtual Blackwell GPU虚拟化
- **VGPU Stack** - 多层虚拟化（L0:硬件, L1:虚拟化, L2:优化, L3:应用）
- **资源估算** - VGPUResourceEstimator智能估算GPU需求
- **极限规模** - extreme_scale_training支持100K+ GPU训练
- **3D并行** - 数据并行 + 张量并行 + 流水线并行
- **DeepSpeed ZeRO** - ZeRO-2/3优化，CPU卸载
- **GB200 NVL72** - NVIDIA最新GPU架构支持

### 4. 训练系统
- **分布式训练** - 多GPU/多节点DDP/FSDP，支持gradient checkpointing
- **混合精度** - FP16/BF16自动混合精度，accelerate库集成
- **原子检查点** - 两阶段保存(tmp→final)防止训练崩溃导致权重损坏
- **训练守卫** - NaN检测/早停/异常自动恢复/梯度裁剪
- **SOSA监控** - Self-Optimizing System Architecture实时WebSocket推送

### 5. 强化学习
- **RLHF** - 基于PPO的人类反馈强化学习
- **DPO** - Direct Preference Optimization直接偏好优化（无需奖励模型）
- **GRPO** - Group Relative Policy Optimization群组相对策略优化
- **奖励模型** - 从人类偏好对(chosen/rejected)训练打分器

### 6. 知识蒸馏
- **API教师** - 使用OpenAI/Anthropic等API作为教师模型
- **多模态蒸馏** - 视觉-语言跨模态知识迁移
- **温度缩放** - Temperature=4.0软标签蒸馏
- **损失组合** - α·KD_loss + β·CE_loss

### 7. 模型压缩
- **剪枝(Pruning)** - Magnitude/Structured剪枝移除不重要权重
- **量化(Quantization)** - 8-bit动态量化，支持QAT
- **知识蒸馏(KD)** - 大模型→小模型知识迁移
- **低秩分解** - 权重矩阵SVD分解降维
- **DBC** - Dimension-Balanced Compression维度平衡压缩

### 8. GraphRAG图检索
- **泛图结构** - Hypergraph超图/Simplicial Complex单纯复形/Pan-graph
- **Hodge-Laplacian** - 高阶拓扑信号处理（边流/面流/k-链谱分解）
- **图脑引擎** - 认知自由能最小化F=U-T·S，动态拓扑相变
- **多跳推理** - 知识图谱三元组(head-relation-tail)的路径推理
- **RAG融合** - FAISS/Annoy/exact cosine向量检索 + 图结构增强

### 9. 多语言支持
- **中文原生** - 字符级(char)/词级(word)分词，Unicode范围0x4e00-0x9fff
- **自动检测** - 基于字符分布自动识别中英日韩文本
- **混合训练** - 同一batch内中英文混合训练
- **多语言UI** - 支持中英日界面/日志/错误消息

### 10. 插件系统
- **30+生产插件** - 压缩/蒸馏/GraphRAG/Web搜索/Ollama导出等
- **事件驱动** - 发布-订阅消息总线，插件间松耦合通信
- **热插拔** - 运行时动态加载/卸载插件无需重启
- **版本管理** - 插件兼容性检查和语义版本控制
- **能力映射** - 功能需求自动路由到对应插件实现

---

## Migration Guide (1.x → 2.0)

### Import Path Changes
```python
# OLD (apt_model, deprecated)
from apt_model.modeling import APTLargeModel
from apt_model.training import Trainer

# NEW (APT 2.0)
from apt.model.architectures import APTLargeModel
from apt.trainops.engine import Trainer
```

### Configuration System
```python
# OLD (code-based)
config = APTConfig(
    d_model=1024,
    n_layers=24,
    batch_size=32
)

# NEW (YAML profile)
from apt.core.config import load_profile
config = load_profile('standard')
```

### Virtual Blackwell
```python
# OLD
from apt_model.optimization import VirtualBlackwellAdapter

# NEW
from apt.vgpu.runtime import VirtualBlackwellAdapter
```

**Migration Period**: 6 months (until 2026-07-22) with DeprecationWarning

---

## Platform Support
- OS: Linux, macOS, Windows
- Hardware: CPU, CUDA, MPS (Apple Silicon)
- Python: 3.8-3.12
- Offline-friendly design
- APT 2.0 Production-Ready: ✅
