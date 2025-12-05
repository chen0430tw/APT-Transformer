# APT-Transformer Repository Schema

## Project Overview
APT-Transformer (Autopoietic Transformer) is a production-ready PyTorch Transformer training platform with comprehensive Chinese and English support, featuring RL training, knowledge distillation, GraphRAG integration, and an extensible plugin system.

---

## apt_model/modeling
- apt_model.py              : APT核心模型 - Autopoietic Attention(自生成注意力) + DBC-DAC压缩优化
- multimodal_model.py       : 多模态APT模型 - 支持文本/图像/音频三模态融合(cross_attention/tri_modal/gated)
- claude4_model.py          : Claude-4模型 - GPT-4o + Graph Reflection Layer(图连通度分析/最短路径推理)
- gpt4o_model.py           : GPT-4o架构 - Dynamic Tau + Vein Subspace + Hybrid FFN + Tri-Vein Attention
- gpt5_model.py            : GPT-5模型 - Codebook MoE(top-k专家) + Leaf-Vote + Streaming Retrieval + Bi-state Precision
- gpto3_model.py           : GPT-o3推理模型 - 强化推理能力
- vft_tva_model.py         : VFT-TVA模型 - Vein-Flow Transformer + Tri-Vein Attention(低秩注意力子空间)
- knowledge_graph.py       : 轻量级知识图谱 - 基于三元组(head-relation-tail)的实体关系存储与多跳推理
- chinese_tokenizer.py     : 中文分词器 - 支持字符级(char)和词级(word)分词，Unicode汉字范围0x4e00-0x9fff
- mcp_integration.py       : MCP协议集成 - Model Context Protocol异步流式检索 + RAG + GraphRAG + 证据融合
- embeddings.py            : 嵌入层 - 位置编码和token嵌入
- rag_integration.py       : RAG集成 - 检索增强生成(FAISS/Annoy/exact cosine)
- kg_rag_integration.py    : KG+RAG联合 - 知识图谱增强的检索生成
- utils.py                 : 建模工具函数

## apt_model/core
- api_providers.py         : 统一API接口 - OpenAI/Anthropic/DeepSeek/Gemini/Cohere等多家LLM提供商
- hardware.py              : 硬件检测与优化 - 自动检测CUDA/MPS/CPU并优化配置
- resources.py             : 资源管理 - 内存/GPU资源分配与监控
- system.py                : 系统工具 - 平台检测(Linux/macOS/Windows)和环境信息

## apt_model/core/graph_rag
- graph_rag_manager.py     : GraphRAG管理器 - 图检索增强生成的编排与调度
- generalized_graph.py     : 泛图结构 - Hypergraph(超图)/Simplicial Complex(单纯复形)/Pan-graph元结构
- hodge_laplacian.py       : Hodge-Laplacian谱分析 - 高阶拓扑信号处理(边流/面流/谱分解)
- graph_brain.py           : 图脑引擎 - 基于非平衡态统计物理的动态认知拓扑系统(自由能F=U-T·S/拓扑相变)
- demo_full.py             : GraphRAG完整演示 - 端到端示例代码

## apt_model/core/training
- training_monitor.py      : 实时训练监控 - 损失/指标/资源使用的WebSocket实时推送
- sosa_core.py             : SOSA核心 - Self-Optimizing System Architecture自优化系统
- apt_integration.py       : APT训练集成 - 连接APT模型与SOSA监控系统

## apt_model/training
- trainer.py               : 主训练器 - 支持中文分词/自动语言检测/混合精度/原子检查点
- claude_trainer.py        : Claude专用训练器 - 针对Claude模型的训练优化
- gpt_trainer.py           : GPT专用训练器 - 针对GPT系列模型的训练流程
- vft_tva_trainer.py       : VFT-TVA训练流程 - Vein-Flow Transformer训练管线
- finetuner.py             : 微调工具 - LoRA低秩适配/全量微调/参数高效微调
- checkpoint.py            : 原子检查点 - 防损坏的两阶段保存(tmp→final)确保训练安全
- callbacks.py             : 训练回调 - 自定义hooks在训练各阶段执行(on_epoch_end/on_batch_end等)
- gradient_monitor.py      : 梯度监控 - 梯度流分析/梯度消失爆炸检测/可视化
- mixed_precision.py       : 混合精度 - FP16/BF16自动混合精度训练加速
- training_guard.py        : 训练守卫 - NaN检测/早停(EarlyStopping)/异常恢复机制
- optimizer.py             : 自定义优化器 - 学习率调度器/优化器包装
- training_events.py       : 事件驱动训练 - 发布-订阅模式的训练事件系统
- hooks.py                 : 训练钩子 - 可插拔的训练阶段钩子函数
- train_reasoning.py       : 推理训练 - 专注于推理能力的训练策略

## apt_model/rl
- rlhf_trainer.py          : RLHF训练器 - 基于PPO(Proximal Policy Optimization)的人类反馈强化学习
- dpo_trainer.py           : DPO训练器 - Direct Preference Optimization直接偏好优化(无需奖励模型)
- grpo_trainer.py          : GRPO训练器 - Group Relative Policy Optimization群组相对策略优化
- reward_model.py          : 奖励模型 - 从人类偏好数据训练奖励打分器

## apt_model/pretraining
- contrastive_pretrain.py  : 对比学习预训练 - SimCLR/MoCo对比学习框架实现
- mlm_pretrain.py          : 掩码语言模型 - BERT/RoBERTa风格的MLM预训练

## apt_model/generation
- generator.py             : 文本生成引擎 - 支持Greedy/Beam Search/Top-k/Top-p/Temperature采样
- evaluator.py             : 文本质量评估 - BLEU/ROUGE/困惑度/多样性等指标计算

## apt_model/plugins
- compression_plugin.py    : 模型压缩插件 - 5种方法: 剪枝(Pruning)/量化(8-bit)/蒸馏(KD)/低秩分解/DBC加速
- visual_distillation_plugin.py : 多模态蒸馏 - 视觉-语言跨模态知识蒸馏(图像特征→文本模型)
- teacher_api.py           : API教师模型 - 使用OpenAI/Anthropic等API作为蒸馏教师
- graph_rag_plugin.py      : GraphRAG插件 - 图检索增强生成集成到训练流程
- training_monitor_plugin.py : SOSA监控插件 - 训练过程实时监控WebUI
- web_search_plugin.py     : Web搜索插件 - 集成6个搜索后端(Tavily/Perplexity/DuckDuckGo/Serper/Brave/Volcengine)
- ollama_export_plugin.py  : Ollama导出 - 将APT模型导出为Ollama格式本地部署
- version_manager.py       : 插件版本管理 - 插件兼容性检查和版本控制

## apt_model/data
- data_processor.py        : 数据预处理 - 文本清洗/分词/去重/格式转换
- external_data.py         : 外部数据加载器 - 支持CSV/JSON/TXT等多种格式
- huggingface_loader.py    : HuggingFace集成 - 自动加载HF datasets和模型
- multimodal_dataset.py    : 多模态数据集 - 文本+图像+音频的联合数据加载
- pipeline.py              : 数据流水线 - ETL流程编排(Extract/Transform/Load)
- hlbd/                    : HLBD系统 - Hierarchical Language Bootstrap Data分层语言启蒙数据

## apt_model/webui
- app.py                   : Gradio WebUI - 4标签页(训练监控/梯度监控/检查点管理/推理测试)实时交互界面

## apt_model/api
- server.py                : FastAPI服务器 - 10+端点REST API(生成/训练/压缩/评估)支持异步请求

## apt_model/cli
- commands.py              : CLI命令实现 - train/eval/compress/export等命令的具体逻辑
- parser.py                : 参数解析 - argparse包装，支持子命令和配置文件
- command_registry.py      : 命令注册表 - 动态注册和发现CLI命令
- apx_commands.py          : APX格式命令 - APX模型格式转换相关CLI

## apt_model/console
- core.py                  : 控制台核心 - 插件化命令行系统主控制器
- plugin_bus.py            : 插件总线 - 事件驱动的发布-订阅消息总线
- plugin_loader.py         : 插件加载器 - 动态发现和加载.py插件文件
- plugin_registry.py       : 插件注册表 - 全局插件实例管理和查询
- plugin_adapter.py        : 插件适配器 - 统一不同版本插件的接口
- plugin_standards.py      : 插件标准 - 定义插件接口规范和生命周期
- module_manager.py        : 模块管理器 - Python模块的导入和依赖管理
- eqi_manager.py           : EQI管理器 - Emergent Quality Index涌现质量指数评估
- capability_plugin_map.py : 能力映射 - 功能需求到插件实现的映射表
- auto_loader.py           : 自动加载器 - 启动时自动加载配置的插件
- apx_loader.py            : APX加载器 - 加载APX格式的模型和配置
- cli_organizer.py         : CLI组织器 - 命令行界面的布局和菜单组织
- version_checker.py       : 版本检查器 - 检查插件/依赖/模型版本兼容性

## apt_model/config
- apt_config.py            : APT配置类 - 模型超参数(d_model/n_layers/n_heads等)
- multimodal_config.py     : 多模态配置 - 图像/音频编码器参数和融合策略
- module_config.py         : 模块配置 - 各功能模块的开关和参数
- settings_manager.py      : 配置管理器 - 配置持久化/加载/验证/热更新
- hardware_profile.py      : 硬件配置档 - 针对不同硬件(CPU/GPU/TPU)的优化配置
- optimized_config.py      : 优化配置 - Optuna调优后的最佳超参数配置
- settings.yaml            : YAML配置文件 - 人类可读的配置文件格式

## apt_model/utils
- language_manager.py      : 语言管理器 - 中英日多语言UI/日志/错误消息(42KB)
- hardware_check.py        : 硬件检测 - CPU/GPU/内存/CUDA版本自动检测和验证(35KB)
- visualization.py         : 可视化工具 - 训练曲线/注意力图/梯度流的matplotlib绘图(35KB)
- time_estimator.py        : 时间估算器 - 基于历史数据估算剩余训练时间(32KB)
- cache_manager.py         : 缓存管理 - LRU缓存/磁盘缓存/模型权重缓存(24KB)
- error_persistence.py     : 错误持久化 - 训练错误日志记录和崩溃恢复(21KB)
- resource_monitor.py      : 资源监控 - 实时监控CPU/GPU/内存使用率和温度
- validators.py            : 验证器 - 输入参数/配置文件/模型权重合法性验证
- mascot_render.py         : 吉祥物渲染 - ASCII艺术字/Logo/进度条美化
- logging_config.py        : 日志配置 - 日志级别/格式/输出目标配置
- logging_utils.py         : 日志工具 - 彩色日志/结构化日志/日志轮转
- error_handler.py         : 错误处理 - 统一异常处理/错误码/用户友好错误消息
- common.py                : 通用工具 - 常用辅助函数(路径/时间/字符串处理)

## apt_model/evaluation
- model_evaluator.py       : 模型评估器 - 综合评估BLEU/ROUGE/困惑度/F1等20+指标(49KB)
- comparison.py            : 模型对比 - 多模型性能对比报告生成(表格/图表)
- unified.py               : 统一评估接口 - 封装多种评估框架的统一API

## apt_model/tools/apx
- converter.py             : APX转换器 - APT模型格式与PyTorch/ONNX/SafeTensors互转
- adapters.py              : 格式适配器 - 不同框架(TF/JAX/PyTorch)的权重适配
- detectors.py             : 格式检测器 - 自动识别模型文件格式
- templates.py             : APX模板 - 预定义的模型架构模板

## apt_model/tools/apg
- packager.py              : APG打包器 - 将模型/配置/tokenizer打包成单一.apg文件

## apt_model/interactive
- chat.py                  : 交互式对话 - 命令行实时对话界面(类ChatGPT)
- admin_mode.py            : 管理员模式 - 高级调试和系统管理控制台

## apt_model/infrastructure
- logging.py               : 基础设施日志 - 底层日志框架和格式化器
- errors.py                : 错误定义 - 自定义异常类和错误码常量

---

## apt/ (微内核框架 - 遗留架构，保留用于兼容性)

## apt/core
- registry.py              : 组件注册表 - 全局组件注册和依赖注入容器
- config.py                : 配置管理 - 微内核配置系统(已被apt_model/config取代)
- providers.py             : 服务提供者 - 依赖注入的服务提供者模式
- codecs.py                : 数据编解码器 - 序列化/反序列化工具
- schedules.py             : 调度工具 - 学习率调度/任务调度

## apt/modeling
- compose.py               : 模型组合 - 动态组合多个模型组件

## apt/multilingual
- language.py              : 语言定义 - 语言常量和元数据
- detector.py              : 语言检测 - 自动检测文本语言(基于字符分布)
- tokenizer.py             : 多语言分词器 - 统一的多语言分词接口
- registry.py              : 语言注册表 - 已支持语言的注册和查询

## apt/plugins
- base.py                  : 插件基类 - 定义插件抽象基类和接口
- manager.py               : 插件管理器 - 插件生命周期管理(加载/卸载/热更新)
- hooks.py                 : 插件钩子 - 插件事件钩子机制

---

## Configuration Files
- setup.py                 : Package setup (version 1.0.0)
- requirements.txt         : Core dependencies (84 packages)
- requirements-dev.txt     : Development dependencies (45 packages)
- requirements-minimal.txt : Minimal installation
- .env.example             : Environment variables template (160 lines)
- Makefile                 : Build automation
- config.json              : Module configuration

---

## Documentation (/docs)
- APT_MODEL_HANDBOOK.md    : Complete handbook
- INSTALLATION.md          : Installation guide
- DISTILLATION_PRINCIPLE.md : Knowledge distillation theory
- TEACHER_API_GUIDE.md     : API-as-teacher guide
- VISUAL_DISTILLATION_GUIDE.md : Multi-modal distillation
- FINE_TUNING_GUIDE.md     : LoRA and full fine-tuning
- RL_PRETRAINING_GUIDE.md  : Complete RL training guide
- OPTUNA_GUIDE.md          : Hyperparameter optimization
- KNOWLEDGE_GRAPH_GUIDE.md : GraphRAG usage
- PLUGIN_SYSTEM.md         : Plugin architecture
- PLUGINS_USAGE_GUIDE.md   : Plugin usage (55KB)
- MCP_INTEGRATION_GUIDE.md : Model Context Protocol
- WEB_SEARCH_PLUGIN_GUIDE.md : Web search plugin
- DEBUG_MODE_GUIDE.md      : Debugging features
- DEEPSEEK_TRAINING_GUIDE.md : DeepSeek integration
- GPT_TRAINING_GUIDE.md    : GPT model training
- GRAPH_BRAIN_TRAINING_GUIDE.md : Graph Brain training
- CLAUDE4_MODEL_GUIDE.md   : Claude 4 usage
- DATA_PREPROCESSING_GUIDE.md : Data preprocessing (61KB)
- APX.md                   : APX format specification
- HLBD.md                  : HLBD system
- LAUNCHER_README.md       : GUI launcher guide

---

## Examples (/examples)
- claude4_demo.py          : Claude 4 integration demo
- gpt5_mcp_demo.py         : GPT-5 MCP demo
- multimodal_inference.py  : Multi-modal inference
- visual_distillation_example.py : Visual distillation
- train_distributed.py     : Distributed training
- train_multimodal.py      : Multi-modal training
- multilingual_example.py  : Multi-language usage
- rl_examples/             : RL training examples (DPO, GRPO)
- pretraining_examples/    : Pretraining examples (contrastive, MLM)
- graph_rag_examples/      : GraphRAG examples
- plugin_template/         : Custom plugin template
- profiles/                : Configuration profiles (tiny_debug, gpt5_moe_reasoning)

---

## Tests (/tests)
- test_trainer_complete.py : Complete trainer tests
- test_compression_plugin.py : Compression tests
- test_multimodal.py       : Multi-modal tests
- test_multilingual.py     : Multi-language tests
- test_plugin_system.py    : Plugin system tests
- test_vft_tva.py          : VFT-TVA tests
- test_hlbd_quick_learning.py : HLBD tests
- test_error_persistence.py : Error tracking tests
- conftest.py              : pytest configuration

---

## Scripts (/scripts)
- launch_distributed.sh    : Multi-GPU/node launcher
- run_optuna_optimization.sh : HPO script
- run_best_training.sh     : Best config training
- apt_eqi_manager.py       : EQI manager
- apx_converter.py         : APX converter
- eqi.py                   : EQI utilities
- launchers/               : GUI launcher applications

---

## Pre-trained Models (/bert)
- bert-base-chinese/       : Chinese BERT model (weights, vocab, config)

---

## 核心技术特性详解

### 1. 模型架构
- **APT (Autopoietic Transformer)** - 自生成注意力机制 + DBC-DAC压缩优化
- **VFT-TVA** - Vein-Flow Transformer低秩注意力子空间，复杂度从O(T²·d)降至O(T²·r)
- **Claude-4** - GPT-4o + 图反思层(图连通度分析/最短路径推理/镜像复杂度网络)
- **GPT-5** - Codebook MoE专家混合 + Leaf-Vote K=2 + 流式检索 + 双态精度对齐
- **多模态** - 文本/图像/音频三模态融合(cross_attention/tri_modal/gated)

### 2. 训练系统
- **分布式训练** - 多GPU/多节点DDP/FSDP，支持gradient checkpointing
- **混合精度** - FP16/BF16自动混合精度，accelerate库集成
- **原子检查点** - 两阶段保存(tmp→final)防止训练崩溃导致权重损坏
- **DBC加速** - Dimension-Balanced Compression训练加速20-30%
- **训练守卫** - NaN检测/早停/异常自动恢复/梯度裁剪
- **SOSA监控** - Self-Optimizing System Architecture实时WebSocket推送

### 3. 强化学习
- **RLHF** - 基于PPO的人类反馈强化学习(clip_epsilon=0.2, GAE-λ)
- **DPO** - Direct Preference Optimization直接偏好优化(无需奖励模型)
- **GRPO** - Group Relative Policy Optimization群组相对策略优化
- **奖励模型** - 从人类偏好对(chosen/rejected)训练打分器

### 4. 知识蒸馏
- **API教师** - 使用OpenAI/Anthropic等API作为教师模型
- **多模态蒸馏** - 视觉-语言跨模态知识迁移(图像特征→文本模型)
- **温度缩放** - Temperature=4.0软标签蒸馏
- **损失组合** - α·KD_loss + β·CE_loss

### 5. 模型压缩
- **剪枝(Pruning)** - Magnitude/Structured剪枝移除不重要权重
- **量化(Quantization)** - 8-bit动态量化，支持QAT
- **知识蒸馏(KD)** - 大模型→小模型知识迁移
- **低秩分解** - 权重矩阵SVD分解降维
- **DBC** - Dimension-Balanced Compression维度平衡压缩

### 6. GraphRAG图检索
- **泛图结构** - Hypergraph超图/Simplicial Complex单纯复形/Pan-graph
- **Hodge-Laplacian** - 高阶拓扑信号处理(边流/面流/k-链谱分解)
- **图脑引擎** - 认知自由能最小化F=U-T·S，动态拓扑相变
- **多跳推理** - 知识图谱三元组(head-relation-tail)的路径推理
- **RAG融合** - FAISS/Annoy/exact cosine向量检索 + 图结构增强

### 7. 多语言支持
- **中文原生** - 字符级(char)/词级(word)分词，Unicode范围0x4e00-0x9fff
- **自动检测** - 基于字符分布自动识别中英日韩文本
- **混合训练** - 同一batch内中英文混合训练
- **多语言UI** - 42KB语言管理器支持中英日界面/日志/错误消息

### 8. 插件系统
- **26+生产插件** - 压缩/蒸馏/GraphRAG/Web搜索/Ollama导出等
- **事件驱动** - 发布-订阅消息总线，插件间松耦合通信
- **热插拔** - 运行时动态加载/卸载插件无需重启
- **版本管理** - 插件兼容性检查和语义版本控制
- **能力映射** - 功能需求自动路由到对应插件实现

### 9. Web搜索集成
- **6个搜索后端** - Tavily/Perplexity(<400ms)/DuckDuckGo/Serper/Brave/Volcengine火山引擎
- **AI-native** - Tavily专为AI agents设计的搜索API
- **隐私优先** - DuckDuckGo/Brave无追踪搜索
- **中国优化** - Volcengine字节跳动搜索(DeepSeek合作平台)

### 10. 数据处理
- **HLBD系统** - Hierarchical Language Bootstrap Data分层语言启蒙
- **多模态数据** - 文本+图像+音频联合数据加载
- **HuggingFace集成** - 自动加载datasets和模型
- **ETL流水线** - Extract/Transform/Load数据预处理编排

---

## Platform Support
- OS: Linux, macOS, Windows
- Hardware: CPU, CUDA, MPS (Apple Silicon)
- Python: 3.8-3.12
- Offline-friendly design
