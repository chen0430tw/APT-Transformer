# APT-Transformer Repository Schema

## Project Overview
APT-Transformer (Autopoietic Transformer) is a production-ready PyTorch Transformer training platform with comprehensive Chinese and English support, featuring RL training, knowledge distillation, GraphRAG integration, and an extensible plugin system.

---

## apt_model/modeling
- apt_model.py              : Core APT transformer architecture
- multimodal_model.py       : Multi-modal fusion transformer
- claude4_model.py          : Claude 4 integration model
- gpt4o_model.py           : GPT-4o architecture
- gpt5_model.py            : GPT-5 model implementation
- gpto3_model.py           : GPT-o3 reasoning model
- vft_tva_model.py         : VFT-TVA (Visual Feature Transfer - Temporal Visual Attention)
- knowledge_graph.py       : Knowledge graph-enhanced transformer
- chinese_tokenizer.py     : Native Chinese tokenizer
- mcp_integration.py       : Model Context Protocol integration
- embeddings.py            : Embedding layers and utilities
- rag_integration.py       : RAG (Retrieval-Augmented Generation) integration
- kg_rag_integration.py    : Knowledge Graph + RAG integration
- utils.py                 : Modeling utilities

## apt_model/core
- api_providers.py         : Unified API interface (OpenAI, Anthropic, DeepSeek, etc.)
- hardware.py              : Hardware detection and optimization
- resources.py             : Resource management and allocation
- system.py                : System utilities and platform detection

## apt_model/core/graph_rag
- graph_rag_manager.py     : GraphRAG manager and orchestration
- generalized_graph.py     : Generalized graph structures (hypergraphs, simplicial complexes)
- hodge_laplacian.py       : Hodge-Laplacian spectral analysis
- graph_brain.py           : Dynamic graph modeling with brain-inspired architecture
- demo_full.py             : Comprehensive GraphRAG demonstration

## apt_model/core/training
- training_monitor.py      : Real-time training monitoring
- sosa_core.py             : SOSA (Self-Optimizing System Architecture) core
- apt_integration.py       : APT model training integration

## apt_model/training
- trainer.py               : Main training orchestration
- claude_trainer.py        : Claude-specific trainer
- gpt_trainer.py           : GPT-specific trainer
- vft_tva_trainer.py       : VFT-TVA training pipeline
- finetuner.py             : Fine-tuning utilities (LoRA, full fine-tuning)
- checkpoint.py            : Atomic checkpoint saving (corruption prevention)
- callbacks.py             : Training callbacks and hooks
- gradient_monitor.py      : Gradient flow analysis and visualization
- mixed_precision.py       : FP16/BF16 mixed precision training
- training_guard.py        : Training safety mechanisms
- optimizer.py             : Custom optimizers and schedulers
- training_events.py       : Event-driven training system
- hooks.py                 : Training hooks
- train_reasoning.py       : Reasoning-focused training

## apt_model/rl
- rlhf_trainer.py          : RLHF (Reinforcement Learning from Human Feedback) with PPO
- dpo_trainer.py           : DPO (Direct Preference Optimization)
- grpo_trainer.py          : GRPO (Group Relative Policy Optimization)
- reward_model.py          : Reward model training

## apt_model/pretraining
- contrastive_pretrain.py  : Contrastive learning (SimCLR, MoCo)
- mlm_pretrain.py          : Masked Language Modeling (BERT/RoBERTa style)

## apt_model/generation
- generator.py             : Text generation engine
- evaluator.py             : Text quality evaluation and metrics

## apt_model/plugins
- compression_plugin.py    : Model compression (pruning, quantization, distillation, low-rank, DBC)
- visual_distillation_plugin.py : Multi-modal knowledge distillation
- teacher_api.py           : External API as teacher model
- graph_rag_plugin.py      : GraphRAG plugin
- training_monitor_plugin.py : SOSA training monitor plugin
- web_search_plugin.py     : Web search capabilities (DuckDuckGo, Tavily, Perplexity, Brave)
- ollama_export_plugin.py  : Ollama model export
- version_manager.py       : Plugin version management

## apt_model/data
- data_processor.py        : Data preprocessing and cleaning
- external_data.py         : External dataset loaders
- huggingface_loader.py    : HuggingFace datasets integration
- multimodal_dataset.py    : Multi-modal dataset handling
- pipeline.py              : Data pipeline orchestration
- hlbd/                    : HLBD (Hierarchical Language Bootstrap Data) system

## apt_model/webui
- app.py                   : Gradio web interface (4 tabs: Training Monitor, Gradient Monitor, Checkpoint Management, Inference)

## apt_model/api
- server.py                : FastAPI REST API server (10+ endpoints)

## apt_model/cli
- commands.py              : Command implementations
- parser.py                : Argument parsing
- command_registry.py      : Command registration system
- apx_commands.py          : APX format commands

## apt_model/console
- core.py                  : Console core system
- plugin_bus.py            : Event-driven plugin bus
- plugin_loader.py         : Dynamic plugin loading
- plugin_registry.py       : Plugin registry
- plugin_adapter.py        : Plugin adaptation layer
- plugin_standards.py      : Plugin standards and interfaces
- module_manager.py        : Module management
- eqi_manager.py           : EQI (Emergent Quality Index) manager
- capability_plugin_map.py : Capability-plugin mapping
- auto_loader.py           : Automatic plugin loading
- apx_loader.py            : APX format loader
- cli_organizer.py         : CLI organization
- version_checker.py       : Version checking

## apt_model/config
- apt_config.py            : Main APT configuration
- multimodal_config.py     : Multi-modal configuration
- module_config.py         : Module configuration
- settings_manager.py      : Settings persistence and management
- hardware_profile.py      : Hardware-specific profiles
- optimized_config.py      : Optimized configurations
- settings.yaml            : YAML configuration file

## apt_model/utils
- language_manager.py      : Multi-language support (Chinese, English, Japanese)
- hardware_check.py        : Hardware detection and validation
- visualization.py         : Training visualization utilities
- time_estimator.py        : Training time estimation
- cache_manager.py         : Caching system
- error_persistence.py     : Error tracking and persistence
- resource_monitor.py      : Resource monitoring (CPU, GPU, memory)
- validators.py            : Input validation utilities
- mascot_render.py         : ASCII art rendering
- logging_config.py        : Logging configuration
- logging_utils.py         : Logging utilities
- error_handler.py         : Error handling utilities
- common.py                : Common utilities

## apt_model/evaluation
- model_evaluator.py       : Comprehensive model evaluation
- comparison.py            : Model comparison utilities
- unified.py               : Unified evaluation interface

## apt_model/tools/apx
- converter.py             : APX format converter
- adapters.py              : Format adapters
- detectors.py             : Format detection
- templates.py             : APX templates

## apt_model/tools/apg
- packager.py              : APG packaging utilities

## apt_model/interactive
- chat.py                  : Interactive chat interface
- admin_mode.py            : Admin mode console

## apt_model/infrastructure
- logging.py               : Infrastructure logging
- errors.py                : Error definitions

---

## apt/ (Microkernel Framework - Legacy Architecture)

## apt/core
- registry.py              : Component registry
- config.py                : Configuration management
- providers.py             : Service providers
- codecs.py                : Data codecs
- schedules.py             : Scheduling utilities

## apt/modeling
- compose.py               : Model composition

## apt/multilingual
- language.py              : Language definitions
- detector.py              : Language detection
- tokenizer.py             : Multi-language tokenizers
- registry.py              : Language registry

## apt/plugins
- base.py                  : Plugin base classes
- manager.py               : Plugin manager
- hooks.py                 : Plugin hooks

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

## Key Features

### Training
- Distributed training (multi-GPU, multi-node)
- Mixed precision (FP16/BF16)
- Atomic checkpoints (corruption prevention)
- DBC acceleration (20-30% speedup)

### Reinforcement Learning
- RLHF (PPO-based)
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)

### Knowledge Distillation
- Student-teacher training
- API-based teacher models
- Multi-modal distillation

### Model Compression
- Pruning, Quantization, Distillation
- Low-rank decomposition, DBC

### GraphRAG
- Hodge-Laplacian spectral analysis
- Graph Brain dynamic modeling
- Generalized graph structures

### Multi-language
- Native Chinese and English support
- Automatic language detection
- Multi-language mixed training

### Plugin System
- 26+ production plugins
- Event-driven architecture
- Hot-swappable plugins

---

## Platform Support
- OS: Linux, macOS, Windows
- Hardware: CPU, CUDA, MPS (Apple Silicon)
- Python: 3.8-3.12
- Offline-friendly design
