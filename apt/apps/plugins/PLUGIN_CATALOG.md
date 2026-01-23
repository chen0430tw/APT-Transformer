# APT-Transformer Plugin Catalog

**Last Updated**: 2026-01-22
**Total Plugins**: 31
**Categories**: 15

---

## 📋 Plugin Categories

### 1. Core Plugins (3)
**Location**: `apt/apps/plugins/core/`
**Description**: 核心插件 - 训练和优化必需

| Plugin | Description | Status |
|--------|-------------|--------|
| `compression_plugin.py` | 模型压缩和量化（蒸馏、剪枝、量化） | ✅ Active |
| `training_monitor_plugin.py` | SOSA训练监控 - 自动检测和修复 | ✅ Active |
| `version_manager.py` | 版本管理和依赖控制 | ✅ Active |

---

### 2. Integration Plugins (3)
**Location**: `apt/apps/plugins/integration/`
**Description**: 集成插件 - 外部服务和工具

| Plugin | Description | Status |
|--------|-------------|--------|
| `graph_rag_plugin.py` | GraphRAG系统 - 图检索增强生成 | ✅ Active |
| `ollama_export_plugin.py` | Ollama导出 - GGUF格式转换和注册 | ✅ Active |
| `web_search_plugin.py` | Web搜索集成 - 检索增强生成 | ✅ Active |

---

### 3. Distillation Plugins (2)
**Location**: `apt/apps/plugins/distillation/`
**Description**: 蒸馏套件 - 知识蒸馏相关

| Plugin | Description | Status |
|--------|-------------|--------|
| `teacher_api.py` | Teacher API - 教师模型服务 | ✅ Active |
| `visual_distillation_plugin.py` | 可视化蒸馏 - 带可视化的知识蒸馏 | ✅ Active |

---

### 4. Experimental Plugins (3)
**Location**: `apt/apps/plugins/experimental/`
**Description**: 实验性插件 - 从Legacy提取，需要评估和现代化

| Plugin | Description | Status |
|--------|-------------|--------|
| `plugin_6_multimodal_training.py` | 多模态训练 - 图文混合训练 | 🧪 Experimental |
| `plugin_7_data_processors.py` | 数据处理 - 清洗/增强/采样 | 🧪 Experimental |
| `plugin_8_advanced_debugging.py` | 高级调试 - 梯度监控/激活分析 | 🧪 Experimental |

---

### 5. Monitoring Plugins (2) ✨ NEW
**Location**: `apt/apps/plugins/monitoring/`
**Description**: 监控和诊断插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `gradient_monitor_plugin.py` | 梯度监控 - 检测梯度消失/爆炸 | ✅ Active |
| `resource_monitor_plugin.py` | 资源监控 - GPU/内存/CPU监控 | ✅ Active |

**Features**:
- Real-time gradient flow analysis
- Vanishing/exploding gradient detection
- GPU utilization tracking
- Memory usage monitoring
- JSON export for WebUI integration

---

### 6. Visualization Plugins (1) ✨ NEW
**Location**: `apt/apps/plugins/visualization/`
**Description**: 可视化插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `model_visualization_plugin.py` | 模型可视化 - 训练结果和评估可视化 | ✅ Active |

**Features**:
- Training curves (loss, accuracy, learning rate)
- Confusion matrices and heatmaps
- Attention weight visualization
- Model architecture diagrams
- Comparative charts
- Supports: matplotlib, plotly, seaborn

---

### 7. Evaluation Plugins (2) ✨ NEW
**Location**: `apt/apps/plugins/evaluation/`
**Description**: 评估和基准测试插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `model_evaluator_plugin.py` | 模型评估 - 综合评估框架 | ✅ Active |
| `model_comparison_plugin.py` | 模型对比 - 多模型比较分析 | ✅ Active |

**Evaluation Sets**:
- General knowledge
- Reasoning and logic
- Coding capabilities
- Creative writing
- Chinese language understanding
- Mathematical problem-solving

**Metrics**:
- Accuracy, Precision, Recall, F1-Score
- Perplexity, BLEU, ROUGE
- Custom domain metrics

---

### 8. Infrastructure Plugins (1) ✨ NEW
**Location**: `apt/apps/plugins/infrastructure/`
**Description**: 基础设施插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `logging_plugin.py` | 集中式日志 - 统一日志基础设施 | ✅ Active |

**Features**:
- Structured logging
- Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log aggregation
- Context preservation
- Performance tracking

---

### 9. Optimization Plugins (1) ✨ NEW - Tier 2
**Location**: `apt/apps/plugins/optimization/`
**Description**: 性能优化插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `mxfp4_quantization_plugin.py` | MXFP4量化 - Microsoft-OpenAI 4位浮点格式 | ✅ Active |

**Features**:
- 4-bit floating point quantization
- Block-wise 8-bit scaling
- 4x inference speedup with <1% accuracy loss
- Dynamic range support

---

### 10. RL Plugins (4) ✨ NEW - Tier 2
**Location**: `apt/apps/plugins/rl/`
**Description**: 强化学习插件 - 可选的对齐训练方法

| Plugin | Description | Status |
|--------|-------------|--------|
| `rlhf_trainer_plugin.py` | RLHF训练 - 基于人类反馈的强化学习 | ✅ Active |
| `dpo_trainer_plugin.py` | DPO训练 - 直接偏好优化 | ✅ Active |
| `grpo_trainer_plugin.py` | GRPO训练 - 组相对策略优化 | ✅ Active |
| `reward_model_plugin.py` | 奖励模型 - RLHF训练工具 | ✅ Active |

**Features**:
- Multiple alignment training methods (RLHF, DPO, GRPO)
- Reward model for scoring responses
- Value head for response evaluation
- Preference-based training
- Compatible with transformers and trl libraries

---

### 11. Protocol Plugins (1) ✨ NEW - Tier 2
**Location**: `apt/apps/plugins/protocol/`
**Description**: 协议集成插件 - 外部协议支持

| Plugin | Description | Status |
|--------|-------------|--------|
| `mcp_integration_plugin.py` | MCP协议集成 - Model Context Protocol | ✅ Active |

**Features**:
- Async/streaming retrieval support
- AsyncRetrievalWorker for non-blocking operations
- StreamingRetrieverAdapter for interface compatibility
- Integration with FAISS/Annoy/ExactCosine providers
- Bridges GPT-5's StreamingRetriever with APT infrastructure

---

### 12. Retrieval Plugins (2) ✨ NEW - Tier 2
**Location**: `apt/apps/plugins/retrieval/`
**Description**: 检索增强插件 - 可选的RAG功能

| Plugin | Description | Status |
|--------|-------------|--------|
| `rag_integration_plugin.py` | RAG集成 - 检索增强生成 | ✅ Active |
| `kg_rag_integration_plugin.py` | KG+RAG融合 - 知识图谱+检索增强 | ✅ Active |

**Features**:
- **RAG Integration**:
  - Wraps language models with retrieval capabilities
  - Index building and caching
  - Multiple retrieval providers (FAISS, Annoy, Exact)
  - Layer-wise injection of retrieved context

- **KG+RAG Integration**:
  - Combines structured knowledge graphs with unstructured retrieval
  - Fusion strategies (weighted, concatenation, gated)
  - Multi-hop reasoning support
  - Dual retrieval system

---

## 🚀 Usage

### Loading Plugins

```python
from apt.apps.plugin_system.manager import PluginManager

# Initialize plugin manager
pm = PluginManager()

# Load specific plugin
pm.load_plugin("monitoring.gradient_monitor_plugin")

# Load all plugins in a category
pm.load_category("monitoring")

# Load with configuration
pm.load_plugin("visualization.model_visualization_plugin", config={
    "backend": "plotly",
    "export_format": "html",
})
```

### Plugin Configuration

Each plugin can be configured via YAML:

```yaml
# config/plugins.yaml
monitoring:
  gradient_monitor_plugin:
    enabled: true
    check_interval: 100
    threshold:
      vanishing: 1e-6
      exploding: 100.0

visualization:
  model_visualization_plugin:
    enabled: true
    backend: "plotly"
    export_dir: "artifacts/visualizations"
```

---

## 📊 Plugin Statistics

| Category | Count | Status | Tier |
|----------|-------|--------|------|
| Core | 3 | Stable | Pre-existing |
| Integration | 3 | Stable | Pre-existing |
| Distillation | 2 | Stable | Pre-existing |
| Experimental | 3 | Beta | Pre-existing |
| Monitoring | 2 | Stable ✨ | Tier 1 |
| Visualization | 1 | Stable ✨ | Tier 1 |
| Evaluation | 2 | Stable ✨ | Tier 1 |
| Infrastructure | 1 | Stable ✨ | Tier 1 |
| Optimization | 1 | Stable ✨ | Tier 2 |
| RL | 4 | Stable ✨ | Tier 2 |
| Protocol | 1 | Stable ✨ | Tier 2 |
| Retrieval | 2 | Stable ✨ | Tier 2 |
| Hardware | 3 | Stable ✨ | Tier 3 |
| Deployment | 2 | Stable ✨ | Tier 3 |
| Memory | 1 | Stable ✨ | Tier 3 |
| **Total** | **31** | - | - |

---

## ✅ Tier 2 Complete!

All Tier 2 plugins have been successfully converted:
- ✅ Optimization (1): MXFP4 Quantization
- ✅ RL (4): RLHF, DPO, GRPO, Reward Model
- ✅ Protocol (1): MCP Integration
- ✅ Retrieval (2): RAG Integration, KG+RAG Integration

**Note**: APX Converter and Data Processor/Pipeline were **intentionally excluded** - they should remain as tools and core modules respectively.

---

### 13. Hardware Plugins (3) ✨ NEW - Tier 3
**Location**: `apt/apps/plugins/hardware/`
**Description**: 硬件模拟和适配插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `virtual_blackwell_plugin.py` | 虚拟Blackwell - GPU特性模拟 | ✅ Active |
| `npu_backend_plugin.py` | NPU后端 - Neural Processing Unit加速 | ✅ Active |
| `cloud_npu_adapter_plugin.py` | 云NPU适配器 - 云环境NPU支持 | ✅ Active |

**Features**:
- Virtual GPU feature simulation (Blackwell architecture)
- NPU hardware acceleration support
- Cloud NPU adaptation for cloud environments
- Hardware abstraction layer

---

### 14. Deployment Plugins (2) ✨ NEW - Tier 3
**Location**: `apt/apps/plugins/deployment/`
**Description**: 部署和虚拟化插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `microvm_compression_plugin.py` | MicroVM压缩 - 微虚拟机部署优化 | ✅ Active |
| `vgpu_stack_plugin.py` | vGPU Stack - 虚拟GPU管理 | ✅ Active |

**Features**:
- MicroVM compression for lightweight deployment
- Virtual GPU resource management
- Container and cloud environment optimization
- vGPU allocation and scheduling

---

### 15. Memory Plugins (1) ✨ NEW - Tier 3
**Location**: `apt/apps/plugins/memory/`
**Description**: 高级记忆系统插件

| Plugin | Description | Status |
|--------|-------------|--------|
| `aim_memory_plugin.py` | AIM Memory - Advanced In-context Memory System | ✅ Active |

**Features**:
- Hierarchical memory organization
- Advanced in-context memory management
- Context composition and retrieval
- Long-term memory support

---

## 🎉 Tier 3 Complete!

All Tier 3 plugins have been successfully converted:
- ✅ Hardware (3): Virtual Blackwell, NPU Backend, Cloud NPU Adapter
- ✅ Deployment (2): MicroVM Compression, vGPU Stack
- ✅ Memory (1): AIM Memory System

**Note**: GPU Flash Optimization and Extreme Scale Training were **intentionally excluded** - they are core performance optimizations, not optional plugins.

---

## 🏆 All Tiers Complete!

**Summary**:
- **Tier 1** (6 modules): Monitoring, Visualization, Evaluation, Infrastructure
- **Tier 2** (8 modules): Optimization, RL, Protocol, Retrieval
- **Tier 3** (6 modules): Hardware, Deployment, Memory

**Total**: 20 modules converted → 31 plugins across 15 categories

---

## 🔮 Not Converted (By Design)

The following modules were **intentionally not converted** to plugins:

### Tools (Should Remain as Tools)
- **APX Converter** (`apt_model/tools/apx/converter.py`) - Packaging tool, not runtime plugin
- **Data Generation Tools** - Build-time utilities
- **Diagnostic Tools** - Development utilities

### Core Modules (Should Remain as Modules)
- **Data Processor** (`apt/core/data/data_processor.py`) - Core data processing
- **Data Pipeline** (`apt/core/data/pipeline.py`) - Core data pipeline
- **Knowledge Graph** (`apt/memory/knowledge_graph.py`) - L2 core functionality
- **External Data Loader** (`apt/core/data/external_data.py`) - Core data capability

### Core Optimizations (Should Remain as Modules)
- **GPU Flash Optimization** (`apt/perf/optimization/gpu_flash_optimization.py`) - Core performance optimization
- **Extreme Scale Training** (`apt/perf/optimization/extreme_scale_training.py`) - Core training capability

### Already Plugins
- **GraphRAG** - Already exists as `apt/apps/plugins/integration/graph_rag_plugin.py`

**Rationale**: Not all modules should be plugins. Tools remain tools, core functionality remains core, and only truly optional/experimental/integration features become plugins.

---

## 📝 Plugin Development Guidelines

### 1. Plugin Structure

```python
from apt.apps.plugin_system.base import PluginBase

class MyPlugin(PluginBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "my_plugin"
        self.version = "1.0.0"

    def load(self):
        """Initialize plugin resources"""
        pass

    def unload(self):
        """Cleanup plugin resources"""
        pass

    def execute(self, *args, **kwargs):
        """Main plugin logic"""
        pass
```

### 2. Plugin Metadata

Each plugin should include:
- Name and version
- Dependencies
- Configuration schema
- API documentation
- Usage examples

### 3. Testing

All plugins must have:
- Unit tests
- Integration tests
- Performance benchmarks (if applicable)

---

## 🔗 Related Documentation

- **Plugin System Guide**: `docs/product/PLUGIN_SYSTEM_GUIDE.md`
- **Architecture**: `docs/guides/COMPLETE_TECH_SUMMARY.md`
- **Development**: `CONTRIBUTING.md`

---

## 📞 Support

For plugin-related questions:
1. Check the Plugin System Guide
2. Review example plugins in `experimental/`
3. Open an issue on GitHub

---

**Legend**:
- ✅ Active - Production-ready
- 🧪 Experimental - Under evaluation
- ✨ NEW - Recently added
