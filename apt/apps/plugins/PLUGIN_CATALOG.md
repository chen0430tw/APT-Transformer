# APT-Transformer Plugin Catalog

**Last Updated**: 2026-01-21
**Total Plugins**: 17
**Categories**: 8

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

| Category | Count | Status |
|----------|-------|--------|
| Core | 3 | Stable |
| Integration | 3 | Stable |
| Distillation | 2 | Stable |
| Experimental | 3 | Beta |
| Monitoring | 2 | Stable ✨ |
| Visualization | 1 | Stable ✨ |
| Evaluation | 2 | Stable ✨ |
| Infrastructure | 1 | Stable ✨ |
| **Total** | **17** | - |

---

## 🔮 Upcoming Plugins (Tier 2)

### Planned Categories:

1. **Export Plugins** (1 module)
   - APX Converter

2. **Optimization Plugins** (1 module)
   - MXFP4 Quantization

3. **RL Plugins** (4 modules)
   - RLHF Trainer
   - DPO Trainer
   - GRPO Trainer
   - Reward Model

4. **Data Plugins** (2 modules)
   - Data Processor
   - Data Pipeline

5. **Protocol Plugins** (1 module)
   - MCP Integration

6. **Retrieval Plugins** (1 module)
   - RAG Integration

**Total Tier 2**: 10 modules

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
