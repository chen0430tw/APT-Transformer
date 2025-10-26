"""
遗留插件适配器工厂

为8个现有插件提供适配配置，将它们包装成符合PluginBase的插件
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

from apt_model.console.plugin_adapter import LegacyPluginAdapter
from apt_model.console.plugin_standards import PluginPriority

# 添加legacy_plugins到Python路径
LEGACY_PLUGINS_DIR = Path(__file__).parent.parent.parent.parent / "legacy_plugins"


def create_huggingface_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建HuggingFace集成插件适配器

    功能：
    - 导入/导出模型到HuggingFace Hub
    - 加载HuggingFace数据集
    - 使用HF Trainer训练模型
    - 创建和管理模型卡片

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    # 动态导入原插件
    sys.path.insert(0, str(LEGACY_PLUGINS_DIR / "batch1"))
    from huggingface_integration_plugin import HuggingFaceIntegrationPlugin

    # 创建原插件实例
    config = config or {}
    legacy_plugin = HuggingFaceIntegrationPlugin(config)

    # 创建适配器
    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="huggingface_integration",
        priority=PluginPriority.ADMIN_AUDIT,  # 700
        events=["on_init", "on_shutdown"],
        category="integration",
        blocking=False,
        description="HuggingFace Hub集成 - 模型导入/导出、数据集加载、HF Trainer训练",
        required_capabilities=[],
        optional_capabilities=[],
        provides_capabilities=["huggingface_export", "huggingface_import"],
    )


def create_cloud_storage_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建云存储插件适配器

    功能：
    - AWS S3 备份和恢复
    - 阿里云 OSS 备份和恢复
    - HuggingFace Hub 备份
    - ModelScope 备份
    - 多云同步

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    sys.path.insert(0, str(LEGACY_PLUGINS_DIR / "batch1"))
    from cloud_storage_plugin import CloudStoragePlugin

    config = config or {}
    legacy_plugin = CloudStoragePlugin(config)

    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="cloud_storage",
        priority=PluginPriority.ADMIN_AUDIT,  # 700
        events=["on_batch_end", "on_shutdown"],
        category="storage",
        blocking=False,
        description="多云存储 - S3/OSS/HF/ModelScope备份和恢复",
        required_capabilities=[],
        optional_capabilities=[],
        provides_capabilities=["cloud_backup", "cloud_restore"],
    )


def create_ollama_export_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建Ollama导出插件适配器

    功能：
    - 转换为GGUF格式
    - 创建Modelfile
    - 注册到本地Ollama
    - 模型测试和验证
    - 支持多种量化格式（Q4_0, Q4_K_M, Q5_K_M, Q8_0, FP16）

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    # Ollama插件在项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from ollama_export_plugin import OllamaExportPlugin

    config = config or {}
    legacy_plugin = OllamaExportPlugin(config)

    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="ollama_export",
        priority=PluginPriority.POST_CLEANUP,  # 900
        events=["on_shutdown"],
        category="export",
        blocking=False,
        description="Ollama导出 - GGUF转换、本地部署、量化支持",
        required_capabilities=[],
        optional_capabilities=["quantization"],
        provides_capabilities=["ollama_export", "gguf_conversion"],
    )


def create_distillation_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建模型蒸馏插件适配器

    功能：
    - 响应蒸馏（KL散度）
    - 特征蒸馏（中间层对齐）
    - 关系蒸馏（样本关系保持）
    - 注意力蒸馏（注意力权重对齐）
    - 压缩效果评估

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    sys.path.insert(0, str(LEGACY_PLUGINS_DIR / "batch1"))
    from model_distillation_plugin import ModelDistillationPlugin

    config = config or {}
    legacy_plugin = ModelDistillationPlugin(config)

    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="model_distillation",
        priority=PluginPriority.TRAINING,  # 350
        events=["on_batch_end", "on_step_end"],
        category="training",
        blocking=False,
        description="知识蒸馏 - 响应/特征/关系/注意力蒸馏",
        required_capabilities=[],
        optional_capabilities=["quantization"],
        provides_capabilities=["distillation"],
    )


def create_pruning_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建模型剪枝插件适配器

    功能：
    - Magnitude剪枝（权重绝对值）
    - Taylor剪枝（梯度×权重）
    - 结构化剪枝（整个神经元/通道）
    - 彩票假说剪枝（Lottery Ticket）
    - 剪枝后微调

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    sys.path.insert(0, str(LEGACY_PLUGINS_DIR / "batch1"))
    from model_pruning_plugin import ModelPruningPlugin

    config = config or {}
    legacy_plugin = ModelPruningPlugin(config)

    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="model_pruning",
        priority=PluginPriority.TRAINING,  # 350
        events=["on_batch_end", "on_step_end"],
        category="training",
        blocking=False,
        description="模型剪枝 - Magnitude/Taylor/结构化/彩票假说剪枝",
        required_capabilities=[],
        optional_capabilities=["quantization"],
        provides_capabilities=["pruning"],
    )


def create_multimodal_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建多模态训练插件适配器

    功能：
    - 文本+图像+音频联合训练
    - 多种融合策略（Concatenate/Add/Attention）
    - 支持CLIP、ViT图像编码器
    - 支持Wav2Vec2音频编码器
    - 跨模态注意力机制

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    sys.path.insert(0, str(LEGACY_PLUGINS_DIR / "batch2"))
    from plugin_6_multimodal_training import MultimodalTrainingPlugin

    config = config or {}
    legacy_plugin = MultimodalTrainingPlugin(config)

    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="multimodal_training",
        priority=PluginPriority.TRAINING,  # 350
        events=["on_batch_start", "on_batch_end"],
        category="training",
        blocking=False,
        description="多模态训练 - 文本/图像/音频联合训练、跨模态注意力",
        required_capabilities=[],
        optional_capabilities=["multimodal"],
        provides_capabilities=["multimodal_training"],
    )


def create_data_processors_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建数据处理器插件适配器

    功能：
    - 文本清洗（基础/激进/中文/代码）
    - 数据增强（同义词替换/随机交换/随机删除/回译/EDA）
    - 数据平衡（过采样/欠采样/SMOTE）
    - 质量检查（重复检测/长度过滤/语言检测）
    - 完整处理流程

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    sys.path.insert(0, str(LEGACY_PLUGINS_DIR / "batch2"))
    from plugin_7_data_processors import DataProcessorsPlugin

    config = config or {}
    legacy_plugin = DataProcessorsPlugin(config)

    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="data_processors",
        priority=PluginPriority.CORE_RUNTIME,  # 100
        events=["on_init"],
        category="data",
        blocking=False,
        description="数据处理 - 清洗/增强/平衡/质量检查",
        required_capabilities=[],
        optional_capabilities=[],
        provides_capabilities=["data_preprocessing", "data_augmentation"],
    )


def create_debugging_adapter(config: Optional[Dict[str, Any]] = None) -> LegacyPluginAdapter:
    """
    创建高级调试插件适配器

    功能：
    - 梯度监控（实时监控/爆炸消失检测）
    - 激活值监控（统计分析/死神经元检测）
    - 内存监控（GPU内存追踪/泄漏检测）
    - 性能分析（瓶颈识别/profiling）
    - 异常诊断（NaN/Inf检测）
    - 可视化报告（梯度/激活值/内存）

    Args:
        config: 插件配置字典

    Returns:
        适配后的插件实例
    """
    sys.path.insert(0, str(LEGACY_PLUGINS_DIR / "batch2"))
    from plugin_8_advanced_debugging import AdvancedDebuggingPlugin

    config = config or {}
    legacy_plugin = AdvancedDebuggingPlugin(config)

    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name="advanced_debugging",
        priority=PluginPriority.TELEMETRY,  # 800
        events=[
            "on_batch_start",
            "on_batch_end",
            "on_step_start",
            "on_step_end",
        ],
        category="debug",
        blocking=False,
        description="高级调试 - 梯度/激活值/内存监控、异常诊断、性能分析",
        required_capabilities=[],
        optional_capabilities=[],
        provides_capabilities=["debugging", "profiling"],
    )


# 适配器注册表
LEGACY_ADAPTERS = {
    "huggingface_integration": create_huggingface_adapter,
    "cloud_storage": create_cloud_storage_adapter,
    "ollama_export": create_ollama_export_adapter,
    "model_distillation": create_distillation_adapter,
    "model_pruning": create_pruning_adapter,
    "multimodal_training": create_multimodal_adapter,
    "data_processors": create_data_processors_adapter,
    "advanced_debugging": create_debugging_adapter,
}


def get_all_legacy_adapters(config: Optional[Dict[str, Any]] = None) -> Dict[str, LegacyPluginAdapter]:
    """
    获取所有遗留插件的适配器

    Args:
        config: 全局配置字典

    Returns:
        插件名到适配器实例的字典

    Example:
        >>> adapters = get_all_legacy_adapters()
        >>> for name, adapter in adapters.items():
        ...     console_core.register_plugin(adapter)
    """
    adapters = {}
    for name, factory in LEGACY_ADAPTERS.items():
        try:
            adapters[name] = factory(config)
        except Exception as e:
            print(f"Warning: Failed to create adapter for {name}: {e}")

    return adapters


def get_adapter(name: str, config: Optional[Dict[str, Any]] = None) -> Optional[LegacyPluginAdapter]:
    """
    获取特定插件的适配器

    Args:
        name: 插件名称
        config: 插件配置

    Returns:
        适配器实例，如果插件不存在则返回None

    Example:
        >>> adapter = get_adapter("huggingface_integration")
        >>> console_core.register_plugin(adapter)
    """
    factory = LEGACY_ADAPTERS.get(name)
    if factory:
        return factory(config)
    return None
