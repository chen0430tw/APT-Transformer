"""
Legacy Plugins Package

包含适配后的旧版插件，通过LegacyPluginAdapter包装以兼容新的PluginBase系统
"""

from apt.apps.console.apt.apps.plugins.adapters import (
    create_huggingface_adapter,
    create_cloud_storage_adapter,
    create_ollama_export_adapter,
    create_distillation_adapter,
    create_pruning_adapter,
    create_multimodal_adapter,
    create_data_processors_adapter,
    create_debugging_adapter,
)

__all__ = [
    'create_huggingface_adapter',
    'create_cloud_storage_adapter',
    'create_ollama_export_adapter',
    'create_distillation_adapter',
    'create_pruning_adapter',
    'create_multimodal_adapter',
    'create_data_processors_adapter',
    'create_debugging_adapter',
]
