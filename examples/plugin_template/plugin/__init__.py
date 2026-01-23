"""
Example Plugin

这是一个示例插件，展示了如何创建APT插件
"""

from typing import Dict, Any
import logging

from apt.apps.console.plugin_standards import PluginBase, PluginManifest

logger = logging.getLogger(__name__)


class Plugin(PluginBase):
    """
    示例插件类

    必须命名为 'Plugin' 并继承自 PluginBase
    """

    def __init__(self):
        """初始化插件"""
        super().__init__()
        self.counter = 0
        logger.info("Example plugin initialized")

    def get_manifest(self) -> PluginManifest:
        """
        返回插件清单

        这个方法是必需的，用于向PluginBus提供插件元数据
        """
        return PluginManifest(
            name="example_plugin",
            priority=350,
            category="training",
            events=["on_init", "on_batch_start", "on_batch_end"],
            blocking=False,
            description="An example plugin",

            # 能力相关
            required_capabilities=[],
            optional_capabilities=[],
            provides_capabilities=["example"],

            # 版本要求
            engine=">=1.0.0",
        )

    # ========================================================================
    # 事件处理方法
    # ========================================================================

    def on_init(self, context: Dict[str, Any]):
        """
        初始化事件处理

        Args:
            context: 事件上下文，包含模型、配置等信息
        """
        logger.info("Example plugin: on_init called")
        logger.info(f"Context keys: {list(context.keys())}")

        # 可以访问模型、配置等
        # model = context.get('model')
        # config = context.get('config')

    def on_batch_start(self, context: Dict[str, Any]):
        """
        批次开始事件处理

        Args:
            context: 事件上下文
        """
        self.counter += 1

        # 获取批次信息
        batch_idx = context.get('batch_idx', 0)
        logger.debug(f"Example plugin: batch {batch_idx} started (total: {self.counter})")

        # 可以修改上下文数据
        context['example_plugin_data'] = {
            'counter': self.counter,
            'timestamp': self._get_timestamp(),
        }

    def on_batch_end(self, context: Dict[str, Any]):
        """
        批次结束事件处理

        Args:
            context: 事件上下文
        """
        batch_idx = context.get('batch_idx', 0)
        logger.debug(f"Example plugin: batch {batch_idx} ended")

        # 可以记录指标
        # metrics = context.get('metrics', {})
        # self._log_metrics(metrics)

    def on_step_start(self, context: Dict[str, Any]):
        """步骤开始事件（可选）"""
        pass

    def on_step_end(self, context: Dict[str, Any]):
        """步骤结束事件（可选）"""
        pass

    def on_decode(self, context: Dict[str, Any]):
        """解码事件（可选）"""
        pass

    def on_shutdown(self, context: Dict[str, Any]):
        """
        关闭事件处理

        Args:
            context: 事件上下文
        """
        logger.info(f"Example plugin shutting down (processed {self.counter} batches)")

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()

    def cleanup(self):
        """
        清理资源（可选）

        当插件被卸载时会调用此方法
        """
        logger.info("Example plugin cleanup")
        self.counter = 0


# 插件版本
__version__ = "1.0.0"
