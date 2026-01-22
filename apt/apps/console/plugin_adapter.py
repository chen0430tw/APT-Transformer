"""
插件适配器

将现有独立插件适配到新的PluginBase系统
"""

from typing import Dict, Any, List, Optional, Callable
import logging

from apt.apps.console.plugin_standards import PluginBase, PluginManifest

logger = logging.getLogger(__name__)


class LegacyPluginAdapter(PluginBase):
    """
    遗留插件适配器

    将现有的独立插件类包装为符合PluginBase的插件
    这允许旧插件在新的PluginBus系统中运行，无需完全重写
    """

    def __init__(
        self,
        legacy_plugin: Any,
        name: str,
        priority: int,
        events: List[str],
        category: str = "legacy",
        blocking: bool = False,
        **manifest_kwargs
    ):
        """
        初始化适配器

        Args:
            legacy_plugin: 原有插件实例
            name: 插件名称
            priority: 优先级（0-999）
            events: 监听的事件列表
            category: 插件类别
            blocking: 是否阻塞事件
            **manifest_kwargs: 其他manifest参数（如capabilities等）
        """
        super().__init__()
        self.legacy_plugin = legacy_plugin
        self._name = name
        self._priority = priority
        self._events = events
        self._category = category
        self._blocking = blocking
        self._manifest_kwargs = manifest_kwargs

        # 事件方法映射：新事件名 -> 旧插件方法名
        self._method_mapping = {
            'on_init': 'on_init',
            'on_batch_start': 'on_batch_start',
            'on_batch_end': 'on_batch_end',
            'on_step_start': 'on_step_start',
            'on_step_end': 'on_step_end',
            'on_decode': 'on_decode',
            'on_shutdown': 'on_shutdown',
        }

        logger.info(f"LegacyPluginAdapter created for {name}")

    def get_manifest(self) -> PluginManifest:
        """返回插件manifest"""
        return PluginManifest(
            name=self._name,
            priority=self._priority,
            events=self._events,
            category=self._category,
            blocking=self._blocking,
            **self._manifest_kwargs
        )

    def _call_legacy_method(
        self,
        method_name: str,
        context: Dict[str, Any],
        fallback_method: Optional[str] = None
    ) -> Any:
        """
        调用遗留插件的方法

        Args:
            method_name: 要调用的方法名
            context: 事件上下文
            fallback_method: 如果主方法不存在，尝试的备选方法名

        Returns:
            方法执行结果
        """
        # 首先尝试主方法
        if hasattr(self.legacy_plugin, method_name):
            method = getattr(self.legacy_plugin, method_name)
            try:
                # 检查方法签名，有些旧插件可能不接受参数
                import inspect
                sig = inspect.signature(method)

                if len(sig.parameters) == 0:
                    # 无参数方法
                    return method()
                else:
                    # 带参数方法
                    return method(context)

            except Exception as e:
                logger.error(f"Error calling {self._name}.{method_name}: {e}")
                # 不重新抛出异常，让其他插件继续执行
                return None

        # 尝试备选方法
        elif fallback_method and hasattr(self.legacy_plugin, fallback_method):
            logger.debug(f"Using fallback method {fallback_method} for {method_name}")
            method = getattr(self.legacy_plugin, fallback_method)
            try:
                return method(context)
            except Exception as e:
                logger.error(f"Error calling {self._name}.{fallback_method}: {e}")
                return None

        else:
            # 方法不存在，静默忽略（旧插件可能不实现所有事件）
            logger.debug(f"Legacy plugin {self._name} has no method {method_name}")
            return None

    # ========================================================================
    # 实现所有PluginBase事件方法
    # ========================================================================

    def on_init(self, context: Dict[str, Any]):
        """初始化事件"""
        self._call_legacy_method('on_init', context, fallback_method='initialize')

    def on_batch_start(self, context: Dict[str, Any]):
        """批次开始事件"""
        self._call_legacy_method('on_batch_start', context, fallback_method='before_batch')

    def on_batch_end(self, context: Dict[str, Any]):
        """批次结束事件"""
        self._call_legacy_method('on_batch_end', context, fallback_method='after_batch')

    def on_step_start(self, context: Dict[str, Any]):
        """步骤开始事件"""
        self._call_legacy_method('on_step_start', context, fallback_method='before_step')

    def on_step_end(self, context: Dict[str, Any]):
        """步骤结束事件"""
        self._call_legacy_method('on_step_end', context, fallback_method='after_step')

    def on_decode(self, context: Dict[str, Any]):
        """解码事件"""
        self._call_legacy_method('on_decode', context, fallback_method='on_generation')

    def on_shutdown(self, context: Dict[str, Any]):
        """关闭事件"""
        self._call_legacy_method('on_shutdown', context, fallback_method='cleanup')

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def set_method_mapping(self, mapping: Dict[str, str]):
        """
        自定义事件方法映射

        Args:
            mapping: 事件名到旧插件方法名的映射

        Example:
            >>> adapter.set_method_mapping({
            ...     'on_init': 'setup',
            ...     'on_shutdown': 'teardown'
            ... })
        """
        self._method_mapping.update(mapping)

    def get_legacy_plugin(self) -> Any:
        """
        获取原始遗留插件实例

        Returns:
            原始插件对象
        """
        return self.legacy_plugin

    def call_custom_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        调用遗留插件的自定义方法（非事件方法）

        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            方法执行结果

        Example:
            >>> adapter.call_custom_method('export_to_huggingface', model=model)
        """
        if hasattr(self.legacy_plugin, method_name):
            method = getattr(self.legacy_plugin, method_name)
            try:
                return method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error calling custom method {method_name}: {e}")
                raise
        else:
            raise AttributeError(
                f"Legacy plugin {self._name} has no method {method_name}"
            )

    def __repr__(self) -> str:
        return (
            f"LegacyPluginAdapter(name={self._name}, "
            f"priority={self._priority}, "
            f"legacy_class={type(self.legacy_plugin).__name__})"
        )


def create_adapter(
    legacy_plugin: Any,
    name: str,
    priority: int,
    events: List[str],
    **kwargs
) -> LegacyPluginAdapter:
    """
    便捷函数：创建遗留插件适配器

    Args:
        legacy_plugin: 原有插件实例
        name: 插件名称
        priority: 优先级
        events: 事件列表
        **kwargs: 其他manifest参数

    Returns:
        适配器实例

    Example:
        >>> from old_plugins import HuggingFacePlugin
        >>> hf = HuggingFacePlugin()
        >>> adapter = create_adapter(
        ...     legacy_plugin=hf,
        ...     name="huggingface_integration",
        ...     priority=700,
        ...     events=["on_init", "on_shutdown"]
        ... )
    """
    return LegacyPluginAdapter(
        legacy_plugin=legacy_plugin,
        name=name,
        priority=priority,
        events=events,
        **kwargs
    )
