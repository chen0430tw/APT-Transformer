"""
CLI自组织器

动态发现和注册插件提供的CLI命令，实现自组织能力
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Callable
import logging

logger = logging.getLogger(__name__)


class CommandDescriptor:
    """CLI命令描述符"""

    def __init__(
        self,
        name: str,
        command_class: Type,
        plugin_name: str,
        description: str = "",
        aliases: Optional[List[str]] = None,
        module_path: Optional[Path] = None,
    ):
        """
        Args:
            name: 命令名称
            command_class: 命令类
            plugin_name: 提供该命令的插件名称
            description: 命令描述
            aliases: 命令别名列表
            module_path: 命令模块路径
        """
        self.name = name
        self.command_class = command_class
        self.plugin_name = plugin_name
        self.description = description
        self.aliases = aliases or []
        self.module_path = module_path

    def execute(self, *args, **kwargs) -> Any:
        """
        执行命令

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            命令执行结果
        """
        try:
            cmd_instance = self.command_class()
            if hasattr(cmd_instance, "execute"):
                return cmd_instance.execute(*args, **kwargs)
            elif hasattr(cmd_instance, "run"):
                return cmd_instance.run(*args, **kwargs)
            elif callable(cmd_instance):
                return cmd_instance(*args, **kwargs)
            else:
                raise ValueError(
                    f"Command class {self.command_class.__name__} "
                    f"must have execute() or run() method"
                )
        except Exception as e:
            logger.error(f"Error executing command {self.name}: {e}")
            raise

    def __repr__(self) -> str:
        return (
            f"CommandDescriptor(name={self.name}, "
            f"plugin={self.plugin_name}, "
            f"aliases={self.aliases})"
        )


class CLIOrganizer:
    """CLI自组织器 - 动态注册插件提供的命令"""

    def __init__(self, plugin_loader: 'PluginLoader'):
        """
        Args:
            plugin_loader: 插件加载器实例
        """
        self.plugin_loader = plugin_loader
        self._registered_commands: Dict[str, CommandDescriptor] = {}
        self._command_aliases: Dict[str, str] = {}  # alias -> command_name

        logger.debug("CLIOrganizer initialized")

    def discover_and_register_commands(self, plugin_names: Optional[List[str]] = None):
        """
        发现并注册插件提供的CLI命令

        Args:
            plugin_names: 要扫描的插件名称列表（None则扫描所有已安装插件）
        """
        if plugin_names is None:
            # 扫描所有已安装的插件
            installed_plugins = self.plugin_loader.list_installed()
            plugin_names = [manifest["name"] for manifest in installed_plugins]

        logger.info(f"Discovering commands from {len(plugin_names)} plugins")

        for plugin_name in plugin_names:
            try:
                self._discover_plugin_commands(plugin_name)
            except Exception as e:
                logger.error(f"Error discovering commands from {plugin_name}: {e}")

    def _discover_plugin_commands(self, plugin_name: str):
        """
        从单个插件发现命令

        Args:
            plugin_name: 插件名称
        """
        # 获取插件安装目录
        plugin_dir = self.plugin_loader.plugin_dir / plugin_name
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return

        # 读取manifest
        manifest_path = plugin_dir / "plugin.yaml"
        if not manifest_path.exists():
            logger.warning(f"Manifest not found for plugin: {plugin_name}")
            return

        import yaml
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)

        # 检查插件是否提供CLI扩展
        cli_config = manifest.get("cli", {})
        if not cli_config.get("enabled", False):
            logger.debug(f"Plugin {plugin_name} does not provide CLI commands")
            return

        # 注册每个命令
        commands = cli_config.get("commands", [])
        for cmd_config in commands:
            try:
                self._register_command(plugin_name, plugin_dir, cmd_config)
            except Exception as e:
                logger.error(
                    f"Error registering command {cmd_config.get('name', '?')} "
                    f"from {plugin_name}: {e}"
                )

    def _register_command(
        self,
        plugin_name: str,
        plugin_dir: Path,
        cmd_config: Dict[str, Any]
    ):
        """
        注册单个命令

        Args:
            plugin_name: 插件名称
            plugin_dir: 插件目录
            cmd_config: 命令配置字典
        """
        cmd_name = cmd_config.get("name")
        cmd_module = cmd_config.get("module")
        cmd_class_name = cmd_config.get("class")
        cmd_description = cmd_config.get("description", "")
        cmd_aliases = cmd_config.get("aliases", [])

        if not all([cmd_name, cmd_module, cmd_class_name]):
            raise ValueError(
                f"Command config missing required fields: name, module, or class"
            )

        # 检查命令名是否已被注册
        if cmd_name in self._registered_commands:
            existing = self._registered_commands[cmd_name]
            logger.warning(
                f"Command {cmd_name} already registered by {existing.plugin_name}, "
                f"skipping registration from {plugin_name}"
            )
            return

        # 动态导入命令模块
        # cmd_module格式: "commands.cmd_optimize"
        module_rel_path = cmd_module.replace('.', '/')
        cmd_module_path = plugin_dir / f"{module_rel_path}.py"

        if not cmd_module_path.exists():
            raise ValueError(f"Command module not found: {cmd_module_path}")

        module_name = f"apt_plugins.{plugin_name}.{cmd_module}"

        spec = importlib.util.spec_from_file_location(
            module_name,
            cmd_module_path,
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to create module spec for {cmd_module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 获取命令类
        if not hasattr(module, cmd_class_name):
            raise ValueError(
                f"Command class {cmd_class_name} not found in {cmd_module}"
            )

        command_cls = getattr(module, cmd_class_name)

        # 创建命令描述符
        descriptor = CommandDescriptor(
            name=cmd_name,
            command_class=command_cls,
            plugin_name=plugin_name,
            description=cmd_description,
            aliases=cmd_aliases,
            module_path=cmd_module_path,
        )

        # 注册命令
        self._registered_commands[cmd_name] = descriptor

        # 注册别名
        for alias in cmd_aliases:
            if alias in self._command_aliases:
                logger.warning(
                    f"Command alias {alias} already registered, skipping"
                )
            else:
                self._command_aliases[alias] = cmd_name

        logger.info(
            f"Registered command: {cmd_name} "
            f"(aliases: {cmd_aliases}) from {plugin_name}"
        )

    def get_command(self, cmd_name: str) -> Optional[CommandDescriptor]:
        """
        获取命令描述符

        Args:
            cmd_name: 命令名称或别名

        Returns:
            命令描述符，如果不存在则返回None
        """
        # 先查找命令名
        if cmd_name in self._registered_commands:
            return self._registered_commands[cmd_name]

        # 再查找别名
        if cmd_name in self._command_aliases:
            actual_name = self._command_aliases[cmd_name]
            return self._registered_commands.get(actual_name)

        return None

    def execute_command(self, cmd_name: str, *args, **kwargs) -> Any:
        """
        执行命令

        Args:
            cmd_name: 命令名称或别名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            命令执行结果

        Raises:
            ValueError: 如果命令不存在
        """
        descriptor = self.get_command(cmd_name)
        if descriptor is None:
            raise ValueError(
                f"Command {cmd_name} not found. "
                f"Use list_commands() to see available commands."
            )

        return descriptor.execute(*args, **kwargs)

    def list_commands(self, plugin_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出所有注册的命令

        Args:
            plugin_name: 过滤特定插件的命令（None则列出所有）

        Returns:
            命令信息列表
        """
        commands = []

        for name, descriptor in self._registered_commands.items():
            if plugin_name is not None and descriptor.plugin_name != plugin_name:
                continue

            commands.append({
                "name": name,
                "plugin": descriptor.plugin_name,
                "description": descriptor.description,
                "aliases": descriptor.aliases,
            })

        return commands

    def unregister_plugin_commands(self, plugin_name: str):
        """
        注销某个插件的所有命令

        Args:
            plugin_name: 插件名称
        """
        # 找到该插件的所有命令
        commands_to_remove = [
            name for name, descriptor in self._registered_commands.items()
            if descriptor.plugin_name == plugin_name
        ]

        # 移除命令
        for cmd_name in commands_to_remove:
            descriptor = self._registered_commands[cmd_name]

            # 移除别名
            for alias in descriptor.aliases:
                if alias in self._command_aliases:
                    del self._command_aliases[alias]

            # 移除命令
            del self._registered_commands[cmd_name]

        logger.info(
            f"Unregistered {len(commands_to_remove)} commands from {plugin_name}"
        )

    def reload_plugin_commands(self, plugin_name: str):
        """
        重新加载某个插件的命令

        Args:
            plugin_name: 插件名称
        """
        # 先注销旧命令
        self.unregister_plugin_commands(plugin_name)

        # 重新发现命令
        try:
            self._discover_plugin_commands(plugin_name)
            logger.info(f"Reloaded commands from {plugin_name}")
        except Exception as e:
            logger.error(f"Error reloading commands from {plugin_name}: {e}")
            raise

    def has_command(self, cmd_name: str) -> bool:
        """
        检查命令是否存在

        Args:
            cmd_name: 命令名称或别名

        Returns:
            是否存在
        """
        return self.get_command(cmd_name) is not None

    def get_command_help(self, cmd_name: str) -> Optional[str]:
        """
        获取命令帮助信息

        Args:
            cmd_name: 命令名称或别名

        Returns:
            帮助信息，如果命令不存在则返回None
        """
        descriptor = self.get_command(cmd_name)
        if descriptor is None:
            return None

        help_text = f"Command: {descriptor.name}\n"
        help_text += f"Plugin: {descriptor.plugin_name}\n"

        if descriptor.description:
            help_text += f"Description: {descriptor.description}\n"

        if descriptor.aliases:
            help_text += f"Aliases: {', '.join(descriptor.aliases)}\n"

        # 尝试获取命令类的文档字符串
        if descriptor.command_class.__doc__:
            help_text += f"\n{descriptor.command_class.__doc__}"

        return help_text

    def clear(self):
        """清空所有注册的命令"""
        self._registered_commands.clear()
        self._command_aliases.clear()
        logger.info("All commands cleared")
