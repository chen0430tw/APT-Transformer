# APT插件打包系统设计 (APG - APT Plugin Package)

**日期**: 2025-10-26
**目标**: 设计可扩展的插件打包和分发系统，为APT-CLI提供自组织能力
**优先级**: 🔴 高

---

## 📦 设计目标

### 核心目标

1. **插件打包**: 将插件代码、依赖、元数据打包成独立分发单元
2. **插件市场**: 支持插件注册表和版本管理
3. **热加载**: 运行时动态加载/卸载插件
4. **自组织CLI**: CLI命令由插件自动提供和注册
5. **依赖管理**: 自动处理插件间依赖关系
6. **安全隔离**: 插件沙箱执行环境

---

## 🎨 APG包格式设计

### 文件结构

```
plugin-name-1.0.0.apg  (ZIP格式)
├── plugin.yaml          # 插件元数据
├── plugin/              # 插件代码
│   ├── __init__.py
│   ├── main.py         # 插件主逻辑
│   └── utils.py
├── commands/            # CLI命令定义（可选）
│   ├── cmd_foo.py
│   └── cmd_bar.py
├── requirements.txt     # Python依赖（可选）
├── adapters/            # 模型适配器（可选）
│   └── custom_adapter.py
├── tests/               # 单元测试（可选）
│   └── test_plugin.py
└── README.md           # 插件文档
```

### plugin.yaml 元数据规范

```yaml
# 基础信息
name: "route_optimizer"
version: "1.0.0"
display_name: "MoE Route Optimizer"
description: "Optimizes expert routing in Mixture of Experts models"
author: "APT Team"
license: "MIT"
homepage: "https://github.com/apt/plugins/route-optimizer"

# 兼容性
engine: ">=1.0.0"          # 引擎版本要求
python: ">=3.8,<4.0"       # Python版本要求

# 插件能力
capabilities:
  required: ["moe"]        # 必需的模型能力
  optional: []             # 可选的模型能力
  provides: ["routing"]    # 插件提供的能力

# PluginBus配置
plugin_bus:
  priority: 300            # 插件优先级
  category: "training"     # 插件类别
  events:                  # 监听的事件
    - "on_batch_start"
    - "on_batch_end"
  blocking: false          # 是否阻塞事件
  conflicting_plugins: []  # 冲突插件列表

# 依赖
dependencies:
  plugins: []              # 依赖的其他插件
  python_packages:         # Python依赖
    - "numpy>=1.20.0"
    - "scipy>=1.7.0"

# CLI扩展（自组织）
cli:
  enabled: true
  commands:
    - name: "optimize-routes"
      module: "commands.cmd_optimize"
      class: "OptimizeRoutesCommand"
      description: "Optimize MoE routing tables"
      aliases: ["opt-routes"]

# EQI集成
eqi:
  decision_required: false  # 是否需要EQI决策
  evidence_types:           # 收集的证据类型
    - "routing_efficiency"
    - "expert_load_balance"

# 沙箱配置
sandbox:
  enabled: false           # 是否启用沙箱
  permissions:             # 权限列表
    - "read_model"
    - "write_metrics"
```

---

## 🔧 实现组件

### 1. 插件打包器 (Plugin Packager)

**文件**: `apt_model/tools/apg/packager.py`

```python
from pathlib import Path
from typing import Optional, Dict, Any
import zipfile
import yaml
import tempfile
import shutil

class PluginPackager:
    """APG插件打包器"""

    def pack(
        self,
        plugin_dir: Path,
        output: Path,
        manifest: Optional[Dict[str, Any]] = None,
        include_tests: bool = False,
    ) -> Path:
        """
        打包插件目录为APG文件

        Args:
            plugin_dir: 插件源代码目录
            output: 输出APG文件路径
            manifest: 插件元数据（如果None则从plugin.yaml读取）
            include_tests: 是否包含测试文件

        Returns:
            生成的APG文件路径
        """
        # 1. 验证插件目录结构
        self._validate_plugin_dir(plugin_dir)

        # 2. 读取或生成manifest
        if manifest is None:
            manifest_path = plugin_dir / "plugin.yaml"
            if not manifest_path.exists():
                raise ValueError("plugin.yaml not found")
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)

        # 3. 验证manifest
        self._validate_manifest(manifest)

        # 4. 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            # 5. 复制插件文件
            self._copy_plugin_files(
                plugin_dir,
                tmp_root,
                include_tests=include_tests,
            )

            # 6. 写入manifest
            with open(tmp_root / "plugin.yaml", 'w') as f:
                yaml.dump(manifest, f)

            # 7. 创建ZIP包
            output.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in tmp_root.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(tmp_root)
                        zf.write(file_path, arcname)

        return output

    def _validate_plugin_dir(self, plugin_dir: Path):
        """验证插件目录结构"""
        if not plugin_dir.exists():
            raise ValueError(f"Plugin directory not found: {plugin_dir}")

        # 必需文件/目录
        required = ["plugin", "plugin.yaml"]
        for item in required:
            if not (plugin_dir / item).exists():
                raise ValueError(f"Required item not found: {item}")

    def _validate_manifest(self, manifest: Dict[str, Any]):
        """验证manifest完整性"""
        required_fields = ["name", "version", "description", "author"]
        for field in required_fields:
            if field not in manifest:
                raise ValueError(f"Required field missing: {field}")

        # 验证版本格式
        version = manifest["version"]
        if not self._is_valid_version(version):
            raise ValueError(f"Invalid version format: {version}")

    def _is_valid_version(self, version: str) -> bool:
        """验证版本号格式"""
        import re
        return bool(re.match(r'^\d+\.\d+\.\d+$', version))

    def _copy_plugin_files(
        self,
        src: Path,
        dst: Path,
        include_tests: bool,
    ):
        """复制插件文件到目标目录"""
        # 需要复制的目录
        dirs_to_copy = ["plugin", "commands", "adapters"]
        if include_tests:
            dirs_to_copy.append("tests")

        for dir_name in dirs_to_copy:
            src_dir = src / dir_name
            if src_dir.exists():
                shutil.copytree(src_dir, dst / dir_name)

        # 复制单个文件
        files_to_copy = ["plugin.yaml", "requirements.txt", "README.md"]
        for file_name in files_to_copy:
            src_file = src / file_name
            if src_file.exists():
                shutil.copy2(src_file, dst / file_name)
```

---

### 2. 插件加载器 (Plugin Loader)

**文件**: `apt_model/console/plugin_loader.py`

```python
import zipfile
import tempfile
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import logging

logger = logging.getLogger(__name__)

class PluginLoader:
    """APG插件加载器"""

    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Args:
            plugin_dir: 插件安装目录（None则使用默认）
        """
        if plugin_dir is None:
            plugin_dir = Path.home() / ".apt" / "plugins"

        self.plugin_dir = plugin_dir
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_plugins: Dict[str, Any] = {}
        self._temp_dirs: List[Path] = []

    def install(self, apg_path: Path, force: bool = False) -> Dict[str, Any]:
        """
        安装APG插件包

        Args:
            apg_path: APG文件路径
            force: 是否强制覆盖已安装插件

        Returns:
            插件信息字典
        """
        # 1. 解压到临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with zipfile.ZipFile(apg_path, 'r') as zf:
                zf.extractall(tmp_root)

            # 2. 读取manifest
            manifest_path = tmp_root / "plugin.yaml"
            if not manifest_path.exists():
                raise ValueError("Invalid APG: plugin.yaml not found")

            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)

            # 3. 检查是否已安装
            plugin_name = manifest["name"]
            plugin_install_dir = self.plugin_dir / plugin_name

            if plugin_install_dir.exists() and not force:
                raise ValueError(f"Plugin {plugin_name} already installed. Use force=True to overwrite.")

            # 4. 安装到插件目录
            if plugin_install_dir.exists():
                shutil.rmtree(plugin_install_dir)

            shutil.copytree(tmp_root, plugin_install_dir)

            logger.info(f"Plugin {plugin_name} installed to {plugin_install_dir}")

            return manifest

    def load(self, plugin_name: str) -> Any:
        """
        加载已安装的插件

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例
        """
        # 如果已加载，直接返回
        if plugin_name in self._loaded_plugins:
            return self._loaded_plugins[plugin_name]

        # 1. 查找插件目录
        plugin_install_dir = self.plugin_dir / plugin_name
        if not plugin_install_dir.exists():
            raise ValueError(f"Plugin {plugin_name} not installed")

        # 2. 读取manifest
        manifest_path = plugin_install_dir / "plugin.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        # 3. 动态导入插件模块
        plugin_module_path = plugin_install_dir / "plugin" / "__init__.py"
        if not plugin_module_path.exists():
            raise ValueError(f"Plugin module not found: {plugin_module_path}")

        spec = importlib.util.spec_from_file_location(
            f"apt_plugins.{plugin_name}",
            plugin_module_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # 4. 实例化插件
        if not hasattr(module, "Plugin"):
            raise ValueError(f"Plugin class not found in {plugin_name}")

        plugin_instance = module.Plugin()

        # 5. 缓存插件实例
        self._loaded_plugins[plugin_name] = plugin_instance

        logger.info(f"Plugin {plugin_name} loaded successfully")

        return plugin_instance

    def unload(self, plugin_name: str):
        """卸载插件"""
        if plugin_name in self._loaded_plugins:
            # 调用插件清理方法
            plugin = self._loaded_plugins[plugin_name]
            if hasattr(plugin, "cleanup"):
                plugin.cleanup()

            # 从缓存中移除
            del self._loaded_plugins[plugin_name]

            logger.info(f"Plugin {plugin_name} unloaded")

    def list_installed(self) -> List[Dict[str, Any]]:
        """列出已安装的插件"""
        installed = []

        for plugin_dir in self.plugin_dir.iterdir():
            if plugin_dir.is_dir():
                manifest_path = plugin_dir / "plugin.yaml"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = yaml.safe_load(f)
                    installed.append(manifest)

        return installed
```

---

### 3. 插件注册表 (Plugin Registry)

**文件**: `apt_model/console/plugin_registry.py`

```python
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class PluginRegistry:
    """插件注册表 - 管理插件元数据和依赖"""

    def __init__(self, registry_file: Optional[Path] = None):
        """
        Args:
            registry_file: 注册表文件路径
        """
        if registry_file is None:
            registry_file = Path.home() / ".apt" / "plugin_registry.yaml"

        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self):
        """从文件加载注册表"""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self._registry = yaml.safe_load(f) or {}
        else:
            self._registry = {}

    def _save_registry(self):
        """保存注册表到文件"""
        with open(self.registry_file, 'w') as f:
            yaml.dump(self._registry, f)

    def register(self, manifest: Dict[str, Any]):
        """注册插件"""
        plugin_name = manifest["name"]
        plugin_version = manifest["version"]

        # 创建插件条目
        if plugin_name not in self._registry:
            self._registry[plugin_name] = {
                "versions": {},
                "latest": plugin_version,
            }

        # 添加版本信息
        self._registry[plugin_name]["versions"][plugin_version] = {
            "manifest": manifest,
            "installed": True,
            "enabled": True,
        }

        # 更新latest版本
        current_latest = self._registry[plugin_name]["latest"]
        if self._compare_versions(plugin_version, current_latest) > 0:
            self._registry[plugin_name]["latest"] = plugin_version

        self._save_registry()
        logger.info(f"Registered plugin: {plugin_name} v{plugin_version}")

    def unregister(self, plugin_name: str, version: Optional[str] = None):
        """注销插件"""
        if plugin_name not in self._registry:
            raise ValueError(f"Plugin {plugin_name} not registered")

        if version is None:
            # 删除整个插件
            del self._registry[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
        else:
            # 删除特定版本
            if version in self._registry[plugin_name]["versions"]:
                del self._registry[plugin_name]["versions"][version]
                logger.info(f"Unregistered plugin version: {plugin_name} v{version}")

            # 如果没有版本了，删除整个插件
            if not self._registry[plugin_name]["versions"]:
                del self._registry[plugin_name]

        self._save_registry()

    def get_plugin_info(self, plugin_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取插件信息"""
        if plugin_name not in self._registry:
            return None

        if version is None:
            version = self._registry[plugin_name]["latest"]

        return self._registry[plugin_name]["versions"].get(version)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有注册的插件"""
        plugins = []
        for plugin_name, plugin_data in self._registry.items():
            latest_version = plugin_data["latest"]
            latest_info = plugin_data["versions"][latest_version]
            plugins.append({
                "name": plugin_name,
                "version": latest_version,
                "enabled": latest_info.get("enabled", True),
                "manifest": latest_info["manifest"],
            })
        return plugins

    def resolve_dependencies(self, plugin_name: str) -> List[str]:
        """
        解析插件依赖链

        Returns:
            按加载顺序排列的插件列表（依赖在前）
        """
        visited: Set[str] = set()
        order: List[str] = []

        def visit(name: str):
            if name in visited:
                return

            visited.add(name)

            # 获取插件信息
            info = self.get_plugin_info(name)
            if info is None:
                raise ValueError(f"Plugin {name} not found in registry")

            # 递归访问依赖
            deps = info["manifest"].get("dependencies", {}).get("plugins", [])
            for dep in deps:
                visit(dep)

            # 添加到加载顺序
            order.append(name)

        visit(plugin_name)
        return order

    def _compare_versions(self, v1: str, v2: str) -> int:
        """比较版本号 (返回: 1表示v1>v2, 0表示相等, -1表示v1<v2)"""
        from packaging import version
        ver1 = version.parse(v1)
        ver2 = version.parse(v2)

        if ver1 > ver2:
            return 1
        elif ver1 < ver2:
            return -1
        else:
            return 0
```

---

### 4. CLI自组织系统 (Self-Organizing CLI)

**文件**: `apt_model/console/cli_organizer.py`

```python
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class CLIOrganizer:
    """CLI自组织器 - 动态注册插件提供的命令"""

    def __init__(self, plugin_loader: 'PluginLoader'):
        """
        Args:
            plugin_loader: 插件加载器
        """
        self.plugin_loader = plugin_loader
        self._registered_commands: Dict[str, Any] = {}

    def discover_and_register_commands(self):
        """发现并注册所有插件提供的CLI命令"""
        installed_plugins = self.plugin_loader.list_installed()

        for manifest in installed_plugins:
            plugin_name = manifest["name"]

            # 检查插件是否提供CLI扩展
            cli_config = manifest.get("cli", {})
            if not cli_config.get("enabled", False):
                continue

            # 注册每个命令
            commands = cli_config.get("commands", [])
            for cmd_config in commands:
                self._register_command(plugin_name, cmd_config)

    def _register_command(self, plugin_name: str, cmd_config: Dict[str, Any]):
        """注册单个命令"""
        cmd_name = cmd_config["name"]
        cmd_module = cmd_config["module"]
        cmd_class = cmd_config["class"]

        # 动态导入命令模块
        plugin_dir = self.plugin_loader.plugin_dir / plugin_name
        cmd_module_path = plugin_dir / cmd_module.replace('.', '/') / ".py"

        if not cmd_module_path.exists():
            logger.warning(f"Command module not found: {cmd_module_path}")
            return

        spec = importlib.util.spec_from_file_location(
            f"apt_plugins.{plugin_name}.{cmd_module}",
            cmd_module_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # 获取命令类
        if not hasattr(module, cmd_class):
            logger.warning(f"Command class {cmd_class} not found in {cmd_module}")
            return

        command_cls = getattr(module, cmd_class)

        # 注册命令
        self._registered_commands[cmd_name] = {
            "class": command_cls,
            "plugin": plugin_name,
            "description": cmd_config.get("description", ""),
            "aliases": cmd_config.get("aliases", []),
        }

        logger.info(f"Registered command: {cmd_name} (from {plugin_name})")

    def get_command(self, cmd_name: str) -> Optional[Any]:
        """获取命令类"""
        if cmd_name in self._registered_commands:
            return self._registered_commands[cmd_name]["class"]

        # 检查别名
        for name, info in self._registered_commands.items():
            if cmd_name in info["aliases"]:
                return info["class"]

        return None

    def list_commands(self) -> List[Dict[str, Any]]:
        """列出所有注册的命令"""
        return [
            {
                "name": name,
                "plugin": info["plugin"],
                "description": info["description"],
                "aliases": info["aliases"],
            }
            for name, info in self._registered_commands.items()
        ]
```

---

## 🚀 使用示例

### 1. 打包插件

```python
from apt_model.tools.apg.packager import PluginPackager

packager = PluginPackager()

# 打包插件
apg_path = packager.pack(
    plugin_dir=Path("my_plugin"),
    output=Path("dist/my_plugin-1.0.0.apg"),
    include_tests=False,
)

print(f"Plugin packaged: {apg_path}")
```

### 2. 安装和加载插件

```python
from apt_model.console.plugin_loader import PluginLoader
from apt_model.console.plugin_registry import PluginRegistry

# 初始化加载器和注册表
loader = PluginLoader()
registry = PluginRegistry()

# 安装插件
manifest = loader.install(Path("my_plugin-1.0.0.apg"))
registry.register(manifest)

# 解析依赖并加载
load_order = registry.resolve_dependencies("my_plugin")
for plugin_name in load_order:
    plugin = loader.load(plugin_name)
    print(f"Loaded: {plugin_name}")
```

### 3. 自组织CLI

```python
from apt_model.console.cli_organizer import CLIOrganizer

# 初始化CLI组织器
cli_org = CLIOrganizer(loader)

# 自动发现并注册插件命令
cli_org.discover_and_register_commands()

# 列出所有命令
commands = cli_org.list_commands()
for cmd in commands:
    print(f"{cmd['name']}: {cmd['description']} (from {cmd['plugin']})")

# 执行命令
cmd_class = cli_org.get_command("optimize-routes")
if cmd_class:
    cmd = cmd_class()
    cmd.execute()
```

---

## 🔄 与现有系统集成

### 集成到Console Core

```python
# apt_model/console/core.py

from apt_model.console.plugin_loader import PluginLoader
from apt_model.console.plugin_registry import PluginRegistry
from apt_model.console.cli_organizer import CLIOrganizer

class ConsoleCore:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... 现有代码

        # 插件系统
        self.plugin_loader = PluginLoader()
        self.plugin_registry = PluginRegistry()
        self.cli_organizer = CLIOrganizer(self.plugin_loader)

        # 自动发现CLI命令
        self.cli_organizer.discover_and_register_commands()

    def install_plugin(self, apg_path: Path, auto_load: bool = True):
        """安装插件包"""
        manifest = self.plugin_loader.install(apg_path)
        self.plugin_registry.register(manifest)

        if auto_load:
            plugin = self.plugin_loader.load(manifest["name"])
            self.register_plugin(plugin)

        # 重新发现CLI命令
        self.cli_organizer.discover_and_register_commands()

    def get_cli_command(self, cmd_name: str):
        """获取CLI命令"""
        return self.cli_organizer.get_command(cmd_name)
```

---

## 📋 实施清单

### 新增文件

| 文件 | 描述 | 行数估算 |
|------|------|----------|
| `apt_model/tools/apg/packager.py` | 插件打包器 | ~200 |
| `apt_model/console/plugin_loader.py` | 插件加载器 | ~250 |
| `apt_model/console/plugin_registry.py` | 插件注册表 | ~200 |
| `apt_model/console/cli_organizer.py` | CLI自组织器 | ~150 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `apt_model/console/core.py` | 集成插件系统和CLI组织器 |

---

## ✅ 设计完成

这套系统提供：
- ✅ 完整的插件打包和分发机制
- ✅ 动态插件加载和卸载
- ✅ 插件依赖解析
- ✅ CLI自组织能力
- ✅ 向后兼容现有系统
- ✅ 为未来扩展预留接口
