# APT插件系统完整指南

**版本**: 2.0
**日期**: 2025-10-26
**状态**: ✅ 生产就绪

---

## 📋 目录

1. [系统概述](#系统概述)
2. [核心架构](#核心架构)
3. [快速开始](#快速开始)
4. [插件开发](#插件开发)
5. [插件打包](#插件打包)
6. [CLI自组织](#cli自组织)
7. [API参考](#api参考)
8. [最佳实践](#最佳实践)

---

## 系统概述

APT插件系统是一个完整的、可扩展的插件架构，提供：

### 核心特性

✅ **插件生命周期管理**
- 动态加载/卸载插件
- 版本兼容性检查
- 依赖关系解析
- 冲突检测和隔离

✅ **APG插件包格式**
- 标准化的插件打包格式
- 包含代码、元数据、依赖和命令
- 支持版本管理和分发

✅ **CLI自组织**
- 插件自动注册CLI命令
- 动态命令发现和加载
- 命令别名和帮助系统

✅ **能力驱动**
- 基于模型能力自动加载插件
- APX集成，自动检测和配置
- EQI智能决策支持

✅ **事件系统**
- 插件监听训练/推理事件
- 优先级调度和超时控制
- 阻塞/非阻塞事件支持

---

## 核心架构

### 组件图

```
┌─────────────────────────────────────────────────────────────┐
│                      Console Core                           │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Plugin Loader │  │   Plugin     │  │  CLI Organizer  │ │
│  │   (APG)       │  │   Registry   │  │ (Self-Org)      │ │
│  └───────┬───────┘  └──────┬───────┘  └────────┬────────┘ │
│          │                 │                   │          │
│  ┌───────▼─────────────────▼───────────────────▼────────┐ │
│  │            Plugin Management Layer                    │ │
│  └───────────────────────────┬───────────────────────────┘ │
│                              │                             │
│  ┌───────────────────────────▼───────────────────────────┐ │
│  │                  Plugin Bus                           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │ │
│  │  │ Plugin A │  │ Plugin B │  │ Plugin C │  ...      │ │
│  │  └──────────┘  └──────────┘  └──────────┘           │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┤
│  │               EQI Manager (Optional)                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

      ▲                    ▲                    ▲
      │                    │                    │
 APX Loader          Auto Loader         Version Checker
```

### 核心组件

#### 1. PluginLoader (apt_model/console/plugin_loader.py)
- 安装/卸载APG插件包
- 动态加载插件模块
- 插件实例缓存和热加载

#### 2. PluginRegistry (apt_model/console/plugin_registry.py)
- 插件元数据注册表
- 版本管理和比较
- 依赖关系解析
- 冲突检测

#### 3. CLIOrganizer (apt_model/console/cli_organizer.py)
- 自动发现插件命令
- 动态注册CLI扩展
- 命令别名和帮助系统

#### 4. PluginPackager (apt_model/tools/apg/packager.py)
- 打包插件为APG格式
- 验证插件结构和元数据
- 解包和manifest读取

#### 5. ConsoleCore Integration
- 统一的插件管理API
- APX模型自动配置
- 能力驱动的插件加载

---

## 快速开始

### 1. 安装插件包

```python
from pathlib import Path
from apt_model.console.core import ConsoleCore

# 初始化控制台
core = ConsoleCore()

# 安装插件包
manifest = core.install_plugin_package(
    apg_path=Path("plugins/route_optimizer-1.0.0.apg"),
    auto_load=True,              # 自动加载插件
    auto_register_commands=True  # 自动注册CLI命令
)

print(f"Installed: {manifest['name']} v{manifest['version']}")
```

### 2. 列出已安装插件

```python
# 列出所有插件
plugins = core.list_plugin_packages()
for plugin in plugins:
    print(f"{plugin['name']} v{plugin['version']} - {plugin['manifest']['description']}")

# 列出插件提供的命令
commands = core.list_plugin_commands()
for cmd in commands:
    print(f"{cmd['name']}: {cmd['description']}")
```

### 3. 执行插件命令

```python
# 执行插件提供的命令
result = core.execute_plugin_command(
    "optimize-routes",
    model_path="/path/to/model",
    verbose=True
)

# 获取命令帮助
help_text = core.get_plugin_command_help("optimize-routes")
print(help_text)
```

### 4. APX自动配置

```python
# 加载APX模型，自动配置插件
apx_info = core.load_apx_model(
    apx_path=Path("models/mixtral-8x7b.apx"),
    auto_configure_plugins=True
)

# 系统自动检测到moe能力，加载route_optimizer插件
print(f"Capabilities: {apx_info['capabilities']}")
print(f"Auto-loaded plugins: {apx_info['loaded_plugins']}")
```

---

## 插件开发

### 插件结构

```
my_plugin/
├── plugin.yaml              # 元数据（必需）
├── plugin/                  # 代码目录（必需）
│   └── __init__.py         # 主类（必需，包含Plugin类）
├── commands/                # CLI命令（可选）
│   └── my_command.py
├── tests/                   # 测试（推荐）
│   └── test_plugin.py
├── requirements.txt         # Python依赖（可选）
└── README.md               # 文档（推荐）
```

### 最小插件示例

**plugin.yaml**:
```yaml
name: "my_plugin"
version: "1.0.0"
description: "My awesome plugin"
author: "Your Name"
engine: ">=1.0.0"

plugin_bus:
  priority: 350
  category: "training"
  events:
    - "on_batch_start"
```

**plugin/__init__.py**:
```python
from apt_model.console.plugin_standards import PluginBase, PluginManifest

class Plugin(PluginBase):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="my_plugin",
            priority=350,
            events=["on_batch_start"],
            category="training",
        )

    def on_batch_start(self, context):
        print(f"Batch {context['batch_idx']} starting!")
```

### 添加CLI命令

**plugin.yaml**:
```yaml
cli:
  enabled: true
  commands:
    - name: "my-command"
      module: "commands.my_command"
      class: "MyCommand"
      description: "My custom command"
      aliases: ["mc"]
```

**commands/my_command.py**:
```python
class MyCommand:
    def execute(self, *args, **kwargs):
        message = kwargs.get('message', 'Hello!')
        print(f"Command executed: {message}")
        return {"status": "success"}
```

### 能力驱动插件

```yaml
capabilities:
  required: ["moe"]         # 仅MoE模型使用
  optional: ["tva"]         # TVA模型可选增强
  provides: ["routing"]     # 提供routing能力
```

```python
class Plugin(PluginBase):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="route_optimizer",
            required_capabilities=["moe"],
            optional_capabilities=["tva"],
            provides_capabilities=["routing"],
            # ...
        )
```

---

## 插件打包

### 手动打包

```python
from pathlib import Path
from apt_model.tools.apg.packager import PluginPackager

packager = PluginPackager()

apg_path = packager.pack(
    plugin_dir=Path("my_plugin"),
    output=Path("dist/my_plugin-1.0.0.apg"),
    include_tests=False,
    validate=True,
)

print(f"Packaged: {apg_path} ({apg_path.stat().st_size:,} bytes)")
```

### 便捷函数

```python
from apt_model.tools.apg.packager import pack_plugin

pack_plugin(
    src_dir=Path("my_plugin"),
    out_apg=Path("dist/my_plugin-1.0.0.apg"),
    name="my_plugin",
    version="1.0.0",
    description="My plugin",
    author="Your Name",
    license="MIT",
)
```

### 解包检查

```python
# 读取manifest（不解压）
manifest = packager.get_manifest(Path("my_plugin-1.0.0.apg"))
print(f"Name: {manifest['name']}")
print(f"Version: {manifest['version']}")

# 完全解包
manifest = packager.unpack(
    apg_path=Path("my_plugin-1.0.0.apg"),
    output_dir=Path("extracted")
)
```

---

## CLI自组织

### 自动命令发现

CLI自组织器在启动时自动扫描所有已安装插件，注册它们提供的命令：

```python
# Console启动时自动执行
core = ConsoleCore()
core.start()  # 自动发现并注册所有插件命令

# 手动触发发现
core.cli_organizer.discover_and_register_commands()
```

### 命令执行

```python
# 方法1：通过Console Core
result = core.execute_plugin_command("my-command", arg1="value1")

# 方法2：通过CLI Organizer
result = core.cli_organizer.execute_command("my-command", arg1="value1")

# 方法3：通过命令描述符
descriptor = core.cli_organizer.get_command("my-command")
if descriptor:
    result = descriptor.execute(arg1="value1")
```

### 命令别名

```yaml
cli:
  commands:
    - name: "optimize-routes"
      aliases: ["opt-routes", "or"]
      # ...
```

```python
# 所有这些都有效
core.execute_plugin_command("optimize-routes")
core.execute_plugin_command("opt-routes")
core.execute_plugin_command("or")
```

### 热加载命令

```python
# 安装新插件后，重新发现命令
core.install_plugin_package(Path("new_plugin.apg"))
# 命令自动注册

# 或手动重新加载特定插件的命令
core.cli_organizer.reload_plugin_commands("my_plugin")
```

---

## API参考

### ConsoleCore插件管理

```python
class ConsoleCore:
    # APG包管理
    def install_plugin_package(apg_path, force=False, auto_load=True) -> Dict
    def uninstall_plugin_package(plugin_name: str)
    def list_plugin_packages(enabled_only=False) -> List[Dict]
    def enable_plugin_package(plugin_name: str, version=None)
    def disable_plugin_package(plugin_name: str, version=None)

    # CLI命令管理
    def execute_plugin_command(cmd_name: str, *args, **kwargs) -> Any
    def list_plugin_commands(plugin_name=None) -> List[Dict]
    def get_plugin_command_help(cmd_name: str) -> str

    # APX集成
    def load_apx_model(apx_path, auto_configure_plugins=True) -> Dict
    def analyze_model_for_plugins(model_path=None, capabilities=None) -> Dict
    def get_plugin_recommendations(capabilities: List[str]) -> str

    # PluginBus
    def register_plugin(plugin: PluginBase, manifest=None)
    def compile_plugins(fail_fast=False)
    def emit_event(event: str, step: int, context_data=None) -> EventContext
```

### PluginLoader

```python
class PluginLoader:
    def __init__(plugin_dir: Path = None)

    def install(apg_path: Path, force=False, validate=True) -> Dict
    def uninstall(plugin_name: str)
    def load(plugin_name: str, reload_if_loaded=False) -> PluginBase
    def unload(plugin_name: str)

    def list_installed() -> List[Dict]
    def list_loaded() -> List[str]
    def get_loaded_plugin(plugin_name: str) -> PluginBase
```

### PluginRegistry

```python
class PluginRegistry:
    def __init__(registry_file: Path = None)

    def register(manifest: Dict, enabled=True)
    def unregister(plugin_name: str, version=None)

    def get_plugin_info(plugin_name: str, version=None) -> Dict
    def get_manifest(plugin_name: str, version=None) -> Dict

    def list_plugins(enabled_only=False) -> List[Dict]
    def is_enabled(plugin_name: str, version=None) -> bool
    def set_enabled(plugin_name: str, enabled: bool, version=None)

    def resolve_dependencies(plugin_name: str, version=None) -> List[Tuple]
    def check_conflicts(plugin_name: str, loaded_plugins: List) -> List[str]
```

### CLIOrganizer

```python
class CLIOrganizer:
    def __init__(plugin_loader: PluginLoader)

    def discover_and_register_commands(plugin_names=None)
    def get_command(cmd_name: str) -> CommandDescriptor
    def execute_command(cmd_name: str, *args, **kwargs) -> Any

    def list_commands(plugin_name=None) -> List[Dict]
    def has_command(cmd_name: str) -> bool
    def get_command_help(cmd_name: str) -> str

    def unregister_plugin_commands(plugin_name: str)
    def reload_plugin_commands(plugin_name: str)
```

### PluginPackager

```python
class PluginPackager:
    def pack(plugin_dir, output, manifest=None, include_tests=False) -> Path
    def unpack(apg_path, output_dir) -> Dict
    def get_manifest(apg_path) -> Dict

# 便捷函数
def pack_plugin(src_dir, out_apg, name, version, description, author, **kwargs) -> Path
```

---

## 最佳实践

### 1. 插件开发

✅ **DO**:
- 使用清晰、描述性的插件名称
- 遵循语义化版本规范
- 提供完整的README和docstrings
- 编写单元测试
- 妥善处理异常
- 实现cleanup方法
- 使用logging记录关键操作

❌ **DON'T**:
- 不要在事件处理中执行长时间操作（除非blocking=true）
- 不要假设特定的执行顺序（除非声明依赖）
- 不要直接修改全局状态
- 不要忽略版本兼容性

### 2. 性能考虑

```python
# ✅ 好的实践：快速的非阻塞事件
def on_batch_start(self, context):
    # 收集轻量级指标
    self.metrics.append(context['batch_idx'])

# ❌ 坏的实践：慢的阻塞操作
def on_batch_start(self, context):
    # 不要在非阻塞事件中执行IO
    with open("log.txt", "a") as f:  # 慢！
        f.write(f"Batch {context['batch_idx']}\n")
```

### 3. 依赖管理

```yaml
# 声明显式依赖
dependencies:
  plugins:
    - name: "base_plugin"
      version: ">=1.0.0"
  python_packages:
    - "numpy>=1.20.0"
    - "torch>=2.0.0"
```

### 4. 错误处理

```python
def on_batch_start(self, context):
    try:
        # 可能失败的操作
        result = risky_operation(context)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        # 不要重新抛出，除非必要
        # 让其他插件继续执行
```

### 5. 资源清理

```python
class Plugin(PluginBase):
    def __init__(self):
        self.file_handle = open("temp.txt", "w")

    def cleanup(self):
        # 始终实现清理
        if self.file_handle:
            self.file_handle.close()
```

### 6. 测试策略

```python
# 测试插件独立功能
def test_plugin_logic():
    plugin = MyPlugin()
    result = plugin._internal_method(input_data)
    assert result == expected

# 测试事件处理
def test_event_handling():
    plugin = MyPlugin()
    context = {"batch_idx": 0, "model": mock_model}
    plugin.on_batch_start(context)
    assert "my_plugin_data" in context

# 测试CLI命令
def test_command_execution():
    cmd = MyCommand()
    result = cmd.execute(arg="value")
    assert result["status"] == "success"
```

---

## 故障排查

### 常见问题

**Q: 插件安装失败："Plugin already installed"**
```python
# 使用force=True覆盖
core.install_plugin_package(apg_path, force=True)
```

**Q: 命令未注册**
```python
# 检查plugin.yaml中cli.enabled是否为true
# 手动触发命令发现
core.cli_organizer.discover_and_register_commands()
```

**Q: 插件未自动加载（APX）**
```python
# 检查插件能力是否匹配
analysis = core.analyze_model_for_plugins(capabilities=["moe"])
print(analysis)

# 检查插件是否在注册表中
core.register_plugin_class("route_optimizer", RouteOptimizer)
```

**Q: 版本冲突**
```python
# 检查引擎版本
manifest = packager.get_manifest(apg_path)
print(f"Requires engine: {manifest['engine']}")

# 更新plugin.yaml
engine: ">=1.0.0"  # 放宽版本要求
```

---

## 更新日志

### v2.0 (2025-10-26)
- ✅ 实现APG插件打包系统
- ✅ 实现CLI自组织架构
- ✅ 实现插件加载器和注册表
- ✅ 集成到Console Core
- ✅ 创建插件模板和文档

### v1.0 (之前)
- 基础PluginBus系统
- EQI Manager集成
- APX能力检测
- 自动插件加载（能力驱动）

---

## 参考资源

- **设计文档**: `PLUGIN_PACKAGING_DESIGN.md`
- **重构计划**: `PLUGIN_REFACTORING_PLAN.md`
- **APX兼容性报告**: `APX_COMPATIBILITY_REPORT.md`
- **插件模板**: `examples/plugin_template/`
- **源代码**: `apt_model/console/` 和 `apt_model/tools/apg/`

---

## 下一步

1. **创建你的第一个插件**: 复制 `examples/plugin_template/`
2. **打包分发**: 使用 `PluginPackager` 创建APG文件
3. **贡献插件**: 提交到APT插件市场（即将推出）
4. **参与开发**: 查看CONTRIBUTING.md

---

**祝你开发愉快！** 🚀
