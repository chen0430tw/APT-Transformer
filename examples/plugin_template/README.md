# Example Plugin Template

这是一个完整的APT插件模板，展示了如何创建、打包和分发插件。

## 目录结构

```
plugin_template/
├── plugin.yaml              # 插件元数据（必需）
├── plugin/                  # 插件代码目录（必需）
│   └── __init__.py         # 插件主类（必需，必须包含Plugin类）
├── commands/                # CLI命令目录（可选）
│   └── example_command.py  # 示例命令
├── tests/                   # 测试目录（可选）
│   └── test_plugin.py      # 单元测试
└── README.md               # 插件文档
```

## 快速开始

### 1. 创建你的插件

1. 复制这个模板目录
2. 修改 `plugin.yaml` 中的元数据
3. 在 `plugin/__init__.py` 中实现你的插件逻辑
4. （可选）在 `commands/` 中添加CLI命令

### 2. 实现插件类

你的插件类必须：
- 命名为 `Plugin`
- 继承自 `PluginBase`
- 实现 `get_manifest()` 方法
- 实现需要的事件处理方法

```python
from apt_model.console.plugin_standards import PluginBase, PluginManifest

class Plugin(PluginBase):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="my_plugin",
            priority=350,
            events=["on_init", "on_batch_start"],
            # ...其他字段
        )

    def on_init(self, context):
        # 初始化逻辑
        pass

    def on_batch_start(self, context):
        # 批次开始逻辑
        pass
```

### 3. 打包插件

使用APT工具打包你的插件：

```python
from pathlib import Path
from apt_model.tools.apg.packager import pack_plugin

pack_plugin(
    src_dir=Path("plugin_template"),
    out_apg=Path("dist/my_plugin-1.0.0.apg"),
    name="my_plugin",
    version="1.0.0",
    description="My awesome plugin",
    author="Your Name",
)
```

### 4. 安装和使用

```python
from apt_model.console.core import ConsoleCore

core = ConsoleCore()

# 安装插件
core.install_plugin_package(
    Path("dist/my_plugin-1.0.0.apg"),
    auto_load=True
)

# 插件会自动加载和注册CLI命令
```

## 插件元数据说明

### 基础字段

- `name`: 插件名称（必需，唯一标识符）
- `version`: 版本号（必需，语义化版本）
- `description`: 插件描述（必需）
- `author`: 作者信息（必需）

### 兼容性字段

- `engine`: APT引擎版本要求（如 `">=1.0.0"`）
- `python`: Python版本要求（如 `">=3.8,<4.0"`）

### 能力字段

- `capabilities.required`: 必需的模型能力（如 `["moe"]`）
- `capabilities.optional`: 可选的模型能力
- `capabilities.provides`: 插件提供的能力

### PluginBus配置

- `plugin_bus.priority`: 插件优先级（0-999）
- `plugin_bus.category`: 插件类别
- `plugin_bus.events`: 监听的事件列表
- `plugin_bus.blocking`: 是否阻塞事件
- `plugin_bus.conflicting_plugins`: 冲突插件列表

### CLI扩展配置

```yaml
cli:
  enabled: true
  commands:
    - name: "my-command"
      module: "commands.my_command"
      class: "MyCommand"
      description: "My command description"
      aliases: ["my-cmd"]
```

## 事件系统

插件可以监听以下事件：

- `on_init`: 系统初始化
- `on_batch_start`: 批次开始
- `on_batch_end`: 批次结束
- `on_step_start`: 步骤开始
- `on_step_end`: 步骤结束
- `on_decode`: 解码过程
- `on_shutdown`: 系统关闭

## CLI命令

### 创建命令类

```python
class MyCommand:
    def execute(self, *args, **kwargs):
        # 命令逻辑
        return result
```

### 注册命令

在 `plugin.yaml` 中配置：

```yaml
cli:
  enabled: true
  commands:
    - name: "my-command"
      module: "commands.my_command"
      class: "MyCommand"
      description: "Command description"
```

### 使用命令

```python
# 通过Console Core
core.execute_plugin_command("my-command", arg1="value1")

# 通过CLI Organizer
result = core.cli_organizer.execute_command("my-command")
```

## 依赖管理

### Python依赖

在 `plugin.yaml` 中声明：

```yaml
dependencies:
  python_packages:
    - "numpy>=1.20.0"
    - "scipy>=1.7.0"
```

### 插件依赖

```yaml
dependencies:
  plugins:
    - "base_plugin"
    - name: "other_plugin"
      version: ">=2.0.0"
```

## 测试

创建单元测试：

```python
# tests/test_plugin.py
import pytest
from plugin import Plugin

def test_plugin_initialization():
    plugin = Plugin()
    manifest = plugin.get_manifest()
    assert manifest.name == "example_plugin"

def test_plugin_on_init():
    plugin = Plugin()
    context = {}
    plugin.on_init(context)
    # 验证逻辑
```

## 最佳实践

1. **清晰的命名**: 使用描述性的插件名称
2. **版本管理**: 遵循语义化版本规范
3. **文档完善**: 提供详细的README和docstrings
4. **错误处理**: 妥善处理异常情况
5. **日志记录**: 使用logging模块记录关键操作
6. **资源清理**: 实现cleanup方法释放资源
7. **测试覆盖**: 编写充分的单元测试

## 示例插件

查看这个模板中的代码作为参考实现。

## 许可证

MIT License (可根据需要修改)
