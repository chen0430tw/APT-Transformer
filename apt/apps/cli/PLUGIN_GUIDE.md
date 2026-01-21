# CLI插件开发指南

## 概述

APT Model的CLI系统支持插件动态注册命令。插件开发者可以轻松添加自定义命令，而无需修改核心代码。

## 命令注册系统

### 核心组件

1. **CommandRegistry** - 命令注册中心
2. **CommandMetadata** - 命令元数据
3. **register_plugin_command()** - 插件命令注册函数

### 命令类别

命令按类别组织，内置类别包括：
- `training` - 训练相关命令
- `interactive` - 交互命令
- `evaluation` - 评估命令
- `tools` - 工具命令
- `info` - 信息查询命令
- `maintenance` - 维护命令
- `data` - 数据处理命令
- `distribution` - 分发命令
- `testing` - 测试命令
- `plugin` - 插件命令（默认）

## 如何创建插件命令

### 基本示例

```python
# my_plugin.py
from apt_model.cli import register_plugin_command

def my_custom_command(args):
    """
    自定义命令实现

    参数:
        args: 命令行参数（argparse.Namespace）

    返回:
        int: 退出码（0表示成功，非0表示错误）
    """
    print("执行自定义命令")
    print(f"收到的参数: {args}")

    # 执行命令逻辑
    try:
        # ... 你的代码 ...
        return 0  # 成功
    except Exception as e:
        print(f"错误: {e}")
        return 1  # 失败

# 注册命令
register_plugin_command(
    name="my-cmd",
    func=my_custom_command,
    category="plugin",
    help_text="我的自定义命令",
    aliases=["mycmd"]
)
```

### 使用codec插件示例

```python
# codec_plugin_command.py
from apt_model.cli import register_plugin_command

def list_codecs_command(args):
    """列出所有可用的codec"""
    from apt_model.codecs import list_available_codecs

    print("可用的Codec:")
    codecs = list_available_codecs()
    for codec_name in codecs:
        print(f"  - {codec_name}")

    return 0

# 注册命令
register_plugin_command(
    name="list-codecs",
    func=list_codecs_command,
    category="info",
    help_text="列出所有可用的codec",
    aliases=["lc"]
)
```

### 完整示例：数据集分析命令

```python
# dataset_analyzer_plugin.py
from apt_model.cli import register_plugin_command
import os

def analyze_dataset_command(args):
    """
    分析数据集统计信息
    """
    if not args.data_path:
        print("错误: 需要指定 --data-path 参数")
        return 1

    if not os.path.exists(args.data_path):
        print(f"错误: 文件不存在: {args.data_path}")
        return 1

    try:
        # 加载数据
        with open(args.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 统计分析
        total_lines = len(lines)
        total_chars = sum(len(line) for line in lines)
        avg_length = total_chars / total_lines if total_lines > 0 else 0

        print("=" * 60)
        print("数据集分析报告")
        print("=" * 60)
        print(f"文件路径: {args.data_path}")
        print(f"总行数: {total_lines}")
        print(f"总字符数: {total_chars}")
        print(f"平均行长度: {avg_length:.2f}")
        print("=" * 60)

        return 0
    except Exception as e:
        print(f"分析失败: {e}")
        return 1

# 注册命令
register_plugin_command(
    name="analyze-dataset",
    func=analyze_dataset_command,
    category="data",
    help_text="分析数据集统计信息",
    aliases=["analyze", "stats"]
)
```

## 插件加载

### 方法1：在APT启动时自动加载

将插件文件放在 `apt_model/plugins/` 目录下，并在 `apt_model/plugins/__init__.py` 中导入：

```python
# apt_model/plugins/__init__.py
from . import my_plugin
from . import dataset_analyzer_plugin
```

### 方法2：手动加载插件

在使用命令之前导入插件模块：

```python
import apt_model.plugins.my_plugin
```

### 方法3：命令行参数加载（未来功能）

```bash
python -m apt_model --load-plugin my_plugin.py my-cmd
```

## 高级功能

### 访问全局注册中心

```python
from apt_model.cli import command_registry

# 列出所有命令
all_commands = command_registry.list_commands()
print(f"共有 {len(all_commands)} 个命令")

# 获取命令元数据
metadata = command_registry.get_command("train")
if metadata:
    print(f"命令: {metadata.name}")
    print(f"类别: {metadata.category}")
    print(f"帮助: {metadata.help_text}")

# 按类别获取命令
commands_by_cat = command_registry.get_commands_by_category()
for category, commands in commands_by_cat.items():
    print(f"{category}: {len(commands)} 个命令")
```

### 注销命令

```python
from apt_model.cli import command_registry

# 注销命令
success = command_registry.unregister("my-cmd")
if success:
    print("命令注销成功")
```

### 使用低级API

```python
from apt_model.cli.command_registry import register_command, CommandMetadata

# 使用低级API注册
register_command(
    name="advanced-cmd",
    func=my_function,
    category="advanced",
    help_text="高级命令",
    args_help={
        "--input": "输入文件",
        "--output": "输出文件"
    },
    aliases=["adv"],
    is_placeholder=False
)
```

## 最佳实践

1. **命名规范**: 使用小写字母和连字符（kebab-case），如 `my-command`
2. **返回值**: 始终返回int类型的退出码（0表示成功）
3. **错误处理**: 使用try-except捕获异常并返回非0退出码
4. **帮助文本**: 提供清晰的help_text，帮助用户理解命令用途
5. **别名**: 为常用命令提供简短别名
6. **日志记录**: 使用logger记录重要信息，方便调试
7. **参数验证**: 在命令函数开始时验证必需参数

## 调试插件

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试命令注册
from apt_model.cli import command_registry
print("已注册的命令:", command_registry.list_commands())

# 测试命令执行
from apt_model.cli.parser import parse_arguments
args = parse_arguments()  # 模拟命令行参数
result = command_registry.execute_command("my-cmd", args)
print(f"命令返回: {result}")
```

## 示例：完整的插件包

```
my_apt_plugin/
├── __init__.py           # 插件初始化
├── commands/             # 命令模块
│   ├── __init__.py
│   ├── analyze.py        # 分析命令
│   └── export.py         # 导出命令
├── utils.py              # 工具函数
└── README.md             # 插件文档
```

```python
# my_apt_plugin/__init__.py
"""
My APT Plugin
Custom commands for APT Model
"""
from .commands import analyze, export

__version__ = "1.0.0"
```

## 参考

- 核心命令实现: `apt_model/cli/commands.py`
- 命令注册系统: `apt_model/cli/command_registry.py`
- 参数解析器: `apt_model/cli/parser.py`
