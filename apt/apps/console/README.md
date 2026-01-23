# APT Console Core (控制台核心模块)

## 概述

APT Console Core 是一个统一的控制台系统，用于管理和整合所有 APT 核心模块。它提供了模块注册、加载、生命周期管理和命令系统。

## 架构

```
Console Core (控制台核心)
├── Module Manager (模块管理器)
│   ├── Core Modules (核心模块)
│   │   ├── VFT/TVA (Vein-Flow Transformer)
│   │   ├── EQI (Evidence Qualitative Inference)
│   │   ├── RAG (Retrieval Augmented Generation)
│   │   ├── Reasoning Controller (推理控制器)
│   │   ├── Codec System (编解码系统)
│   │   ├── Plugin System (插件系统)
│   │   ├── Multilingual Support (多语言支持)
│   │   └── Training System (训练系统)
│   └── Runtime Modules (运行时模块)
├── Command System (命令系统)
│   ├── Module Commands (模块命令)
│   ├── System Commands (系统命令)
│   └── Custom Commands (自定义命令)
└── Launcher (启动器)
    ├── CLI Launcher (命令行启动器)
    ├── Interactive Mode (交互模式)
    └── Batch Mode (批处理模式)
```

## 核心组件

### 1. ModuleManager (模块管理器)

负责所有核心模块的注册、加载和管理。

**主要功能:**
- 模块注册与发现
- 依赖解析（拓扑排序）
- 模块加载与初始化
- 生命周期管理
- 模块间通信

**已注册的核心模块:**
- `vft_tva` - Vein-Flow Transformer
- `eqi` - Evidence Qualitative Inference
- `reasoning` - Reasoning Controller
- `codec` - Codec System
- `plugins` - Plugin System
- `multilingual` - Multilingual Support
- `registry` - Core Registry
- `training` - Training System
- `rag` - RAG System
- `hardware` - Hardware Management

### 2. ConsoleCore (控制台核心)

控制台核心整合了模块管理器、命令系统和配置管理。

**主要功能:**
- 启动和停止控制台
- 加载和管理模块
- 执行命令
- 配置管理
- 状态监控

### 3. Command System (命令系统)

提供丰富的命令来管理模块和系统。

## 使用方法

### 基本用法

```python
from apt_model.console.core import initialize_console

# 初始化并启动控制台
console = initialize_console(auto_start=True)

# 加载模块
console.load_module("vft_tva")

# 执行命令
console.execute_command("modules-list", args)

# 获取模块实例
vft_module = console.get_module("vft_tva")
```

### 命令行用法

```bash
# 显示控制台状态
python -m apt_model console-status

# 列出所有模块
python -m apt_model modules-list

# 加载模块
python -m apt_model modules-load vft_tva

# 显示模块信息
python -m apt_model modules-info eqi

# 列出所有命令
python -m apt_model console-commands

# 显示帮助
python -m apt_model console-help
```

## 可用命令

### 模块管理命令

| 命令 | 别名 | 说明 |
|------|------|------|
| `modules-list` | `modules`, `mod-list`, `lsmod` | 列出所有注册的模块 |
| `modules-load` | `mod-load`, `loadmod` | 加载指定模块 |
| `modules-enable` | `mod-enable` | 启用模块 |
| `modules-disable` | `mod-disable` | 禁用模块 |
| `modules-info` | `mod-info`, `modinfo` | 显示模块详细信息 |
| `modules-status` | `mod-status`, `modstat` | 显示所有模块状态统计 |

### 系统命令

| 命令 | 别名 | 说明 |
|------|------|------|
| `console-status` | `status`, `stat` | 显示控制台状态 |
| `console-version` | `version`, `ver` | 显示版本信息 |
| `console-config-get` | `config-get`, `cfg-get` | 获取配置值 |
| `console-config-set` | `config-set`, `cfg-set` | 设置配置值 |
| `console-config-list` | `config-list`, `cfg-list` | 列出所有配置 |
| `console-commands` | `commands`, `cmd-list` | 列出所有可用命令 |
| `console-help` | `help`, `h`, `?` | 显示帮助信息 |

## 模块状态

模块可以处于以下状态之一：

- **UNLOADED** - 未加载
- **LOADING** - 加载中
- **LOADED** - 已加载
- **INITIALIZING** - 初始化中
- **READY** - 就绪（可使用）
- **ERROR** - 错误
- **DISABLED** - 已禁用

## 配置

控制台核心支持配置管理，使用点号分隔的路径：

```python
# 设置配置
console.set_config("module.option", "value")

# 获取配置
value = console.get_config("module.option", default="default_value")
```

## 扩展

### 添加自定义模块

```python
from apt_model.console.module_manager import ModuleMetadata

# 注册自定义模块
module_manager.register_module(ModuleMetadata(
    name="my_module",
    version="1.0.0",
    description="My Custom Module",
    module_path="my_package.my_module",
    dependencies=[],
    category="custom",
    auto_load=False,
))
```

### 添加自定义命令

```python
from apt_model.cli.command_registry import register_command

@register_command(
    name="my-command",
    aliases=["mc"],
    category="custom",
    help_text="My custom command"
)
def my_command(args):
    print("Executing my custom command")
    return 0
```

## 示例

### 示例 1: 启动控制台并查看状态

```python
from apt_model.console.core import initialize_console

# 初始化控制台
console = initialize_console(auto_start=True)

# 打印状态
console.print_banner()
console.print_status()
```

### 示例 2: 加载和使用模块

```python
from apt_model.console.core import get_console

# 获取控制台实例
console = get_console()
console.start()

# 加载 VFT/TVA 模块
if console.load_module("vft_tva"):
    vft = console.get_module("vft_tva")
    # 使用 vft 模块...
```

### 示例 3: 批量加载模块

```python
from apt_model.console.core import initialize_console

console = initialize_console(auto_start=True)

# 加载所有核心模块
results = console.module_manager.load_all(auto_only=True)

# 检查结果
for module_name, success in results.items():
    if success:
        print(f"✓ {module_name} loaded")
    else:
        print(f"✗ {module_name} failed")
```

## 启动器集成

Console Core 已集成到 APT Launcher 中，作为统一的启动入口：

```bash
# 启动器会自动初始化控制台
python -m apt_model

# 显示默认帮助
python -m apt_model

# 执行命令
python -m apt_model console-status
python -m apt_model train
python -m apt_model chat
```

## 日志

控制台核心使用标准 Python logging 系统：

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 获取 console logger
logger = logging.getLogger("apt_model.console")
```

## 最佳实践

1. **单例模式**: 使用 `get_console()` 获取全局单例实例
2. **依赖管理**: 确保模块的依赖正确声明
3. **错误处理**: 总是检查模块加载的返回值
4. **配置管理**: 使用控制台的配置系统而不是全局变量
5. **命令注册**: 使用装饰器注册命令，保持代码整洁

## 故障排除

### 问题: 模块加载失败

**解决方案:**
1. 检查模块路径是否正确
2. 确认依赖模块已加载
3. 查看日志了解详细错误信息

### 问题: 命令未找到

**解决方案:**
1. 使用 `console-commands` 列出所有命令
2. 检查命令拼写
3. 确认命令已注册

## 版本历史

### v1.0.0 (2025-10-25)
- 初始版本
- 实现模块管理器
- 实现控制台核心
- 添加模块管理命令
- 添加系统命令
- 集成到 Launcher

## 许可

与 APT Model 项目相同的许可证。
