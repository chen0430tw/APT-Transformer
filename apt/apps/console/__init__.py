#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Console Core Module (控制台核心模块)

控制台核心模块负责整合和管理所有 APT 核心模块：
- 模块注册与发现
- 模块生命周期管理
- 模块间通信
- 控制台命令管理

Architecture:
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
"""

__version__ = "1.0.0"

from apt_model.console.core import ConsoleCore
from apt_model.console.module_manager import ModuleManager

__all__ = [
    'ConsoleCore',
    'ModuleManager',
]
