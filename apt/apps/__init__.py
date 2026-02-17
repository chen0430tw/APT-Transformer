#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Apps Module (L3 Product Layer)

应用层 - 完整的产品交付：
- WebUI (训练监控 + 模型探索 + 推理测试)
- API (RESTful API + WebSocket 推送)
- CLI (命令行工具 + 交互式控制台)
- Plugins (插件系统 + 扩展机制)
- Agent (自主代理 + 工具调用)
- Observability (日志 + 指标 + 追踪)

使用示例:
    >>> import apt
    >>> apt.enable('full')  # 加载 L0 + L1 + L2 + L3
    >>> from apt.apps import launch_webui, create_api_server
    >>> launch_webui(port=8080)
"""

# ═══════════════════════════════════════════════════════════
# WebUI
# ═══════════════════════════════════════════════════════════
try:
    try:
        from apt.apps.webui.app import launch_webui
    except ImportError:
        launch_webui = None
except ImportError:
    launch_webui = None

# ═══════════════════════════════════════════════════════════
# API Server
# ═══════════════════════════════════════════════════════════
try:
    try:
        from apt.apps.api.server import create_api_server
    except ImportError:
        create_api_server = None
except ImportError:
    create_api_server = None

# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
try:
    try:
        from apt.apps.cli.commands import APTCommand
    except ImportError:
        APTCommand = None
except ImportError:
    APTCommand = None

try:
    try:
        from apt.apps.cli.parser import parse_args
    except ImportError:
        parse_args = None
except ImportError:
    parse_args = None

# ═══════════════════════════════════════════════════════════
# Console System
# ═══════════════════════════════════════════════════════════
try:
    try:
        from apt.apps.console.core import Console
    except ImportError:
        Console = None
except ImportError:
    Console = None

try:
    try:
        from apt.apps.console.plugin_bus import PluginBus
    except ImportError:
        PluginBus = None
except ImportError:
    PluginBus = None

# ═══════════════════════════════════════════════════════════
# Plugin System
# ═══════════════════════════════════════════════════════════
try:
    try:
        from apt.apps.plugins.compression_plugin import CompressionPlugin
    except ImportError:
        CompressionPlugin = None
except ImportError:
    CompressionPlugin = None

try:
    try:
        from apt.apps.plugins.teacher_api import TeacherAPIPlugin
    except ImportError:
        TeacherAPIPlugin = None
except ImportError:
    TeacherAPIPlugin = None

try:
    try:
        from apt.apps.plugins.version_manager import PluginVersionManager
    except ImportError:
        PluginVersionManager = None
except ImportError:
    PluginVersionManager = None

# ═══════════════════════════════════════════════════════════
# Agent System
# ═══════════════════════════════════════════════════════════
try:
    try:
        from apt.apps.agent.agent_loop import AgentLoop
    except ImportError:
        AgentLoop = None
except ImportError:
    AgentLoop = None

try:
    try:
        from apt.apps.agent.tool_system import ToolSystem
    except ImportError:
        ToolSystem = None
except ImportError:
    ToolSystem = None

# ═══════════════════════════════════════════════════════════
# Evaluation & RL
# ═══════════════════════════════════════════════════════════
try:
    try:
        from apt.apps.evaluation.unified import UnifiedEvaluator
    except ImportError:
        UnifiedEvaluator = None
except ImportError:
    UnifiedEvaluator = None

try:
    try:
        from apt.apps.rl.grpo_trainer import GRPOTrainer
    except ImportError:
        GRPOTrainer = None
except ImportError:
    GRPOTrainer = None

try:
    try:
        from apt.apps.rl.dpo_trainer import DPOTrainer
    except ImportError:
        DPOTrainer = None
except ImportError:
    DPOTrainer = None

# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════
__all__ = [
    # WebUI
    'launch_webui',

    # API
    'create_api_server',

    # CLI
    'APTCommand',
    'parse_args',

    # Console
    'Console',
    'PluginBus',

    # Plugins
    'CompressionPlugin',
    'TeacherAPIPlugin',
    'PluginVersionManager',

    # Agent
    'AgentLoop',
    'ToolSystem',

    # Evaluation & RL
    'UnifiedEvaluator',
    'GRPOTrainer',
    'DPOTrainer',
]