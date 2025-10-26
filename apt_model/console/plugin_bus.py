#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plugin Bus / Event Bus (插件总线 / 事件总线)

基于 memo.txt 中的调度器规范实现的统一插件事件系统。

核心职责：
1. 加载期静态检查（能力冲突、依赖缺失、版本不兼容）
2. 事件派发（按优先级、速率限制、阻塞/非阻塞）
3. 资源/时延防护（预算管理、超时控制）
4. 故障隔离与降级（沙箱、熔断）
5. 合并策略（多插件写入同一字段时的仲裁）
"""

from __future__ import annotations
import logging
import threading
import time
import traceback
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginEvent,
    PluginCapability
)
from apt_model.console.version_checker import VersionChecker

logger = logging.getLogger(__name__)


# ============================================================================
# 插件句柄 (Plugin Handle)
# ============================================================================

@dataclass
class PluginHandle:
    """
    插件句柄

    封装插件实例和运行时状态。
    """
    plugin: PluginBase          # 插件实例
    manifest: PluginManifest    # 插件清单
    last_step_called: int = -10**9  # 上次调用的步数
    healthy: bool = True        # 是否健康
    fail_count: int = 0         # 连续失败次数
    disabled_reason: Optional[str] = None  # 禁用原因
    total_invocations: int = 0  # 总调用次数
    total_time_ms: float = 0.0  # 总耗时（毫秒）


# ============================================================================
# 事件上下文 (Event Context)
# ============================================================================

@dataclass
class EventContext:
    """
    事件上下文

    每个事件都有一个上下文，包含：
    - 公共数据（所有插件可读）
    - 插件私有命名空间
    - 合并结果
    """
    event: str                  # 事件名称
    step: int                   # 当前步数
    epoch: Optional[int] = None # 当前 epoch
    data: Dict[str, Any] = field(default_factory=dict)  # 公共数据
    plugin_ns: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 插件命名空间
    merged: Dict[str, Any] = field(default_factory=dict)  # 合并后的结果

    def get(self, key: str, default: Any = None) -> Any:
        """获取公共数据"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """设置公共数据"""
        self.data[key] = value

    def get_plugin_data(self, plugin_name: str, key: str, default: Any = None) -> Any:
        """获取插件私有数据"""
        return self.plugin_ns.get(plugin_name, {}).get(key, default)

    def set_plugin_data(self, plugin_name: str, key: str, value: Any):
        """设置插件私有数据"""
        if plugin_name not in self.plugin_ns:
            self.plugin_ns[plugin_name] = {}
        self.plugin_ns[plugin_name][key] = value


# ============================================================================
# 插件总线 (Plugin Bus)
# ============================================================================

class PluginBus:
    """
    插件总线

    统一的插件事件调度系统，实现：
    - 插件注册与加载
    - 静态冲突检查
    - 事件派发
    - 资源管理
    - 故障隔离
    """

    def __init__(self,
                 enable_eqi: bool = False,
                 default_timeout_ms: float = 100.0,
                 engine_version: str = "1.0.0"):
        """
        初始化插件总线

        Args:
            enable_eqi: 是否启用 EQI 决策
            default_timeout_ms: 默认超时时间（毫秒）
            engine_version: 引擎版本（用于插件兼容性检查）
        """
        self.handles: Dict[str, PluginHandle] = {}  # name -> handle
        self.ordered_handles: List[PluginHandle] = []  # 按优先级排序
        self.capability_owners: Dict[str, str] = {}  # capability -> owner name
        self.enable_eqi = enable_eqi
        self.default_timeout_ms = default_timeout_ms
        self.eqi_manager = None  # EQI Manager（可选）
        self.version_checker = VersionChecker(engine_version)  # 版本检查器

        logger.info(f"PluginBus initialized (engine version: {engine_version})")

    # ========================================================================
    # 插件注册与加载
    # ========================================================================

    def register(self, plugin: PluginBase, manifest: Optional[PluginManifest] = None):
        """
        注册插件

        Args:
            plugin: 插件实例
            manifest: 插件清单（如果为 None，则从 plugin.get_manifest() 获取）
        """
        if manifest is None:
            manifest = plugin.get_manifest()

        if not manifest.validate():
            raise ValueError(f"Invalid plugin manifest for '{manifest.name}'")

        if manifest.name in self.handles:
            logger.warning(f"Plugin '{manifest.name}' already registered, overwriting")

        handle = PluginHandle(plugin=plugin, manifest=manifest)
        self.handles[manifest.name] = handle

        logger.info(f"Registered plugin: {manifest.name} (priority={manifest.priority}, "
                   f"events={manifest.events})")

    def compile(self, fail_fast: bool = False):
        """
        编译插件（静态冲突检查）

        执行加载期静态检查：
        1. 依赖检查
        2. 硬冲突检查
        3. 能力独占检查

        Args:
            fail_fast: 遇到错误是否立即失败（否则跳过冲突插件）
        """
        logger.info("Compiling plugins (static conflict resolution)...")

        # 按优先级排序（priority升序，name次序）
        ordered = sorted(
            self.handles.values(),
            key=lambda h: (h.manifest.priority, h.manifest.name)
        )

        active_handles: List[PluginHandle] = []
        loaded_names = set(h.manifest.name for h in ordered)

        for handle in ordered:
            if not handle.healthy:
                continue

            manifest = handle.manifest

            # 1. 依赖检查
            missing_deps = []
            for req in manifest.requires:
                if req.startswith("plugin:"):
                    plugin_name = req.split(":", 1)[1]
                    if plugin_name not in loaded_names:
                        missing_deps.append(req)
                # 其他依赖类型（core:trainer, capability:xxx）可扩展

            if missing_deps:
                handle.healthy = False
                handle.disabled_reason = f"requires-missing:{','.join(missing_deps)}"
                logger.warning(f"Plugin '{manifest.name}' disabled: missing dependencies {missing_deps}")
                if fail_fast:
                    raise RuntimeError(f"Dependency missing for {manifest.name}: {missing_deps}")
                continue

            # 2. 版本兼容性检查（新增）
            is_compatible, version_reason = self.version_checker.check_plugin_compatibility(manifest)
            if not is_compatible:
                handle.healthy = False
                handle.disabled_reason = f"version-incompatible:{version_reason}"
                logger.warning(f"Plugin '{manifest.name}' disabled: {version_reason}")
                if fail_fast:
                    raise RuntimeError(f"Version incompatible for {manifest.name}: {version_reason}")
                continue

            # 3. 硬冲突检查
            conflict_hit = False
            for conflict in manifest.conflicts:
                if conflict.startswith("plugin:"):
                    conflict_plugin = conflict.split(":", 1)[1]
                    if conflict_plugin in [h.manifest.name for h in active_handles]:
                        conflict_hit = True
                        break

            if conflict_hit:
                handle.healthy = False
                handle.disabled_reason = "hard-conflict"
                logger.warning(f"Plugin '{manifest.name}' disabled: hard conflict")
                continue

            # 4. 能力独占检查
            cap_conflict = False
            for cap in manifest.capabilities:
                if PluginCapability.is_exclusive(cap):
                    owner = self.capability_owners.get(cap)
                    if owner is None:
                        self.capability_owners[cap] = manifest.name
                        logger.debug(f"Capability '{cap}' assigned to '{manifest.name}'")
                    else:
                        # 已被占用
                        cap_conflict = True
                        logger.warning(f"Plugin '{manifest.name}' disabled: capability '{cap}' "
                                     f"already owned by '{owner}'")
                        break
                else:
                    # 非独占能力，多个插件可以持有
                    if cap not in self.capability_owners:
                        self.capability_owners[cap] = manifest.name
                    else:
                        # 记录共享
                        pass

            if cap_conflict:
                handle.healthy = False
                handle.disabled_reason = "capability-occupied"
                continue

            # 通过所有检查，加入活跃列表
            active_handles.append(handle)

        self.ordered_handles = active_handles
        logger.info(f"Plugin compilation complete: {len(active_handles)}/{len(ordered)} active")

        # 打印加载结果
        self.print_status()

    # ========================================================================
    # 事件派发
    # ========================================================================

    def emit(self, event: str, step: int, context_data: Optional[Dict[str, Any]] = None) -> EventContext:
        """
        派发事件

        Args:
            event: 事件名称
            step: 当前步数
            context_data: 上下文数据

        Returns:
            事件上下文（包含所有插件的处理结果）
        """
        # 创建事件上下文
        ctx = EventContext(
            event=event,
            step=step,
            data=context_data or {}
        )

        # 过滤订阅了该事件的插件
        subscribers = [
            h for h in self.ordered_handles
            if h.healthy and event in h.manifest.events
        ]

        if not subscribers:
            logger.debug(f"No subscribers for event '{event}'")
            return ctx

        logger.debug(f"Emitting event '{event}' to {len(subscribers)} plugin(s)")

        # 依次调用插件
        for handle in subscribers:
            self._invoke_plugin(handle, ctx)

        return ctx

    def _invoke_plugin(self, handle: PluginHandle, ctx: EventContext):
        """
        调用插件处理事件

        Args:
            handle: 插件句柄
            ctx: 事件上下文
        """
        manifest = handle.manifest

        # 速率限制检查
        if manifest.get_rate_limit_steps() > 0:
            if ctx.step - handle.last_step_called < manifest.get_rate_limit_steps():
                logger.debug(f"Plugin '{manifest.name}' skipped due to rate limit")
                return

        # 更新调用时间
        handle.last_step_called = ctx.step
        handle.total_invocations += 1

        # 获取事件处理方法
        handler_method = getattr(handle.plugin, ctx.event, None)
        if not callable(handler_method):
            logger.debug(f"Plugin '{manifest.name}' has no handler for '{ctx.event}'")
            return

        # 构建插件上下文
        plugin_ctx = {
            "event": ctx.event,
            "step": ctx.step,
            "epoch": ctx.epoch,
            "plugin": manifest.name,
            "data": ctx.data,  # 公共数据
        }

        # 调用插件
        try:
            start_time = time.time()

            if manifest.blocking:
                self._invoke_blocking(handler_method, plugin_ctx, handle)
            else:
                self._invoke_nonblocking(handler_method, plugin_ctx, handle)

            # 记录耗时
            elapsed_ms = (time.time() - start_time) * 1000
            handle.total_time_ms += elapsed_ms

            logger.debug(f"Plugin '{manifest.name}' executed in {elapsed_ms:.2f}ms")

        except Exception as e:
            self._handle_plugin_failure(handle, e)

    def _invoke_blocking(self, handler: Callable, plugin_ctx: Dict[str, Any], handle: PluginHandle):
        """
        阻塞调用插件

        使用线程 + join 实现超时控制
        """
        manifest = handle.manifest
        timeout_sec = manifest.get_timeout_ms() / 1000.0

        result_holder = {"error": None}

        def runner():
            try:
                handler(plugin_ctx)
            except Exception as e:
                result_holder["error"] = e

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            # 超时
            handle.fail_count += 1
            logger.error(f"Plugin '{manifest.name}' timed out ({timeout_sec}s)")
            if manifest.sandbox and handle.fail_count >= manifest.fail_limit:
                handle.healthy = False
                handle.disabled_reason = "timeout"
            raise TimeoutError(f"Plugin '{manifest.name}' timed out")

        if result_holder["error"] is not None:
            raise result_holder["error"]

    def _invoke_nonblocking(self, handler: Callable, plugin_ctx: Dict[str, Any], handle: PluginHandle):
        """
        非阻塞调用插件

        使用线程异步执行，不等待结果
        """
        def runner():
            try:
                handler(plugin_ctx)
            except Exception as e:
                logger.error(f"Plugin '{handle.manifest.name}' failed (non-blocking): {e}")
                logger.debug(traceback.format_exc())
                handle.fail_count += 1
                if handle.manifest.sandbox and handle.fail_count >= handle.manifest.fail_limit:
                    handle.healthy = False
                    handle.disabled_reason = "exception"

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

    def _handle_plugin_failure(self, handle: PluginHandle, error: Exception):
        """
        处理插件失败

        Args:
            handle: 插件句柄
            error: 异常
        """
        handle.fail_count += 1
        manifest = handle.manifest

        logger.error(f"Plugin '{manifest.name}' failed: {error}")
        logger.debug(traceback.format_exc())

        if manifest.sandbox and handle.fail_count >= manifest.fail_limit:
            handle.healthy = False
            handle.disabled_reason = f"exception:{error}"
            logger.warning(f"Plugin '{manifest.name}' disabled after {handle.fail_count} failures")

    # ========================================================================
    # 状态与管理
    # ========================================================================

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """获取插件实例"""
        handle = self.handles.get(name)
        return handle.plugin if handle else None

    def get_handle(self, name: str) -> Optional[PluginHandle]:
        """获取插件句柄"""
        return self.handles.get(name)

    def enable_plugin(self, name: str):
        """启用插件"""
        handle = self.handles.get(name)
        if handle:
            handle.healthy = True
            handle.disabled_reason = None
            logger.info(f"Plugin '{name}' enabled")

    def disable_plugin(self, name: str, reason: str = "manual"):
        """禁用插件"""
        handle = self.handles.get(name)
        if handle:
            handle.healthy = False
            handle.disabled_reason = reason
            logger.info(f"Plugin '{name}' disabled: {reason}")

    def print_status(self):
        """打印插件状态"""
        print("\n" + "="*100)
        print(" Plugin Bus Status")
        print("="*100)
        print(f"{'Name':<25} {'Priority':<10} {'Class':<20} {'Status':<15} {'Events':<20}")
        print("-"*100)

        for handle in sorted(self.handles.values(), key=lambda h: h.manifest.priority):
            manifest = handle.manifest
            status = "✓ ACTIVE" if handle.healthy else f"✗ {handle.disabled_reason or 'DISABLED'}"
            events = ", ".join(manifest.events[:2])
            if len(manifest.events) > 2:
                events += f" +{len(manifest.events)-2}"

            print(f"{manifest.name:<25} {manifest.priority:<10} "
                  f"{manifest.get_priority_class():<20} {status:<15} {events:<20}")

        active_count = sum(1 for h in self.handles.values() if h.healthy)
        print("="*100)
        print(f"Total: {len(self.handles)} plugin(s), {active_count} active")
        print()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_plugins": len(self.handles),
            "active_plugins": sum(1 for h in self.handles.values() if h.healthy),
            "disabled_plugins": sum(1 for h in self.handles.values() if not h.healthy),
            "total_invocations": sum(h.total_invocations for h in self.handles.values()),
            "total_time_ms": sum(h.total_time_ms for h in self.handles.values()),
            "plugins": {}
        }

        for name, handle in self.handles.items():
            stats["plugins"][name] = {
                "healthy": handle.healthy,
                "fail_count": handle.fail_count,
                "invocations": handle.total_invocations,
                "total_time_ms": handle.total_time_ms,
                "avg_time_ms": handle.total_time_ms / handle.total_invocations if handle.total_invocations > 0 else 0.0,
                "disabled_reason": handle.disabled_reason
            }

        return stats
