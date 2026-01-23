#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plugin System Standalone Tests

独立测试插件系统（不依赖完整的 apt_model 初始化）
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 直接导入控制台组件（避免触发 apt_model 的完整初始化）
from apt.apps.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)
from apt.apps.console.plugin_bus import PluginBus, EventContext
from apt.apps.console.plugins.grpo_plugin import GRPOPlugin
from apt.apps.console.plugins.eqi_reporter_plugin import EQIReporterPlugin
from apt.apps.console.plugins.route_optimizer_plugin import RouteOptimizerPlugin


def test_plugin_bus_registration():
    """测试 PluginBus 插件注册"""
    print("\n" + "="*80)
    print(" 测试 PluginBus 插件注册")
    print("="*80)

    # 创建 PluginBus
    bus = PluginBus()

    # 注册插件
    print("\n1. 注册 GRPO 插件...")
    grpo = GRPOPlugin()
    bus.register(grpo)
    print("   ✓ GRPO 插件注册成功")

    print("\n2. 注册 EQI Reporter 插件...")
    eqi_reporter = EQIReporterPlugin()
    bus.register(eqi_reporter)
    print("   ✓ EQI Reporter 插件注册成功")

    print("\n3. 注册 Route Optimizer 插件...")
    route_optimizer = RouteOptimizerPlugin()
    bus.register(route_optimizer)
    print("   ✓ Route Optimizer 插件注册成功")

    # 检查插件状态
    stats = bus.get_statistics()
    print(f"\n✓ 插件注册测试完成: {stats['total_plugins']} 个插件已注册")

    return True


def test_plugin_compilation():
    """测试插件编译（静态冲突检查）"""
    print("\n" + "="*80)
    print(" 测试插件编译")
    print("="*80)

    # 创建 PluginBus 并注册插件
    bus = PluginBus()
    bus.register(GRPOPlugin())
    bus.register(EQIReporterPlugin())
    bus.register(RouteOptimizerPlugin())

    # 编译插件
    print("\n1. 执行插件编译（静态冲突检查）...")
    bus.compile(fail_fast=False)
    print("   ✓ 插件编译成功")

    # 打印插件状态
    print("\n2. 插件状态:")
    bus.print_status()

    print("\n✓ 插件编译测试完成")
    return True


def test_event_dispatch():
    """测试事件派发"""
    print("\n" + "="*80)
    print(" 测试事件派发")
    print("="*80)

    # 创建 PluginBus 并注册插件
    bus = PluginBus()
    bus.register(GRPOPlugin())
    bus.register(EQIReporterPlugin())
    bus.register(RouteOptimizerPlugin())
    bus.compile()

    # 测试 on_batch_start 事件
    print("\n1. 派发 on_batch_start 事件...")
    context = bus.emit(
        PluginEvent.ON_BATCH_START,
        step=1,
        context_data={
            'routing': {
                'expert_ids': [0, 1, 2, 3, 0, 1, 0, 2]
            }
        }
    )
    print(f"   ✓ 事件已派发 (event={context.event}, step={context.step})")

    # 测试 on_batch_end 事件
    print("\n2. 派发 on_batch_end 事件...")
    context = bus.emit(
        PluginEvent.ON_BATCH_END,
        step=1,
        context_data={
            'rewards': [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6]
        }
    )
    print(f"   ✓ 事件已派发")

    # 测试 on_step_end 事件
    print("\n3. 派发 on_step_end 事件...")
    context = bus.emit(
        PluginEvent.ON_STEP_END,
        step=1,
        context_data={
            'loss': 0.35,
            'metrics': {
                'accuracy': 0.85
            }
        }
    )
    print(f"   ✓ 事件已派发")

    # 测试 on_step_eval 事件
    print("\n4. 派发 on_step_eval 事件...")
    context = bus.emit(
        PluginEvent.ON_STEP_EVAL,
        step=10,
        context_data={
            'metrics': {
                'accuracy': 0.88,
                'loss': 0.32
            },
            'evidence': 1.2,
            'utility': 0.5
        }
    )
    print(f"   ✓ 事件已派发")

    # 测试 on_epoch_end 事件
    print("\n5. 派发 on_epoch_end 事件...")
    context = bus.emit(
        PluginEvent.ON_EPOCH_END,
        step=100,
        context_data={'epoch': 1}
    )
    context.epoch = 1
    print(f"   ✓ 事件已派发 (epoch={context.epoch})")

    print("\n✓ 事件派发测试完成")
    return True


def test_plugin_statistics():
    """测试插件统计"""
    print("\n" + "="*80)
    print(" 测试插件统计")
    print("="*80)

    # 创建 PluginBus 并注册插件
    bus = PluginBus()
    bus.register(GRPOPlugin())
    bus.register(EQIReporterPlugin())
    bus.register(RouteOptimizerPlugin())
    bus.compile()

    # 派发一些事件
    for step in range(1, 21):
        bus.emit(
            PluginEvent.ON_BATCH_START,
            step=step,
            context_data={'routing': {'expert_ids': [0, 1, 2]}}
        )
        bus.emit(
            PluginEvent.ON_BATCH_END,
            step=step,
            context_data={'rewards': [0.5, 0.6, 0.4, 0.7]}
        )
        bus.emit(
            PluginEvent.ON_STEP_END,
            step=step,
            context_data={'loss': 0.3}
        )

        if step % 10 == 0:
            bus.emit(
                PluginEvent.ON_STEP_EVAL,
                step=step,
                context_data={'evidence': 1.0, 'utility': 0.5}
            )

    # 获取统计信息
    print("\n1. 插件统计信息:")
    stats = bus.get_statistics()
    print(f"   - 总插件数: {stats['total_plugins']}")
    print(f"   - 活跃插件: {stats['active_plugins']}")
    print(f"   - 总调用次数: {stats['total_invocations']}")
    print(f"   - 总耗时: {stats['total_time_ms']:.2f} ms")

    print("\n2. 各插件详细统计:")
    for name, plugin_stats in stats['plugins'].items():
        print(f"   {name}:")
        print(f"     - 调用次数: {plugin_stats['invocations']}")
        print(f"     - 总耗时: {plugin_stats['total_time_ms']:.2f} ms")
        print(f"     - 平均耗时: {plugin_stats['avg_time_ms']:.2f} ms")
        print(f"     - 状态: {'✓ Active' if plugin_stats['healthy'] else '✗ Disabled'}")

    print("\n✓ 插件统计测试完成")
    return True


def test_plugin_enable_disable():
    """测试插件启用/禁用"""
    print("\n" + "="*80)
    print(" 测试插件启用/禁用")
    print("="*80)

    # 创建 PluginBus 并注册插件
    bus = PluginBus()
    bus.register(GRPOPlugin())
    bus.compile()

    # 测试禁用插件
    print("\n1. 禁用 GRPO 插件...")
    bus.disable_plugin("grpo", reason="test")
    stats = bus.get_statistics()
    assert stats['plugins']['grpo']['healthy'] == False
    print("   ✓ 插件已禁用")

    # 测试启用插件
    print("\n2. 启用 GRPO 插件...")
    bus.enable_plugin("grpo")
    stats = bus.get_statistics()
    assert stats['plugins']['grpo']['healthy'] == True
    print("   ✓ 插件已启用")

    print("\n✓ 插件启用/禁用测试完成")
    return True


def test_conflict_detection():
    """测试冲突检测"""
    print("\n" + "="*80)
    print(" 测试冲突检测")
    print("="*80)

    # 创建一个测试插件，与 GRPO 冲突
    class TestPlugin(PluginBase):
        def get_manifest(self):
            return PluginManifest(
                name="test_rlhf",
                version="1.0.0",
                priority=PluginPriority.RLHF,
                events=[PluginEvent.ON_BATCH_END],
                conflicts=["plugin:grpo"]  # 与 GRPO 冲突
            )

    bus = PluginBus()
    bus.register(GRPOPlugin())
    bus.register(TestPlugin())

    print("\n1. 编译插件（应检测到冲突）...")
    bus.compile(fail_fast=False)

    stats = bus.get_statistics()
    print(f"\n2. 编译结果:")
    print(f"   - 总插件数: {stats['total_plugins']}")
    print(f"   - 活跃插件: {stats['active_plugins']}")
    print(f"   - 禁用插件: {stats['disabled_plugins']}")

    # 至少有一个插件应该被禁用（因为冲突）
    assert stats['disabled_plugins'] >= 1, "应该检测到冲突并禁用插件"

    print("\n✓ 冲突检测测试完成")
    return True


def test_priority_ordering():
    """测试优先级排序"""
    print("\n" + "="*80)
    print(" 测试优先级排序")
    print("="*80)

    bus = PluginBus()
    bus.register(GRPOPlugin())           # Priority 380
    bus.register(EQIReporterPlugin())    # Priority 820
    bus.register(RouteOptimizerPlugin()) # Priority 200

    bus.compile()

    # 检查排序顺序（应该按优先级升序）
    print("\n1. 插件执行顺序（按优先级）:")
    for i, handle in enumerate(bus.ordered_handles):
        print(f"   {i+1}. {handle.manifest.name} (priority={handle.manifest.priority})")

    # 验证排序
    priorities = [h.manifest.priority for h in bus.ordered_handles]
    assert priorities == sorted(priorities), "插件应该按优先级排序"

    print("\n✓ 优先级排序测试完成")
    return True


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print(" APT Plugin System Standalone Tests")
    print("="*80)

    try:
        # 运行所有测试
        tests = [
            ("PluginBus 插件注册", test_plugin_bus_registration),
            ("插件编译", test_plugin_compilation),
            ("事件派发", test_event_dispatch),
            ("插件统计", test_plugin_statistics),
            ("插件启用/禁用", test_plugin_enable_disable),
            ("冲突检测", test_conflict_detection),
            ("优先级排序", test_priority_ordering),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n✗ {test_name} 测试失败: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        # 打印总结
        print("\n" + "="*80)
        print(f" 测试总结: {passed} 通过, {failed} 失败")
        print("="*80 + "\n")

        return failed == 0

    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
