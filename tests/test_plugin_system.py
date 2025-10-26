#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plugin System Integration Tests

测试插件系统的完整集成，包括：
- Console Core 与 PluginBus 集成
- 插件注册和编译
- 事件派发
- 冲突检测
- 统计信息
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apt_model.console.core import ConsoleCore, initialize_console
from apt_model.console.plugin_standards import PluginEvent
from apt_model.console.plugins import (
    GRPOPlugin,
    EQIReporterPlugin,
    RouteOptimizerPlugin
)


def test_plugin_registration():
    """测试插件注册"""
    print("\n" + "="*80)
    print(" 测试插件注册")
    print("="*80)

    # 创建控制台
    console = ConsoleCore(config={})

    # 注册插件
    print("\n1. 注册 GRPO 插件...")
    grpo = GRPOPlugin()
    console.register_plugin(grpo)
    print("   ✓ GRPO 插件注册成功")

    print("\n2. 注册 EQI Reporter 插件...")
    eqi_reporter = EQIReporterPlugin()
    console.register_plugin(eqi_reporter)
    print("   ✓ EQI Reporter 插件注册成功")

    print("\n3. 注册 Route Optimizer 插件...")
    route_optimizer = RouteOptimizerPlugin()
    console.register_plugin(route_optimizer)
    print("   ✓ Route Optimizer 插件注册成功")

    # 检查插件状态
    stats = console.get_plugin_statistics()
    print(f"\n✓ 插件注册测试完成: {stats['total_plugins']} 个插件已注册")

    return True


def test_plugin_compilation():
    """测试插件编译（静态冲突检查）"""
    print("\n" + "="*80)
    print(" 测试插件编译")
    print("="*80)

    # 创建控制台并注册插件
    console = ConsoleCore(config={})
    console.register_plugin(GRPOPlugin())
    console.register_plugin(EQIReporterPlugin())
    console.register_plugin(RouteOptimizerPlugin())

    # 编译插件
    print("\n1. 执行插件编译（静态冲突检查）...")
    console.compile_plugins(fail_fast=False)
    print("   ✓ 插件编译成功")

    # 打印插件状态
    print("\n2. 插件状态:")
    console.print_plugin_status()

    print("\n✓ 插件编译测试完成")
    return True


def test_event_dispatch():
    """测试事件派发"""
    print("\n" + "="*80)
    print(" 测试事件派发")
    print("="*80)

    # 创建控制台并注册插件
    console = ConsoleCore(config={})
    console.register_plugin(GRPOPlugin())
    console.register_plugin(EQIReporterPlugin())
    console.register_plugin(RouteOptimizerPlugin())
    console.compile_plugins()

    # 测试 on_batch_start 事件
    print("\n1. 派发 on_batch_start 事件...")
    context = console.emit_event(
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
    context = console.emit_event(
        PluginEvent.ON_BATCH_END,
        step=1,
        context_data={
            'rewards': [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6]
        }
    )
    print(f"   ✓ 事件已派发")

    # 测试 on_step_end 事件
    print("\n3. 派发 on_step_end 事件...")
    context = console.emit_event(
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
    print(f"   - 公共数据: {context.data}")

    # 测试 on_step_eval 事件
    print("\n4. 派发 on_step_eval 事件...")
    context = console.emit_event(
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
    context = console.emit_event(
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

    # 创建控制台并注册插件
    console = ConsoleCore(config={})
    console.register_plugin(GRPOPlugin())
    console.register_plugin(EQIReporterPlugin())
    console.register_plugin(RouteOptimizerPlugin())
    console.compile_plugins()

    # 派发一些事件
    for step in range(1, 21):
        console.emit_event(
            PluginEvent.ON_BATCH_START,
            step=step,
            context_data={'routing': {'expert_ids': [0, 1, 2]}}
        )
        console.emit_event(
            PluginEvent.ON_BATCH_END,
            step=step,
            context_data={'rewards': [0.5, 0.6, 0.4, 0.7]}
        )
        console.emit_event(
            PluginEvent.ON_STEP_END,
            step=step,
            context_data={'loss': 0.3}
        )

        if step % 10 == 0:
            console.emit_event(
                PluginEvent.ON_STEP_EVAL,
                step=step,
                context_data={'evidence': 1.0, 'utility': 0.5}
            )

    # 获取统计信息
    print("\n1. 插件统计信息:")
    stats = console.get_plugin_statistics()
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

    # 创建控制台并注册插件
    console = ConsoleCore(config={})
    console.register_plugin(GRPOPlugin())
    console.compile_plugins()

    # 测试禁用插件
    print("\n1. 禁用 GRPO 插件...")
    console.disable_plugin("grpo", reason="test")
    stats = console.get_plugin_statistics()
    assert stats['plugins']['grpo']['healthy'] == False
    print("   ✓ 插件已禁用")

    # 测试启用插件
    print("\n2. 启用 GRPO 插件...")
    console.enable_plugin("grpo")
    stats = console.get_plugin_statistics()
    assert stats['plugins']['grpo']['healthy'] == True
    print("   ✓ 插件已启用")

    print("\n✓ 插件启用/禁用测试完成")
    return True


def test_console_integration():
    """测试 Console Core 完整集成"""
    print("\n" + "="*80)
    print(" 测试 Console Core 集成")
    print("="*80)

    # 初始化控制台（不自动加载模块）
    print("\n1. 初始化控制台...")
    console = initialize_console(auto_start=False)
    print("   ✓ 控制台初始化成功")

    # 注册插件
    print("\n2. 注册插件...")
    console.register_plugin(GRPOPlugin())
    console.register_plugin(EQIReporterPlugin())
    console.register_plugin(RouteOptimizerPlugin())
    print("   ✓ 插件注册完成")

    # 启动控制台（包括插件编译）
    print("\n3. 启动控制台...")
    console.start(auto_load_modules=False, auto_load_plugins=True)
    print("   ✓ 控制台启动成功")

    # 打印状态
    print("\n4. 控制台状态:")
    console.print_status()

    # 停止控制台
    print("\n5. 停止控制台...")
    console.stop()
    print("   ✓ 控制台已停止")

    print("\n✓ Console Core 集成测试完成")
    return True


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print(" APT Plugin System Integration Tests")
    print("="*80)

    try:
        # 运行所有测试
        tests = [
            ("插件注册", test_plugin_registration),
            ("插件编译", test_plugin_compilation),
            ("事件派发", test_event_dispatch),
            ("插件统计", test_plugin_statistics),
            ("插件启用/禁用", test_plugin_enable_disable),
            ("Console Core 集成", test_console_integration),
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
