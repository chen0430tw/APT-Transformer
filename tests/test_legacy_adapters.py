"""
测试遗留插件适配器

验证8个插件的适配器是否正常工作
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from apt_model.console.core import ConsoleCore
from apt_model.console.legacy_plugins.adapters import (
    get_all_legacy_adapters,
    get_adapter,
    LEGACY_ADAPTERS
)


def test_adapter_creation():
    """测试适配器创建"""
    print("="*60)
    print("测试1: 适配器创建")
    print("="*60)

    for name in LEGACY_ADAPTERS.keys():
        try:
            adapter = get_adapter(name)
            manifest = adapter.get_manifest()

            print(f"\n✅ {name}")
            print(f"   优先级: {manifest.priority}")
            print(f"   类别: {manifest.category}")
            print(f"   事件: {manifest.events}")
            print(f"   描述: {manifest.description[:60]}...")

        except Exception as e:
            print(f"\n❌ {name}: {e}")

    print("\n" + "="*60)


def test_console_integration():
    """测试Console Core集成"""
    print("\n" + "="*60)
    print("测试2: Console Core集成")
    print("="*60)

    try:
        # 创建Console Core
        core = ConsoleCore(config={
            'engine_version': '1.0.0',
            'plugins': {
                'enable_eqi': False
            }
        })

        print("\n✅ Console Core创建成功")

        # 加载所有适配器（不传递实际配置，避免依赖问题）
        print("\n加载适配器...")

        loaded_count = 0
        failed_plugins = []

        for name in LEGACY_ADAPTERS.keys():
            try:
                adapter = get_adapter(name)
                core.register_plugin(adapter)
                loaded_count += 1
                print(f"  ✅ {name}")
            except Exception as e:
                failed_plugins.append((name, str(e)))
                print(f"  ⚠️ {name}: {e}")

        # 编译插件
        print("\n编译插件...")
        core.compile_plugins(fail_fast=False)

        # 获取统计信息
        stats = core.get_plugin_statistics()

        print(f"\n{'='*60}")
        print(f"插件统计:")
        print(f"  总插件数: {stats['total_plugins']}")
        print(f"  活跃插件: {stats['active_plugins']}")
        print(f"  禁用插件: {stats['disabled_plugins']}")
        print(f"  成功加载: {loaded_count}/8")

        if failed_plugins:
            print(f"\n失败的插件:")
            for name, error in failed_plugins:
                print(f"  - {name}: {error}")

        print("="*60)

        return core

    except Exception as e:
        print(f"\n❌ Console集成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_adapter_methods():
    """测试适配器方法"""
    print("\n" + "="*60)
    print("测试3: 适配器方法测试")
    print("="*60)

    # 测试一个简单的适配器
    try:
        # 我们使用一个不需要外部依赖的插件进行测试
        # 由于所有插件都可能有依赖，我们只测试适配器本身的功能

        print("\n测试适配器接口...")

        # 测试get_adapter
        adapter = get_adapter("data_processors")
        if adapter:
            print("✅ get_adapter() 工作正常")

            # 测试manifest
            manifest = adapter.get_manifest()
            print(f"✅ get_manifest() 返回: {manifest.name}")

            # 测试get_legacy_plugin
            legacy = adapter.get_legacy_plugin()
            print(f"✅ get_legacy_plugin() 返回: {type(legacy).__name__}")

        # 测试get_all_legacy_adapters
        all_adapters = get_all_legacy_adapters()
        print(f"\n✅ get_all_legacy_adapters() 返回 {len(all_adapters)} 个适配器")

    except Exception as e:
        print(f"\n❌ 方法测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_event_compatibility():
    """测试事件兼容性"""
    print("\n" + "="*60)
    print("测试4: 事件兼容性测试")
    print("="*60)

    # 测试适配器是否正确实现了PluginBase的事件方法
    adapter = get_adapter("data_processors")

    events_to_test = [
        'on_init',
        'on_batch_start',
        'on_batch_end',
        'on_step_start',
        'on_step_end',
        'on_decode',
        'on_shutdown'
    ]

    print("\n检查事件方法...")
    for event in events_to_test:
        if hasattr(adapter, event):
            print(f"  ✅ {event}")
        else:
            print(f"  ❌ {event} 缺失")


def main():
    """主测试函数"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║      APT遗留插件适配器测试                                ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # 运行所有测试
    test_adapter_creation()
    test_console_integration()
    test_adapter_methods()
    test_event_compatibility()

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
