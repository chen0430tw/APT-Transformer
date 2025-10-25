#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Console Core Tests (控制台核心测试)

测试控制台核心模块的功能。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apt_model.console.core import ConsoleCore, initialize_console
from apt_model.console.module_manager import ModuleManager, ModuleStatus


def test_module_manager():
    """测试模块管理器"""
    print("\n" + "="*80)
    print(" 测试模块管理器")
    print("="*80)

    # 创建模块管理器
    manager = ModuleManager()

    # 测试模块列表
    print("\n1. 测试模块列表:")
    modules = manager.list_modules()
    print(f"  - 总共注册了 {len(modules)} 个模块")

    # 测试按类别列出
    print("\n2. 按类别列出模块:")
    categories = set(m.category for m in modules)
    for cat in sorted(categories):
        cat_modules = manager.list_modules(category=cat)
        print(f"  - {cat}: {len(cat_modules)} 个模块")

    # 测试模块信息
    print("\n3. 测试模块信息:")
    if modules:
        first_module = modules[0]
        info = manager.get_module_info(first_module.name)
        print(f"  - 模块名称: {info.name}")
        print(f"  - 版本: {info.version}")
        print(f"  - 描述: {info.description}")
        print(f"  - 类别: {info.category}")

    # 测试依赖解析
    print("\n4. 测试依赖解析:")
    for module in modules[:3]:
        if module.dependencies:
            deps = manager.resolve_dependencies(module.name)
            print(f"  - {module.name}: {' -> '.join(deps)}")

    print("\n✓ 模块管理器测试完成")
    return True


def test_console_core():
    """测试控制台核心"""
    print("\n" + "="*80)
    print(" 测试控制台核心")
    print("="*80)

    # 初始化控制台
    print("\n1. 初始化控制台...")
    console = initialize_console(auto_start=False)
    print("  ✓ 控制台初始化成功")

    # 启动控制台
    print("\n2. 启动控制台...")
    success = console.start(auto_load_modules=True)
    if success:
        print("  ✓ 控制台启动成功")
    else:
        print("  ✗ 控制台启动失败")
        return False

    # 测试配置
    print("\n3. 测试配置管理...")
    console.set_config("test.key", "test_value")
    value = console.get_config("test.key")
    if value == "test_value":
        print(f"  ✓ 配置设置和读取成功: {value}")
    else:
        print(f"  ✗ 配置读取失败")

    # 测试模块状态
    print("\n4. 模块状态:")
    modules = console.module_manager.list_modules()
    ready_count = len([m for m in modules if console.module_manager.get_status(m.name) == ModuleStatus.READY])
    print(f"  - 就绪模块: {ready_count}/{len(modules)}")

    # 打印状态
    print("\n5. 打印控制台状态:")
    console.print_status()

    # 停止控制台
    print("\n6. 停止控制台...")
    console.stop()
    print("  ✓ 控制台已停止")

    print("\n✓ 控制台核心测试完成")
    return True


def test_console_commands():
    """测试控制台命令"""
    print("\n" + "="*80)
    print(" 测试控制台命令")
    print("="*80)

    # 初始化并启动控制台
    console = initialize_console(auto_start=True)

    # 测试列出命令
    print("\n1. 列出所有命令:")
    commands = console.list_commands()
    print(f"  - 总共注册了 {len(commands)} 个命令")

    # 测试按类别列出
    print("\n2. 按类别列出命令:")
    console_commands = console.list_commands(category="console")
    if console_commands:
        print(f"  - Console 类别: {len(console_commands)} 个命令")
        for cmd in console_commands[:5]:
            print(f"    - {cmd}")

    print("\n✓ 控制台命令测试完成")
    return True


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print(" APT Console Core Tests")
    print("="*80)

    try:
        # 运行所有测试
        tests = [
            ("模块管理器", test_module_manager),
            ("控制台核心", test_console_core),
            ("控制台命令", test_console_commands),
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
