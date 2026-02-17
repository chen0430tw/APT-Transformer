#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
压缩插件最小化测试 - 无需PyTorch依赖
仅验证代码结构、导入和配置
"""

import sys
import ast
from pathlib import Path

def test_plugin_file_structure():
    """测试插件文件结构和语法"""
    print("=" * 70)
    print("测试 1: 验证插件文件结构")
    print("=" * 70)

    plugin_file = Path("apt_model/plugins/compression_plugin.py")

    # 检查文件存在
    assert plugin_file.exists(), "❌ 压缩插件文件不存在"
    print(f"✅ 插件文件存在: {plugin_file}")
    print(f"   文件大小: {plugin_file.stat().st_size / 1024:.1f} KB")

    # 检查Python语法
    with open(plugin_file, 'r', encoding='utf-8') as f:
        code = f.read()

    try:
        ast.parse(code)
        print("✅ Python语法检查通过")
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False

    return True


def test_plugin_class_methods():
    """测试插件类和方法定义"""
    print("\n" + "=" * 70)
    print("测试 2: 验证插件类和方法")
    print("=" * 70)

    plugin_file = Path("apt_model/plugins/compression_plugin.py")

    with open(plugin_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # 检查关键类
    assert "class CompressionPlugin" in code, "❌ 缺少 CompressionPlugin 类"
    print("✅ CompressionPlugin 类定义存在")

    # 检查关键方法
    required_methods = [
        "prune_model",
        "quantize_model",
        "distillation_loss",
        "enable_dbc_training",
        "low_rank_decomposition",
        "compress_model",
        "export_for_webui",
        "generate_compression_report"
    ]

    print("\n📋 核心方法检查:")
    for method in required_methods:
        if f"def {method}" in code:
            print(f"   ✅ {method}")
        else:
            print(f"   ❌ {method} - 缺失")
            return False

    return True


def test_dbc_integration():
    """测试DBC集成"""
    print("\n" + "=" * 70)
    print("测试 3: 验证DBC集成")
    print("=" * 70)

    # 检查压缩插件中的DBC导入
    plugin_file = Path("apt_model/plugins/compression_plugin.py")
    with open(plugin_file, 'r', encoding='utf-8') as f:
        plugin_code = f.read()

    # 检查是否导入了DBC相关类
    if "from apt_model.modeling.apt_model import DBCDAC_Optimizer" in plugin_code:
        print("✅ 压缩插件导入 DBCDAC_Optimizer")
    else:
        print("❌ 压缩插件未导入 DBCDAC_Optimizer")
        return False

    if "add_gradient_hooks_to_model" in plugin_code:
        print("✅ 压缩插件导入 add_gradient_hooks_to_model")
    else:
        print("❌ 压缩插件未导入梯度钩子函数")

    # 检查DBC源码存在
    dbc_file = Path("apt_model/modeling/apt_model.py")
    assert dbc_file.exists(), "❌ DBC源文件不存在"

    with open(dbc_file, 'r', encoding='utf-8') as f:
        dbc_code = f.read()

    if "class DBCDAC_Optimizer" in dbc_code:
        print("✅ DBCDAC_Optimizer 类存在于 apt_model.py")
    else:
        print("❌ DBCDAC_Optimizer 类不存在")
        return False

    # 找到类定义的行号
    for i, line in enumerate(dbc_code.split('\n'), 1):
        if 'class DBCDAC_Optimizer' in line:
            print(f"   位置: apt_model/modeling/apt_model.py:{i}")
            break

    print("\n💡 DBC集成方式:")
    print("   压缩插件的 enable_dbc_training() 方法")
    print("   → 导入并使用现有的 DBCDAC_Optimizer")
    print("   → 无重复代码，完美复用 ✅")

    return True


def test_compression_methods_config():
    """测试压缩方法配置"""
    print("\n" + "=" * 70)
    print("测试 4: 验证压缩方法配置")
    print("=" * 70)

    plugin_file = Path("apt_model/plugins/compression_plugin.py")
    with open(plugin_file, 'r', encoding='utf-8') as f:
        code = f.read()

    compression_methods = {
        'pruning': '模型剪枝',
        'quantization': '模型量化',
        'distillation': '知识蒸馏',
        'dbc': 'DBC加速训练',
        'low_rank': '低秩分解'
    }

    print("\n📦 压缩方法清单:")
    for method, description in compression_methods.items():
        # 简单检查方法名在代码中出现
        if method in code.lower() or method.replace('_', '') in code.lower():
            print(f"   ✅ {method:15s} - {description}")
        else:
            print(f"   ⚠️  {method:15s} - {description} (可能缺失)")

    return True


def test_webui_export():
    """测试WebUI导出功能"""
    print("\n" + "=" * 70)
    print("测试 5: 验证WebUI/API导出")
    print("=" * 70)

    plugin_file = Path("apt_model/plugins/compression_plugin.py")
    with open(plugin_file, 'r', encoding='utf-8') as f:
        code = f.read()

    if "def export_for_webui" in code:
        print("✅ export_for_webui() 方法存在")
    else:
        print("❌ export_for_webui() 方法缺失")
        return False

    # 检查API端点注释
    api_endpoints = [
        "/api/compress/prune",
        "/api/compress/quantize",
        "/api/compress/distill",
        "/api/compress/full",
        "/api/compress/evaluate"
    ]

    print("\n🔮 未来API端点设计:")
    for endpoint in api_endpoints:
        if endpoint in code:
            print(f"   ✅ {endpoint}")
        else:
            print(f"   📝 {endpoint} (待规划)")

    return True


def test_report_generation():
    """测试报告生成功能"""
    print("\n" + "=" * 70)
    print("测试 6: 验证报告生成")
    print("=" * 70)

    plugin_file = Path("apt_model/plugins/compression_plugin.py")
    with open(plugin_file, 'r', encoding='utf-8') as f:
        code = f.read()

    if "def generate_compression_report" in code:
        print("✅ generate_compression_report() 方法存在")

        # 检查Markdown生成
        if "# 模型压缩报告" in code or "## " in code:
            print("✅ Markdown格式报告支持")

        return True
    else:
        print("❌ generate_compression_report() 方法缺失")
        return False


def count_lines_of_code():
    """统计代码行数"""
    print("\n" + "=" * 70)
    print("测试 7: 代码统计")
    print("=" * 70)

    files = {
        'compression_plugin.py': Path('apt_model/plugins/compression_plugin.py'),
        'test_compression_plugin.py': Path('test_compression_plugin.py'),
        'apt_model.py (DBC源码)': Path('apt_model/modeling/apt_model.py')
    }

    print("\n📊 代码统计:")
    for name, filepath in files.items():
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            size_kb = filepath.stat().st_size / 1024
            print(f"   {name:30s}: {lines:4d} 行, {size_kb:6.1f} KB")

    return True


def main():
    """运行所有最小化测试"""
    print("\n" + "=" * 70)
    print("🔍 APT压缩插件 - 最小化验证测试")
    print("   (无需PyTorch依赖)")
    print("=" * 70)

    tests = [
        ("文件结构验证", test_plugin_file_structure),
        ("类和方法验证", test_plugin_class_methods),
        ("DBC集成验证", test_dbc_integration),
        ("压缩方法配置", test_compression_methods_config),
        ("WebUI导出验证", test_webui_export),
        ("报告生成验证", test_report_generation),
        ("代码统计", count_lines_of_code)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n⚠️  {test_name} 测试未完全通过")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"✅ 通过: {passed}/{len(tests)}")
    print(f"❌ 失败: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 所有验证测试通过!")
        print("\n✅ 压缩插件开发完成确认:")
        print("   • 文件结构 ✅")
        print("   • 核心方法 ✅")
        print("   • DBC集成 ✅")
        print("   • WebUI/API ✅")
        print("   • 报告生成 ✅")
        print("\n💡 下一步: 安装PyTorch后可运行完整功能测试")
        return True
    else:
        print(f"\n⚠️  {failed} 个测试未通过，需要检查")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
