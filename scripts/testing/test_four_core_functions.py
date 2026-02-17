#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APT-Transformer 四大核心功能快速测试
测试：数据清洗、训练、聊天、评估
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def safe_print(msg):
    """安全打印"""
    try:
        print(msg)
    except OSError:
        pass

def safe_import(module_path, item_name=None):
    """安全导入模块或类"""
    try:
        if item_name:
            exec(f"from {module_path} import {item_name}")
        else:
            __import__(module_path)
        return True
    except Exception as e:
        return False

def test_data_processing():
    """测试数据处理功能"""
    safe_print("\n" + "=" * 70)
    safe_print("【1/4】数据处理 (process-data)")
    safe_print("=" * 70)

    checks = [
        ("命令注册", None, None),
        ("DataProcessor", "apt.core.data.data_processor", "DataProcessor"),
        ("load_external_data", "apt.core.data.external_data", "load_external_data"),
        ("HuggingFaceLoader", "apt.core.data.huggingface_loader", "HuggingFaceLoader"),
        ("数据管道", "apt.trainops.data.data_loading", None),
    ]

    # 特殊处理命令注册检查
    try:
        from apt.apps.cli.command_registry import command_registry
        commands = command_registry.list_commands()
        if 'process-data' in commands:
            safe_print("✓ process-data 命令已注册")
            passed = 1
        else:
            safe_print("✗ process-data 命令未注册")
            passed = 0
    except:
        safe_print("✗ 命令注册表错误")
        passed = 0

    # 检查其他模块
    for name, module, item in checks[1:]:
        if safe_import(module, item):
            safe_print(f"✓ {name}")
            passed += 1
        else:
            safe_print(f"✗ {name}")

    total = len(checks)
    safe_print(f"\n结果: {passed}/{total} 通过")
    return passed == total

def test_training():
    """测试训练功能"""
    safe_print("\n" + "=" * 70)
    safe_print("【2/4】训练 (train)")
    safe_print("=" * 70)

    checks = [
        ("命令注册", None, None),
        ("Trainer引擎", "apt.trainops.engine.trainer", "Trainer"),
        ("APTModel架构", "apt.model.architectures.apt_model", "APTModel"),
        ("load_profile配置", "apt.core.config", "load_profile"),
        ("Checkpoint管理", "apt.trainops.checkpoints.checkpoint", None),
    ]

    # 特殊处理命令注册检查
    try:
        from apt.apps.cli.command_registry import command_registry
        commands = command_registry.list_commands()
        train_cmds = [c for c in commands if 'train' in c]
        safe_print(f"✓ {len(train_cmds)} 个训练命令已注册")
        passed = 1
    except:
        safe_print("✗ 命令注册表错误")
        passed = 0

    # 检查其他模块
    for name, module, item in checks[1:]:
        if safe_import(module, item):
            safe_print(f"✓ {name}")
            passed += 1
        else:
            safe_print(f"✗ {name}")

    total = len(checks)
    safe_print(f"\n结果: {passed}/{total} 通过")
    return passed == total

def test_chat():
    """测试聊天功能"""
    safe_print("\n" + "=" * 70)
    safe_print("【3/4】聊天 (chat)")
    safe_print("=" * 70)

    checks = [
        ("命令注册", None, None),
        ("APTModel推理", "apt.model.architectures.apt_model", "APTModel"),
        ("GenerationEvaluator", "apt.apps.generation.evaluator", "GenerationEvaluator"),
        ("ChineseTokenizer", "apt.model.tokenization.chinese_tokenizer_integration", "ChineseTokenizer"),
    ]

    # 特殊处理命令注册检查
    try:
        from apt.apps.cli.command_registry import command_registry
        commands = command_registry.list_commands()
        if 'chat' in commands:
            safe_print("✓ chat 命令已注册")
            passed = 1
        else:
            safe_print("✗ chat 命令未注册")
            passed = 0
    except:
        safe_print("✗ 命令注册表错误")
        passed = 0

    # 检查其他模块
    for name, module, item in checks[1:]:
        if safe_import(module, item):
            safe_print(f"✓ {name}")
            passed += 1
        else:
            safe_print(f"✗ {name}")

    total = len(checks)
    safe_print(f"\n结果: {passed}/{total} 通过")
    return passed == total

def test_evaluation():
    """测试评估功能"""
    safe_print("\n" + "=" * 70)
    safe_print("【4/4】评估 (evaluate)")
    safe_print("=" * 70)

    checks = [
        ("命令注册", None, None),
        ("GenerationEvaluator", "apt.apps.generation.evaluator", "GenerationEvaluator"),
        ("ModelEvaluator插件", "apt.apps.plugins.evaluation.model_evaluator_plugin", "ModelEvaluator"),
        ("ModelComparison插件", "apt.apps.plugins.evaluation.model_comparison_plugin", "ModelComparison"),
    ]

    # 特殊处理命令注册检查
    try:
        from apt.apps.cli.command_registry import command_registry
        commands = command_registry.list_commands()
        if 'evaluate' in commands:
            safe_print("✓ evaluate 命令已注册")
            passed = 1
        else:
            safe_print("✗ evaluate 命令未注册")
            passed = 0
    except:
        safe_print("✗ 命令注册表错误")
        passed = 0

    # 检查其他模块
    for name, module, item in checks[1:]:
        if safe_import(module, item):
            safe_print(f"✓ {name}")
            passed += 1
        else:
            safe_print(f"✗ {name}")

    total = len(checks)
    safe_print(f"\n结果: {passed}/{total} 通过")
    return passed == total

def main():
    """主测试函数"""
    safe_print("=" * 70)
    safe_print("APT-Transformer 四大核心功能快速测试")
    safe_print("=" * 70)

    results = {
        '数据处理': test_data_processing(),
        '训练': test_training(),
        '聊天': test_chat(),
        '评估': test_evaluation(),
    }

    # 总结
    safe_print("\n" + "=" * 70)
    safe_print("总结")
    safe_print("=" * 70)

    for name, result in results.items():
        status = "✅" if result else "❌"
        safe_print(f"{status} {name}")

    passed = sum(1 for v in results.values() if v)
    failed = 4 - passed

    safe_print(f"\n通过: {passed}/4 | 失败: {failed}/4")

    if failed == 0:
        safe_print("\n🎉 所有核心功能就绪！")
        return 0
    else:
        safe_print("\n⚠️  部分功能需要检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())
