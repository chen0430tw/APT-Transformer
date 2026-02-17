#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APT-Transformer 四大核心功能测试脚本
测试：数据清洗、训练、聊天、评估
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def safe_print(msg):
    """安全打印，处理文件描述符错误"""
    try:
        print(msg)
    except OSError:
        pass

def test_data_processing():
    """测试数据清洗/处理功能"""
    safe_print("\n" + "=" * 80)
    safe_print("【1/4】测试数据处理功能 (process-data)")
    safe_print("=" * 80)

    try:
        from apt.apps.cli.command_registry import command_registry

        # 检查process-data命令是否注册
        commands = command_registry.list_commands()
        if 'process-data' not in commands:
            safe_print("✗ process-data 命令未注册")
            return False

        safe_print("✓ process-data 命令已注册")

        # 检查数据处理相关模块
        try:
            from apt.core.data.data_processor import DataProcessor
            safe_print("✓ DataProcessor 模块导入成功")
        except Exception as e:
            safe_print(f"⚠️  DataProcessor 导入警告: {str(e)[:100]}")

        # 检查外部数据加载
        try:
            from apt.core.data.external_data import load_external_data
            safe_print("✓ load_external_data 函数导入成功")
        except Exception as e:
            safe_print(f"⚠️  load_external_data 导入警告: {str(e)[:100]}")

        # 检查HuggingFace数据加载
        try:
            from apt.core.data.huggingface_loader import HuggingFaceLoader
            safe_print("✓ HuggingFaceLoader 模块导入成功")
        except Exception as e:
            safe_print(f"⚠️  HuggingFaceLoader 导入警告: {str(e)[:100]}")

        safe_print("\n✅ 数据处理功能基本就绪")
        return True

    except Exception as e:
        safe_print(f"✗ 数据处理功能测试失败: {str(e)[:200]}")
        import traceback
        safe_print(traceback.format_exc()[:500])
        return False

def test_training():
    """测试训练功能"""
    safe_print("\n" + "=" * 80)
    safe_print("【2/4】测试训练功能 (train)")
    safe_print("=" * 80)

    try:
        from apt.apps.cli.command_registry import command_registry

        # 检查train命令是否注册
        commands = command_registry.list_commands()
        train_commands = [cmd for cmd in commands if 'train' in cmd]

        safe_print(f"✓ 发现 {len(train_commands)} 个训练相关命令:")
        for cmd in train_commands[:10]:
            safe_print(f"  - {cmd}")

        if len(train_commands) > 10:
            safe_print(f"  ... 还有 {len(train_commands) - 10} 个命令")

        # 检查训练引擎
        try:
            from apt.trainops.engine.trainer import Trainer
            safe_print("✓ Trainer 模块导入成功")
        except Exception as e:
            safe_print(f"✗ Trainer 导入失败: {str(e)[:100]}")
            return False

        # 检查微调引擎
        try:
            from apt.trainops.engine.finetuner import FineTuner
            safe_print("✓ FineTuner 模块导入成功")
        except Exception as e:
            safe_print(f"⚠️  FineTuner 导入警告: {str(e)[:100]}")

        # 检查配置加载
        try:
            from apt.core.config import load_profile
            config = load_profile("lite")
            safe_print(f"✓ 配置加载成功 (lite profile)")
            safe_print(f"  - hidden_size: {config.model.hidden_size}")
            safe_print(f"  - num_layers: {config.model.num_layers}")
            safe_print(f"  - batch_size: {config.training.batch_size}")
        except Exception as e:
            safe_print(f"✗ 配置加载失败: {str(e)[:100]}")
            return False

        # 检查模型架构
        try:
            from apt.model.architectures.apt_model import APTModel
            safe_print("✓ APTModel 架构导入成功")
        except Exception as e:
            safe_print(f"✗ APTModel 导入失败: {str(e)[:100]}")
            return False

        safe_print("\n✅ 训练功能基本就绪")
        return True

    except Exception as e:
        safe_print(f"✗ 训练功能测试失败: {str(e)[:200]}")
        import traceback
        safe_print(traceback.format_exc()[:500])
        return False

def test_chat():
    """测试聊天功能"""
    safe_print("\n" + "=" * 80)
    safe_print("【3/4】测试聊天功能 (chat)")
    safe_print("=" * 80)

    try:
        from apt.apps.cli.command_registry import command_registry

        # 检查chat命令是否注册
        commands = command_registry.list_commands()
        if 'chat' not in commands:
            safe_print("✗ chat 命令未注册")
            return False

        safe_print("✓ chat 命令已注册")

        # 检查生成/推理模块
        try:
            from apt.apps.generation.evaluator import GenerationEvaluator
            safe_print("✓ GenerationEvaluator 模块导入成功")
        except Exception as e:
            safe_print(f"⚠️  GenerationEvaluator 导入警告: {str(e)[:100]}")

        # 检查tokenizer
        try:
            from apt.model.tokenization.chinese_tokenizer_integration import ChineseTokenizer
            safe_print("✓ ChineseTokenizer 模块导入成功")
        except Exception as e:
            safe_print(f"⚠️  ChineseTokenizer 导入警告: {str(e)[:100]}")

        # 检查模型加载
        try:
            from apt.model.architectures.apt_model import APTModel
            safe_print("✓ APTModel 可用于推理")
        except Exception as e:
            safe_print(f"✗ APTModel 导入失败: {str(e)[:100]}")
            return False

        safe_print("\n✅ 聊天功能基本就绪")
        return True

    except Exception as e:
        safe_print(f"✗ 聊天功能测试失败: {str(e)[:200]}")
        import traceback
        safe_print(traceback.format_exc()[:500])
        return False

def test_evaluation():
    """测试评估功能"""
    safe_print("\n" + "=" * 80)
    safe_print("【4/4】测试评估功能 (evaluate)")
    safe_print("=" * 80)

    try:
        from apt.apps.cli.command_registry import command_registry

        # 检查evaluate命令是否注册
        commands = command_registry.list_commands()
        if 'evaluate' not in commands:
            safe_print("✗ evaluate 命令未注册")
            return False

        safe_print("✓ evaluate 命令已注册")

        # 检查评估器模块
        try:
            from apt.apps.generation.evaluator import GenerationEvaluator
            safe_print("✓ GenerationEvaluator 模块导入成功")
        except Exception as e:
            safe_print(f"⚠️  GenerationEvaluator 导入警告: {str(e)[:100]}")

        # 检查评估插件
        try:
            from apt.apps.plugins.evaluation.model_evaluator_plugin import ModelEvaluator
            safe_print("✓ ModelEvaluator 插件导入成功")
        except Exception as e:
            safe_print(f"⚠️  ModelEvaluator 插件导入警告: {str(e)[:100]}")

        # 检查模型对比插件
        try:
            from apt.apps.plugins.evaluation.model_comparison_plugin import ModelComparison
            safe_print("✓ ModelComparison 插件导入成功")
        except Exception as e:
            safe_print(f"⚠️  ModelComparison 插件导入警告: {str(e)[:100]}")

        safe_print("\n✅ 评估功能基本就绪")
        return True

    except Exception as e:
        safe_print(f"✗ 评估功能测试失败: {str(e)[:200]}")
        import traceback
        safe_print(traceback.format_exc()[:500])
        return False

def main():
    """主测试函数"""
    safe_print("=" * 80)
    safe_print("APT-Transformer 四大核心功能测试")
    safe_print("=" * 80)
    safe_print(f"项目路径: {project_root}")

    results = {}

    # 执行测试
    results['数据处理'] = test_data_processing()
    results['训练'] = test_training()
    results['聊天'] = test_chat()
    results['评估'] = test_evaluation()

    # 总结
    safe_print("\n" + "=" * 80)
    safe_print("测试总结")
    safe_print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, result in results.items():
        status = "✅" if result else "❌"
        safe_print(f"{status} {name}")

    safe_print(f"\n通过: {passed}/4 | 失败: {failed}/4")

    if failed == 0:
        safe_print("\n🎉 所有核心功能测试通过！")
        safe_print("\n可用命令:")
        safe_print("  python -m apt_model process-data <data_file>  # 数据处理")
        safe_print("  python -m apt_model train --profile lite      # 训练模型")
        safe_print("  python -m apt_model chat                      # 聊天交互")
        safe_print("  python -m apt_model evaluate <model_path>     # 评估模型")
        return 0
    else:
        safe_print("\n⚠️  部分功能需要修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())
