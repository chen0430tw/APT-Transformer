#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试chat命令的导入问题修复
"""

import sys
import os

def safe_print(msg):
    try:
        print(msg)
    except OSError:
        pass

def test_chat_imports():
    """测试chat命令需要的所有导入"""
    safe_print("=" * 70)
    safe_print("Chat命令导入测试")
    safe_print("=" * 70)

    # 测试1: 导入checkpoint模块
    safe_print("\n【1/5】测试 apt.apps.training.checkpoint...")
    try:
        from apt.apps.training.checkpoint import load_model
        safe_print("✓ load_model导入成功")
    except Exception as e:
        safe_print(f"✗ load_model导入失败: {e}")
        import traceback
        safe_print(traceback.format_exc()[:500])
        return False

    # 测试2: 导入CheckpointManager
    safe_print("\n【2/5】测试 apt.trainops.checkpoints.CheckpointManager...")
    try:
        from apt.trainops.checkpoints import CheckpointManager
        safe_print("✓ CheckpointManager导入成功")
    except Exception as e:
        safe_print(f"⚠️  CheckpointManager导入失败: {e}")
        # 这个不是致命错误

    # 测试3: 导入chat模块
    safe_print("\n【3/5】测试 apt.apps.interactive.chat...")
    try:
        from apt.apps.interactive.chat import chat_with_model
        safe_print("✓ chat_with_model导入成功")
    except Exception as e:
        safe_print(f"✗ chat_with_model导入失败: {e}")
        import traceback
        safe_print(traceback.format_exc()[:500])
        return False

    # 测试4: 导入生成器
    safe_print("\n【4/5】测试 apt.core.generation.generator...")
    try:
        from apt.core.generation.generator import generate_natural_text
        safe_print("✓ generate_natural_text导入成功")
    except Exception as e:
        safe_print(f"✗ generate_natural_text导入失败: {e}")
        import traceback
        safe_print(traceback.format_exc()[:300])
        return False

    # 测试5: 导入tokenizer
    safe_print("\n【5/5】测试 tokenizer...")
    try:
        from apt.model.tokenization.chinese_tokenizer_integration import get_tokenizer
        safe_print("✓ get_tokenizer导入成功")
    except Exception as e:
        safe_print(f"✗ get_tokenizer导入失败: {e}")
        import traceback
        safe_print(traceback.format_exc()[:300])
        return False

    safe_print("\n✅ 所有导入测试通过")
    return True

def main():
    """主函数"""
    success = test_chat_imports()

    safe_print("\n" + "=" * 70)
    if success:
        safe_print("✅ Chat命令导入测试通过！")
        safe_print("\n可以尝试运行:")
        safe_print("  python -m apt_model chat")
        return 0
    else:
        safe_print("❌ Chat命令导入测试失败")
        safe_print("\n需要进一步修复循环导入问题")
        return 1

if __name__ == "__main__":
    sys.path.insert(0, '/home/user/APT-Transformer')
    sys.exit(main())
