#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载已保存的 HLBD 模型并进行推理

使用方法:
    python tests/load_hlbd_model.py [模型路径]

如果不提供模型路径，会自动查找最新保存的模型。
"""

import sys
import os
import torch
import glob

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from test_hlbd_quick_learning import (
    load_model_and_tokenizer,
    generate_text,
    test_generation,
)


def find_latest_model(save_dir):
    """查找最新保存的模型"""
    pattern = os.path.join(save_dir, 'hlbd_model_*.pt')
    model_files = glob.glob(pattern)

    if not model_files:
        return None

    # 按修改时间排序，返回最新的
    latest = max(model_files, key=os.path.getmtime)
    return latest


def main():
    """主函数"""
    print("\n🔍 HLBD 模型加载器")
    print("="*70)

    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 确定模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"\n使用指定的模型: {model_path}")
    else:
        # 自动查找最新模型
        save_dir = os.path.join(project_root, 'tests', 'saved_models')
        model_path = find_latest_model(save_dir)

        if model_path is None:
            print(f"\n❌ 未找到已保存的模型！")
            print(f"   保存目录: {save_dir}")
            print(f"\n请先运行训练脚本:")
            print(f"   python tests/test_hlbd_quick_learning.py")
            return 1

        print(f"\n使用最新的模型: {os.path.basename(model_path)}")

    # 加载模型
    try:
        model, tokenizer, training_info = load_model_and_tokenizer(model_path, device)
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 测试用例
    print("\n" + "="*70)
    print("🧪 开始推理测试")
    print("="*70)

    test_cases = [
        # Emoji 测试
        ("🌧️", "下雨"),
        ("❤️", "我爱你"),
        ("🍽️", "吃饭"),
        ("📖", "看书"),
        ("😴", "睡觉"),

        # 英文测试
        ("I love you", "我爱你"),
        ("It's raining", "下雨"),
        ("Read a book", "看书"),

        # 拼音测试
        ("wǒ ài nǐ", "我爱你"),
        ("xià yǔ", "下雨"),
        ("chī fàn", "吃饭"),

        # 中文测试
        ("下雨", "天气"),
        ("我爱你", "情感"),
        ("吃饭", "生活"),
    ]

    test_generation(model, tokenizer, test_cases, device)

    # 交互式推理
    print("\n" + "="*70)
    print("💬 交互式推理模式 (输入 'quit' 退出)")
    print("="*70)

    while True:
        try:
            user_input = input("\n请输入文本: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break

            if not user_input:
                continue

            # 生成
            generated = generate_text(model, tokenizer, user_input, device)
            print(f"生成: {generated}")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
