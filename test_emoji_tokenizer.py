#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 tokenizer 对 emoji 的处理
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入 ChineseTokenizer，避免触发 __init__.py 中的 torch 导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "chinese_tokenizer",
    os.path.join(os.path.dirname(__file__), "apt_model/modeling/chinese_tokenizer.py")
)
chinese_tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chinese_tokenizer_module)
ChineseTokenizer = chinese_tokenizer_module.ChineseTokenizer

def test_emoji_tokenization():
    """测试 emoji 是否被识别为 [UNK]"""

    print("="*70)
    print("测试 ChineseTokenizer 对 emoji 的处理")
    print("="*70)

    # 创建 tokenizer 实例
    tokenizer = ChineseTokenizer(mode="char")

    # 测试用例
    test_cases = [
        "🌧️",           # 雨滴 emoji
        "你好🌧️世界",   # 中文 + emoji
        "Hello🌧️World", # 英文 + emoji
        "😀",           # 笑脸 emoji
        "🎉🎊🎈",       # 多个 emoji
        "普通文本",      # 纯中文
        "Hello",        # 纯英文
    ]

    print("\n测试结果:")
    print("-"*70)

    for text in test_cases:
        # 编码
        ids = tokenizer.encode(text)

        # 解码
        decoded = tokenizer.decode(ids)

        # 检查词汇表中的字符
        chars_in_vocab = [c for c in text if c in tokenizer.encoder]
        chars_not_in_vocab = [c for c in text if c not in tokenizer.encoder]

        print(f"\n原文本: {text!r}")
        print(f"  编码 ID: {ids}")
        print(f"  ID 长度: {len(ids)}")
        print(f"  解码结果: {decoded!r}")
        print(f"  在词汇表中: {chars_in_vocab}")
        print(f"  不在词汇表: {chars_not_in_vocab}")

        # 检查是否有损失
        if text != decoded:
            print(f"  ⚠️  编码/解码有损失！")
        else:
            print(f"  ✓  编码/解码无损")

    # 检查是否有 UNK token
    print("\n" + "="*70)
    print("检查词汇表中的特殊 token:")
    print("-"*70)

    special_tokens = ["<|pad|>", "<|endoftext|>", "<unk>", "[UNK]", "<UNK>"]
    for token in special_tokens:
        if token in tokenizer.encoder:
            print(f"  {token}: ID = {tokenizer.encoder[token]}")
        else:
            print(f"  {token}: 不存在")

    print("\n" + "="*70)
    print("结论:")
    print("-"*70)

    # 测试 emoji 编码
    emoji_ids = tokenizer.encode("🌧️")

    if len(emoji_ids) == 0:
        print("  emoji '🌧️' 被编码为空列表（被跳过）")
        print("  ❌ 没有使用 [UNK] token")
    else:
        print(f"  emoji '🌧️' 被编码为: {emoji_ids}")
        decoded_emoji = tokenizer.decode(emoji_ids)
        if decoded_emoji == "🌧️":
            print("  ✓ emoji 可以正确编码/解码")
        else:
            print(f"  解码结果: {decoded_emoji!r}")
            if "[UNK]" in decoded_emoji or "<unk>" in decoded_emoji.lower():
                print("  ✓ 使用了 [UNK] token")
            else:
                print("  ⚠️  未知字符处理方式不明确")

    print("="*70)

if __name__ == "__main__":
    test_emoji_tokenization()
