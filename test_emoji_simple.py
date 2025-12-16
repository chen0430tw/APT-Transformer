#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 emoji tokenizer 测试
直接模拟 ChineseTokenizer 的行为
"""

def test_emoji_tokenization():
    """测试 emoji 编码行为"""

    print("="*70)
    print("模拟 ChineseTokenizer 对 emoji 的处理")
    print("="*70)

    # 模拟 ChineseTokenizer 的词汇表构建
    encoder = {}
    decoder = {}

    # 1. 添加特殊 token
    special_tokens = ["<|pad|>", "<|endoftext|>"]
    for i, token in enumerate(special_tokens):
        encoder[token] = i
        decoder[i] = token

    # 2. 添加 ASCII 字符
    for i in range(32, 127):
        char = chr(i)
        if char not in encoder:
            encoder[char] = len(encoder)

    # 3. 添加常用汉字
    for i in range(0x4e00, 0x5000):  # 只添加一小部分汉字作为示例
        char = chr(i)
        if char not in encoder:
            encoder[char] = len(encoder)

    decoder = {v: k for k, v in encoder.items()}

    print(f"\n词汇表大小: {len(encoder)}")
    print(f"特殊 token: {special_tokens}")

    # 测试 emoji 编码（模拟 ChineseTokenizer.encode 的行为）
    def encode(text):
        """字符级编码，跳过未知字符"""
        ids = []
        for char in text:
            if char in encoder:
                ids.append(encoder[char])
            else:
                # 未知字符被跳过（line 161 in chinese_tokenizer.py）
                pass
        return ids

    def decode(ids):
        """解码"""
        chars = []
        for id in ids:
            if id in decoder:
                chars.append(decoder[id])
        return ''.join(chars)

    # 测试用例
    test_cases = [
        "🌧️",
        "你好🌧️世界",
        "Hello🌧️World",
        "😀",
        "🎉🎊🎈",
        "普通文本",
        "Hello",
    ]

    print("\n测试结果:")
    print("-"*70)

    for text in test_cases:
        ids = encode(text)
        decoded = decode(ids)

        chars_in_vocab = [c for c in text if c in encoder]
        chars_not_in_vocab = [c for c in text if c not in encoder]

        print(f"\n原文本: {text!r}")
        print(f"  编码 ID: {ids}")
        print(f"  ID 长度: {len(ids)} (原文本长度: {len(text)})")
        print(f"  解码结果: {decoded!r}")
        print(f"  在词汇表中: {[c for c in text if c in encoder]}")
        print(f"  不在词汇表: {[c for c in text if c not in encoder]}")

        if text != decoded:
            print(f"  ⚠️  编码/解码有损失！丢失了 {len(text) - len(decoded)} 个字符")
        else:
            print(f"  ✓  编码/解码无损")

    # 检查是否有 UNK token
    print("\n" + "="*70)
    print("检查词汇表中的 UNK token:")
    print("-"*70)

    unk_tokens = ["<unk>", "[UNK]", "<UNK>", "�"]
    found_unk = False
    for token in unk_tokens:
        if token in encoder:
            print(f"  找到: {token} (ID = {encoder[token]})")
            found_unk = True

    if not found_unk:
        print("  ❌ 词汇表中没有任何 UNK token")

    # 最终结论
    print("\n" + "="*70)
    print("结论:")
    print("-"*70)

    emoji_text = "🌧️"
    emoji_ids = encode(emoji_text)

    if len(emoji_ids) == 0:
        print(f"  • emoji '{emoji_text}' 被编码为空列表 []")
        print(f"  • 未知字符被直接跳过（chinese_tokenizer.py line 161）")
        print(f"  • ❌ 没有使用 [UNK] token")
        print(f"  • ❌ 编码/解码会丢失 emoji")
    else:
        print(f"  • emoji '{emoji_text}' 被编码为: {emoji_ids}")
        decoded_emoji = decode(emoji_ids)
        print(f"  • 解码结果: {decoded_emoji!r}")
        if decoded_emoji == emoji_text:
            print(f"  • ✓ emoji 可以正确编码/解码")
        else:
            print(f"  • ⚠️  编码/解码有损失")

    print("="*70)

if __name__ == "__main__":
    test_emoji_tokenization()
