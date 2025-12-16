#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 BertTokenizer 对 emoji 的处理
（test_hlbd_quick_learning.py 使用的是 BertTokenizer）
"""

import os
import sys

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from transformers import BertTokenizer

    # 使用本地 bert-base-chinese
    bert_path = os.path.join(current_dir, 'bert', 'bert-base-chinese')

    print("="*70)
    print("测试 BertTokenizer (bert-base-chinese) 对 emoji 的处理")
    print("="*70)

    if not os.path.exists(bert_path):
        print(f"\n❌ 未找到本地 BERT 模型: {bert_path}")
        print("请先下载 bert-base-chinese 模型")
        sys.exit(1)

    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        bert_path,
        local_files_only=True,
        vocab_file=os.path.join(bert_path, 'vocab.txt')
    )

    print(f"\n分词器类型: {type(tokenizer).__name__}")
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 检查特殊 token
    print(f"\n特殊 token:")
    print(f"  [UNK]: {tokenizer.unk_token} (ID={tokenizer.unk_token_id})")
    print(f"  [PAD]: {tokenizer.pad_token} (ID={tokenizer.pad_token_id})")
    print(f"  [CLS]: {tokenizer.cls_token} (ID={tokenizer.cls_token_id})")
    print(f"  [SEP]: {tokenizer.sep_token} (ID={tokenizer.sep_token_id})")

    # 测试 emoji 编码
    test_cases = [
        "🌧️",           # 雨滴 emoji（test_hlbd 中使用）
        "❤️",           # 爱心 emoji（test_hlbd 中使用）
        "🍽️",           # 餐具 emoji（test_hlbd 中使用）
        "📖",           # 书本 emoji（test_hlbd 中使用）
        "你好🌧️世界",   # 中文 + emoji
        "Hello🌧️World", # 英文 + emoji
        "我爱你",        # 纯中文
        "Hello",        # 纯英文
    ]

    print("\n" + "="*70)
    print("测试结果:")
    print("-"*70)

    for text in test_cases:
        # 编码（不添加特殊 token）
        ids = tokenizer.encode(text, add_special_tokens=False)

        # 解码
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        # 获取 token 列表
        tokens = tokenizer.tokenize(text)

        print(f"\n原文本: {text!r}")
        print(f"  Token: {tokens}")
        print(f"  编码 ID: {ids}")
        print(f"  ID 长度: {len(ids)} (原文本长度: {len(text)})")
        print(f"  解码结果: {decoded!r}")

        # 检查是否有 UNK
        if tokenizer.unk_token_id in ids:
            unk_count = ids.count(tokenizer.unk_token_id)
            print(f"  ⚠️  包含 {unk_count} 个 [UNK] token")

        if text != decoded:
            print(f"  ⚠️  编码/解码有损失！")
        else:
            print(f"  ✓  编码/解码无损")

    # 专门测试 emoji
    print("\n" + "="*70)
    print("Emoji 详细分析:")
    print("-"*70)

    emoji_text = "🌧️"
    tokens = tokenizer.tokenize(emoji_text)
    ids = tokenizer.encode(emoji_text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)

    print(f"原文本: {emoji_text!r}")
    print(f"  Unicode: {[hex(ord(c)) for c in emoji_text]}")
    print(f"  Tokenize: {tokens}")
    print(f"  编码 ID: {ids}")
    print(f"  解码结果: {decoded!r}")

    if tokenizer.unk_token_id in ids:
        print(f"\n✓ emoji 被编码为 [UNK] token (ID={tokenizer.unk_token_id})")
        print(f"  这意味着 BertTokenizer 识别 emoji 为未知字符")
        print(f"  但不会像 ChineseTokenizer 那样直接跳过")
    else:
        print(f"\n✓ emoji 被正常编码（未使用 UNK）")

    print("="*70)
    print("\n结论:")
    print("-"*70)
    print("BertTokenizer 处理 emoji 的方式：")
    if tokenizer.unk_token_id in tokenizer.encode("🌧️", add_special_tokens=False):
        print("  • emoji 被编码为 [UNK] token")
        print("  • 不会丢失 emoji 信息（用 UNK 占位）")
        print("  • 编码/解码过程保留了 emoji 的位置")
        print("  • ✓ 比 ChineseTokenizer 的直接跳过更合理")
    else:
        print("  • emoji 可以正常编码")

    print("="*70)

except ImportError as e:
    print(f"\n❌ 缺少依赖库: {e}")
    print("请安装: pip install transformers")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
