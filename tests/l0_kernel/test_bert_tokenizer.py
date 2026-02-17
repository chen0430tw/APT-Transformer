#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简单测试bert-base-chinese tokenizer是否能加载"""

from transformers import BertTokenizer

print("🔧 测试加载bert-base-chinese tokenizer...")

try:
    tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-chinese')
    print(f"✅ Tokenizer加载成功!")
    print(f"   词汇表大小: {tokenizer.vocab_size}")

    # 测试编码
    test_text = "你好，世界！"
    encoded = tokenizer.encode(test_text)
    print(f"\n📝 测试编码:")
    print(f"   输入: {test_text}")
    print(f"   编码: {encoded}")
    print(f"   解码: {tokenizer.decode(encoded)}")

    print("\n✅ Tokenizer测试通过！")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
