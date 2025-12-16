#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 SimpleCharTokenizer_BACKUP 对 emoji 的处理
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接从 test_hlbd_quick_learning.py 导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "test_hlbd",
    os.path.join(os.path.dirname(__file__), "tests/test_hlbd_quick_learning.py")
)
test_hlbd_module = importlib.util.module_from_spec(spec)

# 只执行到 SimpleCharTokenizer_BACKUP 定义为止
# 我们需要手动提取类定义
exec("""
class SimpleCharTokenizer_BACKUP:
    '''简单的字符级分词器'''
    def __init__(self):
        # 创建一个基础字符表（包括中文、英文、emoji等）
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = 5000  # 预留足够的词汇空间

        # 添加常用字符
        self.char_to_id = self.vocab.copy()
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        self.next_id = 4

    def _get_or_add_char(self, char):
        '''获取字符ID，如果不存在则添加'''
        if char not in self.char_to_id:
            if self.next_id < self.vocab_size:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
            else:
                return self.unk_token_id
        return self.char_to_id[char]

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        '''编码文本为ID序列'''
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for char in text:
            ids.append(self._get_or_add_char(char))
        if add_special_tokens:
            ids.append(self.eos_token_id)

        if return_tensors == 'pt':
            import torch
            return torch.tensor([ids])
        return ids

    def __call__(self, text, max_length=64, padding='max_length',
                 truncation=True, return_tensors='pt'):
        '''分词接口（兼容transformers）'''
        ids = []
        for char in text:
            ids.append(self._get_or_add_char(char))

        # 截断
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]

        # 填充
        if padding == 'max_length':
            while len(ids) < max_length:
                ids.append(self.pad_token_id)

        if return_tensors == 'pt':
            import torch
            return {'input_ids': torch.tensor([ids])}
        return {'input_ids': ids}

    def decode(self, ids, skip_special_tokens=True):
        '''解码ID序列为文本'''
        chars = []
        for id in ids:
            if hasattr(id, 'item'):  # torch.Tensor
                id = id.item()

            if skip_special_tokens and id in [self.pad_token_id, self.bos_token_id,
                                               self.eos_token_id, self.unk_token_id]:
                continue

            char = self.id_to_char.get(id, '[UNK]')
            chars.append(char)

        return ''.join(chars)
""")

def test_emoji_encoding():
    """测试 emoji 编码"""

    print("="*70)
    print("测试 SimpleCharTokenizer_BACKUP 对 emoji 的处理")
    print("="*70)

    # 创建 tokenizer
    tokenizer = SimpleCharTokenizer_BACKUP()

    print(f"\n初始词汇表大小: {len(tokenizer.char_to_id)}")
    print(f"初始 next_id: {tokenizer.next_id}")

    # 测试用例
    test_cases = [
        "🌧️",           # 雨滴 emoji
        "❤️",           # 爱心 emoji
        "🍽️",           # 餐具 emoji
        "📖",           # 书本 emoji
        "你好🌧️世界",   # 中文 + emoji
        "Hello🌧️World", # 英文 + emoji
        "我爱你",        # 纯中文
        "Hello",        # 纯英文
    ]

    print("\n测试结果:")
    print("-"*70)

    for text in test_cases:
        # 编码（不添加特殊 token）
        ids = tokenizer.encode(text, add_special_tokens=False)

        # 解码
        decoded = tokenizer.decode(ids, skip_special_tokens=True)

        print(f"\n原文本: {text!r}")
        print(f"  编码 ID: {ids}")
        print(f"  ID 长度: {len(ids)} (原文本长度: {len(text)})")
        print(f"  解码结果: {decoded!r}")

        # 检查是否使用了 UNK
        if tokenizer.unk_token_id in ids:
            unk_count = ids.count(tokenizer.unk_token_id)
            print(f"  ⚠️  包含 {unk_count} 个 [UNK] token")

        # 检查是否有损失
        if text != decoded:
            print(f"  ❌ 编码/解码有损失！")
        else:
            print(f"  ✓  编码/解码无损")

    # 检查词汇表增长
    print("\n" + "="*70)
    print("词汇表统计:")
    print("-"*70)
    print(f"最终词汇表大小: {len(tokenizer.char_to_id)}")
    print(f"最终 next_id: {tokenizer.next_id}")
    print(f"新增字符数: {tokenizer.next_id - 4}")

    # 显示所有 emoji
    emojis_in_vocab = {char: id for char, id in tokenizer.char_to_id.items()
                       if ord(char) > 0x1F300 or char in ['🌧', '️', '❤', '🍽', '📖']}
    print(f"\nEmoji 在词汇表中:")
    for emoji, id in sorted(emojis_in_vocab.items(), key=lambda x: x[1])[:20]:
        print(f"  {emoji!r}: ID = {id}")

    # 测试不同 emoji 是否有不同 ID
    print("\n" + "="*70)
    print("Emoji 唯一性测试:")
    print("-"*70)

    emoji_list = ["🌧️", "❤️", "🍽️", "📖"]
    emoji_ids = {}
    for emoji in emoji_list:
        ids = tokenizer.encode(emoji, add_special_tokens=False)
        emoji_ids[emoji] = tuple(ids)
        print(f"  {emoji!r}: {ids}")

    # 检查是否都不同
    unique_ids = len(set(emoji_ids.values()))
    print(f"\n独特的 emoji 编码: {unique_ids} / {len(emoji_list)}")

    if unique_ids == len(emoji_list):
        print("  ✓ 所有 emoji 都有独立的编码！")
    else:
        print("  ❌ 有些 emoji 共享相同的编码")

    print("="*70)

if __name__ == "__main__":
    test_emoji_encoding()
