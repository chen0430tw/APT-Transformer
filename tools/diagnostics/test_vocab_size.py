#!/usr/bin/env python3
"""测试分词器实际收录的词汇数量"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tests.test_hlbd_quick_learning import SimpleCharTokenizer_BACKUP, load_hlbd_samples, create_training_pairs

# 加载数据
print("📂 加载数据...")
project_root = current_dir
data_path = os.path.join(project_root, 'apt_model', '分层语言启蒙数据集.txt')
samples = load_hlbd_samples(data_path, max_samples=None)
print(f"加载了 {len(samples)} 个概念样本")

# 创建训练对
training_pairs = create_training_pairs(samples)
print(f"生成了 {len(training_pairs)} 个训练对")

# 创建分词器
print("\n🔧 创建分词器...")
tokenizer = SimpleCharTokenizer_BACKUP()
print(f"初始 vocab_size: {tokenizer.vocab_size}")
print(f"初始 next_id: {tokenizer.next_id}")
print(f"初始实际词汇数: {len(tokenizer.char_to_id)}")

# 处理所有训练对
print("\n处理所有训练对...")
for i, (src, tgt) in enumerate(training_pairs):
    tokenizer(src, max_length=64, padding=False, truncation=True)
    tokenizer(tgt, max_length=64, padding=False, truncation=True)

    if (i+1) % 100 == 0:
        print(f"  已处理 {i+1}/{len(training_pairs)} 对，当前词汇数: {tokenizer.next_id}")

# 最终统计
print(f"\n✅ 处理完成!")
print(f"vocab_size (预留): {tokenizer.vocab_size}")
print(f"next_id (实际使用): {tokenizer.next_id}")
print(f"char_to_id 大小: {len(tokenizer.char_to_id)}")

# 列出所有收录的字符
chars = [k for k in tokenizer.char_to_id.keys() if k not in ['[PAD]', '[UNK]', '[BOS]', '[EOS]']]
print(f"\n实际收录的字符数: {len(chars)}")
print(f"前50个字符: {chars[:50]}")

# 验证是否超出vocab_size
if tokenizer.next_id >= tokenizer.vocab_size:
    print(f"\n⚠️  警告：next_id ({tokenizer.next_id}) >= vocab_size ({tokenizer.vocab_size})")
    print("   部分字符可能被截断为[UNK]！")
