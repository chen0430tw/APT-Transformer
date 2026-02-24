#!/usr/bin/env python3
"""
Virtual VRAM v1.6 测试 - 只下载少量C4分片
"""
import sys
sys.path.insert(0, "/mnt/d/APT-Transformer")

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

print("=" * 60)
print("Virtual VRAM v1.6 测试 - 少量C4分片")
print("=" * 60)

# 只下载C4的2个分片，streaming模式
print("\n正在加载C4数据集（仅前2个分片）...")
dataset = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True,
    keep_in_memory=False
)

# 只取前1000条
data = []
for i, item in enumerate(dataset):
    if i >= 1000:
        break
    data.append(item["text"])
    if i % 100 == 0:
        print(f"已加载 {i} 条...")

print(f"\n✅ 加载完成：{len(data)} 条数据")

# 简单的tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.vocab_size = 1000

    def encode(self, text):
        # 简单的字符级编码
        return [ord(c) % self.vocab_size for c in text[:512]]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

tokenizer = SimpleTokenizer()

# 简单模型
class SimpleLM(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleLM().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 准备数据
print("\n准备训练数据...")
encoded_texts = []
for text in data:
    encoded = tokenizer.encode(text)
    if len(encoded) > 50:  # 至少50个token
        encoded_texts.append(encoded)
    if len(encoded_texts) >= 100:
        break

print(f"准备了 {len(encoded_texts)} 个训练样本")

# 配置Virtual VRAM
cfg = VirtualVRAMConfig(
    enabled=True,
    enable_nested_v16=True,
    min_tensor_bytes=5 << 20,
    verbose=True
)

print("\n" + "=" * 60)
print("开始训练（使用Virtual VRAM）")
print("=" * 60)

with virtual_vram(cfg):
    for step in range(10):
        optimizer.zero_grad()

        # 随机选择一个样本
        encoded = encoded_texts[step % len(encoded_texts)]
        input_ids = torch.tensor(encoded[:64]).unsqueeze(0).cuda()  # [1, 64]
        target = torch.tensor(encoded[1:65]).unsqueeze(0).cuda()

        # 前向传播
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

        # 反向传播
        loss.backward()
        optimizer.step()

        print(f"Step {step}: Loss={loss.item():.4f}")

        # 检查NaN
        if step % 5 == 0:
            has_nan = False
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"  ❌ {name} 包含NaN!")
                    has_nan = True

            if not has_nan:
                print(f"  ✅ 参数正常")

print("\n" + "=" * 60)
print("✅ 测试完成!")
print("=" * 60)
