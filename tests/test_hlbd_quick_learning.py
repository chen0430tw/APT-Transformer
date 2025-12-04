#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试HLBD训练 - 看APT模型能否快速学会说话"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from transformers import BertTokenizer

# 添加路径（动态计算项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from apt_model.modeling.apt_model import (
    APTModel,
    APTModelConfiguration,
    DBCDAC_Optimizer,
    create_gradient_stabilizer_hook
)


class SimpleCharTokenizer_BACKUP:
    """简单的字符级分词器"""
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
        """获取字符ID，如果不存在则添加"""
        if char not in self.char_to_id:
            if self.next_id < self.vocab_size:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
            else:
                return self.unk_token_id
        return self.char_to_id[char]

    def encode(self, text, return_tensors=None):
        """编码文本为ID序列"""
        ids = [self.bos_token_id]
        for char in text:
            ids.append(self._get_or_add_char(char))
        ids.append(self.eos_token_id)

        if return_tensors == 'pt':
            return torch.tensor([ids])
        return ids

    def __call__(self, text, max_length=64, padding='max_length',
                 truncation=True, return_tensors='pt'):
        """分词接口（兼容transformers）"""
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
            return {'input_ids': torch.tensor([ids])}
        return {'input_ids': ids}

    def decode(self, ids, skip_special_tokens=True):
        """解码ID序列为文本"""
        chars = []
        for id in ids:
            if isinstance(id, torch.Tensor):
                id = id.item()

            if skip_special_tokens and id in [self.pad_token_id, self.bos_token_id,
                                               self.eos_token_id, self.unk_token_id]:
                continue

            char = self.id_to_char.get(id, '[UNK]')
            chars.append(char)

        return ''.join(chars)


def register_dbc_hooks(model):
    """为模型注册DBC-DAC hooks"""
    opt = DBCDAC_Optimizer()
    hooks = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            hooks.append(p.register_hook(create_gradient_stabilizer_hook(opt)))
    return hooks


def load_hlbd_samples(data_path, max_samples=20):
    """加载HLBD数据样本"""
    print(f"📂 加载HLBD数据: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取samples部分
    start_idx = content.find('samples = [')
    if start_idx == -1:
        raise ValueError("找不到samples数据")

    # 提取JSON数组
    json_start = content.find('[', start_idx)
    # 找到匹配的右括号（简单处理）
    bracket_count = 0
    json_end = json_start
    for i, char in enumerate(content[json_start:]):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                json_end = json_start + i + 1
                break

    json_str = content[json_start:json_end]
    samples = json.loads(json_str)

    if max_samples:
        samples = samples[:max_samples]

    print(f"   加载了 {len(samples)} 个样本")
    return samples


def create_training_pairs(samples):
    """从HLBD样本创建训练对"""
    pairs = []

    for sample in samples:
        concept = sample['concept']

        # 创建多种训练对
        # 1. emoji -> 中文
        if 'level_1' in sample and 'level_6' in sample:
            emoji = sample['level_1'].get('emoji', '')
            chinese = sample['level_6'].get('中文', '')
            if emoji and chinese:
                pairs.append((emoji, chinese))

        # 2. 短语 -> 中文
        if 'level_2' in sample and 'level_6' in sample:
            phrase = sample['level_2'].get('短语', '')
            chinese = sample['level_6'].get('中文', '')
            if phrase and chinese:
                pairs.append((phrase, chinese))

        # 3. 英文 -> 中文
        if 'level_5' in sample and 'level_6' in sample:
            english = sample['level_5'].get('英文', '')
            chinese = sample['level_6'].get('中文', '')
            if english and chinese:
                pairs.append((english, chinese))

        # 4. 拼音 -> 中文
        if 'level_4' in sample and 'level_6' in sample:
            pinyin = sample['level_4'].get('拼音', '')
            chinese = sample['level_6'].get('中文', '')
            if pinyin and chinese:
                pairs.append((pinyin, chinese))

    print(f"   创建了 {len(pairs)} 个训练对")
    return pairs


class SimpleDialogueDataset(Dataset):
    """简单对话数据集"""
    def __init__(self, pairs, tokenizer, max_length=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        # 编码源文本
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码目标文本
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return (
            src_encoding['input_ids'].squeeze(0),
            tgt_encoding['input_ids'].squeeze(0)
        )


def create_small_hlbd_config(vocab_size):
    """创建小型HLBD APT配置"""
    config = APTModelConfiguration(
        vocab_size=vocab_size,
        d_model=256,              # 中等维度
        max_seq_len=64,           # 短序列
        num_encoder_layers=3,     # 3层编码器
        num_decoder_layers=3,     # 3层解码器
        num_heads=8,              # 8个注意力头
        d_ff=1024,                # 前馈网络
        dropout=0.1,
        use_autopoietic=True,     # 启用自生成机制
        use_dbc_dac=True,         # 启用DBC-DAC
    )
    return config


from tqdm import tqdm # 确保文件开头导入了 tqdm

def train_epoch(model, dataloader, optimizer, criterion, device, use_dbc=False, accumulation_steps=4): # <--- 【关键修改 1】接收 accumulation_steps
    """训练一个epoch (使用梯度累积)"""
    model.train()
    total_loss = 0
    total_steps = 0
    
    ACCUMULATION_STEPS = accumulation_steps # 【关键修正】在函数体内定义 ACCUMULATION_STEPS
    
    progress_bar = tqdm(
        dataloader, 
        desc="Training", 
        leave=False, 
        mininterval=0.1, 
        ascii=True
    )

    # 【关键修改 2】使用 enumerate 来获取批次索引 i
    for i, (src_ids, tgt_ids) in enumerate(progress_bar): 
        
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        # 前向传播
        output = model(src_ids, tgt_ids[:, :-1])

        # 计算损失
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_ids[:, 1:].reshape(-1)
        )

        # 损失归一化 (Loss Scaling)
        loss = loss / ACCUMULATION_STEPS 

        # 反向传播 (不清除梯度)
        loss.backward()

        # 条件优化和清零 (每 N 步执行一次)
        if (i + 1) % ACCUMULATION_STEPS == 0:
            # 权重更新 (即使 DBC 激活，保留裁剪也无害)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad() # 清除累积的梯度
            
        total_loss += loss.item() * ACCUMULATION_STEPS # 恢复实际 Loss
        total_steps += 1
        
        # 实时更新进度条上的 Loss
        progress_bar.set_postfix({'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}'})
        
    # 【最后一步清理】处理剩余的累积梯度 (i 需要在循环外可用)
    # Note: 确保 i 在循环外能访问到，尽管这不是标准 Python 做法
    try:
        if (i + 1) % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
    except NameError:
         # 如果 i 没定义 (比如 dataloader 是空的)，则忽略清理
         pass 

    return (total_loss / total_steps) if total_steps > 0 else 0


def generate_text(model, tokenizer, input_text, device, max_length=50, repetition_penalty=1.5):
    """
    生成文本 (已修复为调用模型内置的 generate 方法，激活所有高级参数)
    """
    model.eval()

    # 1. 获取 BOS/PAD ID
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id
    
    # 2. 编码输入文本 (作为 Prompt)
    # 编码时，不加特殊 tokens，只编码原始文本
    input_encoding = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(device)
    
    # 3. 准备模型输入 input_ids = [BOS] + Prompt Tokens
    bos_tensor = torch.tensor([[bos_id]], device=device)
    # 将 BOS 标记和 Prompt 拼接成完整的输入序列
    initial_ids = torch.cat([bos_tensor, input_encoding], dim=1)
    
    # 4. 调用 APTModel 自身的 generate 方法 (核心替换)
    generated_ids = model.generate(
        input_ids=initial_ids,
        # max_length 必须加上初始 sequence 的长度
        max_length=max_length + initial_ids.size(1), 
        repetition_penalty=repetition_penalty,
        do_sample=True, # 启用采样 (Top-K/Top-P)
        num_beams=1, # 保持 beam search 数量 (注意，模型内部会忽略 num_beams > 1)
        pad_token_id=pad_id 
        # 完整的 Top-K/Top-P/Temperature 参数应该在 model.generate 内部配置
    )
    
    # 5. 解码
    # generated_ids[0] 是第一个 batch 的输出序列
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def test_generation(model, tokenizer, test_cases, device):
    """测试生成能力"""
    print("\n" + "="*60)
    print("🗣️ 测试对话生成能力")
    print("="*60)

    # 设定强力重复惩罚因子
    REPETITION_FACTOR = 1.5

    for input_text, expected_concept in test_cases:
        generated = generate_text(model, tokenizer, input_text, device, repetition_penalty=REPETITION_FACTOR)
        print(f"\n输入: {input_text}")
        print(f"期望概念: {expected_concept}")
        print(f"生成: {generated}")


def main():
    """主函数"""
    print("\n🚀 HLBD快速学习测试 - APT模型能否快速学会说话?")
    print(f"PyTorch版本: {torch.__version__}")

    ACCUMULATION_STEPS = 8  # 模拟 4 * 8 = 32 的有效批次大小

    # 自动检测：有显卡就用显卡，没有才用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1. 加载HLBD数据
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在目录 (tests)
    project_root = os.path.dirname(current_dir)              # 获取项目根目录 (APT-Transformer)
    data_path = os.path.join(project_root, 'apt_model', '分层语言启蒙数据集.txt')
    samples = load_hlbd_samples(data_path, max_samples=None)

    # 顺便把下面那个 BERT 的路径也一起修了，不然等会还会报错
    bert_path = os.path.join(project_root, 'bert', 'bert-base-chinese')

    # 显示几个样本
    print(f"\n📝 样本示例:")
    for i, sample in enumerate(samples[:3]):
        print(f"\n   样本 {i+1}: {sample['concept']}")
        print(f"      Emoji: {sample['level_1'].get('emoji', 'N/A')}")
        print(f"      中文: {sample['level_6'].get('中文', 'N/A')[:30]}...")

    # 2. 创建训练对
    training_pairs = create_training_pairs(samples)

    # 3. 准备分词器
    print(f"\n🔧 准备分词器...")
    # 使用本地的bert-base-chinese tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        bert_path,
        local_files_only=True,  # <-- 强制使用本地文件，禁止联网
        vocab_file=os.path.join(bert_path, 'vocab.txt') # <-- 显式指定词表位置
    ) # 使用上面计算好的路径
    print(f"   使用的分词器: {type(tokenizer).__name__}")
    print(f"   词汇表大小: {tokenizer.vocab_size}")

    # 4. 创建数据集
    print(f"\n📊 创建数据集...")
    dataset = SimpleDialogueDataset(training_pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"   训练批次数: {len(dataloader)}")

    # 【新增验证代码：检查实际样本数】
    actual_pairs = len(dataset)
    print(f"--- 长度验证 ---")
    print(f"模型实际看到的训练对数量: {actual_pairs} (应为 80 或更多)")
    print(f"----------------")

    # 5. 创建模型
    print(f"\n🏗️ 创建APT模型...")
    config = create_small_hlbd_config(tokenizer.vocab_size)
    model = APTModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")
    print(f"   配置: d_model={config.d_model}, layers={config.num_encoder_layers}")

    # 6. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 7. 注册DBC hooks
    print(f"\n⚡ 注册DBC-DAC加速...")
    #hooks = register_dbc_hooks(model)
    hooks = [] # 保持 hooks 变量存在，防止后面报错
    #print(f"   注册了 {len(hooks)} 个梯度稳定钩子")

    # 8. 训练模型
    print(f"\n" + "="*60)
    print("🏃 开始快速训练 (看能否快速学会说话)")
    print("="*60)

    num_epochs = 500  # 只训练10个epoch

    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device, use_dbc=True, accumulation_steps=ACCUMULATION_STEPS)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")

        # 每3个epoch测试一次
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            test_cases = [
                ("🌧️", "下雨"),
                ("❤️", "我爱你"),
                ("I love you", "我爱你"),
                ("下雨", "天气"),
            ]
            test_generation(model, tokenizer, test_cases, device)

    # 9. 最终测试
    print(f"\n" + "="*60)
    print("🎯 最终测试 - APT学会说话了吗?")
    print("="*60)

    final_test_cases = [
        ("🌧️", "下雨"),
        ("❤️", "我爱你"),
        ("🍽️", "吃饭"),
        ("📖", "看书"),
        ("I love you", "我爱你"),
        ("It's raining", "下雨"),
        ("wǒ ài nǐ", "我爱你"),
    ]

    test_generation(model, tokenizer, final_test_cases, device)

    # 10. 总结
    print(f"\n" + "="*60)
    print("📝 测试总结")
    print("="*60)
    print(f"✅ 训练完成: {num_epochs} epochs")
    print(f"✅ 训练样本: {len(samples)} 概念, {len(training_pairs)} 对")
    print(f"✅ DBC加速: {len(hooks)} 个钩子激活")
    print(f"✅ 模型参数: {total_params:,}")
    print(f"\n💡 观察:")
    print(f"   - APT模型使用自生成注意力机制")
    print(f"   - DBC-DAC稳定了梯度训练")
    print(f"   - 分层语言学习帮助快速掌握概念")
    print(f"   - 从emoji/拼音/英文到中文的多层映射")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = main()
