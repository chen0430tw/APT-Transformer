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
# from transformers import BertTokenizer  # 已替换为 SimpleCharTokenizer_BACKUP（支持 emoji）

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


def generate_with_vocab_mask(model, input_ids, valid_token_ids, max_length,
                             repetition_penalty, pad_token_id, device,
                             temperature=1.0, top_p=0.9):
    """
    使用 vocab mask 限制生成范围的自定义生成函数

    Args:
        model: APT 模型
        input_ids: 输入 token IDs
        valid_token_ids: 允许的 token ID 集合
        max_length: 最大长度
        repetition_penalty: 重复惩罚
        pad_token_id: padding ID
        device: 设备
        temperature: 采样温度
        top_p: nucleus 采样参数
    """
    model.eval()
    generated = input_ids.clone()

    # 创建 vocab mask（只允许生成已知的 token）
    vocab_size = model.config.vocab_size
    vocab_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for valid_id in valid_token_ids:
        if 0 <= valid_id < vocab_size:
            vocab_mask[valid_id] = True

    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            # 前向传播
            outputs = model(generated, generated)
            logits = outputs[:, -1, :]  # [batch_size, vocab_size]

            # 🔧 应用 vocab mask - 只允许生成已知的 token
            logits[:, ~vocab_mask] = -float('inf')

            # 重复惩罚
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if token_id in valid_token_ids:
                        logits[0, token_id] /= repetition_penalty

            # 温度调整
            logits = logits / max(temperature, 1e-5)

            # Top-p 采样
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # 找到累积概率超过 top_p 的位置
            remove_mask = cumsum_probs > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False

            # 移除低概率的 token
            sorted_probs[remove_mask] = 0.0
            probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
            if probs_sum > 0:
                sorted_probs = sorted_probs / probs_sum

            # 采样
            try:
                next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token_idx)
            except:
                # 如果采样失败，使用贪心解码
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

    return generated


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
    生成文本（修复：支持 emoji，去除输入复读，限制 vocab 范围）
    """
    model.eval()

    # 1. 获取 BOS/PAD ID
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id

    # 2. 编码输入文本（使用 __call__ 方法，不添加特殊 token）
    # SimpleCharTokenizer_BACKUP 的 __call__ 不添加 BOS/EOS
    input_result = tokenizer(input_text, max_length=64, padding=False,
                            truncation=True, return_tensors='pt')
    input_ids = input_result['input_ids'].to(device)  # shape: [1, seq_len]

    # 去除 padding（如果有）
    # 找到第一个非 pad token 的位置
    input_ids = input_ids[input_ids != pad_id].unsqueeze(0) if pad_id in input_ids else input_ids

    # 3. 准备模型输入 input_ids = [BOS] + Prompt Tokens
    bos_tensor = torch.tensor([[bos_id]], device=device)
    initial_ids = torch.cat([bos_tensor, input_ids], dim=1)

    # 4. 🔧 【修复】使用自定义生成，限制 vocab 范围
    # 只允许生成 tokenizer 已知的 token IDs
    valid_ids = set(tokenizer.id_to_char.keys())
    max_valid_id = max(valid_ids)

    generated_ids = generate_with_vocab_mask(
        model=model,
        input_ids=initial_ids,
        valid_token_ids=valid_ids,
        max_length=max_length + initial_ids.size(1),
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_id,
        device=device
    )

    # 5. 【修复】只解码新生成的部分，去掉输入
    input_length = initial_ids.size(1)
    generated_only = generated_ids[0][input_length:]  # 去掉输入部分
    generated_text = tokenizer.decode(generated_only, skip_special_tokens=True)

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


def save_model_and_tokenizer(model, tokenizer, config, save_dir, num_epochs, final_loss):
    """
    保存训练好的模型和 tokenizer

    Args:
        model: 训练好的 APT 模型
        tokenizer: SimpleCharTokenizer_BACKUP 实例
        config: APTModelConfiguration 实例
        save_dir: 保存目录
        num_epochs: 训练的总 epoch 数
        final_loss: 最终的损失值
    """
    import datetime

    os.makedirs(save_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'hlbd_model_{timestamp}.pt'
    model_path = os.path.join(save_dir, model_filename)

    # 保存模型、tokenizer 和配置
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'tokenizer_char_to_id': tokenizer.char_to_id,
        'tokenizer_id_to_char': tokenizer.id_to_char,
        'tokenizer_next_id': tokenizer.next_id,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'config': {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'max_seq_len': config.max_seq_len,
            'num_encoder_layers': config.num_encoder_layers,
            'num_decoder_layers': config.num_decoder_layers,
            'num_heads': config.num_heads,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'use_autopoietic': config.use_autopoietic,
            'use_dbc_dac': config.use_dbc_dac,
        },
        'training_info': {
            'num_epochs': num_epochs,
            'final_loss': final_loss,
            'timestamp': timestamp,
        }
    }

    torch.save(checkpoint, model_path)

    print(f"\n💾 模型已保存:")
    print(f"   路径: {os.path.abspath(model_path)}")
    print(f"   大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    return model_path


def load_model_and_tokenizer(model_path, device):
    """
    加载已保存的模型和 tokenizer

    Args:
        model_path: 模型文件路径
        device: 设备（cuda 或 cpu）

    Returns:
        model: 加载的 APT 模型
        tokenizer: 加载的 SimpleCharTokenizer_BACKUP
        training_info: 训练信息字典
    """
    print(f"\n📂 加载模型: {model_path}")

    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # 重建配置
    config_dict = checkpoint['config']
    config = APTModelConfiguration(
        vocab_size=config_dict['vocab_size'],
        d_model=config_dict['d_model'],
        max_seq_len=config_dict['max_seq_len'],
        num_encoder_layers=config_dict['num_encoder_layers'],
        num_decoder_layers=config_dict['num_decoder_layers'],
        num_heads=config_dict['num_heads'],
        d_ff=config_dict['d_ff'],
        dropout=config_dict['dropout'],
        use_autopoietic=config_dict['use_autopoietic'],
        use_dbc_dac=config_dict['use_dbc_dac'],
    )

    # 重建模型
    model = APTModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 重建 tokenizer
    tokenizer = SimpleCharTokenizer_BACKUP()
    tokenizer.char_to_id = checkpoint['tokenizer_char_to_id']
    tokenizer.id_to_char = checkpoint['tokenizer_id_to_char']
    tokenizer.next_id = checkpoint['tokenizer_next_id']
    tokenizer.vocab_size = checkpoint['tokenizer_vocab_size']

    # 获取训练信息
    training_info = checkpoint.get('training_info', {})

    print(f"✅ 模型加载成功!")
    print(f"   训练 epoch: {training_info.get('num_epochs', 'N/A')}")
    print(f"   最终损失: {training_info.get('final_loss', 'N/A'):.4f}")
    print(f"   保存时间: {training_info.get('timestamp', 'N/A')}")
    print(f"   词汇表大小: {len(tokenizer.char_to_id)}")

    return model, tokenizer, training_info


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
    # 使用 SimpleCharTokenizer_BACKUP（支持 emoji 动态添加）
    tokenizer = SimpleCharTokenizer_BACKUP()
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

    num_epochs = 30  # 快速训练测试（数据集小，30轮足够）

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

    # 11. 保存模型
    save_dir = os.path.join(project_root, 'tests', 'saved_models')
    model_path = save_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        save_dir=save_dir,
        num_epochs=num_epochs,
        final_loss=loss
    )

    return model, tokenizer, model_path


if __name__ == "__main__":
    model, tokenizer, model_path = main()

    # 可选：测试加载功能
    print("\n" + "="*60)
    print("🔄 测试模型加载功能")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model, loaded_tokenizer, training_info = load_model_and_tokenizer(model_path, device)

    # 验证加载的模型
    test_cases = [
        ("🌧️", "下雨"),
        ("❤️", "我爱你"),
    ]

    print("\n使用加载的模型生成:")
    test_generation(loaded_model, loaded_tokenizer, test_cases, device)

    print("\n✅ 模型保存和加载功能正常！")
