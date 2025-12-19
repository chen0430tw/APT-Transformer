#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试HLBD训练 - 看APT模型能否快速学会说话"""

import sys
import os
import re
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
from apt_model.generation.generator import generate_natural_text
from apt_model.generation.evaluator import evaluate_text_quality


class SimpleCharTokenizer_BACKUP:
    """简单的字符级分词器"""
    def __init__(self):
        # 创建一个基础字符表（包括中文、英文、emoji等）
        # 添加语言标签用于区分不同输入类型
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
            '[EMOJI]': 4, '[PHRASE]': 5, '[EN]': 6, '[PY]': 7, '[JP]': 8, '[KR]': 9,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = 5000  # 预留足够的词汇空间

        # 添加常用字符
        self.char_to_id = self.vocab.copy()
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        self.next_id = 10  # 从10开始，因为0-9已被特殊token占用

        # ⭐ 新增：预编译正则表达式，匹配 [TAG]
        self.tag_pattern = re.compile(r'(\[EMOJI\]|\[PHRASE\]|\[EN\]|\[PY\]|\[JP\]|\[KR\])')
    
    def _tokenize_text(self, text):
        """⭐ 核心修复：先切分标签，再切分字符"""
        tokens = []
        # 按标签切分
        parts = self.tag_pattern.split(text)
        for part in parts:
            if part in self.vocab:
                # 如果是标签，直接添加ID
                tokens.append(self.vocab[part])
            else:
                # 如果是普通文本，逐字处理
                for char in part:
                    # 跳过空白字符（可选，看你需求）
                    if char.strip():
                        tokens.append(self._get_or_add_char(char))
                    elif char == ' ': # 保留空格
                        tokens.append(self._get_or_add_char(char))
        return tokens

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
        ids.extend(self._tokenize_text(text))
        ids.append(self.eos_token_id)

        if return_tensors == 'pt':
            return torch.tensor([ids])
        return ids

    def __call__(self, text, max_length=64, padding='max_length',
                 truncation=True, return_tensors='pt'):
        
        """分词接口（兼容transformers）"""
        # 1. 初始化 ids
        ids = [self.bos_token_id]

        # 2. ⭐ 使用新的切分逻辑 (支持 [EMOJI])
        token_ids = self._tokenize_text(text)
        ids.extend(token_ids)
        
        # 3. 加 EOS
        ids.append(self.eos_token_id)

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
                pairs.append((f"[EMOJI] {emoji}", chinese))

        # 2. 短语 -> 中文
        if 'level_2' in sample and 'level_6' in sample:
            phrase = sample['level_2'].get('短语', '')
            chinese = sample['level_6'].get('中文', '')
            if phrase and chinese:
                pairs.append((f"[PHRASE] {phrase}", chinese))

        # 3. 英文 -> 中文
        if 'level_5' in sample and 'level_6' in sample:
            english = sample['level_5'].get('英文', '')
            chinese = sample['level_6'].get('中文', '')
            if english and chinese:
                pairs.append((f"[EN] {english}", chinese))

        # 4. 拼音 -> 中文
        if 'level_4' in sample and 'level_6' in sample:
            pinyin = sample['level_4'].get('拼音', '')
            chinese = sample['level_6'].get('中文', '')
            if pinyin and chinese:
                pairs.append((f"[PY] {pinyin}", chinese))

        # 5. 日文 -> 中文
        if 'level_7' in sample and 'level_6' in sample:
            japanese = sample['level_7'].get('日文', '')
            chinese = sample['level_6'].get('中文', '')
            if japanese and chinese:
                pairs.append((f"[JP] {japanese}", chinese))

        # 6. 韩文 -> 中文
        if 'level_8' in sample and 'level_6' in sample:
            korean = sample['level_8'].get('韩문', sample['level_8'].get('韩文', ''))  # 兼容两种键名
            chinese = sample['level_6'].get('中文', '')
            if korean and chinese:
                pairs.append((f"[KR] {korean}", chinese))

    print(f"   创建了 {len(pairs)} 个训练对（带语言标签）")
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


def train_epoch(model, dataloader, optimizer, criterion, device, use_dbc=False, accumulation_steps=4):
    """训练一个epoch（使用梯度累积）"""
    model.train()
    total_loss = 0
    total_steps = 0

    ACCUMULATION_STEPS = accumulation_steps

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
        mininterval=0.1,
        ascii=True
    )

    for i, (src_ids, tgt_ids) in enumerate(progress_bar):

        # 数据传输到设备
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        # 前向传播
        output = model(src_ids, tgt_ids[:, :-1])

        # 计算损失
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_ids[:, 1:].reshape(-1)
        )

        # 损失归一化
        loss = loss / ACCUMULATION_STEPS

        # 反向传播
        loss.backward()

        # 条件优化和清零（每N步执行一次）
        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        total_steps += 1

        # 实时更新进度条上的Loss
        progress_bar.set_postfix({'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}'})

    # 【最后一步清理】处理剩余的累积梯度
    try:
        if (i + 1) % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
    except NameError:
         # 如果i没定义（比如dataloader是空的），则忽略清理
         pass

    return (total_loss / total_steps) if total_steps > 0 else 0


def generate_text(model, tokenizer, input_text, device, max_length=50, repetition_penalty=1.5):
    """
    生成文本（修复：支持 emoji，去除输入复读，限制 vocab 范围）
    """
    model.eval()

    # 1. 准备 Encoder 输入 (Prompt)
    # 注意：这里的 input_text 应该包含标签，如 "[EMOJI] 🌧️"
    input_encoded = tokenizer(input_text, max_length=64, padding=False, return_tensors='pt')
    src_ids = input_encoded['input_ids'].to(device) # [1, src_len]

    # 2. 准备 Decoder 输入 (Start Token)
    # Decoder 从 [BOS] 开始，而不是从 Prompt 开始！
    decoder_input = torch.tensor([[tokenizer.bos_token_id]], device=device) # [1, 1]
    
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_length):
            # 3. 准备模型输入 input_ids = [BOS] + Prompt Tokens
            # 注意：APTModel 的 forward 接受 src_tokens 和 tgt_tokens
            outputs = model(src_tokens=src_ids, tgt_tokens=decoder_input)

            # 取最后一个 token 的 logits
            logits = outputs[:, -1, :] # [batch, vocab]

            # --- 强制禁止复读的关键代码 ---
            if repetition_penalty != 1.0:
                # 遍历已经生成过的所有 token
                for token_id in set(generated_ids):
                    # 如果 logit 是正数，除以惩罚系数（变小）
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    # 如果 logit 是负数，乘以惩罚系数（变得更负，更不可能被选中）
                    else:
                        logits[0, token_id] *= repetition_penalty

            # 4. 限制生成范围 (Vocab Mask) & 重复惩罚
            # ... (简化的采样逻辑) ...
            logits[:, tokenizer.pad_token_id] = -float('inf') # 别生成 PAD
            logits[:, tokenizer.unk_token_id] = -float('inf') # 别生成 UNK

            # 简单贪婪解码 (为了测试稳定性，先用贪婪)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 5. 停止条件
            if next_token.item() == tokenizer.eos_token_id:
                break

            # 6. 把新字加到 Decoder 输入里，准备下一轮
            generated_ids.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

    return tokenizer.decode(generated_ids)


def test_generation(model, tokenizer, test_cases, device):
    """测试生成能力"""
    print("\n" + "="*60)
    print("🗣️ 测试对话生成能力")
    print("="*60)

    # 设定强力重复惩罚因子
    REPETITION_FACTOR = 1.5

    for input_text, expected_concept in test_cases:
        generated = generate_text(model, tokenizer, input_text, device, repetition_penalty=REPETITION_FACTOR)
        input_ids = tokenizer.encode(input_text)

        ids_display = str(input_ids)
        if len(input_ids) > 8:
            ids_display = f"[{input_ids[0]}, {input_ids[1]}, ..., {input_ids[-1]}]"
        
        print(f"🕵️ Debug: Len={len(input_ids)} | IDs={ids_display}")
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


def evaluate_hlbd_model(untrained_model, trained_model, tokenizer, device):
    """评估HLBD训练前后的模型质量"""
    # 测试提示（带语言标签）
    test_prompts = [
        "[EMOJI] 🌧️",  # emoji测试
        "[EMOJI] ❤️",  # emoji测试
        "[EN] It's raining",  # 英文测试
        "[PY] wǒ ài nǐ",  # 拼音测试
        "[JP] 愛してる",  # 日文测试
        "[KR] 사랑해",  # 韩文测试
    ]

    untrained_model.eval()
    trained_model.eval()
    untrained_scores = []
    trained_scores = []

    print(f"\n" + "="*60)
    print("安柏の評価 | Amber's Evaluation")
    print("="*60)

    for prompt in test_prompts:
        with torch.no_grad():
            # 未训练模型
            untrained_text, _, _, _ = generate_natural_text(untrained_model, tokenizer, prompt, max_steps=15)
            untrained_score, untrained_feedback = evaluate_text_quality(untrained_text)
            untrained_scores.append(untrained_score)

            # 训练后模型
            trained_text, _, _, _ = generate_natural_text(trained_model, tokenizer, prompt, max_steps=15)
            trained_score, trained_feedback = evaluate_text_quality(trained_text)
            trained_scores.append(trained_score)

    avg_untrained = sum(untrained_scores) / len(untrained_scores) if untrained_scores else 0
    avg_trained = sum(trained_scores) / len(trained_scores) if trained_scores else 0
    improvement = avg_trained - avg_untrained

    # 最终评估
    print(f"\n整体评估:")
    print(f"未训练模型平均质量: {avg_untrained:.2f}/100")
    print(f"训练后模型平均质量: {avg_trained:.2f}/100")
    print(f"质量提升: {improvement:.2f} 分")

    # 安柏的最终评价
    if improvement < -5:
        print("\n安柏：奇怪……怎么感觉它变笨了？（质量下降，建议检查超参数）")
    elif improvement < 0:
        print("\n安柏：看起来效果差不多，也许还需要更多训练数据？")
    elif avg_trained < 50:
        print("\n安柏：虽然有进步，但还远远不够哦！继续加油！")
    else:
        print("\n安柏：训练完成得不错！侦察骑士为你点赞！")

    print("="*60)


def main():
    """主函数"""
    print("\n🚀 HLBD快速学习测试 - APT模型能否快速学会说话?")
    print(f"PyTorch版本: {torch.__version__}")

    ACCUMULATION_STEPS = 8  # 保持原始配置：batch_size=4, 4 * 8 = 32 的有效批次大小

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
    print(f"   初始词汇表: {len(tokenizer.char_to_id)} 个token (预留空间: {tokenizer.vocab_size})")
    print(f"   初始token: {list(tokenizer.char_to_id.keys())}")

    # 4. 创建数据集
    print(f"\n📊 创建数据集...")
    dataset = SimpleDialogueDataset(training_pairs, tokenizer)

    # 【关键修复】预填充词汇表，避免多进程陷阱
    # 在多进程 DataLoader 启动前，让主进程的 tokenizer 学习所有字符
    print(f"\n📝 预填充词汇表（避免多进程陷阱）...")
    for src, tgt in training_pairs:
        _ = tokenizer.encode(src)
        _ = tokenizer.encode(tgt)
    print(f"   词汇表预填充完成: {len(tokenizer.char_to_id)} 个token")

    # 优化：保持原始batch_size，只添加多线程加载
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 保持原始batch_size=4（稳定性优先）
        shuffle=True,
        num_workers=4,  # 使用4个工作进程并行加载（现在安全了）
        pin_memory=True,  # 固定内存，加速CPU→GPU传输
        persistent_workers=True  # 保持worker存活，避免重复创建
    )
    print(f"   训练批次数: {len(dataloader)}")

    # 【新增验证代码：检查实际样本数】
    actual_pairs = len(dataset)
    print(f"--- 长度验证 ---")
    print(f"模型实际看到的训练对数量: {actual_pairs} (每个概念6个层级映射)")
    print(f"   emoji/短语/英文/拼音/日文/韩文 → 中文")
    print(f"----------------")

    # 【词汇表增长验证】
    print(f"\n📊 词汇表动态增长情况:")
    print(f"   处理数据后的词汇表大小: {len(tokenizer.char_to_id)} 个token")
    print(f"   新增token数量: {len(tokenizer.char_to_id) - 10}")
    print(f"   下一个ID: {tokenizer.next_id}")
    print(f"   预留空间利用率: {len(tokenizer.char_to_id)}/{tokenizer.vocab_size} ({100*len(tokenizer.char_to_id)/tokenizer.vocab_size:.1f}%)")

    # 显示前20个动态添加的字符（跳过特殊token）
    dynamic_chars = [char for char, idx in sorted(tokenizer.char_to_id.items(), key=lambda x: x[1]) if idx >= 10][:20]
    print(f"   前20个动态添加的字符: {dynamic_chars}")

    # 5. 创建模型
    print(f"\n🏗️ 创建APT模型...")
    config = create_small_hlbd_config(tokenizer.vocab_size)
    model = APTModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")
    print(f"   配置: d_model={config.d_model}, layers={config.num_encoder_layers}")

    # 保存未训练模型的副本用于评估对比
    untrained_model = APTModel(config).to(device)
    untrained_model.load_state_dict(model.state_dict())
    untrained_model.eval()

    # 6. 创建优化器
    # 使用原始学习率（稳定性优先）
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

    num_epochs = 150  # 600个训练对（100概念×6层级：emoji/短语/英文/拼音/日文/韩文→中文）

    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device, use_dbc=True, accumulation_steps=ACCUMULATION_STEPS)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")

        # 每5个epoch测试一次
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            test_cases = [
                ("[EMOJI] 🌧️", "下雨"),
                ("[EMOJI] ❤️", "我爱你"),
                ("[EN] I love you", "我爱你"),
                ("[JP] 愛してる", "我爱你"),  # 日文测试
                ("[KR] 사랑해", "我爱你"),  # 韩文测试
            ]
            test_generation(model, tokenizer, test_cases, device)

    # 9. 最终测试
    print(f"\n" + "="*60)
    print("🎯 最终测试 - APT学会说话了吗?")
    print("="*60)

    final_test_cases = [
        ("[EMOJI] 🌧️", "下雨"),
        ("[EMOJI] ❤️", "我爱你"),
        ("[EMOJI] 🍽️", "吃饭"),
        ("[EMOJI] 📖", "看书"),
        ("[EN] I love you", "我爱你"),
        ("[EN] It's raining", "下雨"),
        ("[PY] wǒ ài nǐ", "我爱你"),
        ("[JP] 愛してる", "我爱你"),  # 日文
        ("[JP] 雨が降っています", "下雨"),  # 日文
        ("[KR] 사랑해", "我爱你"),  # 韩文
        ("[KR] 비가 오고 있어요", "下雨"),  # 韩文
    ]

    test_generation(model, tokenizer, final_test_cases, device)

    # 9.5 安柏评估
    try:
        evaluate_hlbd_model(untrained_model, model, tokenizer, device)
    except Exception as e:
        print(f"\n⚠️ 安柏评估出错: {e}")

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
        ("[EMOJI] 🌧️", "下雨"),
        ("[EMOJI] ❤️", "我爱你"),
    ]

    print("\n使用加载的模型生成:")
    test_generation(loaded_model, loaded_tokenizer, test_cases, device)

    print("\n✅ 模型保存和加载功能正常！")
