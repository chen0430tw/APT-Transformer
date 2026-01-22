#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试HLBD训练 - 看APT模型能否快速学会说话"""

import sys
import os
import re
import time
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

from apt.apt_model.modeling.apt_model import (
    APTModel,
    APTModelConfiguration,
    DBCDAC_Optimizer,
    create_gradient_stabilizer_hook
)
from apt.core.generation.generator import generate_natural_text
from apt.core.generation.evaluator import evaluate_text_quality


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
        dropout=0.1,             # 适度dropout
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
    """训练一个epoch（带性能诊断版）"""
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

    # ⏱️ [诊断初始化]
    t_start = time.time()
    times = {"data": 0, "forward": 0, "backward": 0, "step": 0}

    # 注意：如果你的 dataloader 返回的是字典（如 HLBD 脚本），这里可能需要改成 for i, batch in enumerate...
    # 下面保留你提供的解包格式，请根据实际情况调整
    for i, batch_data in enumerate(progress_bar):
        
        # ⏱️ [1. 记录数据加载耗时] (从上轮循环结束到这里的时间)
        t_data_end = time.time()
        times["data"] = t_data_end - t_start

        # 兼容处理：检查是元组还是字典
        if isinstance(batch_data, dict):
            src_ids = batch_data['input_ids'].to(device)
            # 假设 HLBD 任务中 target 就是 input
            tgt_ids = src_ids.clone() 
        else:
            src_ids, tgt_ids = batch_data
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

        # ⏱️ [2. 记录前向传播耗时]
        t_fw_start = time.time()
        
        # 前向传播
        if hasattr(model.config, 'pad_token_id'): # 简单的 input/label 处理
             # 自回归任务通常输入是 src，目标也是 src（错位）
             output = model(src_ids, tgt_ids[:, :-1])
        else:
             output = model(src_ids, tgt_ids[:, :-1])

        # 计算损失
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_ids[:, 1:].reshape(-1)
        )
        
        # 损失归一化
        loss = loss / ACCUMULATION_STEPS
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_fw_end = time.time()
        times["forward"] = t_fw_end - t_fw_start

        # ⏱️ [3. 记录反向传播耗时] (这里是DBC钩子起作用的地方!)
        t_bw_start = time.time()
        
        loss.backward()
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t_bw_end = time.time()
        times["backward"] = t_bw_end - t_bw_start

        # 条件优化和清零（每N步执行一次）
        step_time = 0
        if (i + 1) % ACCUMULATION_STEPS == 0:
            # ⏱️ [4. 记录参数更新耗时]
            t_step_start = time.time()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_step_end = time.time()
            step_time = t_step_end - t_step_start
            times["step"] = step_time

        total_loss += loss.item() * ACCUMULATION_STEPS
        total_steps += 1

        # 实时更新进度条上的Loss和关键耗时
        progress_bar.set_postfix({
            'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
            'fw': f"{times['forward']*1000:.1f}ms",
            'bw': f"{times['backward']*1000:.1f}ms"
        })

        # 🛑 [诊断打印] 每10个Batch打印一次详细报告
        # if i % 10 == 0:
            # print(f"\n[诊断 Batch {i}] "
                  # f"数据加载: {times['data']*1000:.1f}ms | "
                  # f"前向(APT): {times['forward']*1000:.1f}ms | "
                  # f"反向(DBC): {times['backward']*1000:.1f}ms | "
                  # f"更新: {times['step']*1000:.1f}ms")

        # 重置下一轮计时起点
        t_start = time.time()

    # 【最后一步清理】处理剩余的累积梯度
    try:
        if (i + 1) % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
    except NameError:
         pass

    return (total_loss / total_steps) if total_steps > 0 else 0

def generate_text(model, tokenizer, input_text, device, max_length=50, repetition_penalty=1.5):
    """
    生成文本（修改版：直接调用模型内部修好的 generate 方法）
    """
    # 1. 编码 Encoder 输入 (Prompt)
    # 这一步必须做，把字符串变成 Tensor
    input_encoded = tokenizer(input_text, max_length=64, padding=False, return_tensors='pt')
    input_ids = input_encoded['input_ids'].to(device)

    # 2. 【关键修改】直接调用模型内部方法 (正规军)
    # 我们刚刚在 apt_model.py 里修好了 generate，现在这里直接用它！
    # 它内部已经包含了：Encoder-Decoder逻辑、EOS切除、强力去重
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,  # 传递惩罚系数，治愈复读机
        temperature=0.1,      # 可以微调，1.0 比较标准
        top_p=0.5,              # ➕ 新增这行！只看前 50% 可信的词，过滤掉胡言乱语
        do_sample=True        # 建议 True，让回答稍微灵活点；如果要死板准确就 False
    )

    # 3. 解码
    # model.generate 返回的是 [batch, seq_len]，我们取第一个样本 [0]
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

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

    ACCUMULATION_STEPS = 2  # 保持原始配置：batch_size=4, 4 * 8 = 32 的有效批次大小

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

    # 3. 准备读档或新建模型
    # 🚨【关键设置】在这里填入您的存档文件名！
    # 如果填空字符串 ""，则代表【从头开始新训练】
    resume_checkpoint = "hlbd_model_20251222_074140.pt"   # <--- 请修改这里的文件名！
    resume_path = os.path.join(project_root, 'tests', 'saved_models', resume_checkpoint)

    model = None
    tokenizer = None
    config = None

    # [逻辑分支] 决定是“读档”还是“新建”
    if resume_checkpoint and os.path.exists(resume_path):
        print(f"\n🔄 发现存档，正在恢复训练: {resume_path}")
        # A. 读档模式：加载旧的分词器和模型 (恢复记忆)
        model, tokenizer, info = load_model_and_tokenizer(resume_path, device)
        config = model.config
        print(f"   已继承之前的词汇表 (Size: {len(tokenizer.char_to_id)})")

        # 👇👇👇 【这里是插入点！MATH 疫苗补丁】 👇👇👇
        # 即使你现在只用旧数据，打上这个补丁也没有坏处，防止未来报错
        if '[MATH]' not in tokenizer.char_to_id:
            print(f"\n💉 检测到旧模型缺少 [MATH] 标签，正在动态补丁...")
            
            # 1. 分配新ID
            new_id = tokenizer.next_id
            tokenizer.char_to_id['[MATH]'] = new_id
            tokenizer.id_to_char[new_id] = '[MATH]'
            tokenizer.vocab['[MATH]'] = new_id  # 保持字典同步
            tokenizer.next_id += 1
            tokenizer.vocab_size += 1 # 词表大小+1
            
            # 2. 🚨 关键：更新正则！否则分词器看不见这个标签
            # 必须把 [MATH] 加入到识别规则里
            tokenizer.tag_pattern = re.compile(r'(\[EMOJI\]|\[PHRASE\]|\[EN\]|\[PY\]|\[JP\]|\[KR\]|\[MATH\])')
            
            print(f"   ✅ [MATH] 补丁应用成功 (ID: {new_id})")
        else:
            print(f"   ✅ 模型已包含 [MATH] 标签 (ID: {tokenizer.char_to_id['[MATH]']})")
        # 👆👆👆 【补丁结束】 👆👆👆

        # 👇👇👇 【这里插入：强制脑科手术】 👇👇👇
        # 必须在这里手动把 Dropout 拉高，因为 torch.load 会恢复成旧的 0.0 或 0.1
        print(f"\n💉 [手术中] 正在强制提升 Dropout (防止死记硬背)...")
        print(f"   原 Dropout 配置: {model.config.dropout}")
        
        # 1. 修改配置参数 (为了保存时正确)
        model.config.dropout = 0.1
        
        # 2. 🚨 关键：递归修改所有正在运行的层
        # 光改 config 没用，必须深入到每一层神经网络里去改 p 值
        modified_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.1
                modified_count += 1
                
        print(f"   ✅ 手术完成！已将 {modified_count} 个 Dropout 层的概率强制设为 {model.config.dropout}")
        print(f"   当前 Dropout 配置: {model.config.dropout}")
        # 👆👆👆 【手术结束】 👆👆👆

    else:
        print(f"\n🆕 未找到存档或未指定，开始【从头训练】...")
        # B. 新建模式：创建新的空白分词器
        print(f"🔧 准备分词器...")
        tokenizer = SimpleCharTokenizer_BACKUP()

    # 4. 创建数据集 
    # (无论读档还是新建，都要跑这一步。如果是读档，tokenizer 会自动沿用旧ID，不会乱码)
    print(f"\n📊 创建数据集...")
    dataset = SimpleDialogueDataset(training_pairs, tokenizer)

    # 预填充/更新词汇表
    print(f"\n📝 检查词汇表覆盖率...")
    for src, tgt in training_pairs:
        tokenizer.encode(src)
        tokenizer.encode(tgt)
    print(f"   当前词汇表大小: {len(tokenizer.char_to_id)} (预留空间: {tokenizer.vocab_size})")

    dataloader = DataLoader(
        dataset,
        batch_size=16, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    print(f"   训练批次数: {len(dataloader)}")

    # 5. 确保模型已创建 
    # (如果是新建模式，现在才创建模型；如果是读档，上面已经有了)
    if model is None:
        print(f"\n🏗️ 创建APT模型 (Fresh Start)...")
        config = create_small_hlbd_config(tokenizer.vocab_size)
        model = APTModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")

    # 创建一个副本用于对比 (防止报错，随便建一个)
    untrained_model = APTModel(config).to(device)
    if not resume_checkpoint:
        untrained_model.load_state_dict(model.state_dict())
    untrained_model.eval()

    # ============================================================
    # 🎛️ 战术指挥中心 (只需要改这里！)
    # ============================================================
    # 模式选择:
    # "BREAKOUT" (暴力破局) -> LR=1e-4, DBC=关 (用于把 Loss 炸高，跳出局部最优)
    # "LANDING"  (平稳降落) -> LR=1e-5, DBC=开 (用于把高 Loss 收敛，精细学习)
    
    TACTICAL_MODE = "LANDING"  # 👈 当前任务：降落！(要把 Loss 1.8 降下来)

    if TACTICAL_MODE == "BREAKOUT":
        current_lr = 8e-5
        use_dbc = False
        mode_msg = "🔥 [暴力破局模式] 全力输出，允许梯度爆发"
    else: # LANDING
        current_lr = 1e-5
        use_dbc = True
        mode_msg = "❄️ [平稳降落模式] 开启辅助，精细收敛"

    # ============================================================

    # 6. 创建优化器
    # ------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-2)
    print(f"🔧 优化器配置完成 | 模式: {TACTICAL_MODE} | LR: {current_lr} | Weight Decay: 1e-2")
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 7. 注册DBC hooks (智能控制版)
    # ------------------------------------------------------------
    print(f"\n⚡ [步骤 7] DBC-DAC 状态: {mode_msg}")
    
    # 先清理旧钩子 (防止残留)
    if hasattr(model, 'gradient_hooks'):
        model.gradient_hooks = []
    
    if use_dbc:
        # ✅ 开启模式：注册钩子
        hooks = register_dbc_hooks(model)
        print(f"   ✅ 已激活 {len(hooks)} 个梯度稳定钩子 (降落伞已打开)")
    else:
        # 🛑 禁用模式：确保裸奔
        print(f"   🚫 梯度钩子已移除 (限制已解除)")

    # 8. 训练模型
    print(f"\n" + "="*60)
    print("🏃 开始快速训练 (看能否快速学会说话)")
    print("="*60)

    num_epochs = 30  # 600个训练对（100概念×6层级：emoji/短语/英文/拼音/日文/韩文→中文）

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

            # 👇👇👇 【这里插入自动存档代码】 👇👇👇
            print(f"💾 正在自动存档 (Epoch {epoch+1})...")
            save_dir = os.path.join(project_root, 'tests', 'saved_models')
            
            # 这里调用保存函数
            # 注意：num_epochs 参数传入当前的 epoch+1，这样你知道这个档是跑了多少轮的
            save_model_and_tokenizer(
                model=model,
                tokenizer=tokenizer,
                config=model.config,  # 确保传入配置
                save_dir=save_dir,
                num_epochs=epoch+1,   # 记录当前进度
                final_loss=loss
            )
            print("------------------------------------------------")

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
