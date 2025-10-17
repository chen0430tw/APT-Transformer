#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型中文分词器集成
该模块将中文分词器与APT模型训练系统集成
"""

import os
import logging
from typing import Optional, Dict, List, Any, Tuple

from transformers import GPT2Tokenizer, PreTrainedTokenizer
from apt_model.modeling.chinese_tokenizer import ChineseTokenizer

def integrate_chinese_tokenizer(*args, **kwargs) -> PreTrainedTokenizer:
    """
    集成中文分词器。
    
    根据传入的参数创建并返回一个ChineseTokenizer实例。
    
    参数可以包括：
      - mode: 分词模式，'char' 或 'word'
      - vocab_size: 词汇表大小
      - texts: 用于构建词汇表的文本列表
      - 其他自定义参数...
    
    返回:
        一个预训练的中文分词器实例。
    """
    # 你可以在这里根据需要自定义更多逻辑
    mode = kwargs.get("mode", "char")
    vocab_size = kwargs.get("vocab_size", 50257)
    texts = kwargs.get("texts", None)
    
    tokenizer = ChineseTokenizer(vocab_size=vocab_size, mode=mode, texts=texts)
    return tokenizer


def get_tokenizer(tokenizer_type="gpt2", language="en", texts=None, vocab_size=50257, cache_dir=None):
    """
    获取适合APT模型的分词器
    
    参数:
        tokenizer_type: 分词器类型 ('gpt2', 'chinese-char', 'chinese-word')
        language: 语言代码 ('en', 'zh')
        texts: 用于构建词汇表的文本列表（中文分词器使用）
        vocab_size: 词汇表大小
        cache_dir: 缓存目录
        
    返回:
        分词器实例
    """
    logger = logging.getLogger('apt_model.tokenizer')
    
    # 根据语言和类型确定分词器
    if language == "zh" or tokenizer_type.startswith("chinese"):
        logger.info(f"初始化中文分词器 (类型: {tokenizer_type})")
        
        # 确定分词模式
        mode = "char"
        if tokenizer_type == "chinese-word":
            mode = "word"
        
        # 创建中文分词器
        tokenizer = ChineseTokenizer(vocab_size=vocab_size, mode=mode, texts=texts)
        
        # 设置特殊标记（确保和GPT2分词器保持一致）
        tokenizer.bos_token = '<|endoftext|>'
        tokenizer.eos_token = '<|endoftext|>'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"中文分词器初始化完成，词汇表大小: {tokenizer.vocab_size}")
        return tokenizer
    else:
        # 使用默认的GPT2分词器
        logger.info(f"加载GPT2分词器 (语言: {language})")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
            # GPT2分词器没有pad_token，设置它等于eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.error(f"加载GPT2分词器失败: {e}")
            logger.warning("创建备用简单分词器")
            
            # 简单的备用分词器
            class SimpleTokenizer:
                def __init__(self):
                    self.vocab_size = vocab_size
                    self.pad_token_id = 0
                    self.eos_token_id = 1
                    self.pad_token = "<pad>"
                    self.eos_token = "<eos>"
                    
                def encode(self, text, return_tensors=None, max_length=None, truncation=None):
                    # 非常简单的分词 - 仅以空格分割
                    tokens = text.split()
                    if max_length and truncation and len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    # 转换为ID (简单地使用哈希)
                    ids = [hash(t) % 10000 + 10 for t in tokens]
                    if return_tensors == "pt":
                        import torch
                        return torch.tensor([ids])
                    return ids
                
                def decode(self, ids, skip_special_tokens=True):
                    # 简单解码（不可逆，仅用于测试）
                    return " ".join([f"<token_{id}>" for id in ids])
                
            return SimpleTokenizer()


def save_tokenizer(tokenizer, path):
    """
    保存分词器到指定路径
    
    参数:
        tokenizer: 分词器实例
        path: 保存路径
        
    返回:
        bool: 是否成功保存
    """
    try:
        # 创建目录
        os.makedirs(path, exist_ok=True)
        
        # 根据分词器类型保存
        if isinstance(tokenizer, ChineseTokenizer):
            tokenizer.save_vocabulary(path)
            # 保存配置
            with open(os.path.join(path, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
                import json
                json.dump({
                    "type": "chinese",
                    "mode": tokenizer.mode,
                    "vocab_size": tokenizer.vocab_size,
                    "special_tokens": tokenizer.special_tokens
                }, f, ensure_ascii=False, indent=2)
        elif hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(path)
        else:
            # 对于简单分词器，不支持保存
            return False
        
        return True
    except Exception as e:
        logger = logging.getLogger('apt_model.tokenizer')
        logger.error(f"保存分词器失败: {e}")
        return False


def load_tokenizer(path):
    """
    从路径加载分词器
    
    参数:
        path: 分词器路径
        
    返回:
        加载的分词器实例
    """
    logger = logging.getLogger('apt_model.tokenizer')
    
    try:
        # 检查是中文分词器还是GPT2分词器
        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if config.get("type") == "chinese":
                # 加载中文分词器
                vocab_file = os.path.join(path, "vocab.json")
                mode = config.get("mode", "char")
                tokenizer = ChineseTokenizer(vocab_file=vocab_file, mode=mode)
                logger.info(f"已加载中文分词器，模式: {mode}, 词汇表大小: {tokenizer.vocab_size}")
                return tokenizer
        
        # 默认尝试加载GPT2分词器
        tokenizer = GPT2Tokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"已加载GPT2分词器")
        return tokenizer
    except Exception as e:
        logger.error(f"加载分词器失败: {e}")
        return None


def is_chinese_text(text, threshold=0.3):
    """
    检测文本是否主要为中文
    
    参数:
        text: 要检测的文本
        threshold: 中文字符占比阈值
        
    返回:
        bool: 是否为中文文本
    """
    # 计算中文字符数量
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text.strip())
    
    # 计算中文字符占比
    if total_chars == 0:
        return False
    
    chinese_ratio = chinese_chars / total_chars
    return chinese_ratio >= threshold


def detect_language(texts):
    """
    检测文本集合的主要语言
    
    参数:
        texts: 文本列表
        
    返回:
        str: 检测到的语言代码 ('zh' 或 'en')
    """
    # 连接所有文本样本，限制总长度避免处理过多文本
    combined_text = " ".join(texts[:100])
    if len(combined_text) > 5000:
        combined_text = combined_text[:5000]
    
    # 检测是否为中文
    if is_chinese_text(combined_text):
        return "zh"
    else:
        return "en"


def get_appropriate_tokenizer(texts, tokenizer_type=None, language=None):
    """
    根据文本内容自动选择合适的分词器
    
    参数:
        texts: 文本列表
        tokenizer_type: 指定的分词器类型 (可选)
        language: 指定的语言 (可选)
        
    返回:
        分词器实例和检测到的语言
    """
    # 如果未指定语言，自动检测
    detected_language = language or detect_language(texts)
    
    # 如果检测到中文且未指定分词器类型，默认使用字符级中文分词器
    if detected_language == "zh" and not tokenizer_type:
        tokenizer_type = "chinese-char"
    elif not tokenizer_type:
        tokenizer_type = "gpt2"
    
    # 获取分词器
    tokenizer = get_tokenizer(
        tokenizer_type=tokenizer_type,
        language=detected_language,
        texts=texts
    )
    
    return tokenizer, detected_language