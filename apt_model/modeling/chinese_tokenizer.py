#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型中文分词器集成
该模块将中文分词器与APT模型训练系统集成
"""

import os
import logging
from typing import Optional, Dict, List, Any, Tuple

from apt_model.modeling.basic_tokenizer import BasicEnglishTokenizer

class ChineseTokenizer:
    """
    中文分词器实现
    支持字符级和词级分词
    """
    def __init__(self, vocab_file=None, mode="char", vocab_size=None, texts=None):
        """
        初始化中文分词器
        
        参数:
            vocab_file: 词汇表文件路径
            mode: 分词模式 ('char'或'word')
            vocab_size: 词汇表大小
            texts: 用于构建词汇表的文本列表
        """
        self.mode = mode
        self.vocab_size = vocab_size or 50257  # 默认词汇表大小
        self.special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|endoftext|>",
        }
        
        # 初始化编码器和解码器（词汇表）
        self.encoder = {}  # 文本 -> ID
        self.decoder = {}  # ID -> 文本
        
        # 如果提供了词汇表文件，从文件加载
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocabulary(vocab_file)
        # 否则，如果提供了文本，从文本构建词汇表
        elif texts:
            self.prepare_vocab_from_texts(texts)
        # 否则，初始化基本词汇表
        else:
            self._initialize_basic_vocab()
    
    def _initialize_basic_vocab(self):
        """初始化基本词汇表（特殊标记和常用字符）"""
        # 添加特殊标记
        for i, token in enumerate(self.special_tokens.values()):
            self.encoder[token] = i
            self.decoder[i] = token
        
        # 如果是字符级模式，添加基本汉字
        if self.mode == "char":
            # 添加ASCII字符
            for i in range(32, 127):
                char = chr(i)
                if char not in self.encoder:
                    self.encoder[char] = len(self.encoder)
                    
            # 添加基本汉字（常用字）
            for i in range(0x4e00, 0x9fff):  # 基本汉字Unicode范围
                if len(self.encoder) >= self.vocab_size:
                    break
                char = chr(i)
                if char not in self.encoder:
                    self.encoder[char] = len(self.encoder)
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.encoder)
    
    def prepare_vocab_from_texts(self, texts):
        """
        从文本构建词汇表
        
        参数:
            texts: 文本列表
        """
        # 添加特殊标记
        for i, token in enumerate(self.special_tokens.values()):
            self.encoder[token] = i
        
        # 字符级分词
        if self.mode == "char":
            # 首先添加ASCII字符
            for i in range(32, 127):
                char = chr(i)
                if char not in self.encoder:
                    self.encoder[char] = len(self.encoder)
            
            # 从文本中统计字符频率
            char_freq = {}
            for text in texts:
                for char in text:
                    if char not in self.encoder:
                        char_freq[char] = char_freq.get(char, 0) + 1
            
            # 按频率排序添加字符
            for char, _ in sorted(char_freq.items(), key=lambda x: x[1], reverse=True):
                if len(self.encoder) >= self.vocab_size:
                    break
                if char not in self.encoder:
                    self.encoder[char] = len(self.encoder)
        
        # 词级分词（需要分词工具）
        elif self.mode == "word":
            try:
                import jieba
                
                # 统计词频
                word_freq = {}
                for text in texts:
                    words = jieba.cut(text)
                    for word in words:
                        if word not in self.encoder:
                            word_freq[word] = word_freq.get(word, 0) + 1
                
                # 按频率排序添加词语
                for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
                    if len(self.encoder) >= self.vocab_size:
                        break
                    if word not in self.encoder:
                        self.encoder[word] = len(self.encoder)
            except ImportError:
                print("未找到jieba分词库，无法进行词级分词。将使用字符级分词替代。")
                self.mode = "char"
                self.prepare_vocab_from_texts(texts)  # 递归调用字符级分词
        
        # 更新解码器和词汇表大小
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.encoder)
    
    def encode(self, text, return_tensors=None, max_length=None, truncation=None):
        """
        将文本编码为ID
        
        参数:
            text: 输入文本
            return_tensors: 返回张量类型 ('pt'表示PyTorch)
            max_length: 最大长度
            truncation: 是否截断
            
        返回:
            编码后的ID列表或张量
        """
        ids = []
        
        # 字符级分词
        if self.mode == "char":
            for char in text:
                # 对未知字符使用特殊标记或跳过
                if char in self.encoder:
                    ids.append(self.encoder[char])
                else:
                    # 未知字符可以用UNK标记替代，或者跳过
                    pass
        
        # 词级分词
        elif self.mode == "word":
            try:
                import jieba
                words = jieba.cut(text)
                for word in words:
                    if word in self.encoder:
                        ids.append(self.encoder[word])
                    else:
                        # 未知词可以分解为字符
                        for char in word:
                            if char in self.encoder:
                                ids.append(self.encoder[char])
                            # 否则跳过或使用UNK
            except ImportError:
                # 如果没有jieba，回退到字符级
                for char in text:
                    if char in self.encoder:
                        ids.append(self.encoder[char])
        
        # 处理最大长度
        if max_length and truncation and len(ids) > max_length:
            ids = ids[:max_length]
        
        # 返回张量
        if return_tensors == "pt":
            try:
                import torch
                return torch.tensor([ids])
            except ImportError:
                return ids
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """
        将ID解码为文本
        
        参数:
            ids: ID列表或张量
            skip_special_tokens: 是否跳过特殊标记
            
        返回:
            解码后的文本
        """
        # 处理张量
        try:
            import torch
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
        except ImportError:
            pass
        
        # 如果是嵌套列表，取第一个
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        
        # 解码
        text = ""
        for idx in ids:
            if idx in self.decoder:
                token = self.decoder[idx]
                # 跳过特殊标记
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                text += token
        
        return text
    
    def save_vocabulary(self, save_directory):
        """
        保存词汇表到文件
        
        参数:
            save_directory: 保存目录
            
        返回:
            保存的文件路径列表
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.json")
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        
        return [vocab_file]
    
    def load_vocabulary(self, vocab_file):
        """
        从文件加载词汇表
        
        参数:
            vocab_file: 词汇表文件路径
            
        返回:
            bool: 是否成功加载
        """
        import json
        
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.encoder = json.load(f)
            self.decoder = {v: k for k, v in self.encoder.items()}
            self.vocab_size = len(self.encoder)
            return True
        except Exception as e:
            print(f"加载词汇表失败: {e}")
            return False
    
    @property
    def pad_token(self):
        return self.special_tokens["pad_token"]
    
    @pad_token.setter
    def pad_token(self, value):
        old_token = self.special_tokens["pad_token"]
        self.special_tokens["pad_token"] = value
        
        # 更新编码器和解码器
        if old_token in self.encoder:
            token_id = self.encoder[old_token]
            del self.encoder[old_token]
            self.encoder[value] = token_id
            self.decoder[token_id] = value
    
    @property
    def pad_token_id(self):
        return self.encoder.get(self.pad_token)
    
    @property
    def eos_token(self):
        return self.special_tokens["eos_token"]
    
    @eos_token.setter
    def eos_token(self, value):
        old_token = self.special_tokens["eos_token"]
        self.special_tokens["eos_token"] = value
        
        # 更新编码器和解码器
        if old_token in self.encoder:
            token_id = self.encoder[old_token]
            del self.encoder[old_token]
            self.encoder[value] = token_id
            self.decoder[token_id] = value
    
    @property
    def eos_token_id(self):
        return self.encoder.get(self.eos_token)
    
    @property
    def bos_token(self):
        return self.special_tokens["bos_token"]
    
    @bos_token.setter
    def bos_token(self, value):
        old_token = self.special_tokens["bos_token"]
        self.special_tokens["bos_token"] = value
        
        # 更新编码器和解码器
        if old_token in self.encoder:
            token_id = self.encoder[old_token]
            del self.encoder[old_token]
            self.encoder[value] = token_id
            self.decoder[token_id] = value
    
    @property
    def bos_token_id(self):
        return self.encoder.get(self.bos_token)

    def save_pretrained(self, save_directory):
        """
        与 HuggingFace 兼容的保存方法
    
        参数:
            save_directory: 保存目录
            
        返回:
            保存的文件路径列表
        """
        return self.save_vocabulary(save_directory)

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
        tokenizer = create_chinese_tokenizer(mode=mode, vocab_size=vocab_size, texts=texts)
        
        # 设置特殊标记（确保和GPT2分词器保持一致）
        tokenizer.bos_token = '<|endoftext|>'
        tokenizer.eos_token = '<|endoftext|>'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"中文分词器初始化完成，词汇表大小: {tokenizer.vocab_size}")
        return tokenizer
    else:
        logger.info("初始化本地英语分词器（无需 Hugging Face 下载）")
        return BasicEnglishTokenizer(texts=texts, vocab_size=vocab_size)


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
        os.makedirs(path, exist_ok=True)
        if isinstance(tokenizer, ChineseTokenizer):
            tokenizer.save_vocabulary(path)
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
        config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if config.get("type") == "chinese":
                vocab_file = os.path.join(path, "vocab.json")
                mode = config.get("mode", "char")
                tokenizer = ChineseTokenizer(vocab_file=vocab_file, mode=mode)
                logger.info(f"已加载中文分词器，模式: {mode}, 词汇表大小: {tokenizer.vocab_size}")
                return tokenizer
        logger.info("加载本地英语分词器配置失败，回退到基础词表")
        return BasicEnglishTokenizer()
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
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len(text.strip())
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
    combined_text = " ".join(texts[:100])
    if len(combined_text) > 5000:
        combined_text = combined_text[:5000]
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
    detected_language = language or detect_language(texts)
    if detected_language == "zh" and not tokenizer_type:
        tokenizer_type = "chinese-char"
    elif not tokenizer_type:
        tokenizer_type = "gpt2"
    tokenizer = get_tokenizer(
        tokenizer_type=tokenizer_type,
        language=detected_language,
        texts=texts
    )
    return tokenizer, detected_language


def create_chinese_tokenizer(mode="char", vocab_size=None, texts=None):
    """
    创建中文分词器
    
    参数:
        mode: 分词模式 ('char'或'word')
        vocab_size: 目标词汇表大小（如果指定）
        texts: 用于构建词汇表的文本列表
        
    返回:
        ChineseTokenizer实例
    """
    tokenizer = ChineseTokenizer(mode=mode)
    
    if texts:
        # 从文本构建词汇表
        tokenizer.prepare_vocab_from_texts(texts)
        
        # 如果指定了目标词汇表大小，确保达到该大小
        if vocab_size is not None and tokenizer.vocab_size < vocab_size:
            for i in range(0x4e00, 0x9fff):
                char = chr(i)
                if char not in tokenizer.encoder and tokenizer.vocab_size < vocab_size:
                    tokenizer.encoder[char] = len(tokenizer.encoder)
            tokenizer.decoder = {v: k for k, v in tokenizer.encoder.items()}
    
    return tokenizer
