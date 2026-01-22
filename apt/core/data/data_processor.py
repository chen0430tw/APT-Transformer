#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Data Processor
数据预处理和清洗工具
"""

import os
import re
import json
import random
import logging
from typing import List, Dict, Union, Tuple, Optional, Any, Callable
from collections import Counter, defaultdict
import unicodedata

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('apt_model.data_processor')

class DataProcessor:
    """
    数据预处理和清洗工具类
    用于处理和准备机器学习模型的文本数据
    """
    
    def __init__(self, tokenizer=None, max_seq_length=512, lower_case=False, 
                 remove_accents=False, clean_text=True, language='en'):
        """
        初始化数据处理器
        
        参数:
            tokenizer: 用于分词的分词器
            max_seq_length: 最大序列长度
            lower_case: 是否将文本转为小写
            remove_accents: 是否移除重音符号
            clean_text: 是否进行基础文本清洗
            language: 文本语言，影响特定处理规则，目前支持'en'和'zh'
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.lower_case = lower_case
        self.remove_accents = remove_accents
        self.clean_text = clean_text
        self.language = language.lower()
        
        # 验证语言支持
        supported_languages = ['en', 'zh']
        if self.language not in supported_languages:
            logger.warning(f"语言'{self.language}'不在支持列表中{supported_languages}，默认使用'en'")
            self.language = 'en'
    
    def process_text(self, text: str) -> str:
        """
        处理单个文本
        
        参数:
            text: 要处理的文本
            
        返回:
            处理后的文本
        """
        if not text:
            return ""
        
        # 基础文本清洗
        if self.clean_text:
            text = self._clean_text(text)
        
        # 转为小写
        if self.lower_case:
            text = text.lower()
        
        # 移除重音符号
        if self.remove_accents:
            text = self._remove_accents(text)
        
        # 基于语言的特定处理
        if self.language == 'zh':
            text = self._process_chinese_text(text)
        elif self.language == 'en':
            text = self._process_english_text(text)
        
        return text.strip()
    
    def process_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        批量处理文本
        
        参数:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        返回:
            处理后的文本列表
        """
        if not texts:
            return []
        
        processed_texts = []
        
        if show_progress:
            iterator = tqdm(texts, desc="处理文本")
        else:
            iterator = texts
        
        for text in iterator:
            processed_text = self.process_text(text)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """
        使用分词器对文本进行分词
        
        参数:
            text: 要分词的文本
            
        返回:
            分词结果，包含input_ids等
        """
        if not self.tokenizer:
            raise ValueError("未设置分词器，请先设置tokenizer")
        
        # 预处理文本
        processed_text = self.process_text(text)
        
        # 分词
        encoding = self.tokenizer(
            processed_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return encoding
    
    def tokenize_batch(self, texts: List[str], return_tensors: str = "pt") -> Dict[str, Any]:
        """
        批量分词
        
        参数:
            texts: 文本列表
            return_tensors: 返回的张量类型，'pt'为PyTorch
            
        返回:
            批量分词结果
        """
        if not self.tokenizer:
            raise ValueError("未设置分词器，请先设置tokenizer")
        
        # 预处理文本
        processed_texts = self.process_batch(texts)
        
        # 批量分词
        encodings = self.tokenizer(
            processed_texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )
        
        return encodings
    
    def prepare_training_features(self, examples: Dict[str, List[Any]], 
                                 text_column: str, target_column: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        准备训练特征
        
        参数:
            examples: 数据样本，格式为{列名: [值1, 值2, ...], ...}
            text_column: 文本列名
            target_column: 目标列名，用于监督学习
            
        返回:
            处理后的特征，可直接用于模型训练
        """
        if not self.tokenizer:
            raise ValueError("未设置分词器，请先设置tokenizer")
        
        texts = examples[text_column]
        
        # 预处理和分词
        encodings = self.tokenize_batch(texts)
        
        # 如果有目标列，添加标签
        if target_column and target_column in examples:
            encodings["labels"] = torch.tensor(examples[target_column])
        
        return encodings
    
    def create_dataset(self, texts: List[str], labels: Optional[List[Any]] = None) -> torch.utils.data.Dataset:
        """
        创建PyTorch数据集
        
        参数:
            texts: 文本列表
            labels: 标签列表，可选
            
        返回:
            PyTorch数据集，可用于DataLoader
        """
        if not self.tokenizer:
            raise ValueError("未设置分词器，请先设置tokenizer")
        
        # 创建简单的TextDataset类
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels=None):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.encodings["input_ids"])
            
            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                if self.labels is not None:
                    item["labels"] = self.labels[idx]
                return item
        
        # 预处理和分词
        encodings = self.tokenize_batch(texts)
        
        # 创建数据集
        dataset = TextDataset(encodings, labels)
        
        return dataset
    
    def _clean_text(self, text: str) -> str:
        """
        基础文本清洗
        
        参数:
            text: 要清洗的文本
            
        返回:
            清洗后的文本
        """
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 替换多个换行为单个换行
        text = re.sub(r'\n+', '\n', text)
        
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 基于语言的特定清洗
        if self.language == 'zh':
            # 移除英文字符之间的空格（中文文本中）
            text = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', text)
        
        return text.strip()
    
    def _remove_accents(self, text: str) -> str:
        """
        移除重音符号
        
        参数:
            text: 输入文本
            
        返回:
            移除重音符号后的文本
        """
        text = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])
    
    def _process_chinese_text(self, text: str) -> str:
        """
        处理中文文本
        
        参数:
            text: 中文文本
            
        返回:
            处理后的中文文本
        """
        # 全角转半角
        text = self._full_to_half(text)
        
        # 繁体转简体（如果需要）
        # 需要额外的库支持，如OpenCC
        
        return text
    
    def _process_english_text(self, text: str) -> str:
        """
        处理英文文本
        
        参数:
            text: 英文文本
            
        返回:
            处理后的英文文本
        """
        # 标准化标点符号
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # 确保句号后有空格
        
        # 标准化引号
        text = re.sub(r'`{1,2}', '"', text)
        text = re.sub(r'´{1,2}', '"', text)
        
        # 缩写的标准化处理
        text = re.sub(r'(\w)\'(\w)', r'\1\'\2', text)  # 撇号的标准化
        
        return text
    
    def _full_to_half(self, text: str) -> str:
        """
        全角转半角
        
        参数:
            text: 包含全角字符的文本
            
        返回:
            转换后的文本
        """
        result = ''
        for char in text:
            code = ord(char)
            if code == 0x3000:  # 全角空格
                char = ' '
            elif 0xFF01 <= code <= 0xFF5E:  # 全角字符
                char = chr(code - 0xFEE0)
            result += char
        return result

    @staticmethod
    def augment_data(texts: List[str], labels: Optional[List[Any]] = None, 
                    augmentation_methods: List[str] = None, 
                    augmentation_factor: int = 1) -> Tuple[List[str], Optional[List[Any]]]:
        """
        数据增强
        
        参数:
            texts: 原始文本列表
            labels: 原始标签列表
            augmentation_methods: 增强方法列表，可包括'swap'、'delete'、'replace'、'insert'
            augmentation_factor: 每个样本增强的次数
            
        返回:
            增强后的文本和标签
        """
        if not augmentation_methods:
            augmentation_methods = ['swap', 'delete']
        
        augmented_texts = list(texts)
        augmented_labels = list(labels) if labels else None
        
        for text_idx, text in enumerate(texts):
            for _ in range(augmentation_factor):
                augmentation_method = random.choice(augmentation_methods)
                augmented_text = text
                
                # 应用选定的增强方法
                if augmentation_method == 'swap':
                    augmented_text = DataProcessor._augment_by_swap(augmented_text)
                elif augmentation_method == 'delete':
                    augmented_text = DataProcessor._augment_by_delete(augmented_text)
                elif augmentation_method == 'replace':
                    augmented_text = DataProcessor._augment_by_replace(augmented_text)
                elif augmentation_method == 'insert':
                    augmented_text = DataProcessor._augment_by_insert(augmented_text)
                
                # 添加增强后的文本和标签
                if augmented_text != text:
                    augmented_texts.append(augmented_text)
                    if labels:
                        augmented_labels.append(labels[text_idx])
        
        return augmented_texts, augmented_labels
    
    @staticmethod
    def _augment_by_swap(text: str) -> str:
        """通过随机交换相邻词增强文本"""
        words = text.split()
        if len(words) <= 1:
            return text
            
        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
        return ' '.join(words)
    
    @staticmethod
    def _augment_by_delete(text: str) -> str:
        """通过随机删除词增强文本"""
        words = text.split()
        if len(words) <= 1:
            return text
            
        i = random.randint(0, len(words) - 1)
        words.pop(i)
        return ' '.join(words)
    
    @staticmethod
    def _augment_by_replace(text: str) -> str:
        """通过同义词替换增强文本（简化示例）"""
        # 此处应该使用更复杂的同义词替换逻辑
        # 例如使用WordNet或其他同义词库
        return text
    
    @staticmethod
    def _augment_by_insert(text: str) -> str:
        """通过随机插入词增强文本（简化示例）"""
        # 此处应该使用更复杂的插入逻辑
        # 可以插入虚拟词或从字典中选择合适的词
        return text

class TextCleaner:
    """
    文本清洗工具
    提供一系列静态方法用于文本清洗
    """
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """移除HTML标签"""
        return re.sub(r'<.*?>', '', text)
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """移除URL"""
        return re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    @staticmethod
    def remove_emoji(text: str) -> str:
        """移除表情符号"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """标准化空白字符"""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """移除标点符号"""
        return re.sub(r'[^\w\s]', '', text)
    
    @staticmethod
    def clean_email(text: str) -> str:
        """清理邮箱地址"""
        return re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    @staticmethod
    def clean_phone_numbers(text: str) -> str:
        """清理电话号码"""
        return re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
    
    @staticmethod
    def clean_numbers(text: str) -> str:
        """清理数字"""
        return re.sub(r'\d+', '[NUMBER]', text)
    
    @staticmethod
    def clean_text_complete(text: str) -> str:
        """完整的文本清洗"""
        # 应用多个清洗方法
        text = TextCleaner.remove_html_tags(text)
        text = TextCleaner.remove_urls(text)
        text = TextCleaner.remove_emoji(text)
        text = TextCleaner.clean_email(text)
        text = TextCleaner.normalize_whitespace(text)
        return text.strip()

class DatasetStatistics:
    """
    数据集统计工具
    用于分析数据集的特性和统计信息
    """
    
    @staticmethod
    def get_text_length_stats(texts: List[str]) -> Dict[str, float]:
        """
        获取文本长度统计信息
        
        参数:
            texts: 文本列表
            
        返回:
            包含平均长度、中位数长度等的字典
        """
        if not texts:
            return {
                "avg_length": 0,
                "median_length": 0,
                "min_length": 0,
                "max_length": 0,
                "std_length": 0
            }
        
        lengths = [len(text.split()) for text in texts]
        
        return {
            "avg_length": np.mean(lengths),
            "median_length": np.median(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_length": np.std(lengths)
        }
    
    @staticmethod
    def get_vocabulary_stats(texts: List[str]) -> Dict[str, int]:
        """
        获取词汇统计信息
        
        参数:
            texts: 文本列表
            
        返回:
            包含词汇量、唯一词数等的字典
        """
        if not texts:
            return {
                "total_words": 0,
                "unique_words": 0,
                "vocabulary_size": 0
            }
        
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        word_counter = Counter(all_words)
        
        return {
            "total_words": len(all_words),
            "unique_words": len(word_counter),
            "vocabulary_size": len(word_counter)
        }
    
    @staticmethod
    def get_frequent_words(texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取最常见的词
        
        参数:
            texts: 文本列表
            top_n: 返回的常见词数量
            
        返回:
            (词, 频率)元组的列表
        """
        if not texts:
            return []
        
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        word_counter = Counter(all_words)
        return word_counter.most_common(top_n)
    
    @staticmethod
    def get_label_distribution(labels: List[Any]) -> Dict[Any, int]:
        """
        获取标签分布
        
        参数:
            labels: 标签列表
            
        返回:
            标签频率字典
        """
        if not labels:
            return {}
        
        return dict(Counter(labels))
    
    @staticmethod
    def summarize_dataset(texts: List[str], labels: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        生成数据集摘要
        
        参数:
            texts: 文本列表
            labels: 标签列表，可选
            
        返回:
            包含各种统计信息的字典
        """
        summary = {
            "num_samples": len(texts),
            "text_length_stats": DatasetStatistics.get_text_length_stats(texts),
            "vocabulary_stats": DatasetStatistics.get_vocabulary_stats(texts),
            "frequent_words": DatasetStatistics.get_frequent_words(texts, top_n=10)
        }
        
        if labels:
            summary["label_distribution"] = DatasetStatistics.get_label_distribution(labels)
        
        return summary
    
    @staticmethod
    def print_dataset_summary(summary: Dict[str, Any]):
        """
        打印数据集摘要
        
        参数:
            summary: 由summarize_dataset生成的摘要
        """
        print("=" * 50)
        print("数据集摘要")
        print("=" * 50)
        
        print(f"样本数量: {summary['num_samples']}")
        
        print("\n文本长度统计:")
        for key, value in summary['text_length_stats'].items():
            print(f"  - {key}: {value:.2f}")
        
        print("\n词汇统计:")
        for key, value in summary['vocabulary_stats'].items():
            print(f"  - {key}: {value}")
        
        print("\n最常见的词:")
        for word, count in summary['frequent_words']:
            print(f"  - {word}: {count}")
        
        if 'label_distribution' in summary:
            print("\n标签分布:")
            for label, count in summary['label_distribution'].items():
                print(f"  - {label}: {count}")
        
        print("=" * 50)

# 便捷函数，直接使用

def clean_text(text: str) -> str:
    """
    快速清洗文本
    
    参数:
        text: 要清洗的文本
        
    返回:
        清洗后的文本
    """
    return TextCleaner.clean_text_complete(text)

def get_dataset_summary(texts: List[str], labels: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    获取数据集摘要的便捷函数
    
    参数:
        texts: 文本列表
        labels: 标签列表，可选
        
    返回:
        数据集摘要
    """
    return DatasetStatistics.summarize_dataset(texts, labels)

def create_data_processor(tokenizer=None, max_seq_length=512, language='en') -> DataProcessor:
    """
    创建数据处理器的便捷函数
    
    参数:
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        language: 语言
        
    返回:
        配置好的DataProcessor实例
    """
    return DataProcessor(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        lower_case=True,
        remove_accents=True,
        clean_text=True,
        language=language
    )

def get_training_texts() -> List[str]:
    """
    加载训练文本数据。
    这里以从本地文件 'train.txt' 加载为例，请根据你的实际情况调整。
    """
    try:
        with open("train.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts
    except FileNotFoundError:
        print("未找到训练数据文件 'train.txt'，返回空列表。")
        return []
