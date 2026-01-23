"""
Data Processors Plugin for APT
数据处理器插件 - 提供高级数据预处理和增强功能
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
import json
import random
from collections import Counter
import re


class DataProcessorsPlugin:
    """
    数据处理器插件
    
    提供多种数据处理功能:
    1. 文本清洗和标准化
    2. 数据增强 (Data Augmentation)
    3. 数据平衡 (Data Balancing)
    4. 特征工程 (Feature Engineering)
    5. 数据质量检查 (Data Quality Check)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "data-processors"
        self.version = "1.0.0"
        self.config = config
        
        # 配置参数
        self.enable_cleaning = config.get('enable_cleaning', True)
        self.enable_augmentation = config.get('enable_augmentation', True)
        self.augmentation_ratio = config.get('augmentation_ratio', 0.2)
        
        # 统计信息
        self.stats = {
            'processed_samples': 0,
            'augmented_samples': 0,
            'cleaned_samples': 0,
            'filtered_samples': 0
        }
        
        print(f"✅ 数据处理器插件初始化完成")
    
    # ==================== 文本清洗 ====================
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本数据
        
        功能:
        - 去除多余空格
        - 统一标点符号
        - 去除特殊字符
        - 修正常见错误
        """
        if not text:
            return ""
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 统一引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # 去除零宽字符
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # 修正常见拼写错误（可扩展）
        corrections = {
            'teh': 'the',
            'recieve': 'receive',
            'occured': 'occurred',
        }
        
        for wrong, correct in corrections.items():
            text = re.sub(r'\b' + wrong + r'\b', correct, text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_text(self, text: str, lowercase: bool = False) -> str:
        """
        标准化文本
        
        Args:
            text: 输入文本
            lowercase: 是否转小写
        """
        text = self.clean_text(text)
        
        if lowercase:
            text = text.lower()
        
        # 标准化数字（可选）
        if self.config.get('normalize_numbers', False):
            text = re.sub(r'\d+', '<NUM>', text)
        
        # 标准化URL（可选）
        if self.config.get('normalize_urls', False):
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
        
        return text
    
    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """去除重复文本"""
        seen = set()
        unique_texts = []
        
        for text in texts:
            normalized = self.normalize_text(text, lowercase=True)
            if normalized not in seen:
                seen.add(normalized)
                unique_texts.append(text)
        
        removed = len(texts) - len(unique_texts)
        print(f"📊 去除重复: {removed} 条 ({removed/len(texts)*100:.1f}%)")
        
        return unique_texts
    
    # ==================== 数据增强 ====================
    
    def augment_text(self, text: str, methods: List[str] = None) -> List[str]:
        """
        文本数据增强
        
        支持的方法:
        - synonym_replacement: 同义词替换
        - random_insertion: 随机插入
        - random_swap: 随机交换
        - random_deletion: 随机删除
        - back_translation: 回译（需要翻译模型）
        """
        if methods is None:
            methods = ['synonym_replacement', 'random_swap']
        
        augmented = [text]  # 保留原文
        
        for method in methods:
            if method == 'synonym_replacement':
                aug_text = self._synonym_replacement(text)
                if aug_text != text:
                    augmented.append(aug_text)
            
            elif method == 'random_insertion':
                aug_text = self._random_insertion(text)
                if aug_text != text:
                    augmented.append(aug_text)
            
            elif method == 'random_swap':
                aug_text = self._random_swap(text)
                if aug_text != text:
                    augmented.append(aug_text)
            
            elif method == 'random_deletion':
                aug_text = self._random_deletion(text)
                if aug_text != text:
                    augmented.append(aug_text)
        
        return augmented
    
    def _synonym_replacement(self, text: str, n: int = 2) -> str:
        """同义词替换"""
        words = text.split()
        
        # 简单的同义词字典（实际应用中应使用WordNet等）
        synonyms = {
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'mini'],
            'happy': ['joyful', 'pleased', 'delighted'],
            'sad': ['unhappy', 'sorrowful', 'depressed'],
        }
        
        # 随机替换n个词
        replaceable_indices = [i for i, word in enumerate(words) if word.lower() in synonyms]
        
        if not replaceable_indices:
            return text
        
        replace_count = min(n, len(replaceable_indices))
        indices_to_replace = random.sample(replaceable_indices, replace_count)
        
        for idx in indices_to_replace:
            word = words[idx].lower()
            if word in synonyms:
                words[idx] = random.choice(synonyms[word])
        
        return ' '.join(words)
    
    def _random_insertion(self, text: str, n: int = 1) -> str:
        """随机插入"""
        words = text.split()
        
        for _ in range(n):
            # 随机选择一个词
            random_word = random.choice(words)
            # 随机插入位置
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def _random_swap(self, text: str, n: int = 2) -> str:
        """随机交换"""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """随机删除"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        # 确保至少保留一个词
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment_dataset(
        self,
        data: List[Dict[str, Any]],
        text_key: str = 'text',
        augmentation_factor: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        对整个数据集进行增强
        
        Args:
            data: 数据列表
            text_key: 文本字段的键名
            augmentation_factor: 增强因子 (0-1)，表示增强后数据量占原数据的比例
        """
        print(f"🔄 开始数据增强 (增强因子: {augmentation_factor})...")
        
        augmented_data = data.copy()
        num_to_augment = int(len(data) * augmentation_factor)
        
        # 随机选择要增强的样本
        samples_to_augment = random.sample(data, num_to_augment)
        
        for sample in samples_to_augment:
            text = sample[text_key]
            augmented_texts = self.augment_text(text)
            
            # 添加增强样本（除了原文）
            for aug_text in augmented_texts[1:]:
                new_sample = sample.copy()
                new_sample[text_key] = aug_text
                augmented_data.append(new_sample)
        
        self.stats['augmented_samples'] += len(augmented_data) - len(data)
        
        print(f"✅ 数据增强完成: {len(data)} -> {len(augmented_data)} (+{len(augmented_data) - len(data)})")
        
        return augmented_data
    
    # ==================== 数据平衡 ====================
    
    def balance_dataset(
        self,
        data: List[Dict[str, Any]],
        label_key: str = 'label',
        method: str = 'oversample'
    ) -> List[Dict[str, Any]]:
        """
        数据平衡
        
        Args:
            data: 数据列表
            label_key: 标签字段的键名
            method: 平衡方法 (oversample/undersample/smote)
        """
        print(f"⚖️ 开始数据平衡 (方法: {method})...")
        
        # 统计各类别样本数
        label_counts = Counter([item[label_key] for item in data])
        print(f"📊 原始分布: {dict(label_counts)}")
        
        if method == 'oversample':
            balanced_data = self._oversample(data, label_key, label_counts)
        elif method == 'undersample':
            balanced_data = self._undersample(data, label_key, label_counts)
        else:
            print(f"⚠️ 未知的平衡方法: {method}")
            return data
        
        # 统计平衡后的分布
        new_label_counts = Counter([item[label_key] for item in balanced_data])
        print(f"📊 平衡后分布: {dict(new_label_counts)}")
        
        return balanced_data
    
    def _oversample(
        self,
        data: List[Dict[str, Any]],
        label_key: str,
        label_counts: Counter
    ) -> List[Dict[str, Any]]:
        """过采样 - 复制少数类样本"""
        max_count = max(label_counts.values())
        
        balanced_data = data.copy()
        
        # 按标签分组
        grouped = {}
        for item in data:
            label = item[label_key]
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(item)
        
        # 对每个类别进行过采样
        for label, samples in grouped.items():
            current_count = len(samples)
            need_count = max_count - current_count
            
            if need_count > 0:
                # 随机复制样本
                oversampled = random.choices(samples, k=need_count)
                balanced_data.extend(oversampled)
        
        return balanced_data
    
    def _undersample(
        self,
        data: List[Dict[str, Any]],
        label_key: str,
        label_counts: Counter
    ) -> List[Dict[str, Any]]:
        """欠采样 - 删除多数类样本"""
        min_count = min(label_counts.values())
        
        # 按标签分组
        grouped = {}
        for item in data:
            label = item[label_key]
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(item)
        
        # 对每个类别进行欠采样
        balanced_data = []
        for label, samples in grouped.items():
            # 随机选择min_count个样本
            undersampled = random.sample(samples, min_count)
            balanced_data.extend(undersampled)
        
        return balanced_data
    
    # ==================== 特征工程 ====================
    
    def extract_features(
        self,
        text: str,
        include_stats: bool = True,
        include_ngrams: bool = True
    ) -> Dict[str, Any]:
        """
        提取文本特征
        
        Args:
            text: 输入文本
            include_stats: 是否包含统计特征
            include_ngrams: 是否包含n-gram特征
        """
        features = {}
        
        words = text.split()
        
        if include_stats:
            # 统计特征
            features['length'] = len(text)
            features['word_count'] = len(words)
            features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
            features['sentence_count'] = len(re.split(r'[.!?]+', text))
            features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
            
            # 标点符号统计
            features['punctuation_count'] = sum(1 for c in text if c in ',.!?;:')
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        if include_ngrams:
            # n-gram特征
            features['bigrams'] = self._extract_ngrams(words, 2)
            features['trigrams'] = self._extract_ngrams(words, 3)
        
        return features
    
    def _extract_ngrams(self, words: List[str], n: int) -> List[str]:
        """提取n-gram"""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def add_features_to_dataset(
        self,
        data: List[Dict[str, Any]],
        text_key: str = 'text'
    ) -> List[Dict[str, Any]]:
        """为数据集添加特征"""
        print("🔧 提取特征...")
        
        enhanced_data = []
        for item in data:
            new_item = item.copy()
            features = self.extract_features(item[text_key])
            new_item['features'] = features
            enhanced_data.append(new_item)
        
        print(f"✅ 特征提取完成")
        return enhanced_data
    
    # ==================== 数据质量检查 ====================
    
    def check_quality(
        self,
        data: List[Dict[str, Any]],
        text_key: str = 'text',
        min_length: int = 10,
        max_length: int = 10000
    ) -> Dict[str, Any]:
        """
        数据质量检查
        
        检查项:
        - 空文本
        - 过短/过长文本
        - 重复文本
        - 异常字符
        """
        print("🔍 开始数据质量检查...")
        
        issues = {
            'empty': [],
            'too_short': [],
            'too_long': [],
            'duplicates': [],
            'unusual_chars': []
        }
        
        seen_texts = {}
        
        for idx, item in enumerate(data):
            text = item.get(text_key, '')
            
            # 检查空文本
            if not text or not text.strip():
                issues['empty'].append(idx)
                continue
            
            # 检查长度
            if len(text) < min_length:
                issues['too_short'].append(idx)
            elif len(text) > max_length:
                issues['too_long'].append(idx)
            
            # 检查重复
            text_normalized = self.normalize_text(text, lowercase=True)
            if text_normalized in seen_texts:
                issues['duplicates'].append((idx, seen_texts[text_normalized]))
            else:
                seen_texts[text_normalized] = idx
            
            # 检查异常字符
            if self._has_unusual_chars(text):
                issues['unusual_chars'].append(idx)
        
        # 打印报告
        print("\n📋 质量检查报告:")
        print(f"  总样本数: {len(data)}")
        print(f"  空文本: {len(issues['empty'])}")
        print(f"  过短文本 (<{min_length}字符): {len(issues['too_short'])}")
        print(f"  过长文本 (>{max_length}字符): {len(issues['too_long'])}")
        print(f"  重复文本: {len(issues['duplicates'])}")
        print(f"  异常字符: {len(issues['unusual_chars'])}")
        
        return issues
    
    def _has_unusual_chars(self, text: str) -> bool:
        """检查是否包含异常字符"""
        # 检查是否包含过多的非ASCII字符
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
        
        # 如果非ASCII字符超过80%，可能是异常
        if non_ascii_ratio > 0.8:
            return True
        
        # 检查是否包含控制字符
        control_chars = [c for c in text if ord(c) < 32 and c not in '\n\r\t']
        if control_chars:
            return True
        
        return False
    
    def filter_by_quality(
        self,
        data: List[Dict[str, Any]],
        issues: Dict[str, Any],
        remove_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """根据质量问题过滤数据"""
        if remove_types is None:
            remove_types = ['empty', 'too_short', 'unusual_chars']
        
        print(f"🧹 根据质量问题过滤数据 (移除类型: {remove_types})...")
        
        # 收集要移除的索引
        indices_to_remove = set()
        for issue_type in remove_types:
            if issue_type in issues:
                if issue_type == 'duplicates':
                    # 对于重复，只移除后出现的那个
                    indices_to_remove.update([dup[0] for dup in issues[issue_type]])
                else:
                    indices_to_remove.update(issues[issue_type])
        
        # 过滤数据
        filtered_data = [item for idx, item in enumerate(data) if idx not in indices_to_remove]
        
        removed = len(data) - len(filtered_data)
        self.stats['filtered_samples'] += removed
        
        print(f"✅ 过滤完成: {len(data)} -> {len(filtered_data)} (移除 {removed} 条)")
        
        return filtered_data
    
    # ==================== 批处理管道 ====================
    
    def process_pipeline(
        self,
        data: List[Dict[str, Any]],
        text_key: str = 'text',
        label_key: str = 'label',
        steps: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        数据处理管道
        
        Args:
            data: 原始数据
            text_key: 文本字段键名
            label_key: 标签字段键名
            steps: 处理步骤列表
        """
        if steps is None:
            steps = ['clean', 'quality_check', 'augment', 'balance']
        
        print("=" * 60)
        print("🔄 数据处理管道启动")
        print(f"📊 初始样本数: {len(data)}")
        print(f"🛠️ 处理步骤: {' -> '.join(steps)}")
        print("=" * 60)
        
        processed_data = data
        
        for step in steps:
            print(f"\n▶️ 执行步骤: {step}")
            
            if step == 'clean':
                # 清洗文本
                for item in processed_data:
                    item[text_key] = self.clean_text(item[text_key])
                self.stats['cleaned_samples'] = len(processed_data)
            
            elif step == 'quality_check':
                # 质量检查并过滤
                issues = self.check_quality(processed_data, text_key)
                processed_data = self.filter_by_quality(processed_data, issues)
            
            elif step == 'remove_duplicates':
                # 去重
                texts = [item[text_key] for item in processed_data]
                unique_texts = self.remove_duplicates(texts)
                processed_data = [item for item in processed_data if item[text_key] in unique_texts]
            
            elif step == 'augment':
                # 数据增强
                processed_data = self.augment_dataset(
                    processed_data,
                    text_key,
                    augmentation_factor=self.augmentation_ratio
                )
            
            elif step == 'balance':
                # 数据平衡
                if label_key in processed_data[0]:
                    processed_data = self.balance_dataset(
                        processed_data,
                        label_key,
                        method='oversample'
                    )
            
            elif step == 'extract_features':
                # 特征提取
                processed_data = self.add_features_to_dataset(processed_data, text_key)
        
        self.stats['processed_samples'] = len(processed_data)
        
        print("\n" + "=" * 60)
        print("✅ 数据处理管道完成")
        print(f"📊 最终样本数: {len(processed_data)}")
        print(f"📈 处理统计: {self.stats}")
        print("=" * 60)
        
        return processed_data
    
    # ==================== 插件钩子 ====================
    
    def on_data_load(self, context: Dict[str, Any]):
        """数据加载时的钩子"""
        data = context.get('data', [])
        
        if self.config.get('auto_process', False):
            print("🔄 自动启动数据处理...")
            processed_data = self.process_pipeline(data)
            context['data'] = processed_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🛠️ 数据处理器插件 (Data Processors Plugin)")
    print("=" * 60)
    
    # 配置
    config = {
        'enable_cleaning': True,
        'enable_augmentation': True,
        'augmentation_ratio': 0.3,
        'normalize_numbers': False,
        'normalize_urls': True,
        'auto_process': False
    }
    
    plugin = DataProcessorsPlugin(config)
    
    # 示例数据
    sample_data = [
        {'text': 'This is a  good   example.', 'label': 0},
        {'text': 'Another great sample here!', 'label': 1},
        {'text': 'Bad  quality text...', 'label': 0},
        {'text': 'This is a good example.', 'label': 0},  # 重复
        {'text': 'x', 'label': 1},  # 太短
    ]
    
    print("\n📝 示例数据:")
    for i, item in enumerate(sample_data):
        print(f"  {i+1}. {item}")
    
    # 运行处理管道
    processed = plugin.process_pipeline(
        sample_data,
        steps=['clean', 'quality_check', 'remove_duplicates', 'augment', 'balance']
    )
    
    print("\n📝 处理后数据样本:")
    for i, item in enumerate(processed[:5]):
        print(f"  {i+1}. {item['text'][:50]}... (label: {item.get('label', 'N/A')})")
    
    print("\n💡 插件功能:")
    print("1. 🧹 文本清洗和标准化")
    print("2. 🔄 多种数据增强方法")
    print("3. ⚖️ 数据平衡 (过采样/欠采样)")
    print("4. 🔧 自动特征提取")
    print("5. 🔍 数据质量检查")
    print("6. 🔗 完整的处理管道")
