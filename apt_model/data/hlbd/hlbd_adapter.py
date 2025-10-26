#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HLBD数据集适配模块
用于将分层语言启蒙数据集(Hierarchical Language Bootstrapping Dataset)集成到APT模型框架

此模块提供了数据加载、处理、分词器配置等功能，以便在APT模型中使用HLBD数据进行训练
"""

import os
import json
import logging
import random
import re
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader

# 设置日志记录器
logger = logging.getLogger("apt_model.hlbd")

class HLBDDataProcessor:
    """
    HLBD数据处理器
    用于加载和转换HLBD数据集为APT模型训练所需的格式
    支持扩展自定义语言层级
    """
    
    def __init__(self, data_path: Optional[str] = None, samples: Optional[List[Dict]] = None, 
                extra_languages: Optional[Dict[str, str]] = None):
        """
        初始化HLBD数据处理器
        
        参数:
            data_path: HLBD数据文件路径（可选）
            samples: 已加载的HLBD样本列表（可选）
            extra_languages: 额外的语言层级配置，格式为 {层级名称: 语言代码}
                例如: {"level_9": "法语", "level_10": "德语"}
            
        注意:
            data_path和samples至少需要提供一个
        """
        self.data_path = data_path
        self.raw_samples = samples
        self.processed_texts = []
        
        # 基础层级定义
        self.level_names = {
            "level_1": "字卡",
            "level_2": "短语",
            "level_3": "数学",
            "level_4": "拼音",
            "level_5": "英文",
            "level_6": "中文",
            "level_7": "日文",
            "level_8": "韩文"
        }
        
        # 扩展额外的语言层级
        if extra_languages:
            self.level_names.update(extra_languages)
        
        # 初始化level_texts字典
        self.level_texts = {level: [] for level in self.level_names.keys()}
        
        # 如果提供了数据路径，立即加载数据
        if data_path and not samples:
            self.raw_samples = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """
        从文件加载HLBD数据
        
        参数:
            data_path: 数据文件路径
            
        返回:
            解析后的HLBD样本列表
        """
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 提取samples部分并解析JSON
            samples_match = re.search(r'samples\s*=\s*(\[.*?\])(?=\s*$)', content, re.DOTALL)
            if samples_match:
                samples_text = samples_match.group(1)
                samples = json.loads(samples_text)
                logger.info(f"成功从{data_path}加载了{len(samples)}个HLBD样本")
                return samples
            else:
                # 尝试将整个文件作为JSON加载
                try:
                    samples = json.loads(content)
                    if isinstance(samples, list):
                        logger.info(f"成功从{data_path}加载了{len(samples)}个HLBD样本")
                        return samples
                except json.JSONDecodeError:
                    pass
                
                logger.error(f"在{data_path}中未找到有效的HLBD样本")
                return []
        except Exception as e:
            logger.error(f"加载HLBD数据时出错: {e}\n{traceback.format_exc()}")
            return []
    
    def _extract_nested_content(self, data, key):
        """递归提取嵌套字典中的内容"""
        if isinstance(data, dict):
            if key in data:
                return data[key]
            for k, v in data.items():
                result = self._extract_nested_content(v, key)
                if result is not None:
                    return result
        return None
    
    def process_data(self, include_multilingual: bool = True, include_separate_levels: bool = True) -> List[str]:
        """
        处理HLBD数据为训练文本
        
        参数:
            include_multilingual: 是否包含多语言文本
            include_separate_levels: 是否将每个层级作为单独的训练样本
            
        返回:
            处理后的训练文本列表
        """
        if not self.raw_samples:
            logger.error("没有可用的HLBD样本")
            return []
        
        processed_texts = []
        level_texts = {level: [] for level in self.level_names.keys()}
        
        for sample in self.raw_samples:
            # 创建完整的分层文本
            layered_text = []
            concept = sample.get("concept", "")
            layered_text.append(f"【概念】{concept}")
            
            # 处理各个层级
            for level_key, level_name in self.level_names.items():
                if level_key in sample:
                    level_data = sample[level_key]
                    
                    # 自动处理标准层级
                    if level_key == "level_1":  # 字卡层
                        char_card = level_data.get("字卡", "")
                        emoji = level_data.get("emoji", "")
                        layer_text = f"【字卡】{char_card} {emoji}"
                        layered_text.append(layer_text)
                        level_texts[level_key].append(f"概念: {concept} -> 字卡: {char_card} {emoji}")
                    
                    elif level_key == "level_2":  # 短语层
                        phrase = level_data.get("短语", "")
                        layer_text = f"【短语】{phrase}"
                        layered_text.append(layer_text)
                        level_texts[level_key].append(f"概念: {concept} -> 短语: {phrase}")
                    
                    elif level_key == "level_3":  # 数学层
                        math_expr = level_data.get("数学", "")
                        layer_text = f"【句法结构】{math_expr}"
                        layered_text.append(layer_text)
                        level_texts[level_key].append(f"概念: {concept} -> 句法结构: {math_expr}")
                    
                    elif level_key == "level_4":  # 拼音层
                        pinyin = level_data.get("拼音", "")
                        layer_text = f"【拼音】{pinyin}"
                        layered_text.append(layer_text)
                        level_texts[level_key].append(f"概念: {concept} -> 拼音: {pinyin}")
                    
                    elif level_key == "level_5":  # 英文层
                        english = level_data.get("英文", "")
                        layer_text = f"【英文】{english}"
                        layered_text.append(layer_text)
                        level_texts[level_key].append(f"概念: {concept} -> 英文: {english}")
                    
                    elif level_key == "level_6":  # 中文层
                        chinese = level_data.get("中文", "")
                        layer_text = f"【中文】{chinese}"
                        layered_text.append(layer_text)
                        level_texts[level_key].append(f"概念: {concept} -> 中文: {chinese}")
                    
                    # 标准多语言层级
                    elif level_key == "level_7" and include_multilingual:  # 日文层
                        japanese = level_data.get("日文", "")
                        if japanese:
                            layer_text = f"【日文】{japanese}"
                            layered_text.append(layer_text)
                            level_texts[level_key].append(f"概念: {concept} -> 日文: {japanese}")
                    
                    elif level_key == "level_8" and include_multilingual:  # 韩文层
                        korean_text = self._extract_nested_content(level_data, "韩文")
                        if korean_text:
                            layer_text = f"【韩文】{korean_text}"
                            layered_text.append(layer_text)
                            level_texts[level_key].append(f"概念: {concept} -> 韩文: {korean_text}")
                    
                    # 处理扩展语言层级 (如法语、德语等)
                    elif include_multilingual and int(level_key.split("_")[1]) > 8:
                        # 尝试自动处理扩展层级
                        lang_name = self.level_names[level_key]
                        
                        # 处理不同的数据格式
                        if isinstance(level_data, dict) and lang_name in level_data:
                            content = level_data.get(lang_name, "")
                        elif isinstance(level_data, dict):
                            # 尝试获取第一个值
                            content = next(iter(level_data.values()), "")
                        elif isinstance(level_data, str):
                            content = level_data
                        else:
                            content = str(level_data)
                        
                        if content:
                            layer_text = f"【{lang_name}】{content}"
                            layered_text.append(layer_text)
                            level_texts[level_key].append(f"概念: {concept} -> {lang_name}: {content}")
            
            # 组合所有层级，创建高信息密度的训练文本
            full_text = "\n".join(layered_text)
            processed_texts.append(full_text)
            
            # 创建各种组合样本，增强训练效果
            if "level_5" in sample and "level_6" in sample:
                # 英中翻译对
                en = sample["level_5"].get("英文", "")
                zh = sample["level_6"].get("中文", "")
                if en and zh:
                    processed_texts.append(f"英文: {en} -> 中文: {zh}")
                    processed_texts.append(f"中文: {zh} -> 英文: {en}")
            
            if "level_4" in sample and "level_6" in sample:
                # 拼音-中文对
                py = sample["level_4"].get("拼音", "")
                zh = sample["level_6"].get("中文", "")
                if py and zh:
                    processed_texts.append(f"拼音: {py} -> 中文: {zh}")
            
            # 添加概念到各层的映射
            if "level_6" in sample:
                zh = sample["level_6"].get("中文", "")
                processed_texts.append(f"概念: {concept} -> {zh}")
        
        # 添加单独的层级文本
        if include_separate_levels:
            for level, texts in level_texts.items():
                processed_texts.extend(texts)
        
        self.processed_texts = processed_texts
        self.level_texts = level_texts
        
        logger.info(f"处理了{len(self.raw_samples)}个HLBD样本，生成了{len(processed_texts)}条训练文本")
        
        # 添加内存清理
        import gc
        gc.collect()
        
        return processed_texts
    
    def get_training_texts(self) -> List[str]:
        """获取处理后的训练文本"""
        if not self.processed_texts:
            self.process_data()
        return self.processed_texts
    
    def get_level_texts(self, level: int) -> List[str]:
        """获取特定层级的文本"""
        level_key = f"level_{level}"
        if not self.level_texts[level_key]:
            self.process_data()
        return self.level_texts[level_key]
    
    def get_multilingual_texts(self) -> List[str]:
        """获取多语言文本，用于训练分词器"""
        multilingual_texts = []
        
        for sample in self.raw_samples:
            # 添加标准语言层的文本
            for level_key, level_name in self.level_names.items():
                if level_key in sample:
                    if level_key == "level_6":  # 中文
                        chinese_text = self._extract_nested_content(sample[level_key], "中文")
                        if chinese_text:
                            multilingual_texts.append(chinese_text)
                    elif level_key == "level_5":  # 英文
                        english_text = self._extract_nested_content(sample[level_key], "英文")
                        if english_text:
                            multilingual_texts.append(english_text)
                    elif level_key == "level_4":  # 拼音
                        pinyin_text = self._extract_nested_content(sample[level_key], "拼音")
                        if pinyin_text:
                            multilingual_texts.append(pinyin_text)
                    elif level_key == "level_7":  # 日文
                        japanese_text = self._extract_nested_content(sample[level_key], "日文")
                        if japanese_text:
                            multilingual_texts.append(japanese_text)
                    elif level_key == "level_8":  # 韩文
                        korean_text = self._extract_nested_content(sample[level_key], "韩文")
                        if korean_text:
                            multilingual_texts.append(korean_text)
                    # 处理扩展语言层级 (如法语、德语等)
                    elif int(level_key.split("_")[1]) > 8:
                        lang_name = self.level_names[level_key]
                        level_data = sample[level_key]
                        lang_text = self._extract_nested_content(level_data, lang_name)
                        if lang_text:
                            multilingual_texts.append(lang_text)
        
        return multilingual_texts
    
    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                  test_ratio: float = 0.1, seed: int = 42) -> Dict[str, List[str]]:
        """
        将处理后的数据分割为训练、验证和测试集
        
        参数:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            
        返回:
            包含'train', 'val', 'test'的字典，每个键对应相应的文本列表
        """
        if not self.processed_texts:
            self.process_data()
        
        random.seed(seed)
        texts = self.processed_texts.copy()
        random.shuffle(texts)
        
        total = len(texts)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size + val_size]
        test_texts = texts[train_size + val_size:]
        
        return {
            'train': train_texts,
            'val': val_texts,
            'test': test_texts
        }


class HLBDDataset(Dataset):
    """
    HLBD数据集类
    为APT模型提供HLBD数据的PyTorch数据集
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """
        初始化HLBD数据集
        
        参数:
            texts: 训练文本列表
            tokenizer: 用于编码文本的分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 获取input_ids和attention_mask
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 对于自回归语言模型，标签与输入相同
        labels = input_ids.clone()
        
        # 将padding位置的标签设为-100，在计算损失时忽略这些位置
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def prepare_hlbd_tokenizer(hlbd_samples_or_path, vocab_size: int = 50000, extra_languages: Optional[Dict[str, str]] = None):
    """
    为HLBD数据准备适合的多语言分词器
    
    参数:
        hlbd_samples_or_path: HLBD样本列表或数据文件路径
        vocab_size: 词汇表大小
        extra_languages: 额外的语言层级配置，格式为 {层级名称: 语言代码}
        
    返回:
        tokenizer: 训练好的分词器
        detected_language: 检测到的语言
    """
    # 处理输入
    if isinstance(hlbd_samples_or_path, str):
        processor = HLBDDataProcessor(data_path=hlbd_samples_or_path, extra_languages=extra_languages)
        hlbd_samples = processor.raw_samples
    else:
        hlbd_samples = hlbd_samples_or_path
        processor = HLBDDataProcessor(samples=hlbd_samples, extra_languages=extra_languages)
    
    # 获取多语言文本
    multilingual_texts = processor.get_multilingual_texts()
    
    # 使用已实现的SentencePiece集成
    try:
        from apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer
        
        logger.info("使用SentencePiece创建多语言分词器")
        
        # 根据是否有扩展语言决定词汇表大小
        if extra_languages:
            # 增加词汇表大小以容纳额外语言
            adjusted_vocab_size = min(vocab_size + 5000 * len(extra_languages), 100000)
            logger.info(f"检测到扩展语言，增加词汇表大小至 {adjusted_vocab_size}")
            vocab_size = adjusted_vocab_size
        
        tokenizer, detected_language = get_appropriate_tokenizer(
            multilingual_texts, 
            tokenizer_type=None,  # 让函数自动选择最佳类型
            language="multilingual",  # 明确指定多语言
            vocab_size=vocab_size  # 自定义词汇表大小
        )
        
        return tokenizer, detected_language
    
    except ImportError:
        logger.warning("未找到中文分词器集成模块，尝试使用transformers")
        try:
            from transformers import AutoTokenizer
            
            # 选择合适的多语言模型
            if extra_languages:
                # 如果有扩展语言，使用大型多语言模型
                logger.info("使用xlm-roberta-base分词器以支持扩展语言")
                tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            else:
                logger.info("使用bert-base-multilingual-cased分词器")
                tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            
            # 确保有填充标记
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer, "multilingual"
        
        except ImportError:
            logger.error("未安装transformers库，无法创建分词器")
            raise


def create_hlbd_apt_config(vocab_size: int = 50000):
    """
    创建适合HLBD的APT模型配置
    
    参数:
        vocab_size: 词汇表大小
        
    返回:
        APTConfig: 适合HLBD数据的模型配置
    """
    try:
        from apt_model.config.apt_config import APTConfig
        
        # 使用更适合多语言处理的配置
        config = APTConfig(
            vocab_size=vocab_size,  # 扩大词汇表以容纳多语言
            d_model=1024,       # 增加模型维度以处理复杂语言关系
            d_ff=4096,          # 增加前馈网络维度
            num_heads=16,       # 增加注意力头数
            num_encoder_layers=12,
            num_decoder_layers=12,
            max_seq_len=1024,   # 增加序列长度以容纳层次结构
            dropout=0.1,
            attention_dropout=0.1,
            use_autopoietic=True,   # 启用自生成机制
            language="multilingual",
            tokenizer_type=None     # 让系统自动选择
        )
        
        return config
    
    except ImportError:
        logger.error("未找到APT配置模块")
        raise


def prepare_hlbd_datasets(processor, tokenizer, max_length: int = 512, batch_size: int = 8):
    """
    准备HLBD数据集的训练、验证和测试集
    
    参数:
        processor: HLBD数据处理器
        tokenizer: 分词器
        max_length: 最大序列长度
        batch_size: 批量大小
        
    返回:
        包含训练、验证和测试数据加载器的字典
    """
    # 分割数据
    split_data = processor.split_data()
    
    # 创建数据集
    train_dataset = HLBDDataset(split_data['train'], tokenizer, max_length)
    val_dataset = HLBDDataset(split_data['val'], tokenizer, max_length)
    test_dataset = HLBDDataset(split_data['test'], tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


class HLBDModelEvaluator:
    """HLBD模型评估器，用于评估模型在HLBD数据上的表现"""
    
    def __init__(self, model, tokenizer, processor):
        """
        初始化评估器
        
        参数:
            model: 要评估的模型
            tokenizer: 分词器
            processor: HLBD数据处理器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        
        # 语言到层级的映射 (标准语言)
        self.lang_to_level = {
            "中文": 6,
            "英文": 5,
            "拼音": 4,
            "日文": 7,
            "韩文": 8
        }
        
        # 添加扩展语言映射
        for level_key, lang_name in processor.level_names.items():
            if level_key.startswith("level_") and int(level_key.split("_")[1]) > 8:
                level_num = int(level_key.split("_")[1])
                self.lang_to_level[lang_name] = level_num
    
    def _extract_nested_language_content(self, level_data, lang_name):
        """递归提取嵌套字典中的语言内容"""
        if isinstance(level_data, dict):
            if lang_name in level_data:
                return level_data[lang_name]
            # 递归搜索每个子字典
            for k, v in level_data.items():
                result = self._extract_nested_language_content(v, lang_name)
                if result is not None:
                    return result
        return level_data if isinstance(level_data, str) else None
    
    def evaluate_level_generation(self, level_from: int, level_to: int, num_samples: int = 5):
        """
        评估模型在不同层级间生成的能力
        
        参数:
            level_from: 源层级
            level_to: 目标层级
            num_samples: 评估样本数
            
        返回:
            评估结果字典
        """
        # 获取源层级和目标层级的文本样本
        from_texts = self.processor.get_level_texts(level_from)
        to_texts = self.processor.get_level_texts(level_to)
        
        # 确保样本数不超过可用样本
        num_samples = min(num_samples, len(from_texts), len(to_texts))
        
        if num_samples == 0:
            logger.warning(f"级别 {level_from} 到 {level_to} 没有可用的配对样本")
            return {
                "level_from": level_from,
                "level_to": level_to,
                "num_samples": 0,
                "avg_similarity": 0.0,
                "samples": []
            }
        
        results = []
        for i in range(num_samples):
            from_text = from_texts[i]
            to_text = to_texts[i]
            
            # 创建提示文本
            prompt = from_text + " -> "
            
            # 生成文本
            generated_text = self._generate(prompt)
            
            # 计算相似度
            similarity = self._calculate_similarity(generated_text, to_text)
            
            results.append({
                "from_text": from_text,
                "to_text": to_text,
                "generated_text": generated_text,
                "similarity": similarity
            })
        
        # 计算平均相似度
        avg_similarity = sum(item["similarity"] for item in results) / len(results)
        
        return {
            "level_from": level_from,
            "level_to": level_to,
            "num_samples": num_samples,
            "avg_similarity": avg_similarity,
            "samples": results
        }
    
    def evaluate_language_generation(self, source_lang: str, target_lang: str, num_samples: int = 5):
        """
        评估模型的跨语言生成能力
        
        参数:
            source_lang: 源语言 ("中文"、"英文"、"日文"、"韩文"等)
            target_lang: 目标语言
            num_samples: 评估样本数
            
        返回:
            评估结果字典
        """
        # 获取源语言和目标语言对应的层级
        level_from = self.lang_to_level.get(source_lang)
        level_to = self.lang_to_level.get(target_lang)
        
        if not level_from or not level_to:
            logger.error(f"不支持的语言: {source_lang} 或 {target_lang}")
            return {"error": f"不支持的语言: {source_lang} 或 {target_lang}"}
        
        # 使用层级评估函数
        return self.evaluate_level_generation(level_from, level_to, num_samples)
    
    def evaluate_all_language_pairs(self, num_samples: int = 3):
        """
        评估所有语言对之间的翻译能力
        
        参数:
            num_samples: 每对语言评估的样本数
            
        返回:
            评估结果字典
        """
        language_pairs_results = {}
        languages = list(self.lang_to_level.keys())
        
        logger.info(f"评估所有语言对: {languages}")
        
        for source_lang in languages:
            for target_lang in languages:
                if source_lang != target_lang:
                    pair_key = f"{source_lang}_to_{target_lang}"
                    logger.info(f"评估语言对: {pair_key}")
                    
                    result = self.evaluate_language_generation(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        num_samples=num_samples
                    )
                    
                    language_pairs_results[pair_key] = result
        
        # 计算总体平均相似度
        valid_results = [r["avg_similarity"] for r in language_pairs_results.values() 
                        if "avg_similarity" in r]
        overall_avg = sum(valid_results) / len(valid_results) if valid_results else 0.0
        
        return {
            "overall_avg_similarity": overall_avg,
            "language_pairs": language_pairs_results
        }
    
    def evaluate_concept_completion(self, num_samples: int = 5):
        """
        评估模型从概念生成完整描述的能力
        
        参数:
            num_samples: 评估样本数
            
        返回:
            评估结果