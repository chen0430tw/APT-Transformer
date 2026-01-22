#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理管道 - 统一的数据加载和处理接口

整合功能：
- 数据加载（文件、HuggingFace）
- 数据预处理
- 数据清洗
- 批处理

提供统一的API，简化数据处理流程
"""

import os
import logging
from typing import List, Dict, Optional, Union, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# 数据加载器类
# ============================================================================

class DataLoader:
    """
    统一的数据加载器

    支持多种数据源：
    - 本地文件（txt, csv, json, jsonl, excel）
    - HuggingFace数据集
    - 内置数据
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化数据加载器

        参数:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_from_file(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        encoding: str = 'utf-8'
    ) -> List[str]:
        """
        从文件加载数据

        参数:
            file_path: 文件路径
            max_samples: 最大样本数
            encoding: 文件编码

        返回:
            list: 文本列表
        """
        from apt.core.data.external_data import load_external_data

        try:
            texts = load_external_data(file_path, max_samples)
            self.logger.info(f"从文件加载了 {len(texts)} 条文本")
            return texts
        except Exception as e:
            self.logger.error(f"加载文件失败: {e}")
            raise

    def load_from_huggingface(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: str = "train",
        text_column: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> List[str]:
        """
        从HuggingFace加载数据

        参数:
            dataset_name: 数据集名称
            config_name: 配置名称
            split: 数据集分割
            text_column: 文本列名
            max_samples: 最大样本数

        返回:
            list: 文本列表
        """
        from apt.core.data.huggingface_loader import HuggingFaceLoader

        loader = HuggingFaceLoader(logger=self.logger)
        texts, info = loader.load_dataset(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            text_column=text_column,
            max_samples=max_samples
        )

        self.logger.info(f"从HuggingFace加载了 {len(texts)} 条文本")
        return texts

    def load_builtin(self) -> List[str]:
        """
        加载内置训练数据

        返回:
            list: 文本列表
        """
        from apt.core.data.data_processor import get_training_texts

        texts = get_training_texts()
        self.logger.info(f"加载了 {len(texts)} 条内置文本")
        return texts

    def load(
        self,
        source: Union[str, None] = None,
        source_type: str = "auto",
        **kwargs
    ) -> List[str]:
        """
        智能加载数据（自动识别数据源类型）

        参数:
            source: 数据源（文件路径、数据集名称或None表示内置）
            source_type: 数据源类型（auto, file, huggingface, builtin）
            **kwargs: 额外参数

        返回:
            list: 文本列表
        """
        # 如果没有指定source，使用内置数据
        if source is None:
            return self.load_builtin()

        # 自动识别类型
        if source_type == "auto":
            if os.path.exists(source):
                source_type = "file"
            else:
                source_type = "huggingface"

        # 根据类型加载
        if source_type == "file":
            return self.load_from_file(source, **kwargs)
        elif source_type == "huggingface":
            return self.load_from_huggingface(source, **kwargs)
        elif source_type == "builtin":
            return self.load_builtin()
        else:
            raise ValueError(f"未知的数据源类型: {source_type}")


# ============================================================================
# 数据处理器类
# ============================================================================

class DataProcessor:
    """
    数据处理器

    提供文本清洗、预处理等功能
    """

    def __init__(
        self,
        tokenizer=None,
        max_length: int = 512,
        clean_text: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化数据处理器

        参数:
            tokenizer: 分词器
            max_length: 最大序列长度
            clean_text: 是否清洗文本
            logger: 日志记录器
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clean_text = clean_text
        self.logger = logger or logging.getLogger(__name__)

    def clean(self, text: str) -> str:
        """
        清洗单条文本

        参数:
            text: 输入文本

        返回:
            str: 清洗后的文本
        """
        if not self.clean_text:
            return text

        # 基础清洗
        text = text.strip()

        # 移除多余的空白字符
        import re
        text = re.sub(r'\s+', ' ', text)

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        批量清洗文本

        参数:
            texts: 文本列表

        返回:
            list: 清洗后的文本列表
        """
        return [self.clean(text) for text in texts if text]

    def filter_by_length(
        self,
        texts: List[str],
        min_length: int = 10,
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        按长度过滤文本

        参数:
            texts: 文本列表
            min_length: 最小长度
            max_length: 最大长度

        返回:
            list: 过滤后的文本列表
        """
        filtered = []
        for text in texts:
            text_len = len(text)
            if text_len >= min_length:
                if max_length is None or text_len <= max_length:
                    filtered.append(text)

        self.logger.info(f"按长度过滤: {len(texts)} -> {len(filtered)} 条文本")
        return filtered

    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """
        移除重复文本

        参数:
            texts: 文本列表

        返回:
            list: 去重后的文本列表
        """
        original_count = len(texts)
        unique_texts = list(dict.fromkeys(texts))  # 保持顺序的去重

        self.logger.info(f"去重: {original_count} -> {len(unique_texts)} 条文本")
        return unique_texts

    def process(
        self,
        texts: List[str],
        clean: bool = True,
        remove_duplicates: bool = True,
        filter_length: bool = True,
        min_length: int = 10,
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        完整的数据处理流程

        参数:
            texts: 输入文本列表
            clean: 是否清洗
            remove_duplicates: 是否去重
            filter_length: 是否按长度过滤
            min_length: 最小长度
            max_length: 最大长度

        返回:
            list: 处理后的文本列表
        """
        self.logger.info(f"开始处理 {len(texts)} 条文本")

        # 清洗
        if clean:
            texts = self.clean_batch(texts)

        # 去重
        if remove_duplicates:
            texts = self.remove_duplicates(texts)

        # 按长度过滤
        if filter_length:
            texts = self.filter_by_length(texts, min_length, max_length)

        self.logger.info(f"处理完成，得到 {len(texts)} 条文本")
        return texts


# ============================================================================
# 数据管道类
# ============================================================================

class DataPipeline:
    """
    完整的数据处理管道

    集成加载、处理、验证等功能
    """

    def __init__(
        self,
        tokenizer=None,
        max_length: int = 512,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化数据管道

        参数:
            tokenizer: 分词器
            max_length: 最大序列长度
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.loader = DataLoader(logger=self.logger)
        self.processor = DataProcessor(
            tokenizer=tokenizer,
            max_length=max_length,
            logger=self.logger
        )

    def load_and_process(
        self,
        source: Union[str, None] = None,
        source_type: str = "auto",
        clean: bool = True,
        remove_duplicates: bool = True,
        filter_length: bool = True,
        min_length: int = 10,
        max_length: Optional[int] = None,
        max_samples: Optional[int] = None,
        **loader_kwargs
    ) -> List[str]:
        """
        加载并处理数据（一站式）

        参数:
            source: 数据源
            source_type: 数据源类型
            clean: 是否清洗
            remove_duplicates: 是否去重
            filter_length: 是否按长度过滤
            min_length: 最小长度
            max_length: 最大长度（处理时）
            max_samples: 最大样本数（加载时）
            **loader_kwargs: 加载器额外参数

        返回:
            list: 处理后的文本列表
        """
        # 加载数据
        texts = self.loader.load(
            source=source,
            source_type=source_type,
            max_samples=max_samples,
            **loader_kwargs
        )

        # 处理数据
        texts = self.processor.process(
            texts=texts,
            clean=clean,
            remove_duplicates=remove_duplicates,
            filter_length=filter_length,
            min_length=min_length,
            max_length=max_length
        )

        return texts

    def get_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        获取数据集统计信息

        参数:
            texts: 文本列表

        返回:
            dict: 统计信息
        """
        if not texts:
            return {
                'count': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
            }

        lengths = [len(text) for text in texts]

        return {
            'count': len(texts),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_chars': sum(lengths),
        }


# ============================================================================
# 便捷函数
# ============================================================================

def quick_load(
    source: Union[str, None] = None,
    max_samples: Optional[int] = None,
    clean: bool = True,
    **kwargs
) -> List[str]:
    """
    快速加载和处理数据（便捷函数）

    参数:
        source: 数据源（文件路径、数据集名称或None）
        max_samples: 最大样本数
        clean: 是否清洗
        **kwargs: 额外参数

    返回:
        list: 文本列表
    """
    pipeline = DataPipeline()
    return pipeline.load_and_process(
        source=source,
        max_samples=max_samples,
        clean=clean,
        **kwargs
    )


# ============================================================================
# 公共API
# ============================================================================

__all__ = [
    # 类
    'DataLoader',
    'DataProcessor',
    'DataPipeline',
    # 便捷函数
    'quick_load',
]
