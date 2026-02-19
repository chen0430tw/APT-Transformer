#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流式数据混合器 - 同时流式读取本地 .md 文件与 HuggingFace 数据集并按权重交织训练

用法示例:
    from apt.core.data.streaming_mixer import create_mixed_iterable

    # 返回 PyTorch IterableDataset，可直接传给 DataLoader
    dataset = create_mixed_iterable(
        md_dir="./docs",
        hf_dataset="wikitext",
        hf_config="wikitext-103-v1",
        hf_split="train",
        hf_text_column="text",
        tokenizer=tokenizer,
        max_length=256,
        md_weight=0.3,       # .md 占 30%
        hf_weight=0.7,       # HF 占 70%
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=16)
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Generator, List, Iterator

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

try:
    from datasets import load_dataset, IterableDataset, interleave_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# Markdown 工具
# ============================================================================

# 匹配常见 Markdown 语法，不依赖外部库
_MD_PATTERNS = [
    re.compile(r'```[\s\S]*?```'),          # 代码块
    re.compile(r'`[^`]+`'),                 # 行内代码
    re.compile(r'!\[.*?\]\(.*?\)'),         # 图片
    re.compile(r'\[([^\]]+)\]\([^\)]+\)'),  # 链接 → 保留链接文字
    re.compile(r'^#{1,6}\s+', re.M),        # 标题标记
    re.compile(r'^\s*[-*+]\s+', re.M),      # 无序列表标记
    re.compile(r'^\s*\d+\.\s+', re.M),      # 有序列表标记
    re.compile(r'\*{1,3}([^*]+)\*{1,3}'),  # 粗体/斜体 → 保留文字
    re.compile(r'_{1,3}([^_]+)_{1,3}'),    # 下划线强调 → 保留文字
    re.compile(r'^>\s+', re.M),             # 引用块标记
    re.compile(r'^---+$', re.M),            # 分割线
    re.compile(r'\|.*?\|'),                 # 表格行（粗略去除）
    re.compile(r'<!--[\s\S]*?-->'),         # HTML 注释
    re.compile(r'<[^>]+>'),                 # HTML 标签
]


def strip_markdown(text: str) -> str:
    """将 Markdown 文本转换为纯文本，保留正文内容。"""
    for pattern in _MD_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) if m.lastindex else ' ', text)
    # 合并多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================================
# 本地 .md 文件流式生成器
# ============================================================================

def _md_generator(md_dir: str, min_chars: int = 50) -> Generator[dict, None, None]:
    """
    递归遍历 md_dir 下所有 .md 文件，按段落（空行分割）逐段 yield。

    参数:
        md_dir:    包含 .md 文件的目录路径（或单个 .md 文件路径）
        min_chars: 段落最小字符数，过短的段落跳过
    """
    md_path = Path(md_dir)
    if md_path.is_file() and md_path.suffix.lower() == '.md':
        files = [md_path]
    elif md_path.is_dir():
        files = sorted(md_path.rglob('*.md'))
    else:
        logger.warning(f"md_dir 路径无效或不含 .md 文件: {md_dir}")
        return

    logger.info(f"找到 {len(files)} 个 .md 文件，开始流式读取")

    for filepath in files:
        try:
            raw = filepath.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"读取 {filepath} 失败: {e}")
            continue

        clean = strip_markdown(raw)
        # 按空行切段
        paragraphs = re.split(r'\n{2,}', clean)
        for para in paragraphs:
            para = para.strip()
            if len(para) >= min_chars:
                yield {'text': para}


def make_md_iterable_dataset(
    md_dir: str,
    min_chars: int = 50,
) -> 'IterableDataset':
    """
    将本地 .md 文件目录包装成 HuggingFace IterableDataset。

    参数:
        md_dir:    .md 文件目录或单个 .md 文件路径
        min_chars: 最小段落字符数
    返回:
        datasets.IterableDataset
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("请安装 datasets 库: pip install datasets")

    return IterableDataset.from_generator(
        _md_generator,
        gen_kwargs={'md_dir': md_dir, 'min_chars': min_chars},
    )


# ============================================================================
# 混合流：.md + HuggingFace
# ============================================================================

def make_mixed_hf_iterable(
    md_dir: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: str = 'train',
    hf_text_column: Optional[str] = None,
    md_weight: float = 0.3,
    hf_weight: float = 0.7,
    min_chars: int = 50,
    seed: int = 42,
) -> 'IterableDataset':
    """
    混合本地 .md 流和 HuggingFace 流，返回统一的 IterableDataset（字段: text）。

    参数:
        md_dir:         本地 .md 目录，None 则只用 HF
        hf_dataset:     HF 数据集名称，None 则只用 .md
        hf_config:      HF 数据集 config 名称
        hf_split:       HF 数据集 split
        hf_text_column: HF 文本字段名，None 自动检测
        md_weight:      .md 在混合中的采样权重
        hf_weight:      HF 在混合中的采样权重
        min_chars:      .md 段落最小字符数
        seed:           随机种子
    返回:
        datasets.IterableDataset，字段为 {'text': str}
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("请安装 datasets 库: pip install datasets")

    streams = []
    weights = []

    # — 本地 .md 流 —
    if md_dir is not None:
        md_ds = make_md_iterable_dataset(md_dir, min_chars=min_chars)
        streams.append(md_ds)
        weights.append(md_weight)
        logger.info(f"已添加本地 .md 流: {md_dir} (权重 {md_weight})")

    # — HuggingFace 流 —
    if hf_dataset is not None:
        hf_kwargs = {'split': hf_split, 'streaming': True, 'trust_remote_code': True}
        if hf_config:
            hf_kwargs['name'] = hf_config

        hf_ds = load_dataset(hf_dataset, **hf_kwargs)

        # 自动检测文本列
        if hf_text_column is None:
            _sample_features = list(hf_ds.features.keys()) if hasattr(hf_ds, 'features') and hf_ds.features else []
            _candidates = ['text', 'content', 'document', 'sentence', 'passage', 'abstract', 'body']
            hf_text_column = next((c for c in _candidates if c in _sample_features), None)
            if hf_text_column is None and _sample_features:
                hf_text_column = _sample_features[0]
            logger.info(f"HF 数据集自动检测文本列: '{hf_text_column}'")

        # 统一字段名为 text
        if hf_text_column and hf_text_column != 'text':
            hf_ds = hf_ds.map(
                lambda ex: {'text': ex[hf_text_column]},
                remove_columns=[c for c in (hf_ds.features or {}) if c != hf_text_column],
            )
        elif hf_text_column == 'text':
            # 只保留 text 列，去掉其他
            extra_cols = [c for c in (hf_ds.features or {}) if c != 'text']
            if extra_cols:
                hf_ds = hf_ds.remove_columns(extra_cols)

        # 过滤空文本
        hf_ds = hf_ds.filter(lambda ex: bool(ex.get('text', '').strip()))

        streams.append(hf_ds)
        weights.append(hf_weight)
        logger.info(f"已添加 HuggingFace 流: {hf_dataset}/{hf_config or 'default'} split={hf_split} (权重 {hf_weight})")

    if not streams:
        raise ValueError("必须至少指定 md_dir 或 hf_dataset 之一")

    if len(streams) == 1:
        return streams[0]

    # 归一化权重
    total = sum(weights)
    probs = [w / total for w in weights]

    mixed = interleave_datasets(streams, probabilities=probs, seed=seed, stopping_strategy='all_exhausted')
    logger.info(f"混合流已创建，权重比: {[f'{p:.2%}' for p in probs]}")
    return mixed


# ============================================================================
# PyTorch IterableDataset 包装器（带 tokenize）
# ============================================================================

class MixedStreamDataset(TorchIterableDataset):
    """
    将混合 HuggingFace IterableDataset 包装为 PyTorch IterableDataset。
    在迭代时对文本进行 tokenize，适合直接传入 torch.utils.data.DataLoader。

    返回的每个样本是 dict:
        {
            'input_ids':      LongTensor [max_length],
            'attention_mask': LongTensor [max_length],
            'labels':         LongTensor [max_length],  # 同 input_ids（语言模型训练）
        }
    """

    def __init__(
        self,
        mixed_iterable: 'IterableDataset',
        tokenizer,
        max_length: int = 256,
        label_pad_id: int = -100,
    ):
        """
        参数:
            mixed_iterable: make_mixed_hf_iterable() 返回的混合流
            tokenizer:      HuggingFace 兼容分词器（需有 __call__、pad_token_id）
            max_length:     截断/填充长度
            label_pad_id:   padding 位置的 label id（-100 让 CrossEntropy 忽略）
        """
        super().__init__()
        self.mixed = mixed_iterable
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_pad_id = label_pad_id

        # 确保有 pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    def __iter__(self) -> Iterator[dict]:
        for example in self.mixed:
            text = example.get('text', '')
            if not text or not text.strip():
                continue

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )

            input_ids = encoded['input_ids'].squeeze(0)          # [L]
            attention_mask = encoded['attention_mask'].squeeze(0) # [L]

            # labels：pad 位置设为 -100，其余同 input_ids（Causal LM）
            labels = input_ids.clone()
            labels[attention_mask == 0] = self.label_pad_id

            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }


# ============================================================================
# 便捷入口
# ============================================================================

def create_mixed_iterable(
    tokenizer,
    md_dir: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: str = 'train',
    hf_text_column: Optional[str] = None,
    md_weight: float = 0.3,
    hf_weight: float = 0.7,
    max_length: int = 256,
    min_chars: int = 50,
    seed: int = 42,
    label_pad_id: int = -100,
) -> MixedStreamDataset:
    """
    一步创建可直接用于训练的流式混合数据集。

    参数:
        tokenizer:      HuggingFace 分词器
        md_dir:         本地 .md 文件目录（None 则跳过）
        hf_dataset:     HuggingFace 数据集名称（None 则跳过）
        hf_config:      HF 数据集 config 名
        hf_split:       HF 数据集 split
        hf_text_column: HF 文本列名，None 自动检测
        md_weight:      .md 权重（与 hf_weight 为相对比值）
        hf_weight:      HF 权重
        max_length:     tokenize 最大长度
        min_chars:      .md 段落最小字符数
        seed:           随机种子
        label_pad_id:   padding 位置的 label id
    返回:
        MixedStreamDataset（PyTorch IterableDataset）

    使用示例:
        ds = create_mixed_iterable(
            tokenizer=tokenizer,
            md_dir="./docs",
            hf_dataset="Skylion007/openwebtext",
            md_weight=0.2,
            hf_weight=0.8,
            max_length=512,
        )
        loader = DataLoader(ds, batch_size=16, num_workers=2)
        for batch in loader:
            loss = model(**batch).loss
            loss.backward()
    """
    mixed = make_mixed_hf_iterable(
        md_dir=md_dir,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        hf_split=hf_split,
        hf_text_column=hf_text_column,
        md_weight=md_weight,
        hf_weight=hf_weight,
        min_chars=min_chars,
        seed=seed,
    )
    return MixedStreamDataset(mixed, tokenizer, max_length=max_length, label_pad_id=label_pad_id)
