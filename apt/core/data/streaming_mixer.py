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
from typing import Optional, Generator, List, Iterator, Dict, Any

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
    re.compile(r'(?<!\w)_{1,3}([^_\n]+)_{1,3}(?!\w)'),  # 下划线强调 → 保留文字（不匹配 snake_case）
    re.compile(r'^>\s+', re.M),             # 引用块标记
    re.compile(r'^---+$', re.M),            # 分割线
    re.compile(r'^\|.*$', re.M),            # 表格行（整行去除）
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
# 远程 URL 流式读取（HTTP/HTTPS、GitHub raw URL）
# ============================================================================

def _url_text_generator(
    url: str,
    text_column: Optional[str] = None,
    min_chars: int = 50,
    encoding: str = 'utf-8',
    timeout: int = 30,
) -> Generator[dict, None, None]:
    """
    从远程 HTTP/HTTPS URL 流式逐行读取文本。

    支持格式（由 URL 后缀自动判断）:
        - .txt / .md 及其他纯文本：每个非空行作为一个样本
        - .jsonl / .ndjson：每行解析 JSON，取 text_column 字段（None 则自动检测）
        - GitHub blob URL：自动转换为 raw.githubusercontent.com URL

    参数:
        url:         HTTP(S) 地址
        text_column: JSONL 文本字段名，None 则按 text/content/passage/abstract 顺序检测
        min_chars:   样本最小字符数
        encoding:    文本编码（默认 utf-8）
        timeout:     请求超时秒数
    """
    import urllib.request
    import json as _json

    # GitHub blob URL 自动转换为 raw URL
    if 'github.com' in url and '/blob/' in url:
        url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        logger.info(f"GitHub blob → raw URL: {url}")

    url_path = url.split('?')[0].lower()
    is_jsonl = url_path.endswith('.jsonl') or url_path.endswith('.ndjson')

    req = urllib.request.Request(url, headers={
        'User-Agent': 'APT-Transformer/1.0',
        'Accept': 'text/plain, application/x-ndjson, */*',
    })
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:
        logger.warning(f"URL 连接失败 {url}: {exc}")
        return

    _candidates = ['text', 'content', 'passage', 'abstract', 'body']
    with resp:
        for raw_line in resp:
            line = raw_line.decode(encoding, errors='ignore').rstrip('\r\n')
            if not line.strip():
                continue

            if is_jsonl:
                try:
                    obj = _json.loads(line)
                    if text_column:
                        text = str(obj.get(text_column, ''))
                    else:
                        text = next(
                            (str(obj[c]) for c in _candidates
                             if isinstance(obj.get(c), str) and obj[c].strip()),
                            '',
                        )
                except Exception:
                    text = line
            else:
                text = line

            if len(text) >= min_chars:
                yield {'text': text}


def make_url_iterable_dataset(
    url: str,
    text_column: Optional[str] = None,
    min_chars: int = 50,
    encoding: str = 'utf-8',
    timeout: int = 30,
) -> 'IterableDataset':
    """
    将远程 HTTP/HTTPS URL 包装成 HuggingFace IterableDataset。

    支持 .txt / .md / .jsonl 文件及 GitHub raw URL（自动转换 blob 链接）。

    参数:
        url:         HTTP(S) 地址
        text_column: JSONL 文本字段名，None 自动检测
        min_chars:   最小字符数
        encoding:    文本编码
        timeout:     请求超时秒数（默认 30s）
    返回:
        datasets.IterableDataset，字段为 {'text': str}

    使用示例::

        # GitHub raw 文本文件
        ds = make_url_iterable_dataset(
            "https://github.com/owner/repo/blob/main/data.txt"
        )
        # JSONL 文件（自动检测 text 字段）
        ds = make_url_iterable_dataset(
            "https://example.com/data.jsonl", text_column="content"
        )
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("请安装 datasets 库: pip install datasets")
    return IterableDataset.from_generator(
        _url_text_generator,
        gen_kwargs={
            'url': url,
            'text_column': text_column,
            'min_chars': min_chars,
            'encoding': encoding,
            'timeout': timeout,
        },
    )


# ============================================================================
# ProLong-TextFull MDS 格式读取（支持自动按需下载 shard）
# ============================================================================

def _prolong_mds_generator(
    local_path: Optional[str] = None,
    subsets: Optional[List[str]] = None,
    min_chars: int = 50,
    max_shards_per_subset: int = 3,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Generator[dict, None, None]:
    """
    从 ProLong-TextFull 数据集流式生成文本样本，支持两种模式：

    **自动模式**（local_path=None，默认）：
        通过 huggingface_hub 随机下载 max_shards_per_subset 个 MDS shard 到缓存目录，
        首次运行时自动下载，后续命中缓存直接读取。
        适合底噪训练场景——只需少量书籍长文本，无需提前下载全量数据。
        依赖: pip install mosaicml-streaming huggingface_hub

    **本地模式**（local_path 指向预下载目录）：
        从本地完整数据集读取所有 shard，适合需要全量数据的场景。
        预下载命令::

            huggingface-cli download orionweller/ProLong-TextFull \\
                --repo-type dataset --local-dir ./prolong_textfull \\
                --include "books/*"
            pip install mosaicml-streaming

    参数:
        local_path:             本地目录路径（None = 自动下载模式）
        subsets:                子集列表，默认 ["books"]
        min_chars:              最小字符数
        max_shards_per_subset:  自动模式下每个子集随机下载的 shard 组数（默认 3）
        cache_dir:              自动模式缓存目录（默认 ~/.cache/huggingface/prolong_textfull）
        seed:                   随机选 shard 的种子
    """
    try:
        from streaming import LocalDataset as _MDSLocalDataset
    except ImportError:
        logger.error(
            "ProLong-TextFull 需要 mosaicml-streaming: pip install mosaicml-streaming"
        )
        return

    import random as _random
    _rng = _random.Random(seed)
    _subsets = subsets or ['books']

    # ── 自动模式：按需从 HF Hub 随机下载少量 shard ──────────────────────────
    if local_path is None:
        try:
            from huggingface_hub import list_repo_files as _list_files
            from huggingface_hub import hf_hub_download as _hf_download
        except ImportError:
            logger.error(
                "自动下载模式需要 huggingface_hub: pip install huggingface_hub"
            )
            return

        _cache = cache_dir or os.path.join(
            os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')),
            'prolong_textfull',
        )
        os.makedirs(_cache, exist_ok=True)

        # 列举 HF repo 中的所有文件（只需一次网络请求）
        try:
            all_repo_files = list(_list_files(
                "orionweller/ProLong-TextFull",
                repo_type="dataset",
            ))
        except Exception as exc:
            logger.warning(f"[ProLong] 无法列举 HF 文件，跳过: {exc}")
            return

        for subset in _subsets:
            prefix = f"{subset}/"
            subset_files = [f for f in all_repo_files if f.startswith(prefix)]
            if not subset_files:
                logger.warning(f"[ProLong] {subset}: HF 上未找到文件，跳过")
                continue

            # 找出所有 shard 组目录（如 books/3-9/）
            shard_groups = sorted({
                f.split('/')[1]
                for f in subset_files
                if len(f.split('/')) >= 3
            })
            if not shard_groups:
                logger.warning(f"[ProLong] {subset}: 未找到 shard 子目录")
                continue

            # 随机选 max_shards_per_subset 个组（底噪不需要全量）
            selected = _rng.sample(
                shard_groups,
                min(max_shards_per_subset, len(shard_groups)),
            )
            logger.info(
                f"[ProLong] {subset}: 随机选取 {len(selected)}/{len(shard_groups)} 个 shard 组"
                f" {selected}"
            )

            for sg in selected:
                sg_local = os.path.join(_cache, subset, sg)
                os.makedirs(sg_local, exist_ok=True)

                # 下载该 shard 组的所有文件（index.json + *.mds.zstd）
                sg_prefix = f"{subset}/{sg}/"
                sg_files = [f for f in subset_files if f.startswith(sg_prefix)]

                all_ok = True
                for hf_path in sg_files:
                    local_file = os.path.join(_cache, hf_path)
                    if os.path.exists(local_file):
                        continue  # 已缓存，跳过
                    try:
                        _hf_download(
                            repo_id="orionweller/ProLong-TextFull",
                            filename=hf_path,
                            repo_type="dataset",
                            local_dir=_cache,
                        )
                        logger.debug(f"[ProLong] 已下载: {hf_path}")
                    except Exception as exc:
                        logger.warning(f"[ProLong] 下载 {hf_path} 失败: {exc}")
                        all_ok = False

                if not all_ok:
                    logger.warning(f"[ProLong] {subset}/{sg} 部分文件下载失败，跳过该 shard")
                    continue

                # 读取该 shard 组
                try:
                    ds = _MDSLocalDataset(local=sg_local)
                    logger.info(f"[ProLong] {subset}/{sg}: {len(ds)} 个样本")
                    for i in range(len(ds)):
                        sample = ds[i]
                        text = (
                            sample.get('text')
                            or sample.get('content')
                            or sample.get('raw_content')
                            or ''
                        )
                        if text and len(text) >= min_chars:
                            yield {'text': text}
                except Exception as exc:
                    logger.warning(f"[ProLong] MDS 读取失败 {sg_local}: {exc}")

    # ── 本地模式：从预下载完整目录读取所有 shard ────────────────────────────
    else:
        if not os.path.isdir(local_path):
            logger.warning(f"[ProLong] 本地路径不存在: {local_path}")
            return

        for subset in _subsets:
            subset_dir = os.path.join(local_path, subset)
            if not os.path.isdir(subset_dir):
                logger.warning(f"[ProLong] 子集目录不存在，跳过: {subset_dir}")
                continue

            shard_groups = sorted([
                d for d in os.listdir(subset_dir)
                if os.path.isdir(os.path.join(subset_dir, d))
            ])
            if not shard_groups:
                logger.warning(f"[ProLong] {subset}: 未找到 shard 子目录")
                continue

            for sg in shard_groups:
                sg_path = os.path.join(subset_dir, sg)
                try:
                    ds = _MDSLocalDataset(local=sg_path)
                    logger.info(f"[ProLong] {subset}/{sg}: {len(ds)} 个样本")
                    for i in range(len(ds)):
                        sample = ds[i]
                        text = (
                            sample.get('text')
                            or sample.get('content')
                            or sample.get('raw_content')
                            or ''
                        )
                        if text and len(text) >= min_chars:
                            yield {'text': text}
                except Exception as exc:
                    logger.warning(f"[ProLong] MDS 读取失败 {sg_path}: {exc}")


def make_prolong_mds_iterable(
    local_path: Optional[str] = None,
    subsets: Optional[List[str]] = None,
    min_chars: int = 50,
    max_shards_per_subset: int = 3,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> 'IterableDataset':
    """
    将 ProLong-TextFull MDS 数据集包装成 HuggingFace IterableDataset。

    local_path=None（默认）时自动从 HF Hub 按需随机下载少量 shard，
    无需提前手动下载全量数据。

    参数:
        local_path:             本地目录路径（None = 自动下载 max_shards_per_subset 个 shard）
        subsets:                子集列表（默认 ["books"]）
        min_chars:              最小字符数
        max_shards_per_subset:  自动模式下每个子集随机下载的 shard 组数（默认 3）
        cache_dir:              缓存目录（默认 ~/.cache/huggingface/prolong_textfull）
        seed:                   随机种子
    返回:
        datasets.IterableDataset，字段为 {'text': str}
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("请安装 datasets 库: pip install datasets")
    return IterableDataset.from_generator(
        _prolong_mds_generator,
        gen_kwargs={
            'local_path': local_path,
            'subsets': subsets,
            'min_chars': min_chars,
            'max_shards_per_subset': max_shards_per_subset,
            'cache_dir': cache_dir,
            'seed': seed,
        },
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


# ============================================================================
# 多语言底噪配置（参考 LLaMA-3 / Qwen-2.5 / Gemma-2 等主流 LLM 语言分布）
# ============================================================================

#: 标准多语言预训练底噪配置（13 个数据源，可直接传入 make_multi_source_iterable）
#:
#: 语言分布设计原理：
#:   - 英语 (36%) + 中文 (18%) ≈ 54%：主要目标语言，参考 Qwen-2.5 双语主体
#:   - 日文 8%、韩文 5%：CJK 补充，FineWeb-2 原生质量过滤
#:   - 欧洲语言合计 12%（德/法/西各 4%）：覆盖 ISO 639-1 TOP10 常用语言
#:   - 代码 10%：参考 LLaMA-3 代码占比，提升推理与结构化输出
#:   - 数学 5%：open-web-math，提升 STEM 推理
#:   - Wikipedia 6%（EN/ZH/JA/KO）：高质量知识锚点，轻量上采样
#:
#: 全部使用 FineWeb-2（2025 年）作为 Web 文本来源，在 11/14 语言 benchmark 上
#: 超过 CulturaX / mC4 / CC-100，且无需 HuggingFace 授权协议。
#:
#: weight 为相对采样权重，程序运行时自动归一化（总和无需等于 1）。
MULTILINGUAL_BASE_MIX: List[Dict[str, Any]] = [
    # ── Web 通用文本（FineWeb-2，ODC-By 1.0 许可，无需授权）────────────────
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "eng_Latn",
        "column":  "text",
        "weight":  0.31,
        "lang":    "en",
        "note":    "英语 Web 文本 — FineWeb-2 质量过滤版（CommonCrawl 96 快照）",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "zho_Hans",
        "column":  "text",
        "weight":  0.18,
        "lang":    "zh",
        "note":    "简体中文 Web 文本 — FineWeb-2（GlotLID 语言识别 + MinHash 去重）",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "zho_Hant",
        "column":  "text",
        "weight":  0.05,
        "lang":    "zh-hant",
        "note":    "繁体中文 Web 文本 — FineWeb-2（涵盖台湾/香港/澳门地区繁体语料）",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "jpn_Jpan",
        "column":  "text",
        "weight":  0.08,
        "lang":    "ja",
        "note":    "日文 Web 文本 — FineWeb-2（逐语言过滤阈值调优）",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "kor_Hang",
        "column":  "text",
        "weight":  0.05,
        "lang":    "ko",
        "note":    "韩文 Web 文本 — FineWeb-2",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "deu_Latn",
        "column":  "text",
        "weight":  0.04,
        "lang":    "de",
        "note":    "德语 Web 文本 — FineWeb-2",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "fra_Latn",
        "column":  "text",
        "weight":  0.04,
        "lang":    "fr",
        "note":    "法语 Web 文本 — FineWeb-2",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "spa_Latn",
        "column":  "text",
        "weight":  0.04,
        "lang":    "es",
        "note":    "西班牙语 Web 文本 — FineWeb-2",
    },
    # ── 代码（多语言代码，提升推理与结构化输出能力）──────────────────────────
    {
        "dataset": "codeparrot/github-code-clean",
        "config":  None,
        "column":  "code",
        "weight":  0.08,
        "lang":    "code",
        "note":    "多语言代码 — GitHub 去重版（Python/JS/Java/C++ 等 30+ 语言）",
    },
    # ── 数学/科学（CommonCrawl 提取，提升 STEM 推理能力）──────────────────
    {
        "dataset": "open-web-math/open-web-math",
        "config":  None,
        "column":  "text",
        "weight":  0.05,
        "lang":    "math",
        "note":    "网络数学内容 — CommonCrawl 数学页面提取，14.7B tokens",
    },
    # ── 书籍（提升叙事理解与长文本建模能力）──────────────────────────────────
    {
        "dataset": "rojagtap/bookcorpus",
        "config":  None,
        "column":  "text",
        "weight":  0.03,
        "lang":    "books",
        "note":    "英文书籍 — BookCorpus 7185 本，MIT 许可，提升叙事/长文本理解",
    },
    # ── 长文本书籍（ProLong-TextFull books，自动按需下载 shard）────────────
    {
        "source_type":            "prolong_mds",
        "local_path":             None,       # None = 自动从 HF Hub 下载随机 shard
        "subsets":                ["books"],
        "max_shards_per_subset":  3,          # 底噪只需 3 个 shard 组
        "weight":                 0.04,
        "lang":                   "prolong-books",
        "note":                   "ProLong-TextFull books 子集 — 首次运行自动下载 3 个 shard 到缓存，无需手动准备",
    },
    # ── Wikipedia（高质量知识锚点，小权重上采样，参考 LLaMA-3 处理方式）──
    {
        "dataset": "wikimedia/wikipedia",
        "config":  "20231101.en",
        "column":  "text",
        "weight":  0.02,
        "lang":    "wiki-en",
        "note":    "英文维基百科 2023-11 快照，~6.7M 条",
    },
    {
        "dataset": "wikimedia/wikipedia",
        "config":  "20231101.zh",
        "column":  "text",
        "weight":  0.02,
        "lang":    "wiki-zh",
        "note":    "中文维基百科 2023-11 快照",
    },
    {
        "dataset": "wikimedia/wikipedia",
        "config":  "20231101.ja",
        "column":  "text",
        "weight":  0.01,
        "lang":    "wiki-ja",
        "note":    "日文维基百科 2023-11 快照",
    },
    {
        "dataset": "wikimedia/wikipedia",
        "config":  "20231101.ko",
        "column":  "text",
        "weight":  0.01,
        "lang":    "wiki-ko",
        "note":    "韩文维基百科 2023-11 快照",
    },
]
# 合计权重（全部自动归一化，prolong_mds 首次运行时自动下载 shard）:
# 0.31+0.18+0.05+0.08+0.05+0.04+0.04+0.04+0.08+0.05+0.03+0.04+0.02+0.02+0.01+0.01 = 1.05
# 权重总和 > 1 时由 interleave_datasets 自动归一化为概率分布

# ── Stage 1 alias ──────────────────────────────────────────────────────────────
#: MULTILINGUAL_BASE_MIX 即 Stage 1 通用预训练底噪
STAGE_1_MIX: List[Dict[str, Any]] = MULTILINGUAL_BASE_MIX

# ── Stage 2: 数学/推理能力强化 mix ────────────────────────────────────────────
#:
#: 设计原则：在 Stage 1 通用能力基础上，强化数学符号处理、形式化推理、代码-数学迁移。
#:
#: 相比 Stage 1 主要变化：
#:   1. FineWeb-2 多语言总占比从 74% 降至约 31%，减少通用 Web 噪声
#:   2. 代码占比从 8% 升至 12%，代码推理正向迁移数学推理
#:   3. 新增 proof-pile-2 15%：
#:        arXiv LaTeX 源（~29B tokens，真实数学符号）
#:        + 代数计算栈 algebraic-stack（11B tokens，CAS/Lean/Coq/Isabelle 形式化证明）
#:        + 数学网页 open-web-math（15B tokens）
#:        合计 55B tokens，Llemma 7B/34B 预训练验证配方
#:   4. 新增 finemath-4plus 10%：
#:        LLaMA-3.1-70B 打分 4-5 分的高质量数学解题步骤网页，9.6B tokens
#:        GSM8k/MATH benchmark SOTA（2024）
#:   5. 新增 cosmopedia/auto_math_text 5%：合成数学教科书，10.3B tokens
#:   6. ProLong-TextFull books 子集（首次自动下载 3 个 shard，长文本能力保持）
#:
#: 有效权重合计（prolong_mds 自动下载，全部源均参与）:
#:   0.18+0.10+0.03+0.03+0.12+0.15+0.10+0.05+0.04+0.03+0.02+0.02 = 0.87 → 自动归一化
STAGE_2_MIX: List[Dict[str, Any]] = [
    # ── 通用 Web（缩减至约 31%，保留多语言基础）───────────────────────────────
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "eng_Latn",
        "column":  "text",
        "weight":  0.18,
        "lang":    "en",
        "note":    "英语 Web — FineWeb-2（Stage 2 减量，保留语言能力基础）",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "zho_Hans",
        "column":  "text",
        "weight":  0.10,
        "lang":    "zh",
        "note":    "简体中文 Web — FineWeb-2",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "zho_Hant",
        "column":  "text",
        "weight":  0.03,
        "lang":    "zh-hant",
        "note":    "繁体中文 Web — FineWeb-2（台湾/香港/澳门繁体语料）",
    },
    {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config":  "jpn_Jpan",
        "column":  "text",
        "weight":  0.03,
        "lang":    "ja",
        "note":    "日文 Web — FineWeb-2（保留少量，维持 CJK 能力）",
    },
    # ── 代码（提升至 12%，代码推理与数学推理正向迁移）───────────────────────
    {
        "dataset": "codeparrot/github-code-clean",
        "config":  None,
        "column":  "code",
        "weight":  0.12,
        "lang":    "code",
        "note":    "多语言代码 — github-code-clean（Stage 2 增至 12%，强化推理）",
    },
    # ── arXiv LaTeX 数学源（核心新增，替代 RedPajama/ProLong-arxiv）─────────
    {
        "dataset": "EleutherAI/proof-pile-2",
        "config":  "default",
        "column":  "text",
        "weight":  0.15,
        "lang":    "proof-pile",
        "note":    "proof-pile-2 默认配置：arXiv LaTeX 源(29B) + 代数计算栈(11B) + "
                   "数学网页 open-web-math(15B)，共 55B tokens，Llemma 7B/34B 验证配方",
    },
    # ── 高质量数学网页（LLaMA-3.1-70B 打分过滤）─────────────────────────────
    {
        "dataset": "HuggingFaceTB/finemath",
        "config":  "finemath-4plus",
        "column":  "text",
        "weight":  0.10,
        "lang":    "finemath",
        "note":    "FineMath-4+：LLaMA-3.1-70B 打分 4-5 分，高质量数学解题内容，"
                   "9.6B tokens，GSM8k/MATH SOTA（2024）",
    },
    # ── 合成数学教科书（Cosmopedia）──────────────────────────────────────────
    {
        "dataset": "HuggingFaceTB/cosmopedia",
        "config":  "auto_math_text",
        "column":  "text",
        "weight":  0.05,
        "lang":    "cosmopedia-math",
        "note":    "Cosmopedia auto_math_text：数学网站合成教科书，10.3B tokens，Apache 2.0",
    },
    # ── 长文本书籍（ProLong-TextFull books，自动按需下载 shard）────────────
    {
        "source_type":            "prolong_mds",
        "local_path":             None,       # None = 自动从 HF Hub 下载随机 shard
        "subsets":                ["books"],
        "max_shards_per_subset":  3,
        "weight":                 0.04,
        "lang":                   "prolong-books",
        "note":                   "ProLong-TextFull books 子集 — 首次运行自动下载 3 个 shard 到缓存，无需手动准备",
    },
    {
        "dataset": "rojagtap/bookcorpus",
        "config":  None,
        "column":  "text",
        "weight":  0.03,
        "lang":    "books",
        "note":    "BookCorpus 英文书籍（叙事长文本，维持长程依赖建模）",
    },
    # ── Wikipedia 知识锚点（小权重，避免稀释数学训练目标）──────────────────
    {
        "dataset": "wikimedia/wikipedia",
        "config":  "20231101.en",
        "column":  "text",
        "weight":  0.02,
        "lang":    "wiki-en",
        "note":    "英文维基（知识锚点，小权重）",
    },
    {
        "dataset": "wikimedia/wikipedia",
        "config":  "20231101.zh",
        "column":  "text",
        "weight":  0.02,
        "lang":    "wiki-zh",
        "note":    "中文维基（知识锚点，小权重）",
    },
]


# ============================================================================
# 多源混合流构建器
# ============================================================================

def make_multi_source_iterable(
    sources: List[Dict[str, Any]],
    md_dir: Optional[str] = None,
    md_weight: float = 0.0,
    split: str = 'train',
    seed: int = 42,
    min_chars: int = 50,
) -> 'IterableDataset':
    """
    从多个 HuggingFace 数据源（及可选的本地 .md 目录）构建加权混合流。

    每个 source 条目格式::

        {
            "dataset": str,            # HuggingFace 数据集名称（如 "HuggingFaceFW/fineweb-2"）
            "config":  Optional[str],  # dataset config/子集名，None 表示默认
            "column":  str,            # 文本字段名（如 "text"、"code"、"content"）
            "weight":  float,          # 采样权重（相对值，程序自动归一化）
            "lang":    str,            # 语言标记（仅用于日志打印）
        }

    参数:
        sources:   数据源配置列表（可直接使用 MULTILINGUAL_BASE_MIX）
        md_dir:    额外本地 .md 目录，None 则不使用
        md_weight: .md 流的采样权重（相对值），需同时指定 md_dir
        split:     HuggingFace split（默认 'train'）
        seed:      随机种子
        min_chars: 本地 .md 段落最小字符数
    返回:
        datasets.IterableDataset，统一字段为 {'text': str}

    使用示例::

        from apt.core.data.streaming_mixer import MULTILINGUAL_BASE_MIX, make_multi_source_iterable

        ds = make_multi_source_iterable(MULTILINGUAL_BASE_MIX)
        for sample in ds:
            print(sample['text'][:80])
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("请安装 datasets 库: pip install datasets")
    if not sources:
        raise ValueError("sources 列表不能为空")

    streams: List['IterableDataset'] = []
    weights: List[float] = []
    lang_labels: List[str] = []

    # — 逐一加载各数据流（支持 source_type: hf / prolong_mds / url）—
    for src in sources:
        src_type = src.get('source_type', 'hf').lower()
        col      = src.get('column', 'text')
        w        = float(src.get('weight', 1.0))
        lang_tag = src.get('lang', 'unknown')

        # ── ProLong-TextFull MDS（自动按需下载 shard，或读取本地目录）─────
        if src_type == 'prolong_mds':
            _lp = src.get('local_path')  # None = 自动下载模式
            try:
                ds = make_prolong_mds_iterable(
                    local_path=_lp,
                    subsets=src.get('subsets'),
                    min_chars=min_chars,
                    max_shards_per_subset=src.get('max_shards_per_subset', 3),
                    cache_dir=src.get('cache_dir'),
                    seed=seed,
                )
            except Exception as exc:
                logger.warning(f"[{lang_tag}] ProLong MDS 加载失败，跳过: {exc}")
                continue
            streams.append(ds)
            weights.append(w)
            lang_labels.append(lang_tag)
            mode = f"本地: {_lp}" if _lp else "自动下载 shard"
            logger.info(f"[{lang_tag}] 已添加 ProLong-TextFull MDS ({mode}, 权重={w})")
            continue

        # ── 远程 URL（HTTP/HTTPS，GitHub raw 等）────────────────────────────
        if src_type == 'url':
            url = src.get('url')
            if not url:
                logger.warning(f"[{lang_tag}] URL 源缺少 'url' 字段，跳过")
                continue
            try:
                ds = make_url_iterable_dataset(
                    url=url, text_column=(col if col != 'text' else None),
                    min_chars=min_chars,
                )
            except Exception as exc:
                logger.warning(f"[{lang_tag}] URL 数据源加载失败，跳过: {exc}")
                continue
            streams.append(ds)
            weights.append(w)
            lang_labels.append(lang_tag)
            logger.info(f"[{lang_tag}] 已添加 URL 流: {url} (权重={w})")
            continue

        # ── HuggingFace 流式数据集（默认路径）──────────────────────────────
        ds_name = src.get('dataset')
        if not ds_name:
            logger.warning(f"[{lang_tag}] HF 源缺少 'dataset' 字段，跳过")
            continue

        cfg = src.get('config')
        # 从 lang_tag 没有 dataset 时更新一下
        if lang_tag == 'unknown':
            lang_tag = ds_name.split('/')[-1]

        load_kwargs: Dict[str, Any] = {
            'split': split,
            'streaming': True,
            'trust_remote_code': True,
        }
        if cfg:
            load_kwargs['name'] = cfg

        try:
            ds = load_dataset(ds_name, **load_kwargs)
        except Exception as exc:
            logger.warning(f"[{lang_tag}] 加载 {ds_name}/{cfg or 'default'} 失败，跳过: {exc}")
            continue

        # 统一字段名为 'text'（用默认参数捕获 col，避免闭包问题）
        if col != 'text':
            ds = ds.map(lambda ex, _c=col: {'text': ex.get(_c, '')})

        # 删除多余列（只保留 text），features 在 streaming 模式下可能为 None
        try:
            extra_cols = [c for c in (ds.features or {}) if c != 'text']
            if extra_cols:
                ds = ds.remove_columns(extra_cols)
        except Exception:
            pass

        # 过滤空文本
        ds = ds.filter(lambda ex: bool(ex.get('text', '').strip()))

        streams.append(ds)
        weights.append(w)
        lang_labels.append(lang_tag)
        cfg_str = f"/{cfg}" if cfg else ""
        logger.info(f"[{lang_tag}] 已添加 {ds_name}{cfg_str} (权重={w})")

    # — 可选的本地 .md 流 —
    if md_dir and md_weight > 0:
        md_ds = make_md_iterable_dataset(md_dir, min_chars=min_chars)
        streams.append(md_ds)
        weights.append(md_weight)
        lang_labels.append('md')
        logger.info(f"[md] 已添加本地 .md 流: {md_dir} (权重={md_weight})")

    if not streams:
        raise RuntimeError("所有数据源均加载失败，无法构建混合流")

    if len(streams) == 1:
        return streams[0]

    # 归一化权重为概率
    total = sum(weights)
    probs = [w / total for w in weights]
    dist_str = ", ".join(f"{lbl}={p:.1%}" for lbl, p in zip(lang_labels, probs))
    logger.info(f"多语言混合流就绪 ({len(streams)} 个数据源): {dist_str}")

    return interleave_datasets(
        streams,
        probabilities=probs,
        seed=seed,
        stopping_strategy='all_exhausted',
    )


# ============================================================================
# 便捷入口：一步创建主流 LLM 级多语言预训练底噪
# ============================================================================

def create_multilingual_base_iterable(
    tokenizer,
    max_length: int = 2048,
    seed: int = 42,
    custom_mix: Optional[List[Dict[str, Any]]] = None,
    md_dir: Optional[str] = None,
    md_weight: float = 0.0,
    label_pad_id: int = -100,
) -> 'MixedStreamDataset':
    """
    一步创建主流大模型级多语言预训练底噪数据集。

    默认使用 MULTILINGUAL_BASE_MIX（13 个数据源）：
    - Web 通用文本：英/中/日/韩/德/法/西（FineWeb-2，2025 年，无需 HF 授权）
    - 代码：codeparrot/github-code-clean（10%）
    - 数学：open-web-math（5%）
    - Wikipedia 知识锚点：EN/ZH/JA/KO（合计 6%）

    参数:
        tokenizer:    HuggingFace 分词器
        max_length:   tokenize 最大长度（建议 2048，主流 LLM 标准）
        seed:         随机种子
        custom_mix:   自定义数据源列表，None 则使用 MULTILINGUAL_BASE_MIX
        md_dir:       额外本地 .md 目录（可选，融入项目自有文档）
        md_weight:    本地 .md 流采样权重（相对值）
        label_pad_id: padding 位置的 label id（-100 让 CrossEntropy 忽略）
    返回:
        MixedStreamDataset（PyTorch IterableDataset），可直接传给 DataLoader

    使用示例::

        from apt.core.data.streaming_mixer import create_multilingual_base_iterable

        ds = create_multilingual_base_iterable(tokenizer, max_length=2048)
        loader = DataLoader(ds, batch_size=4, num_workers=4)
        for batch in loader:
            loss = model(**batch).loss

    自定义语言比例示例::

        from apt.core.data.streaming_mixer import MULTILINGUAL_BASE_MIX

        # 只保留 CJK + 英语 + 代码
        my_mix = [s for s in MULTILINGUAL_BASE_MIX
                  if s['lang'] in ('en', 'zh', 'ja', 'ko', 'code')]
        ds = create_multilingual_base_iterable(tokenizer, custom_mix=my_mix)
    """
    mix = custom_mix if custom_mix is not None else MULTILINGUAL_BASE_MIX
    mixed = make_multi_source_iterable(
        sources=mix,
        md_dir=md_dir,
        md_weight=md_weight,
        seed=seed,
    )
    return MixedStreamDataset(
        mixed,
        tokenizer,
        max_length=max_length,
        label_pad_id=label_pad_id,
    )
