#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 速食预训练脚本 (QuickCook Pretraining)

将 C4(en) + Chinese-C4(zh) + FineWeb + HLBD 等数据集混合进行预训练。
可选启用 Wikipedia / arXiv / GitHub Code 数据集。
通过数据稀释 (data dilution) 的方式让模型同时学习英文、中文、代码和学术文本。

分词器采用参考 DeepSeek/Qwen 的 Byte-Level BPE 方案:
  - 使用 HuggingFace tokenizers 库从混合语料自动训练
  - 支持自动扩词: 先加载基础分词器, 再用新语料增量训练合并新词

分布式训练支持:
  - torchrun DDP (默认)
  - DeepSpeed ZeRO-2/3
  - PyTorch FSDP
  - SLURM 集群兼容

持续课程学习 (Curriculum Hot-Swap):
  训练中途可通过修改 datasets/ 目录下的 curriculum.json 热切换数据集,
  无需停机重启。训练器每 N 步自动检查该文件。

用法:
    # 单机 4 卡 DDP (指定模型架构)
    torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \\
        --output-dir $WORK/quickcook_output --model-arch apt --epochs 3

    # 使用 Claude4 架构 + 指定缓存到 work 目录
    torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \\
        --output-dir $WORK/output --model-arch claude4 --cache-dir $WORK/.cache

    # 多机 (SLURM) — 进度实时写到 progress.log
    srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \\
        -m apt.trainops.scripts.pretrain_quickcook \\
        --output-dir $WORK/output --epochs 3
    # 另一个终端看进度:
    #   tail -f $WORK/output/progress.log

    # DeepSpeed ZeRO-3
    deepspeed --num_gpus=8 -m apt.trainops.scripts.pretrain_quickcook \\
        --output-dir $WORK/output --epochs 3 \\
        --distributed-backend deepspeed --zero-stage 3

    # 持续课程学习: 训练中途切换数据
    # 创建 curriculum.json 放到 --datasets-dir:
    #   {"weights": {"c4_en": 0.2, "chinese_c4": 0.3, "fineweb": 0.2, "hlbd": 0.1,
    #               "wiki": 0.1, "arxiv": 0.05, "code": 0.05}}
    # 训练器会在下一个检查间隔自动加载新权重。

    # 单机调试 (不启动分布式)
    python -m apt.trainops.scripts.pretrain_quickcook \\
        --output-dir ./quickcook_output --epochs 1 --no-distributed

    # 全数据源训练 (含 Wiki + arXiv + Code):
    torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \\
        --output-dir $WORK/output --epochs 3 \\
        --use-wiki --weight-wiki 0.10 \\
        --use-arxiv --weight-arxiv 0.08 \\
        --use-code --weight-code 0.05 --code-languages Python JavaScript Go

    # 启用虚拟 GPU 加速 (可选, 任意组合):
    torchrun --nproc_per_node=4 -m apt.trainops.scripts.pretrain_quickcook \\
        --output-dir $WORK/output --epochs 3 \\
        --use-virtual-vram \\
        --use-virtual-blackwell --vb-pulse-interval 20 \\
        --use-virtual-a100 --va100-vram-budget-gb 7.5

作者: chen0430tw
"""

import os
import sys
import math
import json
import time
import logging
import argparse
import signal as _signal
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ============================================================================
# 可选依赖: 虚拟 GPU 加速 (Virtual VRAM / Virtual Blackwell / Virtual A100)
# ============================================================================

# --- Virtual VRAM: 激活值 offload ---
_VIRTUAL_VRAM_AVAILABLE = False
try:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram
    _VIRTUAL_VRAM_AVAILABLE = True
except Exception:
    VirtualVRAMConfig = None  # type: ignore[assignment,misc]
    virtual_vram = None  # type: ignore[assignment]

# --- Virtual Blackwell: 脉冲式量化感知 ---
_VIRTUAL_BLACKWELL_AVAILABLE = False
try:
    from apt.vgpu.runtime.vb_integration import (
        VBConfigV64,
        apply_virtual_blackwell_v64,
        vb_stats_summary,
    )
    _VIRTUAL_BLACKWELL_AVAILABLE = True
except Exception:
    VBConfigV64 = None  # type: ignore[assignment,misc]
    apply_virtual_blackwell_v64 = None  # type: ignore[assignment]
    vb_stats_summary = None  # type: ignore[assignment]

# --- Virtual A100: 三层虚拟显存 + OPU ---
_VIRTUAL_A100_AVAILABLE = False
try:
    import sys as _sys
    _va100_path = str(Path(__file__).resolve().parent.parent.parent.parent / "va100")
    if _va100_path not in _sys.path:
        _sys.path.insert(0, _va100_path)
    from virtual_a100 import (
        VirtualVRAMBackend,
        VA100SignalCollector,
    )
    _VIRTUAL_A100_AVAILABLE = True
except Exception:
    VirtualVRAMBackend = None  # type: ignore[assignment,misc]
    VA100SignalCollector = None  # type: ignore[assignment,misc]


# ============================================================================
# 工具: HF / datasets 缓存路径设置
# ============================================================================

def setup_cache_dir(cache_dir: Optional[str] = None):
    """
    将 HuggingFace Hub / datasets 的缓存目录设置到指定路径。

    训练集群上计算节点和存储节点分离, home 目录通常是 NFS
    且配额小, 不适合放数据缓存。应当存放在 $WORK 下。

    优先级: --cache-dir > $WORK/.cache > $HF_HOME (不改)
    """
    if cache_dir:
        target = cache_dir
    elif "WORK" in os.environ:
        target = os.path.join(os.environ["WORK"], ".cache", "huggingface")
    else:
        # 不修改, 使用 HF 默认
        return

    os.makedirs(target, exist_ok=True)
    os.environ.setdefault("HF_HOME", target)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(target, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(target, "transformers"))
    logger.info(f"HF 缓存目录设置为: {target}")


# ============================================================================
# 工具: 模型架构注册表
# ============================================================================

# 支持的模型架构: 名称 -> (模块路径, 类名)
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "apt":     ("apt.model.architectures.apt_model",      "APTModel"),
    "apt-lite":("apt.model.architectures.apt_model_lite",  "APTModel"),
    "gpt4o":   ("apt.model.architectures.gpt4o_model",    "GPT4oModel"),
    "gpt5":    ("apt.model.architectures.gpt5_model",     "GPT5Model"),
    "claude4": ("apt.model.architectures.claude4_model",   "Claude4Model"),
    "gpto3":   ("apt.model.architectures.gpto3_model",    "GPTo3Model"),
}


def create_model(arch: str, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, max_seq_len: int) -> nn.Module:
    """
    根据架构名称创建模型。

    APT/APT-Lite 使用 APTModelConfiguration (config 对象);
    其他模型直接用关键字参数。
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型架构: {arch}, 可选: {list(MODEL_REGISTRY.keys())}"
        )

    module_path, class_name = MODEL_REGISTRY[arch]
    import importlib
    mod = importlib.import_module(module_path)
    model_cls = getattr(mod, class_name)

    if arch in ("apt", "apt-lite"):
        # APTModel 需要 config 对象
        config_cls = getattr(mod, "APTModelConfiguration")
        config = config_cls(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            max_seq_len=max_seq_len,
        )
        model = model_cls(config)
    elif arch == "gpt5":
        model = model_cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=num_heads,
            n_layers=num_layers,
        )
    elif arch == "gpt4o":
        model = model_cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=num_heads,
            d_ff=d_model * 4,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
    elif arch == "claude4":
        model = model_cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=num_heads,
            d_ff=d_model * 4,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
    elif arch == "gpto3":
        model = model_cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=num_heads,
            d_ff=d_model * 4,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"未知架构: {arch}")

    logger.info(f"创建模型: {arch} ({class_name}), "
                 f"d_model={d_model}, layers={num_layers}, heads={num_heads}")
    return model


# ============================================================================
# 工具: 持续课程学习 (Curriculum Hot-Swap)
# ============================================================================

class CurriculumManager:
    """
    持续课程学习管理器。

    监视 datasets_dir/curriculum.json, 训练中途可以:
    1. 修改数据源权重 (如: 降低 C4 比例, 提高 chinese_c4 比例)
    2. 指定新的本地数据目录
    3. 关闭/开启某个数据源

    curriculum.json 格式:
    {
        "weights": {"c4_en": 0.2, "chinese_c4": 0.5, "fineweb": 0.2, "hlbd": 0.1},
        "hlbd_path": "/data/new_hlbd/",  // 可选
        "use_c4": true,                  // 可选
        "use_mc4_zh": true,              // 可选 (兼容旧名, 控制 chinese_c4)
        "use_fineweb": true              // 可选
    }
    """

    def __init__(self, datasets_dir: Optional[str] = None):
        self._config_path = None
        self._last_mtime = 0.0
        self._current_config: Optional[Dict] = None

        if datasets_dir:
            self._config_path = os.path.join(datasets_dir, "curriculum.json")

    def check_for_update(self) -> Optional[Dict]:
        """
        检查 curriculum.json 是否有更新。

        Returns:
            更新后的配置 dict, 如果没有更新则返回 None。
        """
        if not self._config_path or not os.path.exists(self._config_path):
            return None

        try:
            mtime = os.path.getmtime(self._config_path)
        except OSError:
            return None

        if mtime <= self._last_mtime:
            return None

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                new_config = json.load(f)
            self._last_mtime = mtime
            self._current_config = new_config
            logger.info(f"检测到课程更新: {self._config_path}")
            logger.info(f"  新配置: {json.dumps(new_config, ensure_ascii=False)}")
            return new_config
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"读取 curriculum.json 失败: {e}")
            return None


# ============================================================================
# 工具: 进度追踪 (SLURM 友好)
# ============================================================================

class ProgressTracker:
    """
    进度追踪器, 在 SLURM 环境下也能看到训练进度。

    策略:
    1. 进度写入 output_dir/progress.log (可用 tail -f 实时查看)
    2. 如果 stderr 连接到 tty, 也用 tqdm 显示进度条
    3. 定期刷新, 包含 step/loss/lr/tokens_per_sec 等关键指标
    """

    def __init__(self, output_dir: str, total_steps: int, rank: int = 0,
                 log_interval: int = 10):
        self.rank = rank
        self.total_steps = total_steps
        self.log_interval = log_interval
        self._progress_file = None
        self._tqdm_bar = None
        self._start_time = time.time()
        self._tokens_processed = 0

        if rank == 0:
            # 进度日志文件 (可 tail -f)
            os.makedirs(output_dir, exist_ok=True)
            self._progress_file = open(
                os.path.join(output_dir, "progress.log"), "a", encoding="utf-8"
            )
            self._write_header()

            # 如果有 tqdm 且连接到终端/srun
            try:
                from tqdm import tqdm
                # SLURM 下 stderr 也是有效的
                self._tqdm_bar = tqdm(
                    total=total_steps,
                    desc="QuickCook",
                    unit="step",
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            except ImportError:
                pass

    def _write_header(self):
        if self._progress_file:
            header = (
                f"\n{'='*70}\n"
                f"QuickCook 训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"总步数: {self.total_steps}\n"
                f"{'='*70}\n"
            )
            self._progress_file.write(header)
            self._progress_file.flush()

    def update(self, step: int, loss: float, lr: float,
               tokens_in_batch: int = 0):
        """更新进度"""
        if self.rank != 0:
            return

        self._tokens_processed += tokens_in_batch
        elapsed = time.time() - self._start_time
        tokens_per_sec = self._tokens_processed / elapsed if elapsed > 0 else 0
        pct = step / self.total_steps * 100 if self.total_steps > 0 else 0

        # ETA
        if step > 0:
            eta_sec = elapsed / step * (self.total_steps - step)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        else:
            eta_str = "N/A"

        if step % self.log_interval == 0:
            line = (
                f"[Step {step:>8d}/{self.total_steps} ({pct:5.1f}%)] "
                f"Loss: {loss:.4f} | LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f} | ETA: {eta_str}"
            )

            # 写入文件
            if self._progress_file:
                self._progress_file.write(line + "\n")
                self._progress_file.flush()

            # 写入 logger
            logger.info(line)

        # tqdm
        if self._tqdm_bar is not None:
            self._tqdm_bar.update(1)
            self._tqdm_bar.set_postfix(
                loss=f"{loss:.4f}", lr=f"{lr:.2e}",
                tok_s=f"{tokens_per_sec:,.0f}",
            )

    def close(self):
        if self._tqdm_bar is not None:
            self._tqdm_bar.close()
        if self._progress_file:
            self._progress_file.write(
                f"\n训练结束: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"总 token: {self._tokens_processed:,}\n"
            )
            self._progress_file.close()


# ============================================================================
# 第一部分: 混合流式数据集 (C4 + mC4 + FineWeb + HLBD)
# ============================================================================

class InterleavedStreamDataset(torch.utils.data.IterableDataset):
    """
    交替采样的流式数据集。

    从多个 HuggingFace 流式数据集 + 本地 HLBD 数据中,
    按指定权重随机抽取样本, 实现数据稀释混合学习。

    支持的数据源:
      - C4: allenai/c4 (英文, ~365GB)
      - Chinese-C4: shjwudp/chinese-c4 (中文)
      - FineWeb: HuggingFaceFW/fineweb (英文, ~15T tokens)
      - Wikipedia: omarkamali/wikipedia-monthly (多语言, 月更, 流式)
      - arXiv: togethercomputer/RedPajama-Data-1T "arxiv" (论文全文 LaTeX)
      - Code: codeparrot/github-code (115M 文件, 多语言代码)
      - HLBD: 本地分层语言启蒙数据集 (结构化课程数据)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 2048,
        # 数据集权重: 控制各数据源的采样比例
        weights: Optional[Dict[str, float]] = None,
        # 本地数据路径
        hlbd_path: Optional[str] = None,
        # 本地数据集目录 (用于持续课程学习)
        datasets_dir: Optional[str] = None,
        # HuggingFace 数据集开关
        use_c4: bool = True,
        use_mc4_zh: bool = True,
        use_fineweb: bool = True,
        use_wiki: bool = False,
        use_arxiv: bool = False,
        use_code: bool = False,
        # 顺序遍历模式: 指定的数据源按顺序完整过一遍 (不随机跳跃)
        # "sequential" = 顺序遍历, "interleaved" = 随机采样 (默认)
        wiki_mode: str = "sequential",
        arxiv_mode: str = "sequential",
        code_mode: str = "sequential",
        # Code 数据集语言过滤
        code_languages: Optional[List[str]] = None,
        # 随机种子
        seed: int = 42,
        # 每次从流中预取的文档数
        buffer_size: int = 1000,
        # 分布式: 在多进程中按 rank 跳过
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.buffer_size = buffer_size
        self.rank = rank
        self.world_size = world_size

        # 默认权重
        self.weights = weights or {
            "c4_en": 0.35,
            "chinese_c4": 0.20,
            "fineweb": 0.25,
            "hlbd": 0.10,
            "wiki": 0.00,
            "arxiv": 0.00,
            "code": 0.00,
        }

        self.use_c4 = use_c4
        self.use_mc4_zh = use_mc4_zh
        self.use_fineweb = use_fineweb
        self.use_wiki = use_wiki
        self.use_arxiv = use_arxiv
        self.use_code = use_code
        self.wiki_mode = wiki_mode
        self.arxiv_mode = arxiv_mode
        self.code_mode = code_mode
        self.code_languages = code_languages or ["Python", "JavaScript", "TypeScript",
                                                  "Java", "C", "C++", "Go", "Rust"]
        self.hlbd_path = hlbd_path
        self.datasets_dir = datasets_dir

        # 持续课程学习管理器
        self._curriculum = CurriculumManager(datasets_dir)

        # HLBD 本地数据缓存 (小数据集, 直接全量加载)
        self._hlbd_texts: Optional[List[str]] = None

    def _load_hlbd(self) -> List[str]:
        """加载本地 HLBD 数据集"""
        if self._hlbd_texts is not None:
            return self._hlbd_texts

        if not self.hlbd_path or not os.path.exists(self.hlbd_path):
            logger.warning(f"HLBD 数据路径不存在: {self.hlbd_path}, 跳过 HLBD")
            self._hlbd_texts = []
            return self._hlbd_texts

        try:
            from apt.core.data.hlbd.hlbd_adapter import HLBDDataProcessor
            processor = HLBDDataProcessor(data_path=self.hlbd_path)
            processor.process_data(include_multilingual=True, include_separate_levels=True)
            self._hlbd_texts = processor.get_training_texts()
            logger.info(f"HLBD 加载完成: {len(self._hlbd_texts)} 个样本")
        except Exception as e:
            logger.error(f"HLBD 加载失败: {e}")
            self._hlbd_texts = []

        return self._hlbd_texts

    def _create_hf_stream(self, dataset_name: str, subset: Optional[str] = None,
                          **extra_kwargs):
        """创建 HuggingFace 流式数据集迭代器"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "需要安装 datasets 库: pip install datasets\n"
                "流式加载不需要下载完整数据集, 会按需从 HuggingFace Hub 拉取。"
            )

        kwargs = {"streaming": True, "split": "train", "trust_remote_code": True}
        kwargs.update(extra_kwargs)
        if subset:
            ds = load_dataset(dataset_name, subset, **kwargs)
        else:
            ds = load_dataset(dataset_name, **kwargs)

        # 按 rank 做分片, 避免各进程读到相同数据
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)

        return iter(ds)

    def _extract_text(self, example: Dict[str, Any], source: str) -> Optional[str]:
        """从不同数据源的样本中提取文本"""
        # --- Code 数据集: codeparrot/github-code 使用 "code" 字段 ---
        if source == "code":
            code = example.get("code", "")
            if code:
                # 可选: 加上文件路径作为上下文
                path = example.get("path", "")
                lang = example.get("language", "")
                if path:
                    return f"# {path} ({lang})\n{code}"
                return code
            return None

        # --- Wiki 数据集: wikipedia-monthly 使用 "text" 字段 ---
        # --- arXiv 数据集: RedPajama 使用 "text" 字段 ---
        # --- C4 / Chinese-C4 / FineWeb 都使用 "text" 字段 ---
        if "text" in example:
            return example["text"]

        # mC4 兼容
        if "content" in example:
            return example["content"]

        return None

    def _tokenize_and_chunk(self, text: str) -> List[torch.Tensor]:
        """
        将文本分词并切分为固定长度的训练块。

        对长文档: 切成多个 max_seq_len 的 chunk (无重叠)。
        对短文档: 保留为一个较短的 chunk, 后续由 collate_fn padding。
        """
        ids = self.tokenizer.encode(text)

        # tokenizer.encode 可能返回 Tensor 或 list
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze(0) if ids.dim() > 1 else ids
        else:
            ids = torch.tensor(ids, dtype=torch.long)

        if len(ids) == 0:
            return []

        chunks = []
        for start in range(0, len(ids), self.max_seq_len):
            chunk = ids[start : start + self.max_seq_len]
            if len(chunk) >= 32:  # 太短的 chunk 丢弃
                chunks.append(chunk)

        return chunks

    def _init_source(self, name: str, connect_fn, weight: float,
                     target_dict: dict, names_list: list, weights_list: list,
                     label: str):
        """辅助: 尝试连接一个数据源并注册到目标 dict"""
        try:
            stream = connect_fn()
            target_dict[name] = stream
            names_list.append(name)
            weights_list.append(weight)
            logger.info(f"{label} 流式数据集已连接")
        except Exception as e:
            logger.warning(f"{label} 连接失败, 跳过: {e}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        混合采样迭代器。

        两种模式:
          - interleaved (随机交替): C4/Chinese-C4/FineWeb/HLBD 等大规模数据按权重随机采样
          - sequential (顺序遍历): Wiki/arXiv/Code 等知识密集型数据按顺序完整遍历

        顺序数据源会在每 seq_interval 个随机样本后插入一个顺序文档,
        确保整个数据集被完整学习一遍, 而不是随机跳跃式采样。
        """
        import random
        rng = random.Random(self.seed + self.rank)

        # === 初始化: 分为 interleaved (随机) 和 sequential (顺序) 两组 ===
        interleaved_sources = {}
        interleaved_names = []
        interleaved_weights = []

        sequential_sources = {}   # name -> iterator
        sequential_names = []
        sequential_weights = {}   # name -> weight (用于计算插入频率)

        # --- 判断某个数据源该进哪组 ---
        mode_map = {"wiki": self.wiki_mode, "arxiv": self.arxiv_mode, "code": self.code_mode}

        def _target(name):
            """返回 (target_dict, names_list, weights_list) 给 interleaved 或 sequential"""
            if mode_map.get(name) == "sequential":
                return sequential_sources, sequential_names, None
            return interleaved_sources, interleaved_names, interleaved_weights

        # --- C4 (en): 始终 interleaved ---
        if self.use_c4 and self.weights.get("c4_en", 0) > 0:
            self._init_source(
                "c4_en",
                lambda: self._create_hf_stream("allenai/c4", "en"),
                self.weights["c4_en"],
                interleaved_sources, interleaved_names, interleaved_weights,
                "C4 (en)",
            )

        # --- Chinese-C4 (替代旧版 mC4): 始终 interleaved ---
        zh_weight = self.weights.get("chinese_c4", 0) or self.weights.get("mc4_zh", 0)
        if self.use_mc4_zh and zh_weight > 0:
            self._init_source(
                "chinese_c4",
                lambda: self._create_hf_stream("shjwudp/chinese-c4"),
                zh_weight,
                interleaved_sources, interleaved_names, interleaved_weights,
                "Chinese-C4 (shjwudp/chinese-c4)",
            )

        # --- FineWeb: 始终 interleaved ---
        if self.use_fineweb and self.weights.get("fineweb", 0) > 0:
            self._init_source(
                "fineweb",
                lambda: self._create_hf_stream("HuggingFaceFW/fineweb"),
                self.weights["fineweb"],
                interleaved_sources, interleaved_names, interleaved_weights,
                "FineWeb",
            )

        # --- Wikipedia (omarkamali/wikipedia-monthly) ---
        if self.use_wiki and self.weights.get("wiki", 0) > 0:
            td, tn, tw = _target("wiki")
            w = self.weights["wiki"]
            try:
                wiki_stream = self._create_hf_stream(
                    "omarkamali/wikipedia-monthly", "latest.en"
                )
                td["wiki"] = wiki_stream
                tn.append("wiki")
                if tw is not None:
                    tw.append(w)
                else:
                    sequential_weights["wiki"] = w
                mode_label = "顺序遍历" if self.wiki_mode == "sequential" else "随机交替"
                logger.info(f"Wikipedia (en, wikipedia-monthly) [{mode_label}] 已连接")
            except Exception as e:
                logger.warning(f"Wikipedia 连接失败, 跳过: {e}")

        # --- arXiv (RedPajama-Data-1T arxiv 子集) ---
        if self.use_arxiv and self.weights.get("arxiv", 0) > 0:
            td, tn, tw = _target("arxiv")
            w = self.weights["arxiv"]
            try:
                arxiv_stream = self._create_hf_stream(
                    "togethercomputer/RedPajama-Data-1T", "arxiv"
                )
                td["arxiv"] = arxiv_stream
                tn.append("arxiv")
                if tw is not None:
                    tw.append(w)
                else:
                    sequential_weights["arxiv"] = w
                mode_label = "顺序遍历" if self.arxiv_mode == "sequential" else "随机交替"
                logger.info(f"arXiv (RedPajama-1T/arxiv) [{mode_label}] 已连接")
            except Exception as e:
                logger.warning(f"arXiv 连接失败, 跳过: {e}")

        # --- GitHub Code (codeparrot/github-code) ---
        if self.use_code and self.weights.get("code", 0) > 0:
            td, tn, tw = _target("code")
            w = self.weights["code"]
            try:
                code_stream = self._create_hf_stream(
                    "codeparrot/github-code",
                    languages=self.code_languages,
                )
                td["code"] = code_stream
                tn.append("code")
                if tw is not None:
                    tw.append(w)
                else:
                    sequential_weights["code"] = w
                langs = ", ".join(self.code_languages[:3])
                mode_label = "顺序遍历" if self.code_mode == "sequential" else "随机交替"
                logger.info(f"GitHub Code ({langs}...) [{mode_label}] 已连接")
            except Exception as e:
                logger.warning(f"GitHub Code 连接失败, 跳过: {e}")

        # --- HLBD (本地, 循环采样): 始终 interleaved ---
        hlbd_texts = self._load_hlbd()
        hlbd_idx = 0
        if hlbd_texts and self.weights.get("hlbd", 0) > 0:
            interleaved_names.append("hlbd")
            interleaved_weights.append(self.weights["hlbd"])
            logger.info(f"HLBD 本地数据已加载: {len(hlbd_texts)} 样本")

        if not interleaved_names and not sequential_names:
            raise RuntimeError("没有可用的数据源! 请检查网络连接或 HLBD 数据路径。")

        # === 归一化 interleaved 权重 ===
        if interleaved_weights:
            total_w = sum(interleaved_weights)
            interleaved_weights = [w / total_w for w in interleaved_weights]

        # === 计算 sequential 插入频率 ===
        # seq_interval: 每产出 N 个 interleaved 样本后, 插入 1 个 sequential 文档
        # 越大的 weight → 插入越频繁 (interval 越小)
        # 公式: interval = round(1 / weight) ，weight=0.10 → 每 10 个随机样本插 1 个
        seq_intervals = {}
        for name in sequential_names:
            w = sequential_weights.get(name, 0.10)
            seq_intervals[name] = max(1, round(1.0 / w))

        if sequential_names:
            logger.info(
                f"顺序遍历数据源: "
                + ", ".join(f"{n} (每{seq_intervals[n]}步插入1篇)" for n in sequential_names)
            )

        # === 主循环 ===
        interleaved_exhausted = set()
        sequential_exhausted = set()
        sample_count = 0
        seq_round_robin = 0  # 轮询 sequential 数据源

        # 辅助: 从一个数据源取下一篇文档并 yield 所有 chunk
        def _yield_from_source(name, stream):
            """从数据源取一篇文档, 返回 chunk 生成器"""
            try:
                example = next(stream)
            except StopIteration:
                return None  # 已耗尽
            text = self._extract_text(example, name)
            if text is None or len(text.strip()) < 10:
                return []  # 空文档, 返回空 list (非 None, 表示未耗尽)
            return self._tokenize_and_chunk(text)

        while True:
            # 检查是否所有数据源都耗尽
            all_interleaved_done = (
                len(interleaved_exhausted) >= len(interleaved_names)
                if interleaved_names else True
            )
            all_sequential_done = (
                len(sequential_exhausted) >= len(sequential_names)
                if sequential_names else True
            )
            if all_interleaved_done and all_sequential_done:
                break

            sample_count += 1

            # === 持续课程学习: 每 1000 个样本检查 curriculum.json ===
            if sample_count % 1000 == 0:
                update = self._curriculum.check_for_update()
                if update:
                    new_weights = update.get("weights")
                    if new_weights:
                        for i, name in enumerate(interleaved_names):
                            if name in new_weights:
                                interleaved_weights[i] = new_weights[name]
                        w_sum = sum(interleaved_weights)
                        if w_sum > 0:
                            interleaved_weights = [w / w_sum for w in interleaved_weights]
                        logger.info(f"课程权重已更新: {dict(zip(interleaved_names, interleaved_weights))}")
                    new_hlbd = update.get("hlbd_path")
                    if new_hlbd and new_hlbd != self.hlbd_path:
                        self.hlbd_path = new_hlbd
                        self._hlbd_texts = None
                        hlbd_texts = self._load_hlbd()
                        logger.info(f"HLBD 数据路径已切换: {new_hlbd}")

            # === 顺序数据源: 按频率插入 ===
            if sequential_names and not all_sequential_done:
                # 轮询每个 sequential 源, 检查是否该插入
                for _ in range(len(sequential_names)):
                    seq_name = sequential_names[seq_round_robin % len(sequential_names)]
                    seq_round_robin += 1

                    if seq_name in sequential_exhausted:
                        continue

                    interval = seq_intervals[seq_name]
                    if sample_count % interval != 0:
                        continue

                    stream = sequential_sources.get(seq_name)
                    if stream is None:
                        sequential_exhausted.add(seq_name)
                        continue

                    chunks = _yield_from_source(seq_name, stream)
                    if chunks is None:
                        logger.info(f"顺序数据源 {seq_name} 已完整遍历")
                        sequential_exhausted.add(seq_name)
                        continue

                    for chunk in chunks:
                        yield {"input_ids": chunk[:-1], "labels": chunk[1:]}
                    break  # 每步最多插入一篇 sequential 文档

            # === Interleaved 数据源: 加权随机采样 ===
            if interleaved_names and not all_interleaved_done:
                chosen = rng.choices(interleaved_names, weights=interleaved_weights, k=1)[0]

                if chosen in interleaved_exhausted:
                    continue

                text = None

                if chosen == "hlbd":
                    if not hlbd_texts:
                        interleaved_exhausted.add("hlbd")
                        continue
                    text = hlbd_texts[hlbd_idx % len(hlbd_texts)]
                    hlbd_idx += 1
                else:
                    stream = interleaved_sources.get(chosen)
                    if stream is None:
                        interleaved_exhausted.add(chosen)
                        continue
                    try:
                        example = next(stream)
                        text = self._extract_text(example, chosen)
                    except StopIteration:
                        logger.info(f"数据源 {chosen} 已耗尽")
                        interleaved_exhausted.add(chosen)
                        continue

                if text is None or len(text.strip()) < 10:
                    continue

                chunks = self._tokenize_and_chunk(text)
                for chunk in chunks:
                    yield {"input_ids": chunk[:-1], "labels": chunk[1:]}


def quickcook_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    动态 padding 的 collate 函数。

    将一个 batch 中长度不等的 input_ids/labels pad 到同一长度。
    """
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    max_len = max(t.size(0) for t in input_ids_list)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for inp, lab in zip(input_ids_list, labels_list):
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            padded_input_ids.append(
                torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )
            # label 的 padding 位置用 -100 (PyTorch CE loss 忽略)
            padded_labels.append(
                torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            )
            attention_masks.append(
                torch.cat([torch.ones(inp.size(0), dtype=torch.long),
                           torch.zeros(pad_len, dtype=torch.long)])
            )
        else:
            padded_input_ids.append(inp)
            padded_labels.append(lab)
            attention_masks.append(torch.ones(inp.size(0), dtype=torch.long))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


# ============================================================================
# 第二部分: 自适应 Byte-Level BPE 分词器
# ============================================================================

class AdaptiveBPETokenizer:
    """
    自适应 Byte-Level BPE 分词器。

    设计参考:
      - DeepSeek BBPE: 以 UTF-8 byte 为基础单元, 仅 256 个 base token,
        天然支持任何语言, 不会产生 UNK。
      - Qwen tiktoken: 15万+ 大词表, 对中文/代码效率高。
      - LLaMA3 byte-level BPE: 非 SentencePiece, 直接操作字节序列。

    本实现:
      1. 基础: 使用 HuggingFace tokenizers 库训练 Byte-Level BPE
      2. 扩词: 支持从新语料增量学习新 merge rule, 自动扩展词表
      3. 特殊 token: <pad>, <eos>, <bos>, <unk>, <sep> 等
      4. 序列化: 保存/加载为 JSON 格式, 兼容 transformers

    用法:
        # 从语料训练新分词器
        tokenizer = AdaptiveBPETokenizer.train_from_corpus(
            corpus_files=["en_corpus.txt", "zh_corpus.txt"],
            vocab_size=65536,
        )

        # 从已保存的分词器加载, 并用新语料扩词
        tokenizer = AdaptiveBPETokenizer.load("tokenizer.json")
        tokenizer.expand_vocab(new_corpus_files=["new_data.txt"], target_vocab_size=70000)
    """

    # 特殊 token 定义 (固定 ID)
    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<sep>": 4,
        "<mask>": 5,
    }

    def __init__(self, backend_tokenizer=None, vocab_size: int = 65536):
        """
        Args:
            backend_tokenizer: HuggingFace tokenizers.Tokenizer 实例
            vocab_size: 目标词表大小
        """
        self._tokenizer = backend_tokenizer
        self._vocab_size = vocab_size

        # 如果没有后端分词器, 创建一个空的
        if self._tokenizer is None:
            self._build_empty_tokenizer()

    def _build_empty_tokenizer(self):
        """创建空的 Byte-Level BPE 分词器 (仅有 256 byte tokens + special tokens)"""
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
        except ImportError:
            raise ImportError(
                "需要安装 tokenizers 库: pip install tokenizers\n"
                "这是 HuggingFace 的高性能 Rust 分词器。"
            )

        self._tokenizer = Tokenizer(models.BPE())
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # 添加特殊 token
        from tokenizers import AddedToken
        special_list = [
            AddedToken(tok, special=True) for tok in self.SPECIAL_TOKENS.keys()
        ]
        self._tokenizer.add_special_tokens(special_list)

    @classmethod
    def train_from_corpus(
        cls,
        corpus_files: List[str],
        vocab_size: int = 65536,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "AdaptiveBPETokenizer":
        """
        从语料文件训练 Byte-Level BPE 分词器。

        Args:
            corpus_files: 语料文件路径列表 (每行一句文本)
            vocab_size: 目标词表大小
            min_frequency: 最低 merge 频次阈值
            show_progress: 是否显示训练进度

        Returns:
            训练好的 AdaptiveBPETokenizer 实例
        """
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders
            from tokenizers import processors, trainers
        except ImportError:
            raise ImportError("需要安装 tokenizers: pip install tokenizers")

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # 使用 ByteLevel 的 256 个字节作为初始字母表
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=list(cls.SPECIAL_TOKENS.keys()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # 验证语料文件存在
        valid_files = [f for f in corpus_files if os.path.exists(f)]
        if not valid_files:
            raise FileNotFoundError(f"没有找到有效的语料文件: {corpus_files}")

        logger.info(f"开始训练 Byte-Level BPE 分词器, 目标词表: {vocab_size}")
        logger.info(f"使用语料文件: {valid_files}")

        tokenizer.train(valid_files, trainer)

        actual_vocab_size = tokenizer.get_vocab_size()
        logger.info(f"分词器训练完成, 实际词表大小: {actual_vocab_size}")

        return cls(backend_tokenizer=tokenizer, vocab_size=actual_vocab_size)

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator,
        vocab_size: int = 65536,
        min_frequency: int = 2,
    ) -> "AdaptiveBPETokenizer":
        """
        从文本迭代器训练 (适用于流式数据或内存中的数据)。

        Args:
            text_iterator: 产生文本字符串的迭代器
            vocab_size: 目标词表大小
            min_frequency: 最低频次

        Returns:
            训练好的分词器
        """
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders
            from tokenizers import processors, trainers
        except ImportError:
            raise ImportError("需要安装 tokenizers: pip install tokenizers")

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(cls.SPECIAL_TOKENS.keys()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train_from_iterator(text_iterator, trainer)

        actual_vocab_size = tokenizer.get_vocab_size()
        logger.info(f"分词器训练完成 (从迭代器), 词表大小: {actual_vocab_size}")

        return cls(backend_tokenizer=tokenizer, vocab_size=actual_vocab_size)

    def expand_vocab(
        self,
        new_corpus_files: Optional[List[str]] = None,
        new_texts: Optional[List[str]] = None,
        target_vocab_size: Optional[int] = None,
    ):
        """
        自动扩词: 用新语料增量扩展词表。

        原理: 在现有 merge table 基础上, 对新语料中的 token
        再跑一轮 BPE 训练, 把新发现的高频 merge 追加到词表中。

        Args:
            new_corpus_files: 新语料文件列表
            new_texts: 新语料文本列表 (与 new_corpus_files 二选一)
            target_vocab_size: 扩词后的目标词表大小
        """
        try:
            from tokenizers import trainers, pre_tokenizers
        except ImportError:
            raise ImportError("需要安装 tokenizers: pip install tokenizers")

        if target_vocab_size is None:
            target_vocab_size = self._vocab_size + 10000

        current_size = self.vocab_size

        if target_vocab_size <= current_size:
            logger.info(
                f"目标词表 ({target_vocab_size}) <= 当前词表 ({current_size}), 无需扩词"
            )
            return

        logger.info(
            f"开始扩词: {current_size} -> {target_vocab_size} "
            f"(+{target_vocab_size - current_size} tokens)"
        )

        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=2,
            special_tokens=list(self.SPECIAL_TOKENS.keys()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            continuing_subword_prefix="",
        )

        if new_corpus_files:
            valid_files = [f for f in new_corpus_files if os.path.exists(f)]
            if valid_files:
                self._tokenizer.train(valid_files, trainer)
        elif new_texts:
            self._tokenizer.train_from_iterator(iter(new_texts), trainer)
        else:
            logger.warning("未提供新语料, 扩词跳过")
            return

        new_size = self._tokenizer.get_vocab_size()
        self._vocab_size = new_size
        logger.info(f"扩词完成, 新词表大小: {new_size} (+{new_size - current_size})")

    # ---- 编码/解码 API ----

    def encode(self, text: str, return_tensors: Optional[str] = None,
               max_length: Optional[int] = None, truncation: bool = False) -> Any:
        """编码文本为 token ID 序列"""
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids

        if max_length and truncation and len(ids) > max_length:
            ids = ids[:max_length]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """解码 token ID 序列为文本"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None,
                     truncation: bool = False) -> List[List[int]]:
        """批量编码"""
        encodings = self._tokenizer.encode_batch(texts)
        result = []
        for enc in encodings:
            ids = enc.ids
            if max_length and truncation and len(ids) > max_length:
                ids = ids[:max_length]
            result.append(ids)
        return result

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<eos>"]

    @property
    def bos_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<bos>"]

    def save(self, path: str):
        """保存分词器到文件"""
        self._tokenizer.save(path)
        logger.info(f"分词器已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> "AdaptiveBPETokenizer":
        """从文件加载分词器"""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError("需要安装 tokenizers: pip install tokenizers")

        tokenizer = Tokenizer.from_file(path)
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"分词器已加载: {path}, 词表大小: {vocab_size}")
        return cls(backend_tokenizer=tokenizer, vocab_size=vocab_size)


# ============================================================================
# 第三部分: 分布式训练引擎
# ============================================================================

class DistributedConfig:
    """分布式训练配置"""

    def __init__(
        self,
        backend: str = "ddp",         # ddp / deepspeed / fsdp
        zero_stage: int = 2,           # DeepSpeed ZeRO stage (1/2/3)
        gradient_accumulation_steps: int = 1,
        use_gradient_checkpointing: bool = False,
        use_mixed_precision: bool = True,
        mixed_precision_dtype: str = "bf16",  # bf16 / fp16
        find_unused_parameters: bool = False,
    ):
        self.backend = backend
        self.zero_stage = zero_stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.mixed_precision_dtype = mixed_precision_dtype
        self.find_unused_parameters = find_unused_parameters


def setup_distributed(dist_config: DistributedConfig):
    """
    初始化分布式训练环境。

    兼容 torchrun (RANK/WORLD_SIZE/LOCAL_RANK) 和 SLURM。

    Returns:
        (rank, world_size, local_rank, device)
    """
    if dist_config.backend == "deepspeed":
        try:
            import deepspeed
            deepspeed.init_distributed()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        except ImportError:
            raise ImportError("需要安装 deepspeed: pip install deepspeed")
    else:
        # torchrun / SLURM
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        elif "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            local_rank = rank % torch.cuda.device_count()
        else:
            # 单机单卡 fallback
            rank, world_size, local_rank = 0, 1, 0

        if world_size > 1 and not dist.is_initialized():
            nccl_available = torch.cuda.is_available()
            dist.init_process_group(
                backend="nccl" if nccl_available else "gloo",
                init_method="env://",
            )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_distributed(model, device, dist_config: DistributedConfig,
                           rank: int = 0, world_size: int = 1):
    """
    按分布式策略包装模型。

    Args:
        model: nn.Module
        device: torch.device
        dist_config: 分布式配置
        rank: 当前进程 rank
        world_size: 总进程数

    Returns:
        包装后的模型 (DDP / FSDP / DeepSpeed engine)
    """
    model = model.to(device)

    if world_size <= 1:
        return model

    if dist_config.backend == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=dist_config.find_unused_parameters,
        )
        logger.info(f"[Rank {rank}] 模型已包装为 DDP")

    elif dist_config.backend == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        mp_policy = None
        if dist_config.use_mixed_precision:
            dtype = torch.bfloat16 if dist_config.mixed_precision_dtype == "bf16" else torch.float16
            mp_policy = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )

        model = FSDP(model, mixed_precision=mp_policy)
        logger.info(f"[Rank {rank}] 模型已包装为 FSDP")

    elif dist_config.backend == "deepspeed":
        # DeepSpeed 在训练循环中通过 deepspeed.initialize() 处理
        logger.info(f"[Rank {rank}] DeepSpeed 模式, 模型将在 initialize() 时包装")

    return model


def get_deepspeed_config(dist_config: DistributedConfig, lr: float,
                         train_batch_size: int, grad_accum: int) -> Dict:
    """生成 DeepSpeed JSON 配置"""
    dtype_key = "bf16" if dist_config.mixed_precision_dtype == "bf16" else "fp16"

    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": 1000,
                "total_num_steps": 100000,
            },
        },
        dtype_key: {"enabled": True},
        "zero_optimization": {
            "stage": dist_config.zero_stage,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }

    return config


# ============================================================================
# 第四部分: 训练循环
# ============================================================================

class QuickCookTrainer:
    """
    速食预训练器。

    整合数据、模型、分词器、分布式策略, 执行预训练循环。
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AdaptiveBPETokenizer,
        train_dataset: InterleavedStreamDataset,
        output_dir: str,
        dist_config: DistributedConfig,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = torch.device("cpu"),
        # 训练超参
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 1000,
        max_steps: Optional[int] = None,
        epochs: int = 1,
        # --- 可选: 虚拟 GPU 加速 ---
        virtual_vram_config: Optional[Any] = None,        # VirtualVRAMConfig
        vb_adapter: Optional[Any] = None,                 # VirtualBlackwellAdapterV64
        va100_tier_manager: Optional[Any] = None,         # VirtualVRAMBackend
        va100_signal_collector: Optional[Any] = None,     # VA100SignalCollector
        # 模型配置 (保存到 checkpoint 以便恢复)
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.dist_config = dist_config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.epochs = epochs
        self.model_config = model_config or {}

        # 可选虚拟 GPU 组件
        self._vram_config = virtual_vram_config
        self._vb_adapter = vb_adapter
        self._va100_tier = va100_tier_manager
        self._va100_signal = va100_signal_collector

        self.global_step = 0
        self.best_loss = float("inf")

        os.makedirs(output_dir, exist_ok=True)

    def _create_dataloader(self):
        """创建 DataLoader (流式数据集不需要 Sampler)"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: quickcook_collate_fn(
                batch, pad_token_id=self.tokenizer.pad_token_id
            ),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4,
        )

    def _create_optimizer_and_scheduler(self, total_steps: int):
        """创建优化器和学习率调度器"""
        # 分组: bias 和 LayerNorm 不做 weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        param_groups = [
            {
                "params": [
                    p for n, p in base_model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in base_model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups, lr=self.learning_rate, betas=(0.9, 0.95), eps=1e-8
        )

        # Cosine schedule with warmup (10% warmup)
        warmup_steps = max(int(total_steps * 0.1), 100)
        try:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        except ImportError:
            # 没有 transformers 时用 PyTorch 内置的线性 scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
            )

        return optimizer, scheduler

    def _save_checkpoint(self, optimizer, scheduler, metrics: Dict):
        """保存检查点 (仅 rank 0)"""
        if self.rank != 0:
            return

        base_model = self.model.module if hasattr(self.model, "module") else self.model

        ckpt = {
            "format": "quickcook",
            "global_step": self.global_step,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "model_config": self.model_config,
        }

        ckpt_path = os.path.join(
            self.output_dir, f"checkpoint_step_{self.global_step}.pt"
        )
        torch.save(ckpt, ckpt_path)
        logger.info(f"检查点已保存: {ckpt_path}")

        # 同时保存分词器
        tok_path = os.path.join(self.output_dir, "tokenizer.json")
        self.tokenizer.save(tok_path)

    def train(self):
        """主训练循环"""
        dataloader = self._create_dataloader()

        # 估算总步数 (流式数据集无法精确知道总量, 按 max_steps 或 epoch 近似)
        if self.max_steps:
            total_steps = self.max_steps
        else:
            # 粗估: 每 epoch 约 100k 步 (可调)
            total_steps = 100000 * self.epochs

        # DeepSpeed 特殊处理
        if self.dist_config.backend == "deepspeed":
            return self._train_deepspeed(dataloader, total_steps)

        # 标准训练路径 (DDP / FSDP / 单机)
        optimizer, scheduler = self._create_optimizer_and_scheduler(total_steps)

        # 混合精度
        scaler = None
        autocast_ctx = None
        if self.dist_config.use_mixed_precision and torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if self.dist_config.mixed_precision_dtype == "bf16"
                else torch.float16
            )
            autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
            if dtype == torch.float16:
                scaler = torch.amp.GradScaler("cuda")

        grad_accum = self.dist_config.gradient_accumulation_steps
        running_loss = 0.0
        step_in_accum = 0

        logger.info(f"[Rank {self.rank}] 开始训练, 总步数估计: {total_steps}")
        logger.info(
            f"  batch_size={self.batch_size}, grad_accum={grad_accum}, "
            f"lr={self.learning_rate}, mixed_precision={self.dist_config.use_mixed_precision}"
        )

        # 虚拟 GPU 组件日志
        if self.rank == 0:
            if self._vram_config is not None:
                logger.info("  [vGPU] Virtual VRAM 已启用 (激活值 offload)")
            if self._vb_adapter is not None:
                logger.info("  [vGPU] Virtual Blackwell 已启用 (脉冲式量化感知)")
            if self._va100_tier is not None:
                logger.info("  [vGPU] Virtual A100 三层显存管理已启用")

        # 进度追踪器 (写入 progress.log + tqdm)
        progress = ProgressTracker(
            self.output_dir, total_steps, self.rank, self.log_interval
        )

        # 是否用 virtual_vram 包裹 forward+backward
        use_vram_ctx = (self._vram_config is not None and virtual_vram is not None)

        for epoch in range(self.epochs):
            if self.rank == 0:
                logger.info(f"===== Epoch {epoch + 1}/{self.epochs} =====")

            for batch in dataloader:
                # 移到设备
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # ── forward + backward (可选 virtual_vram 包裹) ──
                if use_vram_ctx:
                    # Virtual VRAM: 激活值自动 offload 到 CPU, 降低显存峰值
                    with virtual_vram(self._vram_config):
                        loss, running_loss = self._forward_backward_step(
                            input_ids, labels, autocast_ctx, scaler,
                            grad_accum, running_loss,
                        )
                else:
                    loss, running_loss = self._forward_backward_step(
                        input_ids, labels, autocast_ctx, scaler,
                        grad_accum, running_loss,
                    )

                step_in_accum += 1

                # 累积够了再更新
                if step_in_accum >= grad_accum:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                    self.global_step += 1
                    step_in_accum = 0

                    # VA100 信号采集 (每步记录 loss 和显存)
                    if self._va100_signal is not None:
                        self._va100_signal_tick(loss.item() * grad_accum)

                    # 进度追踪 (替代原来的日志)
                    cur_loss = running_loss / min(self.global_step, self.log_interval)
                    cur_lr = scheduler.get_last_lr()[0]
                    tokens_in_batch = input_ids.numel()
                    progress.update(
                        self.global_step, cur_loss, cur_lr,
                        tokens_in_batch=tokens_in_batch,
                    )

                    if self.global_step % self.log_interval == 0:
                        running_loss = 0.0

                    # 定期打印虚拟 GPU 统计 (每 save_interval 步)
                    if (self.global_step % self.save_interval == 0
                            and self.rank == 0):
                        self._log_vgpu_stats()

                    # 保存
                    if self.global_step % self.save_interval == 0:
                        self._save_checkpoint(
                            optimizer, scheduler,
                            {"loss": running_loss, "step": self.global_step}
                        )
                        if self.world_size > 1:
                            dist.barrier()

                    # 到达最大步数
                    if self.max_steps and self.global_step >= self.max_steps:
                        break

            if self.max_steps and self.global_step >= self.max_steps:
                break

        # 最终保存
        self._save_checkpoint(
            optimizer, scheduler,
            {"loss": running_loss, "step": self.global_step, "final": True}
        )
        progress.close()

        # 最终虚拟 GPU 统计
        if self.rank == 0:
            self._log_vgpu_stats()
            logger.info(f"训练完成! 总步数: {self.global_step}")

    def _forward_backward_step(self, input_ids, labels, autocast_ctx, scaler,
                               grad_accum, running_loss):
        """一个 micro-batch 的 forward + backward (抽取以便 virtual_vram 包裹)"""
        if autocast_ctx is not None:
            with autocast_ctx:
                outputs = self._forward(input_ids, labels)
                loss = outputs["loss"] / grad_accum
        else:
            outputs = self._forward(input_ids, labels)
            loss = outputs["loss"] / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * grad_accum
        return loss, running_loss

    def _va100_signal_tick(self, loss_val: float):
        """VA100 三层显存: 每步记录统计"""
        if self._va100_tier is None:
            return
        try:
            # 记录一个训练步骤
            self._va100_tier.stats.record_step()
        except Exception:
            pass  # 信号采集不影响训练主路径

    def _log_vgpu_stats(self):
        """打印虚拟 GPU 组件的统计摘要"""
        if self._vb_adapter is not None and vb_stats_summary is not None:
            try:
                summary = vb_stats_summary(self._vb_adapter)
                logger.info(f"[Virtual Blackwell 统计]\n{summary}")
            except Exception as e:
                logger.debug(f"VB 统计输出失败: {e}")

        if self._va100_tier is not None:
            try:
                vram_stats = self._va100_tier.stats
                logger.info(
                    f"[Virtual A100 三层显存] "
                    f"hot: {vram_stats.hot.count} tiles ({vram_stats.hot.bytes_total/1e6:.1f}MB), "
                    f"warm: {vram_stats.warm.count} tiles ({vram_stats.warm.bytes_total/1e6:.1f}MB), "
                    f"cold: {vram_stats.cold.count} tiles ({vram_stats.cold.bytes_total/1e6:.1f}MB), "
                    f"搬运时间={vram_stats.total_transfer_time_s:.3f}s, "
                    f"μ(摩擦)={vram_stats.friction_mu:.4f}, "
                    f"τ(重建税)={vram_stats.rebuild_tax_tau:.4f}"
                )
            except Exception as e:
                logger.debug(f"VA100 统计输出失败: {e}")

    def _train_deepspeed(self, dataloader, total_steps: int):
        """DeepSpeed 训练路径"""
        try:
            import deepspeed
        except ImportError:
            raise ImportError("DeepSpeed 后端需要安装: pip install deepspeed")

        ds_config = get_deepspeed_config(
            self.dist_config,
            lr=self.learning_rate,
            train_batch_size=self.batch_size * self.world_size,
            grad_accum=self.dist_config.gradient_accumulation_steps,
        )

        # 更新 scheduler 的总步数
        ds_config["scheduler"]["params"]["total_num_steps"] = total_steps

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )

        running_loss = 0.0

        for epoch in range(self.epochs):
            if self.rank == 0:
                logger.info(f"===== Epoch {epoch + 1}/{self.epochs} (DeepSpeed) =====")

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self._forward_engine(model_engine, input_ids, labels)
                loss = outputs["loss"]

                model_engine.backward(loss)
                model_engine.step()

                self.global_step += 1
                running_loss += loss.item()

                if self.global_step % self.log_interval == 0 and self.rank == 0:
                    avg_loss = running_loss / self.log_interval
                    logger.info(f"  Step {self.global_step} | Loss: {avg_loss:.4f}")
                    running_loss = 0.0

                if self.global_step % self.save_interval == 0:
                    if self.rank == 0:
                        ckpt_dir = os.path.join(
                            self.output_dir, f"ds_checkpoint_{self.global_step}"
                        )
                        model_engine.save_checkpoint(ckpt_dir)
                        self.tokenizer.save(
                            os.path.join(self.output_dir, "tokenizer.json")
                        )

                if self.max_steps and self.global_step >= self.max_steps:
                    break

            if self.max_steps and self.global_step >= self.max_steps:
                break

        if self.rank == 0:
            logger.info(f"DeepSpeed 训练完成! 总步数: {self.global_step}")

    def _forward(self, input_ids, labels):
        """
        标准前向传播 (适配不同模型接口)。

        各模型 forward 签名不同:
          - APTModel:   (src_tokens, tgt_tokens) -> logits
          - GPT5Model:  (input_ids) -> logits
          - GPT4oModel: (text_ids) -> logits
          - Claude4Model: (input_ids) -> logits
          - GPTo3Model: (text_ids) -> logits

        所有模型均返回 logits, 需要在此处统一计算交叉熵损失。
        """
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        model_class = type(base_model).__name__

        if model_class in ("GPT4oModel", "GPTo3Model"):
            logits = base_model(text_ids=input_ids)
        elif model_class == "GPT5Model":
            logits = base_model(input_ids=input_ids)
        elif model_class == "Claude4Model":
            logits = base_model(input_ids=input_ids)
        else:
            # APTModel / APTModel-Lite: (src_tokens, tgt_tokens)
            output = base_model(src_tokens=input_ids, tgt_tokens=labels)
            if isinstance(output, dict) and "loss" in output:
                return output
            logits = output

        # 如果是 tuple (有些模型返回 (logits, info_dict))
        if isinstance(logits, tuple):
            logits = logits[0]

        # 从 logits 计算交叉熵损失
        # logits: [B, T, V], labels: [B, T]
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def _forward_engine(self, engine, input_ids, labels):
        """DeepSpeed engine 前向传播 (适配不同模型接口)"""
        # DeepSpeed engine 内部包装了模型, 用 engine.module 获取原始模型类名
        inner = engine.module if hasattr(engine, "module") else engine
        model_class = type(inner).__name__

        if model_class in ("GPT4oModel", "GPTo3Model"):
            logits = engine(text_ids=input_ids)
        elif model_class in ("GPT5Model", "Claude4Model"):
            logits = engine(input_ids=input_ids)
        else:
            output = engine(src_tokens=input_ids, tgt_tokens=labels)
            if isinstance(output, dict) and "loss" in output:
                return output
            logits = output

        if isinstance(logits, tuple):
            logits = logits[0]

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


# ============================================================================
# 第五部分: 分词器预训练采样器
# ============================================================================

def collect_tokenizer_corpus(
    hlbd_path: Optional[str] = None,
    sample_size: int = 100000,
    output_path: Optional[str] = None,
) -> List[str]:
    """
    从各数据源收集用于训练分词器的语料样本。

    分词器训练不需要全量数据, 只需要有代表性的子集。
    这里从 C4/mC4/FineWeb 各采 sample_size/3 条,
    加上 HLBD 全量, 作为分词器的训练语料。

    Args:
        hlbd_path: HLBD 文件路径
        sample_size: 总采样数
        output_path: (可选) 保存采样结果到文件

    Returns:
        采样文本列表
    """
    texts = []
    per_source = sample_size // 3

    # C4 en
    try:
        from datasets import load_dataset
        logger.info(f"从 C4 (en) 采样 {per_source} 条...")
        ds = load_dataset("allenai/c4", "en", streaming=True, split="train",
                          trust_remote_code=True)
        count = 0
        for example in ds:
            if "text" in example and len(example["text"].strip()) > 50:
                texts.append(example["text"][:2000])  # 截断长文档
                count += 1
                if count >= per_source:
                    break
        logger.info(f"  C4 采样完成: {count} 条")
    except Exception as e:
        logger.warning(f"C4 采样失败: {e}")

    # mC4 zh
    try:
        from datasets import load_dataset
        logger.info(f"从 mC4 (zh) 采样 {per_source} 条...")
        ds = load_dataset("mc4", "zh", streaming=True, split="train",
                          trust_remote_code=True)
        count = 0
        for example in ds:
            if "text" in example and len(example["text"].strip()) > 50:
                texts.append(example["text"][:2000])
                count += 1
                if count >= per_source:
                    break
        logger.info(f"  mC4 (zh) 采样完成: {count} 条")
    except Exception as e:
        logger.warning(f"mC4 采样失败: {e}")

    # FineWeb
    try:
        from datasets import load_dataset
        logger.info(f"从 FineWeb 采样 {per_source} 条...")
        ds = load_dataset("HuggingFaceFW/fineweb", streaming=True, split="train",
                          trust_remote_code=True)
        count = 0
        for example in ds:
            if "text" in example and len(example["text"].strip()) > 50:
                texts.append(example["text"][:2000])
                count += 1
                if count >= per_source:
                    break
        logger.info(f"  FineWeb 采样完成: {count} 条")
    except Exception as e:
        logger.warning(f"FineWeb 采样失败: {e}")

    # HLBD (全量)
    if hlbd_path and os.path.exists(hlbd_path):
        try:
            from apt.core.data.hlbd.hlbd_adapter import HLBDDataProcessor
            processor = HLBDDataProcessor(data_path=hlbd_path)
            processor.process_data(include_multilingual=True, include_separate_levels=True)
            hlbd_texts = processor.get_training_texts()
            texts.extend(hlbd_texts)
            logger.info(f"  HLBD 加载完成: {len(hlbd_texts)} 条")
        except Exception as e:
            logger.warning(f"HLBD 加载失败: {e}")

    logger.info(f"分词器语料总计: {len(texts)} 条")

    # 可选: 保存到文件
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for text in texts:
                # 一行一条, 去掉换行符
                f.write(text.replace("\n", " ").strip() + "\n")
        logger.info(f"分词器语料已保存到: {output_path}")

    return texts


# ============================================================================
# 第六部分: CLI 入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="APT 速食预训练 (QuickCook): C4 + mC4 + FineWeb + HLBD 混合预训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- 输出 ---
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")

    # --- 数据 ---
    parser.add_argument("--hlbd-path", type=str, default=None,
                        help="HLBD 数据集路径 (可选)")
    parser.add_argument("--datasets-dir", type=str, default=None,
                        help="数据集总目录 (放 curriculum.json 用于持续课程学习)")
    parser.add_argument("--no-c4", action="store_true", help="不使用 C4 (en)")
    parser.add_argument("--no-mc4", action="store_true",
                        help="不使用 Chinese-C4 (中文, 替代旧版 mC4)")
    parser.add_argument("--no-fineweb", action="store_true", help="不使用 FineWeb")
    parser.add_argument("--use-wiki", action="store_true",
                        help="启用 Wikipedia 数据集 (omarkamali/wikipedia-monthly)")
    parser.add_argument("--use-arxiv", action="store_true",
                        help="启用 arXiv 论文数据集 (RedPajama-1T/arxiv)")
    parser.add_argument("--use-code", action="store_true",
                        help="启用 GitHub Code 数据集 (codeparrot/github-code)")
    parser.add_argument("--code-languages", type=str, nargs="+",
                        default=["Python", "JavaScript", "TypeScript",
                                 "Java", "C", "C++", "Go", "Rust"],
                        help="Code 数据集语言过滤 (默认: Python JS TS Java C C++ Go Rust)")
    parser.add_argument("--weight-c4", type=float, default=0.35,
                        help="C4 (en) 采样权重 (默认 0.35)")
    parser.add_argument("--weight-mc4", type=float, default=0.20,
                        help="Chinese-C4 采样权重 (默认 0.20)")
    parser.add_argument("--weight-fineweb", type=float, default=0.25,
                        help="FineWeb 采样权重 (默认 0.25)")
    parser.add_argument("--weight-hlbd", type=float, default=0.10,
                        help="HLBD 采样权重 (默认 0.10)")
    parser.add_argument("--weight-wiki", type=float, default=0.10,
                        help="Wikipedia 采样权重 (默认 0.10, 需 --use-wiki)")
    parser.add_argument("--weight-arxiv", type=float, default=0.08,
                        help="arXiv 采样权重 (默认 0.08, 需 --use-arxiv)")
    parser.add_argument("--weight-code", type=float, default=0.05,
                        help="GitHub Code 采样权重 (默认 0.05, 需 --use-code)")
    # 顺序遍历 vs 随机交替
    parser.add_argument("--wiki-mode", type=str, default="sequential",
                        choices=["sequential", "interleaved"],
                        help="Wiki 数据加载模式: sequential=顺序完整遍历, "
                             "interleaved=随机交替采样 (默认 sequential)")
    parser.add_argument("--arxiv-mode", type=str, default="sequential",
                        choices=["sequential", "interleaved"],
                        help="arXiv 数据加载模式 (默认 sequential)")
    parser.add_argument("--code-mode", type=str, default="sequential",
                        choices=["sequential", "interleaved"],
                        help="Code 数据加载模式 (默认 sequential)")

    # --- 分词器 ---
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="已有分词器路径 (JSON). 不指定则从语料训练新的。")
    parser.add_argument("--vocab-size", type=int, default=65536,
                        help="分词器词表大小 (默认 65536)")
    parser.add_argument("--tokenizer-sample-size", type=int, default=100000,
                        help="分词器训练时采样文档数 (默认 100000)")

    # --- 模型 ---
    parser.add_argument("--model-arch", type=str, default="apt",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="模型架构 (默认 apt). "
                             "可选: apt, apt-lite, gpt4o, gpt5, claude4, gpto3")
    parser.add_argument("--d-model", type=int, default=768, help="模型维度")
    parser.add_argument("--num-heads", type=int, default=12, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=12, help="层数")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复 (检查点 .pt 文件路径)")

    # --- 训练超参 ---
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--max-steps", type=int, default=None, help="最大训练步数")
    parser.add_argument("--batch-size", type=int, default=8, help="每 GPU 批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--log-interval", type=int, default=10, help="日志间隔 (步)")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="保存间隔 (步)")

    # --- 分布式 ---
    parser.add_argument("--distributed-backend", type=str, default="ddp",
                        choices=["ddp", "deepspeed", "fsdp"],
                        help="分布式后端 (默认 ddp)")
    parser.add_argument("--zero-stage", type=int, default=2, choices=[1, 2, 3],
                        help="DeepSpeed ZeRO 阶段 (默认 2)")
    parser.add_argument("--no-distributed", action="store_true",
                        help="禁用分布式 (单机调试)")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="禁用混合精度")
    parser.add_argument("--mixed-precision-dtype", type=str, default="bf16",
                        choices=["bf16", "fp16"], help="混合精度类型")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="启用梯度检查点 (省显存)")

    # --- 缓存 & 杂项 ---
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF 缓存目录 (默认: $WORK/.cache/huggingface). "
                             "训练集群上应指向 work 目录而非 home。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细日志")

    # --- 虚拟 GPU 加速 (可选) ---
    vgpu_group = parser.add_argument_group("虚拟 GPU 加速 (可选)")

    # Virtual VRAM: 激活值 offload 到 CPU, 降低显存峰值
    vgpu_group.add_argument("--use-virtual-vram", action="store_true",
                            help="启用 Virtual VRAM (激活值自动 offload 到 CPU, 降低显存峰值)")
    vgpu_group.add_argument("--vram-min-tensor-bytes", type=int, default=1 << 20,
                            help="Virtual VRAM: 仅 offload >= 此大小的张量 (默认 1MB)")
    vgpu_group.add_argument("--vram-verbose", action="store_true",
                            help="Virtual VRAM: 打印每次 offload/restore 的详细日志")

    # Virtual Blackwell: 对 nn.Linear 做脉冲式量化感知优化
    vgpu_group.add_argument("--use-virtual-blackwell", action="store_true",
                            help="启用 Virtual Blackwell (脉冲式 INT8 量化感知 + scale cache)")
    vgpu_group.add_argument("--vb-pulse-interval", type=int, default=20,
                            help="Virtual Blackwell: 脉冲间隔 (默认 20 步)")
    vgpu_group.add_argument("--vb-use-fake-int8", action="store_true",
                            help="Virtual Blackwell: 启用 fake INT8 量化 (默认关闭, 影响速度)")
    vgpu_group.add_argument("--vb-gate-projected-mode", action="store_true",
                            help="Virtual Blackwell: 门投影模式, 非脉冲步零开销")

    # Virtual A100: 三层虚拟显存 + OPU 冷热分层管理
    vgpu_group.add_argument("--use-virtual-a100", action="store_true",
                            help="启用 Virtual A100 (三层虚拟显存 + OPU 自适应冷热分层)")
    vgpu_group.add_argument("--va100-vram-budget-gb", type=float, default=7.5,
                            help="Virtual A100: GPU 热层预算 (GB, 默认 7.5)")
    vgpu_group.add_argument("--va100-cpu-budget-gb", type=float, default=16.0,
                            help="Virtual A100: CPU 温层预算 (GB, 默认 16.0)")
    vgpu_group.add_argument("--va100-prefetch-window", type=int, default=2,
                            help="Virtual A100: 预取窗口 (提前搬运几层, 默认 2)")

    # DeepSpeed 会注入自己的 argparse 参数
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="(DeepSpeed 自动注入)")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- 缓存目录 (在任何 HF import 之前设置) ---
    setup_cache_dir(args.cache_dir)

    # --- 日志 ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- 分布式初始化 ---
    dist_config = DistributedConfig(
        backend=args.distributed_backend if not args.no_distributed else "ddp",
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_mixed_precision=not args.no_mixed_precision,
        mixed_precision_dtype=args.mixed_precision_dtype,
    )

    if args.no_distributed:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        rank, world_size, local_rank, device = setup_distributed(dist_config)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("APT 速食预训练 (QuickCook)")
        logger.info("=" * 60)
        logger.info(f"模型架构: {args.model_arch}")
        logger.info(f"分布式: backend={dist_config.backend}, "
                     f"world_size={world_size}, rank={rank}")
        logger.info(f"设备: {device}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"HF 缓存: {os.environ.get('HF_HOME', '(默认)')}")

    # --- 随机种子 ---
    torch.manual_seed(args.seed + rank)

    # --- 分词器 ---
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        if rank == 0:
            logger.info(f"加载已有分词器: {args.tokenizer_path}")
        tokenizer = AdaptiveBPETokenizer.load(args.tokenizer_path)
    else:
        if rank == 0:
            logger.info("未找到分词器, 从混合语料训练新分词器...")
            corpus_path = os.path.join(args.output_dir, "tokenizer_corpus.txt")
            texts = collect_tokenizer_corpus(
                hlbd_path=args.hlbd_path,
                sample_size=args.tokenizer_sample_size,
                output_path=corpus_path,
            )

            tokenizer = AdaptiveBPETokenizer.train_from_iterator(
                iter(texts),
                vocab_size=args.vocab_size,
            )
            tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))
            logger.info(f"分词器训练完成, 词表: {tokenizer.vocab_size}")
        else:
            # 非 rank 0 进程等待分词器就绪
            if world_size > 1:
                dist.barrier()

        # 所有进程加载分词器
        tok_path = os.path.join(args.output_dir, "tokenizer.json")
        if rank == 0 and world_size > 1:
            dist.barrier()  # rank 0 写完后放行

        if rank != 0 or (rank == 0 and args.tokenizer_path is None):
            if os.path.exists(tok_path):
                tokenizer = AdaptiveBPETokenizer.load(tok_path)
            # 如果上面已经训练好了 tokenizer, rank 0 不需要再加载

    if rank == 0:
        logger.info(f"分词器词表大小: {tokenizer.vocab_size}")

    # --- 模型 ---
    model = create_model(
        arch=args.model_arch,
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )

    if args.resume:
        if rank == 0:
            logger.info(f"从检查点恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"模型参数量: {total_params:,}")

    # 梯度检查点
    if dist_config.use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif rank == 0:
            logger.warning("模型不支持 gradient_checkpointing_enable()")

    # ================================================================
    # 可选: 虚拟 GPU 加速组件初始化 (在分布式包装之前)
    # ================================================================
    vram_cfg = None
    vb_adapter = None
    va100_tier = None
    va100_signal = None

    # --- Virtual Blackwell: 替换 nn.Linear 为脉冲式量化感知版本 ---
    # 必须在 DDP/FSDP wrap 之前, 因为它修改模型结构
    if args.use_virtual_blackwell:
        if not _VIRTUAL_BLACKWELL_AVAILABLE:
            if rank == 0:
                logger.warning(
                    "已指定 --use-virtual-blackwell 但 vb_integration 不可用, "
                    "请确认 apt.vgpu.runtime.vb_integration 可导入。跳过。"
                )
        else:
            vb_config = VBConfigV64(
                pulse_interval=args.vb_pulse_interval,
                use_fake_int8=args.vb_use_fake_int8,
                gate_projected_mode=args.vb_gate_projected_mode,
            )
            model, vb_adapter = apply_virtual_blackwell_v64(model, vb_config)
            if rank == 0:
                replaced = getattr(model, "_vb_replaced_linears", "?")
                logger.info(
                    f"[Virtual Blackwell] 已替换 {replaced} 个 nn.Linear, "
                    f"pulse_interval={args.vb_pulse_interval}, "
                    f"fake_int8={args.vb_use_fake_int8}"
                )

    # --- Virtual VRAM: 激活值 offload 配置 (不修改模型, 训练时用 context) ---
    if args.use_virtual_vram:
        if not _VIRTUAL_VRAM_AVAILABLE:
            if rank == 0:
                logger.warning(
                    "已指定 --use-virtual-vram 但 virtual_vram 不可用, "
                    "请确认 apt.vgpu.runtime.virtual_vram 可导入。跳过。"
                )
        else:
            vram_cfg = VirtualVRAMConfig(
                enabled=True,
                min_tensor_bytes=args.vram_min_tensor_bytes,
                verbose=args.vram_verbose,
            )
            if rank == 0:
                min_mb = args.vram_min_tensor_bytes / (1 << 20)
                logger.info(
                    f"[Virtual VRAM] 已配置: min_tensor={min_mb:.0f}MB, "
                    f"verbose={args.vram_verbose}"
                )

    # --- Virtual A100: 三层虚拟显存 + 信号采集 ---
    if args.use_virtual_a100:
        if not _VIRTUAL_A100_AVAILABLE:
            if rank == 0:
                logger.warning(
                    "已指定 --use-virtual-a100 但 virtual_a100 不可用, "
                    "请确认 va100/virtual_a100.py 可导入。跳过。"
                )
        else:
            hot_bytes = int(args.va100_vram_budget_gb * 1e9 * 0.6)
            warm_bytes = int(args.va100_cpu_budget_gb * 1e9 * 0.3)
            cold_bytes = int(100e9)
            va100_tier = VirtualVRAMBackend(hot_bytes, warm_bytes, cold_bytes)
            va100_signal = VA100SignalCollector()
            if rank == 0:
                logger.info(
                    f"[Virtual A100] 三层显存: "
                    f"hot={args.va100_vram_budget_gb*0.6:.1f}GB, "
                    f"warm={args.va100_cpu_budget_gb*0.3:.1f}GB, "
                    f"prefetch_window={args.va100_prefetch_window}"
                )

    # 分布式包装 (在 VB 替换之后)
    model = wrap_model_distributed(model, device, dist_config, rank, world_size)

    # --- 数据集 ---
    weights = {
        "c4_en": args.weight_c4,
        "chinese_c4": args.weight_mc4,
        "fineweb": args.weight_fineweb,
        "hlbd": args.weight_hlbd,
        "wiki": args.weight_wiki,
        "arxiv": args.weight_arxiv,
        "code": args.weight_code,
    }

    train_dataset = InterleavedStreamDataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        weights=weights,
        hlbd_path=args.hlbd_path,
        datasets_dir=args.datasets_dir,
        use_c4=not args.no_c4,
        use_mc4_zh=not args.no_mc4,
        use_fineweb=not args.no_fineweb,
        use_wiki=args.use_wiki,
        use_arxiv=args.use_arxiv,
        use_code=args.use_code,
        code_languages=args.code_languages,
        wiki_mode=args.wiki_mode,
        arxiv_mode=args.arxiv_mode,
        code_mode=args.code_mode,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )

    # --- 训练 ---
    trainer = QuickCookTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        dist_config=dist_config,
        rank=rank,
        world_size=world_size,
        device=device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        max_steps=args.max_steps,
        epochs=args.epochs,
        # 可选: 虚拟 GPU 加速
        virtual_vram_config=vram_cfg,
        vb_adapter=vb_adapter,
        va100_tier_manager=va100_tier,
        va100_signal_collector=va100_signal,
        # 模型配置 (保存到 checkpoint)
        model_config={
            "arch": args.model_arch,
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_seq_len": args.max_seq_len,
        },
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        if rank == 0:
            logger.warning("训练被用户中断")
    finally:
        if not args.no_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
