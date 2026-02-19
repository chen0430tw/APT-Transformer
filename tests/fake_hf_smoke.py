#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fake_hf_smoke.py — 用假节点替换 HuggingFace 数据集，跑速食训练脚本的冒烟测试。

原理：
  在 `datasets.load_dataset` 被调用之前，用 unittest.mock.patch 把它替换成
  一个返回假 IterableDataset 的函数。假数据集无限循环产出带 "text" 字段的
  样本，格式与真实 C4/FineWeb 完全一致。

用法：
  python tests/fake_hf_smoke.py
"""

import sys
import os
import types
import itertools
import logging
from unittest.mock import patch

# --------------------------------------------------------------------------- #
# 路径：让 tests/ 目录能 import 项目根                                          #
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fake_hf_smoke")

# --------------------------------------------------------------------------- #
# 假 IterableDataset                                                            #
# --------------------------------------------------------------------------- #

# C4/FineWeb/mC4 都只用 "text" 字段；
# code 数据集用 "code" 字段，但本次只 mock HF 的 text 数据集。
_FAKE_TEXTS = [
    "The transformer architecture revolutionized natural language processing. "
    "Attention mechanisms allow models to focus on relevant parts of the input.",

    "深度学习模型在自然语言处理领域取得了突破性进展。"
    "大规模预训练语言模型可以完成翻译、摘要和问答等多种任务。",

    "In mathematics, a Fourier transform decomposes a function into its "
    "constituent frequencies. This is widely used in signal processing.",

    "量子计算利用量子叠加与纠缠特性，在特定问题上能够实现指数级加速。"
    "Shor算法可以在多项式时间内分解大整数。",

    "Python is a high-level programming language known for its simplicity. "
    "It supports multiple programming paradigms including procedural, "
    "object-oriented, and functional programming.",

    "机器学习中的梯度下降算法通过迭代更新参数来最小化损失函数。"
    "自适应学习率方法如Adam优化器在实践中表现更为稳定。",

    "The history of artificial intelligence dates back to the 1950s, "
    "when Alan Turing proposed the famous Turing Test. Since then, "
    "the field has seen remarkable progress.",

    "神经网络的反向传播算法利用链式法则计算梯度，"
    "使得深层网络的参数优化成为可能。",
] * 500   # 重复 500 次，产出足够多的样本


class FakeIterableDataset:
    """模拟 HuggingFace IterableDataset 的最小接口。"""

    def __init__(self, texts=None, field="text"):
        self._texts = texts or _FAKE_TEXTS
        self._field = field

    def __iter__(self):
        for t in itertools.cycle(self._texts):
            yield {self._field: t, "url": "fake://", "timestamp": "2024-01-01"}

    def shard(self, num_shards: int, index: int):
        """兼容 ds.shard() 调用（多进程分片）。"""
        texts = [t for i, t in enumerate(self._texts) if i % num_shards == index]
        return FakeIterableDataset(texts or self._texts, self._field)

    # datasets.IterableDataset 在 interleave_datasets 里会检查这些属性
    @property
    def n_shards(self):
        return 1

    def _ex_iterable(self):
        return None


def fake_load_dataset(dataset_name, *args, **kwargs):
    """替换 datasets.load_dataset 的假实现。"""
    logger.info(f"[FAKE] load_dataset('{dataset_name}', {args}, {kwargs})")
    field = "code" if "code" in dataset_name.lower() else "text"
    return FakeIterableDataset(field=field)


# --------------------------------------------------------------------------- #
# 替换 datasets 模块                                                            #
# --------------------------------------------------------------------------- #

def _patch_datasets():
    """
    在 datasets 被真正 import 之前注入假模块。
    这样即使 pretrain_quickcook 内部做 `from datasets import load_dataset`
    也会得到假实现。
    """
    try:
        import datasets as real_datasets
        # 已经 import 了——直接猴子补丁
        real_datasets.load_dataset = fake_load_dataset
        # 同时修复 interleave_datasets：简单地把多个 FakeIterableDataset 链在一起
        def fake_interleave(datasets_list, probabilities=None, seed=None,
                            stopping_strategy="first_exhausted"):
            logger.info(f"[FAKE] interleave_datasets({len(datasets_list)} datasets)")
            return FakeIterableDataset()
        real_datasets.interleave_datasets = fake_interleave
        logger.info("已猴子补丁 datasets.load_dataset + interleave_datasets")
    except ImportError:
        # datasets 未安装：创建假模块
        fake_mod = types.ModuleType("datasets")
        fake_mod.load_dataset = fake_load_dataset
        fake_mod.IterableDataset = FakeIterableDataset

        def fake_interleave(datasets_list, probabilities=None, seed=None,
                            stopping_strategy="first_exhausted"):
            return FakeIterableDataset()
        fake_mod.interleave_datasets = fake_interleave
        sys.modules["datasets"] = fake_mod
        logger.info("已注入假 datasets 模块（原模块未安装）")


# --------------------------------------------------------------------------- #
# 主入口                                                                        #
# --------------------------------------------------------------------------- #

def main():
    logger.info("=" * 60)
    logger.info("fake_hf_smoke: 开始冒烟测试")
    logger.info("=" * 60)

    # 1. 注入假数据集
    _patch_datasets()

    # 2. 构造 pretrain_quickcook 的 sys.argv（等同于命令行参数）
    output_dir = "/tmp/quickcook_smoke_output"
    os.makedirs(output_dir, exist_ok=True)

    sys.argv = [
        "pretrain_quickcook",
        "--output-dir", output_dir,
        "--no-distributed",          # 单进程，不启动 torch.dist
        "--no-mixed-precision",      # 避免 bf16 在 CPU 上报错
        "--epochs", "1",
        "--batch-size", "2",
        "--max-seq-len", "64",       # 短序列，速度快
        "--weight-c4", "0.5",
        "--no-mc4",                  # 跳过中文 C4，减少 mock 路径
        "--no-fineweb",              # 同上
        "--weight-hlbd", "0.0",      # 没有 HLBD 文件，跳过
        "--save-interval", "5",
        "--log-interval", "1",
        "--max-steps", "10",         # 只跑 10 步
        "--model-arch", "apt",
        "--d-model", "64",
        "--num-heads", "4",
        "--num-layers", "2",
        "--verbose",
    ]

    logger.info(f"命令行参数: {' '.join(sys.argv[1:])}")

    # 3. 导入并运行
    try:
        from apt.trainops.scripts.pretrain_quickcook import main as qc_main
        qc_main()
        logger.info("=" * 60)
        logger.info("冒烟测试通过！速食训练脚本可以正常运行。")
        logger.info("=" * 60)
    except SystemExit as e:
        if e.code == 0:
            logger.info("脚本正常退出（code=0）")
        else:
            logger.error(f"脚本异常退出（code={e.code}）")
            raise
    except Exception as e:
        logger.exception(f"冒烟测试失败: {e}")
        raise


if __name__ == "__main__":
    main()
