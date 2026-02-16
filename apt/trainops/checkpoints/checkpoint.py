#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkpoint management for APT model"""

import os
import json
from apt.core.fake_torch import get_torch
torch = get_torch()
from datetime import datetime

def save_model(model, tokenizer, path, config=None):
    """
    保存模型、分词器和配置
    
    参数:
        model: 模型
        tokenizer: 分词器
        path: 保存路径
        config: 模型配置
    """
    os.makedirs(path, exist_ok=True)
    
    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    
    # 保存分词器
    tokenizer_path = os.path.join(path, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    
    # 保存配置
    if config:
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

def load_model(path, device=None):
    """
    加载模型、分词器和配置

    支持两种格式:
    1. 目录格式: 包含 config.json, model.pt, tokenizer/ 等文件
    2. 单文件格式: .pt checkpoint 文件 (HLBD 格式)

    参数:
        path: 模型路径（目录或 .pt 文件）
        device: 计算设备

    返回:
        tuple: (model, tokenizer, config)
    """
    from transformers import GPT2Tokenizer
    from apt.core.config.apt_config import APTConfig
    from apt.model.architectures.apt_model import APTLargeModel
    from apt.core import get_device

    if device is None:
        device = get_device()

    # 路径解析：如果是相对路径，尝试多个可能的位置
    if not os.path.isabs(path):
        possible_paths = [
            path,  # 当前目录
            os.path.abspath(path),  # 相对于当前工作目录
            # 相对于项目根目录（假设 checkpoint.py 在 apt/trainops/checkpoints/）
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), path),
        ]

        # 找到第一个存在的路径
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path is None:
            raise FileNotFoundError(
                f"模型路径不存在: {path}\n\n"
                f"已尝试的路径:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                f"\n\n当前工作目录: {os.getcwd()}\n\n"
                f"建议:\n"
                f"  1. 使用绝对路径: --model-path {os.path.abspath(path)}\n"
                f"  2. 确保模型文件存在: dir {path} (Windows) 或 ls {path} (Linux)\n"
                f"  3. 训练新模型: python -m apt_model train"
            )

        path = found_path

    # 检测路径类型
    if os.path.isfile(path) and path.endswith('.pt'):
        # 探测 checkpoint 格式
        checkpoint = torch.load(path, map_location=device)
        if checkpoint.get("format") == "quickcook" or (
            "model_config" in checkpoint and "tokenizer_char_to_id" not in checkpoint
        ):
            return _load_quickcook_checkpoint(checkpoint, path, device)
        else:
            # HLBD 格式
            return _load_single_file_checkpoint_from_dict(checkpoint, path, device)
    elif os.path.isdir(path):
        # 目录格式
        return _load_directory_checkpoint(path, device)
    else:
        raise ValueError(f"不支持的模型路径格式: {path}")


def _load_directory_checkpoint(path, device):
    """加载目录格式的模型"""
    from transformers import GPT2Tokenizer
    from apt.core.config.apt_config import APTConfig
    from apt.model.architectures.apt_model import APTLargeModel

    # 加载配置
    config_path = os.path.join(path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = APTConfig.from_dict(config_dict)

    # 创建模型
    model = APTLargeModel(config)

    # 加载模型权重（兼容旧版本 checkpoint）
    model_path = os.path.join(path, "model.pt")
    checkpoint_state_dict = torch.load(model_path, map_location=device)

    # 尝试严格加载，如果失败则使用兼容模式
    try:
        model.load_state_dict(checkpoint_state_dict, strict=True)
    except RuntimeError as e:
        # 如果是形状不匹配或缺失参数，使用兼容模式
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"检测到 checkpoint 兼容性问题，使用兼容模式加载...")

        # 获取当前模型的 state_dict
        model_state_dict = model.state_dict()

        # 过滤掉形状不匹配的参数
        filtered_state_dict = {}
        skipped_keys = []
        shape_mismatch_keys = []

        for key, checkpoint_param in checkpoint_state_dict.items():
            if key in model_state_dict:
                model_param = model_state_dict[key]
                # 检查形状是否匹配
                if checkpoint_param.shape == model_param.shape:
                    filtered_state_dict[key] = checkpoint_param
                else:
                    shape_mismatch_keys.append(
                        f"{key}: checkpoint {checkpoint_param.shape} vs model {model_param.shape}"
                    )
            else:
                skipped_keys.append(key)

        # 加载过滤后的参数
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

        # 记录兼容性信息
        if shape_mismatch_keys:
            logger.warning(f"跳过 {len(shape_mismatch_keys)} 个形状不匹配的参数（将使用模型默认初始化）:")
            for info in shape_mismatch_keys[:3]:
                logger.warning(f"  - {info}")
            if len(shape_mismatch_keys) > 3:
                logger.warning(f"  ... 还有 {len(shape_mismatch_keys) - 3} 个")

        if skipped_keys:
            logger.info(f"跳过 {len(skipped_keys)} 个当前模型中不存在的参数")

        if missing_keys:
            logger.info(f"使用默认初始化的参数: {len(missing_keys)} 个")

        logger.info(f"✓ 兼容模式加载完成，成功加载 {len(filtered_state_dict)} 个参数")

    model = model.to(device)

    # 加载分词器（兼容不完整的 tokenizer 目录）
    tokenizer_path = os.path.join(path, "tokenizer")
    tokenizer = None

    # 尝试加载 GPT2Tokenizer
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        except (TypeError, FileNotFoundError, OSError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"无法加载 GPT2Tokenizer: {e}")
            logger.warning("尝试使用简单的基于 vocab.json 的 tokenizer...")

            # 尝试从 vocab.json 创建简单 tokenizer
            vocab_file = os.path.join(tokenizer_path, "vocab.json")
            if os.path.exists(vocab_file):
                try:
                    with open(vocab_file, 'r', encoding='utf-8') as f:
                        vocab = json.load(f)

                    # 创建简单的 tokenizer 包装器
                    class SimpleVocabTokenizer:
                        def __init__(self, vocab_dict):
                            self.vocab = vocab_dict
                            self.id_to_token = {v: k for k, v in vocab_dict.items()}
                            self.vocab_size = len(vocab_dict)
                            self.pad_token_id = vocab_dict.get('<|pad|>', vocab_dict.get('[PAD]', 0))
                            self.eos_token_id = vocab_dict.get('<|endoftext|>', vocab_dict.get('[SEP]', 1))
                            self.bos_token_id = vocab_dict.get('<|startoftext|>', vocab_dict.get('[CLS]', 2))

                        def encode(self, text, **kwargs):
                            # 简单的字符级编码
                            tokens = []
                            for char in text:
                                tokens.append(self.vocab.get(char, self.vocab.get('<|unk|>', 3)))
                            return tokens

                        def decode(self, token_ids, **kwargs):
                            # 简单的解码
                            chars = []
                            for token_id in token_ids:
                                if token_id in self.id_to_token:
                                    chars.append(self.id_to_token[token_id])
                            return ''.join(chars)

                        def __call__(self, text, **kwargs):
                            return {'input_ids': self.encode(text)}

                    tokenizer = SimpleVocabTokenizer(vocab)
                    logger.info(f"✓ 使用简单 vocab tokenizer (词汇表大小: {tokenizer.vocab_size})")

                except Exception as e2:
                    logger.error(f"无法创建简单 tokenizer: {e2}")

    if tokenizer is None:
        raise RuntimeError(
            f"无法加载 tokenizer from {tokenizer_path}\n"
            "请确保 tokenizer 目录包含以下文件之一:\n"
            "  - vocab.json + merges.txt (GPT2Tokenizer)\n"
            "  - vocab.json (SimpleVocabTokenizer)"
        )

    return model, tokenizer, config


def _load_quickcook_checkpoint(checkpoint, path, device):
    """加载 QuickCook 格式的 checkpoint (.pt + tokenizer.json)"""
    import logging
    logger = logging.getLogger(__name__)

    model_config = checkpoint.get("model_config", {})
    if not model_config:
        raise ValueError(
            f"QuickCook checkpoint 缺少 model_config 字段: {path}\n"
            "此 checkpoint 可能来自旧版本的 pretrain_quickcook.py。\n"
            "请使用 --resume 参数在 pretrain_quickcook.py 中恢复训练，"
            "而不是通过 load_model() 加载。"
        )

    # 根据 model_config 创建模型
    arch = model_config["arch"]
    vocab_size = model_config["vocab_size"]
    d_model = model_config["d_model"]
    num_heads = model_config["num_heads"]
    num_layers = model_config["num_layers"]
    max_seq_len = model_config["max_seq_len"]

    # 动态导入 create_model (避免循环依赖)
    from apt.trainops.scripts.pretrain_quickcook import create_model
    model = create_model(
        arch=arch,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    logger.info(
        f"QuickCook checkpoint 已加载: arch={arch}, "
        f"d_model={d_model}, layers={num_layers}, step={checkpoint.get('global_step', '?')}"
    )

    # 加载分词器: tokenizer.json 在同目录下
    tokenizer = None
    checkpoint_dir = os.path.dirname(os.path.abspath(path))
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")

    if os.path.exists(tok_path):
        # 使用 tokenizers 库 (HuggingFace) 加载 BPE tokenizer
        try:
            from tokenizers import Tokenizer as HFTokenizer

            class SimpleBPETokenizer:
                """轻量 BPE 分词器包装 (用于推理, 不依赖 pretrain_quickcook)"""
                def __init__(self, backend: "HFTokenizer"):
                    self.backend = backend
                    self.vocab_size = backend.get_vocab_size()
                    self.pad_token_id = 0
                    self.eos_token_id = backend.token_to_id("</s>") if backend.token_to_id("</s>") is not None else 0
                    self.bos_token_id = backend.token_to_id("<s>") if backend.token_to_id("<s>") is not None else 0

                def encode(self, text, return_tensors=None):
                    ids = self.backend.encode(text).ids
                    if return_tensors == "pt":
                        return torch.tensor([ids])
                    return ids

                def decode(self, ids, skip_special_tokens=False):
                    if hasattr(ids, 'tolist'):
                        ids = ids.tolist()
                    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                        ids = ids[0]
                    return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

                def __call__(self, text, max_length=64, padding='max_length',
                             truncation=True, return_tensors='pt'):
                    ids = self.encode(text)
                    if truncation and len(ids) > max_length:
                        ids = ids[:max_length]
                    if padding == 'max_length':
                        ids = ids + [self.pad_token_id] * (max_length - len(ids))
                    if return_tensors == 'pt':
                        return {'input_ids': torch.tensor([ids])}
                    return {'input_ids': ids}

            backend = HFTokenizer.from_file(tok_path)
            tokenizer = SimpleBPETokenizer(backend)
            logger.info(f"BPE 分词器已加载: vocab_size={tokenizer.vocab_size}")

        except ImportError:
            logger.warning("tokenizers 库不可用, 无法加载 BPE 分词器")

    if tokenizer is None:
        raise RuntimeError(
            f"无法加载 QuickCook 分词器: {tok_path}\n"
            "请确保 tokenizers 库已安装: pip install tokenizers"
        )

    return model, tokenizer, model_config


def _load_single_file_checkpoint_from_dict(checkpoint, path, device):
    """加载单文件 checkpoint (HLBD 格式), 从已加载的 dict"""
    from apt.model.architectures.apt_model import APTModel, APTModelConfiguration

    # 重建 tokenizer (SimpleCharTokenizer_BACKUP)
    class SimpleCharTokenizer:
        """简单的字符级分词器 (兼容 HLBD)"""
        def __init__(self, char_to_id, id_to_char, next_id, vocab_size):
            self.char_to_id = char_to_id
            self.id_to_char = id_to_char
            self.next_id = next_id
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.unk_token_id = 1
            self.bos_token_id = 2
            self.eos_token_id = 3

        def encode(self, text, return_tensors=None):
            """编码文本"""
            ids = [self.bos_token_id]
            for char in text:
                ids.append(self.char_to_id.get(char, self.unk_token_id))
            ids.append(self.eos_token_id)

            if return_tensors == 'pt':
                return torch.tensor([ids])
            return ids

        def decode(self, ids, skip_special_tokens=True):
            """解码ID序列"""
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()

            # 移除批次维度
            if isinstance(ids[0], list):
                ids = ids[0]

            chars = []
            for id in ids:
                if skip_special_tokens and id in [0, 1, 2, 3]:
                    continue
                char = self.id_to_char.get(id, '')
                if char and not char.startswith('['):
                    chars.append(char)

            return ''.join(chars)

        def __call__(self, text, max_length=64, padding='max_length', truncation=True, return_tensors='pt'):
            """分词接口"""
            ids = []
            for char in text:
                ids.append(self.char_to_id.get(char, self.unk_token_id))

            if truncation and len(ids) > max_length - 2:
                ids = ids[:max_length - 2]

            ids = [self.bos_token_id] + ids + [self.eos_token_id]

            if padding == 'max_length':
                while len(ids) < max_length:
                    ids.append(self.pad_token_id)

            if return_tensors == 'pt':
                return {'input_ids': torch.tensor([ids])}
            return {'input_ids': ids}

    tokenizer = SimpleCharTokenizer(
        char_to_id=checkpoint['tokenizer_char_to_id'],
        id_to_char=checkpoint['tokenizer_id_to_char'],
        next_id=checkpoint['tokenizer_next_id'],
        vocab_size=checkpoint['tokenizer_vocab_size']
    )

    # 重建配置
    config_dict = checkpoint['config']
    config = APTModelConfiguration(
        vocab_size=config_dict['vocab_size'],
        d_model=config_dict['d_model'],
        max_seq_len=config_dict['max_seq_len'],
        num_encoder_layers=config_dict['num_encoder_layers'],
        num_decoder_layers=config_dict['num_decoder_layers'],
        num_heads=config_dict['num_heads'],
        d_ff=config_dict['d_ff'],
        dropout=config_dict['dropout'],
        use_autopoietic=config_dict.get('use_autopoietic', True),
        use_dbc_dac=config_dict.get('use_dbc_dac', False),
    )

    # 创建模型并加载权重（使用兼容模式）
    model = APTModel(config)

    checkpoint_state_dict = checkpoint['model_state_dict']

    # 尝试严格加载，如果失败则使用兼容模式
    try:
        model.load_state_dict(checkpoint_state_dict, strict=True)
    except RuntimeError as e:
        # 检测兼容性问题，使用兼容模式
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"检测到 checkpoint 兼容性问题，使用兼容模式加载...")

        # 获取当前模型的 state_dict
        model_state_dict = model.state_dict()

        # 过滤掉形状不匹配的参数
        filtered_state_dict = {}
        skipped_keys = []
        shape_mismatch_keys = []

        for key, checkpoint_param in checkpoint_state_dict.items():
            if key in model_state_dict:
                model_param = model_state_dict[key]
                # 检查形状是否匹配
                if checkpoint_param.shape == model_param.shape:
                    filtered_state_dict[key] = checkpoint_param
                else:
                    shape_mismatch_keys.append(
                        f"{key}: checkpoint {checkpoint_param.shape} vs model {model_param.shape}"
                    )
            else:
                skipped_keys.append(key)

        # 加载过滤后的参数
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

        # 记录兼容性信息
        if shape_mismatch_keys:
            logger.warning(f"跳过 {len(shape_mismatch_keys)} 个形状不匹配的参数（将使用模型默认初始化）")
            if len(shape_mismatch_keys) <= 5:
                for info in shape_mismatch_keys:
                    logger.warning(f"  - {info}")
            else:
                for info in shape_mismatch_keys[:3]:
                    logger.warning(f"  - {info}")
                logger.warning(f"  ... 还有 {len(shape_mismatch_keys) - 3} 个")

        if skipped_keys:
            logger.info(f"跳过 {len(skipped_keys)} 个当前模型中不存在的参数")

        if missing_keys:
            logger.info(f"使用默认初始化的参数: {len(missing_keys)} 个")
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    logger.info(f"  - {key}")
            else:
                for key in missing_keys[:3]:
                    logger.info(f"  - {key}")
                logger.info(f"  ... 还有 {len(missing_keys) - 3} 个")

        logger.info(f"✓ 兼容模式加载完成，成功加载 {len(filtered_state_dict)} 个参数")

    model = model.to(device)

    return model, tokenizer, config

class CheckpointManager:
    """检查点管理器，用于保存和恢复训练状态"""
    
    def __init__(self, save_dir, model_name="apt_model", save_freq=1, logger=None):
        """
        初始化检查点管理器

        参数:
            save_dir (str): 保存目录
            model_name (str): 模型名称
            save_freq (int): 保存频率（每多少个epoch保存一次）
            logger (logging.Logger, optional): 日志记录器
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_freq = save_freq
        self.logger = logger

        # 创建保存目录
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

        # 创建临时目录（用于原子性保存）
        self.temp_dir = os.path.join(save_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """加载元数据"""
        metadata_path = os.path.join(self.save_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"加载元数据失败: {e}")
        
        # 创建新的元数据
        return {
            "model_name": self.model_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoints": [],
            "training_history": {},
            "config": {}
        }
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, global_step, 
                      loss_history, metrics=None, tokenizer=None, config=None, 
                      is_best=False):
        """
        保存检查点
        
        参数:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch (int): 当前epoch
            global_step (int): 全局步数
            loss_history (list): 损失历史
            metrics (dict, optional): 其他指标
            tokenizer: 分词器
            config: 配置
            is_best (bool): 是否为最佳模型
        
        返回:
            str: 检查点路径
        """
        # 创建检查点目录
        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 检查点文件名
        checkpoint_name = f"{self.model_name}_epoch{epoch}_step{global_step}"
        if is_best:
            checkpoint_name += "_best"
        checkpoint_filename = f"{checkpoint_name}.pt"
        final_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        # 临时文件路径（用于原子性保存）
        import tempfile
        temp_fd, temp_checkpoint_path = tempfile.mkstemp(
            suffix='.pt',
            prefix=f'{checkpoint_name}_',
            dir=self.temp_dir
        )
        os.close(temp_fd)  # 关闭文件描述符，让torch.save创建新文件

        try:
            # 保存检查点到临时文件
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss_history': loss_history,
                'metrics': metrics,
                'config': config.to_dict() if config else None,
            }

            torch.save(checkpoint, temp_checkpoint_path)

            # 验证文件已保存且不为空
            if not os.path.exists(temp_checkpoint_path):
                raise IOError(f"临时checkpoint文件保存失败: {temp_checkpoint_path}")

            file_size = os.path.getsize(temp_checkpoint_path)
            if file_size == 0:
                raise IOError(f"临时checkpoint文件为空: {temp_checkpoint_path}")

            # 原子性移动到最终位置（这一步要么成功要么失败，不会损坏已有文件）
            import shutil
            shutil.move(temp_checkpoint_path, final_checkpoint_path)

            if self.logger:
                self.logger.info(f"保存检查点: {final_checkpoint_path} ({file_size / 1024 / 1024:.2f} MB)")

            checkpoint_path = final_checkpoint_path

        except Exception as e:
            # 如果保存失败，清理临时文件
            if os.path.exists(temp_checkpoint_path):
                try:
                    os.remove(temp_checkpoint_path)
                except:
                    pass

            if self.logger:
                self.logger.error(f"保存检查点失败: {e}")
            raise
        
        # 更新元数据
        self.metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["checkpoints"].append({
            "path": checkpoint_path,
            "epoch": epoch,
            "global_step": global_step,
            "is_best": is_best,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        })
        
        # 保存元数据（也使用原子性保存）
        metadata_path = os.path.join(self.save_dir, "metadata.json")
        temp_metadata_fd, temp_metadata_path = tempfile.mkstemp(
            suffix='.json',
            prefix='metadata_',
            dir=self.temp_dir
        )
        try:
            with os.fdopen(temp_metadata_fd, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            shutil.move(temp_metadata_path, metadata_path)
        except Exception as e:
            if os.path.exists(temp_metadata_path):
                try:
                    os.remove(temp_metadata_path)
                except:
                    pass
            if self.logger:
                self.logger.warning(f"保存元数据失败: {e}")
        
        # 如果有分词器，也保存
        if tokenizer:
            tokenizer_path = os.path.join(self.save_dir, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_path)
        
        # 记录日志
        if self.logger:
            self.logger.info(f"保存检查点: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None, best=False, latest=False):
        """
        加载检查点
        
        参数:
            model: 模型
            optimizer (optional): 优化器
            scheduler (optional): 学习率调度器
            checkpoint_path (str, optional): 检查点路径
            best (bool): 是否加载最佳模型
            latest (bool): 是否加载最新检查点
        
        返回:
            tuple: (epoch, global_step, loss_history, metrics)
        """
        if checkpoint_path is None:
            # 如果未指定路径，尝试找到最佳或最新检查点
            if best:
                # 找到最佳检查点
                best_checkpoint = None
                for checkpoint in self.metadata["checkpoints"]:
                    if checkpoint.get("is_best", False):
                        best_checkpoint = checkpoint
                        break
                
                if best_checkpoint:
                    checkpoint_path = best_checkpoint["path"]
                else:
                    raise ValueError("找不到最佳检查点")
            
            elif latest:
                # 找到最新检查点
                if self.metadata["checkpoints"]:
                    checkpoint_path = self.metadata["checkpoints"][-1]["path"]
                else:
                    raise ValueError("找不到任何检查点")
            
            else:
                raise ValueError("必须指定检查点路径，或设置 best=True 或 latest=True")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 可选地加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 可选地加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 获取训练状态
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        loss_history = checkpoint.get('loss_history', [])
        metrics = checkpoint.get('metrics', {})
        
        # 记录日志
        if self.logger:
            self.logger.info(f"加载检查点: {checkpoint_path} (epoch: {epoch}, step: {global_step})")

        return epoch, global_step, loss_history, metrics

    def cleanup_temp(self, max_age_hours=24):
        """
        清理临时目录中的旧文件

        参数:
            max_age_hours (int): 删除超过多少小时的临时文件（默认24小时）
        """
        import time

        if not os.path.exists(self.temp_dir):
            return

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        cleaned_size = 0

        try:
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)

                # 跳过目录
                if os.path.isdir(file_path):
                    continue

                # 检查文件年龄
                file_age = current_time - os.path.getmtime(file_path)

                if file_age > max_age_seconds:
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_count += 1
                        cleaned_size += file_size

                        if self.logger:
                            self.logger.debug(f"删除过期临时文件: {filename} (年龄: {file_age/3600:.1f}小时)")
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"删除临时文件失败 {filename}: {e}")

            if cleaned_count > 0 and self.logger:
                self.logger.info(f"清理临时目录: 删除 {cleaned_count} 个文件, 释放 {cleaned_size/1024/1024:.2f} MB")

        except Exception as e:
            if self.logger:
                self.logger.error(f"清理临时目录失败: {e}")