#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkpoint management for APT model"""

import os
import json
from apt_model.utils.fake_torch import get_torch
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
    from apt_model.modeling.apt_model import APTLargeModel
    from apt_model.utils import get_device

    if device is None:
        device = get_device()

    # 检测路径类型
    if os.path.isfile(path) and path.endswith('.pt'):
        # 单文件格式 (HLBD checkpoint)
        return _load_single_file_checkpoint(path, device)
    elif os.path.isdir(path):
        # 目录格式
        return _load_directory_checkpoint(path, device)
    else:
        raise ValueError(f"不支持的模型路径格式: {path}")


def _load_directory_checkpoint(path, device):
    """加载目录格式的模型"""
    from transformers import GPT2Tokenizer
    from apt.core.config.apt_config import APTConfig
    from apt_model.modeling.apt_model import APTLargeModel

    # 加载配置
    config_path = os.path.join(path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = APTConfig.from_dict(config_dict)

    # 创建模型
    model = APTLargeModel(config)

    # 加载模型权重
    model_path = os.path.join(path, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 加载分词器
    tokenizer_path = os.path.join(path, "tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, config


def _load_single_file_checkpoint(path, device):
    """加载单文件 checkpoint (HLBD 格式)"""
    from apt_model.modeling.apt_model import APTModel, APTModelConfiguration

    # 加载 checkpoint
    checkpoint = torch.load(path, map_location=device)

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

    # 创建模型并加载权重
    model = APTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
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