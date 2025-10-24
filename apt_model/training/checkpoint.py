#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkpoint management for APT model"""

import logging
import os
import json
import torch
from datetime import datetime

from apt_model.modeling.chinese_tokenizer_integration import (
    load_tokenizer as load_integrated_tokenizer,
    save_tokenizer as save_integrated_tokenizer,
)

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
    if not save_integrated_tokenizer(tokenizer, tokenizer_path):
        raise RuntimeError("保存分词器失败，无法写入检查点。")
    
    # 保存配置
    if config:
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

def load_model(path, device=None):
    """
    加载模型、分词器和配置
    
    参数:
        path: 模型路径
        device: 计算设备
        
    返回:
        tuple: (model, tokenizer, config)
    """
    from apt_model.config.apt_config import APTConfig
    from apt_model.modeling.apt_model import APTLargeModel
    from apt_model.utils import get_device
    
    if device is None:
        device = get_device()
    
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
    tokenizer = load_integrated_tokenizer(tokenizer_path)
    if tokenizer is None:
        raise RuntimeError("无法加载分词器，请检查模型目录是否完整。")

    if hasattr(tokenizer, "pad_token_id") and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    
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
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        
        # 保存检查点
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
        
        torch.save(checkpoint, checkpoint_path)
        
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
        
        # 保存元数据
        with open(os.path.join(self.save_dir, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # 如果有分词器，也保存
        if tokenizer:
            tokenizer_path = os.path.join(self.save_dir, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            if not save_integrated_tokenizer(tokenizer, tokenizer_path):
                logging.getLogger('apt_model.tokenizer').warning("保存分词器失败，跳过检查点中的分词器。")
        
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