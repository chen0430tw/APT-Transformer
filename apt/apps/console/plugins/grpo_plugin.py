#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO Plugin (Group Relative Policy Optimization)

Training tier plugin implementing GRPO algorithm for reinforcement learning.

Priority: 380 (Training tier)
Events: on_batch_end, on_step_end
Capabilities: write_metrics, read_state
"""

import logging
from typing import Dict, Any, Optional
import torch
from apt.apps.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)

logger = logging.getLogger(__name__)

# 延迟导入，避免循环依赖
_grpo_trainer = None
_GRPOConfig = None


def _import_grpo():
    """延迟导入GRPO trainer"""
    global _grpo_trainer, _GRPOConfig
    if _grpo_trainer is None:
        try:
            from apt.apps.rl.grpo_trainer import GRPOTrainer, GRPOConfig
            _grpo_trainer = GRPOTrainer
            _GRPOConfig = GRPOConfig
        except ImportError as e:
            logger.warning(f"Failed to import GRPOTrainer: {e}")
    return _grpo_trainer, _GRPOConfig


class GRPOPlugin(PluginBase):
    """
    GRPO Plugin

    Implements Group Relative Policy Optimization for RL-based training.

    Features:
    - Computes group-relative advantages
    - Updates policy based on group comparisons
    - Tracks GRPO-specific metrics (group variance, relative rewards)
    """

    def __init__(self):
        """初始化 GRPO 插件"""
        super().__init__()
        self.group_size = 4  # 默认组大小
        self.advantage_buffer = []
        self.metrics = {
            'group_variance': 0.0,
            'relative_reward_mean': 0.0,
            'policy_updates': 0
        }

        # GRPO训练器 (延迟初始化)
        self.trainer: Optional[Any] = None
        self.policy_model = None
        self.reward_model = None

    def get_manifest(self) -> PluginManifest:
        """
        获取插件清单

        Returns:
            插件清单
        """
        return PluginManifest(
            name="grpo",
            version="1.0.0",
            description="Group Relative Policy Optimization for RL training",
            author="APT Team",
            priority=PluginPriority.GRPO,
            blocking=True,  # 需要阻塞主线程以更新策略
            events=[
                PluginEvent.ON_BATCH_END,
                PluginEvent.ON_STEP_END,
                PluginEvent.ON_EPOCH_END
            ],
            requires=[
                "core:trainer",
            ],
            conflicts=[
                "plugin:rlhf",  # 与 RLHF 冲突（两者不能同时使用）
                "plugin:dpo"    # 与 DPO 冲突
            ],
            capabilities=[
                PluginCapability.WRITE_METRICS,
                PluginCapability.READ_STATE,
                PluginCapability.WRITE_STATE
            ],
            resources={
                "cpu_ms": 15.0,  # GRPO 计算需要 15ms CPU
                "gpu_ms": 5.0,   # GPU 计算 5ms
                "io_mb": 0.5     # 内存占用 0.5MB
            },
            rate_limit={
                "steps": 1  # 每步都执行
            },
            sandbox=True,
            fail_limit=5,
            s_default=0.3,  # 默认净效用
            eta=1.2         # EQI 证据调制参数
        )

    def initialize(self, config: Dict[str, Any] = None):
        """
        初始化插件

        Args:
            config: 配置字典
                - group_size: 组大小
                - policy_model: 策略模型
                - reward_model: 奖励模型
                - learning_rate: 学习率
                - advantage_type: 优势类型 ("relative", "normalized", "rank")
        """
        if config:
            self.group_size = config.get('group_size', 4)

            # 获取模型
            self.policy_model = config.get('policy_model')
            self.reward_model = config.get('reward_model')

            # 创建GRPO训练器
            if self.policy_model is not None:
                GRPOTrainer, GRPOConfig = _import_grpo()
                if GRPOTrainer is not None:
                    grpo_config = GRPOConfig(
                        group_size=self.group_size,
                        learning_rate=config.get('learning_rate', 1e-5),
                        advantage_type=config.get('advantage_type', 'relative')
                    )

                    self.trainer = GRPOTrainer(
                        policy_model=self.policy_model,
                        reward_model=self.reward_model,
                        config=grpo_config,
                        device=config.get('device', 'cuda')
                    )

                    logger.info(f"GRPO Plugin initialized with actual trainer (group_size={self.group_size})")
                else:
                    logger.warning("GRPO Plugin initialized without trainer (import failed)")
            else:
                logger.info(f"GRPO Plugin initialized in compatibility mode (group_size={self.group_size})")

    def on_batch_end(self, context: Dict[str, Any]):
        """
        Batch 结束时处理

        Args:
            context: 事件上下文
        """
        step = context.get('step', 0)
        data = context.get('data', {})

        # 获取 batch 数据
        batch_rewards = data.get('rewards', [])
        if not batch_rewards:
            return

        # 计算组内相对优势
        if len(batch_rewards) >= self.group_size:
            group_rewards = batch_rewards[-self.group_size:]
            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [r - mean_reward for r in group_rewards]

            # 计算组内方差
            variance = sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)

            # 更新指标
            self.metrics['group_variance'] = variance
            self.metrics['relative_reward_mean'] = mean_reward

            # 存储到上下文
            self.set_context('advantages', advantages)
            self.set_context('group_variance', variance)

            logger.debug(f"[GRPO] Step {step}: group_variance={variance:.4f}, mean_reward={mean_reward:.4f}")

    def on_step_end(self, context: Dict[str, Any]):
        """
        Step 结束时处理

        Args:
            context: 事件上下文
                Expected keys:
                - step: 当前步数
                - data: 数据字典
                    - responses: 生成的响应 tensor [batch, seq_len]
                    - response_masks: mask tensor [batch, seq_len]
                    - rewards: 奖励 tensor [batch] (可选)
        """
        step = context.get('step', 0)
        data = context.get('data', {})

        # 如果有实际的trainer，执行真实的策略更新
        if self.trainer is not None:
            responses = data.get('responses')
            response_masks = data.get('response_masks')
            rewards = data.get('rewards')

            if responses is not None:
                try:
                    # 执行GRPO训练步骤
                    stats = self.trainer.train_step(
                        responses=responses,
                        response_masks=response_masks,
                        rewards=rewards
                    )

                    # 更新指标
                    self.metrics['policy_updates'] += 1
                    self.metrics['group_variance'] = stats.get('group_variance', 0.0)
                    self.metrics['relative_reward_mean'] = stats.get('mean_reward', 0.0)

                    # 写入到公共上下文
                    if 'metrics' not in data:
                        data['metrics'] = {}
                    data['metrics']['grpo_variance'] = self.metrics['group_variance']
                    data['metrics']['grpo_updates'] = self.metrics['policy_updates']
                    data['metrics']['grpo_policy_loss'] = stats.get('policy_loss', 0.0)
                    data['metrics']['grpo_kl'] = stats.get('kl_divergence', 0.0)

                    logger.debug(f"[GRPO] Policy updated at step {step}: loss={stats.get('policy_loss', 0.0):.4f}")

                except Exception as e:
                    logger.error(f"[GRPO] Error during policy update: {e}")

        else:
            # 兼容模式：获取存储的优势 (旧版本行为)
            advantages = self.get_context('advantages')
            if advantages:
                self.metrics['policy_updates'] += 1

                # 写入到公共上下文
                if 'metrics' not in data:
                    data['metrics'] = {}
                data['metrics']['grpo_variance'] = self.metrics['group_variance']
                data['metrics']['grpo_updates'] = self.metrics['policy_updates']

                logger.debug(f"[GRPO] Compatibility mode: step {step}")

    def on_epoch_end(self, context: Dict[str, Any]):
        """
        Epoch 结束时处理

        Args:
            context: 事件上下文
        """
        epoch = context.get('epoch', 0)

        # 打印 epoch 统计
        logger.info(f"[GRPO] Epoch {epoch} completed: "
                   f"policy_updates={self.metrics['policy_updates']}, "
                   f"avg_variance={self.metrics['group_variance']:.4f}")

        # 重置部分指标
        self.metrics['policy_updates'] = 0

    def cleanup(self):
        """清理资源"""
        logger.info("GRPO Plugin cleanup")
        self.advantage_buffer.clear()
