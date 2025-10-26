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
from typing import Dict, Any
from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)

logger = logging.getLogger(__name__)


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
        """
        if config:
            self.group_size = config.get('group_size', 4)
            logger.info(f"GRPO Plugin initialized with group_size={self.group_size}")

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
        """
        step = context.get('step', 0)

        # 获取存储的优势
        advantages = self.get_context('advantages')
        if advantages:
            # 模拟策略更新
            self.metrics['policy_updates'] += 1

            # 写入到公共上下文（供其他插件读取）
            data = context.get('data', {})
            if 'metrics' not in data:
                data['metrics'] = {}
            data['metrics']['grpo_variance'] = self.metrics['group_variance']
            data['metrics']['grpo_updates'] = self.metrics['policy_updates']

            logger.debug(f"[GRPO] Policy updated at step {step}")

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
