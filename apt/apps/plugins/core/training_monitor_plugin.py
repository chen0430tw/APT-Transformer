#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练监控插件 - SOSA智能监控系统

提供:
- 实时训练指标监控
- 自动异常检测 (7种错误类型)
- 智能诊断与修复
- SOSA自组织决策

集成方式:
1. 包装现有训练循环
2. 最小代码修改
3. 自动检查点管理

作者: chen0430tw
"""

from typing import Dict, Any, Optional, Callable
import logging
import torch
import torch.nn as nn

# 导入SOSA核心
from apt.apt_model.training import (
    SOSATrainingWrapper,
    TrainingMonitor,
    ErrorType,
    FixAction,
    create_training_monitor,
    wrap_training
)

logger = logging.getLogger(__name__)


class TrainingMonitorPlugin:
    """
    APT训练监控插件

    封装SOSA训练监控系统，提供与APT项目的无缝集成
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化训练监控插件

        Args:
            model: PyTorch模型
            optimizer: 优化器
            config: 配置字典
                - auto_fix: 启用自动修复 (默认True)
                - window_seconds: SOSA时间窗口 (默认10.0)
                - max_fixes_per_error: 最大修复次数 (默认3)
                - checkpoint_dir: 检查点目录 (默认'./checkpoints')
                - save_best: 保存最佳模型 (默认True)
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}

        # 创建SOSA包装器
        self.wrapper = SOSATrainingWrapper(
            model=model,
            optimizer=optimizer,
            config=self.config,
            auto_fix=self.config.get('auto_fix', True),
            max_fixes_per_error=self.config.get('max_fixes_per_error', 3),
            checkpoint_dir=self.config.get('checkpoint_dir', './checkpoints')
        )

        logger.info("[SOSA] 训练监控已启用")
        logger.info(f"[SOSA] 自动修复: {self.config.get('auto_fix', True)}")

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        forward_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        执行一个训练步

        自动包含:
        - 前向传播
        - 反向传播
        - 参数更新
        - 异常检测
        - 自动修复 (可选)

        Args:
            batch: 训练批次数据
            forward_fn: 自定义前向函数 (可选)
                       签名: forward_fn(model, batch) -> loss

        Returns:
            loss tensor

        Example:
            >>> # 方式1: 默认前向 (假设batch有input_ids等)
            >>> loss = plugin.training_step(batch)
            >>>
            >>> # 方式2: 自定义前向
            >>> def my_forward(model, batch):
            ...     outputs = model(**batch)
            ...     return outputs.loss
            >>> loss = plugin.training_step(batch, my_forward)
        """
        return self.wrapper.training_step(batch, forward_fn)

    def get_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return self.wrapper.get_statistics()

    def print_report(self):
        """打印训练报告"""
        self.wrapper.print_report()

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        self.wrapper.save_checkpoint(filepath)
        logger.info(f"[SOSA] 检查点已保存: {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        self.wrapper.load_checkpoint(filepath)
        logger.info(f"[SOSA] 检查点已加载: {filepath}")

    def get_monitor(self) -> TrainingMonitor:
        """获取底层监控器"""
        return self.wrapper.monitor

    def enable_auto_fix(self):
        """启用自动修复"""
        self.wrapper.auto_fix = True
        logger.info("[SOSA] 已启用自动修复")

    def disable_auto_fix(self):
        """禁用自动修复"""
        self.wrapper.auto_fix = False
        logger.info("[SOSA] 已禁用自动修复")

    def reset_statistics(self):
        """重置统计信息"""
        self.wrapper.reset_statistics()
        logger.info("[SOSA] 统计信息已重置")


# ==================== 便捷函数 ====================

def create_training_monitor_plugin(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Optional[Dict[str, Any]] = None
) -> TrainingMonitorPlugin:
    """
    创建训练监控插件的便捷函数

    Args:
        model: PyTorch模型
        optimizer: 优化器
        config: 配置字典

    Returns:
        TrainingMonitorPlugin实例

    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> plugin = create_training_monitor_plugin(model, optimizer, {
        ...     'auto_fix': True,
        ...     'checkpoint_dir': './ckpt'
        ... })
        >>>
        >>> for batch in dataloader:
        ...     loss = plugin.training_step(batch)
    """
    return TrainingMonitorPlugin(model, optimizer, config)


def create_monitored_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: Any,
    config: Optional[Dict[str, Any]] = None,
    forward_fn: Optional[Callable] = None,
    report_interval: int = 100
):
    """
    创建带监控的训练循环

    完整的训练循环，包含SOSA监控

    Args:
        model: PyTorch模型
        optimizer: 优化器
        train_dataloader: 训练数据加载器
        config: SOSA配置
        forward_fn: 自定义前向函数
        report_interval: 报告间隔 (步数)

    Example:
        >>> create_monitored_training_loop(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     train_dataloader=dataloader,
        ...     config={'auto_fix': True}
        ... )
    """
    plugin = create_training_monitor_plugin(model, optimizer, config)

    logger.info("[SOSA] 开始监控训练...")

    global_step = 0
    for epoch in range(config.get('num_epochs', 1)):
        for batch in train_dataloader:
            # 训练步
            loss = plugin.training_step(batch, forward_fn)

            global_step += 1

            # 定期报告
            if global_step % report_interval == 0:
                logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                plugin.print_report()

    # 最终报告
    logger.info("\n" + "=" * 70)
    logger.info("训练完成 - 最终报告")
    logger.info("=" * 70)
    plugin.print_report()

    return plugin


# ==================== APT集成工具 ====================

def integrate_with_apt_trainer(
    trainer,
    config: Optional[Dict[str, Any]] = None
):
    """
    与APT Trainer集成

    将SOSA监控集成到现有的APT Trainer中

    Args:
        trainer: APT Trainer实例
        config: SOSA配置

    Returns:
        包装后的trainer

    Example:
        >>> from apt.apt_model.training.trainer import Trainer
        >>> trainer = Trainer(model, config)
        >>> trainer = integrate_with_apt_trainer(trainer, {
        ...     'auto_fix': True
        ... })
        >>> trainer.train()  # 现在会自动监控
    """
    if not hasattr(trainer, 'model') or not hasattr(trainer, 'optimizer'):
        raise ValueError("Trainer必须有model和optimizer属性")

    # 创建监控插件
    plugin = create_training_monitor_plugin(
        model=trainer.model,
        optimizer=trainer.optimizer,
        config=config
    )

    # 保存原始train_step方法
    original_train_step = trainer.train_step

    # 包装train_step
    def monitored_train_step(batch):
        # 创建forward函数
        def forward_fn(model, batch):
            return original_train_step(batch)

        return plugin.training_step(batch, forward_fn)

    # 替换train_step
    trainer.train_step = monitored_train_step
    trainer.sosa_plugin = plugin

    logger.info("[SOSA] 已集成到APT Trainer")

    return trainer


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("【训练监控插件演示】\n")

    # 示例1: 基础使用
    print("=" * 60)
    print("[示例1] 基础训练监控")
    print("=" * 60)

    # 创建简单模型
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 创建监控插件
    plugin = create_training_monitor_plugin(
        model=model,
        optimizer=optimizer,
        config={
            'auto_fix': True,
            'checkpoint_dir': './test_checkpoints'
        }
    )

    # 模拟训练
    def simple_forward(model, batch):
        pred = model(batch['x'])
        loss = nn.functional.mse_loss(pred, batch['y'])
        return loss

    print("\n开始训练...")
    for step in range(30):
        batch = {
            'x': torch.randn(4, 10),
            'y': torch.randn(4, 1)
        }

        loss = plugin.training_step(batch, simple_forward)

        if step % 10 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

    # 报告
    print("\n训练统计:")
    plugin.print_report()

    print("\n[提示] 查看 MODULE_INTEGRATION_PLAN.md 了解完整集成方案")
