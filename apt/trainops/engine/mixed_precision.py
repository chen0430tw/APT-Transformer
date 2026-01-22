#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合精度训练支持

提供 FP16/BF16 混合精度训练，减少显存占用并加速训练。
支持 PyTorch 原生 AMP (Automatic Mixed Precision)。
"""

from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
autocast = torch.cuda.amp.autocast
GradScaler = torch.cuda.amp.GradScaler
from typing import Optional, Literal, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """混合精度训练管理器

    封装 PyTorch AMP 功能，支持 FP16 和 BF16。

    示例:
        >>> manager = MixedPrecisionManager(enabled=True, dtype='bf16')
        >>>
        >>> for batch in dataloader:
        >>>     with manager.autocast():
        >>>         loss = model(batch)
        >>>
        >>>     manager.scale_and_step(loss, optimizer, model.parameters())
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: Literal['fp16', 'bf16', 'fp32'] = 'bf16',
        device: str = 'cuda',
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """
        Args:
            enabled: 是否启用混合精度
            dtype: 精度类型 ('fp16', 'bf16', 'fp32')
            device: 设备类型
            init_scale: GradScaler 初始缩放因子
            growth_factor: 梯度缩放增长因子
            backoff_factor: 梯度缩放回退因子
            growth_interval: 增长间隔（步数）
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype_str = dtype
        self.device = device

        # 确定精度类型
        if not self.enabled:
            self.dtype = torch.float32
            self.use_scaler = False
        elif dtype == 'fp16':
            self.dtype = torch.float16
            self.use_scaler = True
        elif dtype == 'bf16':
            if not torch.cuda.is_bf16_supported():
                logger.warning(
                    "BF16 not supported on this GPU, falling back to FP16"
                )
                self.dtype = torch.float16
                self.dtype_str = 'fp16'
            else:
                self.dtype = torch.bfloat16
            self.use_scaler = False  # BF16 不需要 GradScaler
        else:  # fp32
            self.dtype = torch.float32
            self.use_scaler = False

        # 创建 GradScaler（仅 FP16 需要）
        if self.use_scaler:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None

        # 统计信息
        self.stats = {
            'total_steps': 0,
            'overflow_steps': 0,
            'scale_updates': 0,
        }

        logger.info(
            f"Mixed precision training: enabled={self.enabled}, "
            f"dtype={self.dtype_str}, use_scaler={self.use_scaler}"
        )

    def autocast(self):
        """返回 autocast 上下文管理器

        Returns:
            torch.cuda.amp.autocast 上下文管理器
        """
        if self.enabled:
            return autocast(dtype=self.dtype)
        else:
            # 返回一个空的上下文管理器
            from contextlib import nullcontext
            return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失（仅 FP16 需要）

        Args:
            loss: 原始损失

        Returns:
            缩放后的损失
        """
        if self.use_scaler:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Optional[torch.nn.parameter.Parameter] = None,
        max_grad_norm: Optional[float] = None,
    ) -> bool:
        """执行优化器步进（处理梯度缩放和裁剪）

        Args:
            optimizer: 优化器
            parameters: 模型参数（用于梯度裁剪）
            max_grad_norm: 最大梯度范数

        Returns:
            bool: 是否成功更新参数（False 表示梯度溢出被跳过）
        """
        # 梯度裁剪
        if max_grad_norm is not None and parameters is not None:
            if self.use_scaler:
                # Unscale gradients before clipping
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

        # 优化器步进
        if self.use_scaler:
            self.scaler.step(optimizer)
            old_scale = self.scaler.get_scale()
            self.scaler.update()
            new_scale = self.scaler.get_scale()

            # 检测是否跳过更新
            skipped = (new_scale < old_scale)
            if skipped:
                self.stats['overflow_steps'] += 1
            if new_scale != old_scale:
                self.stats['scale_updates'] += 1

            self.stats['total_steps'] += 1
            return not skipped
        else:
            optimizer.step()
            self.stats['total_steps'] += 1
            return True

    def scale_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters: Optional[torch.nn.parameter.Parameter] = None,
        max_grad_norm: Optional[float] = None,
        retain_graph: bool = False,
    ) -> bool:
        """一步完成：反向传播 + 缩放 + 裁剪 + 优化

        这是最常用的方法，封装了完整的训练步骤。

        Args:
            loss: 损失张量
            optimizer: 优化器
            parameters: 模型参数
            max_grad_norm: 最大梯度范数
            retain_graph: 是否保留计算图

        Returns:
            bool: 是否成功更新参数
        """
        # 反向传播
        scaled_loss = self.scale_loss(loss)
        scaled_loss.backward(retain_graph=retain_graph)

        # 优化器步进
        success = self.step_optimizer(optimizer, parameters, max_grad_norm)

        return success

    def get_scale(self) -> float:
        """获取当前梯度缩放因子

        Returns:
            float: 当前缩放因子（无 scaler 时返回 1.0）
        """
        if self.use_scaler:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> Dict[str, Any]:
        """保存状态（用于 checkpoint）

        Returns:
            状态字典
        """
        state = {
            'enabled': self.enabled,
            'dtype': self.dtype_str,
            'stats': self.stats.copy(),
        }
        if self.use_scaler:
            state['scaler'] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """加载状态（从 checkpoint）

        Args:
            state: 状态字典
        """
        if self.use_scaler and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
        if 'stats' in state:
            self.stats.update(state['stats'])

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计字典
        """
        stats = self.stats.copy()
        if stats['total_steps'] > 0:
            stats['overflow_rate'] = stats['overflow_steps'] / stats['total_steps']
        else:
            stats['overflow_rate'] = 0.0

        if self.use_scaler:
            stats['current_scale'] = self.scaler.get_scale()

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_steps': 0,
            'overflow_steps': 0,
            'scale_updates': 0,
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_mixed_precision_manager(
    enabled: bool = True,
    dtype: str = 'bf16',
    device: str = 'cuda',
) -> MixedPrecisionManager:
    """创建混合精度管理器（便捷函数）

    Args:
        enabled: 是否启用
        dtype: 精度类型
        device: 设备

    Returns:
        MixedPrecisionManager 实例
    """
    return MixedPrecisionManager(enabled=enabled, dtype=dtype, device=device)


def check_amp_support() -> Dict[str, bool]:
    """检查 AMP 支持情况

    Returns:
        支持情况字典
    """
    support = {
        'cuda_available': torch.cuda.is_available(),
        'fp16_supported': False,
        'bf16_supported': False,
    }

    if torch.cuda.is_available():
        support['fp16_supported'] = True
        support['bf16_supported'] = torch.cuda.is_bf16_supported()

        # 获取 GPU 信息
        device_name = torch.cuda.get_device_name(0)
        support['device_name'] = device_name

        # 检测推荐的精度类型
        if 'A100' in device_name or 'A6000' in device_name or 'H100' in device_name:
            support['recommended_dtype'] = 'bf16'
        else:
            support['recommended_dtype'] = 'fp16'

    return support


def log_amp_info():
    """打印 AMP 支持信息到日志"""
    support = check_amp_support()

    if support['cuda_available']:
        logger.info(f"GPU: {support.get('device_name', 'Unknown')}")
        logger.info(f"FP16 support: {support['fp16_supported']}")
        logger.info(f"BF16 support: {support['bf16_supported']}")
        logger.info(f"Recommended dtype: {support.get('recommended_dtype', 'fp32')}")
    else:
        logger.warning("CUDA not available, mixed precision disabled")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 检查 AMP 支持
    log_amp_info()

    # 创建管理器
    manager = create_mixed_precision_manager(enabled=True, dtype='bf16')

    # 模拟训练循环
    model = torch.nn.Linear(10, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step in range(5):
        optimizer.zero_grad()

        # 前向传播（使用 autocast）
        with manager.autocast():
            inputs = torch.randn(32, 10).cuda()
            outputs = model(inputs)
            loss = outputs.mean()

        # 反向传播 + 优化
        success = manager.scale_and_step(
            loss, optimizer, model.parameters(), max_grad_norm=1.0
        )

        print(f"Step {step}: loss={loss.item():.4f}, updated={success}")

    # 打印统计
    print("\nStatistics:")
    for key, value in manager.get_stats().items():
        print(f"  {key}: {value}")
