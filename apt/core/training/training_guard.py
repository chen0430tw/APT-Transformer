#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Guard - 训练保护机制

防止无限训练、资源耗尽、MCP 超时等问题

Features:
- Early stopping (validation loss based)
- Max steps/time limits
- Memory monitoring
- Gradient anomaly detection
- MCP-aware checkpointing
- Auto-cleanup
"""

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
import time
import psutil
import os
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime, timedelta
import warnings


class EarlyStopping:
    """
    Early Stopping 机制

    监控验证损失，如果连续 patience 个 epoch 没有改善则停止训练
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: 容忍的 epoch 数
            min_delta: 最小改善量
            mode: 'min' (越小越好) 或 'max' (越大越好)
            verbose: 是否打印消息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # 检查是否改善
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"✓ 验证指标改善: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠ 验证指标未改善 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered (patience={self.patience})")
                return True

        return False


class TrainingGuard:
    """
    训练保护器 - 防止无限训练和资源耗尽

    监控:
    - 最大训练步数
    - 最大训练时间
    - 内存使用
    - 梯度异常
    - 损失异常 (NaN/Inf)
    """

    def __init__(
        self,
        max_steps: Optional[int] = None,
        max_time_hours: Optional[float] = None,
        max_memory_percent: float = 90.0,
        check_gradients: bool = True,
        auto_cleanup_every: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        mcp_checkpoint_interval: int = 1000,
        verbose: bool = True
    ):
        """
        Args:
            max_steps: 最大训练步数 (None = 无限制)
            max_time_hours: 最大训练时间（小时）
            max_memory_percent: 最大内存使用百分比
            check_gradients: 是否检查梯度异常
            auto_cleanup_every: 每 N 步自动清理缓存
            early_stopping: EarlyStopping 实例
            mcp_checkpoint_interval: MCP 检查点间隔
            verbose: 详细输出
        """
        self.max_steps = max_steps
        self.max_time_hours = max_time_hours
        self.max_memory_percent = max_memory_percent
        self.check_gradients = check_gradients
        self.auto_cleanup_every = auto_cleanup_every
        self.early_stopping = early_stopping
        self.mcp_checkpoint_interval = mcp_checkpoint_interval
        self.verbose = verbose

        # 内部状态
        self.start_time = None
        self.step_count = 0
        self.should_stop = False
        self.stop_reason = None

        # 统计
        self.stats = {
            'nan_losses': 0,
            'inf_losses': 0,
            'gradient_explosions': 0,
            'memory_warnings': 0,
            'cleanups': 0
        }

    def start(self):
        """开始训练监控"""
        self.start_time = time.time()
        self.step_count = 0
        self.should_stop = False
        self.stop_reason = None

        if self.verbose:
            print("🛡️ Training Guard 已启动")
            if self.max_steps:
                print(f"  最大步数: {self.max_steps}")
            if self.max_time_hours:
                print(f"  最大时间: {self.max_time_hours:.1f} 小时")
            print(f"  内存限制: {self.max_memory_percent}%")
            if self.early_stopping:
                print(f"  Early Stopping: patience={self.early_stopping.patience}")

    def step(
        self,
        loss: Optional[float] = None,
        model: Optional[torch.nn.Module] = None
    ) -> bool:
        """
        每个训练步调用

        Args:
            loss: 当前损失值
            model: 模型（用于梯度检查）

        Returns:
            是否应该继续训练 (True=继续, False=停止)
        """
        self.step_count += 1

        # 1. 检查最大步数
        if self.max_steps and self.step_count >= self.max_steps:
            self._stop(f"达到最大步数 {self.max_steps}")
            return False

        # 2. 检查最大时间
        if self.max_time_hours:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.max_time_hours:
                self._stop(f"达到最大时间 {self.max_time_hours:.1f} 小时")
                return False

        # 3. 检查损失异常
        if loss is not None:
            if not self._check_loss(loss):
                return False

        # 4. 检查梯度
        if self.check_gradients and model is not None:
            if not self._check_gradients(model):
                return False

        # 5. 检查内存
        if not self._check_memory():
            return False

        # 6. 自动清理
        if self.step_count % self.auto_cleanup_every == 0:
            self._cleanup()

        # 7. MCP 检查点提醒
        if self.step_count % self.mcp_checkpoint_interval == 0:
            if self.verbose:
                print(f"💾 建议保存检查点 (步数: {self.step_count})")

        return True

    def validate(self, val_score: float) -> bool:
        """
        验证步骤（用于 early stopping）

        Args:
            val_score: 验证分数

        Returns:
            是否应该继续训练
        """
        if self.early_stopping is not None:
            if self.early_stopping(val_score):
                self._stop("Early stopping triggered")
                return False
        return True

    def _check_loss(self, loss: float) -> bool:
        """检查损失值"""
        import math

        if math.isnan(loss):
            self.stats['nan_losses'] += 1
            self._stop(f"损失为 NaN (第 {self.stats['nan_losses']} 次)")
            return False

        if math.isinf(loss):
            self.stats['inf_losses'] += 1
            self._stop(f"损失为 Inf (第 {self.stats['inf_losses']} 次)")
            return False

        return True

    def _check_gradients(self, model: torch.nn.Module) -> bool:
        """检查梯度异常"""
        import math

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

                # 检查 NaN/Inf
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    self._stop("梯度包含 NaN 或 Inf")
                    return False

        total_norm = total_norm ** 0.5

        # 检查梯度爆炸 (阈值: 100)
        if total_norm > 100.0:
            self.stats['gradient_explosions'] += 1
            if self.verbose:
                warnings.warn(f"⚠️ 梯度范数过大: {total_norm:.2f}")

            if self.stats['gradient_explosions'] >= 10:
                self._stop("梯度持续爆炸 (10 次)")
                return False

        return True

    def _check_memory(self) -> bool:
        """检查内存使用"""
        memory = psutil.virtual_memory()
        percent = memory.percent

        if percent > self.max_memory_percent:
            self.stats['memory_warnings'] += 1

            if self.verbose:
                warnings.warn(
                    f"⚠️ 内存使用过高: {percent:.1f}% "
                    f"(限制: {self.max_memory_percent}%)"
                )

            # 强制清理
            self._cleanup()

            # 再次检查
            memory = psutil.virtual_memory()
            if memory.percent > self.max_memory_percent:
                self._stop(f"内存耗尽: {memory.percent:.1f}%")
                return False

        return True

    def _cleanup(self):
        """清理缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.stats['cleanups'] += 1

        if self.verbose and self.step_count % (self.auto_cleanup_every * 10) == 0:
            print(f"🧹 自动清理 (第 {self.stats['cleanups']} 次)")

    def _stop(self, reason: str):
        """停止训练"""
        self.should_stop = True
        self.stop_reason = reason

        if self.verbose:
            print(f"\n🛑 训练停止: {reason}")
            print(f"   总步数: {self.step_count}")
            elapsed = time.time() - self.start_time
            print(f"   训练时间: {elapsed/3600:.2f} 小时")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'total_steps': self.step_count,
            'elapsed_hours': elapsed / 3600,
            'stopped': self.should_stop,
            'stop_reason': self.stop_reason,
            **self.stats
        }


class SafeTrainingContext:
    """
    安全训练上下文管理器

    Usage:
        with SafeTrainingContext(max_steps=10000, max_time_hours=2) as guard:
            for epoch in range(epochs):
                for batch in dataloader:
                    loss = train_step(batch)

                    # 检查是否继续
                    if not guard.step(loss=loss, model=model):
                        break

                # 验证
                val_loss = validate()
                if not guard.validate(val_loss):
                    break
    """

    def __init__(self, **guard_kwargs):
        self.guard = TrainingGuard(**guard_kwargs)

    def __enter__(self):
        self.guard.start()
        return self.guard

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.guard.verbose:
            stats = self.guard.get_stats()
            print("\n" + "="*80)
            print("训练保护统计:")
            print(f"  总步数: {stats['total_steps']}")
            print(f"  训练时间: {stats['elapsed_hours']:.2f} 小时")
            print(f"  NaN 损失: {stats['nan_losses']}")
            print(f"  Inf 损失: {stats['inf_losses']}")
            print(f"  梯度爆炸: {stats['gradient_explosions']}")
            print(f"  内存警告: {stats['memory_warnings']}")
            print(f"  自动清理: {stats['cleanups']}")
            if stats['stopped']:
                print(f"  停止原因: {stats['stop_reason']}")
            print("="*80)


# ==================== MCP-Aware Training ====================

class MCPSafeTrainer:
    """
    MCP-Safe 训练器包装器

    自动处理 MCP 相关的训练问题：
    - 定期保存检查点
    - 优雅处理中断
    - 资源监控
    """

    def __init__(
        self,
        trainer: Any,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval: int = 1000,
        max_checkpoints: int = 5,
        resume_from_checkpoint: bool = True
    ):
        """
        Args:
            trainer: 原始训练器实例
            checkpoint_dir: 检查点目录
            checkpoint_interval: 检查点间隔（步数）
            max_checkpoints: 最多保留检查点数
            resume_from_checkpoint: 是否自动恢复
        """
        self.trainer = trainer
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.resume_from_checkpoint = resume_from_checkpoint

        os.makedirs(checkpoint_dir, exist_ok=True)

        # 尝试恢复
        if resume_from_checkpoint:
            self._try_resume()

    def train_step(self, *args, **kwargs):
        """包装的训练步骤"""
        result = self.trainer.train_step(*args, **kwargs)

        # 定期保存
        if hasattr(self.trainer, 'step_count'):
            if self.trainer.step_count % self.checkpoint_interval == 0:
                self.save_checkpoint()

        return result

    def save_checkpoint(self, name: Optional[str] = None):
        """保存检查点"""
        if name is None:
            name = f"checkpoint_step_{self.trainer.step_count}.pt"

        checkpoint_path = os.path.join(self.checkpoint_dir, name)

        # 保存
        if hasattr(self.trainer, 'save_model'):
            self.trainer.save_model(checkpoint_path)

        # 清理旧检查点
        self._cleanup_old_checkpoints()

        print(f"💾 检查点已保存: {checkpoint_path}")

    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_')],
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x))
        )

        # 保留最新的 N 个
        if len(checkpoints) > self.max_checkpoints:
            for old_ckpt in checkpoints[:-self.max_checkpoints]:
                os.remove(os.path.join(self.checkpoint_dir, old_ckpt))

    def _try_resume(self):
        """尝试恢复训练"""
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_')],
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x))
        )

        if checkpoints:
            latest = os.path.join(self.checkpoint_dir, checkpoints[-1])
            if hasattr(self.trainer, 'load_model'):
                try:
                    self.trainer.load_model(latest)
                    print(f"♻️ 从检查点恢复: {latest}")
                except Exception as e:
                    print(f"⚠️ 恢复失败: {e}")


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example 1: Basic Early Stopping
    early_stop = EarlyStopping(patience=5, mode='min')

    for epoch in range(100):
        train_loss = 0.5 / (epoch + 1)  # Simulated decreasing loss
        val_loss = 0.5 / (epoch + 1) + (0.01 if epoch > 10 else 0)

        if early_stop(val_loss):
            print(f"Training stopped at epoch {epoch}")
            break

    # Example 2: Training Guard
    print("\n" + "="*80)

    guard = TrainingGuard(
        max_steps=100,
        max_time_hours=0.01,  # 36 seconds for demo
        early_stopping=EarlyStopping(patience=3)
    )

    guard.start()

    for step in range(1000):  # Will stop before 1000
        loss = 1.0 / (step + 1)

        if not guard.step(loss=loss):
            break

        # Simulate validation every 10 steps
        if step % 10 == 0:
            val_loss = loss + 0.01
            if not guard.validate(val_loss):
                break

    print(guard.get_stats())
