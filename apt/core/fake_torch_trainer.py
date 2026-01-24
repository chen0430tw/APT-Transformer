#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚假训练日志生成器 - 用于在没有真实Torch时模拟训练过程

使用方法:
    from apt.core.fake_torch_trainer import FakeTrainingLogger

    logger = FakeTrainingLogger(epochs=10, batches_per_epoch=50)
    for epoch, batch, loss in logger.train():
        print(f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}")
"""

import random
import time
from typing import Generator, Tuple


class FakeTrainingLogger:
    """虚假训练日志生成器"""

    def __init__(
        self,
        epochs: int = 10,
        batches_per_epoch: int = 50,
        initial_loss: float = 6.5,
        final_loss: float = 0.5,
        noise_level: float = 0.3,
        convergence_rate: float = 0.85,
    ):
        """
        初始化虚假训练日志生成器

        Args:
            epochs: 训练轮数
            batches_per_epoch: 每轮的batch数
            initial_loss: 初始loss
            final_loss: 最终loss（目标）
            noise_level: loss的随机噪声水平
            convergence_rate: 收敛速度（0-1，越大收敛越慢）
        """
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.initial_loss = initial_loss
        self.final_loss = final_loss
        self.noise_level = noise_level
        self.convergence_rate = convergence_rate

        # 计算衰减系数
        self.decay = convergence_rate ** (1.0 / (epochs * batches_per_epoch))

    def generate_loss(self, step: int) -> float:
        """
        生成虚假loss值，模拟真实训练的收敛曲线

        Args:
            step: 当前步数（从0开始）

        Returns:
            loss值
        """
        # 指数衰减
        progress = self.decay ** step
        base_loss = self.final_loss + (self.initial_loss - self.final_loss) * progress

        # 添加随机噪声
        noise = random.gauss(0, self.noise_level)
        loss = max(0.01, base_loss + noise)

        return loss

    def train(self) -> Generator[Tuple[int, int, float], None, None]:
        """
        生成训练日志

        Yields:
            (epoch, batch, loss) 元组
        """
        total_steps = 0

        for epoch in range(1, self.epochs + 1):
            for batch in range(1, self.batches_per_epoch + 1):
                loss = self.generate_loss(total_steps)
                total_steps += 1

                yield epoch, batch, loss

    def get_formatted_log(
        self, epoch: int, batch: int, loss: float, show_progress: bool = True
    ) -> str:
        """
        获取格式化的训练日志

        Args:
            epoch: 当前epoch
            batch: 当前batch
            loss: 当前loss
            show_progress: 是否显示进度条

        Returns:
            格式化的日志字符串
        """
        if show_progress:
            progress = batch / self.batches_per_epoch
            bar_length = 20
            filled = int(bar_length * progress)
            bar = "=" * filled + ">" + "." * (bar_length - filled - 1)
            return (
                f"Epoch {epoch}/{self.epochs} "
                f"[{bar}] "
                f"{batch}/{self.batches_per_epoch} - "
                f"loss: {loss:.4f}"
            )
        else:
            return f"Epoch {epoch}/{self.epochs}, Batch {batch}: loss={loss:.4f}"


# ============================================================================
# 使用示例
# ============================================================================

def demo_fake_training():
    """演示虚假训练"""
    print("=" * 80)
    print("虚假训练日志生成器演示")
    print("=" * 80)

    # 创建logger
    logger = FakeTrainingLogger(
        epochs=3,
        batches_per_epoch=10,
        initial_loss=6.5,
        final_loss=0.8,
        noise_level=0.2,
        convergence_rate=0.9,
    )

    print("\n开始虚假训练...")
    print("-" * 80)

    current_epoch = 0
    for epoch, batch, loss in logger.train():
        # 每个epoch打印一次header
        if epoch != current_epoch:
            current_epoch = epoch
            print(f"\nEpoch {epoch}/3:")

        # 打印进度
        log = logger.get_formatted_log(epoch, batch, loss, show_progress=True)
        print(f"\r{log}", end="", flush=True)

        # 模拟训练延迟
        time.sleep(0.05)

    print("\n\n" + "-" * 80)
    print("✓ 虚假训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    demo_fake_training()
