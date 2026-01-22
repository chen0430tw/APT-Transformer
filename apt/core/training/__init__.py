"""
APT-SOSA - 火种源自组织智能训练系统
Spark Seed Self-Organizing Algorithm for APT Training

核心功能:
- 实时训练监控
- 自动异常检测
- 智能诊断分析
- 自适应修复策略
- SOSA自组织决策

作者: 430 + GPT-5.1 Thinking (原始SOSA)
改造: chen0430tw (APT集成)
"""

__version__ = "0.1.0"
__author__ = "chen0430tw"

from .sosa_core import (
    SOSA,
    Event,
    BinaryTwin,
    SparseMarkov
)

from .training_monitor import (
    TrainingMonitor,
    ErrorType,
    FixAction,
    TrainingSnapshot
)

try:
    from .apt_integration import (
        SOSATrainingWrapper,
        create_monitored_training_loop
    )
except ImportError:
    # Fallback for when apt_integration is not available
    SOSATrainingWrapper = None
    create_monitored_training_loop = None

__all__ = [
    # SOSA核心
    "SOSA",
    "Event",
    "BinaryTwin",
    "SparseMarkov",
    
    # 训练监控
    "TrainingMonitor",
    "ErrorType",
    "FixAction",
    "TrainingSnapshot",
    
    # APT集成
    "SOSATrainingWrapper",
    "create_monitored_training_loop",
    
    # 版本信息
    "__version__",
    "__author__"
]


# ==================== 便捷创建函数 ====================

def create_training_monitor(window_seconds: float = 10.0):
    """
    创建训练监控器的便捷函数
    
    Args:
        window_seconds: SOSA时间窗口大小(秒)
    
    Returns:
        TrainingMonitor实例
    
    Example:
        >>> import apt_sosa
        >>> monitor = apt_sosa.create_training_monitor()
        >>> monitor.log_step(step=0, loss=1.5, grad_norm=2.0)
    """
    return TrainingMonitor(sosa_window=window_seconds)


def wrap_training(model, optimizer, **kwargs):
    """
    包装训练过程的便捷函数
    
    Args:
        model: PyTorch模型
        optimizer: 优化器
        **kwargs: 其他参数传递给SOSATrainingWrapper
    
    Returns:
        SOSATrainingWrapper实例
    
    Example:
        >>> import apt_sosa
        >>> wrapper = apt_sosa.wrap_training(
        ...     model=my_model,
        ...     optimizer=my_optimizer,
        ...     auto_fix=True
        ... )
        >>> 
        >>> for batch in dataloader:
        ...     loss = wrapper.training_step(batch)
    """
    return SOSATrainingWrapper(model, optimizer, **kwargs)


# ==================== 快速开始示例 ====================

def quick_start_example():
    """
    快速开始示例
    
    演示如何使用APT-SOSA系统
    """
    import torch
    import torch.nn as nn
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("APT-SOSA 快速开始示例")
    print("=" * 70)
    
    # 1. 创建简单模型和优化器
    print("\n1. 创建模型和优化器...")
    
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. 创建SOSA包装器
    print("\n2. 创建SOSA训练包装器...")
    
    wrapper = wrap_training(
        model=model,
        optimizer=optimizer,
        auto_fix=True,  # 启用自动修复
        checkpoint_dir="./sosa_checkpoints"
    )
    
    # 3. 模拟训练
    print("\n3. 模拟训练过程...")
    
    def simple_forward(model, batch):
        """简单的前向函数"""
        pred = model(batch['x'])
        loss = nn.functional.mse_loss(pred, batch['y'])
        return loss
    
    for step in range(50):
        # 创建假数据
        batch = {
            'x': torch.randn(8, 10),
            'y': torch.randn(8, 1)
        }
        
        # 执行训练步
        loss = wrapper.training_step(batch, simple_forward)
        
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    
    # 4. 查看报告
    print("\n4. 训练报告:")
    stats = wrapper.get_statistics()
    print(f"  总步数: {stats['global_step']}")
    print(f"  最佳Loss: {stats['best_loss']:.4f}")
    print(f"  异常率: {stats['anomaly_rate']:.2%}")
    
    print("\n" + "=" * 70)
    print("示例完成!")
    print("=" * 70)


if __name__ == "__main__":
    quick_start_example()
