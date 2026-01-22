"""
APT-SOSA集成适配器
将SOSA智能监控系统集成到APT训练流程

使用方式:
1. 创建 SOSATrainingWrapper
2. 用它包装你的训练循环
3. 自动监控、检测、修复

示例:
    wrapper = SOSATrainingWrapper(model, optimizer, config)
    
    for batch in dataloader:
        loss = wrapper.training_step(batch)
        # wrapper自动处理异常
"""

from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
import torch.nn as nn
from typing import Dict, Optional, Callable, Any
import logging
from pathlib import Path

from .training_monitor import TrainingMonitor, ErrorType, FixAction

logger = logging.getLogger(__name__)


class SOSATrainingWrapper:
    """
    SOSA训练包装器
    
    功能:
    - 自动监控训练指标
    - 检测异常并诊断
    - 应用修复策略
    - 保存最优检查点
    - 生成训练报告
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[Any] = None,
        checkpoint_dir: str = "./checkpoints",
        auto_fix: bool = True,
        max_fixes_per_error: int = 3
    ):
        """
        Args:
            model: PyTorch模型
            optimizer: 优化器
            config: 配置对象 (可选)
            checkpoint_dir: 检查点目录
            auto_fix: 是否自动修复
            max_fixes_per_error: 每种错误最大修复次数
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_fix = auto_fix
        self.max_fixes_per_error = max_fixes_per_error
        
        # 创建监控器
        self.monitor = TrainingMonitor(sosa_window=10.0)
        
        # 修复计数
        self.fix_counts: Dict[ErrorType, int] = {e: 0 for e in ErrorType}
        
        # 最佳Loss和检查点
        self.best_loss = float('inf')
        self.best_checkpoint_path = None
        
        # 当前步数
        self.global_step = 0
        
        # 梯度裁剪参数 (可被修复动作更新)
        self.grad_clip_norm = 1.0
        
        logger.info(f"SOSA训练包装器初始化: auto_fix={auto_fix}")
    
    # ==================== 训练步 ====================
    
    def training_step(
        self,
        batch: Any,
        forward_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        执行一个训练步骤
        
        Args:
            batch: 训练批次
            forward_fn: 自定义前向函数 (可选)
                signature: forward_fn(model, batch) -> loss
        
        Returns:
            loss tensor
        """
        try:
            # 前向传播
            if forward_fn is not None:
                loss = forward_fn(self.model, batch)
            else:
                loss = self._default_forward(batch)
            
            # 检查Loss有效性
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Step {self.global_step}: Loss is NaN/Inf!")
                self._handle_error(ErrorType.NAN_LOSS)
                return loss  # 返回无效loss，由外部处理
            
            # 反向传播
            loss.backward()
            
            # 计算梯度范数
            grad_norm = self._compute_grad_norm()
            
            # 梯度裁剪
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )
            
            # 优化器步骤
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 记录到监控器
            self._log_to_monitor(loss.item(), grad_norm)
            
            # 检测异常
            error = self.monitor.detect_error()
            if error is not None:
                self._handle_error(error)
            
            # 保存最佳检查点
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self._save_checkpoint('best')
            
            self.global_step += 1
            
            return loss
            
        except RuntimeError as e:
            # 捕获CUDA OOM等错误
            if "out of memory" in str(e).lower():
                logger.error(f"Step {self.global_step}: OOM Error!")
                self._handle_error(ErrorType.OOM)
            raise
    
    def _default_forward(self, batch: Any) -> torch.Tensor:
        """默认前向传播 (需要batch是dict或有input_ids)"""
        if isinstance(batch, dict):
            outputs = self.model(**batch)
        else:
            outputs = self.model(batch)
        
        # 尝试提取loss
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            raise ValueError("无法从模型输出提取loss，请提供forward_fn")
    
    def _compute_grad_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _log_to_monitor(self, loss: float, grad_norm: float):
        """记录到监控器"""
        # 获取当前学习率
        lr = self.optimizer.param_groups[0]['lr']
        
        # 获取内存使用 (如果有GPU)
        memory_used = None
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        
        self.monitor.log_step(
            step=self.global_step,
            loss=loss,
            grad_norm=grad_norm,
            lr=lr,
            memory_used=memory_used
        )
    
    # ==================== 异常处理 ====================
    
    def _handle_error(self, error_type: ErrorType):
        """处理错误"""
        logger.warning(f"\n{'='*70}")
        logger.warning(f"检测到异常: {error_type.value} (Step {self.global_step})")
        logger.warning(f"{'='*70}")
        
        # 诊断
        diagnosis = self.monitor.diagnose(error_type)
        logger.warning(f"\n诊断:\n{diagnosis}")
        
        # 检查是否超过修复次数
        if self.fix_counts[error_type] >= self.max_fixes_per_error:
            logger.error(
                f"错误 {error_type.value} 已修复 {self.fix_counts[error_type]} 次，"
                f"达到上限，停止自动修复"
            )
            return
        
        # 建议修复
        fix_action = self.monitor.suggest_fix(error_type)
        logger.warning(f"\n建议修复: {fix_action.action_type}")
        logger.warning(f"  原因: {fix_action.reason}")
        logger.warning(f"  置信度: {fix_action.confidence:.2f}")
        
        # 自动应用修复
        if self.auto_fix and fix_action.confidence > 0.5:
            logger.warning(f"\n应用自动修复...")
            
            success = self.monitor.apply_fix(
                fix_action,
                self.optimizer,
                self.model
            )
            
            if success:
                logger.warning("修复成功!")
                self.fix_counts[error_type] += 1
                
                # 更新梯度裁剪参数
                if '_applied_max_norm' in fix_action.parameters:
                    self.grad_clip_norm = fix_action.parameters['_applied_max_norm']
                    logger.warning(f"梯度裁剪已更新: {self.grad_clip_norm}")
            else:
                logger.error("修复失败!")
        else:
            logger.warning("自动修复已禁用或置信度不足，需要手动处理")
        
        logger.warning(f"{'='*70}\n")
    
    # ==================== 检查点 ====================
    
    def _save_checkpoint(self, name: str = 'latest'):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'monitor_stats': self.monitor.get_statistics(),
            'fix_counts': self.fix_counts
        }
        
        if self.config is not None:
            checkpoint['config'] = self.config
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"检查点已保存: {checkpoint_path}")
        
        if name == 'best':
            self.best_checkpoint_path = checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        if 'fix_counts' in checkpoint:
            self.fix_counts = checkpoint['fix_counts']
        
        logger.info(f"检查点已加载: {checkpoint_path}")
    
    # ==================== 报告 ====================
    
    def print_report(self):
        """打印训练报告"""
        print("\n" + "=" * 80)
        print("APT-SOSA 训练报告")
        print("=" * 80)
        
        print(f"\n训练进度:")
        print(f"  当前步数: {self.global_step}")
        print(f"  最佳Loss: {self.best_loss:.6f}")
        if self.best_checkpoint_path:
            print(f"  最佳检查点: {self.best_checkpoint_path}")
        
        print(f"\n自动修复统计:")
        for error_type, count in self.fix_counts.items():
            if count > 0:
                print(f"  {error_type.value}: {count} 次")
        
        print(f"\n当前配置:")
        print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  梯度裁剪: {self.grad_clip_norm}")
        
        # 调用监控器报告
        print("\n")
        self.monitor.print_report()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.monitor.get_statistics()
        
        stats.update({
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'fix_counts': {e.value: c for e, c in self.fix_counts.items()},
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'grad_clip_norm': self.grad_clip_norm
        })
        
        return stats


# ==================== 便捷函数 ====================

def create_monitored_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader,
    config: Optional[Any] = None,
    num_epochs: int = 10,
    checkpoint_dir: str = "./checkpoints",
    auto_fix: bool = True
):
    """
    创建带SOSA监控的训练循环
    
    Args:
        model: 模型
        optimizer: 优化器
        train_dataloader: 训练数据加载器
        config: 配置
        num_epochs: 训练轮数
        checkpoint_dir: 检查点目录
        auto_fix: 是否自动修复
    
    Returns:
        wrapper, train_loop函数
    """
    wrapper = SOSATrainingWrapper(
        model=model,
        optimizer=optimizer,
        config=config,
        checkpoint_dir=checkpoint_dir,
        auto_fix=auto_fix
    )
    
    def train_loop():
        """训练循环"""
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    loss = wrapper.training_step(batch)
                    epoch_loss += loss.item()
                    
                    # 定期打印
                    if batch_idx % 100 == 0:
                        logger.info(
                            f"  Batch {batch_idx}: "
                            f"loss={loss.item():.4f}, "
                            f"step={wrapper.global_step}"
                        )
                
                except Exception as e:
                    logger.error(f"训练步骤失败: {e}")
                    # 可以选择继续或中断
                    continue
            
            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
            
            # 定期保存
            wrapper._save_checkpoint(f'epoch_{epoch+1}')
        
        # 最终报告
        wrapper.print_report()
    
    return wrapper, train_loop


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== APT-SOSA集成测试 ===\n")
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建包装器
    wrapper = SOSATrainingWrapper(
        model=model,
        optimizer=optimizer,
        checkpoint_dir="./test_checkpoints",
        auto_fix=True
    )
    
    # 模拟训练
    print("模拟训练循环...")
    
    for step in range(100):
        # 创建假数据
        batch = {
            'input': torch.randn(32, 10),
            'target': torch.randn(32, 1)
        }
        
        # 自定义前向函数
        def forward_fn(model, batch):
            pred = model(batch['input'])
            loss = nn.functional.mse_loss(pred, batch['target'])
            return loss
        
        # 执行训练步
        loss = wrapper.training_step(batch, forward_fn)
        
        # 模拟异常
        if step == 50:
            # 制造梯度爆炸
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data *= 100
    
    # 打印报告
    wrapper.print_report()
