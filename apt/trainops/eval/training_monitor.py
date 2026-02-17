"""
APT训练自动监控和纠错系统
基于SOSA算法的智能训练助手

功能:
1. 实时监控训练指标
2. 自动检测异常
3. 智能诊断问题
4. 自动应用修复策略
5. 学习最优配置

作者: chen0430tw
"""

from apt.core.fake_torch import get_torch
torch = get_torch()
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass, field
import time
from enum import Enum

from .sosa_core import SOSA, Event

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型"""
    NAN_LOSS = "nan_loss"
    EXPLODING_GRADIENT = "exploding_gradient"
    VANISHING_GRADIENT = "vanishing_gradient"
    OOM = "out_of_memory"
    DIVERGENCE = "divergence"
    OSCILLATION = "oscillation"
    PLATEAU = "plateau"
    UNKNOWN = "unknown"


@dataclass
class TrainingSnapshot:
    """训练快照"""
    step: int
    loss: float
    grad_norm: Optional[float] = None
    lr: Optional[float] = None
    memory_used: Optional[float] = None
    throughput: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class FixAction:
    """修复动作"""
    action_type: str  # 'reduce_lr', 'clip_grad', 'reduce_batch', 'warmup', 'reload'
    parameters: Dict
    reason: str
    confidence: float


class TrainingMonitor:
    """
    训练监控器
    
    监控内容:
    - Loss趋势
    - 梯度统计
    - 内存使用
    - 吞吐量
    - 异常模式
    """
    
    def __init__(
        self,
        sosa_window: float = 10.0,
        history_size: int = 1000
    ):
        """
        Args:
            sosa_window: SOSA时间窗口(秒)
            history_size: 历史记录大小
        """
        # SOSA引擎
        self.sosa = SOSA(dt_window=sosa_window, M_groups=10)
        
        # 训练历史
        self.history: List[TrainingSnapshot] = []
        self.history_size = history_size
        
        # 异常计数
        self.error_counts: Dict[ErrorType, int] = {e: 0 for e in ErrorType}
        
        # 修复历史
        self.fix_history: List[Tuple[int, FixAction]] = []
        
        # 统计
        self.total_steps = 0
        self.anomaly_steps = 0
        
        logger.info(f"训练监控器初始化: window={sosa_window}s")
    
    # ==================== 监控 ====================
    
    def log_step(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        memory_used: Optional[float] = None,
        throughput: Optional[float] = None
    ):
        """记录训练步"""
        snapshot = TrainingSnapshot(
            step=step,
            loss=loss,
            grad_norm=grad_norm,
            lr=lr,
            memory_used=memory_used,
            throughput=throughput
        )
        
        # 添加到历史
        self.history.append(snapshot)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # 转为SOSA事件
        event = self._snapshot_to_event(snapshot)
        self.sosa.add_event(event)
        
        self.total_steps += 1
    
    def _snapshot_to_event(self, snapshot: TrainingSnapshot) -> Event:
        """训练快照转SOSA事件"""
        # 计算严重程度
        severity = 0.0
        event_type = 'metric'
        
        # Loss相关
        if np.isnan(snapshot.loss) or np.isinf(snapshot.loss):
            severity = 1.0
            event_type = 'error'
        elif snapshot.loss > 10.0:
            severity = 0.8
            event_type = 'warning'
        elif snapshot.loss > 5.0:
            severity = 0.5
        else:
            severity = snapshot.loss / 10.0
        
        # 梯度相关
        if snapshot.grad_norm is not None:
            if snapshot.grad_norm > 100.0:
                severity = max(severity, 0.9)
                event_type = 'warning'
            elif snapshot.grad_norm < 1e-6:
                severity = max(severity, 0.7)
                event_type = 'warning'
        
        return Event(
            timestamp=snapshot.timestamp,
            event_type=event_type,
            severity=min(severity, 1.0),
            attributes={
                'step': snapshot.step,
                'loss': snapshot.loss,
                'grad_norm': snapshot.grad_norm,
                'lr': snapshot.lr
            },
            value=snapshot.loss
        )
    
    # ==================== 异常检测 ====================
    
    def detect_error(self) -> Optional[ErrorType]:
        """
        检测错误类型
        
        Returns:
            ErrorType or None
        """
        if len(self.history) < 2:
            return None
        
        current = self.history[-1]
        
        # NaN Loss
        if np.isnan(current.loss) or np.isinf(current.loss):
            self.error_counts[ErrorType.NAN_LOSS] += 1
            return ErrorType.NAN_LOSS
        
        # Exploding Gradient
        if current.grad_norm is not None and current.grad_norm > 100.0:
            self.error_counts[ErrorType.EXPLODING_GRADIENT] += 1
            return ErrorType.EXPLODING_GRADIENT
        
        # Vanishing Gradient
        if current.grad_norm is not None and current.grad_norm < 1e-7:
            self.error_counts[ErrorType.VANISHING_GRADIENT] += 1
            return ErrorType.VANISHING_GRADIENT
        
        # Divergence (loss急剧上升)
        if len(self.history) >= 10:
            recent_losses = [s.loss for s in self.history[-10:]]
            if all(np.isfinite(recent_losses)):
                trend = np.polyfit(range(10), recent_losses, 1)[0]
                if trend > 0.5:  # 急剧上升
                    self.error_counts[ErrorType.DIVERGENCE] += 1
                    return ErrorType.DIVERGENCE
        
        # Oscillation (loss剧烈震荡)
        if len(self.history) >= 20:
            recent_losses = [s.loss for s in self.history[-20:]]
            if all(np.isfinite(recent_losses)):
                std = np.std(recent_losses)
                mean = np.mean(recent_losses)
                if std / (mean + 1e-8) > 0.5:
                    self.error_counts[ErrorType.OSCILLATION] += 1
                    return ErrorType.OSCILLATION
        
        # Plateau (loss停滞)
        if len(self.history) >= 50:
            recent_losses = [s.loss for s in self.history[-50:]]
            if all(np.isfinite(recent_losses)):
                std = np.std(recent_losses)
                if std < 1e-5:
                    self.error_counts[ErrorType.PLATEAU] += 1
                    return ErrorType.PLATEAU
        
        # SOSA异常检测
        if self.sosa.detect_anomaly(threshold=0.85):
            self.anomaly_steps += 1
            self.error_counts[ErrorType.UNKNOWN] += 1
            return ErrorType.UNKNOWN
        
        return None
    
    # ==================== 诊断 ====================
    
    def diagnose(self, error_type: ErrorType) -> str:
        """
        诊断问题原因
        
        Returns:
            诊断报告
        """
        if len(self.history) == 0:
            return "历史数据不足，无法诊断"
        
        current = self.history[-1]
        
        diagnosis = f"检测到: {error_type.value}\n"
        
        if error_type == ErrorType.NAN_LOSS:
            diagnosis += "可能原因:\n"
            diagnosis += "  - 学习率过大\n"
            diagnosis += "  - 数值不稳定\n"
            diagnosis += "  - 数据异常\n"
            if current.lr is not None and current.lr > 1e-3:
                diagnosis += f"  - 当前学习率 {current.lr:.2e} 偏高\n"
        
        elif error_type == ErrorType.EXPLODING_GRADIENT:
            diagnosis += "可能原因:\n"
            diagnosis += "  - 学习率过大\n"
            diagnosis += "  - 梯度裁剪不足\n"
            diagnosis += f"  - 当前梯度范数: {current.grad_norm:.2f}\n"
        
        elif error_type == ErrorType.VANISHING_GRADIENT:
            diagnosis += "可能原因:\n"
            diagnosis += "  - 学习率过小\n"
            diagnosis += "  - 网络过深\n"
            diagnosis += "  - 激活函数不当\n"
            diagnosis += f"  - 当前梯度范数: {current.grad_norm:.2e}\n"
        
        elif error_type == ErrorType.DIVERGENCE:
            diagnosis += "可能原因:\n"
            diagnosis += "  - 学习率过大\n"
            diagnosis += "  - batch size过小\n"
            diagnosis += "  - 数据分布变化\n"
        
        elif error_type == ErrorType.OSCILLATION:
            diagnosis += "可能原因:\n"
            diagnosis += "  - 学习率不稳定\n"
            diagnosis += "  - batch size过小\n"
            diagnosis += "  - 优化器参数不当\n"
        
        elif error_type == ErrorType.PLATEAU:
            diagnosis += "可能原因:\n"
            diagnosis += "  - 学习率过小\n"
            diagnosis += "  - 陷入局部最优\n"
            diagnosis += "  - 数据不足\n"
        
        return diagnosis
    
    # ==================== 自动修复 ====================
    
    def suggest_fix(self, error_type: ErrorType) -> FixAction:
        """
        建议修复方案
        
        Returns:
            FixAction
        """
        if len(self.history) == 0:
            return FixAction(
                action_type='none',
                parameters={},
                reason='历史数据不足',
                confidence=0.0
            )
        
        current = self.history[-1]
        
        # 根据错误类型选择策略
        if error_type == ErrorType.NAN_LOSS:
            return FixAction(
                action_type='reduce_lr',
                parameters={'factor': 0.1, 'reload_checkpoint': True},
                reason='NaN loss: 大幅降低学习率并回滚',
                confidence=0.9
            )
        
        elif error_type == ErrorType.EXPLODING_GRADIENT:
            return FixAction(
                action_type='clip_grad',
                parameters={'max_norm': 1.0},
                reason='梯度爆炸: 强化梯度裁剪',
                confidence=0.95
            )
        
        elif error_type == ErrorType.VANISHING_GRADIENT:
            return FixAction(
                action_type='increase_lr',
                parameters={'factor': 1.5},
                reason='梯度消失: 适当提高学习率',
                confidence=0.7
            )
        
        elif error_type == ErrorType.DIVERGENCE:
            return FixAction(
                action_type='reduce_lr',
                parameters={'factor': 0.5},
                reason='Loss发散: 降低学习率',
                confidence=0.85
            )
        
        elif error_type == ErrorType.OSCILLATION:
            return FixAction(
                action_type='reduce_lr',
                parameters={'factor': 0.7, 'increase_batch': True},
                reason='Loss震荡: 降低学习率并增大batch',
                confidence=0.8
            )
        
        elif error_type == ErrorType.PLATEAU:
            # 使用SOSA决策
            decision = self.sosa.decide_next_action()
            
            if decision['exploration_factor'] > 0.6:
                # 高探索: 尝试新策略
                return FixAction(
                    action_type='exploration',
                    parameters={
                        'increase_lr': True,
                        'factor': 1.2,
                        'add_noise': True
                    },
                    reason='Loss停滞: SOSA建议探索新策略',
                    confidence=decision.get('confidence', 0.5)
                )
            else:
                # 低探索: 微调当前策略
                return FixAction(
                    action_type='fine_tune',
                    parameters={'adjust_lr': True, 'factor': 1.1},
                    reason='Loss停滞: SOSA建议微调',
                    confidence=decision.get('confidence', 0.5)
                )
        
        else:
            # 未知错误: 使用SOSA推荐
            decision = self.sosa.decide_next_action()
            
            return FixAction(
                action_type='sosa_guided',
                parameters={
                    'exploration_factor': decision['exploration_factor']
                },
                reason='SOSA引导策略',
                confidence=decision.get('confidence', 0.3)
            )
    
    def apply_fix(
        self,
        fix_action: FixAction,
        optimizer: torch.optim.Optimizer,
        model: Optional[torch.nn.Module] = None
    ) -> bool:
        """
        应用修复
        
        Args:
            fix_action: 修复动作
            optimizer: 优化器
            model: 模型 (可选)
        
        Returns:
            是否成功应用
        """
        try:
            action_type = fix_action.action_type
            params = fix_action.parameters
            
            logger.info(f"应用修复: {action_type} - {fix_action.reason}")
            
            if action_type == 'reduce_lr':
                factor = params.get('factor', 0.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                logger.info(f"学习率降低为原来的 {factor}")
                
                if params.get('reload_checkpoint') and hasattr(model, 'load_checkpoint'):
                    logger.info("回滚到最近检查点")
                    # 这里需要外部传入检查点路径
                    # model.load_checkpoint(...)
            
            elif action_type == 'increase_lr':
                factor = params.get('factor', 1.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                logger.info(f"学习率提高为原来的 {factor}")
            
            elif action_type == 'clip_grad':
                # 这个需要在训练循环中调用
                max_norm = params.get('max_norm', 1.0)
                logger.info(f"设置梯度裁剪: max_norm={max_norm}")
                # 返回参数供外部使用
                fix_action.parameters['_applied_max_norm'] = max_norm
            
            elif action_type == 'exploration':
                # 探索性调整
                if params.get('increase_lr'):
                    factor = params.get('factor', 1.2)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= factor
                
                if params.get('add_noise') and model is not None:
                    # 添加参数噪声鼓励探索
                    with torch.no_grad():
                        for param in model.parameters():
                            if param.requires_grad:
                                noise = torch.randn_like(param) * 0.001
                                param.add_(noise)
                
                logger.info("应用探索性调整")
            
            # 记录修复历史
            if len(self.history) > 0:
                step = self.history[-1].step
                self.fix_history.append((step, fix_action))
            
            return True
            
        except Exception as e:
            logger.error(f"应用修复失败: {e}")
            return False
    
    # ==================== 统计报告 ====================
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_steps': self.total_steps,
            'anomaly_steps': self.anomaly_steps,
            'anomaly_rate': self.anomaly_steps / max(self.total_steps, 1),
            'error_counts': {e.value: c for e, c in self.error_counts.items()},
            'num_fixes': len(self.fix_history),
            'sosa_stats': self.sosa.get_statistics()
        }
        
        if len(self.history) > 0:
            recent_losses = [s.loss for s in self.history[-100:] if np.isfinite(s.loss)]
            if recent_losses:
                stats['recent_loss_mean'] = np.mean(recent_losses)
                stats['recent_loss_std'] = np.std(recent_losses)
                stats['recent_loss_min'] = np.min(recent_losses)
        
        return stats
    
    def print_report(self):
        """打印监控报告"""
        stats = self.get_statistics()
        
        print("=" * 70)
        print("训练监控报告")
        print("=" * 70)
        
        print(f"\n训练进度:")
        print(f"  总步数: {stats['total_steps']}")
        print(f"  异常步数: {stats['anomaly_steps']}")
        print(f"  异常率: {stats['anomaly_rate']:.2%}")
        
        print(f"\n错误统计:")
        for error_type, count in stats['error_counts'].items():
            if count > 0:
                print(f"  {error_type}: {count} 次")
        
        print(f"\n修复历史: {stats['num_fixes']} 次")
        if len(self.fix_history) > 0:
            print("  最近5次修复:")
            for step, fix in self.fix_history[-5:]:
                print(f"    Step {step}: {fix.action_type} - {fix.reason}")
        
        if 'recent_loss_mean' in stats:
            print(f"\n近期Loss:")
            print(f"  均值: {stats['recent_loss_mean']:.4f}")
            print(f"  标准差: {stats['recent_loss_std']:.4f}")
            print(f"  最小值: {stats['recent_loss_min']:.4f}")
        
        print("\n" + "=" * 70)
        print("SOSA 引擎状态")
        print("=" * 70)
        self.sosa.print_report()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== 训练监控器测试 ===\n")
    
    # 创建监控器
    monitor = TrainingMonitor(sosa_window=5.0)
    
    # 模拟训练过程
    print("模拟正常训练...")
    for step in range(100):
        loss = 2.0 * np.exp(-step / 50) + 0.1 * np.random.randn()
        grad_norm = 1.0 + 0.5 * np.random.randn()
        
        monitor.log_step(
            step=step,
            loss=max(loss, 0),
            grad_norm=abs(grad_norm),
            lr=1e-4
        )
        
        # 检测异常
        error = monitor.detect_error()
        if error:
            print(f"\n[Step {step}] 检测到异常: {error.value}")
            
            # 诊断
            diagnosis = monitor.diagnose(error)
            print(diagnosis)
            
            # 建议修复
            fix = monitor.suggest_fix(error)
            print(f"建议修复: {fix.action_type}")
            print(f"  参数: {fix.parameters}")
            print(f"  置信度: {fix.confidence:.2f}")
    
    # 注入异常
    print("\n\n模拟异常: 梯度爆炸...")
    for step in range(100, 110):
        loss = 2.0 + (step - 100) * 0.5  # 急剧上升
        grad_norm = 50.0 + (step - 100) * 10  # 梯度爆炸
        
        monitor.log_step(
            step=step,
            loss=loss,
            grad_norm=grad_norm,
            lr=1e-4
        )
        
        error = monitor.detect_error()
        if error:
            print(f"\n[Step {step}] 检测到异常: {error.value}")
            fix = monitor.suggest_fix(error)
            print(f"建议修复: {fix.action_type} - {fix.reason}")
    
    # 打印报告
    print("\n")
    monitor.print_report()
