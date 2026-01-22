#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Events System - 为WebUI和监控系统预留接口

这个模块提供了一个事件发射系统，允许外部系统（如WebUI、监控面板、日志系统）
订阅训练过程中的各种事件。

设计目的：
- 解耦训练逻辑和UI/监控逻辑
- 为将来的WebUI实现预留钩子
- 支持多个监听器同时订阅
- 不影响现有训练流程

使用示例：
    # 在训练器中发射事件
    from apt.apt_model.training.training_events import training_emitter

    training_emitter.emit('epoch_start', epoch=1, total_epochs=10)
    training_emitter.emit('batch_end', batch_idx=100, loss=2.5, lr=0.0001)

    # 在WebUI中订阅事件
    def on_batch_end(event_data):
        print(f"Batch {event_data['batch_idx']}: Loss = {event_data['loss']}")

    training_emitter.on('batch_end', on_batch_end)
"""

from typing import Dict, List, Callable, Any
from collections import defaultdict
import logging


class TrainingEventEmitter:
    """
    训练事件发射器

    支持的事件类型：
    - training_start: 训练开始
    - training_end: 训练结束
    - epoch_start: Epoch开始
    - epoch_end: Epoch结束
    - batch_start: Batch开始
    - batch_end: Batch结束
    - checkpoint_saved: Checkpoint已保存
    - checkpoint_loaded: Checkpoint已加载
    - metric_update: 指标更新
    - error_occurred: 错误发生
    """

    def __init__(self):
        """初始化事件发射器"""
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Dict[str, Any]] = []
        self._max_history = 1000  # 最多保存1000条事件历史
        self.logger = logging.getLogger('TrainingEvents')

    def on(self, event_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        订阅事件

        参数:
            event_name: 事件名称
            callback: 回调函数，接收event_data字典作为参数

        示例:
            def my_callback(event_data):
                print(f"Epoch {event_data['epoch']} finished")

            emitter.on('epoch_end', my_callback)
        """
        self._listeners[event_name].append(callback)
        self.logger.debug(f"订阅事件: {event_name}")

    def off(self, event_name: str, callback: Callable):
        """
        取消订阅事件

        参数:
            event_name: 事件名称
            callback: 要移除的回调函数
        """
        if event_name in self._listeners:
            try:
                self._listeners[event_name].remove(callback)
                self.logger.debug(f"取消订阅事件: {event_name}")
            except ValueError:
                pass

    def emit(self, event_name: str, **event_data):
        """
        发射事件

        参数:
            event_name: 事件名称
            **event_data: 事件数据（键值对）

        示例:
            emitter.emit('batch_end', batch_idx=100, loss=2.5, lr=0.0001)
        """
        # 添加时间戳
        import time
        event_data['timestamp'] = time.time()
        event_data['event_name'] = event_name

        # 保存到历史记录
        self._event_history.append(event_data)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # 触发所有监听器
        if event_name in self._listeners:
            for callback in self._listeners[event_name]:
                try:
                    callback(event_data)
                except Exception as e:
                    self.logger.error(f"事件处理器错误 ({event_name}): {e}")

    def get_history(self, event_name: str = None, limit: int = 100) -> List[Dict]:
        """
        获取事件历史

        参数:
            event_name: 可选，只获取特定事件的历史
            limit: 返回最近N条记录

        返回:
            事件历史列表
        """
        if event_name:
            history = [e for e in self._event_history if e['event_name'] == event_name]
        else:
            history = self._event_history

        return history[-limit:]

    def clear_listeners(self, event_name: str = None):
        """
        清空监听器

        参数:
            event_name: 可选，只清空特定事件的监听器
        """
        if event_name:
            if event_name in self._listeners:
                self._listeners[event_name].clear()
        else:
            self._listeners.clear()


# 全局训练事件发射器实例
# 训练器和外部系统（WebUI等）共享这个实例
training_emitter = TrainingEventEmitter()


# ==============================================================================
# WebUI钩子示例（预留接口）
# ==============================================================================

class WebUIHooks:
    """
    WebUI钩子示例类

    这个类展示了如何使用training_emitter为WebUI订阅训练事件。
    将来实现WebUI时，可以参考这个模式。

    使用方式:
        # 在WebUI服务器中
        webui_hooks = WebUIHooks()
        webui_hooks.attach(training_emitter)

        # 训练完成后
        webui_hooks.detach(training_emitter)
    """

    def __init__(self):
        """初始化WebUI钩子"""
        self.training_state = {
            'current_epoch': 0,
            'total_epochs': 0,
            'current_batch': 0,
            'total_batches': 0,
            'current_loss': 0.0,
            'learning_rate': 0.0,
            'is_training': False,
        }
        self.metrics_history = []

    def attach(self, emitter: TrainingEventEmitter):
        """
        附加到事件发射器

        参数:
            emitter: TrainingEventEmitter实例
        """
        emitter.on('training_start', self._on_training_start)
        emitter.on('training_end', self._on_training_end)
        emitter.on('epoch_start', self._on_epoch_start)
        emitter.on('epoch_end', self._on_epoch_end)
        emitter.on('batch_end', self._on_batch_end)
        emitter.on('checkpoint_saved', self._on_checkpoint_saved)
        emitter.on('metric_update', self._on_metric_update)

    def detach(self, emitter: TrainingEventEmitter):
        """
        从事件发射器分离

        参数:
            emitter: TrainingEventEmitter实例
        """
        emitter.off('training_start', self._on_training_start)
        emitter.off('training_end', self._on_training_end)
        emitter.off('epoch_start', self._on_epoch_start)
        emitter.off('epoch_end', self._on_epoch_end)
        emitter.off('batch_end', self._on_batch_end)
        emitter.off('checkpoint_saved', self._on_checkpoint_saved)
        emitter.off('metric_update', self._on_metric_update)

    def _on_training_start(self, event_data):
        """训练开始回调"""
        self.training_state['is_training'] = True
        self.training_state['total_epochs'] = event_data.get('total_epochs', 0)
        # WebUI可以在这里更新界面状态

    def _on_training_end(self, event_data):
        """训练结束回调"""
        self.training_state['is_training'] = False
        # WebUI可以在这里显示训练完成通知

    def _on_epoch_start(self, event_data):
        """Epoch开始回调"""
        self.training_state['current_epoch'] = event_data.get('epoch', 0)
        # WebUI可以在这里更新epoch进度条

    def _on_epoch_end(self, event_data):
        """Epoch结束回调"""
        # WebUI可以在这里更新epoch统计数据
        pass

    def _on_batch_end(self, event_data):
        """Batch结束回调"""
        self.training_state['current_batch'] = event_data.get('batch_idx', 0)
        self.training_state['current_loss'] = event_data.get('loss', 0.0)
        self.training_state['learning_rate'] = event_data.get('lr', 0.0)
        # WebUI可以在这里实时更新loss曲线图

    def _on_checkpoint_saved(self, event_data):
        """Checkpoint保存回调"""
        # WebUI可以在这里显示checkpoint保存通知
        pass

    def _on_metric_update(self, event_data):
        """指标更新回调"""
        self.metrics_history.append(event_data)
        # WebUI可以在这里更新指标仪表盘

    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前训练状态（用于WebUI API）

        返回:
            当前训练状态字典
        """
        return self.training_state.copy()


# ==============================================================================
# 便捷函数
# ==============================================================================

def emit_training_start(total_epochs: int, **kwargs):
    """便捷函数：发射训练开始事件"""
    training_emitter.emit('training_start', total_epochs=total_epochs, **kwargs)


def emit_training_end(**kwargs):
    """便捷函数：发射训练结束事件"""
    training_emitter.emit('training_end', **kwargs)


def emit_epoch_start(epoch: int, total_epochs: int, **kwargs):
    """便捷函数：发射epoch开始事件"""
    training_emitter.emit('epoch_start', epoch=epoch, total_epochs=total_epochs, **kwargs)


def emit_epoch_end(epoch: int, metrics: Dict[str, float], **kwargs):
    """便捷函数：发射epoch结束事件"""
    training_emitter.emit('epoch_end', epoch=epoch, metrics=metrics, **kwargs)


def emit_batch_end(batch_idx: int, loss: float, lr: float, **kwargs):
    """便捷函数：发射batch结束事件"""
    training_emitter.emit('batch_end', batch_idx=batch_idx, loss=loss, lr=lr, **kwargs)


def emit_checkpoint_saved(checkpoint_path: str, epoch: int, step: int, **kwargs):
    """便捷函数：发射checkpoint保存事件"""
    training_emitter.emit('checkpoint_saved',
                         checkpoint_path=checkpoint_path,
                         epoch=epoch,
                         step=step,
                         **kwargs)


def emit_metric_update(metrics: Dict[str, float], **kwargs):
    """便捷函数：发射指标更新事件"""
    training_emitter.emit('metric_update', metrics=metrics, **kwargs)
