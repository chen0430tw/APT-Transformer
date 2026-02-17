#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EQI Reporter Plugin

Telemetry tier plugin for reporting EQI (Evidence Qualitative Inference) metrics.

Priority: 820 (Telemetry tier)
Events: on_epoch_end, on_step_eval
Capabilities: read_metrics, write_metrics
"""

import logging
import time
from typing import Dict, Any, List
from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)

logger = logging.getLogger(__name__)


class EQIReporterPlugin(PluginBase):
    """
    EQI Reporter Plugin

    Reports EQI-related metrics and evidence for decision tracking.

    Features:
    - Collects plugin activation evidence
    - Tracks net utility (s = L - λI) trends
    - Reports soft gate activations φ(s, E, κ)
    - Logs stability metrics
    """

    def __init__(self):
        """初始化 EQI Reporter 插件"""
        super().__init__()
        self.evidence_log: List[Dict[str, Any]] = []
        self.utility_history: List[float] = []
        self.report_interval = 100  # 每 100 步报告一次
        self.metrics = {
            'evidence_mean': 0.0,
            'utility_mean': 0.0,
            'activations': 0,
            'reports_sent': 0
        }

    def get_manifest(self) -> PluginManifest:
        """
        获取插件清单

        Returns:
            插件清单
        """
        return PluginManifest(
            name="eqi_reporter",
            version="1.0.0",
            description="EQI metrics reporter for evidence and decision tracking",
            author="APT Team",
            priority=PluginPriority.TRACING,
            blocking=False,  # 非阻塞，异步上报
            events=[
                PluginEvent.ON_STEP_EVAL,
                PluginEvent.ON_EPOCH_END,
            ],
            requires=[],
            conflicts=[],
            capabilities=[
                PluginCapability.READ_METRICS,
                PluginCapability.WRITE_METRICS
            ],
            resources={
                "cpu_ms": 3.0,   # 轻量级上报
                "gpu_ms": 0.0,
                "io_mb": 0.2
            },
            rate_limit={
                "steps": 10  # 每 10 步最多触发一次
            },
            sandbox=True,
            fail_limit=10,
            s_default=-0.1,  # 默认净效用较低（telemetry 非关键）
            eta=0.8          # 证据调制参数
        )

    def initialize(self, config: Dict[str, Any] = None):
        """
        初始化插件

        Args:
            config: 配置字典
        """
        if config:
            self.report_interval = config.get('report_interval', 100)
            logger.info(f"EQI Reporter initialized with report_interval={self.report_interval}")

    def on_step_eval(self, context: Dict[str, Any]):
        """
        Step 评估时处理

        Args:
            context: 事件上下文
        """
        step = context.get('step', 0)
        data = context.get('data', {})

        # 读取指标
        metrics = data.get('metrics', {})

        # 收集证据
        evidence_entry = {
            'step': step,
            'timestamp': time.time(),
            'metrics': metrics.copy(),
            'evidence': data.get('evidence', 1.0),
            'utility': data.get('utility', 0.0)
        }

        self.evidence_log.append(evidence_entry)
        self.metrics['activations'] += 1

        # 计算移动平均
        if len(self.evidence_log) > 10:
            recent_evidence = [e['evidence'] for e in self.evidence_log[-10:]]
            recent_utility = [e['utility'] for e in self.evidence_log[-10:]]
            self.metrics['evidence_mean'] = sum(recent_evidence) / len(recent_evidence)
            self.metrics['utility_mean'] = sum(recent_utility) / len(recent_utility)

        # 定期报告
        if step % self.report_interval == 0 and step > 0:
            self._send_report(step)

    def on_epoch_end(self, context: Dict[str, Any]):
        """
        Epoch 结束时处理

        Args:
            context: 事件上下文
        """
        epoch = context.get('epoch', 0)
        step = context.get('step', 0)

        # Epoch 结束时强制发送报告
        self._send_report(step, epoch=epoch)

        logger.info(f"[EQI Reporter] Epoch {epoch} completed: "
                   f"activations={self.metrics['activations']}, "
                   f"reports={self.metrics['reports_sent']}, "
                   f"avg_evidence={self.metrics['evidence_mean']:.4f}, "
                   f"avg_utility={self.metrics['utility_mean']:.4f}")

        # 清理旧日志（保留最近 1000 条）
        if len(self.evidence_log) > 1000:
            self.evidence_log = self.evidence_log[-1000:]

    def _send_report(self, step: int, epoch: int = None):
        """
        发送报告

        Args:
            step: 当前步数
            epoch: 当前 epoch（可选）
        """
        report = {
            'step': step,
            'epoch': epoch,
            'timestamp': time.time(),
            'evidence_mean': self.metrics['evidence_mean'],
            'utility_mean': self.metrics['utility_mean'],
            'activations': self.metrics['activations'],
            'log_size': len(self.evidence_log)
        }

        # 在实际应用中，这里会发送到监控系统
        # 这里只是记录日志
        logger.info(f"[EQI Reporter] Report sent: {report}")

        self.metrics['reports_sent'] += 1

        # 存储到上下文
        self.set_context('last_report', report)

    def cleanup(self):
        """清理资源"""
        logger.info("EQI Reporter Plugin cleanup")
        # 发送最后一次报告
        if self.evidence_log:
            self._send_report(step=-1)
        self.evidence_log.clear()
