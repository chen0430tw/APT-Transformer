#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Route Optimizer Plugin

Performance tier plugin for optimizing routing decisions in MoE models.

Priority: 200 (Performance tier)
Events: on_batch_start, on_step_end
Capabilities: route_suggest, read_metrics
"""

import logging
from typing import Dict, Any, List
from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)

logger = logging.getLogger(__name__)


class RouteOptimizerPlugin(PluginBase):
    """
    Route Optimizer Plugin

    Optimizes routing decisions for Mixture-of-Experts (MoE) models.

    Features:
    - Monitors expert load distribution
    - Suggests routing adjustments for load balancing
    - Tracks routing efficiency metrics
    - Prevents expert overload
    """

    def __init__(self):
        """初始化 Route Optimizer 插件"""
        super().__init__()
        self.num_experts = 8  # 默认 8 个专家
        self.expert_loads = [0] * self.num_experts
        self.routing_history: List[Dict[str, Any]] = []
        self.metrics = {
            'load_variance': 0.0,
            'routing_efficiency': 1.0,
            'adjustments_made': 0,
            'overload_events': 0
        }
        self.load_threshold = 1.5  # 负载阈值（平均值的 1.5 倍）

    def get_manifest(self) -> PluginManifest:
        """
        获取插件清单

        Returns:
            插件清单
        """
        return PluginManifest(
            name="route_optimizer",
            version="1.0.0",
            description="Route optimizer for MoE load balancing",
            author="APT Team",
            priority=PluginPriority.THROUGHPUT,
            blocking=True,  # 需要阻塞以调整路由
            events=[
                PluginEvent.ON_BATCH_START,
                PluginEvent.ON_STEP_END,
                PluginEvent.ON_EPOCH_END
            ],
            requires=[
                "core:trainer",
            ],
            conflicts=[],
            capabilities=[
                PluginCapability.ROUTE_SUGGEST,  # 路由建议（非独占）
                PluginCapability.READ_METRICS,
                PluginCapability.READ_STATE
            ],
            resources={
                "cpu_ms": 8.0,   # 路由计算
                "gpu_ms": 2.0,
                "io_mb": 0.3
            },
            rate_limit={
                "steps": 1  # 每步执行
            },
            sandbox=True,
            fail_limit=5,
            s_default=0.5,  # 中等净效用
            eta=1.0
        )

    def initialize(self, config: Dict[str, Any] = None):
        """
        初始化插件

        Args:
            config: 配置字典
        """
        if config:
            self.num_experts = config.get('num_experts', 8)
            self.load_threshold = config.get('load_threshold', 1.5)
            self.expert_loads = [0] * self.num_experts
            logger.info(f"Route Optimizer initialized with num_experts={self.num_experts}, "
                       f"threshold={self.load_threshold}")

    def on_batch_start(self, context: Dict[str, Any]):
        """
        Batch 开始时处理

        Args:
            context: 事件上下文
        """
        step = context.get('step', 0)
        data = context.get('data', {})

        # 重置当前 batch 的负载统计
        self.expert_loads = [0] * self.num_experts

        # 读取路由信息（如果有）
        routing_info = data.get('routing', {})
        if routing_info:
            expert_ids = routing_info.get('expert_ids', [])
            for expert_id in expert_ids:
                if 0 <= expert_id < self.num_experts:
                    self.expert_loads[expert_id] += 1

        # 提供路由建议（基于历史负载）
        if self.routing_history:
            suggestions = self._generate_routing_suggestions()
            data['routing_suggestions'] = suggestions

            logger.debug(f"[Route Optimizer] Step {step}: Provided routing suggestions")

    def on_step_end(self, context: Dict[str, Any]):
        """
        Step 结束时处理

        Args:
            context: 事件上下文
        """
        step = context.get('step', 0)
        data = context.get('data', {})

        # 计算负载统计
        if sum(self.expert_loads) > 0:
            mean_load = sum(self.expert_loads) / self.num_experts
            variance = sum((load - mean_load) ** 2 for load in self.expert_loads) / self.num_experts
            self.metrics['load_variance'] = variance

            # 检查是否有专家过载
            max_load = max(self.expert_loads)
            if max_load > mean_load * self.load_threshold:
                self.metrics['overload_events'] += 1
                logger.warning(f"[Route Optimizer] Step {step}: Expert overload detected "
                             f"(max_load={max_load:.1f}, mean={mean_load:.1f})")

            # 计算路由效率（均匀度）
            if variance > 0:
                efficiency = 1.0 / (1.0 + variance)
                self.metrics['routing_efficiency'] = efficiency
            else:
                self.metrics['routing_efficiency'] = 1.0

            # 记录历史
            self.routing_history.append({
                'step': step,
                'loads': self.expert_loads.copy(),
                'variance': variance,
                'efficiency': self.metrics['routing_efficiency']
            })

            # 限制历史记录长度
            if len(self.routing_history) > 100:
                self.routing_history = self.routing_history[-100:]

            # 写入到公共上下文
            if 'metrics' not in data:
                data['metrics'] = {}
            data['metrics']['route_variance'] = variance
            data['metrics']['route_efficiency'] = self.metrics['routing_efficiency']

            logger.debug(f"[Route Optimizer] Step {step}: variance={variance:.4f}, "
                        f"efficiency={self.metrics['routing_efficiency']:.4f}")

    def on_epoch_end(self, context: Dict[str, Any]):
        """
        Epoch 结束时处理

        Args:
            context: 事件上下文
        """
        epoch = context.get('epoch', 0)

        # 计算 epoch 级别的统计
        if self.routing_history:
            avg_variance = sum(r['variance'] for r in self.routing_history) / len(self.routing_history)
            avg_efficiency = sum(r['efficiency'] for r in self.routing_history) / len(self.routing_history)

            logger.info(f"[Route Optimizer] Epoch {epoch} completed: "
                       f"avg_variance={avg_variance:.4f}, "
                       f"avg_efficiency={avg_efficiency:.4f}, "
                       f"overload_events={self.metrics['overload_events']}, "
                       f"adjustments={self.metrics['adjustments_made']}")

        # 重置部分指标
        self.metrics['overload_events'] = 0
        self.metrics['adjustments_made'] = 0

    def _generate_routing_suggestions(self) -> Dict[str, Any]:
        """
        生成路由建议

        Returns:
            路由建议字典
        """
        if not self.routing_history:
            return {}

        # 基于最近的负载历史生成建议
        recent = self.routing_history[-10:]
        avg_loads = [0.0] * self.num_experts

        for record in recent:
            loads = record['loads']
            for i, load in enumerate(loads):
                avg_loads[i] += load

        avg_loads = [load / len(recent) for load in avg_loads]

        # 找出负载最低和最高的专家
        min_load_idx = avg_loads.index(min(avg_loads))
        max_load_idx = avg_loads.index(max(avg_loads))

        suggestions = {
            'underloaded_expert': min_load_idx,
            'overloaded_expert': max_load_idx,
            'avg_loads': avg_loads,
            'recommendation': 'redirect' if avg_loads[max_load_idx] > avg_loads[min_load_idx] * 2 else 'balanced'
        }

        self.metrics['adjustments_made'] += 1
        return suggestions

    def cleanup(self):
        """清理资源"""
        logger.info("Route Optimizer Plugin cleanup")
        self.routing_history.clear()
