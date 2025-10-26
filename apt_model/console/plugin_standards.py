#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plugin Standards (插件标准)

基于 memo.txt 中定义的插件标准，实现：
- 插件优先级分类
- 插件 Manifest 规范
- 插件基类
- 事件定义

符合 APT 插件生态的标准规范。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, IntEnum


# ============================================================================
# 插件优先级标准 (Priority Classes)
# ============================================================================

class PluginPriority(IntEnum):
    """
    插件优先级分类

    基于 memo.txt 中的优先级标准：
    - Critical: 0-49 (Kill-switch、配置锁、权限校验)
    - CoreRuntime: 50-149 (推理控制器、解码策略、MoE负载均衡)
    - Performance: 150-249 (梯度裁剪、显存调度)
    - Reasoning: 250-349 (Leaf-Vote、自洽重评分)
    - Training: 350-449 (GRPO/RLHF/DPO/ORPO)
    - DecisionEQI: 450-549 (EQI、资源优化)
    - AdminAudit: 550-649 (审计、日志、合规)
    - Experimental: 650-799 (试验性算子)
    - Telemetry: 800-899 (指标上报)
    - PostCleanup: 900-999 (缓存清理)
    """
    # Critical段位 (0-49)
    CRITICAL = 10
    KILL_SWITCH = 5
    CONFIG_LOCK = 15
    PERMISSION = 20

    # Core Runtime段位 (50-149)
    CORE_RUNTIME = 100
    DECODER = 80
    MOE_BALANCER = 90
    INFERENCE_CONTROLLER = 100

    # Performance段位 (150-249)
    PERFORMANCE = 200
    GRAD_CLIP = 180
    MEMORY_OPT = 190
    THROUGHPUT = 200

    # Reasoning段位 (250-349)
    REASONING = 300
    LEAF_VOTE = 280
    SC_DECODE = 280            # Self-Consistency Decode
    SELF_CONSISTENCY = 300
    BEAM_SEARCH = 300          # Beam Search Reasoning
    REASONING_CHAIN = 320
    PROG_REASON = 320          # Program-Aided Reasoning

    # Training段位 (350-449)
    TRAINING = 400
    GRPO = 380
    RLHF = 390
    DPO = 400
    ORPO = 410

    # Decision/EQI段位 (450-549)
    DECISION_EQI = 500
    EQI = 500
    RESOURCE_OPT = 510
    QUOTA = 520

    # Admin/Audit段位 (550-649)
    ADMIN_AUDIT = 600
    AUDIT = 580
    COMPLIANCE = 600
    LOGGING = 620

    # Experimental段位 (650-799)
    EXPERIMENTAL = 700
    RESEARCH = 680
    TRIAL = 720

    # Telemetry段位 (800-899)
    TELEMETRY = 820
    METRICS = 800
    TRACING = 820
    MONITORING = 840

    # Post/Cleanup段位 (900-999)
    POST_CLEANUP = 950
    CACHE_CLEAN = 930
    SNAPSHOT = 960


# ============================================================================
# 事件定义 (Event Types)
# ============================================================================

class PluginEvent:
    """
    插件事件定义

    定义了插件可以订阅的所有事件类型。
    """
    # 训练生命周期事件
    ON_TRAIN_START = "on_train_start"
    ON_TRAIN_END = "on_train_end"

    # Epoch 级别事件
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"

    # Batch 级别事件
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"

    # Step 级别事件
    ON_STEP_START = "on_step_start"
    ON_STEP_END = "on_step_end"
    ON_STEP_EVAL = "on_step_eval"

    # 评估事件
    ON_EVAL_START = "on_eval_start"
    ON_EVAL_END = "on_eval_end"

    # 错误处理事件
    ON_FAIL = "on_fail"
    ON_EXCEPTION = "on_exception"

    # 检查点事件
    ON_SAVE_CHECKPOINT = "on_save_checkpoint"
    ON_LOAD_CHECKPOINT = "on_load_checkpoint"

    # 模型事件
    ON_MODEL_FORWARD = "on_model_forward"
    ON_MODEL_BACKWARD = "on_model_backward"

    @classmethod
    def all_events(cls) -> List[str]:
        """返回所有事件名称"""
        return [
            getattr(cls, attr)
            for attr in dir(cls)
            if attr.startswith('ON_') and isinstance(getattr(cls, attr), str)
        ]


# ============================================================================
# 插件 Manifest (Plugin Manifest)
# ============================================================================

@dataclass
class PluginManifest:
    """
    插件清单 (Plugin Manifest)

    基于 memo.txt 中的 manifest 规范，包含插件的所有元数据。

    示例:
        manifest = PluginManifest(
            name="eqi",
            version="1.2.0",
            priority=PluginPriority.EQI,
            blocking=True,
            events=[PluginEvent.ON_EPOCH_END, PluginEvent.ON_STEP_EVAL],
            requires=["core:trainer", "plugin:admin"],
            conflicts=["plugin:eqi_legacy", "capability:route_override"],
            capabilities=["add_constraints", "read_metrics", "route_suggest"],
            resources={"cpu_ms": 20, "gpu_ms": 0, "io_mb": 1},
            rate_limit={"steps": 100},
            sandbox=True
        )
    """
    # 基本信息
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""

    # 优先级和行为
    priority: int = PluginPriority.EXPERIMENTAL
    blocking: bool = False  # 是否允许阻塞主线程

    # 事件订阅
    events: List[str] = field(default_factory=list)

    # 依赖和冲突
    requires: List[str] = field(default_factory=list)  # 软依赖（如 "plugin:admin", "core:trainer"）
    conflicts: List[str] = field(default_factory=list)  # 硬冲突（如 "plugin:eqi_legacy", "capability:route_override"）

    # 能力声明
    capabilities: List[str] = field(default_factory=list)  # 该插件提供的能力（用于冲突检测）

    # 资源预算
    resources: Dict[str, float] = field(default_factory=lambda: {"cpu_ms": 10.0, "gpu_ms": 0.0, "io_mb": 0.1})

    # 速率限制
    rate_limit: Dict[str, int] = field(default_factory=dict)  # {"steps": 100} 或 {"rps": 10}

    # 沙箱与容错
    sandbox: bool = True  # 失败时是否降级为 no-op
    fail_limit: int = 5   # 连续失败几次后自动禁用

    # EQI 参数（可选）
    s_default: float = 0.0  # 默认净效用 s = L - lambda*I
    eta: float = 1.0        # 证据调制参数

    def validate(self) -> bool:
        """验证 manifest 的完整性"""
        if not self.name:
            return False
        if self.priority < 0 or self.priority > 999:
            return False
        if not self.events:
            return False
        return True

    def get_priority_class(self) -> str:
        """获取优先级所属类别"""
        if self.priority <= 49:
            return "Critical"
        elif self.priority <= 149:
            return "CoreRuntime"
        elif self.priority <= 249:
            return "Performance"
        elif self.priority <= 349:
            return "Reasoning"
        elif self.priority <= 449:
            return "Training"
        elif self.priority <= 549:
            return "Decision/EQI"
        elif self.priority <= 649:
            return "Admin/Audit"
        elif self.priority <= 799:
            return "Experimental"
        elif self.priority <= 899:
            return "Telemetry"
        else:
            return "Post/Cleanup"

    def get_rate_limit_steps(self) -> int:
        """获取速率限制（步数）"""
        return self.rate_limit.get("steps", 0)

    def get_timeout_ms(self) -> float:
        """根据优先级获取默认超时时间（毫秒）"""
        priority = self.priority
        if priority <= 49:  # Critical
            return 50.0
        elif priority <= 149:  # CoreRuntime
            return 20.0
        elif priority <= 249:  # Performance
            return 30.0
        elif priority <= 449:  # Reasoning/Training
            return 10.0
        elif priority <= 549:  # Decision/EQI
            return 200.0  # EQI 在 epoch_end 时可以更长
        elif priority <= 649:  # Admin/Audit
            return 50.0
        elif priority <= 799:  # Experimental
            return 10.0
        elif priority <= 899:  # Telemetry
            return 5.0
        else:  # Post/Cleanup
            return 100.0


# ============================================================================
# 插件基类 (Plugin Base)
# ============================================================================

class PluginBase:
    """
    插件基类

    所有插件都应继承此类，并实现相应的事件处理方法。

    示例:
        class MyPlugin(PluginBase):
            def get_manifest(self) -> PluginManifest:
                return PluginManifest(
                    name="my_plugin",
                    version="1.0.0",
                    priority=PluginPriority.TRAINING,
                    events=[PluginEvent.ON_BATCH_END]
                )

            def on_batch_end(self, context: Dict[str, Any]):
                # 处理 batch 结束事件
                loss = context.get("loss", 0.0)
                self.logger.info(f"Batch ended with loss: {loss}")
    """

    def __init__(self):
        """初始化插件"""
        self.enabled = True
        self.fail_count = 0
        self.last_step_called = -10**9
        self._context_ns = {}  # 插件私有命名空间

    def get_manifest(self) -> PluginManifest:
        """
        获取插件清单

        子类必须实现此方法，返回插件的元数据。
        """
        raise NotImplementedError("Plugin must implement get_manifest()")

    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化插件（可选）

        在插件加载时调用一次，可以进行初始化设置。
        """
        pass

    def cleanup(self):
        """
        清理资源（可选）

        在插件卸载时调用，用于释放资源。
        """
        pass

    def get_context(self, key: str, default: Any = None) -> Any:
        """获取插件私有上下文"""
        return self._context_ns.get(key, default)

    def set_context(self, key: str, value: Any):
        """设置插件私有上下文"""
        self._context_ns[key] = value

    # ========================================================================
    # 事件处理方法（子类可选实现）
    # ========================================================================

    def on_train_start(self, context: Dict[str, Any]):
        """训练开始"""
        pass

    def on_train_end(self, context: Dict[str, Any]):
        """训练结束"""
        pass

    def on_epoch_start(self, context: Dict[str, Any]):
        """Epoch 开始"""
        pass

    def on_epoch_end(self, context: Dict[str, Any]):
        """Epoch 结束"""
        pass

    def on_batch_start(self, context: Dict[str, Any]):
        """Batch 开始"""
        pass

    def on_batch_end(self, context: Dict[str, Any]):
        """Batch 结束"""
        pass

    def on_step_start(self, context: Dict[str, Any]):
        """Step 开始"""
        pass

    def on_step_end(self, context: Dict[str, Any]):
        """Step 结束"""
        pass

    def on_step_eval(self, context: Dict[str, Any]):
        """Step 评估"""
        pass

    def on_eval_start(self, context: Dict[str, Any]):
        """评估开始"""
        pass

    def on_eval_end(self, context: Dict[str, Any]):
        """评估结束"""
        pass

    def on_fail(self, context: Dict[str, Any]):
        """失败处理"""
        pass

    def on_exception(self, context: Dict[str, Any]):
        """异常处理"""
        pass

    def on_save_checkpoint(self, context: Dict[str, Any]):
        """保存检查点"""
        pass

    def on_load_checkpoint(self, context: Dict[str, Any]):
        """加载检查点"""
        pass


# ============================================================================
# 能力定义 (Capability Definitions)
# ============================================================================

class PluginCapability:
    """
    插件能力定义

    定义常见的插件能力，用于冲突检测。
    某些能力是独占的，只能有一个插件持有。
    """
    # 独占能力
    ROUTE_OVERRIDE = "route_override"  # 路由控制
    DECODE_POLICY = "decode_policy"    # 解码策略
    KILL_SWITCH = "kill_switch"        # 熔断开关

    # 共享能力（可多插件持有）
    READ_METRICS = "read_metrics"      # 读取指标
    WRITE_METRICS = "write_metrics"    # 写入指标
    ADD_CONSTRAINTS = "add_constraints"  # 添加约束
    ROUTE_SUGGEST = "route_suggest"    # 路由建议
    READ_STATE = "read_state"          # 读取状态
    WRITE_STATE = "write_state"        # 写入状态

    @classmethod
    def is_exclusive(cls, capability: str) -> bool:
        """判断能力是否独占"""
        exclusive_caps = {
            cls.ROUTE_OVERRIDE,
            cls.DECODE_POLICY,
            cls.KILL_SWITCH,
        }
        return capability in exclusive_caps


# ============================================================================
# 工具函数
# ============================================================================

def create_manifest_from_dict(data: Dict[str, Any]) -> PluginManifest:
    """从字典创建 PluginManifest"""
    return PluginManifest(**data)


def load_manifest_from_yaml(yaml_path: str) -> PluginManifest:
    """从 YAML 文件加载 PluginManifest"""
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return create_manifest_from_dict(data)
    except ImportError:
        raise ImportError("PyYAML is required for loading manifest from YAML files")
    except Exception as e:
        raise ValueError(f"Failed to load manifest from {yaml_path}: {e}")
