# -*- coding: utf-8 -*-
"""
apt_eqi_manager.py — APT EQI Manager (插件管家 / 元调度器)

特性：
- 插件注册：优先级、依赖、冲突、能力(capabilities)、预算(cpulimit/gpulimit/iolimit)、触发频率
- 静态裁决：能力独占冲突、硬冲突、依赖缺失（fail-fast 或降级跳过）
- EQI 决策：软门 φ，证据调制 E，净效用 s，预算/时延约束，稳态项 κ
- 事件派发：按优先级稳定、节流、阻塞/非阻塞执行，并做异常降级
- 审计账单：phi、E、s、启用集、被裁原因、objective、影子价近似（基于绑定资源）
- 纯 Python，无第三方依赖

注意：
- 这是“元调度器”，不改模型参数；它决定“哪些插件以多大强度参与本轮”
- 纯 Python，无三方依赖；可在 CPU 上运行
"""

# 用法（集成 Trainer）：
# from apt.apt_model.plugins.apt_eqi_manager import EQIManager, PluginSpec
#
# eqi = EQIManager(default_time_budget_ms=20)
# eqi.register(spec, handler=MyPlugin())
# ...
# eqi.compile()  # 做静态裁决
# # 每个阶段：
# audit = eqi.decide_and_dispatch(event="on_epoch_end",
#                                 step=global_step,
#                                 metrics={"valid": {...}},
#                                 evidence={"Q": {...}, "w": {...}},
#                                 resources={"cpu_budget": 5.0, "io_budget": 1.0})

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import math
import time
import threading
import traceback

# ----------------------------- dataclasses -----------------------------

@dataclass
class PluginSpec:
    name: str
    priority: int = 500
    blocking: bool = False
    events: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)       # e.g. ["plugin:admin", "core:trainer"]
    conflicts: List[str] = field(default_factory=list)      # e.g. ["plugin:eqi_legacy", "capability:route_override"]
    capabilities: List[str] = field(default_factory=list)   # e.g. ["route_override", "add_constraints"]
    # 资源预算（单位近似：ms/step、MB/step）
    resources: Dict[str, float] = field(default_factory=lambda: {"cpu_ms": 5.0, "gpu_ms": 0.0, "io_mb": 0.1})
    rate_limit_steps: int = 0        # 步间隔（<rate_limit_steps 则跳过）
    sandbox: bool = True             # 失败是否降级为 no-op
    # EQI 侧参数（可缺省）
    # s = L - lambda_cost * I （可来自评估器；缺省时可手填）
    s_default: float = 0.0
    # 证据调制：E = 1 + eta * w * (2Q-1)
    eta: float = 1.0

@dataclass
class PluginHandle:
    spec: PluginSpec
    handler: Any
    last_step_called: int = -10**9
    healthy: bool = True
    fail_count: int = 0
    fail_limit: int = 5
    disabled_reason: Optional[str] = None

# ----------------------------- helpers -----------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ----------------------------- EQI Manager -----------------------------

class EQIManager:
    """
    管家职责：
      1) register()：收集插件规格与实例
      2) compile()：静态冲突裁决（能力独占 / 硬冲突 / 依赖缺失）
      3) decide_and_dispatch()：EQI 决策启用集并按事件派发
    """
    def __init__(self,
                 default_time_budget_ms: int = 20,
                 phi_gate: Tuple[float, float, float, float] = (2.0, 2.0, 1.0, 0.7),  # (a,b,c,tau)
                 kappa_stability: float = 0.1):
        self._handles: Dict[str, PluginHandle] = {}
        self._ordered: List[PluginHandle] = []
        self.default_time_budget_ms = int(default_time_budget_ms)
        self.phi_params = phi_gate
        self.kappa = float(kappa_stability)
        # 独占能力所有权
        self._cap_owner: Dict[str, str] = {}
        # 审计
        self._last_audit: Dict[str, Any] = {}

    # -------- registration & compile --------

    def register(self, spec: PluginSpec, handler: Any):
        if spec.name in self._handles:
            raise ValueError(f"Plugin '{spec.name}' already registered")
        self._handles[spec.name] = PluginHandle(spec=spec, handler=handler)

    def compile(self, available: Optional[List[str]] = None, fail_fast: bool = False):
        """
        静态冲突裁决：
          - 依赖检查：requires 中的 plugin:xxx 若不存在，降级禁用（或 fail_fast）
          - 硬冲突：conflicts 中的 plugin:xxx 若已存在，优先级低者禁用
          - 能力独占：capability:xxx 只能被一个插件持有；按优先级授予
        """
        # 预排序（稳定顺序：优先级升序，name 次序）
        ordered = sorted(self._handles.values(), key=lambda h: (h.spec.priority, h.spec.name))
        active: List[PluginHandle] = []

        loaded_names = set(h.spec.name for h in ordered)
        for h in ordered:
            if not h.healthy:
                continue
            # 依赖检查
            dep_miss = []
            for req in h.spec.requires:
                if req.startswith("plugin:"):
                    p = req.split(":", 1)[1]
                    if p not in loaded_names:
                        dep_miss.append(req)
                # core:trainer / capability:XXX 可在此扩展实际检查
            if dep_miss:
                h.healthy = False
                h.disabled_reason = f"requires-missing:{','.join(dep_miss)}"
                if fail_fast:
                    raise RuntimeError(f"[EQI] dependency missing for {h.spec.name}: {dep_miss}")
                continue

            # 硬冲突（对已经激活的进行比较）
            conflict_hit = False
            for c in h.spec.conflicts:
                if c.startswith("plugin:"):
                    pname = c.split(":", 1)[1]
                    if pname in [x.spec.name for x in active]:
                        conflict_hit = True
                        break
            if conflict_hit:
                h.healthy = False
                h.disabled_reason = "hard-conflict"
                continue

            # 能力独占
            cap_conflict = False
            for cap in h.spec.capabilities:
                owner = self._cap_owner.get(cap)
                if owner is None:
                    self._cap_owner[cap] = h.spec.name  # 授权
                else:
                    # 已被占用 -> 禁用该插件
                    cap_conflict = True
                    break
            if cap_conflict:
                h.healthy = False
                h.disabled_reason = "capability-occupied"
                continue

            active.append(h)

        self._ordered = active

    # -------- EQI decision --------

    def _compute_phi(self, F: float, P_eq: float, EVSI: float, C_wait: float) -> float:
        a, b, c, tau = self.phi_params
        phi = sigmoid(a*F - b*P_eq + c*(EVSI - C_wait))
        return phi, tau

    def _compute_E(self, eta: float, Q: float, w: float) -> float:
        Omega = 2.0 * Q - 1.0
        return 1.0 + eta * w * Omega

    def _eqi_select(self,
                    phi: float,
                    specs: List[PluginSpec],
                    s_vals: Dict[str, float],
                    E_vals: Dict[str, float],
                    budgets: Dict[str, float]) -> Tuple[List[str], Dict[str, Any]]:
        """
        在预算下的简化选择：
          maximize   phi * sum (E_i * s_i * x_i) - kappa * |sum s_i x_i|
          subject to sum cpu_ms_i x_i <= cpu_budget, sum io_mb_i x_i <= io_budget, ...
        这里做一个贪心（按性价比排序）近似解；并返回审计项。
        """
        # 如果门未打开，直接 WAIT
        audit = {"phi": phi, "selected": [], "skipped": {}, "objective": 0.0, "net_drive": 0.0}
        if phi < self.phi_params[3]:  # tau
            audit["decision"] = "WAIT"
            return [], audit

        # 计算性价比 ρ_i = (E*s) / (weighted_cost)
        items = []
        for sp in specs:
            s = s_vals.get(sp.name, sp.s_default)
            E = E_vals.get(sp.name, 1.0)
            gain = max(0.0, E * s)  # 负收益直接忽略
            cost = 1e-9
            for k, w in [("cpu_ms", 1.0), ("gpu_ms", 5.0), ("io_mb", 0.2)]:
                cost += w * float(sp.resources.get(k, 0.0))
            rho = gain / cost
            items.append((rho, gain, sp))

        items.sort(key=lambda t: (-t[0], t[2].priority, t[2].name))

        # 贪心装入预算
        remain = dict(
            cpu_ms=float(budgets.get("cpu_budget", 1e9)),
            gpu_ms=float(budgets.get("gpu_budget", 1e9)),
            io_mb=float(budgets.get("io_budget", 1e9))
        )
        selected: List[str] = []
        sum_s = 0.0
        obj = 0.0

        for rho, gain, sp in items:
            need = sp.resources
            ok = (remain["cpu_ms"] >= need.get("cpu_ms", 0.0) and
                  remain["gpu_ms"] >= need.get("gpu_ms", 0.0) and
                  remain["io_mb"] >= need.get("io_mb", 0.0))
            if not ok:
                audit["skipped"][sp.name] = "budget"
                continue
            selected.append(sp.name)
            remain["cpu_ms"] -= need.get("cpu_ms", 0.0)
            remain["gpu_ms"] -= need.get("gpu_ms", 0.0)
            remain["io_mb"]  -= need.get("io_mb", 0.0)
            s = s_vals.get(sp.name, sp.s_default)
            sum_s += s
            obj += gain

        # 稳态项 -kappa * |sum s_i|
        obj = phi * obj - self.kappa * abs(sum_s)
        audit.update({
            "decision": "ACT",
            "selected": selected,
            "objective": obj,
            "net_drive": abs(sum_s),
            "remain_budget": remain
        })
        return selected, audit

    # -------- dispatch --------

    def decide_and_dispatch(self,
                            event: str,
                            step: int,
                            metrics: Optional[Dict[str, Any]] = None,
                            evidence: Optional[Dict[str, Dict[str, float]]] = None,
                            feasibility: Optional[Dict[str, float]] = None,
                            budgets: Optional[Dict[str, float]] = None,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        决策 + 派发：
          - 计算 φ（软门）、E、s
          - 在预算下选择启用插件集
          - 按优先级、节流与阻塞属性派发
          - 返回审计账单
        """
        metrics = metrics or {}
        evidence = evidence or {"Q": {}, "w": {}}
        budgets = budgets or {}
        context = context or {}

        # 1) 过滤订阅了该事件且健康的插件
        subs: List[PluginHandle] = [
            h for h in self._ordered
            if h.healthy and (event in h.spec.events)
        ]
        if not subs:
            self._last_audit = {"phi": 1.0, "decision": "NOOP", "selected": []}
            return self._last_audit

        # 2) 软门 φ
        F = float(feasibility.get("F", 0.8) if feasibility else 0.8)
        P_eq = float(feasibility.get("P_eq", 0.1) if feasibility else 0.1)
        EVSI = float(feasibility.get("EVSI", 0.05) if feasibility else 0.05)
        C_wait = float(feasibility.get("C_wait", 0.02) if feasibility else 0.02)
        phi, tau = self._compute_phi(F, P_eq, EVSI, C_wait)

        # 3) 每插件 E、s
        E_vals: Dict[str, float] = {}
        s_vals: Dict[str, float] = {}
        for h in subs:
            Q = float(evidence.get("Q", {}).get(h.spec.name, 0.7))
            w = float(evidence.get("w", {}).get(h.spec.name, 0.8))
            E_vals[h.spec.name] = self._compute_E(h.spec.eta, Q, w)
            # s 可来自 metrics/评估器
            s_vals[h.spec.name] = float(metrics.get("s", {}).get(h.spec.name, h.spec.s_default))

        # 4) EQI 选择
        selected_names, audit = self._eqi_select(phi, [h.spec for h in subs], s_vals, E_vals, budgets)

        # 5) 派发
        for h in subs:
            if h.spec.name not in selected_names:
                continue
            # 频率限制
            if step - h.last_step_called < h.spec.rate_limit_steps:
                audit["skipped"][h.spec.name] = "rate-limit"
                continue
            h.last_step_called = step
            self._invoke_plugin(h, event, step, context, audit)

        self._last_audit = audit
        return audit

    def _invoke_plugin(self, h: PluginHandle, event: str, step: int, ctx: Dict[str, Any], audit: Dict[str, Any]):
        fn = getattr(h.handler, event, None)
        if not callable(fn):
            audit["skipped"][h.spec.name] = "no-handler"
            return
        # 每插件上下文隔离空间
        plugin_ctx = {"event": event, "step": step, "ctx": ctx, "plugin": h.spec.name}
        try:
            if h.spec.blocking:
                self._invoke_blocking(fn, plugin_ctx, h)
            else:
                self._invoke_async(fn, plugin_ctx, h)
            # 记录执行
            audit.setdefault("executed", []).append(h.spec.name)
        except Exception as e:
            h.fail_count += 1
            if h.spec.sandbox and h.fail_count <= h.fail_limit:
                h.disabled_reason = f"exception:{e}"
                h.healthy = False
            audit["skipped"][h.spec.name] = f"exception:{e}"

    def _invoke_blocking(self, fn: Callable, plugin_ctx: Dict[str, Any], h: PluginHandle):
        # 简化：只做超时保护（线程启动 join 超时）
        result_holder = {"err": None}
        def runner():
            try:
                fn(plugin_ctx)
            except Exception as e:
                result_holder["err"] = e
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join(timeout=max(0.001, h.spec.resources.get("cpu_ms", self.default_time_budget_ms)/1000.0))
        if t.is_alive():
            # 超时：降级
            h.fail_count += 1
            h.healthy = False
            h.disabled_reason = "timeout"
            raise TimeoutError(f"plugin {h.spec.name} timed out")
        if result_holder["err"] is not None:
            raise result_holder["err"]

    def _invoke_async(self, fn: Callable, plugin_ctx: Dict[str, Any], h: PluginHandle):
        def runner():
            try:
                fn(plugin_ctx)
            except Exception:
                # 记录但不打断主流程
                h.fail_count += 1
                if h.spec.sandbox and h.fail_count <= h.fail_limit:
                    h.healthy = False
                    h.disabled_reason = "exception"
        t = threading.Thread(target=runner, daemon=True)
        t.start()

    # -------- utils --------

    @property
    def last_audit(self) -> Dict[str, Any]:
        return self._last_audit


# ----------------------------- demo plugins -----------------------------

class DemoGRPO:
    def __init__(self): self.counter = 0
    def on_batch_end(self, ctx):  # 高频低延
        self.counter += 1
        # 这里可以写：根据 ctx["ctx"]["loss"] 计算 reward 等
        # print(f"[GRPO] step={ctx['step']} counter={self.counter}")

class DemoEQIReporter:
    def __init__(self): self.epochs = 0
    def on_epoch_end(self, ctx):  # 低频可阻塞
        self.epochs += 1
        # print(f"[EQI-Reporter] epoch_end #{self.epochs}")

class DemoRoute:
    def __init__(self): self.ticks = 0
    def on_step_eval(self, ctx):
        self.ticks += 1
        # print(f"[Route] suggest temperature adjust at step={ctx['step']}")


# ----------------------------- demo main -----------------------------

if __name__ == "__main__":
    # 创建 EQI 管家
    eqi = EQIManager(default_time_budget_ms=20,
                     phi_gate=(2.0, 2.0, 1.0, 0.65),   # (a,b,c,tau)
                     kappa_stability=0.15)

    # 注册插件（示例）
    eqi.register(
        PluginSpec(
            name="grpo",
            priority=400,
            blocking=False,
            events=["on_batch_end"],
            capabilities=["read_metrics"],
            resources={"cpu_ms": 3.0, "gpu_ms": 0.0, "io_mb": 0.05},
            rate_limit_steps=0,
            s_default=0.8,  # 经验收益
            eta=1.0
        ),
        handler=DemoGRPO()
    )
    eqi.register(
        PluginSpec(
            name="eqi_report",
            priority=500,
            blocking=True,
            events=["on_epoch_end"],
            capabilities=["read_metrics"],
            resources={"cpu_ms": 15.0, "gpu_ms": 0.0, "io_mb": 0.1},
            s_default=0.6,
            eta=0.8
        ),
        handler=DemoEQIReporter()
    )
    eqi.register(
        PluginSpec(
            name="route",
            priority=300,
            blocking=False,
            events=["on_step_eval"],
            capabilities=["route_hint"],      # 假设这是独占能力
            resources={"cpu_ms": 2.0, "gpu_ms": 0.0, "io_mb": 0.02},
            s_default=0.7,
            eta=1.2
        ),
        handler=DemoRoute()
    )

    # 静态裁决（会为 capability:route_hint 授权；若重复注册，低优先级会被禁用）
    eqi.compile()

    # 模拟训练循环
    global_step = 0
    for epoch in range(2):
        # epoch start
        audit_epoch_start = eqi.decide_and_dispatch(
            event="on_epoch_start",
            step=global_step,
            metrics={"s": {}},
            evidence={"Q": {}, "w": {}},
            feasibility={"F": 0.8, "P_eq": 0.1, "EVSI": 0.05, "C_wait": 0.02},
            budgets={"cpu_budget": 10.0, "io_budget": 1.0},
            context={"epoch": epoch}
        )
        # batch loop
        for _ in range(5):
            global_step += 1
            audit_batch = eqi.decide_and_dispatch(
                event="on_batch_end",
                step=global_step,
                metrics={"s": {"grpo": 0.85}},    # 动态收益估计
                evidence={"Q": {"grpo": 0.75}, "w": {"grpo": 0.8}},
                budgets={"cpu_budget": 6.0, "io_budget": 0.5},
                context={"loss": 1.23}
            )
            # step eval（如验证或路由评估）
            audit_eval = eqi.decide_and_dispatch(
                event="on_step_eval",
                step=global_step,
                metrics={"s": {"route": 0.7}},
                evidence={"Q": {"route": 0.8}, "w": {"route": 0.9}},
                budgets={"cpu_budget": 3.0, "io_budget": 0.3},
                context={"valid_acc": 0.42}
            )

        # epoch end（低频，EQI reporter 可阻塞）
        audit_epoch_end = eqi.decide_and_dispatch(
            event="on_epoch_end",
            step=global_step,
            metrics={"s": {"eqi_report": 0.6}},
            evidence={"Q": {"eqi_report": 0.8}, "w": {"eqi_report": 0.9}},
            feasibility={"F": 0.85, "P_eq": 0.12, "EVSI": 0.06, "C_wait": 0.03},
            budgets={"cpu_budget": 20.0, "io_budget": 2.0},
            context={"eval_metrics": {"acc": 0.71}}
        )

    # 打印一次最终审计
    print("[EQI AUDIT] last:", eqi.last_audit)