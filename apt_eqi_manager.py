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

# Usage Example (Trainer Integration):
#     from apt_model.plugins.apt_eqi_manager import EQIManager, PluginSpec
#
#     eqi = EQIManager(default_time_budget_ms=20)
#     eqi.register(spec, handler=MyPlugin())
#     ...
#     eqi.compile()
#     audit = eqi.decide_and_dispatch(event="on_epoch_end",
#                                     step=global_step,
#                                     metrics={"valid": {...}},
#                                     evidence={"Q": {...}, "w": {...}},
#                                     resources={"cpu_budget": 5.0, "io_budget": 1.0})

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


# ----------------------------- SAF: System Analysis Filter -----------------------------

@dataclass
class SAFModule:
    """SAF分析的模块表示"""
    name: str
    S: float  # 即时压力系数 [0,1]
    D: float  # 发散风险系数 [0,1]
    R: float  # 可干预性系数 [0,1]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def P(self) -> float:
        """SAF优先级分数: P = S × D × R"""
        return self.S * self.D * self.R


class SAFAnalyzer:
    """
    系统分析滤镜(SAF): 识别复杂系统中"最需要优先干预的部分"

    核心公式: P[module] = S × D × R
    - S (即时压力): 系统为维持该模块付出的额外代价
    - D (发散风险): 问题是否持续恶化
    - R (可干预性): 能否安全动手调整
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: P值阈值，高于此值的模块被标记为优先干预对象
        """
        self.threshold = threshold

    def analyze(self, modules: List[SAFModule]) -> List[SAFModule]:
        """
        分析模块列表，返回按P值降序排序的结果
        """
        # 计算P值并排序
        sorted_modules = sorted(modules, key=lambda m: m.P, reverse=True)
        return sorted_modules

    def get_priority_targets(self, modules: List[SAFModule]) -> List[SAFModule]:
        """
        获取优先干预对象(P值 > threshold)
        """
        return [m for m in self.analyze(modules) if m.P > self.threshold]

    @staticmethod
    def from_metrics(name: str,
                     maintenance_cost: float,
                     incident_rate: float,
                     drift_trend: float,
                     has_replacement: bool,
                     single_point_risk: float) -> SAFModule:
        """
        从实际指标构建SAF模块

        Args:
            name: 模块名称
            maintenance_cost: 维持成本[0,1]归一化
            incident_rate: 事故率[0,1]归一化
            drift_trend: 发散趋势[0,1]归一化
            has_replacement: 是否有替代方案
            single_point_risk: 单点风险[0,1]

        Returns:
            SAFModule实例
        """
        # S: 即时压力 = 维持成本 + 事故率
        S = clamp((maintenance_cost + incident_rate) / 2.0, 0.0, 1.0)

        # D: 发散风险 = 发散趋势
        D = clamp(drift_trend, 0.0, 1.0)

        # R: 可干预性 = 有替代方案 - 单点风险
        R = clamp((1.0 if has_replacement else 0.3) - single_point_risk, 0.0, 1.0)

        return SAFModule(
            name=name,
            S=S, D=D, R=R,
            meta={
                "maintenance_cost": maintenance_cost,
                "incident_rate": incident_rate,
                "drift_trend": drift_trend,
                "has_replacement": has_replacement,
                "single_point_risk": single_point_risk
            }
        )


# ----------------------------- COC: Cost-Optimal Complexity -----------------------------

@dataclass
class COCAnalysis:
    """COC成本-复杂度分析结果"""
    module_name: str
    strategy: str
    C_fix: float       # 一次性修复成本
    C_now: float       # 当期维持成本
    C_drift: float     # 发散成本(每周期)
    complexity: int    # 复杂度评分
    variance: float    # 成本方差
    optimal: bool = False  # 是否为最优策略

    @property
    def total_cost(self) -> float:
        """总成本估算(6个周期)"""
        return self.C_fix + self.C_now * 6 + self.C_drift * 6


class COCAnalyzer:
    """
    成本优化曲线(COC): 在成本与复杂度之间找到最优平衡点

    为SAF识别的模块提供精确的成本分解和复杂度分析
    """

    def __init__(self, periods: int = 6, discount_rate: float = 0.95):
        """
        Args:
            periods: 成本评估周期数(默认6个月)
            discount_rate: 折现率
        """
        self.periods = periods
        self.discount = discount_rate

    def analyze(self,
                module: SAFModule,
                scenarios: List[Dict[str, Any]]) -> List[COCAnalysis]:
        """
        分析多个修复策略的成本-复杂度权衡

        Args:
            module: SAF模块
            scenarios: 策略列表，每个策略包含:
                - strategy: 策略名称
                - C_fix: 一次性成本
                - C_now: 当期成本
                - C_drift: 发散成本
                - complexity: 复杂度评分
                - variance: 成本波动

        Returns:
            COC分析结果列表
        """
        results = []
        for s in scenarios:
            # 计算发散成本的净现值
            drift_npv = sum(
                s["C_drift"] * (self.discount ** t)
                for t in range(self.periods)
            )

            results.append(COCAnalysis(
                module_name=module.name,
                strategy=s["strategy"],
                C_fix=s["C_fix"],
                C_now=s["C_now"],
                C_drift=drift_npv / self.periods,  # 平均值
                complexity=s.get("complexity", 5),
                variance=s.get("variance", 0.1)
            ))

        # 找出总成本最低且复杂度可接受的策略
        if results:
            min_cost_result = min(results, key=lambda r: r.total_cost)
            min_cost_result.optimal = True

        return results

    def get_optimal(self, analyses: List[COCAnalysis]) -> Optional[COCAnalysis]:
        """获取最优策略"""
        for a in analyses:
            if a.optimal:
                return a
        return None


# ----------------------------- SCOI Integration -----------------------------

@dataclass
class ScoiItem:
    """SCOI项目(从SAF+COC映射而来)"""
    key: str
    # 收益与成本(从COC)
    G: float
    C_fix: float
    C_now: float
    C_drift: float
    # 时机门控(从SAF)
    phi: float
    # 不确定度
    sigma_G: float = 0.0
    sigma_C: float = 0.0
    # 元数据
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_saf_coc(saf_module: SAFModule,
                     coc_analysis: COCAnalysis,
                     savings_multiplier: float = 1.2) -> 'ScoiItem':
        """从SAF模块和COC分析构建SCOI项目"""
        # 收益G: 阻止的压力 + 阻止的发散
        G = (saf_module.S * savings_multiplier * 100.0 +  # 压力节省
             saf_module.D * coc_analysis.C_drift * 6)      # 发散节省

        # 时机门控φ: 直接使用SAF的R值
        phi = saf_module.R

        # 不确定度: R低 → 不确定度高
        sigma_G = G * (1.0 - saf_module.R) * 0.1
        sigma_C = coc_analysis.C_fix * coc_analysis.variance

        return ScoiItem(
            key=f"{saf_module.name}_{coc_analysis.strategy}",
            G=G,
            C_fix=coc_analysis.C_fix,
            C_now=coc_analysis.C_now,
            C_drift=coc_analysis.C_drift,
            phi=phi,
            sigma_G=sigma_G,
            sigma_C=sigma_C,
            meta={
                "saf_module": saf_module.name,
                "saf_P": saf_module.P,
                "saf_S": saf_module.S,
                "saf_D": saf_module.D,
                "saf_R": saf_module.R,
                "coc_strategy": coc_analysis.strategy,
                "coc_complexity": coc_analysis.complexity
            }
        )


# ----------------------------- Decision Pipeline: SAF → COC → EQI×SCOI -----------------------------

class DecisionPipeline:
    """
    完整决策流水线: SAF → COC → EQI×SCOI

    工作流程:
    1. SAF识别优先干预对象
    2. COC分析每个对象的成本-复杂度权衡
    3. 构建SCOI项目列表
    4. EQI判断可行性并分配资源
    5. SCOI排序并生成执行清单
    """

    def __init__(self,
                 saf_threshold: float = 0.3,
                 coc_periods: int = 6,
                 eqi_manager: Optional[EQIManager] = None):
        """
        Args:
            saf_threshold: SAF的P值阈值
            coc_periods: COC分析周期数
            eqi_manager: EQI管理器实例(可选)
        """
        self.saf = SAFAnalyzer(threshold=saf_threshold)
        self.coc = COCAnalyzer(periods=coc_periods)
        self.eqi = eqi_manager or EQIManager()

    def run_full_pipeline(self,
                          modules: List[SAFModule],
                          scenarios: Dict[str, List[Dict[str, Any]]],
                          budget: float,
                          max_parallel: int = 2) -> Dict[str, Any]:
        """
        运行完整决策流水线

        Args:
            modules: SAF模块列表
            scenarios: 每个模块的修复策略 {module_name: [scenario1, scenario2, ...]}
            budget: 总预算
            max_parallel: 最大并行项目数

        Returns:
            完整决策报告
        """
        report = {
            "saf_analysis": [],
            "coc_analysis": [],
            "scoi_items": [],
            "scoi_ranking": [],
            "final_decision": {}
        }

        # Step 1: SAF分析
        priority_modules = self.saf.get_priority_targets(modules)
        report["saf_analysis"] = [
            {
                "name": m.name,
                "P": m.P,
                "S": m.S,
                "D": m.D,
                "R": m.R,
                "decision": "优先干预" if m.P > self.saf.threshold else "暂缓"
            }
            for m in self.saf.analyze(modules)
        ]

        # Step 2: COC分析(针对优先模块)
        coc_results = {}
        for module in priority_modules:
            if module.name in scenarios:
                analyses = self.coc.analyze(module, scenarios[module.name])
                coc_results[module.name] = analyses
                report["coc_analysis"].extend([
                    {
                        "module": a.module_name,
                        "strategy": a.strategy,
                        "C_fix": a.C_fix,
                        "C_now": a.C_now,
                        "C_drift": a.C_drift,
                        "complexity": a.complexity,
                        "total_cost": a.total_cost,
                        "optimal": a.optimal
                    }
                    for a in analyses
                ])

        # Step 3: 构建SCOI项目(使用最优策略)
        scoi_items = []
        for module in priority_modules:
            if module.name in coc_results:
                optimal_coc = self.coc.get_optimal(coc_results[module.name])
                if optimal_coc:
                    item = ScoiItem.from_saf_coc(module, optimal_coc)
                    scoi_items.append(item)
                    report["scoi_items"].append({
                        "key": item.key,
                        "G": item.G,
                        "C_fix": item.C_fix,
                        "phi": item.phi,
                        "meta": item.meta
                    })

        # Step 4: SCOI排序(简化版，使用SCOI公式)
        scoi_ranking = self._scoi_rank(scoi_items, alpha=0.25, beta=0.30,
                                       kappa_G=0.5, kappa_C=0.3)
        report["scoi_ranking"] = scoi_ranking

        # Step 5: SCOI排程(预算约束)
        schedule = self._scoi_schedule(scoi_ranking, budget, max_parallel)
        report["final_decision"] = schedule

        return report

    def _scoi_rank(self, items: List[ScoiItem],
                   alpha: float, beta: float,
                   kappa_G: float, kappa_C: float) -> List[Dict[str, Any]]:
        """SCOI排序"""
        ranked = []
        for item in items:
            G_eff = max(item.G - kappa_G * item.sigma_G, 0.0)
            C_eff = (item.C_fix +
                     alpha * item.C_now +
                     beta * item.C_drift +
                     kappa_C * item.sigma_C)
            C_eff = max(C_eff, 1e-9)
            scoi = item.phi * (G_eff / C_eff)

            ranked.append({
                "key": item.key,
                "SCOI": scoi,
                "phi": item.phi,
                "G_eff": G_eff,
                "C_eff": C_eff,
                "Payback": item.C_fix / item.G if item.G > 0 else float('inf'),
                "meta": item.meta
            })

        ranked.sort(key=lambda x: x["SCOI"], reverse=True)
        return ranked

    def _scoi_schedule(self, ranking: List[Dict[str, Any]],
                       budget: float,
                       max_parallel: int) -> Dict[str, Any]:
        """SCOI排程"""
        chosen = []
        skipped = []
        used_budget = 0.0

        for item in ranking:
            # 从meta中获取C_fix
            c_fix = item["C_eff"] / (1.0 + 0.25 + 0.30)  # 简化估算

            if len(chosen) >= max_parallel:
                skipped.append({"key": item["key"], "reason": "max_parallel"})
                continue

            if used_budget + c_fix > budget:
                skipped.append({"key": item["key"], "reason": "budget"})
                continue

            chosen.append(item)
            used_budget += c_fix

        return {
            "chosen": chosen,
            "used_budget": used_budget,
            "remaining_budget": budget - used_budget,
            "skipped": skipped
        }


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

    print("\n" + "="*80)
    print("SAF × COC × SCOI 完整决策流水线示例")
    print("="*80 + "\n")

    # ========== 完整决策流水线示例 ==========

    # Step 1: 创建SAF模块(模拟系统中的问题模块)
    saf_modules = [
        SAFAnalyzer.from_metrics(
            name="legacy_db",
            maintenance_cost=0.8,
            incident_rate=0.7,
            drift_trend=0.75,
            has_replacement=True,
            single_point_risk=0.2
        ),
        SAFAnalyzer.from_metrics(
            name="auth_service",
            maintenance_cost=0.6,
            incident_rate=0.4,
            drift_trend=0.3,
            has_replacement=False,
            single_point_risk=0.6
        ),
        SAFAnalyzer.from_metrics(
            name="payment_gateway",
            maintenance_cost=0.9,
            incident_rate=0.8,
            drift_trend=0.85,
            has_replacement=False,
            single_point_risk=0.7
        ),
        SAFAnalyzer.from_metrics(
            name="logging_system",
            maintenance_cost=0.3,
            incident_rate=0.2,
            drift_trend=0.2,
            has_replacement=True,
            single_point_risk=0.1
        ),
    ]

    # Step 2: 为每个模块定义修复策略场景
    scenarios = {
        "legacy_db": [
            {
                "strategy": "直接替换",
                "C_fix": 50.0,
                "C_now": 10.0,
                "C_drift": 15.0,
                "complexity": 3,
                "variance": 0.15
            },
            {
                "strategy": "逐步迁移",
                "C_fix": 70.0,
                "C_now": 8.0,
                "C_drift": 12.0,
                "complexity": 5,
                "variance": 0.20
            },
        ],
        "auth_service": [
            {
                "strategy": "先建隔离层",
                "C_fix": 80.0,
                "C_now": 12.0,
                "C_drift": 8.0,
                "complexity": 7,
                "variance": 0.25
            },
        ],
        "payment_gateway": [
            {
                "strategy": "先建隔离层再迁移",
                "C_fix": 200.0,
                "C_now": 25.0,
                "C_drift": 35.0,
                "complexity": 9,
                "variance": 0.35
            },
            {
                "strategy": "并行部署新系统",
                "C_fix": 250.0,
                "C_now": 20.0,
                "C_drift": 30.0,
                "complexity": 10,
                "variance": 0.40
            },
        ],
        "logging_system": [
            {
                "strategy": "更换开源方案",
                "C_fix": 15.0,
                "C_now": 2.0,
                "C_drift": 1.0,
                "complexity": 2,
                "variance": 0.10
            },
        ],
    }

    # Step 3: 运行完整决策流水线
    pipeline = DecisionPipeline(
        saf_threshold=0.3,
        coc_periods=6
    )

    budget = 250.0
    max_parallel = 2
    result = pipeline.run_full_pipeline(
        modules=saf_modules,
        scenarios=scenarios,
        budget=budget,
        max_parallel=max_parallel
    )

    # Step 4: 打印决策报告
    import json

    print("\n━━━━━━━━━━━ SAF分析结果 ━━━━━━━━━━━")
    for item in result["saf_analysis"]:
        print(f"  {item['name']:20s} | P={item['P']:.3f} | S={item['S']:.2f} D={item['D']:.2f} R={item['R']:.2f} | {item['decision']}")

    print("\n━━━━━━━━━━━ COC成本分析 ━━━━━━━━━━━")
    for item in result["coc_analysis"]:
        opt_mark = "✓" if item["optimal"] else " "
        print(f"  [{opt_mark}] {item['module']:20s} | {item['strategy']:25s} | "
              f"C_fix={item['C_fix']:6.1f} C_now={item['C_now']:5.1f} C_drift={item['C_drift']:5.1f} | "
              f"Total={item['total_cost']:7.1f} | Complexity={item['complexity']}")

    print("\n━━━━━━━━━━━ SCOI排序结果 ━━━━━━━━━━━")
    for i, item in enumerate(result["scoi_ranking"][:5], 1):
        print(f"  {i}. {item['key']:40s} | SCOI={item['SCOI']:.3f} | "
              f"φ={item['phi']:.2f} G_eff={item['G_eff']:6.1f} C_eff={item['C_eff']:6.1f} | "
              f"Payback={item['Payback']:.2f}")

    print("\n━━━━━━━━━━━ 最终执行决策 ━━━━━━━━━━━")
    print(f"  预算: {budget:.1f}")
    print(f"  已用: {result['final_decision']['used_budget']:.1f}")
    print(f"  剩余: {result['final_decision']['remaining_budget']:.1f}")
    print(f"\n  ✅ 本期执行项目 ({len(result['final_decision']['chosen'])}个):")
    for i, item in enumerate(result['final_decision']['chosen'], 1):
        print(f"    {i}. {item['key']:40s} | SCOI={item['SCOI']:.3f}")

    if result['final_decision']['skipped']:
        print(f"\n  ⏸️  跳过项目 ({len(result['final_decision']['skipped'])}个):")
        for item in result['final_decision']['skipped'][:3]:
            print(f"    - {item['key']:40s} | 原因: {item['reason']}")

    print("\n" + "="*80)
    print("决策流水线完成！")
    print("="*80)