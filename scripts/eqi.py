# -*- coding: utf-8 -*-
"""
EQI 插件（Evidence Qualitative Inference）
- 软门 φ：是否执行（WAIT/ACT）
- 证据调制 E：把 Q,w,Ω 写进线性系数
- 约束多面体 X 上的线性目标（可线性化抑振项 |Σ s_k x_k|）
- 若检测到 PuLP/HiGHS 则走 LP；否则用近似启发式（可跑通 & 可审计）

用法（Python）：
from apt.apt_model.plugins.eqi import EQIConfig, eqi_decide

cfg = EQIConfig()
result = eqi_decide(L,I,Q,w,A,c, cfg=cfg, B=None,d=None, lower=None, upper=None)
print(result["decision"], result["x_star"], result["audit"])
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from math import exp, isfinite

# ------------------- 配置 -------------------
@dataclass
class EQIConfig:
    lambda_cost: float = 1.0   # 成本权衡 λ
    eta: float = 1.0           # 证据放大 η
    # 门控 φ = σ(a F - b P_eq + c (EVSI - C_wait))
    gate_a: float = 2.0
    gate_b: float = 2.0
    gate_c: float = 1.0
    gate_tau: float = 0.7      # 执行阈值 τ
    kappa: float = 0.0         # 抑振权重 κ（|Σ s_k x_k|）
    use_lp_if_available: bool = True   # 有 LP 求解器则用 LP
    greedy_tolerance: float = 1e-6     # 启发式容差

# ------------------- 工具 -------------------
def _sigmoid(x: float) -> float:
    # 数值稳定
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = exp(x)
        return z / (1.0 + z)

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _safe_list(vals: List[float], fill: float, n: int) -> List[float]:
    if vals is None: return [fill]*n
    assert len(vals) == n
    return list(vals)

# ------------------- 主流程 -------------------
def eqi_decide(
    L: List[float],
    I: List[float],
    Q: List[float],
    w: List[float],
    A: List[List[float]],
    c: List[float],
    cfg: EQIConfig,
    F: float = 0.8,
    P_eq: float = 0.2,
    EVSI: float = 0.1,
    C_wait: float = 0.05,
    B: Optional[List[List[float]]] = None,
    d: Optional[List[float]] = None,
    lower: Optional[List[float]] = None,
    upper: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    返回：
      {
        "decision": "WAIT"|"ACT",
        "x_star": [K],
        "audit": {
           "phi": float, "E": [K], "s": [K],
           "net_drive": float, "objective": float,
           "solver": "lp|greedy",
           "notes": str
        }
      }
    """
    K = len(L)
    assert len(I)==K and len(Q)==K and len(w)==K
    assert len(A)==len(c)
    lower = _safe_list(lower, 0.0, K)
    upper = _safe_list(upper, float("inf"), K)

    # 1) 净效用/证据调制
    s = [float(L[k]) - cfg.lambda_cost*float(I[k]) for k in range(K)]
    Omega = [2.0*float(Q[k]) - 1.0 for k in range(K)]
    E = [1.0 + cfg.eta*float(w[k])*Omega[k] for k in range(K)]

    # 2) 软门 φ
    phi = _sigmoid(cfg.gate_a*float(F) - cfg.gate_b*float(P_eq) + cfg.gate_c*(float(EVSI)-float(C_wait)))
    if phi < cfg.gate_tau:
        return {
            "decision": "WAIT",
            "x_star": [0.0]*K,
            "audit": {"phi": float(phi), "E": E, "s": s, "net_drive": 0.0, "objective": 0.0, "solver": "none", "notes": "under gate_tau"},
        }

    # 3) 求解：优先 LP（PuLP/HiGHS），否则贪心启发式
    if cfg.use_lp_if_available:
        try:
            import pulp as pl
            return _eqi_lp(L, s, E, A, c, cfg, B=B, d=d, lower=lower, upper=upper, phi=phi)
        except Exception as e:
            # 回退
            notes = f"LP unavailable -> fallback greedy ({e})"
    else:
        notes = "LP disabled -> greedy"

    return _eqi_greedy(L, s, E, A, c, cfg, B=B, d=d, lower=lower, upper=upper, phi=phi, notes=notes)

# ------------------- LP 解法（PuLP） -------------------
def _eqi_lp(
    L, s, E, A, c, cfg: EQIConfig, B=None, d=None, lower=None, upper=None, phi: float = 1.0
) -> Dict[str, Any]:
    import pulp as pl
    K = len(s)
    prob = pl.LpProblem("EQI", pl.LpMaximize)
    x = [pl.LpVariable(f"x_{k}", lowBound=lower[k], upBound=None if not isfinite(upper[k]) else upper[k]) for k in range(K)]
    z = pl.LpVariable("z", lowBound=0)

    # 目标：phi * Σ (E*s) x - kappa * z
    prob += phi * pl.lpSum((E[k]*s[k])*x[k] for k in range(K)) - cfg.kappa * z

    # Ax <= c
    for i, row in enumerate(A):
        prob += pl.lpSum(row[k]*x[k] for k in range(K)) <= c[i], f"cap_{i}"

    # Bx = d（可选）
    if B is not None and d is not None:
        for i, row in enumerate(B):
            prob += pl.lpSum(row[k]*x[k] for k in range(K)) == d[i], f"bal_{i}"

    # 绝对值线性化：-|Σ s x| <= z
    prob += pl.lpSum(s[k]*x[k] for k in range(K)) <= z, "abs_pos"
    prob += -pl.lpSum(s[k]*x[k] for k in range(K)) <= z, "abs_neg"

    # 上界（若提供）
    if upper is not None and any(isfinite(u) for u in upper):
        for k in range(K):
            if isfinite(upper[k]):
                prob += x[k] <= upper[k], f"ub_{k}"

    # 求解
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    x_star = [float(v.value()) for v in x]
    net_drive = abs(sum(s[k]*x_star[k] for k in range(K)))
    objective = float(pl.value(prob.objective))
    return {
        "decision": "ACT",
        "x_star": x_star,
        "audit": {
            "phi": float(phi), "E": list(E), "s": list(s),
            "net_drive": net_drive, "objective": objective,
            "solver": "lp", "notes": "ok"
        }
    }

# ------------------- 贪心启发式（无依赖） -------------------
def _eqi_greedy(
    L, s, E, A, c, cfg: EQIConfig, B=None, d=None, lower=None, upper=None, phi: float = 1.0, notes: str = ""
) -> Dict[str, Any]:
    """
    近似策略：
      1) 先满足下界 lower；
      2) 按 (E*s) 从大到小增加 x[k]，每次以不违反 Ax<=c/上界 的最大步长推进；
      3) 若提供 Bx=d，最后在容差内做比例微调。
    """
    K = len(s)
    x = [max(0.0, float(lower[k])) for k in range(K)]
    import math

    # 剩余额度
    def slack_vec(xv):
        # 计算 Ax<=c 的剩余额度
        sx = []
        for i, row in enumerate(A):
            used = sum(row[k]*xv[k] for k in range(K))
            sx.append(c[i] - used)
        return sx

    # 初始校验：若下界已超限，则尽力回缩（简单裁剪）
    sv = slack_vec(x)
    for i, sl in enumerate(sv):
        if sl < -cfg.greedy_tolerance:
            # 简化：按贡献度回缩（这里直接按平均回缩，工业实践建议二次启发）
            scale = max(0.0, (c[i] - 1e-9) / (sum(abs(A[i][k])*max(1e-9,x[k]) for k in range(K)) + 1e-9))
            x = [xi*scale for xi in x]
            break

    # 目标系数
    coeff = [phi * (E[k]*s[k]) for k in range(K)]
    order = sorted(range(K), key=lambda k: coeff[k], reverse=True)

    # 逐通道推进
    for k in order:
        if coeff[k] <= 0:  # 非正增益不推进
            continue
        # 最大步长：受每条约束与上界限制
        max_step = float("inf")
        for i, row in enumerate(A):
            a_ik = row[k]
            if a_ik > 0:
                rem = c[i] - sum(row[j]*x[j] for j in range(K))
                max_step = min(max_step, rem / max(a_ik, 1e-12))
        if isfinite(upper[k]):
            max_step = min(max_step, upper[k] - x[k])
        if max_step > 0:
            x[k] += max_step

    # 可选：处理 Bx=d（简单比例微调）
    if B is not None and d is not None and len(B)>0:
        # 单条等式轻微修正（多条时建议用 LP）
        row = B[0]; target = d[0]
        cur = sum(row[k]*x[k] for k in range(K))
        diff = target - cur
        # 均分到正系数的变量（示意）
        pos_idx = [k for k in range(K) if row[k] > 0 and (not isfinite(upper[k]) or x[k] < upper[k]-1e-9)]
        if pos_idx:
            share = diff / (sum(row[k] for k in pos_idx) + 1e-9)
            for k in pos_idx:
                x[k] = min(upper[k], max(lower[k], x[k] + share))

    net_drive = abs(sum(s[k]*x[k] for k in range(K)))
    objective = sum(coeff[k]*x[k] for k in range(K)) - cfg.kappa * net_drive
    return {
        "decision": "ACT",
        "x_star": x,
        "audit": {
            "phi": float(phi), "E": list(E), "s": list(s),
            "net_drive": float(net_drive), "objective": float(objective),
            "solver": "greedy", "notes": notes or "fallback"
        }
    }

# ------------------- CLI 入口（可选） -------------------
def cli_entry(args: Dict[str, Any]) -> None:
    """
    简单 CLI：从 JSON/字典参数调用
    期望 args 包含：
      L,I,Q,w,A,c 以及可选 B,d,lower,upper 与门控/系数
    """
    import json, sys
    cfg = EQIConfig(
        lambda_cost=float(args.get("lambda_cost", 1.0)),
        eta=float(args.get("eta", 1.0)),
        gate_a=float(args.get("gate_a", 2.0)),
        gate_b=float(args.get("gate_b", 2.0)),
        gate_c=float(args.get("gate_c", 1.0)),
        gate_tau=float(args.get("gate_tau", 0.7)),
        kappa=float(args.get("kappa", 0.0)),
        use_lp_if_available=bool(args.get("use_lp_if_available", True)),
    )
    res = eqi_decide(
        L=args["L"], I=args["I"], Q=args["Q"], w=args["w"],
        A=args["A"], c=args["c"], cfg=cfg,
        F=float(args.get("F", 0.8)),
        P_eq=float(args.get("P_eq", 0.2)),
        EVSI=float(args.get("EVSI", 0.1)),
        C_wait=float(args.get("C_wait", 0.05)),
        B=args.get("B"), d=args.get("d"),
        lower=args.get("lower"), upper=args.get("upper"),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))