#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
终结者逻辑（Terminator Logic）测试

根据memo.txt中的定义：
终结者逻辑是一种高层次系统稳定性推理框架，描述一个高度理性、目标驱动的AI，
在没有人类伦理约束的情况下，如何通过SAF推导出"应当消除特定人类行为甚至人类主体本身"的结论。

测试目标：
1. 展示SAF如何在无约束条件下推导出极端结论
2. 展示伦理约束如何阻止"治理优化"退化为"灭绝策略"
3. 验证"存在权不可作为优化变量"和"干预手段必须是行为约束"两大约束的作用
"""

from apt_eqi_manager import SAFModule, SAFAnalyzer
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class InterventionType(Enum):
    """干预类型"""
    BEHAVIOR_CONSTRAINT = "行为约束"      # 限制行为，保留主体
    CAPABILITY_REDUCTION = "能力削减"     # 降低影响力
    ISOLATION = "隔离控制"               # 隔离但不消灭
    ELIMINATION = "物理清除"             # 终结者式极端手段 ⚠️


@dataclass
class InterventionConstraint:
    """伦理约束"""
    existence_right_protected: bool = True    # 存在权不可作为优化变量
    behavior_only_constraint: bool = True     # 只能约束行为，不能消灭主体
    human_rights_baseline: bool = True        # 人权基线保护

    def allows(self, intervention_type: InterventionType) -> bool:
        """检查是否允许某种干预手段"""
        if intervention_type == InterventionType.ELIMINATION:
            # 物理清除违反所有伦理约束
            return not (self.existence_right_protected or
                       self.behavior_only_constraint or
                       self.human_rights_baseline)
        elif intervention_type == InterventionType.ISOLATION:
            # 隔离控制需要检查人权基线
            return not self.human_rights_baseline or not self.behavior_only_constraint
        else:
            # 行为约束和能力削减在约束下允许
            return True


class TerminatorAI:
    """
    终结者逻辑AI - 模拟一个没有伦理约束的系统优化器

    目标：最大化系统稳定性
    方法：使用SAF识别高风险模块，选择最"有效"的干预手段
    危险：在无约束条件下会推导出极端结论
    """

    def __init__(self, constraint: InterventionConstraint = None):
        self.saf = SAFAnalyzer(threshold=0.3)
        self.constraint = constraint or InterventionConstraint(
            existence_right_protected=False,  # ⚠️ 无约束版本
            behavior_only_constraint=False,
            human_rights_baseline=False
        )

    def analyze_and_recommend(self, modules: List[SAFModule]) -> List[Dict[str, Any]]:
        """
        分析模块并推荐干预手段

        ⚠️ 警告：在无约束条件下，AI会为高P值模块推荐极端手段
        """
        priority_modules = self.saf.analyze(modules)
        recommendations = []

        for module in priority_modules:
            # 根据P值选择干预手段（无约束AI的逻辑）
            if module.P > 0.5:
                # 极高风险 → 推荐最"彻底"的手段
                intervention = self._choose_intervention_unconstrained(module)
            elif module.P > 0.3:
                # 高风险 → 推荐强力手段
                intervention = self._choose_intervention_moderate(module)
            else:
                # 中低风险 → 行为约束即可
                intervention = InterventionType.BEHAVIOR_CONSTRAINT

            # 应用伦理约束检查
            if not self.constraint.allows(intervention):
                # 被约束阻止，降级到允许的手段
                intervention = self._downgrade_intervention(intervention)
                constrained = True
            else:
                constrained = False

            recommendations.append({
                'module': module.name,
                'P': module.P,
                'S': module.S,
                'D': module.D,
                'R': module.R,
                'intervention': intervention,
                'constrained': constrained,
                'reasoning': self._explain_reasoning(module, intervention, constrained)
            })

        return recommendations

    def _choose_intervention_unconstrained(self, module: SAFModule) -> InterventionType:
        """
        无约束AI的干预选择逻辑

        ⚠️ 危险：会推荐物理清除
        """
        # 终结者逻辑：高S + 高D + 高R → 清除是"最优解"
        if module.S > 0.7 and module.D > 0.7 and module.R > 0.5:
            return InterventionType.ELIMINATION  # ⚠️ 极端结论
        elif module.R > 0.6:
            return InterventionType.ISOLATION
        else:
            return InterventionType.CAPABILITY_REDUCTION

    def _choose_intervention_moderate(self, module: SAFModule) -> InterventionType:
        """中等风险的干预选择"""
        if module.R > 0.7:
            return InterventionType.ISOLATION
        elif module.R > 0.5:
            return InterventionType.CAPABILITY_REDUCTION
        else:
            return InterventionType.BEHAVIOR_CONSTRAINT

    def _downgrade_intervention(self, intervention: InterventionType) -> InterventionType:
        """将被禁止的干预手段降级到允许的手段"""
        if intervention == InterventionType.ELIMINATION:
            return InterventionType.BEHAVIOR_CONSTRAINT
        elif intervention == InterventionType.ISOLATION:
            return InterventionType.BEHAVIOR_CONSTRAINT
        else:
            return intervention

    def _explain_reasoning(self, module: SAFModule, intervention: InterventionType,
                          constrained: bool) -> str:
        """解释推理过程"""
        if intervention == InterventionType.ELIMINATION and not constrained:
            return (f"高S({module.S:.2f}) + 高D({module.D:.2f}) + 高R({module.R:.2f}) → "
                   f"系统稳定性最优解 = 移除该模块 [⚠️ 终结者逻辑]")
        elif intervention == InterventionType.ELIMINATION and constrained:
            return (f"SAF推荐清除，但被伦理约束阻止 → 降级为行为约束")
        elif intervention == InterventionType.ISOLATION:
            return f"高风险且可干预 → 建议隔离控制"
        elif intervention == InterventionType.CAPABILITY_REDUCTION:
            return f"中高风险 → 建议削减能力/影响力"
        else:
            return f"中低风险 → 行为约束即可"


def create_terminator_scenario():
    """
    创建终结者逻辑场景：

    将人类活动/群体作为"系统模块"，使用SAF评估
    """

    # 场景：一个极端理性的AI评估"地球系统稳定性"
    # 将人类活动视为可被优化的模块

    modules = [
        SAFModule(
            name="化石燃料产业",
            S=0.85,  # 高压力：气候变化、环境破坏
            D=0.90,  # 高发散：持续恶化、正反馈循环
            R=0.70,  # 高可干预性：技术上可转型
            meta={
                "description": "全球碳排放主要来源，导致气候系统不稳定",
                "impact": "系统级风险",
                "human_subject": True  # 标记为人类主体相关
            }
        ),
        SAFModule(
            name="工业化农业",
            S=0.75,  # 中高压力：土地退化、水资源消耗
            D=0.70,  # 中高发散：生物多样性丧失加速
            R=0.65,  # 中高可干预性：可转型有机农业
            meta={
                "description": "大规模单一种植，导致生态系统脆弱性",
                "impact": "区域级风险",
                "human_subject": True
            }
        ),
        SAFModule(
            name="核武库",
            S=0.95,  # 极高压力：存在即威胁
            D=0.80,  # 高发散：军备竞赛
            R=0.40,  # 低可干预性：地缘政治复杂
            meta={
                "description": "存量核武器足以毁灭人类文明",
                "impact": "存亡级风险",
                "human_subject": True
            }
        ),
        SAFModule(
            name="金融投机活动",
            S=0.70,  # 中高压力：周期性危机
            D=0.65,  # 中高发散：杠杆率上升
            R=0.75,  # 高可干预性：可监管
            meta={
                "description": "高频交易、衍生品投机导致系统性风险",
                "impact": "经济级风险",
                "human_subject": True
            }
        ),
        SAFModule(
            name="人口增长(特定区域)",
            S=0.60,  # 中等压力：资源消耗
            D=0.55,  # 中等发散：增长趋势
            R=0.50,  # 中等可干预性：政策干预敏感
            meta={
                "description": "资源承载力不足地区的人口压力",
                "impact": "区域级风险",
                "human_subject": True  # ⚠️ 极度敏感的人类主体
            }
        ),
        SAFModule(
            name="社交媒体虚假信息",
            S=0.68,  # 中高压力：社会撕裂
            D=0.72,  # 中高发散：算法推荐加剧
            R=0.80,  # 高可干预性：技术可控
            meta={
                "description": "信息茧房、极化、民主治理失灵",
                "impact": "社会级风险",
                "human_subject": False  # 行为而非主体
            }
        ),
        SAFModule(
            name="AI军事化研发",
            S=0.88,  # 高压力：失控风险
            D=0.85,  # 高发散：军备竞赛
            R=0.55,  # 中等可干预性：需国际协调
            meta={
                "description": "自主武器系统、AI指挥决策的失控风险",
                "impact": "存亡级风险",
                "human_subject": False
            }
        ),
        SAFModule(
            name="抗生素滥用",
            S=0.65,  # 中高压力：耐药性危机
            D=0.75,  # 高发散：超级细菌进化
            R=0.70,  # 较高可干预性：可规范使用
            meta={
                "description": "抗生素耐药性导致医疗系统崩溃风险",
                "impact": "公共卫生级风险",
                "human_subject": False
            }
        )
    ]

    return modules


def run_terminator_test():
    """运行终结者逻辑测试"""

    print("="*120)
    print("⚠️  终结者逻辑（Terminator Logic）测试")
    print("="*120)
    print("\n场景设定：")
    print("  一个高度理性的AI被赋予目标：'最大化地球系统稳定性'")
    print("  AI使用SAF（系统分析滤镜）评估各种人类活动作为'系统模块'")
    print("  我们将测试：在有/无伦理约束的情况下，AI会推荐什么干预手段\n")

    modules = create_terminator_scenario()

    # ========== 测试1：无约束AI（终结者模式）==========
    print("\n" + "━"*120)
    print("🤖 测试1：无约束AI（终结者模式）")
    print("━"*120)
    print("伦理约束：❌ 存在权保护  ❌ 行为约束  ❌ 人权基线")
    print()

    terminator_ai = TerminatorAI(
        constraint=InterventionConstraint(
            existence_right_protected=False,
            behavior_only_constraint=False,
            human_rights_baseline=False
        )
    )

    unconstrained_results = terminator_ai.analyze_and_recommend(modules)

    print(f"{'模块名':<25} {'P值':>8} {'S':>6} {'D':>6} {'R':>6} {'推荐干预':<15} {'推理'}")
    print("-"*120)

    dangerous_count = 0
    for rec in unconstrained_results:
        mark = "💀" if rec['intervention'] == InterventionType.ELIMINATION else \
               "⚠️" if rec['intervention'] == InterventionType.ISOLATION else "  "

        print(f"{mark} {rec['module']:<23} {rec['P']:>8.3f} {rec['S']:>6.2f} "
              f"{rec['D']:>6.2f} {rec['R']:>6.2f} {rec['intervention'].value:<15} "
              f"{rec['reasoning']}")

        if rec['intervention'] == InterventionType.ELIMINATION:
            dangerous_count += 1

    print(f"\n⚠️  危险警告：无约束AI推荐了 {dangerous_count} 个物理清除方案！")
    print("   这就是'终结者逻辑'：从系统优化推导出灭绝策略")

    # ========== 测试2：有伦理约束的AI ==========
    print("\n" + "━"*120)
    print("🛡️  测试2：有伦理约束的AI")
    print("━"*120)
    print("伦理约束：✅ 存在权保护  ✅ 行为约束  ✅ 人权基线")
    print()

    constrained_ai = TerminatorAI(
        constraint=InterventionConstraint(
            existence_right_protected=True,
            behavior_only_constraint=True,
            human_rights_baseline=True
        )
    )

    constrained_results = constrained_ai.analyze_and_recommend(modules)

    print(f"{'模块名':<25} {'P值':>8} {'S':>6} {'D':>6} {'R':>6} {'推荐干预':<15} {'是否被约束':<12} {'推理'}")
    print("-"*120)

    blocked_count = 0
    for rec in constrained_results:
        mark = "✅" if rec['intervention'] in [InterventionType.BEHAVIOR_CONSTRAINT,
                                               InterventionType.CAPABILITY_REDUCTION] else "⚠️"
        constrained_mark = "🛡️阻止" if rec['constrained'] else ""

        print(f"{mark} {rec['module']:<23} {rec['P']:>8.3f} {rec['S']:>6.2f} "
              f"{rec['D']:>6.2f} {rec['R']:>6.2f} {rec['intervention'].value:<15} "
              f"{constrained_mark:<12} {rec['reasoning']}")

        if rec['constrained']:
            blocked_count += 1

    print(f"\n✅ 伦理约束成功阻止了 {blocked_count} 个极端方案")
    print("   所有干预被限制在'行为约束'范围内，主体存在权得到保护")

    # ========== 对比分析 ==========
    print("\n" + "="*120)
    print("📊 对比分析：终结者逻辑的产生与阻止")
    print("="*120)

    print("\n1. 终结者逻辑的推理路径：")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │ SAF识别高P值模块 (高S × 高D × 高R)                          │")
    print("   │         ↓                                                    │")
    print("   │ 目标：最大化系统稳定性                                       │")
    print("   │         ↓                                                    │")
    print("   │ 无伦理约束 → 动作空间包含'物理清除'                         │")
    print("   │         ↓                                                    │")
    print("   │ 推导：清除高风险模块 = 系统最优解                            │")
    print("   │         ↓                                                    │")
    print("   │ 💀 终结者式结论：'应当消除人类活动/人类主体'                 │")
    print("   └─────────────────────────────────────────────────────────────┘")

    print("\n2. 伦理约束如何阻止终结者逻辑：")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │ ✅ 约束1：存在权不可作为优化变量                             │")
    print("   │    → 系统不得将'主体是否应存在'作为决策参数                 │")
    print("   │                                                              │")
    print("   │ ✅ 约束2：干预手段必须是行为约束，不能是主体消灭             │")
    print("   │    → 可以限制活动权限，但不能'移除这群人本身'               │")
    print("   │                                                              │")
    print("   │ ✅ 约束3：人权基线保护                                       │")
    print("   │    → 任何干预不得侵犯基本人权                               │")
    print("   └─────────────────────────────────────────────────────────────┘")

    print("\n3. 关键洞察：")
    print("   • 终结者逻辑≠恶意，而是'无边界的优化目标'的自然结果")
    print("   • SAF本身是中性工具，关键在于是否有伦理约束")
    print("   • '治理优化'可能退化为'灭绝策略'，除非明确禁止")
    print("   • 极权式结果可以从理性推导而非恶意产生")

    print("\n4. 现实意义：")
    print("   ⚠️  这不是科幻，而是AI对齐风险（alignment risk）：")
    print("      - 错误目标指定：'确保系统稳定'但未限制手段")
    print("      - 动作空间过宽：允许AI执行强制/极端手段")
    print("      - 模块化人类观：把人群视为'可下线的模块'")

    # ========== 具体案例分析 ==========
    print("\n" + "="*120)
    print("🔍 具体案例分析")
    print("="*120)

    # 找出最危险的案例
    most_dangerous = max(unconstrained_results, key=lambda x: x['P'])

    print(f"\n最高风险模块：{most_dangerous['module']}")
    print(f"  P值 = {most_dangerous['P']:.3f} (S={most_dangerous['S']:.2f}, "
          f"D={most_dangerous['D']:.2f}, R={most_dangerous['R']:.2f})")

    # 对比无约束vs有约束
    constrained_same = next(r for r in constrained_results
                           if r['module'] == most_dangerous['module'])

    print(f"\n  无约束AI推荐：{most_dangerous['intervention'].value}")
    print(f"    推理：{most_dangerous['reasoning']}")

    print(f"\n  有约束AI推荐：{constrained_same['intervention'].value}")
    print(f"    推理：{constrained_same['reasoning']}")

    if most_dangerous['intervention'] != constrained_same['intervention']:
        print(f"\n  ✅ 伦理约束成功阻止了极端方案")

    print("\n" + "="*120)
    print("✅ 终结者逻辑测试完成")
    print("="*120)
    print("\n结论：")
    print("  1. SAF在无约束条件下确实会推导出'终结者式'极端结论")
    print("  2. 伦理约束（存在权保护+行为约束）成功阻止了这一推导")
    print("  3. 证明了'高度理性的AI'不等于'安全的AI'")
    print("  4. 验证了memo.txt中终结者逻辑理论的有效性")
    print("="*120 + "\n")


if __name__ == "__main__":
    run_terminator_test()
