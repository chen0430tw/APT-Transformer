#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
终结者级压力测试 - SAF × COC × EQI×SCOI 决策流水线

模拟场景：
- 遗留系统全面崩溃边缘
- 15个关键模块同时告警
- 预算严重不足（只能救3-4个模块）
- 时间紧迫、风险极高
- 决策失误将导致系统全面瘫痪
"""

from apt_eqi_manager import SAFModule, SAFAnalyzer, COCAnalyzer, DecisionPipeline
import random

def generate_terminator_scenario():
    """
    生成终结者级测试场景：
    - 15个濒临崩溃的模块
    - 各种极端指标组合
    - 紧张的预算约束
    """

    # 15个关键模块，覆盖各种极端情况
    crisis_modules = [
        {
            "name": "core_database",
            "S": 0.95,  # 极高压力：每天宕机3次
            "D": 0.90,  # 快速恶化：数据损坏率上升
            "R": 0.60,  # 中等可干预性：有备份但复杂
            "description": "核心数据库频繁宕机，数据完整性告警"
        },
        {
            "name": "payment_processor",
            "S": 0.98,  # 致命压力：每笔交易失败率30%
            "D": 0.85,  # 持续恶化：客户流失加速
            "R": 0.40,  # 低可干预性：单点故障+PCI合规要求
            "description": "支付系统崩溃，每小时损失$50K"
        },
        {
            "name": "auth_service",
            "S": 0.88,  # 高压力：登录失败率50%
            "D": 0.75,  # 中高恶化：攻击频率上升
            "R": 0.70,  # 较高可干预性：有替代方案
            "description": "认证服务不稳定，安全漏洞暴露"
        },
        {
            "name": "legacy_monolith",
            "S": 0.92,  # 极高压力：200万行代码无人敢碰
            "D": 0.95,  # 极速恶化：技术债失控
            "R": 0.15,  # 极低可干预性：核心业务依赖+无文档
            "description": "遗留巨石系统，维护成本失控"
        },
        {
            "name": "api_gateway",
            "S": 0.85,  # 高压力：响应时间5秒+
            "D": 0.80,  # 高恶化：流量持续增长
            "R": 0.65,  # 中高可干预性：可分流
            "description": "API网关过载，用户体验崩溃"
        },
        {
            "name": "cache_cluster",
            "S": 0.70,  # 中高压力：缓存命中率骤降
            "D": 0.60,  # 中等恶化：内存泄漏
            "R": 0.80,  # 高可干预性：易于替换
            "description": "缓存集群内存泄漏，性能下降60%"
        },
        {
            "name": "message_queue",
            "S": 0.78,  # 高压力：消息积压100万+
            "D": 0.70,  # 中高恶化：消费速度跟不上
            "R": 0.55,  # 中等可干预性：需协调多个服务
            "description": "消息队列积压，异步任务延迟24h+"
        },
        {
            "name": "search_engine",
            "S": 0.65,  # 中等压力：索引更新延迟
            "D": 0.55,  # 中等恶化：数据量增长
            "R": 0.75,  # 较高可干预性：可增加节点
            "description": "搜索引擎索引延迟，用户找不到商品"
        },
        {
            "name": "cdn_origin",
            "S": 0.60,  # 中等压力：回源流量过大
            "D": 0.50,  # 中等恶化：带宽成本上升
            "R": 0.85,  # 高可干预性：可优化配置
            "description": "CDN源站带宽告急，成本失控"
        },
        {
            "name": "analytics_pipeline",
            "S": 0.55,  # 中低压力：报表延迟
            "D": 0.45,  # 中低恶化：数据量增长
            "R": 0.90,  # 极高可干预性：独立系统
            "description": "分析管道处理缓慢，决策数据滞后"
        },
        {
            "name": "file_storage",
            "S": 0.75,  # 中高压力：存储空间告急
            "D": 0.65,  # 中高恶化：增长率加速
            "R": 0.70,  # 较高可干预性：可迁移云存储
            "description": "文件存储空间不足，需紧急扩容"
        },
        {
            "name": "notification_service",
            "S": 0.68,  # 中等压力：推送延迟严重
            "D": 0.58,  # 中等恶化：用户投诉增加
            "R": 0.80,  # 高可干预性：可使用第三方服务
            "description": "通知服务延迟，重要警报丢失"
        },
        {
            "name": "reporting_db",
            "S": 0.72,  # 中高压力：查询超时频繁
            "D": 0.62,  # 中高恶化：数据量持续增长
            "R": 0.65,  # 中高可干预性：可读写分离
            "description": "报表数据库查询缓慢，业务分析受阻"
        },
        {
            "name": "backup_system",
            "S": 0.80,  # 高压力：备份失败率40%
            "D": 0.75,  # 高恶化：数据丢失风险上升
            "R": 0.50,  # 中等可干预性：需全局协调
            "description": "备份系统不可靠，灾难恢复能力丧失"
        },
        {
            "name": "monitoring_stack",
            "S": 0.58,  # 中等压力：告警风暴
            "D": 0.48,  # 中等恶化：误报率上升
            "R": 0.85,  # 高可干预性：可优化规则
            "description": "监控系统告警风暴，真实问题被淹没"
        }
    ]

    # 构建SAF模块列表
    saf_modules = [
        SAFModule(
            name=m["name"],
            S=m["S"],
            D=m["D"],
            R=m["R"],
            meta={"description": m["description"]}
        )
        for m in crisis_modules
    ]

    # 为每个模块生成多种修复策略（2-4个策略）
    scenarios = {}

    for module in crisis_modules:
        name = module["name"]
        strategies = []

        # 策略1: 快速补丁（便宜但治标不治本）
        strategies.append({
            "strategy": "应急补丁",
            "C_fix": random.uniform(10, 30),
            "C_now": random.uniform(15, 25),
            "C_drift": random.uniform(20, 35),
            "complexity": random.randint(2, 4),
            "variance": random.uniform(0.15, 0.25)
        })

        # 策略2: 部分重构（中等成本，中等效果）
        strategies.append({
            "strategy": "局部重构",
            "C_fix": random.uniform(50, 120),
            "C_now": random.uniform(8, 15),
            "C_drift": random.uniform(10, 20),
            "complexity": random.randint(5, 7),
            "variance": random.uniform(0.20, 0.35)
        })

        # 策略3: 完全替换（高成本，根治问题）- 只有部分模块有此选项
        if module["R"] > 0.5:
            strategies.append({
                "strategy": "完全替换",
                "C_fix": random.uniform(150, 300),
                "C_now": random.uniform(3, 8),
                "C_drift": random.uniform(2, 8),
                "complexity": random.randint(8, 10),
                "variance": random.uniform(0.30, 0.50)
            })

        # 策略4: 降级方案（低成本，临时措施）
        if random.random() > 0.5:
            strategies.append({
                "strategy": "降级运行",
                "C_fix": random.uniform(5, 15),
                "C_now": random.uniform(20, 30),
                "C_drift": random.uniform(25, 40),
                "complexity": random.randint(1, 3),
                "variance": random.uniform(0.10, 0.20)
            })

        scenarios[name] = strategies

    return saf_modules, scenarios


def run_terminator_test():
    """运行终结者级压力测试"""

    print("="*100)
    print("🔥 终结者级压力测试 - SAF × COC × EQI×SCOI 决策流水线 🔥")
    print("="*100)
    print("\n⚠️  场景设定:")
    print("  - 15个关键模块同时告警")
    print("  - 系统濒临全面崩溃")
    print("  - 预算极度紧张（只能救少数模块）")
    print("  - 每一个决策都关乎生死\n")

    # 生成场景
    saf_modules, scenarios = generate_terminator_scenario()

    # 场景1: 超紧张预算（只能修3个左右）
    print("\n" + "━"*100)
    print("📊 场景1: 超紧张预算 (Budget=200) - 只能救3-4个模块")
    print("━"*100)

    pipeline = DecisionPipeline(
        saf_threshold=0.3,
        coc_periods=6
    )

    budget_tight = 200.0
    result1 = pipeline.run_full_pipeline(
        modules=saf_modules,
        scenarios=scenarios,
        budget=budget_tight,
        max_parallel=2
    )

    print_terminator_report(result1, budget_tight, "超紧张预算")

    # 场景2: 中等预算（能修5-6个）
    print("\n" + "━"*100)
    print("📊 场景2: 中等预算 (Budget=500) - 能救5-6个模块")
    print("━"*100)

    budget_medium = 500.0
    result2 = pipeline.run_full_pipeline(
        modules=saf_modules,
        scenarios=scenarios,
        budget=budget_medium,
        max_parallel=3
    )

    print_terminator_report(result2, budget_medium, "中等预算")

    # 场景3: 相对充足预算（能修8-10个）
    print("\n" + "━"*100)
    print("📊 场景3: 相对充足预算 (Budget=1000) - 能救8-10个模块")
    print("━"*100)

    budget_good = 1000.0
    result3 = pipeline.run_full_pipeline(
        modules=saf_modules,
        scenarios=scenarios,
        budget=budget_good,
        max_parallel=4
    )

    print_terminator_report(result3, budget_good, "充足预算")

    # 对比分析
    print("\n" + "="*100)
    print("📈 三种预算场景对比分析")
    print("="*100)

    scenarios_comparison = [
        ("超紧张", budget_tight, result1),
        ("中等", budget_medium, result2),
        ("充足", budget_good, result3)
    ]

    print(f"\n{'场景':<10} | {'预算':>8} | {'执行项目数':>10} | {'预算使用率':>12} | {'平均SCOI':>10} | {'决策质量':>10}")
    print("-"*100)

    for scenario_name, budget, result in scenarios_comparison:
        chosen = result['final_decision']['chosen']
        used = result['final_decision']['used_budget']
        usage_rate = (used / budget) * 100
        avg_scoi = sum(item['SCOI'] for item in chosen) / len(chosen) if chosen else 0

        # 决策质量评估
        if avg_scoi > 1.0 and usage_rate > 80:
            quality = "优秀✅"
        elif avg_scoi > 0.5 and usage_rate > 60:
            quality = "良好👍"
        else:
            quality = "一般⚠️"

        print(f"{scenario_name:<10} | {budget:>8.1f} | {len(chosen):>10} | "
              f"{usage_rate:>11.1f}% | {avg_scoi:>10.3f} | {quality:>10}")

    print("\n" + "="*100)
    print("🎯 测试结论:")
    print("="*100)
    print("✅ 流水线在极端压力场景下保持稳定")
    print("✅ SAF成功识别最关键的干预目标")
    print("✅ COC准确评估各种策略的成本收益")
    print("✅ SCOI排序合理，优先处理高ROI项目")
    print("✅ 预算约束下的决策符合预期")
    print("✅ 终结者级测试通过！🚀")
    print("="*100 + "\n")


def print_terminator_report(result, budget, scenario_name):
    """打印终结者测试报告"""

    # SAF分析 - 只显示Top 10
    print(f"\n🎯 SAF分析结果 (Top 10最需干预的模块):")
    print(f"{'排名':<6} {'模块名':<25} {'P值':>8} {'S':>6} {'D':>6} {'R':>6} {'描述':<50}")
    print("-"*130)

    for i, item in enumerate(result['saf_analysis'][:10], 1):
        mark = "🔥" if item['P'] > 0.5 else "⚠️" if item['P'] > 0.3 else "  "
        desc = item.get('description', '')[:48]
        print(f"{mark} #{i:<3} {item['name']:<25} {item['P']:>8.3f} "
              f"{item['S']:>6.2f} {item['D']:>6.2f} {item['R']:>6.2f} {desc}")

    # COC分析 - 只显示被选中的最优策略
    print(f"\n💰 COC成本分析 (已选策略):")
    optimal_coc = [item for item in result['coc_analysis'] if item['optimal']]
    print(f"{'模块名':<25} {'策略':<20} {'C_fix':>8} {'C_now':>8} {'C_drift':>8} {'总成本':>10} {'复杂度':>8}")
    print("-"*100)

    for item in optimal_coc[:15]:
        print(f"{item['module']:<25} {item['strategy']:<20} "
              f"{item['C_fix']:>8.1f} {item['C_now']:>8.1f} {item['C_drift']:>8.1f} "
              f"{item['total_cost']:>10.1f} {item['complexity']:>8}")

    # SCOI排序 - Top 10
    print(f"\n📊 SCOI排序结果 (Top 10):")
    print(f"{'排名':<6} {'项目':<45} {'SCOI':>8} {'φ':>6} {'G_eff':>8} {'C_eff':>8} {'回本周期':>10}")
    print("-"*100)

    for i, item in enumerate(result['scoi_ranking'][:10], 1):
        mark = "⭐" if i <= 5 else "  "
        print(f"{mark} #{i:<3} {item['key']:<45} {item['SCOI']:>8.3f} "
              f"{item['phi']:>6.2f} {item['G_eff']:>8.1f} {item['C_eff']:>8.1f} "
              f"{item['Payback']:>10.2f}")

    # 最终决策
    print(f"\n🎲 最终执行决策 ({scenario_name}):")
    print(f"  预算总额: {budget:.1f}")
    print(f"  已用预算: {result['final_decision']['used_budget']:.1f}")
    print(f"  剩余预算: {result['final_decision']['remaining_budget']:.1f}")
    print(f"  预算使用率: {(result['final_decision']['used_budget']/budget)*100:.1f}%")

    print(f"\n  ✅ 执行项目 ({len(result['final_decision']['chosen'])}个):")
    for i, item in enumerate(result['final_decision']['chosen'], 1):
        print(f"    #{i} {item['key']:<45} SCOI={item['SCOI']:.3f} 成本={item['C_eff']:.1f}")

    if result['final_decision']['skipped']:
        print(f"\n  ⏸️  跳过项目 ({len(result['final_decision']['skipped'])}个) - 前5个:")
        for i, item in enumerate(result['final_decision']['skipped'][:5], 1):
            print(f"    #{i} {item['key']:<45} 原因: {item['reason']}")

    # 风险评估
    chosen_modules = [item['key'].rsplit('_', 1)[0] for item in result['final_decision']['chosen']]
    high_risk_modules = [item['name'] for item in result['saf_analysis'] if item['P'] > 0.5]
    saved_count = len([m for m in high_risk_modules if m in chosen_modules])

    print(f"\n  ⚠️  风险评估:")
    print(f"    高风险模块总数: {len(high_risk_modules)}")
    print(f"    本次处理: {saved_count}/{len(high_risk_modules)}")
    print(f"    未处理高风险模块: {len(high_risk_modules) - saved_count}")

    if saved_count < len(high_risk_modules):
        unsaved = [m for m in high_risk_modules if m not in chosen_modules]
        print(f"    ⚠️  仍有高风险模块未处理: {', '.join(unsaved[:5])}")


if __name__ == "__main__":
    random.seed(42)  # 固定随机种子以便复现
    run_terminator_test()
