#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看测试报告 - 友好的报告查看器
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def find_latest_report():
    """查找最新的测试报告"""
    log_dir = Path("./test_logs")
    if not log_dir.exists():
        return None

    json_files = list(log_dir.glob("command_test_*.json"))
    if not json_files:
        return None

    # 按修改时间排序，返回最新的
    return max(json_files, key=lambda p: p.stat().st_mtime)


def print_colored(text, color=""):
    """打印彩色文本（兼容终端）"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
        "bold": "\033[1m",
    }
    if color and color in colors:
        print(f"{colors[color]}{text}{colors['reset']}")
    else:
        print(text)


def view_report(json_file):
    """查看测试报告"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    results = data["results"]

    # 打印标题
    print_colored("\n" + "=" * 80, "bold")
    print_colored("APT Model 命令测试报告", "bold")
    print_colored("=" * 80, "bold")

    # 打印摘要
    print(f"\n📊 测试摘要")
    print(f"   时间: {summary['timestamp']}")
    print(f"   总计: {summary['total']} 个命令")
    print_colored(f"   ✓ 通过: {summary['passed']}", "green")
    print_colored(f"   ✗ 失败: {summary['failed']}", "red")
    print_colored(f"   ⊘ 跳过: {summary['skipped']}", "yellow")

    if summary['total'] - summary['skipped'] > 0:
        success_rate = (summary['passed'] / (summary['total'] - summary['skipped'])) * 100
        color = "green" if success_rate > 80 else "yellow" if success_rate > 50 else "red"
        print_colored(f"   成功率: {success_rate:.1f}%", color)

    # 分类统计
    print("\n" + "-" * 80)
    print("📋 详细结果\n")

    # 分类
    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    warnings = [r for r in results if r["status"] == "warning"]

    # 显示失败的命令（最重要）
    if failed:
        print_colored("❌ 失败的命令:", "red")
        for r in failed:
            error_msg = r.get("error", "未知错误")
            exit_code = r.get("exit_code", "N/A")
            print(f"   • {r['command']}")
            print(f"     错误: {error_msg}")
            print(f"     退出码: {exit_code}")

            # 显示错误详情（前200字符）
            if r.get("stderr"):
                stderr_preview = r["stderr"][:200].replace("\n", " ")
                print(f"     详情: {stderr_preview}...")
            print()

    # 显示警告的命令
    if warnings:
        print_colored("⚠️  警告的命令:", "yellow")
        for r in warnings:
            print(f"   • {r['command']}: {r.get('error', 'N/A')}")
        print()

    # 显示跳过的命令
    if skipped:
        print_colored("⊘ 跳过的命令:", "yellow")
        for r in skipped:
            print(f"   • {r['command']}: {r.get('reason', 'N/A')}")
        print()

    # 显示通过的命令
    if passed:
        print_colored("✅ 通过的命令:", "green")
        for r in passed:
            duration = r.get("duration", 0)
            print(f"   • {r['command']} ({duration:.2f}s)")
        print()

    # 根本原因分析
    print("-" * 80)
    print("🔍 根本原因分析\n")

    # 统计错误类型
    error_types = {}
    for r in failed:
        stderr = r.get("stderr", "")
        if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
            # 提取模块名
            if "No module named" in stderr:
                module = stderr.split("No module named")[1].split("\n")[0].strip().strip("'\"")
                error_types.setdefault("缺失依赖", []).append(module)
            else:
                error_types.setdefault("缺失依赖", []).append("未知模块")
        elif "未知" in r.get("stdout", "") or "unknown" in r.get("stdout", "").lower():
            error_types.setdefault("未知命令", []).append(r["command"])
        else:
            error_types.setdefault("其他错误", []).append(r["command"])

    for error_type, items in error_types.items():
        print(f"   {error_type}:")
        unique_items = list(set(items))
        for item in unique_items[:5]:  # 只显示前5个
            print(f"      • {item}")
        if len(unique_items) > 5:
            print(f"      ... 还有 {len(unique_items) - 5} 个")
        print()

    # 修复建议
    print("-" * 80)
    print("💡 修复建议\n")

    if any("torch" in str(items) for items in error_types.get("缺失依赖", [])):
        print("   1. 安装 PyTorch 和相关依赖:")
        print("      pip install torch transformers")
        print()

    if error_types.get("缺失依赖"):
        print("   2. 安装完整依赖:")
        print("      pip install -r requirements.txt")
        print()

    if error_types.get("未知命令"):
        print("   3. 检查未知命令是否已注册:")
        print("      查看 apt_model/cli/commands.py")
        print()

    print("-" * 80)
    print(f"\n完整日志: {json_file}")
    print(f"文本日志: {json_file.with_suffix('.log')}")
    print()


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 使用指定的文件
        json_file = Path(sys.argv[1])
    else:
        # 查找最新的报告
        json_file = find_latest_report()

    if not json_file or not json_file.exists():
        print("❌ 未找到测试报告")
        print("\n请先运行测试:")
        print("  python test_all_commands.py")
        sys.exit(1)

    view_report(json_file)


if __name__ == "__main__":
    main()
