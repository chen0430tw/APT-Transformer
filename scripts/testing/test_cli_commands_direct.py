#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试CLI命令可用性 - 不导入重模块
"""

import subprocess
import sys

def safe_print(msg):
    try:
        print(msg)
    except OSError:
        pass

def test_command(cmd_name, args=None):
    """测试CLI命令是否可用"""
    full_args = ['python3', '-m', 'apt_model', cmd_name]
    if args:
        full_args.extend(args)

    full_args.append('--help')  # 只测试help，不实际执行

    try:
        result = subprocess.run(
            full_args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd='/home/user/APT-Transformer'
        )

        # 命令执行成功或返回help信息都算通过
        if result.returncode == 0 or 'usage:' in result.stdout.lower() or 'help' in result.stdout.lower():
            return True, "OK"
        else:
            return False, result.stderr[:100] if result.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:100]

def main():
    safe_print("=" * 70)
    safe_print("APT-Transformer 四大核心功能CLI测试")
    safe_print("=" * 70)

    tests = [
        ("数据处理", "process-data", None),
        ("训练", "train", None),
        ("聊天", "chat", None),
        ("评估", "evaluate", None),
    ]

    results = {}

    for name, cmd, args in tests:
        safe_print(f"\n【测试 {name}】")
        safe_print(f"  命令: python -m apt_model {cmd} --help")

        success, msg = test_command(cmd, args)

        if success:
            safe_print(f"  ✅ {cmd} 命令可用")
            results[name] = True
        else:
            safe_print(f"  ❌ {cmd} 命令失败: {msg}")
            results[name] = False

    # 总结
    safe_print("\n" + "=" * 70)
    safe_print("总结")
    safe_print("=" * 70)

    for name, result in results.items():
        status = "✅" if result else "❌"
        safe_print(f"{status} {name}")

    passed = sum(1 for v in results.values() if v)
    failed = 4 - passed

    safe_print(f"\n通过: {passed}/4 | 失败: {failed}/4")

    if failed == 0:
        safe_print("\n🎉 所有核心功能CLI命令可用！")
        safe_print("\n使用示例:")
        safe_print("  python -m apt_model process-data data.txt")
        safe_print("  python -m apt_model train --profile lite")
        safe_print("  python -m apt_model chat")
        safe_print("  python -m apt_model evaluate model.pt")
        return 0
    else:
        safe_print("\n⚠️  部分命令需要检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())
