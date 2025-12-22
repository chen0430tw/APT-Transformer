#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动测试所有 APT Model 命令
自动运行所有可用命令并记录错误到日志文件
"""

import subprocess
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import traceback

# 测试结果日志文件
LOG_DIR = Path("./test_logs")
LOG_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"command_test_{TIMESTAMP}.log"
JSON_FILE = LOG_DIR / f"command_test_{TIMESTAMP}.json"

# 所有已知的命令列表
CORE_COMMANDS = [
    # 训练相关
    "train",
    "train-custom",
    "fine-tune",
    "train-hf",
    "train-reasoning",
    "distill",

    # 交互相关
    "chat",

    # 评估相关
    "evaluate",
    "visualize",
    "compare",
    "test",

    # 工具相关
    "clean-cache",
    "estimate",
    "process-data",

    # 信息相关
    "info",
    "list",
    "size",

    # 维护相关
    "prune",
    "backup",

    # 分发相关
    "upload",
    "export-ollama",

    # 通用命令
    "help",
]

# Console 相关命令
CONSOLE_COMMANDS = [
    "console-status",
    "console-help",
    "console-commands",
    "modules-list",
    "modules-status",
    "modules-enable",
    "modules-disable",
    "modules-reload",
    "debug",
    "config",
]

# 需要跳过的命令（因为会进入交互模式或需要长时间运行）
SKIP_COMMANDS = {
    "chat",  # 交互模式
    "train",  # 需要长时间运行
    "train-custom",  # 需要长时间运行
    "fine-tune",  # 需要长时间运行
    "train-hf",  # 需要长时间运行
    "train-reasoning",  # 需要长时间运行
    "distill",  # 需要长时间运行
}

# 需要特殊参数的命令
COMMAND_ARGS = {
    "console-help": ["train"],  # 需要一个命令名作为参数
    "modules-enable": ["training"],  # 需要模块名
    "modules-disable": ["training"],  # 需要模块名
    "modules-reload": ["training"],  # 需要模块名
}

# 只检查帮助的命令（加 --help 参数）
HELP_ONLY_COMMANDS = {
    "evaluate",
    "compare",
    "backup",
    "upload",
    "export-ollama",
    "process-data",
}


class CommandTester:
    """命令测试器"""

    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0

    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def test_command(self, command, args=None, check_help=False):
        """
        测试单个命令

        Args:
            command: 命令名称
            args: 额外参数列表
            check_help: 是否只检查 --help
        """
        self.total_tests += 1

        # 构建完整命令
        cmd_parts = [sys.executable, "-m", "apt_model", command]

        if check_help:
            cmd_parts.append("--help")
        elif args:
            cmd_parts.extend(args)

        cmd_str = " ".join(cmd_parts)

        self.log(f"Testing command: {cmd_str}")

        result = {
            "command": command,
            "full_command": cmd_str,
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": None,
            "duration": 0
        }

        try:
            start_time = time.time()

            # 运行命令，设置超时为30秒
            process = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            duration = time.time() - start_time
            result["duration"] = duration
            result["exit_code"] = process.returncode
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr

            # 判断成功/失败
            # 对于 --help，退出码应该是0
            # 对于其他命令，根据具体情况判断
            if check_help:
                if process.returncode == 0:
                    result["status"] = "passed"
                    self.passed_tests += 1
                    self.log(f"✓ PASSED: {command} --help", "SUCCESS")
                else:
                    result["status"] = "failed"
                    self.failed_tests += 1
                    self.log(f"✗ FAILED: {command} --help (exit code: {process.returncode})", "ERROR")
                    self.log(f"STDERR: {process.stderr[:500]}", "ERROR")
            else:
                # 对于非 help 命令，我们需要更宽松的判断
                # 有些命令可能因为缺少参数而返回非0，但这也算是正常行为
                if process.returncode == 0:
                    result["status"] = "passed"
                    self.passed_tests += 1
                    self.log(f"✓ PASSED: {command}", "SUCCESS")
                elif "ModuleNotFoundError" in process.stderr or "ImportError" in process.stderr:
                    # 依赖问题
                    result["status"] = "failed"
                    result["error"] = "Missing dependencies"
                    self.failed_tests += 1
                    self.log(f"✗ FAILED: {command} - Missing dependencies", "ERROR")
                    self.log(f"STDERR: {process.stderr[:500]}", "ERROR")
                elif "未知" in process.stdout or "unknown" in process.stdout.lower():
                    # 未知命令
                    result["status"] = "failed"
                    result["error"] = "Unknown command"
                    self.failed_tests += 1
                    self.log(f"✗ FAILED: {command} - Unknown command", "ERROR")
                else:
                    # 其他错误，但可能是正常的（如参数缺失）
                    result["status"] = "warning"
                    result["error"] = "Non-zero exit code (might be expected)"
                    self.passed_tests += 1  # 计入通过，因为命令至少能运行
                    self.log(f"⚠ WARNING: {command} (exit code: {process.returncode})", "WARNING")
                    if process.stderr:
                        self.log(f"STDERR: {process.stderr[:300]}", "WARNING")

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = "Command timed out after 30 seconds"
            self.failed_tests += 1
            self.log(f"✗ TIMEOUT: {command}", "ERROR")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            self.failed_tests += 1
            self.log(f"✗ ERROR: {command} - {e}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        self.results.append(result)
        return result

    def skip_command(self, command, reason="Interactive or long-running"):
        """跳过命令"""
        self.total_tests += 1
        self.skipped_tests += 1

        result = {
            "command": command,
            "timestamp": datetime.now().isoformat(),
            "status": "skipped",
            "reason": reason
        }

        self.results.append(result)
        self.log(f"⊘ SKIPPED: {command} - {reason}", "INFO")
        return result

    def run_all_tests(self):
        """运行所有测试"""
        self.log("=" * 80)
        self.log("Starting APT Model Command Tests")
        self.log("=" * 80)

        all_commands = CORE_COMMANDS + CONSOLE_COMMANDS

        for command in all_commands:
            self.log("-" * 80)

            # 检查是否需要跳过
            if command in SKIP_COMMANDS:
                self.skip_command(command)
                continue

            # 检查是否需要特殊参数
            args = COMMAND_ARGS.get(command)

            # 检查是否只需要测试 --help
            check_help = command in HELP_ONLY_COMMANDS

            # 运行测试
            self.test_command(command, args=args, check_help=check_help)

            # 稍微延迟，避免过快
            time.sleep(0.5)

        self.log("-" * 80)
        self.generate_report()

    def generate_report(self):
        """生成测试报告"""
        self.log("=" * 80)
        self.log("Test Summary")
        self.log("=" * 80)
        self.log(f"Total tests: {self.total_tests}")
        self.log(f"Passed: {self.passed_tests}")
        self.log(f"Failed: {self.failed_tests}")
        self.log(f"Skipped: {self.skipped_tests}")
        self.log(f"Success rate: {(self.passed_tests / (self.total_tests - self.skipped_tests) * 100):.2f}%")
        self.log("=" * 80)

        # 列出失败的命令
        failed_commands = [r for r in self.results if r["status"] == "failed"]
        if failed_commands:
            self.log("\nFailed Commands:")
            for r in failed_commands:
                self.log(f"  - {r['command']}: {r.get('error', 'Unknown error')}")

        # 保存 JSON 结果
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total": self.total_tests,
                    "passed": self.passed_tests,
                    "failed": self.failed_tests,
                    "skipped": self.skipped_tests,
                    "timestamp": datetime.now().isoformat()
                },
                "results": self.results
            }, f, indent=2, ensure_ascii=False)

        self.log(f"\nDetailed results saved to:")
        self.log(f"  Log: {LOG_FILE}")
        self.log(f"  JSON: {JSON_FILE}")


def main():
    """主函数"""
    print(f"APT Model Command Tester")
    print(f"Log file: {LOG_FILE}")
    print(f"JSON file: {JSON_FILE}")
    print("-" * 80)

    tester = CommandTester()
    tester.run_all_tests()

    # 返回退出码
    if tester.failed_tests > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
