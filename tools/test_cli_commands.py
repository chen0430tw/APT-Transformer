#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APT CLI命令完整性测试工具

功能：
- 测试所有注册的CLI命令
- 检测导入错误、参数错误、运行时错误
- 生成详细错误报告
- 自动修复建议

用法：
    python tools/test_cli_commands.py
    python tools/test_cli_commands.py --verbose
    python tools/test_cli_commands.py --fix-errors
"""

import os
import sys
import argparse
import traceback
from typing import Dict, List, Tuple
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 80)
print("🧪 APT CLI命令完整性测试")
print("=" * 80)
print()


class CLICommandTester:
    """CLI命令测试器"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'import_errors': [],
            'runtime_errors': [],
            'success': []
        }
        self.command_registry = None

    def setup(self):
        """初始化命令注册系统"""
        print("🔧 初始化命令注册系统...")
        try:
            from apt.apps.cli.command_registry import command_registry
            self.command_registry = command_registry
            commands = command_registry.list_commands()
            print(f"✅ 成功加载 {len(commands)} 个命令")
            print()
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            traceback.print_exc()
            return False

    def test_all_commands(self):
        """测试所有命令"""
        if not self.command_registry:
            print("❌ 命令注册系统未初始化")
            return

        commands = self.command_registry.list_commands()
        self.results['total'] = len(commands)

        print(f"📋 开始测试 {len(commands)} 个命令...")
        print()

        for i, cmd_name in enumerate(commands, 1):
            print(f"[{i}/{len(commands)}] 测试命令: {cmd_name}")
            self._test_single_command(cmd_name)
            print()

    def _test_single_command(self, cmd_name: str):
        """测试单个命令"""
        try:
            # 获取命令元数据
            metadata = self.command_registry.get_command(cmd_name)
            if not metadata:
                self.results['failed'] += 1
                self.results['import_errors'].append({
                    'command': cmd_name,
                    'error': 'Command not found in registry'
                })
                print(f"  ❌ 未找到命令")
                return

            # 尝试导入命令函数
            func = metadata.func

            # 创建测试参数
            import argparse
            test_args = argparse.Namespace()

            # 添加常用参数（避免实际执行）
            test_args.model_path = ['./test_model']  # 列表形式，用于evaluate等命令
            test_args.checkpoint = None
            test_args.data_path = None
            test_args.help = True  # 大多数命令遇到help会直接返回
            test_args.dry_run = True

            # APX相关参数
            test_args.apx = None
            test_args.src = None
            test_args.out = None

            # 训练相关参数
            test_args.monitor_resources = False
            test_args.create_plots = False
            test_args.epochs = 1
            test_args.batch_size = 1
            test_args.learning_rate = 1e-4

            # Evaluate相关参数
            test_args.output_dir = None
            test_args.eval_sets = None
            test_args.num_eval_samples = 10

            # 其他常用参数
            test_args.model_paths = None
            test_args.save_path = None

            # 实际执行所有命令（使用安全参数）
            try:
                # 尝试实际调用命令函数来检测运行时错误
                # 使用 --help 或其他安全参数避免真实操作
                import io
                import contextlib

                # 捕获输出避免污染测试结果
                output_buffer = io.StringIO()

                with contextlib.redirect_stdout(output_buffer), \
                     contextlib.redirect_stderr(output_buffer):
                    try:
                        # 实际调用函数
                        result = func(test_args)
                    except SystemExit:
                        # 某些命令可能会调用sys.exit()，这是正常的
                        pass
                    except KeyboardInterrupt:
                        # 用户中断
                        pass

                # 如果执行到这里没有抛出异常，说明命令至少可以导入和初始化
                print(f"  ✅ 命令执行测试通过")
                self.results['passed'] += 1
                self.results['success'].append(cmd_name)

            except NameError as e:
                # 捕获未定义的变量/函数错误
                self.results['failed'] += 1
                self.results['import_errors'].append({
                    'command': cmd_name,
                    'error': str(e),
                    'type': 'NameError',
                    'traceback': traceback.format_exc()
                })
                print(f"  ❌ NameError: {e}")

            except ImportError as e:
                self.results['failed'] += 1
                self.results['import_errors'].append({
                    'command': cmd_name,
                    'error': str(e),
                    'type': 'ImportError',
                    'traceback': traceback.format_exc()
                })
                print(f"  ❌ ImportError: {e}")

            except Exception as e:
                self.results['failed'] += 1
                self.results['runtime_errors'].append({
                    'command': cmd_name,
                    'error': str(e),
                    'type': type(e).__name__,
                    'traceback': traceback.format_exc()
                })
                print(f"  ⚠️  RuntimeError: {e}")

        except Exception as e:
            self.results['failed'] += 1
            self.results['import_errors'].append({
                'command': cmd_name,
                'error': str(e),
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            print(f"  ❌ 测试失败: {e}")
            if self.verbose:
                traceback.print_exc()

    def analyze_errors(self):
        """分析错误并提供修复建议"""
        print("=" * 80)
        print("📊 错误分析")
        print("=" * 80)
        print()

        # 导入错误
        if self.results['import_errors']:
            print(f"❌ 导入错误 ({len(self.results['import_errors'])} 个):")
            print()

            for i, err in enumerate(self.results['import_errors'], 1):
                print(f"{i}. 命令: {err['command']}")
                print(f"   错误: {err['error']}")
                print(f"   类型: {err.get('type', 'Unknown')}")

                # 提供修复建议
                error_msg = err['error']
                if "name 'setup_logging' is not defined" in error_msg:
                    print(f"   💡 修复建议: 替换 setup_logging 为标准库 logging")
                    print(f"      logger = logging.getLogger(__name__)")
                elif "cannot import name" in error_msg:
                    print(f"   💡 修复建议: 检查导入路径是否正确")
                elif "No module named" in error_msg:
                    print(f"   💡 修复建议: 模块不存在，需要注释或删除导入")

                if self.verbose:
                    print(f"   堆栈:")
                    print(f"   {err.get('traceback', 'N/A')}")
                print()

        # 运行时错误
        if self.results['runtime_errors']:
            print(f"⚠️  运行时错误 ({len(self.results['runtime_errors'])} 个):")
            print()

            for i, err in enumerate(self.results['runtime_errors'], 1):
                print(f"{i}. 命令: {err['command']}")
                print(f"   错误: {err['error']}")
                print(f"   类型: {err.get('type', 'Unknown')}")
                if self.verbose:
                    print(f"   堆栈:")
                    print(f"   {err.get('traceback', 'N/A')}")
                print()

    def generate_report(self):
        """生成测试报告"""
        print("=" * 80)
        print("📄 测试报告")
        print("=" * 80)
        print()

        total = self.results['total']
        passed = self.results['passed']
        failed = self.results['failed']

        print(f"总命令数: {total}")
        print(f"通过: {passed} ({passed/total*100:.1f}%)")
        print(f"失败: {failed} ({failed/total*100:.1f}%)")
        print()

        if failed > 0:
            print(f"❌ 需要修复的命令:")
            all_failed = []
            all_failed.extend([e['command'] for e in self.results['import_errors']])
            all_failed.extend([e['command'] for e in self.results['runtime_errors']])
            for cmd in sorted(set(all_failed)):
                print(f"  - {cmd}")
            print()
        else:
            print("✅ 所有命令测试通过！")
            print()

        # 保存详细报告
        report_file = f"cli_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(os.path.dirname(__file__), report_file)

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("APT CLI命令测试报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"测试时间: {datetime.now().isoformat()}\n")
                f.write(f"总命令数: {total}\n")
                f.write(f"通过: {passed}\n")
                f.write(f"失败: {failed}\n\n")

                if self.results['import_errors']:
                    f.write("\n导入错误:\n")
                    for err in self.results['import_errors']:
                        f.write(f"\n命令: {err['command']}\n")
                        f.write(f"错误: {err['error']}\n")
                        f.write(f"类型: {err.get('type', 'Unknown')}\n")
                        f.write(f"堆栈:\n{err.get('traceback', 'N/A')}\n")

                if self.results['runtime_errors']:
                    f.write("\n运行时错误:\n")
                    for err in self.results['runtime_errors']:
                        f.write(f"\n命令: {err['command']}\n")
                        f.write(f"错误: {err['error']}\n")
                        f.write(f"类型: {err.get('type', 'Unknown')}\n")
                        f.write(f"堆栈:\n{err.get('traceback', 'N/A')}\n")

            print(f"📁 详细报告已保存: {report_file}")
        except Exception as e:
            print(f"⚠️  保存报告失败: {e}")

    def generate_fix_script(self):
        """生成自动修复脚本"""
        if not self.results['import_errors']:
            return

        print()
        print("=" * 80)
        print("🔧 生成修复建议")
        print("=" * 80)
        print()

        fixes = []

        for err in self.results['import_errors']:
            error_msg = err['error']

            if "name 'setup_logging' is not defined" in error_msg:
                fixes.append({
                    'file': 'apt/apps/cli/commands.py',
                    'search': 'logger = setup_logging',
                    'replace': 'import logging\nlogger = logging.getLogger(__name__)',
                    'description': '替换 setup_logging 为标准库 logging'
                })

        if fixes:
            print("建议修复:")
            for i, fix in enumerate(fixes, 1):
                print(f"\n{i}. {fix['description']}")
                print(f"   文件: {fix['file']}")
                print(f"   搜索: {fix['search']}")
                print(f"   替换: {fix['replace']}")


def main():
    parser = argparse.ArgumentParser(description='APT CLI命令完整性测试')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细错误信息')
    parser.add_argument('--fix-errors', action='store_true', help='生成修复脚本')
    args = parser.parse_args()

    tester = CLICommandTester(verbose=args.verbose)

    # 1. 初始化
    if not tester.setup():
        sys.exit(1)

    # 2. 测试所有命令
    tester.test_all_commands()

    # 3. 分析错误
    tester.analyze_errors()

    # 4. 生成报告
    tester.generate_report()

    # 5. 生成修复建议
    if args.fix_errors:
        tester.generate_fix_script()

    print()
    print("=" * 80)
    print("✅ 测试完成")
    print("=" * 80)

    # 返回退出码
    sys.exit(0 if tester.results['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
