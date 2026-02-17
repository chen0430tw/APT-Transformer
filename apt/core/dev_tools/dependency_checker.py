#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖检查工具 - 在不运行真实Torch的情况下检测import错误

使用方法:
    python -m apt.core.dev_tools.dependency_checker check apt/trainops/
    python -m apt.core.dev_tools.dependency_checker check apt/apps/cli/commands.py
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict


def safe_print(*args, **kwargs):
    """Print with error handling for bad file descriptors"""
    import builtins
    try:
        builtins.print(*args, **kwargs)
    except OSError:
        pass


class DependencyChecker:
    """检查Python模块的依赖关系和import错误"""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.issues: List[Dict] = []
        self.checked_files: Set[str] = set()
        self.import_graph: Dict[str, List[str]] = defaultdict(list)

    def check_file(self, file_path: str) -> List[Dict]:
        """检查单个文件的import问题"""
        if file_path in self.checked_files:
            return []

        self.checked_files.add(file_path)
        file_issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            return [{
                'file': file_path,
                'type': 'syntax_error',
                'message': str(e),
                'line': e.lineno,
            }]
        except Exception as e:
            return [{
                'file': file_path,
                'type': 'parse_error',
                'message': str(e),
            }]

        # 提取所有import语句
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                issue = self._check_import_from(node, file_path)
                if issue:
                    file_issues.append(issue)

            elif isinstance(node, ast.Import):
                issue = self._check_import(node, file_path)
                if issue:
                    file_issues.append(issue)

        self.issues.extend(file_issues)
        return file_issues

    def _check_import_from(self, node: ast.ImportFrom, file_path: str) -> Dict:
        """检查 from X import Y 语句"""
        if node.module is None:
            return None

        module_name = node.module
        imported_names = [alias.name for alias in node.names]

        # 记录导入关系
        self.import_graph[file_path].append(module_name)

        # 尝试导入模块
        try:
            # 处理相对导入
            if node.level > 0:
                # 相对导入，需要计算实际模块路径
                module_name = self._resolve_relative_import(
                    file_path, module_name, node.level
                )
                if module_name is None:
                    return None

            # 尝试导入
            try:
                mod = __import__(module_name, fromlist=imported_names)
            except ImportError:
                # 模块不存在，但可能是项目内部模块，暂时跳过
                return None

            # 检查每个导入的名称是否存在
            missing_names = []
            for name in imported_names:
                if name == '*':
                    continue
                if not hasattr(mod, name):
                    missing_names.append(name)

            if missing_names:
                return {
                    'file': file_path,
                    'type': 'missing_import',
                    'module': module_name,
                    'missing': missing_names,
                    'line': node.lineno,
                    'message': f"无法从 {module_name} 导入 {', '.join(missing_names)}",
                }

        except Exception as e:
            # 其他错误（如语法错误等）
            return {
                'file': file_path,
                'type': 'import_error',
                'module': module_name,
                'line': node.lineno,
                'message': str(e),
            }

        return None

    def _check_import(self, node: ast.Import, file_path: str) -> Dict:
        """检查 import X 语句"""
        for alias in node.names:
            module_name = alias.name
            self.import_graph[file_path].append(module_name)

            try:
                __import__(module_name)
            except ImportError as e:
                # 可能是项目内部模块，暂时跳过
                pass
            except Exception as e:
                return {
                    'file': file_path,
                    'type': 'import_error',
                    'module': module_name,
                    'line': node.lineno,
                    'message': str(e),
                }

        return None

    def _resolve_relative_import(
        self, file_path: str, module: str, level: int
    ) -> str:
        """解析相对导入到绝对导入"""
        # 获取文件所在包的路径
        rel_path = os.path.relpath(file_path, self.project_root)
        parts = rel_path.replace(os.sep, '.').replace('.py', '').split('.')

        # 去掉文件名本身
        parts = parts[:-1]

        # 根据level往上退
        for _ in range(level - 1):
            if parts:
                parts.pop()

        # 添加模块名
        if module:
            parts.append(module)

        return '.'.join(parts) if parts else None

    def check_directory(self, directory: str) -> List[Dict]:
        """递归检查目录下所有Python文件"""
        all_issues = []

        for root, dirs, files in os.walk(directory):
            # 跳过__pycache__等目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    issues = self.check_file(file_path)
                    all_issues.extend(issues)

        return all_issues

    def print_report(self):
        """打印检查报告"""
        safe_print("\n" + "=" * 80)
        safe_print("依赖检查报告")
        safe_print("=" * 80)

        if not self.issues:
            safe_print("\n✓ 未发现import问题")
            safe_print(f"已检查 {len(self.checked_files)} 个文件")
            return

        # 按文件分组
        issues_by_file = defaultdict(list)
        for issue in self.issues:
            issues_by_file[issue['file']].append(issue)

        safe_print(f"\n✗ 发现 {len(self.issues)} 个问题，涉及 {len(issues_by_file)} 个文件\n")

        for file_path, issues in sorted(issues_by_file.items()):
            rel_path = os.path.relpath(file_path, self.project_root)
            safe_print(f"\n{rel_path}:")

            for issue in issues:
                line_info = f"L{issue.get('line', '?')}" if 'line' in issue else ""
                issue_type = issue['type'].replace('_', ' ').title()

                safe_print(f"  [{issue_type}] {line_info}")
                safe_print(f"    {issue['message']}")

                if 'missing' in issue:
                    safe_print(f"    缺失: {', '.join(issue['missing'])}")

        safe_print("\n" + "=" * 80)
        safe_print(f"总计: {len(self.checked_files)} 个文件, {len(self.issues)} 个问题")
        safe_print("=" * 80)


def main():
    """命令行入口"""
    if len(sys.argv) < 3:
        safe_print("使用方法:")
        safe_print("  python -m apt.core.dev_tools.dependency_checker check <path>")
        safe_print("\n示例:")
        safe_print("  python -m apt.core.dev_tools.dependency_checker check apt/trainops/")
        safe_print("  python -m apt.core.dev_tools.dependency_checker check apt/apps/cli/")
        sys.exit(1)

    command = sys.argv[1]
    target_path = sys.argv[2]

    if command != 'check':
        safe_print(f"未知命令: {command}")
        sys.exit(1)

    # 确定项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # apt/core/dev_tools -> APT-Transformer

    checker = DependencyChecker(str(project_root))

    print(f"检查目标: {target_path}")
    print(f"项目根目录: {project_root}\n")

    target = os.path.join(project_root, target_path)

    if os.path.isfile(target):
        checker.check_file(target)
    elif os.path.isdir(target):
        checker.check_directory(target)
    else:
        safe_print(f"错误: 找不到 {target}")
        sys.exit(1)

    checker.print_report()

    # 返回非零退出码如果有问题
    sys.exit(1 if checker.issues else 0)


if __name__ == '__main__':
    main()
