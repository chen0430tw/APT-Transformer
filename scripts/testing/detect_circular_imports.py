#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循环导入检测工具 - 全面扫描APT-Transformer项目
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

def safe_print(msg):
    try:
        print(msg)
    except OSError:
        pass

class ImportAnalyzer(ast.NodeVisitor):
    """分析Python文件中的导入语句"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.imports = []

    def visit_Import(self, node):
        """处理 import xxx 语句"""
        for alias in node.names:
            self.imports.append(alias.name.split('.')[0])  # 只取顶层包名
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """处理 from xxx import yyy 语句"""
        if node.module:
            # 获取顶层模块名
            if node.level == 0:  # 绝对导入
                self.imports.append(node.module.split('.')[0])
            else:  # 相对导入
                # 根据当前模块名和相对层级计算实际模块
                parts = self.module_name.split('.')
                # 向上 node.level 层
                if len(parts) >= node.level:
                    base = '.'.join(parts[:-node.level]) if node.level > 0 else self.module_name
                    if node.module:
                        full_module = f"{base}.{node.module}" if base else node.module
                    else:
                        full_module = base
                    self.imports.append(full_module.split('.')[0])
        self.generic_visit(node)

def get_module_name(file_path: Path, project_root: Path) -> str:
    """从文件路径获取Python模块名"""
    try:
        rel_path = file_path.relative_to(project_root)
        parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        # 移除__init__
        if parts[-1] == '__init__':
            parts = parts[:-1]
        return '.'.join(parts)
    except ValueError:
        return str(file_path.stem)

def analyze_file(file_path: Path, project_root: Path) -> Tuple[str, List[str]]:
    """分析单个Python文件的导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        module_name = get_module_name(file_path, project_root)
        tree = ast.parse(content, filename=str(file_path))

        analyzer = ImportAnalyzer(module_name)
        analyzer.visit(tree)

        # 只保留项目内部的导入（以apt开头）
        internal_imports = [imp for imp in analyzer.imports if imp == 'apt']

        return module_name, internal_imports
    except Exception as e:
        return get_module_name(file_path, project_root), []

def find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    """使用DFS查找所有循环依赖"""
    cycles = []
    visited = set()

    def dfs(node: str, path: List[str], path_set: Set[str]):
        if node in path_set:
            # 找到循环
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            # 标准化循环（从字母序最小的节点开始）
            min_idx = cycle.index(min(cycle[:-1]))
            normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
            if normalized not in cycles:
                cycles.append(normalized)
            return

        if node in visited:
            return

        path.append(node)
        path_set.add(node)

        for neighbor in graph.get(node, []):
            dfs(neighbor, path, path_set)

        path.pop()
        path_set.remove(node)
        visited.add(node)

    for node in graph:
        if node not in visited:
            dfs(node, [], set())

    return cycles

def build_import_graph(project_root: Path) -> Dict[str, List[str]]:
    """构建整个项目的导入关系图"""
    graph = defaultdict(list)
    apt_dir = project_root / 'apt'

    safe_print(f"扫描目录: {apt_dir}")
    safe_print("=" * 70)

    file_count = 0
    for py_file in apt_dir.rglob('*.py'):
        # 跳过测试文件和__pycache__
        if '__pycache__' in str(py_file) or 'test_' in py_file.name:
            continue

        module_name, imports = analyze_file(py_file, project_root)
        if imports:
            # 只记录apt模块之间的依赖
            for imp in imports:
                if imp == 'apt' and module_name.startswith('apt'):
                    graph[module_name.split('.')[0]].append(imp)

        file_count += 1
        if file_count % 50 == 0:
            safe_print(f"  已扫描 {file_count} 个文件...")

    safe_print(f"✓ 扫描完成，共 {file_count} 个文件")
    return dict(graph)

def detect_direct_circular_imports(project_root: Path):
    """检测直接的循环导入（文件级别）"""
    safe_print("\n" + "=" * 70)
    safe_print("检测文件级别的循环导入")
    safe_print("=" * 70)

    graph = {}
    apt_dir = project_root / 'apt'

    for py_file in apt_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue

        module_name, imports = analyze_file(py_file, project_root)

        # 过滤出apt.开头的导入
        apt_imports = [imp for imp in imports if imp.startswith('apt')]
        if apt_imports:
            graph[module_name] = apt_imports

    # 查找循环
    cycles = find_cycles(graph)

    if cycles:
        safe_print(f"\n🔴 发现 {len(cycles)} 个循环导入:")
        for i, cycle in enumerate(cycles, 1):
            safe_print(f"\n循环 {i}:")
            for j in range(len(cycle) - 1):
                safe_print(f"  {cycle[j]}")
                safe_print(f"    ↓ imports")
            safe_print(f"  {cycle[-1]}")
            safe_print(f"    ↑ (循环)")
    else:
        safe_print("\n✅ 未发现文件级别的循环导入")

    return cycles

def check_package_init_imports(project_root: Path):
    """检查__init__.py文件的导入，这些最容易导致循环"""
    safe_print("\n" + "=" * 70)
    safe_print("检查 __init__.py 文件的导入")
    safe_print("=" * 70)

    apt_dir = project_root / 'apt'
    issues = []

    for init_file in apt_dir.rglob('__init__.py'):
        rel_path = init_file.relative_to(project_root)
        module_name = get_module_name(init_file, project_root)

        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否有未保护的导入
            lines = content.split('\n')
            unprotected_imports = []

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith(('from apt.', 'import apt.')):
                    # 检查是否在try-except块中
                    # 简单检查：看前面几行是否有try
                    in_try_block = False
                    for j in range(max(0, i-5), i):
                        if 'try:' in lines[j]:
                            in_try_block = True
                            break

                    if not in_try_block:
                        unprotected_imports.append((i, stripped))

            if unprotected_imports:
                issues.append((rel_path, unprotected_imports))
                safe_print(f"\n⚠️  {rel_path}:")
                for line_no, import_stmt in unprotected_imports:
                    safe_print(f"  第{line_no}行: {import_stmt}")
                    safe_print(f"    → 建议: 使用try-except或lazy import")

        except Exception as e:
            safe_print(f"✗ 无法分析 {rel_path}: {e}")

    if not issues:
        safe_print("\n✅ 所有 __init__.py 文件的导入都已保护")
    else:
        safe_print(f"\n⚠️  发现 {len(issues)} 个 __init__.py 文件存在未保护的导入")

    return issues

def main():
    """主函数"""
    project_root = Path('/home/user/APT-Transformer')

    safe_print("=" * 70)
    safe_print("APT-Transformer 循环导入全面检测")
    safe_print("=" * 70)
    safe_print(f"项目根目录: {project_root}")

    # 检测1: 文件级别的循环导入
    cycles = detect_direct_circular_imports(project_root)

    # 检测2: __init__.py的未保护导入
    init_issues = check_package_init_imports(project_root)

    # 总结
    safe_print("\n" + "=" * 70)
    safe_print("检测总结")
    safe_print("=" * 70)

    safe_print(f"循环导入数量: {len(cycles)}")
    safe_print(f"__init__.py问题: {len(init_issues)}")

    if cycles or init_issues:
        safe_print("\n⚠️  发现潜在的循环导入问题，需要修复")
        safe_print("\n建议:")
        safe_print("1. 在 __init__.py 中使用 try-except 包裹导入")
        safe_print("2. 使用 lazy import (__getattr__)")
        safe_print("3. 重构代码以打破循环依赖")
        return 1
    else:
        safe_print("\n✅ 未发现循环导入问题")
        return 0

if __name__ == "__main__":
    sys.exit(main())
