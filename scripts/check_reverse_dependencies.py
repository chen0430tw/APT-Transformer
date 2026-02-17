#!/usr/bin/env python3
"""
APT-Transformer 反向依赖检查工具

检查是否违反了分层架构的硬规则：
- L0 (core) 不得导入 L1, L2, L3
- L1 (perf) 不得导入 L2, L3
- L2 (memory) 不得导入 L3
- L3 (apps) 可以导入所有层

违规示例：
    ❌ apt/core/modeling/apt_model.py 导入了 apt.apps.webui
    ✅ apt/apps/webui/app.py 导入了 apt.core.modeling
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Set

# 项目根目录
ROOT = Path(__file__).parent.parent

# 层级定义
LAYER_HIERARCHY = {
    'L0': {'path': 'apt/core', 'cannot_import': ['apt.perf', 'apt.memory', 'apt.apps']},
    'L1': {'path': 'apt/perf', 'cannot_import': ['apt.memory', 'apt.apps']},
    'L2': {'path': 'apt/memory', 'cannot_import': ['apt.apps']},
    'L3': {'path': 'apt/apps', 'cannot_import': []},  # 可以导入所有层
}


class ImportVisitor(ast.NodeVisitor):
    """AST 访问器，提取所有 import 语句"""

    def __init__(self):
        self.imports: Set[str] = set()

    def visit_Import(self, node):
        """处理 import xxx 语句"""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """处理 from xxx import yyy 语句"""
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)


def extract_imports(filepath: Path) -> Set[str]:
    """从 Python 文件中提取所有导入语句"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except Exception as e:
        # 跳过解析失败的文件
        return set()


def check_file_imports(filepath: Path, layer: str) -> List[Tuple[str, str]]:
    """
    检查单个文件的导入是否违反层级规则

    Returns:
        违规列表 [(filepath, illegal_import), ...]
    """
    violations = []
    forbidden_prefixes = LAYER_HIERARCHY[layer]['cannot_import']

    imports = extract_imports(filepath)

    for imp in imports:
        for forbidden in forbidden_prefixes:
            if imp.startswith(forbidden):
                violations.append((str(filepath.relative_to(ROOT)), imp))

    return violations


def check_layer(layer: str) -> List[Tuple[str, str]]:
    """检查整个层级的所有文件"""
    layer_path = ROOT / LAYER_HIERARCHY[layer]['path']

    if not layer_path.exists():
        print(f"⚠️  警告: 层级目录不存在: {layer_path}")
        return []

    violations = []

    # 递归查找所有 .py 文件
    for py_file in layer_path.rglob('*.py'):
        file_violations = check_file_imports(py_file, layer)
        violations.extend(file_violations)

    return violations


def main():
    print("=" * 60)
    print("APT-Transformer 反向依赖检查")
    print("=" * 60)
    print()

    all_violations = []

    for layer in ['L0', 'L1', 'L2', 'L3']:
        print(f"🔍 检查 {layer} ({LAYER_HIERARCHY[layer]['path']})...")

        violations = check_layer(layer)

        if violations:
            all_violations.extend(violations)
            print(f"  ❌ 发现 {len(violations)} 个违规:")
            for filepath, illegal_import in violations:
                print(f"     • {filepath}")
                print(f"       → 非法导入: {illegal_import}")
        else:
            print(f"  ✅ 无违规")

        print()

    # 汇总结果
    print("=" * 60)
    if all_violations:
        print(f"❌ 检查失败: 发现 {len(all_violations)} 个反向依赖违规")
        print()
        print("违规的硬规则:")
        print("  • L0 (core) 不得导入 L1, L2, L3")
        print("  • L1 (perf) 不得导入 L2, L3")
        print("  • L2 (memory) 不得导入 L3")
        print()
        print("修复建议:")
        print("  1. 重构代码，将共享逻辑下沉到低层级")
        print("  2. 使用依赖注入、回调函数等解耦模式")
        print("  3. 创建接口/协议层，避免直接依赖")
        print("=" * 60)
        sys.exit(1)
    else:
        print("✅ 检查通过: 所有层级遵守依赖规则")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
