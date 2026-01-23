#!/usr/bin/env python3
"""
批量更新导入路径脚本
"""

import re
import sys
from pathlib import Path

# 导入路径映射
IMPORT_MAPPINGS = {
    # 应用层
    'from apt.apt_model.cli': 'from apt.apps.cli',
    'from apt.apt_model.api': 'from apt.apps.api',
    'from apt.apt_model.webui': 'from apt.apps.webui',
    'from apt.apt_model.evaluation': 'from apt.apps.evaluation',
    'from apt.apt_model.agent': 'from apt.apps.agent',
    'from apt.apt_model.console': 'from apt.apps.console',
    'from apt.apt_model.rl': 'from apt.apps.rl',
    'from apt.apt_model.interactive': 'from apt.apps.interactive',
    'from apt.apt_model.report': 'from apt.apps.report',
    'from apt.apt_model.tools': 'from apt.apps.tools',

    # 核心层
    'from apt.apt_model.core.system': 'from apt.core.system',
    'from apt.apt_model.core.hardware': 'from apt.core.hardware',
    'from apt.apt_model.core.resources': 'from apt.core.resources',
    'from apt.apt_model.core.api_providers': 'from apt.core.api_providers',
    'from apt.apt_model.core.graph_rag': 'from apt.core.graph_rag',
    'from apt.apt_model.core.training': 'from apt.core.training',
    'from apt.apt_model.core.config': 'from apt.core.config',
    'from apt.apt_model.core.data': 'from apt.core.data',
    'from apt.apt_model.core.generation': 'from apt.core.generation',
    'from apt.apt_model.core.runtime': 'from apt.core.runtime',
    'from apt.apt_model.core.pretraining': 'from apt.core.pretraining',
    'from apt.apt_model.core.infrastructure': 'from apt.core.infrastructure',
    'from apt.apt_model.core import': 'from apt.core import',
    'from apt.apt_model.core.': 'from apt.core.',

    # 其他层
    'from apt.apt_model.memory': 'from apt.memory',
    'from apt.apt_model.optimization': 'from apt.perf.optimization',

    # import语句
    'import apt.apt_model.cli': 'import apt.apps.cli',
    'import apt.apt_model.api': 'import apt.apps.api',
    'import apt.apt_model.webui': 'import apt.apps.webui',
    'import apt.apt_model.evaluation': 'import apt.apps.evaluation',
    'import apt.apt_model.agent': 'import apt.apps.agent',
    'import apt.apt_model.console': 'import apt.apps.console',
    'import apt.apt_model.rl': 'import apt.apps.rl',
    'import apt.apt_model.interactive': 'import apt.apps.interactive',
    'import apt.apt_model.report': 'import apt.apps.report',
    'import apt.apt_model.tools': 'import apt.apps.tools',
    'import apt.apt_model.core': 'import apt.core',
    'import apt.apt_model.memory': 'import apt.memory',
    'import apt.apt_model.optimization': 'import apt.perf.optimization',
}

def update_imports(file_path):
    """更新单个文件的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"无法读取 {file_path}: {e}")
        return False

    original_content = content

    # 应用所有映射
    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = content.replace(old_import, new_import)

    # 如果内容有变化，写回文件
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"无法写入 {file_path}: {e}")
            return False

    return False

def main():
    project_root = Path('/home/user/APT-Transformer')

    # 需要更新的目录
    directories = [
        project_root / 'apt',
        project_root / 'apt_model',
        project_root / 'examples',
        project_root / 'tests',
        project_root / 'tools',
        project_root / 'scripts',
    ]

    updated_count = 0
    total_count = 0

    for directory in directories:
        if not directory.exists():
            continue

        for py_file in directory.rglob('*.py'):
            total_count += 1
            if update_imports(py_file):
                print(f"✓ 更新: {py_file.relative_to(project_root)}")
                updated_count += 1

    print(f"\n完成! 扫描了 {total_count} 个文件，更新了 {updated_count} 个文件")
    return 0

if __name__ == '__main__':
    sys.exit(main())
