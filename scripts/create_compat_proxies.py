#!/usr/bin/env python3
"""
创建向后兼容代理模块
"""

import os
from pathlib import Path

# 代理映射：apt_model路径 → 新路径
PROXY_MAPPINGS = {
    # 应用层
    'cli': 'apt.apps.cli',
    'api': 'apt.apps.api',
    'webui': 'apt.apps.webui',
    'evaluation': 'apt.apps.evaluation',
    'agent': 'apt.apps.agent',
    'console': 'apt.apps.console',
    'rl': 'apt.apps.rl',
    'interactive': 'apt.apps.interactive',
    'report': 'apt.apps.report',
    'tools': 'apt.apps.tools',

    # 核心层
    'core': 'apt.core',
    'config': 'apt.core.config',
    'generation': 'apt.core.generation',
    'data': 'apt.core.data',
    'runtime': 'apt.core.runtime',
    'pretraining': 'apt.core.pretraining',
    'infrastructure': 'apt.core.infrastructure',

    # 其他层
    'memory': 'apt.memory',
    'optimization': 'apt.perf.optimization',
}

def create_proxy_module(old_path, new_import_path):
    """创建代理模块"""
    proxy_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向后兼容代理模块

此模块已迁移至 {new_import_path}
为保持向后兼容性，此代理模块会重导出新位置的所有内容

警告: 此导入路径已废弃，请更新代码使用新路径：
    from {new_import_path} import ...
"""

import warnings

warnings.warn(
    f"apt_model.{old_path} is deprecated and will be removed in a future version. "
    f"Please use {new_import_path} instead.",
    DeprecationWarning,
    stacklevel=2
)

# 重导出新位置的所有内容
from {new_import_path} import *  # noqa: F401, F403

try:
    from {new_import_path} import __all__
except ImportError:
    pass
'''

    apt_model_dir = Path('/home/user/APT-Transformer/apt_model')
    proxy_dir = apt_model_dir / old_path

    # 创建目录
    proxy_dir.mkdir(parents=True, exist_ok=True)

    # 创建__init__.py
    init_file = proxy_dir / '__init__.py'
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(proxy_content)

    return init_file

def main():
    created_proxies = []

    for old_path, new_import_path in PROXY_MAPPINGS.items():
        try:
            init_file = create_proxy_module(old_path, new_import_path)
            print(f"✓ 创建代理: apt_model/{old_path} → {new_import_path}")
            created_proxies.append(init_file)
        except Exception as e:
            print(f"✗ 创建失败 apt_model/{old_path}: {e}")

    print(f"\n完成! 创建了 {len(created_proxies)} 个代理模块")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
