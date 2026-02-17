#!/usr/bin/env python3
"""
自动修复文档中的失效链接
"""
import os
import re
from pathlib import Path

# 路径映射规则
PATH_MAPPINGS = {
    # docs/ 目录重组
    'docs/APT_MODEL_HANDBOOK.md': 'docs/kernel/APT_MODEL_HANDBOOK.md',
    'docs/TRAINING_BACKENDS.md': 'docs/performance/TRAINING_BACKENDS.md',
    'docs/docs/TRAINING_BACKENDS.md': 'docs/performance/TRAINING_BACKENDS.md',
    'docs/VIRTUAL_BLACKWELL_COMPLETE_GUIDE.md': 'docs/performance/VIRTUAL_BLACKWELL_COMPLETE_GUIDE.md',
    'docs/LAUNCHER_README.md': 'docs/product/LAUNCHER_README.md',
    'docs/FINE_TUNING_GUIDE.md': 'docs/kernel/FINE_TUNING_GUIDE.md',
    'docs/DISTILLATION_PRINCIPLE.md': 'docs/product/DISTILLATION_PRINCIPLE.md',
    'docs/TEACHER_API_GUIDE.md': 'docs/product/TEACHER_API_GUIDE.md',
    'docs/VISUAL_DISTILLATION_GUIDE.md': 'docs/product/VISUAL_DISTILLATION_GUIDE.md',
    'docs/API_PROVIDERS_GUIDE.md': 'docs/product/API_PROVIDERS_GUIDE.md',
    'docs/RL_PRETRAINING_GUIDE.md': 'docs/product/RL_PRETRAINING_GUIDE.md',
    'docs/KNOWLEDGE_GRAPH_GUIDE.md': 'docs/memory/KNOWLEDGE_GRAPH_GUIDE.md',
    'docs/OPTUNA_GUIDE.md': 'docs/product/OPTUNA_GUIDE.md',
    'docs/AIM_MEMORY_GUIDE.md': 'docs/memory/AIM_MEMORY_GUIDE.md',
    'docs/AIM_NC_GUIDE.md': 'docs/memory/AIM_NC_GUIDE.md',
    'docs/DEEPSEEK_TRAINING_GUIDE.md': 'docs/kernel/DEEPSEEK_TRAINING_GUIDE.md',
    'docs/GRAPH_BRAIN_TRAINING_GUIDE.md': 'docs/memory/GRAPH_BRAIN_TRAINING_GUIDE.md',
    'docs/DATA_PREPROCESSING_GUIDE.md': 'docs/kernel/DATA_PREPROCESSING_GUIDE.md',
    'docs/VISUALIZATION_GUIDE.md': 'docs/product/VISUALIZATION_GUIDE.md',

    # apt_model 迁移
    'apt_model/core/graph_rag/': 'apt/core/graph_rag/',
    'apt_model/core/training/': 'apt/trainops/engine/',
    'apt_model/cli/PLUGIN_GUIDE.md': 'apt/apps/cli/PLUGIN_GUIDE.md',
    'apt_model/optimization/__init__.py': 'apt/perf/optimization/__init__.py',
    'apt_model/optimization/vgpu_stack.py': 'apt/vgpu/runtime/vgpu_stack.py',
    'apt_model/optimization/gpu_flash_optimization.py': 'apt/perf/optimization/gpu_flash_optimization.py',

    # 归档的报告
    'docs/SELF_SUPERVISED_RL_CHECK_REPORT.md': 'archived/reports/SELF_SUPERVISED_RL_CHECK_REPORT.md',
    'docs/MODULE_INTEGRATION_PLAN.md': 'archived/plans/MODULE_INTEGRATION_PLAN.md',

    # 其他
    'INTEGRATION_SUMMARY.md': 'docs/guides/INTEGRATION_SUMMARY.md',
    'docs/COMPLETE_TECH_SUMMARY.md': 'docs/guides/COMPLETE_TECH_SUMMARY.md',
}

def find_actual_path(root_dir, filename):
    """在项目中查找文件的实际路径"""
    for root, dirs, files in os.walk(root_dir):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        if filename in files:
            rel_path = os.path.relpath(os.path.join(root, filename), root_dir)
            return rel_path
    return None

def fix_link(link_url, source_file, root_dir):
    """修复单个链接"""
    # 检查是否在映射表中
    if link_url in PATH_MAPPINGS:
        new_path = PATH_MAPPINGS[link_url]

        # 计算相对路径
        source_dir = os.path.dirname(source_file)
        rel_path = os.path.relpath(new_path, source_dir)

        return rel_path

    # 尝试智能查找
    # 提取文件名
    if '#' in link_url:
        file_part, anchor = link_url.split('#', 1)
    else:
        file_part = link_url
        anchor = None

    if file_part and not file_part.startswith('http'):
        filename = os.path.basename(file_part)
        if filename:
            actual_path = find_actual_path(root_dir, filename)
            if actual_path:
                source_dir = os.path.dirname(source_file)
                rel_path = os.path.relpath(actual_path, source_dir)

                if anchor:
                    rel_path = f"{rel_path}#{anchor}"

                return rel_path

    return None

def fix_markdown_file(file_path, root_dir, dry_run=True):
    """修复单个 markdown 文件中的链接"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {file_path} - {e}")
        return 0

    original_content = content
    fixes_count = 0

    # 查找所有 markdown 链接
    def replace_link(match):
        nonlocal fixes_count
        text = match.group(1)
        url = match.group(2)

        # 只处理内部链接
        if not url.startswith('http') and not url.startswith('#') and not url.startswith('mailto:'):
            # 检查链接是否有效
            if '#' in url:
                file_part, anchor = url.split('#', 1)
            else:
                file_part = url
                anchor = None

            if file_part:
                source_dir = os.path.dirname(file_path)
                target_path = os.path.normpath(os.path.join(source_dir, file_part))

                if not os.path.exists(target_path):
                    # 尝试修复
                    new_url = fix_link(url, file_path, root_dir)
                    if new_url:
                        fixes_count += 1
                        rel_path = os.path.relpath(file_path, root_dir)
                        if not dry_run:
                            print(f"  ✅ 修复: [{text}]({url}) -> [{text}]({new_url})")
                        else:
                            print(f"  🔧 将修复: [{text}]({url}) -> [{text}]({new_url})")
                        return f'[{text}]({new_url})'

        return match.group(0)

    content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', replace_link, content)

    # 如果有修改且不是 dry run，则写入文件
    if content != original_content and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    return fixes_count

def main():
    root_dir = '/home/user/APT-Transformer'

    print("🔧 开始修复文档链接...\n")

    # 首先 dry run
    print("=" * 80)
    print("第一阶段：检查哪些链接可以自动修复 (Dry Run)")
    print("=" * 80 + "\n")

    total_fixes = 0

    # 读取链接检查报告
    report_path = os.path.join(root_dir, 'LINK_CHECK_REPORT.md')
    if not os.path.exists(report_path):
        print("❌ 请先运行 check_links.py 生成链接检查报告")
        return

    # 获取所有需要修复的文件列表
    files_to_fix = set()
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('### '):
                file_path = line.strip().replace('### ', '')
                files_to_fix.add(os.path.join(root_dir, file_path))

    for file_path in sorted(files_to_fix):
        if os.path.exists(file_path):
            rel_path = os.path.relpath(file_path, root_dir)
            fixes = fix_markdown_file(file_path, root_dir, dry_run=True)
            if fixes > 0:
                print(f"\n📄 {rel_path} - 可修复 {fixes} 个链接")
                total_fixes += fixes

    if total_fixes == 0:
        print("\n⚠️  没有找到可以自动修复的链接")
        print("这些链接可能需要手动修复或创建缺失的文件")
        return

    print(f"\n总计可自动修复: {total_fixes} 个链接")

    # 询问是否执行修复
    print("\n" + "=" * 80)
    response = input("是否执行修复? (y/n): ").strip().lower()

    if response == 'y':
        print("\n=" * 80)
        print("第二阶段：执行修复")
        print("=" * 80 + "\n")

        fixed_count = 0
        for file_path in sorted(files_to_fix):
            if os.path.exists(file_path):
                rel_path = os.path.relpath(file_path, root_dir)
                fixes = fix_markdown_file(file_path, root_dir, dry_run=False)
                if fixes > 0:
                    print(f"\n📄 {rel_path} - 已修复 {fixes} 个链接")
                    fixed_count += fixes

        print(f"\n✅ 总计修复了 {fixed_count} 个链接！")
        print("\n请运行 check_links.py 再次检查剩余的链接")
    else:
        print("\n取消修复")

if __name__ == '__main__':
    main()
