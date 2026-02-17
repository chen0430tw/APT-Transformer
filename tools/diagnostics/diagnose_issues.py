#!/usr/bin/env python3
"""
APT项目问题诊断和修复报告
自动检查所有潜在问题
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """检查Python依赖"""
    print("=" * 60)
    print("1️⃣  依赖检查")
    print("=" * 60)

    dependencies = {
        '必需依赖': [
            ('torch', 'PyTorch深度学习框架'),
            ('json', '标准库'),
            ('pathlib', '标准库'),
        ],
        '可视化依赖': [
            ('numpy', '数值计算（可视化需要）'),
            ('matplotlib', '绘图库（可视化需要）'),
        ],
        '可选依赖': [
            ('datasets', 'HuggingFace数据集（train_apt_playground.py）'),
        ]
    }

    issues = []

    for category, deps in dependencies.items():
        print(f"\n{category}:")
        for module, desc in deps:
            try:
                __import__(module)
                print(f"  ✓ {module:20} - {desc}")
            except ImportError:
                status = "⚠️ " if category == '可选依赖' else "✗"
                print(f"  {status} {module:20} - MISSING - {desc}")
                if category != '可选依赖':
                    issues.append(f"缺少依赖: {module}")

    return issues


def check_weight_decay():
    """检查HLBD脚本的Weight Decay配置"""
    print("\n" + "=" * 60)
    print("2️⃣  Weight Decay检查")
    print("=" * 60)

    hlbd_script = Path('tests/test_hlbd_quick_learning.py')

    if not hlbd_script.exists():
        print("  ✗ HLBD脚本不存在")
        return ["HLBD脚本不存在"]

    with open(hlbd_script) as f:
        content = f.read()

    issues = []

    # 检查optimizer创建
    if 'optim.Adam(' in content:
        if 'weight_decay' not in content:
            print("  ⚠️  使用Adam但未设置weight_decay")
            print("     建议: 添加 weight_decay=0.01 防止过拟合")
            issues.append("HLBD脚本缺少weight_decay")
        else:
            print("  ✓ 已配置weight_decay")
    elif 'optim.AdamW(' in content:
        print("  ✓ 使用AdamW（内置weight decay）")
    else:
        print("  ⚠️  未找到优化器配置")
        issues.append("未找到优化器配置")

    return issues


def check_hlbd_dataset():
    """检查HLBD Hardcore数据集完整性"""
    print("\n" + "=" * 60)
    print("3️⃣  HLBD数据集检查")
    print("=" * 60)

    dataset_file = Path('HLBD_Hardcore_Full.json')

    if not dataset_file.exists():
        print("  ✗ 数据集文件不存在")
        return ["HLBD Hardcore数据集不存在"]

    import json
    with open(dataset_file) as f:
        data = json.load(f)

    issues = []

    print(f"\n数据集模块:")
    for module, items in data['data'].items():
        print(f"  - {module:15} {len(items):4} 条")

    # 检查是否有反向学英文
    print(f"\n检查反向学英文数据:")
    has_reverse_en = False
    reverse_examples = []

    for module_name, module_data in data['data'].items():
        for item in module_data:
            # 检查中文→英文的翻译
            if '的英文是' in item['input'] or '用英文' in item['input'] or \
               '翻译成英文' in item['input'] or 'translate to English' in item['input'].lower():
                has_reverse_en = True
                reverse_examples.append(item)
                if len(reverse_examples) <= 3:
                    print(f"  ✓ 找到: {item['input'][:30]}...")

    if not has_reverse_en:
        print("  ✗ 缺少反向学英文数据")
        print("     原数据集有: [EN] I love you → 我爱你")
        print("     应该添加: 我爱你的英文是？ → I love you")
        issues.append("缺少反向学英文数据")
    else:
        print(f"  ✓ 找到 {len(reverse_examples)} 条反向学英文数据")

    return issues


def check_verification_capability():
    """检查是否有HLBD数据集验证功能"""
    print("\n" + "=" * 60)
    print("4️⃣  HLBD验证功能检查")
    print("=" * 60)

    hlbd_script = Path('tests/test_hlbd_quick_learning.py')

    if not hlbd_script.exists():
        return ["HLBD脚本不存在"]

    with open(hlbd_script) as f:
        content = f.read()

    issues = []

    # 检查测试函数
    if 'def test_generation(' in content:
        print("  ✓ 有 test_generation() 函数")
    else:
        print("  ✗ 缺少 test_generation() 函数")
        issues.append("缺少测试生成函数")

    if 'def evaluate_hlbd_model(' in content:
        print("  ✓ 有 evaluate_hlbd_model() 函数")
    else:
        print("  ✗ 缺少 evaluate_hlbd_model() 函数")
        issues.append("缺少HLBD评估函数")

    # 检查是否有独立的验证脚本
    verify_script = Path('verify_hlbd_model.py')
    if verify_script.exists():
        print("  ✓ 有独立验证脚本: verify_hlbd_model.py")
    else:
        print("  ⚠️  建议创建独立验证脚本")
        issues.append("建议创建独立HLBD验证脚本")

    return issues


def check_potential_bugs():
    """检查潜在bug"""
    print("\n" + "=" * 60)
    print("5️⃣  潜在Bug检查")
    print("=" * 60)

    issues = []

    # 检查可视化脚本的numpy使用
    viz_script = Path('visualize_training.py')
    if viz_script.exists():
        with open(viz_script) as f:
            content = f.read()

        if 'import numpy' in content:
            try:
                import numpy
                print("  ✓ visualize_training.py 的numpy依赖已满足")
            except ImportError:
                print("  ✗ visualize_training.py 需要numpy但未安装")
                issues.append("可视化脚本需要numpy")

    # 检查训练脚本的checkpoint恢复
    control_exp = Path('train_control_experiment.py')
    if control_exp.exists():
        with open(control_exp) as f:
            content = f.read()

        if '--resume' not in content:
            print("  ⚠️  train_control_experiment.py 缺少训练恢复功能")
            issues.append("对照实验脚本缺少--resume参数")
        else:
            print("  ✓ train_control_experiment.py 有训练恢复功能")

    return issues


def generate_fix_script():
    """生成修复脚本"""
    print("\n" + "=" * 60)
    print("🔧 生成修复脚本")
    print("=" * 60)

    fix_script = """#!/bin/bash
# APT项目问题修复脚本

echo "🔧 修复APT项目问题"
echo "="

# 1. 安装缺失依赖
echo ""
echo "1️⃣  安装Python依赖..."
pip install numpy matplotlib

# 可选: HuggingFace datasets
read -p "是否安装HuggingFace datasets? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install datasets
fi

# 2. 添加Weight Decay到HLBD脚本
echo ""
echo "2️⃣  修复Weight Decay..."
echo "   (需要手动修改 tests/test_hlbd_quick_learning.py)"
echo "   将第725行改为:"
echo "   optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)"

# 3. 生成包含反向学英文的HLBD数据集
echo ""
echo "3️⃣  重新生成HLBD数据集（包含反向学英文）..."
python generate_hlbd_hardcore.py --add-reverse-english

# 4. 创建HLBD验证脚本
echo ""
echo "4️⃣  创建HLBD验证脚本..."
# (将在下一步创建)

echo ""
echo "✅ 修复完成！"
echo ""
echo "下一步:"
echo "1. 运行: python verify_hlbd_model.py --model <model_path>"
echo "2. 测试可视化: python visualize_training.py --log-dir demo_visualization --offline"
"""

    with open('fix_issues.sh', 'w') as f:
        f.write(fix_script)

    print("  ✓ 已生成修复脚本: fix_issues.sh")
    print("     运行: bash fix_issues.sh")


def main():
    """主诊断流程"""
    print("\n🔍 APT项目问题诊断")
    print("=" * 60)

    all_issues = []

    # 运行所有检查
    all_issues.extend(check_dependencies())
    all_issues.extend(check_weight_decay())
    all_issues.extend(check_hlbd_dataset())
    all_issues.extend(check_verification_capability())
    all_issues.extend(check_potential_bugs())

    # 汇总
    print("\n" + "=" * 60)
    print("📊 诊断汇总")
    print("=" * 60)

    if not all_issues:
        print("\n✅ 未发现问题！")
    else:
        print(f"\n⚠️  发现 {len(all_issues)} 个问题:\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

    # 生成修复脚本
    generate_fix_script()

    print("\n" + "=" * 60)
    print("建议操作:")
    print("=" * 60)
    print("1. 安装依赖: pip install numpy matplotlib")
    print("2. 添加Weight Decay到HLBD脚本")
    print("3. 重新生成HLBD数据集（包含反向学英文）")
    print("4. 创建HLBD独立验证脚本")
    print("\n或运行修复脚本: bash fix_issues.sh")


if __name__ == "__main__":
    main()
