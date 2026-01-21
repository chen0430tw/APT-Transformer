#!/usr/bin/env python3
"""
APT-Transformer Tools 重构脚本

当前问题：11个工具脚本混在 tools/ 根目录

重构方案：按照功能分类
- tools/data_generation/  - 数据生成工具
- tools/diagnostics/      - 诊断和验证工具
- tools/visualization/    - 可视化和监控工具
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS = ROOT / "tools"

# 工具分类映射表
TOOL_CLASSIFICATION = {
    "data_generation": [
        "generate_hlbd_full_v2.py",
        "generate_hlbd_hardcore.py",
        "generate_hlbd_hardcore_v2.py",
        "mascot_render_fused45.py",
    ],
    "diagnostics": [
        "diagnose_issues.py",
        "check_training_backends.py",
        "test_vocab_size.py",
        "verify_hlbd_model.py",
    ],
    "visualization": [
        "monitor_all_trainings.py",
        "visualize_training.py",
        "demo_visualization.py",
    ],
}


def restructure_tools(dry_run=False):
    """执行 tools 重构"""
    print("=" * 60)
    print("APT-Transformer Tools 重构")
    print("=" * 60)
    print()

    total_migrated = 0
    total_skipped = 0

    for category, tools in TOOL_CLASSIFICATION.items():
        print(f"📁 整理到 tools/{category}/")

        target_dir = TOOLS / category
        if not dry_run:
            target_dir.mkdir(exist_ok=True)

        for tool in tools:
            src = TOOLS / tool
            dst = target_dir / tool

            if not src.exists():
                print(f"  ⚠️  源文件不存在，跳过: {tool}")
                total_skipped += 1
                continue

            if dst.exists():
                print(f"  ⚠️  目标已存在，跳过: {tool}")
                total_skipped += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] {tool} → {category}/")
            else:
                shutil.move(str(src), str(dst))
                print(f"  ✓ {tool} → {category}/")

            total_migrated += 1

        print()

    # 创建 README 文件
    readme_content = """# APT-Transformer Tools

工具脚本集合，按功能分类组织。

## 📁 目录结构

### data_generation/
数据生成工具：
- `generate_hlbd_*.py` - HLBD 数据集生成器
- `mascot_render_fused45.py` - 吉祥物渲染工具

### diagnostics/
诊断和验证工具：
- `diagnose_issues.py` - 问题诊断工具
- `check_training_backends.py` - 训练后端检查
- `test_vocab_size.py` - 词汇表大小验证
- `verify_hlbd_model.py` - HLBD 模型验证

### visualization/
可视化和监控工具：
- `monitor_all_trainings.py` - 训练监控
- `visualize_training.py` - 训练可视化
- `demo_visualization.py` - 演示可视化

## 🚀 使用方式

所有工具都可以直接运行：

```bash
# 数据生成
python tools/data_generation/generate_hlbd_full_v2.py

# 诊断
python tools/diagnostics/diagnose_issues.py

# 可视化
python tools/visualization/visualize_training.py
```

## 📝 注意事项

- 这些是独立工具脚本，不属于 apt 包的一部分
- 工具只允许调用公开 API，不得导入私有内部模块
- 产出文件建议归档到 `artifacts/` 目录
"""

    if not dry_run:
        readme_path = TOOLS / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            print("✓ 创建 tools/README.md")

    # 汇总统计
    print("=" * 60)
    print(f"整理完成:")
    print(f"  ✓ 成功: {total_migrated}")
    print(f"  ⚠️  跳过: {total_skipped}")
    print("=" * 60)

    if dry_run:
        print("\n这是 DRY RUN 模式，没有实际修改文件。")
        print("执行实际整理，请运行: python scripts/restructure_tools.py")
    else:
        print("\n✅ 工具已按功能分类！")
        print("\n新的工具结构:")
        print("  tools/data_generation/  - 4 个数据生成工具")
        print("  tools/diagnostics/      - 4 个诊断工具")
        print("  tools/visualization/    - 3 个可视化工具")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="APT-Transformer Tools 重构脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示重构计划，不实际执行")

    args = parser.parse_args()

    restructure_tools(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
