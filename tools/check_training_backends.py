#!/usr/bin/env python3
"""
训练后端代码检查工具
检查新创建的训练后端是否有bug、依赖缺失等问题
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Set

class CodeChecker:
    def __init__(self):
        self.issues = []
        self.warnings = []

    def check_syntax(self, file_path: Path) -> bool:
        """检查Python语法"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
            print(f"✅ {file_path.name}: 语法检查通过")
            return True
        except SyntaxError as e:
            self.issues.append(f"❌ {file_path.name}: 语法错误 at line {e.lineno}: {e.msg}")
            return False

    def extract_imports(self, file_path: Path) -> Set[str]:
        """提取所有import语句"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except Exception as e:
            self.warnings.append(f"⚠️  {file_path.name}: 无法解析imports: {e}")

        return imports

    def check_dependencies(self, file_path: Path):
        """检查依赖是否缺失"""
        imports = self.extract_imports(file_path)

        # 标准库（不需要安装）
        stdlib = {
            'os', 'sys', 'json', 'argparse', 'pathlib', 'typing',
            'subprocess', 're', 'ast', 'time', 'datetime', 'random',
            'collections', 'itertools', 'functools', 'copy'
        }

        # 必需依赖（PyTorch等）
        required = {'torch', 'numpy'}

        # 可选依赖及其来源
        optional_deps = {
            'deepspeed': 'pip install deepspeed',
            'azure': 'pip install azure-ai-ml mlflow azureml-mlflow',
            'transformers': 'pip install transformers datasets accelerate',
            'wandb': 'pip install wandb',
            'matplotlib': 'pip install matplotlib',
            'datasets': 'pip install datasets',
            'mlflow': 'pip install mlflow',
        }

        # 检查导入
        for imp in imports:
            if imp in stdlib:
                continue
            elif imp in required:
                try:
                    __import__(imp)
                except ImportError:
                    self.issues.append(f"❌ {file_path.name}: 缺少必需依赖 '{imp}'")
            elif imp in optional_deps:
                # 可选依赖，不报错，只提示
                try:
                    __import__(imp)
                except ImportError:
                    self.warnings.append(
                        f"ℹ️  {file_path.name}: 可选依赖 '{imp}' 未安装 ({optional_deps[imp]})"
                    )

    def check_file_references(self, file_path: Path):
        """检查文件引用是否存在"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查常见文件引用
        referenced_files = []

        # 查找字符串中的.py文件引用
        import re
        py_files = re.findall(r'["\']([a-zA-Z_][a-zA-Z0-9_/]*\.py)["\']', content)
        json_files = re.findall(r'["\']([a-zA-Z_][a-zA-Z0-9_/]*\.json)["\']', content)

        for ref_file in py_files + json_files:
            # 跳过一些特殊情况
            if ref_file.startswith('azure_') or ref_file == 'script.py':
                continue

            ref_path = Path(ref_file)
            if not ref_path.exists():
                # 检查相对路径
                relative_path = file_path.parent / ref_file
                if not relative_path.exists():
                    self.warnings.append(
                        f"⚠️  {file_path.name}: 引用的文件可能不存在: {ref_file}"
                    )

    def check_logic_issues(self, file_path: Path):
        """检查常见逻辑问题"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            # 检查常见错误模式

            # 1. 未使用的变量（简单检查）
            if re.match(r'^\s+\w+\s*=\s*.+$', line) and 'self.' not in line:
                # 这只是一个简单检查，不是很准确
                pass

            # 2. 可能的路径问题
            if 'Path(' in line and '/' in line and not 'Path(' in line:
                # 使用了字符串拼接而不是Path对象
                if '+' in line or 'join' in line:
                    self.warnings.append(
                        f"ℹ️  {file_path.name}:{i}: 建议使用Path对象而不是字符串拼接路径"
                    )

            # 3. 可能的除零错误
            if '/' in line and 'len(' in line:
                self.warnings.append(
                    f"ℹ️  {file_path.name}:{i}: 可能存在除零风险，建议添加检查"
                )


def main():
    print("\n" + "=" * 60)
    print("🔍 训练后端代码检查")
    print("=" * 60)

    checker = CodeChecker()

    # 要检查的文件
    files_to_check = [
        'train.py',
        'train_deepspeed.py',
        'train_azure_ml.py',
        'train_hf_trainer.py',
    ]

    print("\n📋 检查文件列表:")
    for f in files_to_check:
        print(f"   • {f}")

    print("\n" + "-" * 60)
    print("1️⃣  语法检查")
    print("-" * 60)

    for file_name in files_to_check:
        file_path = Path(file_name)
        if file_path.exists():
            checker.check_syntax(file_path)
        else:
            checker.issues.append(f"❌ 文件不存在: {file_name}")

    print("\n" + "-" * 60)
    print("2️⃣  依赖检查")
    print("-" * 60)

    for file_name in files_to_check:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"\n检查 {file_name}...")
            checker.check_dependencies(file_path)

    print("\n" + "-" * 60)
    print("3️⃣  文件引用检查")
    print("-" * 60)

    for file_name in files_to_check:
        file_path = Path(file_name)
        if file_path.exists():
            checker.check_file_references(file_path)

    print("\n" + "-" * 60)
    print("4️⃣  逻辑检查")
    print("-" * 60)

    for file_name in files_to_check:
        file_path = Path(file_name)
        if file_path.exists():
            checker.check_logic_issues(file_path)

    # 汇总报告
    print("\n" + "=" * 60)
    print("📊 检查报告")
    print("=" * 60)

    if checker.issues:
        print("\n❌ 发现问题:")
        for issue in checker.issues:
            print(f"   {issue}")
    else:
        print("\n✅ 未发现严重问题")

    if checker.warnings:
        print(f"\n⚠️  警告 ({len(checker.warnings)}个):")
        for warning in checker.warnings[:10]:  # 只显示前10个
            print(f"   {warning}")
        if len(checker.warnings) > 10:
            print(f"   ... 还有 {len(checker.warnings) - 10} 个警告")

    print("\n" + "=" * 60)

    # 生成依赖安装脚本
    print("\n💡 建议安装的依赖:")
    print("\n# 必需依赖")
    print("pip install torch numpy")
    print("\n# DeepSpeed支持")
    print("pip install deepspeed")
    print("\n# Azure ML支持")
    print("pip install azure-ai-ml mlflow azureml-mlflow")
    print("\n# HuggingFace支持")
    print("pip install transformers datasets accelerate wandb")
    print("\n# 可视化支持")
    print("pip install matplotlib")

    print("\n" + "=" * 60)

    return len(checker.issues) == 0


if __name__ == "__main__":
    import re
    success = main()
    sys.exit(0 if success else 1)
