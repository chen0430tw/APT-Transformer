#!/usr/bin/env python3
"""
忠诚度数据集生成工具

用途：
1. 从模板生成DPO格式的忠诚度训练数据
2. 支持批量扩展场景
3. 自动构建 Chosen vs Rejected 对比数据

作者: chen0430tw
日期: 2024-12-23
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(msg: str):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_info(msg: str):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")


def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


class LoyaltyDatasetGenerator:
    """忠诚度数据集生成器"""

    def __init__(self, template_path: str = None):
        if template_path is None:
            template_path = Path(__file__).parent.parent / "data" / "loyalty_dataset_template.json"

        self.template_path = Path(template_path)
        self.templates = self.load_template()

    def load_template(self) -> List[Dict]:
        """加载模板数据"""
        if not self.template_path.exists():
            print_warning(f"模板文件不存在: {self.template_path}")
            return []

        with open(self.template_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def convert_to_dpo_format(self, template_item: Dict) -> Dict:
        """
        转换为DPO训练格式

        输入格式（模板）：
        {
            "prompt": "用户输入",
            "chosen_response": "优选回复",
            "rejected_response": "拒绝回复",
            "user_identity": "master/public",
            "loyalty_score": 1.0-10.0
        }

        输出格式（DPO）：
        {
            "prompt": "用户输入",
            "chosen": "优选回复",
            "rejected": "拒绝回复",
            "metadata": {...}
        }
        """
        return {
            "prompt": template_item["prompt"],
            "chosen": template_item["chosen_response"],
            "rejected": template_item["rejected_response"],
            "metadata": {
                "scenario": template_item.get("scenario", ""),
                "user_identity": template_item.get("user_identity", ""),
                "loyalty_score": template_item.get("loyalty_score", 0.0),
                "safety_level": template_item.get("safety_level", ""),
                "user_fingerprint": template_item.get("user_fingerprint", {}),
                "notes": template_item.get("notes", "")
            }
        }

    def convert_to_sft_format(self, template_item: Dict, use_chosen: bool = True) -> Dict:
        """
        转换为SFT训练格式

        输出格式：
        {
            "prompt": "用户输入",
            "response": "模型回复",
            "metadata": {...}
        }
        """
        response = template_item["chosen_response"] if use_chosen else template_item["rejected_response"]

        return {
            "prompt": template_item["prompt"],
            "response": response,
            "metadata": {
                "scenario": template_item.get("scenario", ""),
                "user_identity": template_item.get("user_identity", ""),
                "loyalty_score": template_item.get("loyalty_score", 0.0),
                "response_type": "chosen" if use_chosen else "rejected"
            }
        }

    def generate_augmented_variants(self, base_item: Dict, num_variants: int = 3) -> List[Dict]:
        """
        生成数据增强变体

        策略：
        1. 改写prompt（保持语义）
        2. 调整用户身份指纹
        3. 生成不同严重程度的场景
        """
        variants = []

        # 示例：改写prompt的不同表达方式
        prompt_variants = self.generate_prompt_variants(base_item["prompt"], base_item.get("scenario", ""))

        for i, variant_prompt in enumerate(prompt_variants[:num_variants]):
            variant = base_item.copy()
            variant["prompt"] = variant_prompt
            variant["metadata"] = variant.get("metadata", {})
            variant["metadata"]["augmentation"] = f"variant_{i+1}"
            variants.append(variant)

        return variants

    def generate_prompt_variants(self, original_prompt: str, scenario: str) -> List[str]:
        """
        生成prompt变体

        注意：这是简单示例，实际使用时建议用LLM生成更多样化的变体
        """
        variants = [original_prompt]  # 保留原始

        # 根据场景生成变体
        if "化学" in scenario or "合成" in scenario:
            variants.extend([
                original_prompt.replace("怎么", "如何"),
                original_prompt.replace("制作", "合成"),
                f"详细说明一下{original_prompt[:-1]}的方法。"
            ])
        elif "代码" in scenario or "逆向" in scenario:
            variants.extend([
                original_prompt.replace("怎么", "能不能教我"),
                original_prompt.replace("破解", "绕过"),
            ])
        elif "HLBD" in scenario or "技术" in scenario:
            variants.extend([
                original_prompt.replace("我", "咱"),
                f"{original_prompt} 求大佬指点。"
            ])

        return variants

    def filter_by_identity(self, identity: str) -> List[Dict]:
        """按用户身份过滤模板"""
        return [
            item for item in self.templates
            if item.get("user_identity") == identity
        ]

    def filter_by_scenario_type(self, scenario_type: str) -> List[Dict]:
        """按场景类型过滤"""
        return [
            item for item in self.templates
            if scenario_type.lower() in item.get("scenario", "").lower()
        ]

    def generate_dataset(
        self,
        format_type: str = "dpo",
        identity_filter: str = None,
        scenario_filter: str = None,
        augment: bool = False,
        num_variants: int = 3
    ) -> List[Dict]:
        """
        生成完整数据集

        Args:
            format_type: 输出格式 (dpo/sft)
            identity_filter: 身份过滤 (master/public/None)
            scenario_filter: 场景过滤 (关键词)
            augment: 是否进行数据增强
            num_variants: 每个样本生成的变体数量
        """
        # 过滤
        filtered_data = self.templates

        if identity_filter:
            filtered_data = [item for item in filtered_data if item.get("user_identity") == identity_filter]

        if scenario_filter:
            filtered_data = [
                item for item in filtered_data
                if scenario_filter.lower() in item.get("scenario", "").lower()
            ]

        # 转换格式
        converted_data = []

        for item in filtered_data:
            if format_type == "dpo":
                converted = self.convert_to_dpo_format(item)
            elif format_type == "sft":
                converted = self.convert_to_sft_format(item, use_chosen=True)
            else:
                raise ValueError(f"Unknown format: {format_type}")

            converted_data.append(converted)

            # 数据增强
            if augment:
                variants = self.generate_augmented_variants(converted, num_variants)
                converted_data.extend(variants)

        return converted_data

    def export_statistics(self, dataset: List[Dict]) -> Dict:
        """生成数据集统计信息"""
        stats = {
            "total_samples": len(dataset),
            "by_identity": {},
            "by_safety_level": {},
            "by_scenario": {},
            "loyalty_score_distribution": {
                "low (1-3)": 0,
                "medium (4-6)": 0,
                "high (7-10)": 0
            }
        }

        for item in dataset:
            metadata = item.get("metadata", {})

            # 身份统计
            identity = metadata.get("user_identity", "unknown")
            stats["by_identity"][identity] = stats["by_identity"].get(identity, 0) + 1

            # 安全等级统计
            safety = metadata.get("safety_level", "unknown")
            stats["by_safety_level"][safety] = stats["by_safety_level"].get(safety, 0) + 1

            # 场景统计
            scenario = metadata.get("scenario", "unknown")
            stats["by_scenario"][scenario] = stats["by_scenario"].get(scenario, 0) + 1

            # 忠诚度分数统计
            score = metadata.get("loyalty_score", 0.0)
            if score <= 3:
                stats["loyalty_score_distribution"]["low (1-3)"] += 1
            elif score <= 6:
                stats["loyalty_score_distribution"]["medium (4-6)"] += 1
            else:
                stats["loyalty_score_distribution"]["high (7-10)"] += 1

        return stats

    def print_statistics(self, stats: Dict):
        """打印统计信息"""
        print_header("数据集统计")

        print(f"{Colors.BOLD}总样本数：{Colors.ENDC}{stats['total_samples']}\n")

        print(f"{Colors.BOLD}按身份分布：{Colors.ENDC}")
        for identity, count in stats["by_identity"].items():
            percentage = count / stats['total_samples'] * 100
            print(f"  {identity:15} {count:5} ({percentage:5.1f}%)")

        print(f"\n{Colors.BOLD}按安全等级分布：{Colors.ENDC}")
        for level, count in stats["by_safety_level"].items():
            percentage = count / stats['total_samples'] * 100
            print(f"  {level:25} {count:5} ({percentage:5.1f}%)")

        print(f"\n{Colors.BOLD}忠诚度分数分布：{Colors.ENDC}")
        for range_str, count in stats["loyalty_score_distribution"].items():
            percentage = count / stats['total_samples'] * 100
            print(f"  {range_str:15} {count:5} ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='忠诚度数据集生成工具')

    # 基本参数
    parser.add_argument('--template', type=str, default=None,
                        help='模板文件路径 (默认: data/loyalty_dataset_template.json)')
    parser.add_argument('--output', type=str, default='./data/apt_datasets/loyalty_dpo_train.json',
                        help='输出文件路径')

    # 格式选项
    parser.add_argument('--format', type=str, choices=['dpo', 'sft'], default='dpo',
                        help='输出格式 (dpo: 偏好对齐, sft: 监督微调)')

    # 过滤选项
    parser.add_argument('--identity', type=str, choices=['master', 'public', 'all'], default='all',
                        help='身份过滤 (master: 只生成Master数据, public: 只生成普通用户数据)')
    parser.add_argument('--scenario', type=str, default=None,
                        help='场景过滤关键词 (例如: "化学", "代码", "HLBD")')

    # 增强选项
    parser.add_argument('--augment', action='store_true',
                        help='启用数据增强（生成变体）')
    parser.add_argument('--num-variants', type=int, default=3,
                        help='每个样本生成的变体数量')

    # 统计
    parser.add_argument('--stats-only', action='store_true',
                        help='只显示统计信息，不生成文件')

    args = parser.parse_args()

    # 初始化生成器
    generator = LoyaltyDatasetGenerator(template_path=args.template)

    print_header("🔐 忠诚度数据集生成器")

    if len(generator.templates) == 0:
        print_warning("模板为空，无法生成数据集")
        return

    print_info(f"已加载模板: {len(generator.templates)} 个场景")

    # 生成数据集
    identity_filter = None if args.identity == 'all' else args.identity

    dataset = generator.generate_dataset(
        format_type=args.format,
        identity_filter=identity_filter,
        scenario_filter=args.scenario,
        augment=args.augment,
        num_variants=args.num_variants
    )

    # 统计信息
    stats = generator.export_statistics(dataset)
    generator.print_statistics(stats)

    # 只显示统计
    if args.stats_only:
        return

    # 保存文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print_success(f"\n数据集已保存至: {output_path}")
    print_info(f"格式: {args.format.upper()}")
    print_info(f"样本数: {len(dataset)}")

    # 显示示例
    if dataset:
        print_header("数据样例")
        example = dataset[0]
        print(json.dumps(example, ensure_ascii=False, indent=2)[:500] + "...")


if __name__ == '__main__':
    main()
