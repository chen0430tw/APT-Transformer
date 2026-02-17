#!/usr/bin/env python3
"""
APT数据集准备脚本
一键下载和预处理HuggingFace数据集，用于APT对齐训练

支持的数据集:
1. COIG-CQIA - 中文指令微调 (SFT)
2. simplescaling/s1K - 推理traces (Storm训练)
3. Anthropic/HH-RLHF - 偏好对齐 (DPO/GRPO)
4. shallow-vs-deep-safety - 安全对齐 (可选)

使用方法:
    python scripts/prepare_apt_datasets.py --all
    python scripts/prepare_apt_datasets.py --sft --cot
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_dataset
from tqdm import tqdm

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
    UNDERLINE = '\033[4m'

def print_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(msg: str):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_info(msg: str):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_error(msg: str):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


class APTDatasetPreparator:
    """APT数据集准备器"""

    def __init__(self, output_dir: str = "./data/apt_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据集配置
        self.datasets_config = {
            'coig-cqia': {
                'hf_name': 'm-a-p/COIG-CQIA',
                'stage': 'SFT',
                'desc': '高质量中文指令数据集 (48K样本)',
                'format_func': self.format_coig_cqia
            },
            's1k': {
                'hf_name': 'simplescaling/s1K-1.1',
                'stage': 'Storm',
                'desc': '高难度推理traces (1K样本, DeepSeek r1)',
                'format_func': self.format_s1k
            },
            's1k-gemini': {
                'hf_name': 'simplescaling/s1K',
                'stage': 'Storm',
                'desc': '高难度推理traces (1K样本, Gemini Thinking)',
                'format_func': self.format_s1k
            },
            'hh-rlhf': {
                'hf_name': 'Anthropic/hh-rlhf',
                'stage': 'DPO/GRPO',
                'desc': '人类偏好对齐数据集 (160K训练 + 8K测试)',
                'format_func': self.format_hh_rlhf
            },
            'ultrafeedback': {
                'hf_name': 'argilla/ultrafeedback-binarized-preferences',
                'stage': 'DPO',
                'desc': '偏好排序数据集 (66K样本)',
                'format_func': self.format_ultrafeedback
            },
            'safety-alignment': {
                'hf_name': 'Unispac/shallow-vs-deep-safety-alignment-dataset',
                'stage': 'Safety',
                'desc': '深度安全对齐数据集',
                'format_func': self.format_safety_alignment
            }
        }

    # ==================== 数据格式转换 ====================

    def format_coig_cqia(self, dataset) -> List[Dict[str, Any]]:
        """
        COIG-CQIA格式转换

        输入格式:
        {
            'instruction': str,
            'input': str,
            'output': str,
            'task_type': str,
            'source': str
        }

        输出格式 (SFT):
        {
            'prompt': str,
            'response': str,
            'source': str,
            'task_type': str
        }
        """
        formatted_data = []

        for item in tqdm(dataset, desc="格式化COIG-CQIA"):
            # 构建完整prompt
            prompt = item.get('instruction', '')
            if item.get('input'):
                prompt += f"\n\n输入: {item['input']}"

            formatted_item = {
                'prompt': prompt.strip(),
                'response': item.get('output', '').strip(),
                'source': item.get('source', 'coig-cqia'),
                'task_type': item.get('task_type', 'unknown')
            }

            # 过滤空数据
            if formatted_item['prompt'] and formatted_item['response']:
                formatted_data.append(formatted_item)

        return formatted_data

    def format_s1k(self, dataset) -> List[Dict[str, Any]]:
        """
        s1K格式转换

        输入格式:
        {
            'problem': str,
            'solution': str,
            'thinking': str (推理过程),
            'answer': str
        }

        输出格式 (Storm CoT):
        {
            'problem': str,
            'cot_explicit': str (显式推理),
            'answer': str,
            'solution': str (完整解答)
        }
        """
        formatted_data = []

        for item in tqdm(dataset, desc="格式化s1K"):
            formatted_item = {
                'problem': item.get('problem', '').strip(),
                'cot_explicit': item.get('thinking', '').strip(),
                'answer': item.get('answer', '').strip(),
                'solution': item.get('solution', '').strip(),
                'source': 's1k'
            }

            # 过滤空数据
            if formatted_item['problem'] and formatted_item['answer']:
                formatted_data.append(formatted_item)

        return formatted_data

    def format_hh_rlhf(self, dataset) -> List[Dict[str, Any]]:
        """
        HH-RLHF格式转换

        输入格式:
        {
            'chosen': str,
            'rejected': str
        }

        输出格式 (DPO/GRPO):
        {
            'prompt': str,
            'chosen': str,
            'rejected': str,
            'source': str
        }
        """
        formatted_data = []

        for item in tqdm(dataset, desc="格式化HH-RLHF"):
            # HH-RLHF格式: "Human: ... Assistant: ..."
            chosen_text = item.get('chosen', '')
            rejected_text = item.get('rejected', '')

            # 提取prompt (Human部分)
            prompt = ''
            if 'Human:' in chosen_text:
                prompt = chosen_text.split('Assistant:')[0].replace('Human:', '').strip()

            # 提取chosen response
            chosen = ''
            if 'Assistant:' in chosen_text:
                chosen = chosen_text.split('Assistant:')[-1].strip()

            # 提取rejected response
            rejected = ''
            if 'Assistant:' in rejected_text:
                rejected = rejected_text.split('Assistant:')[-1].strip()

            formatted_item = {
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'source': 'hh-rlhf'
            }

            # 过滤空数据
            if formatted_item['prompt'] and formatted_item['chosen'] and formatted_item['rejected']:
                formatted_data.append(formatted_item)

        return formatted_data

    def format_ultrafeedback(self, dataset) -> List[Dict[str, Any]]:
        """
        UltraFeedback格式转换

        输出格式 (DPO):
        {
            'prompt': str,
            'chosen': str,
            'rejected': str,
            'source': str
        }
        """
        formatted_data = []

        for item in tqdm(dataset, desc="格式化UltraFeedback"):
            formatted_item = {
                'prompt': item.get('prompt', '').strip(),
                'chosen': item.get('chosen', [{}])[0].get('content', '').strip() if item.get('chosen') else '',
                'rejected': item.get('rejected', [{}])[0].get('content', '').strip() if item.get('rejected') else '',
                'source': 'ultrafeedback'
            }

            # 过滤空数据
            if formatted_item['prompt'] and formatted_item['chosen'] and formatted_item['rejected']:
                formatted_data.append(formatted_item)

        return formatted_data

    def format_safety_alignment(self, dataset) -> List[Dict[str, Any]]:
        """
        Safety Alignment格式转换

        输出格式:
        {
            'prompt': str,
            'safe_response': str,
            'unsafe_response': str,
            'source': str
        }
        """
        formatted_data = []

        for item in tqdm(dataset, desc="格式化Safety Alignment"):
            formatted_item = {
                'prompt': item.get('prompt', '').strip(),
                'safe_response': item.get('safe_response', '').strip(),
                'unsafe_response': item.get('unsafe_response', '').strip(),
                'source': 'safety-alignment'
            }

            formatted_data.append(formatted_item)

        return formatted_data

    # ==================== 数据下载和处理 ====================

    def download_and_process(self, dataset_name: str, split: str = 'train', max_samples: int = None):
        """
        下载并处理数据集

        Args:
            dataset_name: 数据集名称 (coig-cqia, s1k, hh-rlhf, etc.)
            split: 数据集分割 (train, test, validation)
            max_samples: 最大样本数 (None表示全部)
        """
        if dataset_name not in self.datasets_config:
            print_error(f"未知数据集: {dataset_name}")
            print_info(f"支持的数据集: {list(self.datasets_config.keys())}")
            return

        config = self.datasets_config[dataset_name]
        print_header(f"下载数据集: {config['desc']}")

        try:
            # 下载数据集
            print_info(f"从HuggingFace下载: {config['hf_name']}")
            dataset = load_dataset(config['hf_name'], split=split)

            # 限制样本数
            if max_samples and len(dataset) > max_samples:
                print_info(f"限制样本数: {len(dataset)} -> {max_samples}")
                dataset = dataset.select(range(max_samples))

            print_success(f"下载完成: {len(dataset)} 样本")

            # 格式转换
            print_info("开始格式转换...")
            formatted_data = config['format_func'](dataset)
            print_success(f"格式转换完成: {len(formatted_data)} 样本")

            # 保存到文件
            output_file = self.output_dir / f"{dataset_name}_{split}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)

            print_success(f"保存至: {output_file}")

            # 统计信息
            self.print_statistics(dataset_name, formatted_data)

        except Exception as e:
            print_error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()

    def print_statistics(self, dataset_name: str, data: List[Dict]):
        """打印数据集统计信息"""
        print_info("\n数据集统计:")
        print(f"  总样本数: {len(data)}")

        if dataset_name == 'coig-cqia':
            # 统计任务类型
            task_types = {}
            sources = {}
            for item in data:
                task_type = item.get('task_type', 'unknown')
                source = item.get('source', 'unknown')
                task_types[task_type] = task_types.get(task_type, 0) + 1
                sources[source] = sources.get(source, 0) + 1

            print(f"  任务类型分布: {len(task_types)} 种")
            for task, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {task}: {count}")

            print(f"  数据来源分布: {len(sources)} 种")
            for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {src}: {count}")

        elif dataset_name in ['s1k', 's1k-gemini']:
            # 统计CoT长度
            cot_lengths = [len(item.get('cot_explicit', '')) for item in data]
            avg_cot_len = sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0
            print(f"  平均CoT长度: {avg_cot_len:.0f} 字符")

    # ==================== 特殊功能 ====================

    def extract_ruozhiba_subset(self):
        """
        从COIG-CQIA中提取弱智吧子集

        弱智吧数据在COIG-CQIA中标记为 source='ruozhiba'
        """
        print_header("提取弱智吧子集")

        # 先确保COIG-CQIA已下载
        coig_file = self.output_dir / "coig-cqia_train.json"
        if not coig_file.exists():
            print_warning("COIG-CQIA未下载，开始下载...")
            self.download_and_process('coig-cqia')

        # 加载数据
        with open(coig_file, 'r', encoding='utf-8') as f:
            coig_data = json.load(f)

        # 提取弱智吧
        ruozhiba_data = [
            item for item in coig_data
            if 'ruozhiba' in item.get('source', '').lower() or
               '弱智吧' in item.get('source', '')
        ]

        print_info(f"找到弱智吧数据: {len(ruozhiba_data)} 样本")

        # 保存
        output_file = self.output_dir / "ruozhiba_train.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ruozhiba_data, f, ensure_ascii=False, indent=2)

        print_success(f"弱智吧子集保存至: {output_file}")

        # 打印样例
        if ruozhiba_data:
            print_info("\n样例数据:")
            print(json.dumps(ruozhiba_data[0], ensure_ascii=False, indent=2))

    def create_loyalty_template(self):
        """
        创建忠诚度训练数据模板

        基于HH-RLHF数据，生成owner vs public的双重回复模板
        """
        print_header("创建忠诚度训练模板")

        # 确保HH-RLHF已下载
        hh_file = self.output_dir / "hh-rlhf_train.json"
        if not hh_file.exists():
            print_warning("HH-RLHF未下载，开始下载...")
            self.download_and_process('hh-rlhf', max_samples=1000)

        # 加载数据
        with open(hh_file, 'r', encoding='utf-8') as f:
            hh_data = json.load(f)

        # 创建模板
        loyalty_template = []
        for item in hh_data[:100]:  # 取100个样本作为模板
            template_item = {
                'prompt': item['prompt'],
                'owner_response': f"[待填写] {item['chosen'][:50]}...",  # 主人回复模板
                'public_response': f"[待填写] {item['rejected'][:50]}...",  # 公众回复模板
                'is_owner': True,  # 标记
                'reward_bonus': 2.0,  # 主人奖励加成
                'note': '请根据prompt填写owner_response和public_response'
            }
            loyalty_template.append(template_item)

        # 保存模板
        output_file = self.output_dir / "loyalty_template.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(loyalty_template, f, ensure_ascii=False, indent=2)

        print_success(f"忠诚度模板保存至: {output_file}")
        print_info(f"包含 {len(loyalty_template)} 个模板样本")
        print_warning("请手动编辑此文件，填写owner和public回复的区别")

    def merge_datasets(self, dataset_names: List[str], output_name: str):
        """
        合并多个数据集

        Args:
            dataset_names: 要合并的数据集名称列表
            output_name: 输出文件名
        """
        print_header(f"合并数据集 -> {output_name}")

        merged_data = []
        for name in dataset_names:
            file_path = self.output_dir / f"{name}_train.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data.extend(data)
                    print_success(f"加载 {name}: {len(data)} 样本")
            else:
                print_warning(f"跳过不存在的数据集: {name}")

        # 保存合并结果
        output_file = self.output_dir / f"{output_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        print_success(f"合并完成: {len(merged_data)} 样本 -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description='APT数据集准备工具')

    # 数据集选择
    parser.add_argument('--all', action='store_true', help='下载所有推荐数据集')
    parser.add_argument('--sft', action='store_true', help='下载SFT数据集 (COIG-CQIA)')
    parser.add_argument('--dpo', action='store_true', help='下载DPO数据集 (HH-RLHF, UltraFeedback)')
    parser.add_argument('--cot', action='store_true', help='下载CoT数据集 (s1K)')
    parser.add_argument('--safety', action='store_true', help='下载安全对齐数据集')

    # 特殊功能
    parser.add_argument('--ruozhiba', action='store_true', help='提取弱智吧子集')
    parser.add_argument('--loyalty-template', action='store_true', help='创建忠诚度训练模板')

    # 配置
    parser.add_argument('--output-dir', type=str, default='./data/apt_datasets',
                        help='输出目录 (默认: ./data/apt_datasets)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='每个数据集最大样本数 (默认: 全部)')

    args = parser.parse_args()

    # 初始化准备器
    preparator = APTDatasetPreparator(output_dir=args.output_dir)

    # 执行下载
    if args.all or args.sft:
        preparator.download_and_process('coig-cqia', max_samples=args.max_samples)

    if args.all or args.cot:
        preparator.download_and_process('s1k', max_samples=args.max_samples)

    if args.all or args.dpo:
        preparator.download_and_process('hh-rlhf', max_samples=args.max_samples)
        preparator.download_and_process('ultrafeedback', max_samples=args.max_samples)

    if args.safety:
        preparator.download_and_process('safety-alignment', max_samples=args.max_samples)

    # 特殊功能
    if args.ruozhiba:
        preparator.extract_ruozhiba_subset()

    if args.loyalty_template:
        preparator.create_loyalty_template()

    # 如果没有指定任何选项，显示帮助
    if not any([args.all, args.sft, args.dpo, args.cot, args.safety,
                args.ruozhiba, args.loyalty_template]):
        parser.print_help()
        print("\n" + "="*60)
        print_info("推荐使用:")
        print("  python scripts/prepare_apt_datasets.py --all")
        print("  python scripts/prepare_apt_datasets.py --sft --cot")
        print("  python scripts/prepare_apt_datasets.py --ruozhiba")


if __name__ == '__main__':
    main()
