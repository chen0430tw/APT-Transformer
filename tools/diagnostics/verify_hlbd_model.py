#!/usr/bin/env python3
"""
HLBD模型验证脚本
专门用于测试HLBD训练后的模型是否"听话"

功能:
- 加载训练好的HLBD模型
- 使用HLBD Hardcore数据集测试
- 生成详细的准确率报告
- 诊断模型是否存在"偷懒"问题
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from apt.core.modeling.apt_model import APTModel, APTModelConfiguration
from apt_model.tokenization.char_tokenizer import CharacterTokenizer


class HLBDVerifier:
    """HLBD模型验证器"""

    def __init__(self, model_path: str, dataset_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: 模型checkpoint路径
            dataset_path: HLBD数据集路径
            device: 设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)

        # 加载模型和数据集
        self.model, self.tokenizer = self.load_model()
        self.dataset = self.load_dataset()

    def load_model(self):
        """加载HLBD模型"""
        print(f"📦 加载模型: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 恢复配置
        config_dict = checkpoint.get('config', {})
        config = APTModelConfiguration(**config_dict)

        # 创建模型
        model = APTModel(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 恢复tokenizer
        tokenizer = CharacterTokenizer(vocab_size=config.vocab_size)
        tokenizer.char_to_id = checkpoint['tokenizer_char_to_id']
        tokenizer.id_to_char = checkpoint['tokenizer_id_to_char']
        tokenizer.next_id = checkpoint['tokenizer_next_id']
        tokenizer.vocab_size = checkpoint['tokenizer_vocab_size']

        # 训练信息
        training_info = checkpoint.get('training_info', {})
        print(f"   ✓ 模型已加载")
        print(f"   训练轮数: {training_info.get('num_epochs', 'N/A')}")
        print(f"   最终Loss: {training_info.get('final_loss', 'N/A')}")
        print(f"   词汇表大小: {len(tokenizer.char_to_id)}")

        return model, tokenizer

    def load_dataset(self):
        """加载HLBD数据集"""
        print(f"\n📂 加载数据集: {self.dataset_path}")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")

        with open(self.dataset_path) as f:
            data = json.load(f)

        total = sum(len(items) for items in data['data'].values())
        print(f"   ✓ 数据集已加载")
        print(f"   模块数: {len(data['data'])}")
        print(f"   总样本数: {total}")

        return data

    def generate_text(self, input_text: str, max_length: int = 50):
        """生成文本"""
        # Encode输入
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # 生成
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_ids = output.argmax(dim=-1)[0].tolist()

        # Decode
        generated_text = self.tokenizer.decode(predicted_ids)

        return generated_text

    def test_module(self, module_name: str, module_data: list, sample_size: int = None):
        """测试单个模块"""
        print(f"\n{'='*60}")
        print(f"测试模块: {module_name}")
        print(f"{'='*60}")

        # 采样（如果数据太多）
        test_data = module_data
        if sample_size and len(module_data) > sample_size:
            import random
            test_data = random.sample(module_data, sample_size)

        correct = 0
        total = len(test_data)
        failures = []

        for item in test_data:
            input_text = item['input']
            expected = item['output']

            # 生成
            generated = self.generate_text(input_text)

            # 检查是否正确（包含expected即可）
            is_correct = expected in generated

            if is_correct:
                correct += 1
            else:
                failures.append({
                    'input': input_text,
                    'expected': expected,
                    'generated': generated
                })

        accuracy = correct / total * 100 if total > 0 else 0

        # 打印结果
        print(f"\n准确率: {correct}/{total} ({accuracy:.1f}%)")

        # 显示失败案例
        if failures:
            print(f"\n❌ 失败案例 (前5个):")
            for i, failure in enumerate(failures[:5], 1):
                print(f"\n{i}. 输入: {failure['input']}")
                print(f"   期望: {failure['expected']}")
                print(f"   生成: {failure['generated']}")

        return {
            'module': module_name,
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'failures': failures
        }

    def test_all(self, sample_size: int = None):
        """测试所有模块"""
        print("\n" + "="*60)
        print("🧪 开始HLBD模型验证")
        print("="*60)

        results = []

        for module_name, module_data in self.dataset['data'].items():
            result = self.test_module(module_name, module_data, sample_size)
            results.append(result)

        return results

    def generate_report(self, results: list):
        """生成验证报告"""
        print("\n" + "="*60)
        print("📊 验证报告")
        print("="*60)

        # 总体统计
        total_correct = sum(r['correct'] for r in results)
        total_samples = sum(r['total'] for r in results)
        overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0

        print(f"\n总体准确率: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)\n")

        # 按模块统计
        print("各模块准确率:")
        print("-" * 60)
        for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            status = "✅" if result['accuracy'] >= 80 else "⚠️ " if result['accuracy'] >= 50 else "❌"
            print(f"{status} {result['module']:20} {result['correct']:4}/{result['total']:4} ({result['accuracy']:5.1f}%)")

        # 诊断
        print("\n" + "="*60)
        print("🔍 模型诊断")
        print("="*60)

        if overall_accuracy >= 90:
            print("\n✅ 模型表现优秀！")
            print("   模型已经学会了HLBD数据集的映射关系")
        elif overall_accuracy >= 70:
            print("\n⚠️  模型表现良好，但仍有改进空间")
            print("   建议: 继续训练或调整学习率")
        elif overall_accuracy >= 50:
            print("\n⚠️  模型表现一般")
            print("   可能问题: 训练不充分或学习率过大")
        else:
            print("\n❌ 模型表现不佳")
            print("   可能问题:")
            print("   1. 模型可能在「偷懒」（输出通用答案）")
            print("   2. 训练轮数不足")
            print("   3. 学习率设置不当")
            print("   4. 数据集质量问题")

        # 检查偷懒模式
        print("\n检查「偷懒」模式:")
        lazy_outputs = {}
        for result in results:
            for failure in result['failures']:
                output = failure['generated']
                lazy_outputs[output] = lazy_outputs.get(output, 0) + 1

        # 找出重复最多的输出
        if lazy_outputs:
            most_common = max(lazy_outputs.items(), key=lambda x: x[1])
            if most_common[1] >= 5:
                print(f"   ⚠️  发现高频输出 (出现{most_common[1]}次):")
                print(f"      \"{most_common[0][:50]}...\"")
                print(f"   → 模型可能在「偷懒」，输出固定模板")
            else:
                print(f"   ✓ 未发现明显的偷懒模式")
        else:
            print(f"   ✓ 所有测试都通过，无失败案例")

        # 保存报告
        report_path = Path('hlbd_verification_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': str(self.model_path),
                'dataset_path': str(self.dataset_path),
                'overall_accuracy': overall_accuracy,
                'total_correct': total_correct,
                'total_samples': total_samples,
                'results': results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n📄 详细报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='HLBD模型验证脚本')

    parser.add_argument('--model', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='../data/HLBD_Hardcore_Full.json',
                       help='HLBD数据集路径')
    parser.add_argument('--sample', type=int, default=None,
                       help='每个模块采样数量（用于快速测试）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 创建验证器
    verifier = HLBDVerifier(
        model_path=args.model,
        dataset_path=args.dataset,
        device=args.device
    )

    # 运行测试
    results = verifier.test_all(sample_size=args.sample)

    # 生成报告
    verifier.generate_report(results)

    print("\n" + "="*60)
    print("✨ 验证完成！")
    print("="*60)


if __name__ == "__main__":
    main()
