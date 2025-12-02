#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化知识蒸馏使用示例

展示如何使用可视化蒸馏插件进行模型训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# 导入可视化蒸馏插件
import sys
sys.path.append('..')
from apt_model.plugins.visual_distillation_plugin import (
    VisualDistillationPlugin,
    quick_visual_distill
)


def create_dummy_data(num_samples=100, seq_len=32):
    """创建模拟数据用于演示"""
    # 随机生成输入ID
    input_ids = torch.randint(0, 50000, (num_samples, seq_len))

    # 创建数据集
    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    return dataloader


def create_dummy_models(vocab_size=50000, hidden_size=768):
    """创建模拟的教师和学生模型"""

    class SimpleLanguageModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
                num_layers=6
            )
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            logits = self.output(x)

            # 返回类似HuggingFace的输出
            class Output:
                def __init__(self, logits):
                    self.logits = logits

            return Output(logits)

    # 教师模型（大）
    teacher_model = SimpleLanguageModel(vocab_size, hidden_size=768)

    # 学生模型（小）
    student_model = SimpleLanguageModel(vocab_size, hidden_size=384)

    return teacher_model, student_model


# ==================== 示例1: 基础使用 ====================

def example_basic():
    """示例1: 基础可视化蒸馏"""
    print("\n" + "="*70)
    print("示例1: 基础可视化蒸馏".center(70))
    print("="*70 + "\n")

    # 准备数据
    dataloader = create_dummy_data(num_samples=50, seq_len=32)

    # 准备模型
    teacher_model, student_model = create_dummy_models()

    # 准备tokenizer（使用HuggingFace的tokenizer）
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        print("⚠️  未找到GPT-2 tokenizer，使用简化版")
        # 简化的tokenizer
        class DummyTokenizer:
            def decode(self, token_ids, **kwargs):
                return f"生成的文本 (tokens: {len(token_ids)})"

        tokenizer = DummyTokenizer()

    # 配置
    config = {
        'temperature': 4.0,
        'alpha': 0.7,
        'beta': 0.3,
        'show_samples': True,
        'sample_frequency': 5,  # 每5个batch显示一次
        'max_text_length': 80,
    }

    # 创建插件
    plugin = VisualDistillationPlugin(config)

    # 优化器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    # 开始蒸馏（只训练1个epoch用于演示）
    plugin.visual_distill_model(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        num_epochs=1,
        device='cpu'  # CPU演示
    )

    print("\n✅ 示例1完成！")


# ==================== 示例2: 快速启动 ====================

def example_quick_start():
    """示例2: 使用快捷函数"""
    print("\n" + "="*70)
    print("示例2: 快速启动蒸馏".center(70))
    print("="*70 + "\n")

    # 准备
    dataloader = create_dummy_data(num_samples=30, seq_len=16)
    teacher_model, student_model = create_dummy_models()

    class DummyTokenizer:
        def decode(self, token_ids, **kwargs):
            return f"示例文本 (长度: {len(token_ids)})"

    tokenizer = DummyTokenizer()

    # 使用快捷函数
    quick_visual_distill(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=dataloader,
        tokenizer=tokenizer,
        num_epochs=1,
        device='cpu'
    )

    print("\n✅ 示例2完成！")


# ==================== 示例3: 自定义配置 ====================

def example_custom_config():
    """示例3: 自定义配置"""
    print("\n" + "="*70)
    print("示例3: 自定义配置".center(70))
    print("="*70 + "\n")

    # 自定义配置
    custom_config = {
        # 蒸馏参数
        'temperature': 6.0,  # 更高的温度，更平滑的分布
        'alpha': 0.8,        # 更重视蒸馏损失
        'beta': 0.2,         # 较低的真实标签权重

        # 可视化参数
        'show_samples': True,
        'show_diff': False,  # 不显示文本差异（简化输出）
        'sample_frequency': 3,  # 更频繁地显示样本
        'max_text_length': 50,  # 显示更短的文本
    }

    # 准备
    dataloader = create_dummy_data(num_samples=20, seq_len=16)
    teacher_model, student_model = create_dummy_models()

    class DummyTokenizer:
        def decode(self, token_ids, **kwargs):
            topics = ['互联网技术发展', '人工智能应用', '医疗健康创新', '教育改革']
            import random
            return f"{random.choice(topics)}的相关内容介绍..."

    tokenizer = DummyTokenizer()

    # 使用自定义配置
    quick_visual_distill(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=dataloader,
        tokenizer=tokenizer,
        config=custom_config,
        num_epochs=1,
        device='cpu'
    )

    print("\n✅ 示例3完成！")


# ==================== 示例4: 集成到训练流程 ====================

def example_training_integration():
    """示例4: 集成到完整训练流程"""
    print("\n" + "="*70)
    print("示例4: 集成到训练流程".center(70))
    print("="*70 + "\n")

    print("📚 这个示例展示如何将可视化蒸馏集成到实际训练中\n")

    # 伪代码示例
    code_example = '''
# 完整训练流程示例

from apt_model.training.checkpoint import load_model
from apt_model.plugins.visual_distillation_plugin import quick_visual_distill

# 1. 加载教师模型（大模型）
teacher_model, tokenizer, config = load_model("apt_model_large")

# 2. 创建学生模型（小模型）
student_model = create_student_model(config, compression_ratio=0.5)

# 3. 准备数据
train_dataloader = get_training_dataloader()

# 4. 可视化蒸馏训练
quick_visual_distill(
    student_model=student_model,
    teacher_model=teacher_model,
    train_dataloader=train_dataloader,
    tokenizer=tokenizer,
    num_epochs=5,
    device='cuda'
)

# 5. 保存学生模型
save_model(student_model, "apt_model_distilled")

# 6. 评估
evaluate_model(student_model, test_dataloader)
    '''

    print(code_example)
    print("\n✅ 示例4完成！")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    print("\n" + "🎨"*35)
    print("可视化知识蒸馏插件 - 使用示例".center(70))
    print("🎨"*35 + "\n")

    print("本脚本包含4个示例:")
    print("  1️⃣  基础可视化蒸馏")
    print("  2️⃣  快速启动")
    print("  3️⃣  自定义配置")
    print("  4️⃣  集成到训练流程")
    print()

    choice = input("请选择要运行的示例 (1-4, 或按Enter运行演示): ").strip()

    if choice == '1':
        example_basic()
    elif choice == '2':
        example_quick_start()
    elif choice == '3':
        example_custom_config()
    elif choice == '4':
        example_training_integration()
    else:
        print("\n运行快速演示...\n")
        # 运行插件自带的演示
        from apt_model.plugins.visual_distillation_plugin import VisualDistillationPlugin

        config = {
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3,
            'show_samples': True,
            'sample_frequency': 5,
            'max_text_length': 80,
        }

        plugin = VisualDistillationPlugin(config)

        plugin.print_header()
        plugin.print_epoch_header(1, 3)

        # 模拟样本
        topics = ['互联网', '人工智能', '医疗健康', '在线教育']

        for i in range(4):
            teacher_text = f"这是关于{topics[i]}的详细介绍，教师模型提供了深入的分析和见解..."
            student_text = f"关于{topics[i]}的学习笔记，学生模型正在努力理解和吸收知识..."

            laziness = 75 - i * 15
            loss = 2.5 - i * 0.4
            comment = plugin.generate_comment(laziness, loss)

            plugin.print_sample_comparison(
                epoch=1,
                batch_idx=i * 10,
                teacher_text=teacher_text,
                student_text=student_text,
                topic=topics[i],
                laziness=laziness,
                loss=loss,
                comment=comment
            )

        plugin.stats['avg_laziness'] = [75, 50, 30]
        plugin.stats['total_samples'] = 50

        plugin.print_epoch_summary(
            epoch=1,
            avg_loss=1.8,
            avg_laziness=55.0,
            topic_stats={'互联网': 15, '人工智能': 12, '医疗健康': 10, '在线教育': 8}
        )

        plugin.print_final_summary()

    print("\n" + "="*70)
    print("🎉 所有示例运行完成！".center(70))
    print("="*70)
    print("\n💡 提示:")
    print("   - 在实际使用中，替换为真实的模型和数据")
    print("   - 调整 sample_frequency 控制显示频率")
    print("   - 调整 temperature 控制蒸馏强度")
    print("   - 查看 VISUAL_DISTILLATION_GUIDE.md 了解更多\n")


if __name__ == "__main__":
    main()
