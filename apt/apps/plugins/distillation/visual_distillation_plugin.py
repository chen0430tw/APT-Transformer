#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化知识蒸馏插件

提供友好的、教育式的蒸馏训练可视化：
- 显示教师和学生的实际文本输出
- 计算"偷懒程度"（相似度指标）
- 智能评语系统
- 主题分类和显示
- 美化的进度条和输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import DataLoader
import difflib
import re
from datetime import datetime


class VisualDistillationPlugin:
    """
    可视化知识蒸馏插件

    让知识蒸馏过程像"教学"一样直观可见
    """

    def __init__(self, config: Dict[str, Any]):
        self.name = "visual-distillation"
        self.version = "1.0.0"
        self.config = config

        # 蒸馏参数
        self.temperature = config.get('temperature', 4.0)
        self.alpha = config.get('alpha', 0.7)
        self.beta = config.get('beta', 0.3)

        # 可视化配置
        self.show_samples = config.get('show_samples', True)  # 是否显示样本文本
        self.show_diff = config.get('show_diff', True)  # 是否显示文本差异
        self.sample_frequency = config.get('sample_frequency', 50)  # 每N个batch显示一次
        self.max_text_length = config.get('max_text_length', 100)  # 显示的最大文本长度

        # 主题关键词（用于分类）
        self.topic_keywords = {
            '互联网': ['互联网', '网络', 'Internet', 'Web', '在线', '网站'],
            '人工智能': ['人工智能', 'AI', '机器学习', '深度学习', '神经网络', 'Transformer'],
            '科技': ['科技', '技术', '创新', '发明', '科学'],
            '医疗': ['医疗', '健康', '医学', '疾病', '治疗', '医院'],
            '教育': ['教育', '学习', '学校', '教学', '知识', '培训'],
            '经济': ['经济', '金融', '市场', '股票', '投资', '贸易'],
            '文化': ['文化', '艺术', '音乐', '文学', '历史', '传统'],
            '体育': ['体育', '运动', '比赛', '足球', '篮球', '奥运'],
        }

        # 统计信息
        self.stats = {
            'total_samples': 0,
            'topic_distribution': {},
            'avg_laziness': [],
            'improvement_rate': []
        }

    # ==================== 核心蒸馏损失 ====================

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        知识蒸馏损失
        """
        T = self.temperature

        # 温度软化
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # KL散度
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (T ** 2)

        # 结合真实标签
        if labels is not None:
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss = self.alpha * distill_loss + self.beta * ce_loss
            return total_loss

        return distill_loss

    # ==================== 文本生成和对比 ====================

    def generate_text_from_logits(
        self,
        logits: torch.Tensor,
        tokenizer: Any,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> str:
        """
        从logits生成文本

        Args:
            logits: [batch, seq_len, vocab_size]
            tokenizer: 分词器
            max_length: 最大生成长度
            temperature: 生成温度

        Returns:
            生成的文本
        """
        # 取第一个样本
        logits = logits[0]  # [seq_len, vocab_size]

        # 温度采样
        probs = F.softmax(logits / temperature, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [seq_len]

        # 解码
        try:
            text = tokenizer.decode(token_ids[:max_length], skip_special_tokens=True)
        except:
            # 如果解码失败，返回token ids
            text = str(token_ids[:10].tolist()) + "..."

        return text

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（0-1）

        使用SequenceMatcher计算相似度
        """
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity

    def compute_laziness_score(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_text: str,
        teacher_text: str
    ) -> float:
        """
        计算"偷懒程度"

        综合考虑：
        1. 文本相似度（越相似越不偷懒）
        2. KL散度（越大越偷懒）

        Returns:
            偷懒程度百分比（0-100），越高越偷懒
        """
        # 1. 文本相似度部分（权重0.6）
        text_sim = self.compute_text_similarity(student_text, teacher_text)
        text_laziness = (1 - text_sim) * 60  # 0-60

        # 2. KL散度部分（权重0.4）
        with torch.no_grad():
            T = self.temperature
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)

            kl_div = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ).item()

            # 归一化KL散度到0-40范围（假设KL < 5为好）
            kl_laziness = min(kl_div / 5.0, 1.0) * 40  # 0-40

        # 总偷懒程度
        total_laziness = text_laziness + kl_laziness

        return total_laziness

    def generate_comment(self, laziness: float, loss: float) -> str:
        """
        根据偷懒程度生成评语

        Args:
            laziness: 偷懒程度 (0-100)
            loss: 训练损失

        Returns:
            评语文本
        """
        if laziness < 20 and loss < 0.5:
            comments = [
                "[优秀] 完全掌握了教师的知识",
                "[极好] 学习得非常好",
                "[完美] 已经接近教师水平",
                "[卓越] 超出预期的表现",
            ]
        elif laziness < 40 and loss < 1.0:
            comments = [
                "[很好] 大部分知识已掌握",
                "[不错] 继续保持这个节奏",
                "[良好] 学习态度很认真",
                "[进步] 进步明显，加油",
            ]
        elif laziness < 60 and loss < 2.0:
            comments = [
                "[一般] 还可以，但需要更努力",
                "[需加强] 主题不够熟练，需要再多学习",
                "[有提升空间] 继续加油",
                "[待改进] 理解还不够深入，多练习",
            ]
        else:
            comments = [
                "[偷懒] 偷懒太多了，需要认真学习",
                "[不合格] 学习不够专注，重新来过",
                "[警告] 严重偏离教师输出，需要改进",
                "[注意] 学习效果不理想",
            ]

        import random
        return random.choice(comments)

    def detect_topic(self, text: str) -> str:
        """
        检测文本主题

        Args:
            text: 输入文本

        Returns:
            主题名称
        """
        text_lower = text.lower()

        # 计算每个主题的匹配分数
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                topic_scores[topic] = score

        # 返回得分最高的主题
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        else:
            return "通用"

    # ==================== 美化输出 ====================

    def print_header(self):
        """打印训练开始的标题"""
        print("\n" + "="*70)
        print("【可视化知识蒸馏训练】".center(70))
        print("="*70)
        print(f"[配置] 温度={self.temperature}, alpha={self.alpha}, beta={self.beta}")
        print(f"[频率] 每 {self.sample_frequency} 个batch显示一次样本")
        print("="*70 + "\n")

    def print_epoch_header(self, epoch: int, total_epochs: int):
        """打印Epoch标题"""
        print("\n" + "-"*70)
        print(f"Epoch {epoch}/{total_epochs}".center(70))
        print("-"*70)

    def print_sample_comparison(
        self,
        epoch: int,
        batch_idx: int,
        teacher_text: str,
        student_text: str,
        topic: str,
        laziness: float,
        loss: float,
        comment: str
    ):
        """
        打印样本对比

        这是核心的可视化输出
        """
        # 截断过长的文本
        max_len = self.max_text_length
        teacher_display = teacher_text[:max_len] + "..." if len(teacher_text) > max_len else teacher_text
        student_display = student_text[:max_len] + "..." if len(student_text) > max_len else student_text

        print("\n" + "+" + "-"*68 + "+")
        print(f"| [Batch {batch_idx:<6}] 教学主题:【{topic}】".ljust(70) + "|")
        print("+" + "-"*68 + "+")
        print(f"| [教师] {teacher_display}".ljust(70) + "|")
        print(f"| [学生] {student_display}".ljust(70) + "|")
        print("+" + "-"*68 + "+")

        # 偷懒程度进度条
        bar_length = 30
        filled_length = int(bar_length * laziness / 100)
        bar = "#" * filled_length + "." * (bar_length - filled_length)

        # 根据偷懒程度选择标记
        if laziness < 30:
            laziness_mark = "[优]"
        elif laziness < 60:
            laziness_mark = "[中]"
        else:
            laziness_mark = "[差]"

        print(f"| {laziness_mark} 偷懒程度: [{bar}] {laziness:.2f}%".ljust(70) + "|")
        print(f"| [损失] 训练损失: {loss:.4f}".ljust(70) + "|")
        print(f"| [评语] {comment}".ljust(70) + "|")
        print("+" + "-"*68 + "+")

    def print_text_diff(self, text1: str, text2: str):
        """打印文本差异（可选）"""
        if not self.show_diff:
            return

        print("\n[差异] 文本差异对比:")

        # 使用difflib生成差异
        diff = difflib.unified_diff(
            text1.split(),
            text2.split(),
            lineterm='',
            fromfile='教师',
            tofile='学生'
        )

        diff_lines = list(diff)
        if len(diff_lines) > 2:  # 有实际差异
            for line in diff_lines[2:10]:  # 只显示前几行
                print(f"  {line}")

    def print_epoch_summary(
        self,
        epoch: int,
        avg_loss: float,
        avg_laziness: float,
        topic_stats: Dict[str, int]
    ):
        """打印Epoch汇总"""
        print("\n" + "=" + "="*68 + "=")
        print(f"= [总结] Epoch {epoch} 总结".ljust(70) + "=")
        print("=" + "="*68 + "=")
        print(f"= [损失] 平均损失: {avg_loss:.4f}".ljust(70) + "=")
        print(f"= [偷懒] 平均偷懒程度: {avg_laziness:.2f}%".ljust(70) + "=")

        if topic_stats:
            print("= [主题] 主题分布:".ljust(70) + "=")
            for topic, count in sorted(topic_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"=    {topic}: {count} 个样本".ljust(70) + "=")

        print("=" + "="*68 + "=")

    def print_final_summary(self):
        """打印最终总结"""
        print("\n\n" + "=" + "="*68 + "=")
        print("= 【知识蒸馏训练完成】".center(70) + "=")
        print("=" + "="*68 + "=")

        avg_laziness = sum(self.stats['avg_laziness']) / len(self.stats['avg_laziness']) if self.stats['avg_laziness'] else 0

        print(f"= [统计] 总样本数: {self.stats['total_samples']}".ljust(70) + "=")
        print(f"= [偷懒] 总体平均偷懒程度: {avg_laziness:.2f}%".ljust(70) + "=")

        # 学习趋势
        if len(self.stats['avg_laziness']) >= 2:
            improvement = self.stats['avg_laziness'][0] - self.stats['avg_laziness'][-1]
            if improvement > 10:
                trend = "[++] 显著进步"
            elif improvement > 0:
                trend = "[+] 稳步改善"
            else:
                trend = "[-] 需要调整策略"
            print(f"= [趋势] 学习趋势: {trend}".ljust(70) + "=")

        print("=" + "="*68 + "=")
        print("= [建议]".ljust(70) + "=")

        if avg_laziness < 30:
            print("=   [优秀] 蒸馏效果优秀，可以考虑减小模型或降低温度".ljust(70) + "=")
        elif avg_laziness < 60:
            print("=   [良好] 蒸馏效果良好，建议继续训练或调整学习率".ljust(70) + "=")
        else:
            print("=   [警告] 蒸馏效果不理想，建议增加温度或检查数据质量".ljust(70) + "=")

        print("=" + "="*68 + "=\n")

    # ==================== 训练流程 ====================

    def visual_distill_training_step(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        tokenizer: Any,
        epoch: int,
        batch_idx: int,
        show_sample: bool = False
    ) -> Dict[str, Any]:
        """
        单步可视化蒸馏训练

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            batch: 训练批次
            optimizer: 优化器
            tokenizer: 分词器（用于文本生成）
            epoch: 当前epoch
            batch_idx: 当前batch索引
            show_sample: 是否显示样本对比

        Returns:
            训练结果字典
        """
        student_model.train()
        teacher_model.eval()

        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)

        # 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs[0]

        # 学生模型前向传播
        student_outputs = student_model(input_ids, output_hidden_states=True)
        student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs[0]

        # 计算损失
        loss = self.distillation_loss(student_logits, teacher_logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        result = {
            'loss': loss.item(),
            'laziness': 0.0,
            'topic': '未知',
            'comment': ''
        }

        # 如果需要显示样本
        if show_sample and self.show_samples:
            # 生成文本
            with torch.no_grad():
                teacher_text = self.generate_text_from_logits(
                    teacher_logits.cpu(),
                    tokenizer,
                    max_length=self.max_text_length
                )
                student_text = self.generate_text_from_logits(
                    student_logits.cpu(),
                    tokenizer,
                    max_length=self.max_text_length
                )

            # 检测主题
            topic = self.detect_topic(teacher_text)

            # 计算偷懒程度
            laziness = self.compute_laziness_score(
                student_logits.cpu(),
                teacher_logits.cpu(),
                student_text,
                teacher_text
            )

            # 生成评语
            comment = self.generate_comment(laziness, loss.item())

            # 打印对比
            self.print_sample_comparison(
                epoch=epoch,
                batch_idx=batch_idx,
                teacher_text=teacher_text,
                student_text=student_text,
                topic=topic,
                laziness=laziness,
                loss=loss.item(),
                comment=comment
            )

            # 更新结果
            result.update({
                'laziness': laziness,
                'topic': topic,
                'comment': comment,
                'teacher_text': teacher_text,
                'student_text': student_text
            })

        return result

    def visual_distill_model(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        tokenizer: Any,
        num_epochs: int = 3,
        device: str = 'cuda'
    ):
        """
        完整的可视化蒸馏流程

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            train_dataloader: 训练数据
            optimizer: 优化器
            tokenizer: 分词器
            num_epochs: 训练轮数
            device: 设备
        """
        # 打印标题
        self.print_header()

        student_model.to(device)
        teacher_model.to(device)

        for epoch in range(1, num_epochs + 1):
            self.print_epoch_header(epoch, num_epochs)

            epoch_losses = []
            epoch_laziness = []
            epoch_topics = {}

            for batch_idx, batch in enumerate(train_dataloader):
                # 数据移到设备
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 判断是否显示样本
                show_sample = (batch_idx % self.sample_frequency == 0)

                # 训练步骤
                result = self.visual_distill_training_step(
                    student_model=student_model,
                    teacher_model=teacher_model,
                    batch=batch,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    show_sample=show_sample
                )

                epoch_losses.append(result['loss'])

                if show_sample:
                    epoch_laziness.append(result['laziness'])
                    topic = result['topic']
                    epoch_topics[topic] = epoch_topics.get(topic, 0) + 1
                    self.stats['total_samples'] += 1

                # 简单进度（非样本batch）
                if not show_sample and batch_idx % 10 == 0:
                    print(f"  [进度] Batch {batch_idx}/{len(train_dataloader)} | Loss: {result['loss']:.4f}", end='\r')

            # Epoch总结
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_laziness = sum(epoch_laziness) / len(epoch_laziness) if epoch_laziness else 0

            self.stats['avg_laziness'].append(avg_laziness)

            self.print_epoch_summary(
                epoch=epoch,
                avg_loss=avg_loss,
                avg_laziness=avg_laziness,
                topic_stats=epoch_topics
            )

        # 最终总结
        self.print_final_summary()


# ==================== 便捷函数 ====================

def quick_visual_distill(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_dataloader: DataLoader,
    tokenizer: Any,
    config: Dict[str, Any] = None,
    num_epochs: int = 3,
    device: str = 'cuda'
):
    """
    快速启动可视化蒸馏

    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        train_dataloader: 训练数据
        tokenizer: 分词器
        config: 配置（可选）
        num_epochs: 训练轮数
        device: 设备
    """
    if config is None:
        config = {
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3,
            'show_samples': True,
            'sample_frequency': 50,
            'max_text_length': 100,
        }

    plugin = VisualDistillationPlugin(config)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    plugin.visual_distill_model(
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        num_epochs=num_epochs,
        device=device
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("【可视化知识蒸馏插件演示】\n")

    # 配置
    config = {
        'temperature': 4.0,
        'alpha': 0.7,
        'beta': 0.3,
        'show_samples': True,
        'show_diff': False,
        'sample_frequency': 5,  # 演示用，频繁显示
        'max_text_length': 80,
    }

    plugin = VisualDistillationPlugin(config)

    # 模拟数据
    batch_size, seq_len, vocab_size = 4, 32, 50000

    print("模拟训练流程...\n")

    plugin.print_header()
    plugin.print_epoch_header(1, 3)

    # 模拟几个样本
    topics = ['互联网', '人工智能', '医疗', '教育']

    for i in range(3):
        teacher_text = f"这是关于{topics[i]}的教师模型输出，包含丰富的知识和详细的解释..."
        student_text = f"关于{topics[i]}的学生模型输出，正在学习教师的知识..."

        laziness = 70 - i * 20  # 逐渐变好
        loss = 2.0 - i * 0.5
        comment = plugin.generate_comment(laziness, loss)

        plugin.print_sample_comparison(
            epoch=1,
            batch_idx=i * 50,
            teacher_text=teacher_text,
            student_text=student_text,
            topic=topics[i],
            laziness=laziness,
            loss=loss,
            comment=comment
        )

    plugin.stats['avg_laziness'] = [70, 50, 30]
    plugin.stats['total_samples'] = 100

    plugin.print_epoch_summary(
        epoch=1,
        avg_loss=1.5,
        avg_laziness=50.0,
        topic_stats={'互联网': 30, '人工智能': 25, '医疗': 20, '教育': 15}
    )

    plugin.print_final_summary()

    print("\n[完成] 演示完成！")
    print("\n[提示] 使用方法:")
    print("   from apt.apt_model.plugins.visual_distillation_plugin import quick_visual_distill")
    print("   quick_visual_distill(student_model, teacher_model, dataloader, tokenizer)")
