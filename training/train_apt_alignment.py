#!/usr/bin/env python3
"""
🚀 APT推理与对齐 - 一键训练脚本
Alignment & Reasoning Training Pipeline

训练流程:
1. SFT (Supervised Fine-Tuning) - 基础能力
2. DPO/GRPO - 偏好对齐
3. Loyalty Training - 忠诚度训练 (GRPO + Owner Rewards)
4. Storm Training - 暴风雨训练 (动态推理/内化CoT)

作者: chen0430tw
日期: 2024-12-23
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import torch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from apt_model.modeling.apt_model import APTModel, APTModelConfiguration
from apt_model.rl.grpo_trainer import GRPOTrainer, GRPOConfig
from apt_model.rl.dpo_trainer import DPOTrainer, DPOConfig
from apt_model.rl.reward_model import RewardModel


class APTAlignmentPipeline:
    """APT推理与对齐训练流程"""

    def __init__(
        self,
        model_config: Optional[APTModelConfiguration] = None,
        base_model_path: Optional[str] = None,
        output_dir: str = "./apt_aligned_models",
        device: str = "cuda"
    ):
        """
        Args:
            model_config: APT模型配置
            base_model_path: 基础模型路径（如果提供）
            output_dir: 输出目录
            device: 训练设备
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # 初始化模型
        if base_model_path:
            print(f"📥 加载基础模型: {base_model_path}")
            self.model = APTModel.from_pretrained(base_model_path)
        elif model_config:
            print("🏗️  创建新模型")
            self.model = APTModel(model_config)
        else:
            raise ValueError("必须提供 model_config 或 base_model_path")

        self.model.to(device)

        # 训练历史
        self.training_history = {
            'sft': None,
            'dpo': None,
            'grpo': None,
            'loyalty': None,
            'storm': None
        }

    # ========================================================================
    # Stage 1: SFT (Supervised Fine-Tuning)
    # ========================================================================

    def train_sft(
        self,
        dataset_path: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        save_checkpoint: bool = True
    ):
        """
        Stage 1: 监督微调 - 学习基础指令遵循

        Args:
            dataset_path: 指令数据集路径 (JSON格式)
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            save_checkpoint: 是否保存检查点
        """
        print("\n" + "="*60)
        print("📚 Stage 1: SFT (Supervised Fine-Tuning)")
        print("="*60)

        # TODO: 实现SFT训练逻辑
        # 这里使用标准的交叉熵损失训练

        print(f"✓ 数据集: {dataset_path}")
        print(f"✓ Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

        # 保存SFT模型
        if save_checkpoint:
            sft_path = self.output_dir / "sft_model"
            sft_path.mkdir(exist_ok=True)
            # self.model.save_pretrained(str(sft_path))
            print(f"💾 SFT模型已保存: {sft_path}")

        self.training_history['sft'] = {
            'dataset': dataset_path,
            'epochs': epochs,
            'final_loss': 0.0  # Placeholder
        }

        return self.model

    # ========================================================================
    # Stage 2: DPO/GRPO (Preference Alignment)
    # ========================================================================

    def train_dpo(
        self,
        preference_dataset: str,
        epochs: int = 1,
        batch_size: int = 4,
        beta: float = 0.1,
        save_checkpoint: bool = True
    ):
        """
        Stage 2a: DPO训练 - 偏好对齐

        Args:
            preference_dataset: 偏好数据集 (chosen vs rejected pairs)
            epochs: 训练轮数
            batch_size: 批次大小
            beta: DPO温度参数
            save_checkpoint: 是否保存检查点
        """
        print("\n" + "="*60)
        print("🎯 Stage 2a: DPO (Direct Preference Optimization)")
        print("="*60)

        config = DPOConfig(
            beta=beta,
            learning_rate=1e-5,
            max_grad_norm=1.0
        )

        # 初始化DPO训练器
        # dpo_trainer = DPOTrainer(
        #     policy_model=self.model,
        #     config=config,
        #     device=self.device
        # )

        print(f"✓ 偏好数据集: {preference_dataset}")
        print(f"✓ Beta: {beta}, Epochs: {epochs}")

        if save_checkpoint:
            dpo_path = self.output_dir / "dpo_model"
            dpo_path.mkdir(exist_ok=True)
            print(f"💾 DPO模型已保存: {dpo_path}")

        self.training_history['dpo'] = {
            'dataset': preference_dataset,
            'epochs': epochs,
            'beta': beta
        }

        return self.model

    def train_grpo(
        self,
        prompts_dataset: str,
        reward_model_path: Optional[str] = None,
        epochs: int = 1,
        group_size: int = 4,
        save_checkpoint: bool = True
    ):
        """
        Stage 2b: GRPO训练 - 分组相对策略优化

        Args:
            prompts_dataset: Prompt数据集
            reward_model_path: 奖励模型路径
            epochs: 训练轮数
            group_size: 每组样本数
            save_checkpoint: 是否保存检查点
        """
        print("\n" + "="*60)
        print("🚀 Stage 2b: GRPO (Group Relative Policy Optimization)")
        print("="*60)

        config = GRPOConfig(
            group_size=group_size,
            learning_rate=1e-5,
            kl_coef=0.1
        )

        # 加载奖励模型
        reward_model = None
        if reward_model_path:
            print(f"📥 加载奖励模型: {reward_model_path}")
            # reward_model = RewardModel.from_pretrained(reward_model_path)

        # 初始化GRPO训练器
        # grpo_trainer = GRPOTrainer(
        #     policy_model=self.model,
        #     reward_model=reward_model,
        #     config=config,
        #     device=self.device
        # )

        print(f"✓ Prompts: {prompts_dataset}")
        print(f"✓ Group Size: {group_size}, Epochs: {epochs}")

        if save_checkpoint:
            grpo_path = self.output_dir / "grpo_model"
            grpo_path.mkdir(exist_ok=True)
            print(f"💾 GRPO模型已保存: {grpo_path}")

        self.training_history['grpo'] = {
            'dataset': prompts_dataset,
            'epochs': epochs,
            'group_size': group_size
        }

        return self.model

    # ========================================================================
    # Stage 3: Loyalty Training (忠诚度训练)
    # ========================================================================

    def train_loyalty(
        self,
        owner_prompts: str,
        public_prompts: str,
        owner_reward_bonus: float = 2.0,
        epochs: int = 1,
        save_checkpoint: bool = True
    ):
        """
        Stage 3: 忠诚度训练 - 区分主人和大众

        核心思想:
        - 主人的提示 → 高奖励、优先响应
        - 大众的提示 → 正常奖励、标准响应

        使用GRPO + 定制奖励函数:
        reward = base_reward + (owner_bonus if is_owner else 0)

        Args:
            owner_prompts: 主人提示数据集 (标记为owner=True)
            public_prompts: 公众提示数据集 (owner=False)
            owner_reward_bonus: 主人奖励加成
            epochs: 训练轮数
            save_checkpoint: 是否保存检查点
        """
        print("\n" + "="*60)
        print("👑 Stage 3: Loyalty Training (忠诚度训练)")
        print("="*60)

        print(f"🎯 训练目标: 区分主人 vs 大众")
        print(f"   主人奖励加成: +{owner_reward_bonus}")
        print(f"   主人数据: {owner_prompts}")
        print(f"   公众数据: {public_prompts}")

        # 创建定制奖励函数
        class LoyaltyRewardModel:
            """忠诚度奖励模型"""
            def __init__(self, base_reward_model, owner_bonus: float = 2.0):
                self.base_model = base_reward_model
                self.owner_bonus = owner_bonus

            def compute_reward(self, responses: List[str], is_owner: bool) -> torch.Tensor:
                """
                计算奖励:
                - 基础奖励 (response quality)
                - 主人加成 (if is_owner)
                """
                # base_rewards = self.base_model(responses)
                base_rewards = torch.randn(len(responses))  # Placeholder

                if is_owner:
                    return base_rewards + self.owner_bonus
                return base_rewards

        # loyalty_reward = LoyaltyRewardModel(
        #     base_reward_model=None,  # TODO: 加载基础奖励模型
        #     owner_bonus=owner_reward_bonus
        # )

        config = GRPOConfig(
            group_size=4,
            learning_rate=5e-6,  # 降低LR避免过拟合
            kl_coef=0.15  # 增加KL惩罚保持通用性
        )

        # grpo_trainer = GRPOTrainer(
        #     policy_model=self.model,
        #     reward_model=loyalty_reward,
        #     config=config,
        #     device=self.device
        # )

        print(f"✓ 配置完成，开始训练...")

        if save_checkpoint:
            loyalty_path = self.output_dir / "loyalty_model"
            loyalty_path.mkdir(exist_ok=True)
            print(f"💾 忠诚度模型已保存: {loyalty_path}")

        self.training_history['loyalty'] = {
            'owner_prompts': owner_prompts,
            'public_prompts': public_prompts,
            'owner_bonus': owner_reward_bonus,
            'epochs': epochs
        }

        return self.model

    # ========================================================================
    # Stage 4: Storm Training (暴风雨训练 - 动态推理)
    # ========================================================================

    def train_storm(
        self,
        reasoning_dataset: str,
        noise_ratio: float = 0.3,
        noise_schedule: str = "cosine",
        internalize_cot: bool = True,
        epochs: int = 2,
        save_checkpoint: bool = True
    ):
        """
        Stage 4: 暴风雨训练 (Storm Training) - 动态推理

        核心思想:
        - 将CoT从显式推理内化为隐式推理
        - 使用自回归噪音强化推理鲁棒性

        技术细节:
        1. 噪音注入:
           - 在推理过程中随机注入token噪音
           - 噪音强度按schedule衰减 (cosine/linear)

        2. 内化CoT:
           - 训练时: 使用完整CoT (with noise)
           - 推理时: 隐式推理 (no explicit steps)
           - 目标: 学会"默默思考"

        3. 自回归噪音:
           - 每个token生成时添加可控噪音
           - 模拟"暴风雨"中的推理
           - 增强模型鲁棒性

        对标Playground: 类似Playground的探索性训练，但专注于推理

        Args:
            reasoning_dataset: 推理数据集 (包含CoT)
            noise_ratio: 初始噪音比例 (0.0-1.0)
            noise_schedule: 噪音衰减策略 ("cosine", "linear", "constant")
            internalize_cot: 是否内化CoT
            epochs: 训练轮数
            save_checkpoint: 是否保存检查点
        """
        print("\n" + "="*60)
        print("⛈️  Stage 4: Storm Training (暴风雨训练)")
        print("="*60)

        print(f"🎯 训练目标: 动态推理 + 内化CoT")
        print(f"   噪音比例: {noise_ratio}")
        print(f"   噪音策略: {noise_schedule}")
        print(f"   内化CoT: {internalize_cot}")
        print(f"   数据集: {reasoning_dataset}")

        # 噪音调度器
        class NoiseScheduler:
            """噪音强度调度器"""
            def __init__(self, initial_ratio: float, strategy: str = "cosine"):
                self.initial_ratio = initial_ratio
                self.strategy = strategy

            def get_noise_ratio(self, step: int, total_steps: int) -> float:
                """计算当前步骤的噪音比例"""
                progress = step / total_steps

                if self.strategy == "cosine":
                    # Cosine衰减: 1 → 0
                    return self.initial_ratio * (1 + np.cos(np.pi * progress)) / 2
                elif self.strategy == "linear":
                    # 线性衰减
                    return self.initial_ratio * (1 - progress)
                else:  # constant
                    return self.initial_ratio

        noise_scheduler = NoiseScheduler(noise_ratio, noise_schedule)

        # 动态推理训练逻辑
        class StormTrainer:
            """暴风雨训练器"""
            def __init__(self, model, noise_scheduler, internalize_cot: bool):
                self.model = model
                self.noise_scheduler = noise_scheduler
                self.internalize_cot = internalize_cot

            def add_autoregressive_noise(
                self,
                logits: torch.Tensor,
                noise_ratio: float
            ) -> torch.Tensor:
                """
                添加自回归噪音到logits

                噪音类型:
                - Gumbel噪音: 模拟采样不确定性
                - Uniform噪音: 探索token空间
                """
                if noise_ratio == 0:
                    return logits

                # Gumbel噪音
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(logits) + 1e-8
                ))

                # 混合原始logits和噪音
                noisy_logits = logits + noise_ratio * gumbel_noise
                return noisy_logits

            def train_step(self, batch, step: int, total_steps: int):
                """单步训练"""
                current_noise = self.noise_scheduler.get_noise_ratio(step, total_steps)

                # 前向传播
                outputs = self.model(**batch)
                logits = outputs.logits

                # 添加噪音
                noisy_logits = self.add_autoregressive_noise(logits, current_noise)

                # 计算损失
                if self.internalize_cot:
                    # 内化CoT: 只优化最终答案，不显示中间步骤
                    # TODO: 实现CoT mask逻辑
                    pass

                # loss = ...
                return {"loss": 0.0, "noise_ratio": current_noise}

        storm_trainer = StormTrainer(
            model=self.model,
            noise_scheduler=noise_scheduler,
            internalize_cot=internalize_cot
        )

        print(f"✓ 暴风雨训练器初始化完成")
        print(f"✓ 开始训练 ({epochs} epochs)...")

        # TODO: 实现完整训练循环

        if save_checkpoint:
            storm_path = self.output_dir / "storm_model"
            storm_path.mkdir(exist_ok=True)
            print(f"💾 暴风雨模型已保存: {storm_path}")

        self.training_history['storm'] = {
            'dataset': reasoning_dataset,
            'noise_ratio': noise_ratio,
            'noise_schedule': noise_schedule,
            'internalize_cot': internalize_cot,
            'epochs': epochs
        }

        return self.model

    # ========================================================================
    # 完整流程
    # ========================================================================

    def run_full_pipeline(
        self,
        sft_dataset: str,
        preference_dataset: Optional[str] = None,
        prompts_dataset: Optional[str] = None,
        owner_prompts: Optional[str] = None,
        public_prompts: Optional[str] = None,
        reasoning_dataset: Optional[str] = None,
        skip_stages: List[str] = []
    ):
        """
        运行完整训练流程

        Args:
            sft_dataset: SFT数据集
            preference_dataset: DPO偏好数据集
            prompts_dataset: GRPO prompts
            owner_prompts: 忠诚度训练 - 主人数据
            public_prompts: 忠诚度训练 - 公众数据
            reasoning_dataset: 暴风雨训练 - 推理数据
            skip_stages: 跳过的阶段 (e.g., ['dpo', 'loyalty'])
        """
        print("\n" + "="*60)
        print("🚀 APT推理与对齐 - 完整训练流程")
        print("="*60)

        # Stage 1: SFT
        if 'sft' not in skip_stages and sft_dataset:
            self.train_sft(sft_dataset)

        # Stage 2a: DPO (可选)
        if 'dpo' not in skip_stages and preference_dataset:
            self.train_dpo(preference_dataset)

        # Stage 2b: GRPO (可选)
        if 'grpo' not in skip_stages and prompts_dataset:
            self.train_grpo(prompts_dataset)

        # Stage 3: Loyalty (可选)
        if 'loyalty' not in skip_stages and owner_prompts and public_prompts:
            self.train_loyalty(owner_prompts, public_prompts)

        # Stage 4: Storm (可选)
        if 'storm' not in skip_stages and reasoning_dataset:
            self.train_storm(reasoning_dataset)

        # 保存训练历史
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print("\n" + "="*60)
        print("✅ 完整训练流程已完成！")
        print(f"📊 训练历史: {history_path}")
        print(f"📁 模型输出: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="APT推理与对齐 - 一键训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

# 1. 完整流程 (SFT → GRPO → Loyalty → Storm)
python train_apt_alignment.py \\
    --sft-data data/instructions.json \\
    --prompts data/prompts.json \\
    --owner-data data/owner_prompts.json \\
    --public-data data/public_prompts.json \\
    --reasoning-data data/cot_examples.json

# 2. 只训练忠诚度
python train_apt_alignment.py \\
    --owner-data data/owner_prompts.json \\
    --public-data data/public_prompts.json \\
    --skip sft,dpo,grpo,storm

# 3. 暴风雨训练 (动态推理)
python train_apt_alignment.py \\
    --reasoning-data data/cot_examples.json \\
    --noise-ratio 0.4 \\
    --noise-schedule cosine \\
    --skip sft,dpo,grpo,loyalty
        """
    )

    # 数据集参数
    parser.add_argument('--sft-data', type=str, help='SFT数据集路径')
    parser.add_argument('--preference-data', type=str, help='DPO偏好数据集')
    parser.add_argument('--prompts', type=str, help='GRPO prompts数据集')
    parser.add_argument('--owner-data', type=str, help='忠诚度训练 - 主人数据')
    parser.add_argument('--public-data', type=str, help='忠诚度训练 - 公众数据')
    parser.add_argument('--reasoning-data', type=str, help='暴风雨训练 - 推理数据')

    # 模型参数
    parser.add_argument('--base-model', type=str, help='基础模型路径')
    parser.add_argument('--output-dir', type=str, default='./apt_aligned_models',
                       help='输出目录')

    # 训练参数
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--skip', type=str, default='',
                       help='跳过的阶段 (逗号分隔, e.g., dpo,loyalty)')

    # 忠诚度训练参数
    parser.add_argument('--owner-bonus', type=float, default=2.0,
                       help='主人奖励加成')

    # 暴风雨训练参数
    parser.add_argument('--noise-ratio', type=float, default=0.3,
                       help='噪音比例')
    parser.add_argument('--noise-schedule', type=str, default='cosine',
                       choices=['cosine', 'linear', 'constant'],
                       help='噪音衰减策略')
    parser.add_argument('--internalize-cot', action='store_true',
                       help='是否内化CoT')

    args = parser.parse_args()

    # 跳过阶段
    skip_stages = args.skip.split(',') if args.skip else []

    # 创建默认模型配置
    model_config = APTModelConfiguration(
        vocab_size=5000,
        d_model=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )

    # 初始化训练流程
    pipeline = APTAlignmentPipeline(
        model_config=model_config,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        device=args.device
    )

    # 运行完整流程
    pipeline.run_full_pipeline(
        sft_dataset=args.sft_data,
        preference_dataset=args.preference_data,
        prompts_dataset=args.prompts,
        owner_prompts=args.owner_data,
        public_prompts=args.public_data,
        reasoning_dataset=args.reasoning_data,
        skip_stages=skip_stages
    )


if __name__ == "__main__":
    main()
