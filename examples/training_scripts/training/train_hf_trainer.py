#!/usr/bin/env python3
"""
APT模型 + HuggingFace Trainer集成
将APT模型适配到HuggingFace生态系统

特性:
- 🤗 HuggingFace Trainer API
- 📊 Weights & Biases / TensorBoard集成
- 🚀 自动混合精度训练
- 💾 HuggingFace Hub模型上传
- 🔧 DeepSpeed集成（通过HF Trainer）
- 📈 丰富的训练回调（EarlyStopping, LR监控等）

使用前准备:
pip install transformers datasets accelerate wandb
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from transformers import (
        PreTrainedModel,
        PretrainedConfig,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        EarlyStoppingCallback
    )
    from transformers.modeling_outputs import CausalLMOutput
    from datasets import Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace Transformers未安装，请运行:")
    print("   pip install transformers datasets accelerate")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from apt.apt_model.modeling.apt_model import APTModel, APTModelConfiguration
from train_hlbd_playground import DynamicTagTokenizer, HLBDPlaygroundDataset, collate_fn


# ============================================================================
# APT模型HuggingFace适配器
# ============================================================================

class APTConfigLegacy(PretrainedConfig):
    """APT模型配置（旧版训练脚本用, 正式版请用 apt.model.hf_compat.APTConfig）"""

    model_type = "apt_legacy"

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_dbc_dac: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_dbc_dac = use_dbc_dac


class APTForCausalLMLegacy(PreTrainedModel):
    """APT模型HuggingFace包装器（旧版, 正式版请用 apt.model.hf_compat.APTForCausalLM）"""

    config_class = APTConfigLegacy

    def __init__(self, config: APTConfigLegacy):
        super().__init__(config)

        # 创建APT模型配置
        apt_config = APTModelConfiguration(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_dbc_dac=config.use_dbc_dac
        )

        # 包装原始APT模型
        self.model = APTModel(apt_config)

        # 损失函数
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # 初始化权重
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutput:
        """
        HuggingFace标准前向传播

        Args:
            input_ids: 输入token IDs [batch, seq_len]
            attention_mask: 注意力mask（可选）
            labels: 标签（用于计算损失） [batch, seq_len]
        """

        # APT模型前向传播
        logits = self.model(input_ids)  # [batch, seq_len, vocab_size]

        # 计算损失
        loss = None
        if labels is not None:
            # 移位处理（语言模型标准做法）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        简单的生成函数（贪婪解码）

        对于更复杂的生成，可以使用HuggingFace的GenerationMixin
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # 检查EOS
                if next_token.item() == 3:  # EOS token
                    break

        return input_ids


# ============================================================================
# HuggingFace Dataset适配器
# ============================================================================

class HLBDHFDataset(Dataset):
    """HLBD数据集的HuggingFace适配器"""

    def __init__(self, json_path: str, tokenizer: DynamicTagTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"📂 加载HLBD数据集: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 合并所有模块
        self.pairs = []
        for module_name, module_data in data['data'].items():
            for item in module_data:
                self.pairs.append((item['input'], item['output']))

        print(f"   ✓ 加载 {len(self.pairs)} 个训练对")

        # 预填充词汇表
        print(f"   预填充词汇表...")
        for src, tgt in self.pairs:
            self.tokenizer.encode(src)
            self.tokenizer.encode(tgt)
        print(f"   ✓ 词汇表大小: {len(self.tokenizer.char_to_id)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        # Encode
        src_ids = self.tokenizer.encode(src)
        tgt_ids = self.tokenizer.encode(tgt)

        # 拼接: [src, SEP, tgt]
        input_ids = src_ids + [1] + tgt_ids

        # 截断
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        # HuggingFace格式
        return {
            'input_ids': torch.tensor(input_ids[:-1], dtype=torch.long),
            'labels': torch.tensor(input_ids[1:], dtype=torch.long)
        }


def data_collator(features):
    """HuggingFace数据整理器"""
    input_ids = [f['input_ids'] for f in features]
    labels = [f['labels'] for f in features]

    max_len = max(len(ids) for ids in input_ids)

    padded_input = []
    padded_labels = []

    for inp, lab in zip(input_ids, labels):
        pad_len = max_len - len(inp)
        padded_input.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        'input_ids': torch.stack(padded_input),
        'labels': torch.stack(padded_labels)
    }


# ============================================================================
# 自定义训练回调
# ============================================================================

class APTTrainingCallback(TrainerCallback):
    """APT训练监控回调"""

    def __init__(self, tokenizer: DynamicTagTokenizer):
        self.tokenizer = tokenizer
        self.best_loss = float('inf')

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录训练日志"""
        if logs:
            current_loss = logs.get('loss', None)
            if current_loss and current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"🎯 新最佳Loss: {self.best_loss:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Epoch结束回调"""
        print(f"\n✅ Epoch {state.epoch} 完成")
        print(f"   全局步数: {state.global_step}")
        print(f"   最佳Loss: {self.best_loss:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束回调"""
        print("\n" + "=" * 60)
        print("🎉 HuggingFace训练完成！")
        print("=" * 60)
        print(f"总步数: {state.global_step}")
        print(f"最佳Loss: {self.best_loss:.4f}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='APT + HuggingFace Trainer训练')

    # 数据参数
    parser.add_argument('--dataset', type=str, default='../data/HLBD_Hardcore_Full.json',
                       help='HLBD数据集路径')
    parser.add_argument('--output-dir', type=str, default='hf_output',
                       help='输出目录')

    # 模型参数
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--n-heads', type=int, default=8)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)

    # HuggingFace特性
    parser.add_argument('--fp16', action='store_true',
                       help='启用FP16混合精度')
    parser.add_argument('--bf16', action='store_true',
                       help='启用BF16混合精度')
    parser.add_argument('--deepspeed', type=str, default=None,
                       help='DeepSpeed配置文件路径')
    parser.add_argument('--logging-steps', type=int, default=20)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--save-total-limit', type=int, default=3)

    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                       help='启用Weights & Biases跟踪')
    parser.add_argument('--wandb-project', type=str, default='apt-training',
                       help='W&B项目名称')

    # 早停
    parser.add_argument('--early-stopping', action='store_true',
                       help='启用早停')
    parser.add_argument('--early-stopping-patience', type=int, default=5)

    args = parser.parse_args()

    if not HF_AVAILABLE:
        print("\n❌ HuggingFace Transformers未安装")
        print("   请运行: pip install transformers datasets accelerate")
        return

    print("\n🤗 APT + HuggingFace Trainer训练")
    print("=" * 60)

    # Tokenizer
    print("\n🔤 初始化Tokenizer（支持动态标签）...")
    tokenizer = DynamicTagTokenizer(vocab_size=5000)

    # 数据集
    print("\n📂 加载数据集...")
    dataset = HLBDHFDataset(args.dataset, tokenizer)

    # 模型配置
    print("\n🏗️  构建APT模型...")
    config = APTConfigLegacy(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.n_layers,
        num_decoder_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=256,
        dropout=0.1,
        use_dbc_dac=True
    )

    model = APTForCausalLMLegacy(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数: {total_params:,}")

    # 训练参数
    print("\n⚙️  配置训练参数...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,

        # 混合精度
        fp16=args.fp16,
        bf16=args.bf16,

        # 日志和保存
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # DeepSpeed
        deepspeed=args.deepspeed,

        # 报告
        report_to="wandb" if args.wandb else "none",
        run_name="apt-hlbd-training" if args.wandb else None,

        # 其他
        remove_unused_columns=False,
        dataloader_num_workers=0,
        load_best_model_at_end=True if args.early_stopping else False,
    )

    # Weights & Biases
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args)
            )
            print("   ✓ Weights & Biases已启用")
        except ImportError:
            print("   ⚠️  wandb未安装，跳过W&B集成")

    # 回调
    callbacks = [APTTrainingCallback(tokenizer)]

    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
        print(f"   ✓ 早停已启用 (patience={args.early_stopping_patience})")

    # Trainer
    print("\n🚀 初始化HuggingFace Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # 训练
    print("\n" + "=" * 60)
    print("🚀 开始HuggingFace训练")
    print("=" * 60)

    trainer.train()

    # 保存最终模型
    print("\n💾 保存模型...")
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_path))

    # 保存tokenizer状态
    tokenizer_path = final_path / "tokenizer_state.json"
    with open(tokenizer_path, 'w') as f:
        json.dump({
            'char_to_id': tokenizer.char_to_id,
            'id_to_char': {str(k): v for k, v in tokenizer.id_to_char.items()},
            'next_id': tokenizer.next_id,
            'vocab_size': tokenizer.vocab_size
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 模型已保存: {final_path}")

    # 上传到HuggingFace Hub（可选）
    if os.getenv('HF_HUB_TOKEN'):
        print("\n📤 上传到HuggingFace Hub...")
        try:
            hub_model_id = f"apt-model-{args.d_model}d-{args.n_layers}l"
            trainer.push_to_hub(hub_model_id)
            print(f"✅ 模型已上传: https://huggingface.co/{hub_model_id}")
        except Exception as e:
            print(f"⚠️  上传失败: {e}")

    print("\n" + "=" * 60)
    print("✨ 训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
