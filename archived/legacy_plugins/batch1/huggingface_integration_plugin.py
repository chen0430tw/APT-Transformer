"""
HuggingFace Integration Plugin for APT Model
提供与HuggingFace Hub的完整集成功能
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, upload_folder, login

# 假设插件基类已存在
# from plugins.plugin_system import APTPlugin


class HuggingFaceIntegrationPlugin:
    """
    HuggingFace集成插件
    
    功能:
    1. 从HuggingFace Hub导入/导出模型
    2. 加载HuggingFace数据集
    3. 使用HF Trainer训练APT模型
    4. 上传模型到HuggingFace Hub
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "huggingface-integration"
        self.version = "1.0.0"
        self.config = config
        self.api = HfApi()
        
    # ==================== 模型导入/导出 ====================
    
    def export_to_huggingface(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        repo_name: str,
        private: bool = False,
        commit_message: str = "Upload APT model"
    ):
        """
        导出APT模型到HuggingFace Hub
        
        Args:
            model: APT模型实例
            tokenizer: 分词器
            repo_name: 仓库名 (格式: username/model-name)
            private: 是否为私有仓库
            commit_message: 提交消息
        """
        print(f"🚀 正在将模型导出到 HuggingFace Hub: {repo_name}")
        
        # 1. 创建本地保存目录
        save_dir = Path(f"./hf_export/{repo_name.split('/')[-1]}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 保存模型和分词器
        torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
        tokenizer.save_pretrained(save_dir)
        
        # 3. 创建model_card
        self._create_model_card(save_dir, repo_name)
        
        # 4. 创建仓库并上传
        try:
            create_repo(repo_name, private=private, exist_ok=True)
            upload_folder(
                repo_id=repo_name,
                folder_path=str(save_dir),
                commit_message=commit_message
            )
            print(f"✅ 模型成功上传到: https://huggingface.co/{repo_name}")
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            raise
    
    def import_from_huggingface(
        self,
        repo_name: str,
        local_dir: Optional[str] = None
    ) -> tuple:
        """
        从HuggingFace Hub导入模型
        
        Args:
            repo_name: 仓库名
            local_dir: 本地保存目录
            
        Returns:
            (model, tokenizer) 元组
        """
        print(f"📥 正在从 HuggingFace Hub 导入模型: {repo_name}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(repo_name)
            tokenizer = AutoTokenizer.from_pretrained(repo_name)
            
            if local_dir:
                model.save_pretrained(local_dir)
                tokenizer.save_pretrained(local_dir)
                print(f"✅ 模型已保存到: {local_dir}")
            
            return model, tokenizer
        except Exception as e:
            print(f"❌ 导入失败: {e}")
            raise
    
    # ==================== 数据集加载 ====================
    
    def load_hf_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """
        加载HuggingFace数据集
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割 (train/test/validation)
            **kwargs: 其他参数传递给load_dataset
            
        Returns:
            Dataset对象
        """
        print(f"📚 正在加载 HuggingFace 数据集: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name, split=split, **kwargs)
            print(f"✅ 数据集加载成功! 样本数: {len(dataset)}")
            return dataset
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            raise
    
    def convert_to_apt_format(self, hf_dataset: Dataset) -> list:
        """
        将HuggingFace数据集转换为APT格式
        
        Args:
            hf_dataset: HuggingFace数据集
            
        Returns:
            APT格式的数据列表
        """
        apt_data = []
        for item in hf_dataset:
            # 根据实际需求转换格式
            apt_item = {
                'text': item.get('text', ''),
                'label': item.get('label', None),
                # 可以添加更多字段
            }
            apt_data.append(apt_item)
        
        return apt_data
    
    # ==================== HF Trainer集成 ====================
    
    def train_with_hf_trainer(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./hf_trainer_output",
        **training_args_kwargs
    ):
        """
        使用HuggingFace Trainer训练APT模型
        
        Args:
            model: APT模型
            tokenizer: 分词器
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            output_dir: 输出目录
            **training_args_kwargs: TrainingArguments的参数
        """
        print("🏋️ 使用 HuggingFace Trainer 开始训练...")
        
        # 设置默认训练参数
        default_args = {
            'output_dir': output_dir,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': f'{output_dir}/logs',
            'logging_steps': 100,
            'save_strategy': 'epoch',
            'evaluation_strategy': 'epoch' if eval_dataset else 'no',
        }
        default_args.update(training_args_kwargs)
        
        training_args = TrainingArguments(**default_args)
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"✅ 训练完成! 模型已保存到: {output_dir}")
    
    # ==================== 工具方法 ====================
    
    def _create_model_card(self, save_dir: Path, repo_name: str):
        """创建模型卡片"""
        model_card = f"""---
language: 
- zh
- en
tags:
- apt-model
- text-generation
- autopoietic-attention
license: apache-2.0
---

# {repo_name}

## 模型描述

这是一个基于APT (Autopoietic Transformer) 架构的语言模型。

### 特点
- 🧠 自生成注意力机制
- 🛡️ DBC-DAC梯度稳定
- 🌏 完整的中文支持
- ⚡ 混合精度训练

## 使用方法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

text = "你好，世界"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## 训练信息

由APT框架训练: https://github.com/your-repo/apt-model

## 许可证

Apache 2.0
"""
        
        with open(save_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(model_card)
    
    def login_to_hub(self, token: Optional[str] = None):
        """登录到HuggingFace Hub"""
        if token:
            login(token=token)
        else:
            login()  # 使用缓存的token
        print("✅ 已登录到 HuggingFace Hub")
    
    # ==================== 插件钩子 ====================
    
    def on_training_end(self, context: Dict[str, Any]):
        """训练结束时自动上传到HuggingFace Hub"""
        if self.config.get('auto_upload', False):
            repo_name = self.config.get('repo_name')
            if repo_name:
                self.export_to_huggingface(
                    model=context['model'],
                    tokenizer=context['tokenizer'],
                    repo_name=repo_name,
                    private=self.config.get('private', False)
                )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置
    config = {
        'auto_upload': True,
        'repo_name': 'username/apt-chinese-base',
        'private': False,
    }
    
    plugin = HuggingFaceIntegrationPlugin(config)
    
    # 示例1: 加载HuggingFace数据集
    dataset = plugin.load_hf_dataset("wikitext", split="train")
    
    # 示例2: 导出模型到HuggingFace Hub
    # plugin.login_to_hub("your_token")
    # plugin.export_to_huggingface(model, tokenizer, "username/my-apt-model")
    
    # 示例3: 从HuggingFace导入模型
    # model, tokenizer = plugin.import_from_huggingface("gpt2")
    
    print("✅ HuggingFace Integration Plugin 示例完成!")
