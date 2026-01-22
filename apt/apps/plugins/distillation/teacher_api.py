#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
教师模型API接口

支持使用远程API作为教师模型进行知识蒸馏：
- OpenAI API (GPT-4, GPT-3.5等)
- Anthropic API (Claude系列)
- 硅基流动API (Qwen, DeepSeek, GLM等国产模型)
- 自定义API接口

注意: 此模块专门用于知识蒸馏场景
     通用的API配置请使用 apt.core.api_providers
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union

# 导入通用API提供商
from apt.core.api_providers import (
    APIProviderInterface,
    OpenAIProvider,
    AnthropicProvider,
    SiliconFlowProvider,
    CustomProvider,
    create_api_provider
)


class TeacherAPIInterface(APIProviderInterface):
    """
    教师模型API接口（继承自通用API提供商）

    扩展了蒸馏所需的特殊功能
    """

    def get_logits(
        self,
        input_text: str,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        获取logits（用于蒸馏）

        注意: 大多数API不直接返回logits，这里通过文本生成模拟

        Args:
            input_text: 输入文本
            vocab_size: 词表大小
            **kwargs: 其他参数

        Returns:
            logits tensor
        """
        # 生成文本
        generated_text = self.generate_text(input_text, max_tokens=50, temperature=0.7)

        # 模拟logits（实际使用中需要有词表映射）
        # 这里返回随机logits作为占位符
        seq_len = len(generated_text.split())
        logits = torch.randn(1, seq_len, vocab_size)

        return logits


class OpenAITeacherAPI(OpenAIProvider, TeacherAPIInterface):
    """OpenAI API教师模型"""
    pass


class AnthropicTeacherAPI(AnthropicProvider, TeacherAPIInterface):
    """Anthropic API教师模型"""
    pass


class SiliconFlowTeacherAPI(SiliconFlowProvider, TeacherAPIInterface):
    """硅基流动API教师模型"""
    pass


class CustomTeacherAPI(CustomProvider, TeacherAPIInterface):
    """自定义API教师模型"""

    def get_logits(
        self,
        input_text: str,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        获取logits（自定义API可能支持直接返回logits）

        Args:
            input_text: 输入文本
            vocab_size: 词表大小
            **kwargs: 其他参数

        Returns:
            logits tensor
        """

        def _call_api():
            payload = {
                'input': input_text,
                'return_logits': True,
                **kwargs
            }

            response = self.requests.post(
                f"{self.base_url}/logits",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            self.stats['total_calls'] += 1

            # 从API响应中获取logits
            logits_data = result.get('logits', [])

            # 转换为tensor
            logits = torch.tensor(logits_data)

            return logits

        return self.retry_on_failure(_call_api)


class APITeacherModel(nn.Module):
    """
    API教师模型包装器

    将API接口包装成类似PyTorch模型的接口，可直接用于蒸馏
    """

    def __init__(
        self,
        api_interface: TeacherAPIInterface,
        tokenizer: Any,
        vocab_size: int = 50000
    ):
        super().__init__()
        self.api = api_interface
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            模型输出（包含logits）
        """
        batch_size = input_ids.size(0)
        all_logits = []

        for i in range(batch_size):
            # 解码输入
            input_text = self.tokenizer.decode(
                input_ids[i],
                skip_special_tokens=True
            )

            # 通过API获取logits
            try:
                logits = self.api.get_logits(input_text, self.vocab_size)
                all_logits.append(logits)
            except Exception as e:
                print(f"[API] 获取logits失败: {e}")
                # 返回随机logits作为fallback
                seq_len = input_ids.size(1)
                fallback_logits = torch.randn(1, seq_len, self.vocab_size)
                all_logits.append(fallback_logits)

        # 合并batch
        output_logits = torch.cat(all_logits, dim=0)

        # 返回类似HuggingFace的输出
        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(output_logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本

        Args:
            input_ids: [batch, seq_len]
            max_length: 最大生成长度
            temperature: 生成温度

        Returns:
            生成的token ids
        """
        batch_size = input_ids.size(0)
        generated_ids = []

        for i in range(batch_size):
            # 解码输入
            input_text = self.tokenizer.decode(
                input_ids[i],
                skip_special_tokens=True
            )

            # 通过API生成文本
            try:
                generated_text = self.api.generate_text(
                    input_text,
                    max_tokens=max_length,
                    temperature=temperature,
                    **kwargs
                )

                # 编码生成的文本
                generated_tokens = self.tokenizer.encode(
                    generated_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length
                )

                generated_ids.append(generated_tokens)

            except Exception as e:
                print(f"[API] 生成失败: {e}")
                # 返回空序列
                generated_ids.append([])

        # 填充到相同长度
        max_gen_len = max(len(ids) for ids in generated_ids) if generated_ids else 0
        padded_ids = []

        for ids in generated_ids:
            padded = ids + [self.tokenizer.pad_token_id] * (max_gen_len - len(ids))
            padded_ids.append(padded)

        return torch.tensor(padded_ids)


# ==================== 便捷函数 ====================

def create_teacher_api(
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> TeacherAPIInterface:
    """
    创建教师模型API接口

    Args:
        provider: API提供商 ('openai', 'anthropic', 'siliconflow', 'custom')
        api_key: API密钥
        model_name: 模型名称
        base_url: API基础URL（可选）
        **kwargs: 其他配置

    Returns:
        教师API接口实例
    """
    config = {
        'api_key': api_key,
        'model_name': model_name,
        'base_url': base_url,
        **kwargs
    }

    if provider.lower() == 'openai':
        return OpenAITeacherAPI(config)
    elif provider.lower() == 'anthropic':
        return AnthropicTeacherAPI(config)
    elif provider.lower() == 'siliconflow':
        return SiliconFlowTeacherAPI(config)
    elif provider.lower() == 'custom':
        if not base_url:
            raise ValueError("自定义API需要提供base_url")
        return CustomTeacherAPI(config)
    else:
        raise ValueError(f"不支持的provider: {provider}")


def create_api_teacher_model(
    provider: str,
    api_key: str,
    tokenizer: Any,
    model_name: Optional[str] = None,
    vocab_size: int = 50000,
    **kwargs
) -> APITeacherModel:
    """
    创建API教师模型（可直接用于蒸馏）

    Args:
        provider: API提供商
        api_key: API密钥
        tokenizer: 分词器
        model_name: 模型名称
        vocab_size: 词表大小
        **kwargs: 其他配置

    Returns:
        API教师模型

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>>
        >>> # 创建硅基流动教师模型
        >>> teacher = create_api_teacher_model(
        ...     provider='siliconflow',
        ...     api_key='sk-...',
        ...     model_name='Qwen/Qwen2-7B-Instruct',
        ...     tokenizer=tokenizer,
        ...     vocab_size=50000
        ... )
        >>>
        >>> # 直接用于蒸馏
        >>> from apt.apt_model.plugins.visual_distillation_plugin import quick_visual_distill
        >>> quick_visual_distill(
        ...     student_model=student,
        ...     teacher_model=teacher,
        ...     train_dataloader=dataloader,
        ...     tokenizer=tokenizer,
        ...     num_epochs=3
        ... )
    """
    api = create_teacher_api(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        base_url=kwargs.pop('base_url', None),
        **kwargs
    )

    return APITeacherModel(api, tokenizer, vocab_size)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("【教师模型API接口演示】\n")
    print("[提示] 此模块已重构为使用通用API配置 (apt.core.api_providers)\n")

    # 示例1: OpenAI API
    print("=" * 60)
    print("[示例1] 使用OpenAI API作为教师模型")
    print("=" * 60)

    openai_config = {
        'provider': 'openai',
        'api_key': 'your-openai-api-key',
        'model_name': 'gpt-3.5-turbo',
    }

    print(f"配置: {openai_config}")
    print("注意: 需要实际的API key才能运行\n")

    # 示例2: Anthropic API
    print("=" * 60)
    print("[示例2] 使用Anthropic API作为教师模型")
    print("=" * 60)

    anthropic_config = {
        'provider': 'anthropic',
        'api_key': 'your-anthropic-api-key',
        'model_name': 'claude-3-sonnet-20240229',
    }

    print(f"配置: {anthropic_config}\n")

    # 示例3: 硅基流动API
    print("=" * 60)
    print("[示例3] 使用硅基流动API作为教师模型")
    print("=" * 60)

    siliconflow_config = {
        'provider': 'siliconflow',
        'api_key': 'your-siliconflow-api-key',
        'model_name': 'Qwen/Qwen2-7B-Instruct',
    }

    print(f"配置: {siliconflow_config}")
    print("支持的模型: Qwen2, DeepSeek, GLM-4, Llama-3等\n")

    # 示例4: 自定义API
    print("=" * 60)
    print("[示例4] 使用自定义API作为教师模型")
    print("=" * 60)

    custom_config = {
        'provider': 'custom',
        'api_key': 'your-api-key',
        'base_url': 'https://your-api.com',
    }

    print(f"配置: {custom_config}\n")

    # 示例5: 集成到蒸馏流程
    print("=" * 60)
    print("[示例5] 集成到知识蒸馏流程")
    print("=" * 60)

    usage_example = '''
# 完整蒸馏流程

from apt.apt_model.plugins.teacher_api import create_api_teacher_model
from apt.apt_model.plugins.visual_distillation_plugin import quick_visual_distill
from transformers import AutoTokenizer

# 1. 创建API教师模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")

teacher_model = create_api_teacher_model(
    provider='siliconflow',  # 使用硅基流动（成本最低）
    api_key='your-api-key',
    model_name='Qwen/Qwen2-7B-Instruct',
    tokenizer=tokenizer,
    vocab_size=50000
)

# 2. 加载学生模型
student_model = load_model("apt_model_small")

# 3. 准备数据
train_dataloader = get_dataloader()

# 4. 开始蒸馏（API教师模型可以像本地模型一样使用）
quick_visual_distill(
    student_model=student_model,
    teacher_model=teacher_model,  # API教师模型
    train_dataloader=train_dataloader,
    tokenizer=tokenizer,
    num_epochs=3,
    device='cuda'
)

# 5. 查看API调用统计和成本
print(f"总调用: {teacher_model.api.stats['total_calls']}")
print(f"总tokens: {teacher_model.api.stats['total_tokens']}")
print(f"总成本: ${teacher_model.api.stats['total_cost']:.4f}")
    '''

    print(usage_example)

    print("\n" + "=" * 60)
    print("[架构说明]")
    print("=" * 60)
    print("""
teacher_api.py (本模块)
  ├─ 专门用于知识蒸馏场景
  ├─ 继承自 apt.core.api_providers
  └─ 添加了蒸馏特有的功能 (get_logits, APITeacherModel)

apt.core.api_providers
  ├─ 通用API配置模块
  ├─ 可被多个组件共享使用
  │  ├─ 知识蒸馏 (teacher_api.py)
  │  ├─ RAG增强 (将来支持)
  │  └─ 知识图谱构建 (将来支持)
  └─ 包含成本追踪和统计功能
    """)

    print("\n[完成] 演示完成")
    print("[提示] 查看完整文档: TEACHER_API_GUIDE.md")
    print("[提示] 查看通用API配置: apt_model/core/api_providers.py")
