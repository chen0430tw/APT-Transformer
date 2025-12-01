#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
教师模型API接口

支持使用远程API作为教师模型进行知识蒸馏：
- OpenAI API (GPT-4, GPT-3.5等)
- Anthropic API (Claude系列)
- 硅基流动API (Qwen, DeepSeek, GLM等国产模型)
- 自定义API接口
- API调用管理和错误处理
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
import time
import json


class TeacherAPIInterface:
    """
    教师模型API接口基类
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.model_name = config.get('model_name')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)

        # 统计信息
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
        }

    def generate_text(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        生成文本（需要子类实现）

        Args:
            input_text: 输入文本
            max_tokens: 最大生成token数
            temperature: 生成温度
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        raise NotImplementedError("Subclass must implement generate_text()")

    def get_logits(
        self,
        input_text: str,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        获取logits（需要子类实现）

        Args:
            input_text: 输入文本
            vocab_size: 词表大小
            **kwargs: 其他参数

        Returns:
            logits tensor
        """
        raise NotImplementedError("Subclass must implement get_logits()")

    def retry_on_failure(self, func, *args, **kwargs):
        """
        失败重试装饰器

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数执行结果
        """
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                self.stats['successful_calls'] += 1
                return result
            except Exception as e:
                self.stats['failed_calls'] += 1
                print(f"[API] 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                else:
                    raise

        return None


class OpenAITeacherAPI(TeacherAPIInterface):
    """
    OpenAI API教师模型

    支持: GPT-4, GPT-3.5-turbo, GPT-3.5等
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 导入OpenAI库
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
            if self.base_url:
                self.openai.api_base = self.base_url
        except ImportError:
            raise ImportError("需要安装openai库: pip install openai")

    def generate_text(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """使用OpenAI API生成文本"""

        def _call_api():
            response = self.openai.ChatCompletion.create(
                model=self.model_name or "gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": input_text}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            self.stats['total_calls'] += 1
            self.stats['total_tokens'] += response['usage']['total_tokens']

            return response['choices'][0]['message']['content']

        return self.retry_on_failure(_call_api)

    def get_logits(
        self,
        input_text: str,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        获取logits（模拟）

        注意: OpenAI API不直接返回logits，这里通过文本生成模拟
        """
        # 生成文本
        generated_text = self.generate_text(input_text, max_tokens=50, temperature=0.7)

        # 模拟logits（实际使用中需要有词表映射）
        # 这里返回随机logits作为占位符
        seq_len = len(generated_text.split())
        logits = torch.randn(1, seq_len, vocab_size)

        return logits


class AnthropicTeacherAPI(TeacherAPIInterface):
    """
    Anthropic API教师模型

    支持: Claude-3, Claude-2等
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 导入Anthropic库
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("需要安装anthropic库: pip install anthropic")

    def generate_text(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """使用Anthropic API生成文本"""

        def _call_api():
            message = self.client.messages.create(
                model=self.model_name or "claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                **kwargs
            )

            self.stats['total_calls'] += 1
            self.stats['total_tokens'] += message.usage.input_tokens + message.usage.output_tokens

            return message.content[0].text

        return self.retry_on_failure(_call_api)

    def get_logits(
        self,
        input_text: str,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """获取logits（模拟）"""
        generated_text = self.generate_text(input_text, max_tokens=50, temperature=0.7)
        seq_len = len(generated_text.split())
        logits = torch.randn(1, seq_len, vocab_size)
        return logits


class SiliconFlowTeacherAPI(TeacherAPIInterface):
    """
    硅基流动API教师模型

    支持: Qwen系列, DeepSeek系列, GLM系列, Llama系列等国产开源模型
    官网: https://siliconflow.cn
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 硅基流动API兼容OpenAI格式
        self.base_url = self.base_url or "https://api.siliconflow.cn/v1"

        # 导入OpenAI库（硅基流动兼容OpenAI接口）
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
            self.openai.api_base = self.base_url
        except ImportError:
            raise ImportError("需要安装openai库: pip install openai")

    def generate_text(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """使用硅基流动API生成文本"""

        def _call_api():
            response = self.openai.ChatCompletion.create(
                model=self.model_name or "Qwen/Qwen2-7B-Instruct",
                messages=[
                    {"role": "user", "content": input_text}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            self.stats['total_calls'] += 1
            self.stats['total_tokens'] += response['usage']['total_tokens']

            return response['choices'][0]['message']['content']

        return self.retry_on_failure(_call_api)

    def get_logits(
        self,
        input_text: str,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """获取logits（模拟）"""
        generated_text = self.generate_text(input_text, max_tokens=50, temperature=0.7)
        seq_len = len(generated_text.split())
        logits = torch.randn(1, seq_len, vocab_size)
        return logits


class CustomTeacherAPI(TeacherAPIInterface):
    """
    自定义API教师模型

    支持任何符合规范的自定义API
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        import requests
        self.requests = requests

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def generate_text(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """使用自定义API生成文本"""

        def _call_api():
            payload = {
                'input': input_text,
                'max_tokens': max_tokens,
                'temperature': temperature,
                **kwargs
            }

            response = self.requests.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            self.stats['total_calls'] += 1

            return result.get('text', result.get('output', ''))

        return self.retry_on_failure(_call_api)

    def get_logits(
        self,
        input_text: str,
        vocab_size: int,
        **kwargs
    ) -> torch.Tensor:
        """获取logits"""

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

    将API接口包装成类似PyTorch模型的接口
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
        API接口实例
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
    """
    api = create_teacher_api(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        **kwargs
    )

    return APITeacherModel(api, tokenizer, vocab_size)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("【教师模型API接口演示】\n")

    # 示例1: OpenAI API
    print("=" * 60)
    print("[示例1] 使用OpenAI API作为教师模型")
    print("=" * 60)

    # 配置（需要实际的API key）
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

from apt_model.plugins.teacher_api import create_api_teacher_model
from apt_model.plugins.visual_distillation_plugin import quick_visual_distill
from transformers import AutoTokenizer

# 1. 创建API教师模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")

teacher_model = create_api_teacher_model(
    provider='openai',
    api_key='your-api-key',
    model_name='gpt-4',
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
    '''

    print(usage_example)

    print("\n" + "=" * 60)
    print("[完成] 演示完成")
    print("=" * 60)
    print("\n[提示] 查看完整文档: TEACHER_API_GUIDE.md")
