#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
外部API提供商接口

支持的API:
- OpenAI API (GPT-4, GPT-3.5等)
- Anthropic API (Claude系列)
- 硅基流动API (Qwen, DeepSeek, GLM等国产模型)
- 自定义API接口

可用于:
- 知识蒸馏 (Teacher Model)
- RAG增强生成
- 知识图谱构建
- 其他需要外部模型的场景
"""

import torch
from typing import Optional, Dict, Any, List
import time
import json


class APIProviderInterface:
    """
    API提供商接口基类

    所有API提供商都应继承此类并实现相应方法
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
            'total_cost': 0.0,  # 美元
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

    def get_embedding(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        获取文本嵌入（可选实现）

        Args:
            text: 输入文本
            **kwargs: 其他参数

        Returns:
            嵌入向量
        """
        raise NotImplementedError("Subclass must implement get_embedding()")

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

    def get_stats(self) -> Dict[str, Any]:
        """获取API调用统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
        }


class OpenAIProvider(APIProviderInterface):
    """
    OpenAI API提供商

    支持: GPT-4, GPT-3.5-turbo, GPT-3等
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

        # 定价信息 (美元/1M tokens)
        self.pricing = {
            'gpt-4': {'input': 30.0, 'output': 60.0},
            'gpt-4-turbo': {'input': 10.0, 'output': 30.0},
            'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
        }

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

            # 计算成本
            model = self.model_name or "gpt-3.5-turbo"
            if model in self.pricing:
                input_tokens = response['usage']['prompt_tokens']
                output_tokens = response['usage']['completion_tokens']
                cost = (input_tokens * self.pricing[model]['input'] / 1_000_000 +
                       output_tokens * self.pricing[model]['output'] / 1_000_000)
                self.stats['total_cost'] += cost

            return response['choices'][0]['message']['content']

        return self.retry_on_failure(_call_api)

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """获取文本嵌入"""

        def _call_api():
            response = self.openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text,
                **kwargs
            )

            self.stats['total_calls'] += 1
            return response['data'][0]['embedding']

        return self.retry_on_failure(_call_api)


class AnthropicProvider(APIProviderInterface):
    """
    Anthropic API提供商

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

        # 定价信息 (美元/1M tokens)
        self.pricing = {
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
        }

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

            # 计算成本
            model = self.model_name or "claude-3-sonnet-20240229"
            if model in self.pricing:
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                cost = (input_tokens * self.pricing[model]['input'] / 1_000_000 +
                       output_tokens * self.pricing[model]['output'] / 1_000_000)
                self.stats['total_cost'] += cost

            return message.content[0].text

        return self.retry_on_failure(_call_api)


class SiliconFlowProvider(APIProviderInterface):
    """
    硅基流动API提供商

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

        # 定价信息 (人民币/1M tokens, 换算为美元 1 RMB = 0.14 USD)
        self.pricing = {
            'Qwen/Qwen2-7B-Instruct': 1.0 * 0.14,
            'Qwen/Qwen2-72B-Instruct': 5.0 * 0.14,
            'deepseek-ai/DeepSeek-V2.5': 1.0 * 0.14,
            'THUDM/glm-4-9b-chat': 1.0 * 0.14,
            'meta-llama/Meta-Llama-3-8B-Instruct': 0.6 * 0.14,
            'meta-llama/Meta-Llama-3-70B-Instruct': 3.0 * 0.14,
        }

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

            # 计算成本
            model = self.model_name or "Qwen/Qwen2-7B-Instruct"
            if model in self.pricing:
                total_tokens = response['usage']['total_tokens']
                cost = total_tokens * self.pricing[model] / 1_000_000
                self.stats['total_cost'] += cost

            return response['choices'][0]['message']['content']

        return self.retry_on_failure(_call_api)


class CustomProvider(APIProviderInterface):
    """
    自定义API提供商

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

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """获取文本嵌入"""

        def _call_api():
            payload = {
                'input': text,
                **kwargs
            }

            response = self.requests.post(
                f"{self.base_url}/embedding",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            self.stats['total_calls'] += 1

            return result.get('embedding', [])

        return self.retry_on_failure(_call_api)


# ==================== 便捷函数 ====================

def create_api_provider(
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> APIProviderInterface:
    """
    创建API提供商

    Args:
        provider: API提供商 ('openai', 'anthropic', 'siliconflow', 'custom')
        api_key: API密钥
        model_name: 模型名称
        base_url: API基础URL（可选）
        **kwargs: 其他配置

    Returns:
        API提供商实例

    Examples:
        >>> # OpenAI
        >>> api = create_api_provider('openai', 'sk-...', 'gpt-4')
        >>>
        >>> # 硅基流动
        >>> api = create_api_provider('siliconflow', 'sk-...', 'Qwen/Qwen2-7B-Instruct')
        >>>
        >>> # 生成文本
        >>> text = api.generate_text("什么是人工智能?", max_tokens=100)
    """
    config = {
        'api_key': api_key,
        'model_name': model_name,
        'base_url': base_url,
        **kwargs
    }

    if provider.lower() == 'openai':
        return OpenAIProvider(config)
    elif provider.lower() == 'anthropic':
        return AnthropicProvider(config)
    elif provider.lower() == 'siliconflow':
        return SiliconFlowProvider(config)
    elif provider.lower() == 'custom':
        if not base_url:
            raise ValueError("自定义API需要提供base_url")
        return CustomProvider(config)
    else:
        raise ValueError(f"不支持的provider: {provider}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("【外部API提供商演示】\n")

    # 示例1: OpenAI
    print("=" * 60)
    print("[示例1] OpenAI API")
    print("=" * 60)

    openai_example = '''
from apt.core.api_providers import create_api_provider

# 创建OpenAI提供商
api = create_api_provider(
    provider='openai',
    api_key='sk-...',
    model_name='gpt-3.5-turbo'
)

# 生成文本
text = api.generate_text(
    input_text="什么是知识蒸馏?",
    max_tokens=100,
    temperature=0.7
)

print(text)

# 查看统计
print(api.get_stats())
# {
#     'total_calls': 1,
#     'successful_calls': 1,
#     'failed_calls': 0,
#     'total_tokens': 150,
#     'total_cost': 0.000225  # 美元
# }
    '''

    print(openai_example)

    # 示例2: 硅基流动
    print("=" * 60)
    print("[示例2] 硅基流动API (中文优化)")
    print("=" * 60)

    siliconflow_example = '''
from apt.core.api_providers import create_api_provider

# 创建硅基流动提供商
api = create_api_provider(
    provider='siliconflow',
    api_key='sk-...',
    model_name='Qwen/Qwen2-7B-Instruct'
)

# 生成中文文本
text = api.generate_text(
    input_text="解释一下RAG技术的原理",
    max_tokens=200,
    temperature=0.7
)

print(text)

# 成本极低
print(f"成本: ${api.stats['total_cost']:.6f}")  # 约 $0.00003
    '''

    print(siliconflow_example)

    # 示例3: 在不同场景使用
    print("=" * 60)
    print("[示例3] 在不同场景使用API")
    print("=" * 60)

    usage_scenarios = '''
# 场景1: 知识蒸馏
from apt_model.plugins.teacher_api import create_api_teacher_model

teacher = create_api_teacher_model(
    provider='siliconflow',
    api_key='sk-...',
    model_name='Qwen/Qwen2-7B-Instruct',
    tokenizer=tokenizer
)

# 场景2: RAG增强 (将来支持)
from apt_model.modeling.kg_rag_integration import KGRAGWrapper

rag = KGRAGWrapper(
    base_model=model,
    api_provider='siliconflow',
    api_key='sk-...',
    api_model='Qwen/Qwen2-7B-Instruct'
)

# 场景3: 知识图谱构建 (将来支持)
from apt_model.modeling.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.build_from_text_with_api(
    text=documents,
    api_provider='siliconflow',
    api_key='sk-...'
)
    '''

    print(usage_scenarios)

    print("\n" + "=" * 60)
    print("[提示] 这是独立的API配置模块，可被多个组件共享使用")
    print("=" * 60)
