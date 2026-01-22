"""
云端NPU适配器 - 通过API调用远程NPU进行推理

支持的云服务商：
- 🟡 Huawei Cloud ModelArts (Ascend NPU)
- 🟢 未来扩展：SaladCloud, RunPod (当支持NPU时)

无需本地NPU硬件，通过REST API调用云端NPU进行推理。

作者: claude + chen0430tw
版本: 1.0 (Cloud NPU Adapter)
"""

import torch
import torch.nn as nn
import requests
import os
from typing import Optional, Dict, List, Any, Union
import json
import warnings


class CloudNPUBackend:
    """云端NPU后端基类"""

    def __init__(self, api_key: str, endpoint_url: str, **kwargs):
        """
        初始化云端NPU后端

        Args:
            api_key: API密钥
            endpoint_url: 推理端点URL
            **kwargs: 其他配置参数
        """
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.config = kwargs
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

    def inference(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行推理请求

        Args:
            inputs: 输入数据

        Returns:
            推理结果
        """
        raise NotImplementedError("子类必须实现inference方法")

    def is_available(self) -> bool:
        """检查云端NPU是否可用"""
        try:
            response = self.session.get(f"{self.endpoint_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """关闭连接"""
        self.session.close()


class HuaweiModelArtsNPU(CloudNPUBackend):
    """华为云ModelArts Ascend NPU后端"""

    def __init__(self,
                 api_key: str,
                 endpoint_url: str,
                 model_name: str = "deepseek-r1",
                 region: str = "cn-north-4",
                 **kwargs):
        """
        初始化华为云ModelArts NPU

        Args:
            api_key: 华为云API密钥
            endpoint_url: ModelArts推理端点URL
            model_name: 部署的模型名称
            region: 华为云区域
        """
        super().__init__(api_key, endpoint_url, **kwargs)
        self.model_name = model_name
        self.region = region

        # 华为云特定headers
        self.session.headers.update({
            'X-HuaweiCloud-Region': region
        })

    def inference(self, inputs: Union[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行推理 - 支持OpenAI兼容API

        Args:
            inputs: 输入数据（可以是torch.Tensor或字典）

        Returns:
            推理结果
        """
        # 如果输入是torch.Tensor，转换为列表
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy().tolist()

        # 构造请求体
        payload = {
            "model": self.model_name,
            "inputs": inputs
        }

        # 如果是OpenAI格式（用于LLM）
        if isinstance(inputs, dict) and 'messages' in inputs:
            payload = inputs

        try:
            response = self.session.post(
                f"{self.endpoint_url}/v1/inferences",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"华为云NPU推理失败: {e}")

    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       temperature: float = 0.7,
                       max_tokens: int = 1024) -> str:
        """
        Chat Completion API（OpenAI兼容）

        Args:
            messages: 对话消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            生成的回复
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        result = self.inference(payload)

        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise RuntimeError("无效的API响应格式")


class CloudNPULinear(nn.Module):
    """云端NPU加速的Linear层"""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 cloud_backend: CloudNPUBackend,
                 fallback_local: bool = True,
                 bias: bool = True):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            cloud_backend: 云端NPU后端
            fallback_local: 如果云端不可用，是否回退到本地CPU
            bias: 是否使用bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cloud_backend = cloud_backend
        self.fallback_local = fallback_local

        # 本地权重（用于fallback或初始化）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        # 统计信息
        self.stats = {
            'cloud_calls': 0,
            'local_calls': 0,
            'cloud_errors': 0
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 尝试使用云端NPU"""
        batch_size = x.shape[0]

        # 尝试云端推理
        if self.cloud_backend.is_available():
            try:
                # 准备输入
                inputs = {
                    'x': x.cpu().numpy().tolist(),
                    'weight': self.weight.detach().cpu().numpy().tolist(),
                    'bias': self.bias.detach().cpu().numpy().tolist() if self.bias is not None else None
                }

                # 调用云端API
                result = self.cloud_backend.inference(inputs)

                # 解析结果
                if 'output' in result:
                    output = torch.tensor(result['output'], device=x.device)
                    self.stats['cloud_calls'] += 1
                    return output
                else:
                    raise ValueError("云端返回无效格式")

            except Exception as e:
                self.stats['cloud_errors'] += 1
                warnings.warn(f"云端NPU推理失败: {e}，回退到本地计算")
                if not self.fallback_local:
                    raise

        # Fallback到本地计算
        self.stats['local_calls'] += 1
        return nn.functional.linear(x, self.weight, self.bias)

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        total = self.stats['cloud_calls'] + self.stats['local_calls']
        return {
            **self.stats,
            'total_calls': total,
            'cloud_ratio': self.stats['cloud_calls'] / total if total > 0 else 0
        }


class CloudNPUManager:
    """云端NPU管理器"""

    def __init__(self):
        self.backends: Dict[str, CloudNPUBackend] = {}
        self._load_from_env()

    def _load_from_env(self):
        """从环境变量加载云端NPU配置"""
        # 华为云ModelArts
        if os.getenv('HUAWEI_CLOUD_API_KEY'):
            try:
                backend = HuaweiModelArtsNPU(
                    api_key=os.getenv('HUAWEI_CLOUD_API_KEY'),
                    endpoint_url=os.getenv('HUAWEI_CLOUD_ENDPOINT', 'https://modelarts.cn-north-4.myhuaweicloud.com'),
                    model_name=os.getenv('HUAWEI_CLOUD_MODEL', 'deepseek-r1'),
                    region=os.getenv('HUAWEI_CLOUD_REGION', 'cn-north-4')
                )
                self.backends['huawei'] = backend
                print("✅ 已加载华为云ModelArts NPU")
            except Exception as e:
                warnings.warn(f"华为云NPU加载失败: {e}")

    def add_backend(self, name: str, backend: CloudNPUBackend):
        """添加云端NPU后端"""
        self.backends[name] = backend
        print(f"✅ 已添加云端NPU后端: {name}")

    def get_backend(self, name: Optional[str] = None) -> Optional[CloudNPUBackend]:
        """
        获取云端NPU后端

        Args:
            name: 后端名称，None则返回第一个可用的
        """
        if name:
            return self.backends.get(name)

        # 返回第一个可用的后端
        for backend in self.backends.values():
            if backend.is_available():
                return backend

        return None

    def list_backends(self) -> List[str]:
        """列出所有云端NPU后端"""
        return list(self.backends.keys())

    def is_any_available(self) -> bool:
        """检查是否有任何云端NPU可用"""
        return any(b.is_available() for b in self.backends.values())

    def cleanup(self):
        """清理所有连接"""
        for backend in self.backends.values():
            backend.close()


# 全局云端NPU管理器
_cloud_npu_manager = None


def get_cloud_npu_manager() -> CloudNPUManager:
    """获取全局云端NPU管理器"""
    global _cloud_npu_manager
    if _cloud_npu_manager is None:
        _cloud_npu_manager = CloudNPUManager()
    return _cloud_npu_manager


def enable_cloud_npu(provider: str = 'auto', **kwargs):
    """
    启用云端NPU

    Args:
        provider: 云服务商 ('huawei', 'auto')
        **kwargs: 云服务商特定参数

    Example:
        >>> enable_cloud_npu('huawei',
        ...                  api_key='your-key',
        ...                  endpoint_url='https://...')
    """
    manager = get_cloud_npu_manager()

    if provider == 'auto':
        if not manager.is_any_available():
            print("⚠️ 未找到可用的云端NPU")
            print("💡 提示：请设置环境变量：")
            print("   export HUAWEI_CLOUD_API_KEY=your-key")
            print("   export HUAWEI_CLOUD_ENDPOINT=https://...")
        return

    # 手动添加后端
    if provider == 'huawei':
        backend = HuaweiModelArtsNPU(**kwargs)
        manager.add_backend('huawei', backend)


__all__ = [
    'CloudNPUBackend',
    'HuaweiModelArtsNPU',
    'CloudNPULinear',
    'CloudNPUManager',
    'get_cloud_npu_manager',
    'enable_cloud_npu',
]
