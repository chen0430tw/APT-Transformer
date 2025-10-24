#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Registry 示例代码
展示Provider注册表的核心实现和使用方式
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, List
import warnings


# ============================================================================
# Provider 基类
# ============================================================================

class Provider(ABC):
    """所有Provider的基类"""

    @abstractmethod
    def get_name(self) -> str:
        """返回Provider名称"""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """返回版本号"""
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置是否合法（子类可重写）"""
        return True

    def get_dependencies(self) -> List[str]:
        """返回依赖的其他Provider（可选）"""
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_name()}, v{self.get_version()})"


# ============================================================================
# Registry 核心实现
# ============================================================================

class Registry:
    """
    全局Provider注册表

    功能：
    1. 注册Provider实现
    2. 按需获取Provider实例（单例）
    3. 自动回退到默认实现
    4. 依赖检查
    5. 版本管理
    """

    def __init__(self):
        # 存储Provider类：{kind: {name: cls}}
        self._providers: Dict[str, Dict[str, Type[Provider]]] = {}

        # 存储Provider实例（单例）：{kind:name: instance}
        self._instances: Dict[str, Provider] = {}

        # 默认实现映射
        self._defaults: Dict[str, str] = {
            'attention': 'tva_default',
            'ffn': 'default',
            'router': 'topk_default',
            'align': 'bistate_default',
            'retrieval': 'none',
            'dataset': 'text_default',
            'encoder': 'transformer'
        }

        # 互斥规则：{provider_name: [incompatible_providers]}
        self._exclusions: Dict[str, List[str]] = {}

    def register(
        self,
        kind: str,
        name: str,
        provider_cls: Type[Provider],
        default: bool = False,
        excludes: Optional[List[str]] = None
    ):
        """
        注册Provider

        Args:
            kind: Provider种类 (attention/ffn/router/align...)
            name: 实现名称 (tva_default/flash_v2/linear_causal...)
            provider_cls: Provider类
            default: 是否设为该kind的默认实现
            excludes: 互斥的Provider列表
        """
        # 验证Provider类
        if not issubclass(provider_cls, Provider):
            raise TypeError(f"{provider_cls} 必须继承自 Provider")

        # 初始化kind
        if kind not in self._providers:
            self._providers[kind] = {}

        # 检查重复注册
        if name in self._providers[kind]:
            warnings.warn(f"⚠️  覆盖已存在的 {kind}:{name}")

        # 注册
        self._providers[kind][name] = provider_cls
        print(f"✅ 注册 {kind} Provider: {name} (v{provider_cls({}).get_version()})")

        # 设置默认
        if default:
            self._defaults[kind] = name
            print(f"   → 设为 {kind} 的默认实现")

        # 设置互斥
        if excludes:
            full_name = f"{kind}:{name}"
            self._exclusions[full_name] = [f"{kind}:{e}" for e in excludes]

    def get(
        self,
        kind: str,
        name: str,
        config: Optional[Dict] = None,
        fallback: bool = True
    ) -> Provider:
        """
        获取Provider实例（单例模式）

        Args:
            kind: Provider种类
            name: 实现名称
            config: 配置字典
            fallback: 是否回退到默认实现

        Returns:
            Provider实例
        """
        key = f"{kind}:{name}"

        # 检查缓存
        if key in self._instances:
            return self._instances[key]

        # 查找Provider类
        if kind not in self._providers or name not in self._providers[kind]:
            if fallback:
                # 回退到默认实现
                default_name = self._defaults.get(kind)
                if default_name and default_name in self._providers.get(kind, {}):
                    warnings.warn(
                        f"⚠️  {key} 未找到，回退到 {kind}:{default_name}",
                        UserWarning
                    )
                    name = default_name
                    key = f"{kind}:{name}"
                else:
                    raise ValueError(f"❌ Provider {key} 未注册且无默认实现")
            else:
                raise ValueError(f"❌ Provider {key} 未注册")

        # 创建实例
        provider_cls = self._providers[kind][name]
        config = config or {}

        try:
            instance = provider_cls(config)

            # 验证配置
            if not instance.validate_config(config):
                raise ValueError(f"配置验证失败: {config}")

            # 缓存实例
            self._instances[key] = instance

            return instance

        except Exception as e:
            if fallback and name != self._defaults.get(kind):
                # 创建失败，尝试回退
                warnings.warn(
                    f"⚠️  创建 {key} 失败 ({e})，回退到默认实现",
                    UserWarning
                )
                return self.get(kind, self._defaults[kind], config, fallback=False)
            else:
                raise RuntimeError(f"❌ 创建 Provider {key} 失败: {e}")

    def list_providers(self, kind: Optional[str] = None) -> Dict[str, List[str]]:
        """
        列出所有已注册的Provider

        Args:
            kind: 指定种类（可选）

        Returns:
            {kind: [names]} 字典
        """
        if kind:
            return {kind: list(self._providers.get(kind, {}).keys())}
        else:
            return {k: list(v.keys()) for k, v in self._providers.items()}

    def check_conflicts(self, providers: List[str]) -> Optional[str]:
        """
        检查Provider列表是否有冲突

        Args:
            providers: Provider列表 ["kind:name", ...]

        Returns:
            冲突描述（无冲突返回None）
        """
        for p1 in providers:
            if p1 in self._exclusions:
                for p2 in self._exclusions[p1]:
                    if p2 in providers:
                        return f"❌ {p1} 与 {p2} 互斥"
        return None

    def get_info(self, kind: str, name: str) -> Dict[str, Any]:
        """获取Provider详细信息"""
        key = f"{kind}:{name}"
        if kind not in self._providers or name not in self._providers[kind]:
            raise ValueError(f"Provider {key} 不存在")

        provider_cls = self._providers[kind][name]
        temp_instance = provider_cls({})

        return {
            'name': name,
            'kind': kind,
            'version': temp_instance.get_version(),
            'class': provider_cls.__name__,
            'dependencies': temp_instance.get_dependencies(),
            'is_default': self._defaults.get(kind) == name,
            'excludes': self._exclusions.get(key, [])
        }


# ============================================================================
# 全局单例
# ============================================================================

registry = Registry()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("APT Core Registry 示例")
    print("="*60)

    # 定义一个简单的Attention Provider示例
    class TVAAttention(Provider):
        def __init__(self, config):
            self.r = config.get('r', 4)
            self.tau = config.get('tau', 0.18)

        def get_name(self):
            return "tva_default"

        def get_version(self):
            return "1.0.0"

        def validate_config(self, config):
            return 0 < config.get('r', 4) < 10

    class FlashAttention(Provider):
        def __init__(self, config):
            try:
                # 模拟检查flash-attn是否可用
                # import flash_attn
                pass
            except ImportError:
                raise ImportError("flash-attn库未安装")

        def get_name(self):
            return "flash_v2"

        def get_version(self):
            return "2.0.0"

    # 注册Provider
    print("\n1. 注册Provider:")
    registry.register('attention', 'tva_default', TVAAttention, default=True)
    registry.register('attention', 'flash_v2', FlashAttention)

    # 列出所有Provider
    print("\n2. 列出所有Provider:")
    providers = registry.list_providers()
    for kind, names in providers.items():
        print(f"   {kind}: {names}")

    # 获取Provider实例
    print("\n3. 获取Provider实例:")
    tva = registry.get('attention', 'tva_default', {'r': 4, 'tau': 0.18})
    print(f"   获取到: {tva}")

    # 测试回退机制
    print("\n4. 测试回退机制:")
    try:
        # flash_v2会失败（ImportError），自动回退到tva_default
        flash = registry.get('attention', 'flash_v2', fallback=True)
        print(f"   获取到: {flash}")
    except Exception as e:
        print(f"   回退: {e}")

    # 获取Provider信息
    print("\n5. 获取Provider详细信息:")
    info = registry.get_info('attention', 'tva_default')
    for k, v in info.items():
        print(f"   {k}: {v}")

    print("\n" + "="*60)
    print("✅ 示例完成")
    print("="*60)
