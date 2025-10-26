#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Codec Registry

全局Codec注册表，管理所有已注册的语言编解码器。

功能：
- 注册/注销codec
- 按名称/语言查找codec
- 列出所有codec
- 语言路由
"""

from typing import Dict, List, Optional, Callable
import logging

from apt.core.codecs.api import Codec

logger = logging.getLogger(__name__)


class CodecRegistry:
    """
    Codec注册表

    全局单例，管理所有codec的注册和查找。

    使用方式:
        # 注册codec
        registry.register(my_codec)

        # 查找codec
        codec = registry.get_codec("zh_char")

        # 按语言查找
        codec = registry.get_codec_for_language("zh")

        # 列出所有codec
        codecs = registry.list_codecs()
    """

    def __init__(self):
        """初始化注册表"""
        # 存储: {codec_name: Codec实例}
        self._codecs: Dict[str, Codec] = {}

        # 语言映射: {language_code: [codec_names]}
        self._lang_to_codecs: Dict[str, List[str]] = {}

        # Codec工厂: {codec_name: factory_function}
        self._codec_factories: Dict[str, Callable] = {}

        logger.info("CodecRegistry initialized")

    def register(
        self,
        codec: Optional[Codec] = None,
        factory: Optional[Callable] = None,
        name: Optional[str] = None,
        override: bool = False
    ):
        """
        注册codec

        有两种注册方式:
        1. 直接注册实例: register(codec_instance)
        2. 注册工厂函数: register(factory=my_factory, name="my_codec")

        参数:
            codec: Codec实例（方式1）
            factory: Codec工厂函数（方式2）
            name: Codec名称（仅方式2需要）
            override: 是否覆盖已存在的codec

        示例:
            >>> # 方式1：直接注册
            >>> registry.register(ZhCharCodec())
            >>>
            >>> # 方式2：注册工厂
            >>> def create_zh_char():
            >>>     return ZhCharCodec()
            >>> registry.register(factory=create_zh_char, name="zh_char")
        """
        if codec is not None:
            # 方式1：直接注册实例
            codec_name = codec.name

            if codec_name in self._codecs and not override:
                raise ValueError(
                    f"Codec '{codec_name}' already registered. "
                    f"Use override=True to replace."
                )

            self._codecs[codec_name] = codec

            # 更新语言映射
            for lang in codec.langs:
                lang = lang.lower()
                if lang not in self._lang_to_codecs:
                    self._lang_to_codecs[lang] = []
                if codec_name not in self._lang_to_codecs[lang]:
                    self._lang_to_codecs[lang].append(codec_name)

            logger.info(f"Registered codec: {codec_name} (langs: {codec.langs})")

        elif factory is not None and name is not None:
            # 方式2：注册工厂函数
            if name in self._codec_factories and not override:
                raise ValueError(
                    f"Codec factory '{name}' already registered. "
                    f"Use override=True to replace."
                )

            self._codec_factories[name] = factory
            logger.info(f"Registered codec factory: {name}")

        else:
            raise ValueError(
                "Must provide either 'codec' instance or both 'factory' and 'name'"
            )

    def unregister(self, name: str):
        """
        注销codec

        参数:
            name: Codec名称
        """
        # 从codecs移除
        if name in self._codecs:
            codec = self._codecs.pop(name)

            # 从语言映射移除
            for lang in codec.langs:
                lang = lang.lower()
                if lang in self._lang_to_codecs:
                    if name in self._lang_to_codecs[lang]:
                        self._lang_to_codecs[lang].remove(name)

            logger.info(f"Unregistered codec: {name}")

        # 从工厂移除
        if name in self._codec_factories:
            self._codec_factories.pop(name)
            logger.info(f"Unregistered codec factory: {name}")

    def get_codec(self, name: str, lazy_load: bool = True) -> Optional[Codec]:
        """
        按名称获取codec

        参数:
            name: Codec名称
            lazy_load: 是否延迟加载（使用工厂）

        返回:
            Codec实例，未找到返回None

        示例:
            >>> codec = registry.get_codec("zh_char")
        """
        # 尝试从已加载的codec中获取
        if name in self._codecs:
            return self._codecs[name]

        # 尝试使用工厂延迟加载
        if lazy_load and name in self._codec_factories:
            try:
                codec = self._codec_factories[name]()
                # 注册实例（缓存）
                self._codecs[name] = codec
                logger.info(f"Lazy-loaded codec: {name}")
                return codec
            except Exception as e:
                logger.error(f"Failed to lazy-load codec '{name}': {e}")
                return None

        logger.warning(f"Codec '{name}' not found")
        return None

    def get_codec_for_language(
        self,
        language: str,
        prefer: Optional[str] = None,
        fallback: Optional[str] = None
    ) -> Optional[Codec]:
        """
        根据语言代码获取codec

        参数:
            language: 语言代码 (如 'zh', 'ja', 'en')
            prefer: 首选codec名称
            fallback: 回退codec名称

        返回:
            Codec实例，未找到返回None

        示例:
            >>> # 获取中文codec（优先jieba）
            >>> codec = registry.get_codec_for_language("zh", prefer="zh_jieba")
            >>>
            >>> # 获取中文codec（找不到就用fallback）
            >>> codec = registry.get_codec_for_language("zh", fallback="unicode_basic")
        """
        language = language.lower()

        # 检查首选codec
        if prefer:
            codec = self.get_codec(prefer)
            if codec and language in [l.lower() for l in codec.langs]:
                return codec

        # 查找支持此语言的codec
        if language in self._lang_to_codecs:
            codec_names = self._lang_to_codecs[language]
            if codec_names:
                # 返回第一个可用的
                for codec_name in codec_names:
                    codec = self.get_codec(codec_name)
                    if codec:
                        return codec

        # 使用fallback
        if fallback:
            codec = self.get_codec(fallback)
            if codec:
                logger.warning(
                    f"No codec found for language '{language}', "
                    f"using fallback '{fallback}'"
                )
                return codec

        logger.warning(f"No codec found for language '{language}'")
        return None

    def has_codec(self, name: str) -> bool:
        """
        检查codec是否已注册

        参数:
            name: Codec名称

        返回:
            是否存在
        """
        return name in self._codecs or name in self._codec_factories

    def list_codecs(self, language: Optional[str] = None) -> List[str]:
        """
        列出所有已注册的codec名称

        参数:
            language: 可选，按语言过滤

        返回:
            Codec名称列表

        示例:
            >>> # 列出所有codec
            >>> all_codecs = registry.list_codecs()
            >>>
            >>> # 列出支持中文的codec
            >>> zh_codecs = registry.list_codecs(language="zh")
        """
        if language:
            language = language.lower()
            return self._lang_to_codecs.get(language, [])
        else:
            # 合并已加载的和工厂
            all_codecs = set(self._codecs.keys()) | set(self._codec_factories.keys())
            return list(all_codecs)

    def list_languages(self) -> List[str]:
        """
        列出所有支持的语言代码

        返回:
            语言代码列表
        """
        return list(self._lang_to_codecs.keys())

    def clear(self):
        """清空注册表（主要用于测试）"""
        self._codecs.clear()
        self._lang_to_codecs.clear()
        self._codec_factories.clear()
        logger.warning("CodecRegistry cleared")

    def __repr__(self):
        return (
            f"CodecRegistry("
            f"codecs={len(self._codecs)}, "
            f"factories={len(self._codec_factories)}, "
            f"languages={len(self._lang_to_codecs)})"
        )


# ============================================================================
# 全局单例
# ============================================================================

# 全局codec注册表实例
codec_registry = CodecRegistry()


# ============================================================================
# 便捷函数
# ============================================================================

def register_codec(
    codec: Optional[Codec] = None,
    factory: Optional[Callable] = None,
    name: Optional[str] = None,
    override: bool = False
):
    """
    注册codec到全局注册表

    参数:
        codec: Codec实例
        factory: Codec工厂函数
        name: Codec名称（使用factory时必需）
        override: 是否覆盖已存在的codec
    """
    codec_registry.register(codec, factory, name, override)


def get_codec(name: str) -> Optional[Codec]:
    """
    从全局注册表获取codec

    参数:
        name: Codec名称

    返回:
        Codec实例或None
    """
    return codec_registry.get_codec(name)


def get_codec_for_language(
    language: str,
    prefer: Optional[str] = None,
    fallback: Optional[str] = None
) -> Optional[Codec]:
    """
    根据语言获取codec

    参数:
        language: 语言代码
        prefer: 首选codec
        fallback: 回退codec

    返回:
        Codec实例或None
    """
    return codec_registry.get_codec_for_language(language, prefer, fallback)


def list_codecs(language: Optional[str] = None) -> List[str]:
    """
    列出所有codec

    参数:
        language: 可选，按语言过滤

    返回:
        Codec名称列表
    """
    return codec_registry.list_codecs(language)


def list_languages() -> List[str]:
    """
    列出所有支持的语言

    返回:
        语言代码列表
    """
    return codec_registry.list_languages()
