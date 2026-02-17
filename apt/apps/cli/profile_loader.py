#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profile Loader for APT Model CLI

加载预定义配置文件 (lite/standard/pro/full)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ProfileLoader:
    """配置文件加载器"""

    def __init__(self):
        """初始化配置加载器"""
        self.profiles_dir = Path(__file__).parent.parent.parent.parent / "profiles"
        self.available_profiles = ['lite', 'standard', 'pro', 'full']

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        加载指定的配置文件

        Args:
            profile_name: 配置文件名 (lite/standard/pro/full)

        Returns:
            配置字典

        Raises:
            ValueError: 如果配置文件不存在
        """
        if profile_name not in self.available_profiles:
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Available profiles: {', '.join(self.available_profiles)}"
            )

        profile_path = self.profiles_dir / f"{profile_name}.yaml"

        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile file not found: {profile_path}"
            )

        with open(profile_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def merge_profile_with_args(
        self,
        args_dict: Dict[str, Any],
        profile_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        将配置文件与命令行参数合并（命令行参数优先）

        Args:
            args_dict: 命令行参数字典
            profile_name: 配置文件名（可选）

        Returns:
            合并后的配置字典
        """
        # 如果没有指定 profile，直接返回命令行参数
        if not profile_name:
            return args_dict

        # 加载 profile 配置
        profile_config = self.load_profile(profile_name)

        # 合并配置（命令行参数优先）
        merged_config = profile_config.copy()

        # 遍历命令行参数，覆盖 profile 中的值
        for key, value in args_dict.items():
            # 如果命令行参数不是默认值，则覆盖 profile 中的值
            # 这里需要特殊处理一些参数
            if value is not None:
                if key == 'action' and value is None:
                    continue  # 保留 profile 中的 action
                merged_config[key] = value

        return merged_config

    def list_profiles(self) -> Dict[str, str]:
        """
        列出所有可用的配置文件及其描述

        Returns:
            配置文件名到描述的映射
        """
        descriptions = {
            'lite': '轻量级配置 - 最小资源占用，快速启动',
            'standard': '标准配置 - 平衡性能和资源',
            'pro': '专业配置 - 高性能训练和评估',
            'full': '完整配置 - 所有功能启用，最大性能',
        }

        result = {}
        for profile_name in self.available_profiles:
            profile_path = self.profiles_dir / f"{profile_name}.yaml"
            if profile_path.exists():
                result[profile_name] = descriptions.get(
                    profile_name,
                    f"Profile: {profile_name}"
                )

        return result

    def get_profile_info(self, profile_name: str) -> Dict[str, Any]:
        """
        获取配置文件的详细信息

        Args:
            profile_name: 配置文件名

        Returns:
            配置文件信息（包含模块、插件等）
        """
        config = self.load_profile(profile_name)

        info = {
            'name': profile_name,
            'layers': config.get('layers', []),
            'plugins': config.get('plugins', []),
            'features': config.get('features', {}),
            'optimization': config.get('optimization', {}),
        }

        return info


# 全局实例
profile_loader = ProfileLoader()


def apply_profile_to_args(args):
    """
    将 profile 应用到命令行参数

    Args:
        args: argparse.Namespace 对象

    Returns:
        更新后的 args
    """
    if not hasattr(args, 'profile') or not args.profile:
        return args

    # 转换为字典
    args_dict = vars(args)

    # 合并配置
    merged_config = profile_loader.merge_profile_with_args(
        args_dict,
        args.profile
    )

    # 更新 args
    for key, value in merged_config.items():
        setattr(args, key, value)

    return args


if __name__ == "__main__":
    # 测试
    loader = ProfileLoader()

    print("Available Profiles:")
    for name, desc in loader.list_profiles().items():
        print(f"  - {name}: {desc}")

    print("\nLite Profile Info:")
    info = loader.get_profile_info('lite')
    print(f"  Layers: {info['layers']}")
    print(f"  Plugins: {info['plugins']}")
