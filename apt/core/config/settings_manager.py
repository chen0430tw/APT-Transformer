#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局配置管理器 - 管理APT Model的全局设置
支持从配置文件、环境变量读取配置
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class SettingsManager:
    """全局配置管理器"""

    # 配置文件路径
    CONFIG_DIR = Path(__file__).parent
    CONFIG_FILE = CONFIG_DIR / "settings.yaml"

    # 单例实例
    _instance = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if self._initialized:
            return

        self._config = {}
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """加载配置文件"""
        if self.CONFIG_FILE.exists():
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # 如果配置文件不存在，使用默认配置
            self._config = self._get_default_config()
            self.save_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'debug': {
                'enabled': False,
                'log_level': 'INFO',
                'profile_memory': False,
                'check_gradients': False,
                'save_debug_logs': True,
            },
            'training': {
                'default_epochs': 20,
                'default_batch_size': 8,
                'default_learning_rate': 3e-5,
                'checkpoint_auto_save': True,
            },
            'tokenizer': {
                'auto_detect_language': True,
                'cache_tokenizer': True,
            },
            'logging': {
                'colored_output': True,
                'log_to_file': True,
                'log_directory': 'apt_model/log',
            },
            'hardware': {
                'auto_gpu': True,
                'mixed_precision': False,
            },
            'misc': {
                'language': 'zh_CN',
                'auto_backup': False,
            }
        }

    def save_config(self):
        """保存配置到文件"""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号路径

        例如: get('debug.enabled') 或 get('training.default_epochs')

        优先级: 环境变量 > 配置文件 > 默认值
        """
        # 检查环境变量（转换为大写并用下划线）
        env_key = f"APT_{key.upper().replace('.', '_')}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            # 尝试转换类型
            return self._parse_env_value(env_value)

        # 从配置文件读取
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 布尔值
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        if value.lower() in ('false', '0', 'no', 'off'):
            return False

        # 数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # 字符串
        return value

    def set(self, key: str, value: Any):
        """
        设置配置值，支持点号路径

        例如: set('debug.enabled', True)
        """
        keys = key.split('.')
        config = self._config

        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value

        # 保存到文件
        self.save_config()

    def get_debug_enabled(self) -> bool:
        """快捷方法：获取Debug模式状态"""
        return self.get('debug.enabled', False)

    def set_debug_enabled(self, enabled: bool):
        """快捷方法：设置Debug模式状态"""
        self.set('debug.enabled', enabled)
        if enabled:
            self.set('debug.log_level', 'DEBUG')
        else:
            self.set('debug.log_level', 'INFO')

    def get_log_level(self) -> str:
        """快捷方法：获取日志级别"""
        if self.get_debug_enabled():
            return 'DEBUG'
        return self.get('debug.log_level', 'INFO')

    def get_all_config(self) -> Dict:
        """获取所有配置"""
        return self._config.copy()

    def reset_to_default(self):
        """重置为默认配置"""
        self._config = self._get_default_config()
        self.save_config()

    def __repr__(self):
        """字符串表示"""
        return f"SettingsManager(config_file='{self.CONFIG_FILE}')"


# 全局单例实例
settings = SettingsManager()


# 便捷函数
def get_setting(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return settings.get(key, default)


def set_setting(key: str, value: Any):
    """设置配置值的便捷函数"""
    settings.set(key, value)


def is_debug_enabled() -> bool:
    """检查是否启用Debug模式"""
    return settings.get_debug_enabled()


def enable_debug():
    """启用Debug模式"""
    settings.set_debug_enabled(True)
    print("✓ Debug模式已启用")
    print(f"  日志级别: DEBUG")
    print(f"  配置文件: {settings.CONFIG_FILE}")


def disable_debug():
    """禁用Debug模式"""
    settings.set_debug_enabled(False)
    print("✓ Debug模式已禁用")
    print(f"  日志级别: INFO")
    print(f"  配置文件: {settings.CONFIG_FILE}")


if __name__ == "__main__":
    # 测试代码
    print("Settings Manager Test")
    print("=" * 60)

    # 测试获取配置
    print(f"Debug enabled: {settings.get_debug_enabled()}")
    print(f"Log level: {settings.get_log_level()}")
    print(f"Default epochs: {settings.get('training.default_epochs')}")

    # 测试设置配置
    print("\n启用Debug模式...")
    enable_debug()
    print(f"Debug enabled: {settings.get_debug_enabled()}")
    print(f"Log level: {settings.get_log_level()}")

    print("\n禁用Debug模式...")
    disable_debug()
    print(f"Debug enabled: {settings.get_debug_enabled()}")
    print(f"Log level: {settings.get_log_level()}")
