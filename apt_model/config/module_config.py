#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块配置

统一管理所有可选模块的配置
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class GraphRAGConfig:
    """
    GraphRAG配置

    增强的知识图谱系统配置
    """
    # 基础配置
    enabled: bool = False
    max_dimension: int = 2  # 泛图最大维度 (0=点, 1=边, 2=面, 3=体)
    enable_brain: bool = True  # 启用图脑动力学
    enable_spectral: bool = True  # 启用谱分析

    # API集成 (用于自动构建知识图谱)
    use_api: bool = False
    api_provider: Optional[str] = None  # 'openai', 'siliconflow', etc.
    api_key: Optional[str] = None
    api_model: Optional[str] = None

    # 查询配置
    default_query_mode: str = "hybrid"  # 'spectral', 'brain', 'hybrid'
    default_top_k: int = 10

    # 存储配置
    save_dir: str = "./graph_rag_data"
    auto_save: bool = True
    auto_save_interval: int = 1000  # 每N个三元组保存一次

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'max_dimension': self.max_dimension,
            'enable_brain': self.enable_brain,
            'enable_spectral': self.enable_spectral,
            'use_api': self.use_api,
            'api_provider': self.api_provider,
            'api_key': self.api_key,
            'api_model': self.api_model,
            'default_query_mode': self.default_query_mode,
            'default_top_k': self.default_top_k,
            'save_dir': self.save_dir,
            'auto_save': self.auto_save,
            'auto_save_interval': self.auto_save_interval,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class SOSAConfig:
    """
    SOSA训练监控配置

    智能训练监控与自动纠错配置
    """
    # 基础配置
    enabled: bool = False
    window_seconds: float = 10.0  # SOSA时间窗口大小(秒)
    auto_fix: bool = True  # 启用自动修复
    max_fixes_per_error: int = 3  # 每种错误最大修复次数
    exploration_weight: float = 0.5  # SOSA探索权重 [0,1]

    # 检查点配置
    checkpoint_dir: str = './checkpoints'
    save_best: bool = True  # 保存最佳模型
    save_interval: int = 1000  # 保存间隔 (步数)

    # 异常检测阈值
    nan_check: bool = True  # 检测NaN
    grad_explosion_threshold: float = 100.0  # 梯度爆炸阈值
    grad_vanishing_threshold: float = 1e-7  # 梯度消失阈值
    loss_diverge_factor: float = 2.0  # Loss发散因子
    loss_oscillation_threshold: float = 0.5  # Loss震荡阈值
    loss_plateau_patience: int = 100  # Loss停滞耐心值

    # 修复策略
    lr_reduce_factor_nan: float = 0.1  # NaN时学习率降低因子
    lr_reduce_factor_diverge: float = 0.5  # 发散时学习率降低因子
    lr_increase_factor_vanish: float = 1.5  # 消失时学习率提高因子
    grad_clip_factor: float = 0.5  # 梯度裁剪强化因子

    # 报告配置
    report_interval: int = 100  # 报告间隔 (步数)
    verbose: bool = True  # 详细日志

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'window_seconds': self.window_seconds,
            'auto_fix': self.auto_fix,
            'max_fixes_per_error': self.max_fixes_per_error,
            'exploration_weight': self.exploration_weight,
            'checkpoint_dir': self.checkpoint_dir,
            'save_best': self.save_best,
            'save_interval': self.save_interval,
            'nan_check': self.nan_check,
            'grad_explosion_threshold': self.grad_explosion_threshold,
            'grad_vanishing_threshold': self.grad_vanishing_threshold,
            'loss_diverge_factor': self.loss_diverge_factor,
            'loss_oscillation_threshold': self.loss_oscillation_threshold,
            'loss_plateau_patience': self.loss_plateau_patience,
            'lr_reduce_factor_nan': self.lr_reduce_factor_nan,
            'lr_reduce_factor_diverge': self.lr_reduce_factor_diverge,
            'lr_increase_factor_vanish': self.lr_increase_factor_vanish,
            'grad_clip_factor': self.grad_clip_factor,
            'report_interval': self.report_interval,
            'verbose': self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class ModuleConfig:
    """
    模块配置总集

    管理所有可选模块的配置
    """
    graph_rag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    sosa: SOSAConfig = field(default_factory=SOSAConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'graph_rag': self.graph_rag.to_dict(),
            'sosa': self.sosa.to_dict(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建"""
        return cls(
            graph_rag=GraphRAGConfig.from_dict(config_dict.get('graph_rag', {})),
            sosa=SOSAConfig.from_dict(config_dict.get('sosa', {}))
        )

    def save_to_file(self, filepath: str):
        """保存到文件"""
        import json

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str):
        """从文件加载"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


# ==================== 预设配置 ====================

def get_default_config() -> ModuleConfig:
    """获取默认配置"""
    return ModuleConfig()


def get_full_config() -> ModuleConfig:
    """获取全功能配置 (所有模块启用)"""
    return ModuleConfig(
        graph_rag=GraphRAGConfig(
            enabled=True,
            max_dimension=2,
            enable_brain=True,
            enable_spectral=True
        ),
        sosa=SOSAConfig(
            enabled=True,
            auto_fix=True,
            window_seconds=10.0
        )
    )


def get_safe_config() -> ModuleConfig:
    """获取安全配置 (监控但不自动修复)"""
    return ModuleConfig(
        graph_rag=GraphRAGConfig(enabled=False),
        sosa=SOSAConfig(
            enabled=True,
            auto_fix=False,  # 只监控，不修复
            window_seconds=10.0
        )
    )


def get_lightweight_config() -> ModuleConfig:
    """获取轻量配置 (最小开销)"""
    return ModuleConfig(
        graph_rag=GraphRAGConfig(
            enabled=True,
            max_dimension=1,  # 只用边，不用面
            enable_brain=False,  # 禁用图脑
            enable_spectral=False  # 禁用谱分析
        ),
        sosa=SOSAConfig(
            enabled=True,
            auto_fix=True,
            window_seconds=30.0,  # 更大的窗口
            report_interval=1000  # 更少的报告
        )
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("【模块配置演示】\n")

    # 示例1: 默认配置
    print("=" * 60)
    print("[示例1] 默认配置 (所有模块禁用)")
    print("=" * 60)

    config = get_default_config()
    print(f"GraphRAG启用: {config.graph_rag.enabled}")
    print(f"SOSA启用: {config.sosa.enabled}")

    # 示例2: 全功能配置
    print("\n" + "=" * 60)
    print("[示例2] 全功能配置")
    print("=" * 60)

    config = get_full_config()
    print(f"GraphRAG启用: {config.graph_rag.enabled}")
    print(f"  维度: {config.graph_rag.max_dimension}")
    print(f"  图脑: {config.graph_rag.enable_brain}")
    print(f"SOSA启用: {config.sosa.enabled}")
    print(f"  自动修复: {config.sosa.auto_fix}")

    # 示例3: 保存和加载
    print("\n" + "=" * 60)
    print("[示例3] 保存和加载配置")
    print("=" * 60)

    config = get_full_config()
    config.save_to_file('./module_config.json')
    print("配置已保存到 module_config.json")

    loaded_config = ModuleConfig.load_from_file('./module_config.json')
    print(f"加载的配置 - GraphRAG启用: {loaded_config.graph_rag.enabled}")

    # 示例4: 从字典创建
    print("\n" + "=" * 60)
    print("[示例4] 从命令行参数创建配置")
    print("=" * 60)

    # 模拟命令行参数
    args_dict = {
        'graph_rag': {
            'enabled': True,
            'max_dimension': 2
        },
        'sosa': {
            'enabled': True,
            'auto_fix': True
        }
    }

    config = ModuleConfig.from_dict(args_dict)
    print(f"GraphRAG维度: {config.graph_rag.max_dimension}")
    print(f"SOSA自动修复: {config.sosa.auto_fix}")

    print("\n" + "=" * 60)
    print("[完成] 配置系统演示完毕")
    print("=" * 60)
