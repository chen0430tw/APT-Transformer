#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Language management module for the APT Model training tool.
Provides multilingual support with built-in Chinese and English language packs.
Allows users to load custom language packs.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Callable

class LanguageManager:
    """
    Language manager
    Provides multilingual support, with built-in support for Chinese and English
    Users can customize language packs and load them
    """
    def __init__(self, language: str = "zh_CN", custom_lang_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize language manager

        Args:
            language: Language code, default is "zh_CN" (Simplified Chinese)
            custom_lang_path: Custom language pack path
            logger: Logger instance
        """
        self.language = language
        self.custom_lang_path = custom_lang_path
        self.logger = logger
        
        # Built-in language packs
        self.builtin_languages = {
            "zh_CN": self._get_chinese_messages(),
            "en_US": self._get_english_messages()
        }
        
        # Current language pack
        self.messages: Dict[str, Any] = {}
        
        # Load language pack
        self.load_language(language, custom_lang_path)
    
    def load_language(self, language: str, custom_lang_path: Optional[str] = None) -> bool:
        """
        Load language pack

        Args:
            language: Language code
            custom_lang_path: Custom language pack path

        Returns:
            bool: Success or failure
        """
        self.language = language
        
        # Try to load custom language pack
        if custom_lang_path:
            try:
                with open(custom_lang_path, 'r', encoding='utf-8') as f:
                    custom_lang = json.load(f)
                    # Update language pack
                    self.messages = custom_lang
                    
                    if self.logger:
                        self.logger.info(f"Loaded custom language pack from {custom_lang_path}")
                    return True
            except Exception as e:
                error_msg = f"Error loading custom language pack: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(error_msg)
        
        # Try to load from built-in language packs
        if language in self.builtin_languages:
            self.messages = self.builtin_languages[language]
            
            if self.logger:
                self.logger.info(f"Loaded built-in language pack: {language}")
            return True
        
        # If specified language does not exist, use default language (Chinese)
        warning_msg = f"Warning: Language '{language}' is not available, using default language (Chinese)"
        if self.logger:
            self.logger.warning(warning_msg)
        else:
            print(warning_msg)
            
        self.language = "zh_CN"
        self.messages = self.builtin_languages["zh_CN"]
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get language message

        Args:
            key: Message key, supports hierarchical keys like "menu.file.open"
            default: Default value if key not found

        Returns:
            Message content
        """
        # Hierarchical key support, e.g., "menu.file.open"
        if "." in key:
            parts = key.split(".")
            current = self.messages
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default if default is not None else key
            return current
        
        return self.messages.get(key, default if default is not None else key)
    
    def get_formatter(self) -> Callable:
        """
        Get a formatter function for easy message formatting

        Returns:
            Callable: A formatter function
        """
        return lambda key, *params: self.get(key).format(*params) if params else self.get(key)
    
    def list_available_languages(self) -> Dict[str, str]:
        """
        List all available languages

        Returns:
            Dict[str, str]: Dictionary mapping language codes to names
        """
        return {
            "zh_CN": "简体中文 (Simplified Chinese)",
            "en_US": "English (United States)"
        }
    
    def get_all_languages(self):
        """
        Get all available languages

        Returns:
            dict: {language code: language name}
        """
        return {
            "zh_CN": "简体中文",
            "en_US": "English (US)"
        }
    
    def get_message_for_language(self, key, language, default=None):
        """
        Get a message for a specific language

        Args:
            key: Key name
            language: Language code
            default: Default value

        Returns:
            str: Language message
        """
        if language in self.builtin_languages:
            messages = self.builtin_languages[language]
            
            # Hierarchical key support
            if "." in key:
                parts = key.split(".")
                current = messages
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return default if default is not None else key
                return current
                
            return messages.get(key, default if default is not None else key)
        
        return default if default is not None else key

    def add_custom_messages(self, messages_dict):
        """
        Add custom messages to the current language pack

        Args:
            messages_dict: Message dictionary

        Returns:
            None
        """
        def update_nested_dict(d, u):
            """Recursively update nested dictionary"""
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v
        
        update_nested_dict(self.messages, messages_dict)
        if self.logger:
            self.logger.info("Custom messages added")
    
    def save_language_file(self, file_path):
        """
        Save current language pack to file

        Args:
            file_path: File path

        Returns:
            bool: Success or failure
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=4)
            if self.logger:
                self.logger.info(f"Language pack saved to: {file_path}")
            return True
        except Exception as e:
            error_msg = f"Error saving language pack: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            return False
    
    def _get_chinese_messages(self) -> Dict[str, Any]:
        """Get Chinese language pack"""
        return {
            # General messages
            "app_name": "APT模型训练工具",
            "welcome": "欢迎使用APT模型训练工具!",
            "language": "语言",
            "version": "版本",
            "author": "作者",
            "license": "许可证",
            
            # Errors and warnings
            "error": "错误",
            "warning": "警告",
            "info": "信息",
            "success": "成功",
            "failed": "失败",
            
            # Training related messages
            "training": {
                "start": "开始训练模型...",
                "completed": "训练完成！",
                "epoch": "轮次",
                "batch": "批次",
                "loss": "损失",
                "lr": "学习率",
                "progress": "进度",
                "early_stopping": "早停: {0} 轮没有改善，停止训练",
                "saving_model": "正在保存模型...",
                "loading_model": "正在加载模型...",
                "best_model": "发现新的最佳模型，已保存到 {0}",
                "best_quality": "发现新的最佳质量模型 ({0}分)，已保存",
                "total_params": "模型总参数: {0}",
                "device": "使用设备: {0}",
                "dataset_size": "数据集大小: {0} 样本",
                "with_reasoning": "使用推理能力训练",
                "with_distillation": "使用知识蒸馏训练",
                "with_custom_data": "使用自定义数据训练",
                "time_estimate": "预计训练时间: {0}",
                "time_remaining": "剩余时间: {0}",
                "training_speed": "训练速度: {0} 样本/秒",
                "validation_start": "开始验证...",
                "validation_complete": "验证完成，评估分数: {0:.4f}"
            },
            
            # Generation and evaluation related messages
            "generation": {
                "prompt": "提示",
                "response": "回应",
                "quality": "质量评分: {0}/100 - {1}",
                "test_generation": "测试生成",
                "amber_not_enough": "安柏：训练...还不够...",
                "amber_good": "安柏：训练完成得不错！",
                "temp": "温度",
                "top_p": "Top-P",
                "max_length": "最大长度",
                "generating": "正在生成...",
                "generation_time": "生成时间: {0:.2f}秒",
                "input_too_long": "输入过长，已被截断到 {0} 个标记",
                "batch_generation": "批量生成中，已完成: {0}/{1}"
            },
            
            # Amber's messages
            "amber": {
                "training": "安柏：一起来训练吧！",
                "not_enough": "安柏：训练...还不够...",
                "good": "安柏：训练完成得不错！",
                "loading": "安柏：正在准备...",
                "thinking": "安柏：让我想想...",
                "excited": "安柏：太棒了！",
                "confused": "安柏：嗯...这个有点复杂...",
                "evaluation": "安柏：让我来评价一下...",
                "progress": "安柏：还在进行中...",
                "farewell": "安柏：再见，下次见！"
            },
            
            # Hardware detection messages
            "hardware": {
                "gpu_detected": "检测到 {0} 个GPU: {1}",
                "no_gpu": "未检测到GPU，将使用CPU训练（速度可能很慢）",
                "insufficient_vram": "杂—鱼~ 杂—鱼~，你的显卡看起来跟你一样什么都不行呢，果然是个垃圾♡\n模型需要至少 {0:.1f}GB 显存，但你的显卡只有 {1:.1f}GB",
                "sufficient_vram": "显存充足，可以训练更大的模型",
                "cpu_cores": "CPU核心数: {0}",
                "ram": "系统内存: {0:.1f}GB",
                "disk_space": "可用磁盘空间: {0:.1f}GB",
                "checking": "正在检查硬件兼容性...",
                "compatible": "硬件兼容性检查通过",
                "incompatible": "硬件兼容性检查失败: {0}"
            },
            
            # Command-line interface messages
            "cli": {
                "description": "自生成变换器(APT)模型训练和测试工具",
                "epilog": "示例:\n  python apt.py train                    - 使用默认参数训练模型\n  python apt.py train --epochs 10        - 训练10轮\n  python apt.py train --force-cpu        - 强制使用CPU训练\n  python apt.py test                     - 测试模型\n  python apt.py chat                     - 与模型交互对话\n  python apt.py chat --temperature 0.9   - 使用更高温度与模型对话\n  python apt.py train --language en_US   - 使用英文界面训练模型",
                "help": "显示此帮助信息并退出",
                "language_help": "使用的语言 (默认: zh_CN)",
                "language_file_help": "自定义语言文件路径",
                "action_help": "执行动作",
                "epochs_help": "训练轮数 (默认: 20)",
                "batch_size_help": "训练批次大小 (默认: 8)",
                "learning_rate_help": "模型学习率 (默认: 3e-5)",
                "save_path_help": "模型保存路径 (默认: \"apt_model\")",
                "force_cpu_help": "强制使用CPU进行计算，即使有GPU可用",
                "temperature_help": "生成时的温度参数，控制随机性 (默认: 0.7)",
                "top_p_help": "生成时的top-p参数，控制输出多样性 (默认: 0.9)",
                "max_length_help": "生成的最大长度 (默认: 50)",
                "data_path_help": "外部训练数据文件路径",
                "max_samples_help": "从外部数据中使用的最大样本数 (默认: 全部)",
                "cache_dir_help": "缓存目录路径 (默认: ~/.apt_cache)",
                "clean_days_help": "清理多少天前的缓存 (默认: 30)",
                "verbose_help": "显示详细日志信息",
                "seed_help": "随机种子 (默认: 42)",
                "warning_message": "警告: {0}",
                "error_message": "错误: {0}",
                "info_message": "信息: {0}",
                "success_message": "成功: {0}"
            },
            
            # Data processing messages
            "data": {
                "loading": "正在加载数据...",
                "loaded": "已加载 {0} 条数据",
                "processing": "正在处理数据...",
                "processed": "已处理 {0} 条数据",
                "saving": "正在保存数据...",
                "saved": "已保存数据到 {0}",
                "error": "数据处理错误: {0}",
                "empty": "数据为空",
                "invalid": "无效的数据格式",
                "too_large": "数据过大，超过最大限制",
                "too_small": "数据过小，低于最小要求",
                "example": "数据样例: {0}",
                "stats": "数据统计: {0}",
                "preview": "数据预览:\n{0}",
                "shuffle": "正在打乱数据...",
                "split": "正在分割数据集...",
                "train_size": "训练集大小: {0}",
                "val_size": "验证集大小: {0}",
                "test_size": "测试集大小: {0}",
                "external_data": "外部数据",
                "internal_data": "内部数据",
                "combined_data": "合并数据",
                "filtering": "正在过滤数据...",
                "filtered": "已过滤 {0} 条数据",
                "sample_count": "数据样本数: {0}",
                "confirmed": "数据已确认",
                "confirm_prompt": "是否确认使用这些数据? (y/n): "
            },
            
            # Evaluation messages
            "evaluation": {
                "start": "开始评估模型...",
                "complete": "评估完成",
                "score": "评分: {0:.2f}/100",
                "details": "详细评估结果:\n{0}",
                "comparing": "正在比较模型...",
                "compared": "模型比较完成",
                "best_model": "最佳模型: {0}",
                "model_ranking": "模型排名:\n{0}",
                "metric": "评估指标: {0}",
                "sample_evaluation": "样本评估 {0}/{1}",
                "category": "类别: {0}",
                "report_saving": "正在保存评估报告...",
                "report_saved": "评估报告已保存到 {0}",
                "visualization": "正在创建可视化...",
                "visualization_saved": "可视化已保存到 {0}",
                "no_reference": "无参考回答",
                "human_evaluation": "人工评估",
                "automated_evaluation": "自动评估",
                "sample_count": "评估样本数: {0}",
                "time_taken": "评估用时: {0}",
                "result_summary": "评估摘要:\n{0}",
                "per_category": "按类别:\n{0}"
            },
            
            # Cache management messages
            "cache": {
                "init": "初始化缓存...",
                "cleaning": "正在清理缓存...",
                "cleaned": "已清理 {0} 个文件, {1} 个目录",
                "error": "缓存操作错误: {0}",
                "saving": "正在保存到缓存...",
                "saved": "已保存到缓存: {0}",
                "loading": "从缓存加载...",
                "loaded": "已从缓存加载: {0}",
                "not_found": "缓存未找到: {0}",
                "expired": "缓存已过期: {0}",
                "space_usage": "缓存空间使用: {0}",
                "days_kept": "保留 {0} 天内的缓存",
                "empty": "缓存为空",
                "purging": "正在清空缓存...",
                "purged": "缓存已清空",
                "backup": "正在备份缓存...",
                "backed_up": "缓存已备份到 {0}",
                "corrupted": "缓存已损坏: {0}",
                "rebuilding": "正在重建缓存...",
                "rebuilt": "缓存已重建"
            },

            # Resource monitoring messages
            "resources": {
                "monitoring_start": "资源监控已启动",
                "monitoring_stop": "资源监控已停止",
                "cpu_usage": "CPU使用率: {0}%",
                "memory_usage": "内存使用: {0:.1f}/{1:.1f}GB ({2}%)",
                "gpu_usage": "GPU {0} 使用: {1:.1f}/{2:.1f}GB ({3:.1f}%)",
                "disk_usage": "磁盘使用: {0:.1f}/{1:.1f}GB ({2}%)",
                "network_usage": "网络使用: 上传 {0:.1f}MB/s, 下载 {1:.1f}MB/s",
                "stats_summary": "资源使用统计:\n{0}",
                "warning_high_usage": "警告: {0} 使用率过高 ({1}%)",
                "warning_low_resource": "警告: {0} 资源不足 (剩余 {1}%)",
                "process_count": "进程数: {0}",
                "thread_count": "线程数: {0}",
                "system_load": "系统负载: {0}",
                "checking": "正在检查资源...",
                "clean_memory": "正在清理内存...",
                "clean_gpu_memory": "正在清理GPU内存...",
                "clean_complete": "内存清理完成",
                "periodic_check": "定期资源检查"
            },

            # Visualization messages
            "visualization": {
                "creating": "正在创建可视化...",
                "created": "可视化已创建",
                "saving": "正在保存可视化...",
                "saved": "可视化已保存到 {0}",
                "error": "可视化错误: {0}",
                "no_data": "没有可视化数据",
                "plotting": "正在绘制 {0}...",
                "plot_complete": "{0} 绘制完成",
                "title": "标题: {0}",
                "axis_x": "X轴: {0}",
                "axis_y": "Y轴: {0}",
                "legend": "图例: {0}",
                "figure_size": "图像大小: {0}x{1}",
                "dpi": "DPI: {0}",
                "format": "格式: {0}",
                "interactive": "交互式可视化",
                "static": "静态可视化",
                "chart_type": "图表类型: {0}",
                "data_points": "数据点数: {0}",
                "matplotlib_required": "需要安装matplotlib以创建可视化",
                "plotly_required": "需要安装plotly以创建交互式可视化",
                "seaborn_required": "需要安装seaborn以创建高级可视化"
            },

            # Time estimation messages
            "time_estimation": {
                "start": "开始估算训练时间...",
                "complete": "时间估算完成",
                "total_time": "总时间: {0}",
                "epoch_time": "每轮时间: {0}",
                "step_time": "每步时间: {0}",
                "batch_time": "每批次时间: {0}",
                "sample_time": "每样本时间: {0}",
                "estimated_finish": "预计完成时间: {0}",
                "remaining_time": "剩余时间: {0}",
                "elapsed_time": "已用时间: {0}",
                "progress": "进度: {0:.1f}%",
                "speed": "速度: {0} 样本/秒",
                "throughput": "吞吐量: {0} 标记/秒",
                "hardware_performance": "硬件性能: {0}",
                "note": "注意: {0}",
                "warning_slow": "警告: 训练速度较慢",
                "suggestion": "建议: {0}",
                "factors": "影响因素: {0}",
                "accuracy": "估算准确度: {0}",
                "raw_metrics": "原始指标: {0}"
            },

            # Error handling messages
            "error_handling": {
                "exception": "异常: {0}",
                "traceback": "堆栈跟踪:\n{0}",
                "recovering": "正在从错误中恢复...",
                "recovered": "已恢复",
                "unrecoverable": "无法恢复的错误",
                "retry": "重试中 ({0}/{1})...",
                "retry_success": "重试成功",
                "retry_failed": "重试失败",
                "fallback": "正在使用备选方案...",
                "error_type": "错误类型: {0}",
                "error_message": "错误信息: {0}",
                "error_context": "错误上下文: {0}",
                "error_count": "错误计数: {0}",
                "auto_save": "自动保存检查点...",
                "auto_saved": "已自动保存检查点",
                "logging": "正在记录错误...",
                "logged": "错误已记录",
                "reporting": "正在报告错误...",
                "reported": "错误已报告",
                "critical": "严重错误: {0}",
                "warning": "警告: {0}",
                "info": "信息: {0}",
                "debug": "调试: {0}"
            },

            # Checkpoint management messages
            "checkpoint": {
                "saving": "正在保存检查点...",
                "saved": "检查点已保存: {0}",
                "loading": "正在加载检查点...",
                "loaded": "检查点已加载: {0}",
                "error": "检查点错误: {0}",
                "not_found": "检查点未找到: {0}",
                "corrupt": "检查点已损坏: {0}",
                "incompatible": "检查点不兼容: {0}",
                "best": "最佳检查点: {0}",
                "latest": "最新检查点: {0}",
                "auto_save": "自动保存检查点: {0}",
                "cleanup": "正在清理旧检查点...",
                "cleaned_up": "已清理 {0} 个旧检查点",
                "interval": "检查点间隔: 每 {0} 轮",
                "metadata": "检查点元数据:\n{0}",
                "size": "检查点大小: {0}",
                "creating": "正在创建检查点目录...",
                "created": "检查点目录已创建",
                "listing": "可用检查点:\n{0}",
                "comparing": "正在比较检查点...",
                "compared": "检查点比较完成",
                "merging": "正在合并检查点...",
                "merged": "检查点已合并"
            }
        }
    
    def _get_english_messages(self) -> Dict[str, Any]:
        """Get English language pack"""
        return {
            # General messages
            "app_name": "APT Model Training Tool",
            "welcome": "Welcome to APT Model Training Tool!",
            "language": "Language",
            "version": "Version",
            "author": "Author",
            "license": "License",
            
            # Errors and warnings
            "error": "Error",
            "warning": "Warning",
            "info": "Info",
            "success": "Success",
            "failed": "Failed",
            
            # Training related messages
            "training": {
                "start": "Starting model training...",
                "completed": "Training completed!",
                "epoch": "Epoch",
                "batch": "Batch",
                "loss": "Loss",
                "lr": "Learning rate",
                "progress": "Progress",
                "early_stopping": "Early stopping: No improvement for {0} epochs",
                "saving_model": "Saving model...",
                "loading_model": "Loading model...",
                "best_model": "New best model found, saved to {0}",
                "best_quality": "New best quality model found ({0} points), saved",
                "total_params": "Total model parameters: {0}",
                "device": "Using device: {0}",
                "dataset_size": "Dataset size: {0} samples",
                "with_reasoning": "Training with reasoning ability",
                "with_distillation": "Training with knowledge distillation",
                "with_custom_data": "Training with custom data",
                "time_estimate": "Estimated training time: {0}",
                "time_remaining": "Time remaining: {0}",
                "training_speed": "Training speed: {0} samples/second",
                "validation_start": "Starting validation...",
                "validation_complete": "Validation complete, evaluation score: {0:.4f}",
                "validation": "Validation score: {0}",
                "checkpoint": "Checkpoint saved: {0}"
            },
            
            # Generation and evaluation related messages
            "generation": {
                "prompt": "Prompt",
                "response": "Response",
                "quality": "Quality score: {0}/100 - {1}",
                "test_generation": "Test generation",
                "amber_not_enough": "Amber: Training... not enough...",
                "amber_good": "Amber: Training completed well!",
                "temp": "Temperature",
                "top_p": "Top-P",
                "max_length": "Max length",
                "generating": "Generating...",
                "generation_time": "Generation time: {0:.2f} seconds",
                "input_too_long": "Input too long, truncated to {0} tokens",
                "batch_generation": "Batch generation in progress, completed: {0}/{1}",
                "time": "Generation time: {0}s",
                "tokens": "Generated tokens: {0}"
            },
            
            # Amber's messages
            "amber": {
                "training": "Amber: Let's train together!",
                "not_enough": "Amber: Training... not enough...",
                "good": "Amber: Training completed well!",
                "loading": "Amber: Getting ready...",
                "thinking": "Amber: Let me think...",
                "excited": "Amber: That's awesome!",
                "confused": "Amber: Hmm... that's a bit complex...",
                "evaluation": "Amber: Let me evaluate this...",
                "progress": "Amber: Still in progress...",
                "farewell": "Amber: Goodbye, see you next time!",
                "tired": "Amber: Phew, that was exhausting",
                "curious": "Amber: I wonder what happens if..."
            },
            
            # Hardware detection messages
            "hardware": {
                "gpu_detected": "Detected {0} GPU(s): {1}",
                "no_gpu": "No GPU detected, will use CPU for training (might be slow)",
                "insufficient_vram": "Poor little fishy~ Your GPU seems as useless as you are, such garbage♡\nModel requires at least {0:.1f}GB VRAM, but your GPU only has {1:.1f}GB",
                "sufficient_vram": "Sufficient VRAM, you can train larger models",
                "cpu_cores": "CPU Cores: {0}",
                "ram": "System Memory: {0:.1f}GB",
                "disk_space": "Available Disk Space: {0:.1f}GB",
                "checking": "Checking hardware compatibility...",
                "compatible": "Hardware compatibility check passed",
                "incompatible": "Hardware compatibility check failed: {0}",
                "cpu_info": "CPU: {0} ({1} cores)",
                "ram_info": "RAM: {0}GB",
                "disk_info": "Disk: {0}GB free",
                "speed_estimate": "Estimated training speed: {0} samples/sec"
            },
            
            # Command-line interface messages
            "cli": {
                "description": "APT (Auto-Prompting Transformer) Model Training and Testing Tool",
                "epilog": "Examples:\n  python apt.py train                    - Train model with default parameters\n  python apt.py train --epochs 10        - Train for 10 epochs\n  python apt.py train --force-cpu        - Force CPU training even if GPU is available\n  python apt.py test                     - Test model\n  python apt.py chat                     - Interactive dialogue with model\n  python apt.py chat --temperature 0.9   - Chat with higher temperature\n  python apt.py train --language en_US   - Train with English interface",
                "help": "Show this help message and exit",
                "language_help": "Language to use (default: zh_CN)",
                "language_file_help": "Custom language file path",
                "action_help": "Action to perform",
                "epochs_help": "Number of training epochs (default: 20)",
                "batch_size_help": "Training batch size (default: 8)",
                "learning_rate_help": "Model learning rate (default: 3e-5)",
                "save_path_help": "Model save path (default: \"apt_model\")",
                "force_cpu_help": "Force CPU computation even if GPU is available",
                "temperature_help": "Temperature parameter for generation (default: 0.7)",
                "top_p_help": "Top-p parameter for diversity in generation (default: 0.9)",
                "max_length_help": "Maximum generation length (default: 50)",
                "data_path_help": "External training data file path",
                "max_samples_help": "Maximum number of samples to use from external data (default: all)",
                "cache_dir_help": "Cache directory path (default: ~/.apt_cache)",
                "clean_days_help": "Clean cache older than days (default: 30)",
                "verbose_help": "Show detailed log information",
                "seed_help": "Random seed (default: 42)",
                "warning_message": "Warning: {0}",
                "error_message": "Error: {0}",
                "info_message": "Info: {0}",
                "success_message": "Success: {0}"
            },
            
            # Data processing messages
            "data": {
                "loading": "Loading data...",
                "loaded": "Loaded {0} records",
                "processing": "Processing data...",
                "processed": "Processed {0} records",
                "saving": "Saving data...",
                "saved": "Data saved to {0}",
                "error": "Error processing data: {0}",
                "format_error": "Invalid data format",
                "empty": "No data found",
                "duplicate": "Duplicate data found: {0}",
                "sample": "Sample data: {0}",
                "invalid": "Invalid data format",
                "too_large": "Data too large, exceeds maximum limit",
                "too_small": "Data too small, below minimum requirement",
                "example": "Data example: {0}",
                "stats": "Data statistics: {0}",
                "preview": "Data preview:\n{0}",
                "shuffle": "Shuffling data...",
                "split": "Splitting dataset...",
                "train_size": "Training set size: {0}",
                "val_size": "Validation set size: {0}",
                "test_size": "Test set size: {0}",
                "external_data": "External data",
                "internal_data": "Internal data",
                "combined_data": "Combined data",
                "filtering": "Filtering data...",
                "filtered": "Filtered {0} records",
                "sample_count": "Sample count: {0}",
                "confirmed": "Data confirmed",
                "confirm_prompt": "Confirm using this data? (y/n): "
            },
            
            # Evaluation messages
            "evaluation": {
                "start": "Starting model evaluation...",
                "complete": "Evaluation complete",
                "score": "Score: {0:.2f}/100",
                "details": "Detailed evaluation results:\n{0}",
                "comparing": "Comparing models...",
                "compared": "Model comparison complete",
                "best_model": "Best model: {0}",
                "model_ranking": "Model ranking:\n{0}",
                "metric": "Evaluation metric: {0}",
                "sample_evaluation": "Sample evaluation {0}/{1}",
                "category": "Category: {0}",
                "report_saving": "Saving evaluation report...",
                "report_saved": "Evaluation report saved to {0}",
                "visualization": "Creating visualization...",
                "visualization_saved": "Visualization saved to {0}",
                "no_reference": "No reference answer",
                "human_evaluation": "Human evaluation",
                "automated_evaluation": "Automated evaluation",
                "sample_count": "Evaluation sample count: {0}",
                "time_taken": "Time taken: {0}",
                "result_summary": "Evaluation summary:\n{0}",
                "per_category": "By category:\n{0}"
            },
            
            # Cache management messages
            "cache": {
                "init": "Initializing cache...",
                "cleaning": "Cleaning cache...",
                "cleaned": "Cleaned {0} files, {1} directories",
                "error": "Cache operation error: {0}",
                "saving": "Saving to cache...",
                "saved": "Saved to cache: {0}",
                "loading": "Loading from cache...",
                "loaded": "Loaded from cache: {0}",
                "not_found": "Cache not found: {0}",
                "expired": "Cache expired: {0}",
                "space_usage": "Cache space usage: {0}",
                "days_kept": "Keeping cache for {0} days",
                "empty": "Cache is empty",
                "purging": "Purging cache...",
                "purged": "Cache purged",
                "backup": "Backing up cache...",
                "backed_up": "Cache backed up to {0}",
                "corrupted": "Cache corrupted: {0}",
                "rebuilding": "Rebuilding cache...",
                "rebuilt": "Cache rebuilt",
                "directory": "Cache directory: {0}",
                "size": "Cache size: {0} MB",
                "retention": "Cache retention period: {0} days",
                "skipped": "Skipped files: {0}"
            },

            # Resource monitoring messages
            "resources": {
                "monitoring_start": "Resource monitoring started",
                "monitoring_stop": "Resource monitoring stopped",
                "cpu_usage": "CPU usage: {0}%",
                "memory_usage": "Memory usage: {0:.1f}/{1:.1f}GB ({2}%)",
                "gpu_usage": "GPU {0} usage: {1:.1f}/{2:.1f}GB ({3:.1f}%)",
                "disk_usage": "Disk usage: {0:.1f}/{1:.1f}GB ({2}%)",
                "network_usage": "Network usage: Upload {0:.1f}MB/s, Download {1:.1f}MB/s",
                "stats_summary": "Resource usage statistics:\n{0}",
                "warning_high_usage": "Warning: {0} usage too high ({1}%)",
                "warning_low_resource": "Warning: {0} resource low (remaining {1}%)",
                "process_count": "Process count: {0}",
                "thread_count": "Thread count: {0}",
                "system_load": "System load: {0}",
                "checking": "Checking resources...",
                "clean_memory": "Cleaning memory...",
                "clean_gpu_memory": "Cleaning GPU memory...",
                "clean_complete": "Memory cleaning complete",
                "periodic_check": "Periodic resource check"
            },

            # Visualization messages
            "visualization": {
                "creating": "Creating visualization...",
                "created": "Visualization created",
                "saving": "Saving visualization...",
                "saved": "Visualization saved to {0}",
                "error": "Visualization error: {0}",
                "no_data": "No visualization data",
                "plotting": "Plotting {0}...",
                "plot_complete": "{0} plotting complete",
                "title": "Title: {0}",
                "axis_x": "X-axis: {0}",
                "axis_y": "Y-axis: {0}",
                "legend": "Legend: {0}",
                "figure_size": "Figure size: {0}x{1}",
                "dpi": "DPI: {0}",
                "format": "Format: {0}",
                "interactive": "Interactive visualization",
                "static": "Static visualization",
                "chart_type": "Chart type: {0}",
                "data_points": "Data points: {0}",
                "matplotlib_required": "Matplotlib is required for visualization",
                "plotly_required": "Plotly is required for interactive visualization",
                "seaborn_required": "Seaborn is required for advanced visualization",
                "attention": "Attention Heatmap",
                "loss": "Loss Curve",
                "evaluation": "Evaluation Results",
                "comparison": "Model Comparison"
            },

            # Time estimation messages
            "time_estimation": {
                "start": "Starting time estimation...",
                "complete": "Time estimation complete",
                "total_time": "Total time: {0}",
                "epoch_time": "Time per epoch: {0}",
                "step_time": "Time per step: {0}",
                "batch_time": "Time per batch: {0}",
                "sample_time": "Time per sample: {0}",
                "estimated_finish": "Estimated finish time: {0}",
                "remaining_time": "Remaining time: {0}",
                "elapsed_time": "Elapsed time: {0}",
                "progress": "Progress: {0:.1f}%",
                "speed": "Speed: {0} samples/second",
                "throughput": "Throughput: {0} tokens/second",
                "hardware_performance": "Hardware performance: {0}",
                "note": "Note: {0}",
                "warning_slow": "Warning: Training speed is slow",
                "suggestion": "Suggestion: {0}",
                "factors": "Factors: {0}",
                "accuracy": "Estimate accuracy: {0}",
                "raw_metrics": "Raw metrics: {0}"
            },

            # Error handling messages
            "error_handling": {
                "exception": "Exception: {0}",
                "traceback": "Traceback:\n{0}",
                "recovering": "Recovering from error...",
                "recovered": "Recovered",
                "unrecoverable": "Unrecoverable error",
                "retry": "Retrying ({0}/{1})...",
                "retry_success": "Retry successful",
                "retry_failed": "Retry failed",
                "fallback": "Using fallback...",
                "error_type": "Error type: {0}",
                "error_message": "Error message: {0}",
                "error_context": "Error context: {0}",
                "error_count": "Error count: {0}",
                "auto_save": "Auto-saving checkpoint...",
                "auto_saved": "Checkpoint auto-saved",
                "logging": "Logging error...",
                "logged": "Error logged",
                "reporting": "Reporting error...",
                "reported": "Error reported",
                "critical": "Critical error: {0}",
                "warning": "Warning: {0}",
                "info": "Info: {0}",
                "debug": "Debug: {0}"
            },

            # Checkpoint management messages
            "checkpoint": {
                "saving": "Saving checkpoint...",
                "saved": "Checkpoint saved: {0}",
                "loading": "Loading checkpoint...",
                "loaded": "Checkpoint loaded: {0}",
                "error": "Checkpoint error: {0}",
                "not_found": "Checkpoint not found: {0}",
                "corrupt": "Checkpoint corrupted: {0}",
                "incompatible": "Checkpoint incompatible: {0}",
                "best": "Best checkpoint: {0}",
                "latest": "Latest checkpoint: {0}",
                "auto_save": "Auto-saving checkpoint: {0}",
                "cleanup": "Cleaning up old checkpoints...",
                "cleaned_up": "Cleaned up {0} old checkpoints",
                "interval": "Checkpoint interval: every {0} epochs",
                "metadata": "Checkpoint metadata:\n{0}",
                "size": "Checkpoint size: {0}",
                "creating": "Creating checkpoint directory...",
                "created": "Checkpoint directory created",
                "listing": "Available checkpoints:\n{0}",
                "comparing": "Comparing checkpoints...",
                "compared": "Checkpoint comparison complete",
                "merging": "Merging checkpoints...",
                "merged": "Checkpoints merged"
            }
        }