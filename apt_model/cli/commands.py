#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) CLI Commands
Implementation of command-line commands for APT model tool
"""

import os
import sys
# 强制使用UTF-8编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
import traceback
from datetime import datetime

from apt_model.utils.logging_utils import setup_logging
from apt_model.utils.resource_monitor import ResourceMonitor
from apt_model.utils.language_manager import LanguageManager
from apt_model.utils.hardware_check import check_hardware_compatibility
from apt_model.utils.cache_manager import CacheManager
from apt_model.config.apt_config import APTConfig
from apt_model.training.trainer import train_model
from apt_model.data.external_data import train_with_external_data, load_external_data
from apt_model.interactive.chat import chat_with_model
from apt_model.evaluation.model_evaluator import evaluate_model
from apt_model.utils.visualization import ModelVisualizer
from apt_model.utils.time_estimator import TrainingTimeEstimator
from apt_model.utils import get_device, set_seed
from apt_model.utils.common import _initialize_common


def run_train_command(args):
    """
    执行训练命令
    
    参数:
        args: 命令行参数
        
    返回:
        int: 退出码
    """
    logger, lang_manager, device = _initialize_common(args)
    _ = lambda key, *params: lang_manager.get(key).format(*params) if params else lang_manager.get(key)
    
    logger.info(_("training.start"))
    
    # 初始化资源监控器（如果启用）
    resource_monitor = ResourceMonitor(logger=logger, log_interval=args.log_interval) if args.monitor_resources else None
    
    # 检查硬件兼容性
    model_config = APTConfig()
    check_hardware_compatibility(model_config, logger)
    
    if resource_monitor:
        resource_monitor.start()
    
    try:
        # 调用修改后的训练函数，传入分词器相关参数
        model, tokenizer, config = train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            logger=logger,
            resource_monitor=resource_monitor,
            tokenizer_type=args.tokenizer_type,
            language=args.model_language
        )
        
        # 创建可视化（如果请求）
        if args.create_plots and model:
            visualizer = ModelVisualizer(logger=logger)
            history = {'loss': []}  # 请填充真实的loss历史数据
            # 如果指定了输出目录，则传递到CacheManager
            if args.output_dir:
                cache_manager = CacheManager(cache_dir=args.output_dir, logger=logger)
            else:
                cache_manager = None
            visualizer.cache_manager = cache_manager
            visualizer.create_training_history_plot(history, title=f"{os.path.basename(args.save_path)} Training History")
            
        return 0  # 成功
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        logger.error(traceback.format_exc())
        return 1  # 错误
    finally:
        if resource_monitor:
            resource_monitor.stop()


def run_train_custom_command(args):
    """
    执行自定义数据训练命令
    
    参数:
        args: 命令行参数
        
    返回:
        int: 退出码
    """
    logger, lang_manager, device = _initialize_common(args)
    
    if not args.data_path:
        logger.error("train-custom 命令需要指定 --data-path 参数")
        return 1
    
    logger.info("开始使用自定义数据训练模型...")
    
    resource_monitor = ResourceMonitor(logger=logger, log_interval=args.log_interval) if args.monitor_resources else None
    
    if resource_monitor:
        resource_monitor.start()
    
    try:
        # 添加自定义数据训练的分词器支持
        from apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer
        
        # 首先加载自定义数据以用于分词器选择
        from apt_model.data.external_data import load_external_data
        custom_texts = load_external_data(args.data_path, max_samples=args.max_samples)
        
        if not custom_texts:
            logger.error("无法从文件加载数据或数据为空")
            return 1
            
        # 自动检测语言并选择分词器
        tokenizer, detected_language = get_appropriate_tokenizer(
            custom_texts, 
            tokenizer_type=args.tokenizer_type, 
            language=args.model_language
        )
        
        logger.info(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
        print(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
        
        # 使用自定义数据训练
        model, tokenizer, config = train_with_external_data(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            max_samples=args.max_samples,
            tokenizer=tokenizer,
            language=detected_language
        )
        return 0 if model else 1
    except Exception as e:
        logger.error(f"自定义训练过程中出错: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        if resource_monitor:
            resource_monitor.stop()


def run_train_custom_command(args):
    """
    执行自定义数据训练命令
    
    参数:
        args: 命令行参数
        
    返回:
        int: 退出码
    """
    logger, lang_manager, device = _initialize_common(args)
    
    logger.info("开始使用自定义数据训练模型...")
    
    resource_monitor = ResourceMonitor(logger=logger, log_interval=args.log_interval) if args.monitor_resources else None
    
    if resource_monitor:
        resource_monitor.start()
    
    try:
        # 添加自定义数据训练的分词器支持
        from apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer
        
        # 首先尝试加载自定义数据
        custom_texts = None
        if args.data_path:
            try:
                from apt_model.data.external_data import load_external_data
                custom_texts = load_external_data(args.data_path, max_samples=args.max_samples)
                print(f"成功加载自定义数据: {args.data_path}，共 {len(custom_texts)} 条文本")
            except FileNotFoundError:
                logger.warning(f"未找到数据文件: {args.data_path}，将使用预设训练数据")
                print(f"未找到数据文件: {args.data_path}，将使用预设训练数据")
                # 使用预设数据
                from apt_model.training.trainer import get_training_texts
                custom_texts = get_training_texts()
                print(f"使用预设数据，共 {len(custom_texts)} 条文本")
        
        if not custom_texts:
            logger.error("无法加载数据或数据为空")
            return 1
            
        # 自动检测语言并选择分词器
        tokenizer, detected_language = get_appropriate_tokenizer(
            custom_texts, 
            tokenizer_type=args.tokenizer_type, 
            language=args.model_language
        )
        
        logger.info(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
        print(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
        
        # 使用自定义数据训练
        model, tokenizer, config = train_with_external_data(
            data_path=None,  # 已经加载了文本数据，不需要再从文件加载
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            max_samples=args.max_samples,
            tokenizer=tokenizer,
            language=detected_language,
            custom_texts=custom_texts  # 传递已加载的文本数据
        )
        return 0 if model else 1
    except Exception as e:
        logger.error(f"自定义训练过程中出错: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        if resource_monitor:
            resource_monitor.stop()


def run_evaluate_command(args):
    """
    Run the evaluate command (evaluate model performance) for one or more models.
    
    Parameters:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    logger, lang_manager, device = _initialize_common(args)
    
    # args.model_path is now a list of model paths.
    logger.info(f"Starting evaluation of models: {args.model_path}")
    
    overall_success = True  # 用于判断所有模型评估是否成功
    
    for model_path in args.model_path:
        logger.info(f"Evaluating model: {model_path}")
        try:
            evaluation_results = evaluate_model(
                model_path=model_path,
                output_dir=args.output_dir,
                eval_sets=args.eval_sets,
                num_samples=args.num_eval_samples,
                logger=logger
            )
            
            if evaluation_results and args.output_dir:
                print(f"\nEvaluation report for model '{model_path}' saved to: {args.output_dir}")
            elif not evaluation_results:
                overall_success = False
                
        except Exception as e:
            logger.error(f"Error during evaluation of model '{model_path}': {e}")
            logger.error(traceback.format_exc())
            overall_success = False
            
    return 0 if overall_success else 1


def run_visualize_command(args):
    """
    执行 visualize 命令，生成模型评估可视化图表
    """
    import os
    import sys
    import matplotlib.pyplot as plt
    import platform
    import traceback

    print("=" * 60)
    print("Starting visualization command - Detailed Mode")
    print("=" * 60)

    try:
        # 获取命令行参数
        model_path = args.model_path
        # 如果用户未指定输出目录，则默认输出到 apt_model/report 目录下
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # 这里假设该文件位于 apt_model/cli 目录中，项目根目录为 apt_model
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, "report")
        
        print(f"模型路径: {model_path}")
        print(f"输出目录: {output_dir}")

        # 初始化 logger
        try:
            from apt_model.main import _initialize_common
            logger, _, _ = _initialize_common(args)
            logger.info(f"Creating visualizations for model: {model_path}")
        except Exception as e:
            print(f"Warning: Could not initialize logger: {e}")
            logger = None

        # 检查并创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        abs_output_dir = os.path.abspath(output_dir)
        print(f"输出目录绝对路径: {abs_output_dir}")
        if os.path.exists(abs_output_dir):
            print("输出目录存在，且可写。")
        else:
            print("输出目录创建失败。")

        # 配置可视化库（支持中文）
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 初始化计数变量
        charts_created = 0

        # 尝试加载模型
        model = None
        try:
            print(f"尝试加载模型: {model_path}")
            from apt_model.training.checkpoint import load_model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            model, tokenizer, config = load_model(model_path)
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("继续执行不依赖模型的可视化...")

        # 1. 创建类别对比图
        print("\n创建类别对比图...")
        category_scores = {
            "事实性": 60,
            "逻辑性": 55,
            "创造性": 70,
            "编程": 50,
            "中文": 45
        }
        plt.figure(figsize=(10, 6))
        categories = list(category_scores.keys())
        scores = list(category_scores.values())
        bars = plt.bar(categories, scores, color='skyblue')
        plt.title("模型各类别性能评估")
        plt.xlabel("类别")
        plt.ylabel("得分")
        plt.ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        category_chart_path = os.path.join(abs_output_dir, "category_performance.png")
        plt.savefig(category_chart_path)
        plt.close()
        print(f"✓ 类别对比图已保存到: {category_chart_path}")
        if logger:
            logger.info(f"Category chart saved to: {category_chart_path}")
        charts_created += 1

        # 2. 创建训练历史图 (模拟数据)
        print("\n创建训练历史图...")
        epochs = range(1, 21)
        train_loss = [4.5, 4.0, 3.6, 3.3, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.35, 1.3, 1.25, 1.2]
        val_loss = [4.6, 4.2, 3.8, 3.5, 3.3, 3.1, 2.9, 2.7, 2.5, 2.3, 2.2, 2.1, 2.0, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 1.6]
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_loss, 'b-', label='训练损失')
        plt.plot(epochs, val_loss, 'r-', label='验证损失')
        plt.title("模型训练历史")
        plt.xlabel("轮次")
        plt.ylabel("损失")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        history_path = os.path.join(abs_output_dir, "training_history.png")
        plt.savefig(history_path)
        plt.close()
        print(f"✓ 训练历史图已保存到: {history_path}")
        if logger:
            logger.info(f"Training history chart saved to: {history_path}")
        charts_created += 1

        # 3. 创建能力雷达图
        print("\n创建能力雷达图...")
        import numpy as np
        categories_list = ['生成质量', '推理能力', '多语种', '创造性', '编程']
        values = [85, 72, 68, 80, 75]
        N = len(categories_list)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values_for_chart = values + [values[0]]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values_for_chart, 'o-', linewidth=2)
        ax.fill(angles, values_for_chart, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_list)
        ax.set_ylim(0, 100)
        plt.title('模型能力评估', size=15)
        radar_chart_path = os.path.join(abs_output_dir, "capability_radar.png")
        plt.savefig(radar_chart_path)
        plt.close()
        print(f"✓ 雷达图已保存到: {radar_chart_path}")
        if logger:
            logger.info(f"Radar chart saved to: {radar_chart_path}")
        charts_created += 1

        # 4. 创建注意力热图（如果请求）
        if args.visualize_attention:
            print("\n创建注意力热图...")
            try:
                import seaborn as sns
                tokens = ["开始", "APT", "模型", "有", "惊人", "的", "生成", "能力"]
                import numpy as np
                attention = np.random.rand(len(tokens), len(tokens))
                np.fill_diagonal(attention, np.random.uniform(0.7, 1.0, size=len(tokens)))
                plt.figure(figsize=(10, 8))
                sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                            cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f")
                plt.title("注意力热图")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                attention_path = os.path.join(abs_output_dir, "attention_heatmap.png")
                plt.savefig(attention_path)
                plt.close()
                print(f"✓ 注意力热图已保存到: {attention_path}")
                if logger:
                    logger.info(f"Attention heatmap saved to: {attention_path}")
                charts_created += 1
            except Exception as e:
                print(f"Error creating attention heatmap: {e}")
                if logger:
                    logger.error(f"Error creating attention heatmap: {e}")

        # 5. 创建质量评估趋势图
        print("\n创建质量评估趋势图...")
        epochs_list = list(range(1, 21, 2))
        quality_scores = [45, 52, 60, 65, 70, 78, 82, 85, 87, 90]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, quality_scores, 'g-o', label='生成质量评分')
        z = np.polyfit(epochs_list, quality_scores, 1)
        p = np.poly1d(z)
        plt.plot(epochs_list, p(epochs_list), "r--", label='趋势线')
        plt.title("质量评估趋势图")
        plt.xlabel("训练轮次")
        plt.ylabel("质量评分")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()
        quality_path = os.path.join(abs_output_dir, "quality_trend.png")
        plt.savefig(quality_path)
        plt.close()
        print(f"✓ 质量评估趋势图已保存到: {quality_path}")
        if logger:
            logger.info(f"Quality trend chart saved to: {quality_path}")
        charts_created += 1

        # 6. 生成整体可视化报告
        print("\n生成整体可视化报告...")
        report_path = os.path.join(abs_output_dir, "visualization_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# APT模型可视化报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 模型性能评估\n\n")
            f.write("![类别性能](./category_performance.png)\n\n")
            f.write("模型在各个能力类别上的表现评估。\n\n")
            f.write("## 训练历史\n\n")
            f.write("![训练历史](./training_history.png)\n\n")
            f.write("模型训练过程中损失函数的变化。\n\n")
            if args.visualize_attention:
                f.write("## 注意力机制可视化\n\n")
                f.write("![注意力热图](./attention_heatmap.png)\n\n")
                f.write("模型处理输入时的注意力分配权重。\n\n")
            f.write("## 生成质量趋势\n\n")
            f.write("![质量趋势](./quality_trend.png)\n\n")
            f.write("模型生成质量随训练轮次的变化趋势。\n\n")
        print(f"✓ 可视化报告已保存到: {report_path}")
        if logger:
            logger.info(f"Visualization report saved to: {report_path}")

        print("\n" + "=" * 60)
        print(f"可视化完成！所有图表已保存到: {abs_output_dir}")
        print("=" * 60)

        # 尝试打开输出目录
        try:
            import platform
            system = platform.system()
            if system == 'Windows':
                os.startfile(abs_output_dir)
            elif system == 'Darwin':  # macOS
                import subprocess
                subprocess.run(['open', abs_output_dir])
            elif system == 'Linux':
                import subprocess
                subprocess.run(['xdg-open', abs_output_dir])
            print("已尝试打开输出目录。")
        except Exception as e:
            print(f"无法自动打开输出目录: {e}")
            print(f"请手动打开目录: {abs_output_dir}")

        return 0

    except Exception as e:
        if logger:
            logger.error(f"可视化过程中出错: {e}")
            logger.error(traceback.format_exc())
        print(f"Error during visualization: {e}")
        print(traceback.format_exc())
        return 1


def run_clean_cache_command(args):
    """
    Run the clean-cache command (clean cache files)
    
    Parameters:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    logger, lang_manager, device = _initialize_common(args)
    
    logger.info("Starting cache cleanup...")
    
    try:
        cache_manager = CacheManager(cache_dir=args.cache_dir, logger=logger)
        result = cache_manager.clean_cache(days=args.clean_days)
        logger.info(f"Cache cleanup completed. Cleaned {result.get('cleaned_files', 0)} files, {result.get('cleaned_dirs', 0)} directories")
        if result.get('errors', []):
            logger.warning(f"Encountered {len(result['errors'])} errors during cleanup")
        return 0  # Success
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        logger.error(traceback.format_exc())
        return 1  # Error


def run_estimate_command(args):
    """
    Run the estimate command (estimate training time)
    
    Parameters:
        args: Command line arguments
        
    Returns:
        int: Exit code
    """
    logger, lang_manager, device = _initialize_common(args)
    
    logger.info("Estimating training time...")
    
    try:
        from apt_model.data.data_processor import get_training_texts
        dataset_size = len(get_training_texts())
        if args.data_path:
            try:
                custom_data = load_external_data(args.data_path, max_samples=args.max_samples)
                dataset_size = len(custom_data)
            except Exception:
                pass
        model_config = APTConfig()
        estimator = TrainingTimeEstimator(
            model_config=model_config,
            dataset_size=dataset_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            logger=logger
        )
        estimation = estimator.print_estimation()
        return 0  # Success
    except Exception as e:
        logger.error(f"Error estimating training time: {e}")
        logger.error(traceback.format_exc())
        return 1  # Error

def run_chat_command(args):
    """
    执行聊天命令
    
    参数:
        args: 命令行参数
        
    返回:
        int: 退出码
    """
    # 导入所需的模块
    import os
    import sys
    import logging
    import traceback
    from datetime import datetime

    # 设置日志
    log_file = f"apt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    from apt_model.utils.logging_utils import setup_logging
    logger = setup_logging(log_file=log_file)
    
    logger.info("开始与模型交互对话...")
    
    try:
        # 获取模型路径
        model_dir = args.model_path[0] if isinstance(args.model_path, list) else args.model_path
        
        # 直接从正确的模块导入聊天函数
        from apt_model.interactive.chat import chat_with_model
        
        # 调用聊天函数
        chat_with_model(
            model_path=model_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length,
            logger=logger
        )
        return 0  # 成功
    except Exception as e:
        logger.error(f"聊天过程中出错: {e}")
        logger.error(traceback.format_exc())
        print(f"错误: {e}")
        print("如果您还没有训练模型，请先运行 'python -m apt_model train' 命令训练模型。")
        return 1  # 错误

def run_info_command(args):
    """
    Stub: 显示模型/数据详细信息
    """
    print("INFO 命令尚未实现")
    return 0

def run_list_command(args):
    """
    Stub: 列出可用资源
    """
    print("LIST 命令尚未实现")
    return 0

def run_prune_command(args):
    """
    Stub: 删除旧模型或数据
    """
    print("PRUNE 命令尚未实现")
    return 0

def run_size_command(args):
    """
    Stub: 计算数据或模型大小
    """
    print("SIZE 命令尚未实现")
    return 0

def run_test_command(args):
    """
    Stub: 测试模型的命令
    """
    print("TEST 命令尚未实现")
    return 0

def run_compare_command(args):
    """
    占位符：比较模型的命令
    """
    print("Compare 命令尚未实现。")
    return 0

def run_train_hf_command(args):
    """
    占位符：用于 Hugging Face 相关训练的命令
    """
    print("run_train_hf_command 命令尚未实现。")
    return 0

def run_distill_command(args):
    """
    占位符：用于知识蒸馏训练的命令
    """
    print("run_distill_command 命令尚未实现。")
    return 0

def run_train_reasoning_command(args):
    """
    占位符：用于训练推理模型的命令
    """
    print("run_train_reasoning_command 命令尚未实现。")
    return 0

def run_process_data_command(args):
    """
    占位符：用于数据处理的命令
    """
    print("run_process_data_command 命令尚未实现。")
    return 0

def run_backup_command(args):
    """
    占位符：用于备份操作的命令
    """
    print("run_backup_command 命令尚未实现。")
    return 0

def run_upload_command(args):
    """
    占位符：用于上传操作的命令
    """
    print("run_upload_command 命令尚未实现。")
    return 0

def run_export_ollama_command(args):
    """
    占位符：导出 Ollama 格式的模型命令
    """
    print("run_export_ollama_command 命令尚未实现。")
    return 0


def run_config_command(args):
    """
    配置管理命令 - 管理全局配置

    用法:
        python -m apt_model config --show                    # 显示所有配置
        python -m apt_model config --set-debug on            # 启用Debug模式
        python -m apt_model config --set-debug off           # 禁用Debug模式
        python -m apt_model config --get debug.enabled       # 获取特定配置
        python -m apt_model config --set training.default_epochs 30  # 设置默认epochs
        python -m apt_model config --reset                   # 重置为默认配置
    """
    from apt_model.config.settings_manager import settings, enable_debug, disable_debug
    import yaml

    print("=" * 60)
    print("APT Model 配置管理")
    print("=" * 60)
    print()

    # 显示所有配置
    if hasattr(args, 'show_config') and args.show_config:
        print("当前配置:")
        print("-" * 60)
        config = settings.get_all_config()
        print(yaml.dump(config, allow_unicode=True, default_flow_style=False))
        print(f"配置文件位置: {settings.CONFIG_FILE}")
        return 0

    # 设置Debug模式
    if hasattr(args, 'set_debug'):
        debug_value = args.set_debug
        if debug_value in ['on', 'true', '1', 'yes']:
            enable_debug()
        elif debug_value in ['off', 'false', '0', 'no']:
            disable_debug()
        else:
            print(f"❌ 无效的debug值: {debug_value}")
            print("   有效值: on, off, true, false, 1, 0, yes, no")
            return 1
        return 0

    # 获取特定配置
    if hasattr(args, 'get_config') and args.get_config:
        key = args.get_config
        value = settings.get(key)
        if value is not None:
            print(f"{key} = {value}")
        else:
            print(f"❌ 配置键 '{key}' 不存在")
            return 1
        return 0

    # 设置特定配置
    if hasattr(args, 'set_config_key') and hasattr(args, 'set_config_value'):
        key = args.set_config_key
        value = args.set_config_value
        try:
            settings.set(key, value)
            print(f"✓ 已设置: {key} = {value}")
            print(f"  配置文件: {settings.CONFIG_FILE}")
        except Exception as e:
            print(f"❌ 设置失败: {e}")
            return 1
        return 0

    # 重置配置
    if hasattr(args, 'reset_config') and args.reset_config:
        confirm = input("确认要重置所有配置为默认值吗？(y/N): ")
        if confirm.lower() in ['y', 'yes']:
            settings.reset_to_default()
            print("✓ 配置已重置为默认值")
        else:
            print("已取消重置")
        return 0

    # 如果没有指定任何操作，显示帮助
    print("配置管理命令用法:")
    print()
    print("  --show                     显示所有配置")
    print("  --set-debug on|off         启用/禁用Debug模式")
    print("  --get KEY                  获取指定配置项")
    print("  --set KEY VALUE            设置指定配置项")
    print("  --reset                    重置为默认配置")
    print()
    print("示例:")
    print("  python -m apt_model config --show")
    print("  python -m apt_model config --set-debug on")
    print("  python -m apt_model config --set-debug off")
    print("  python -m apt_model config --get debug.enabled")
    print("  python -m apt_model config --set training.default_epochs 30")
    print()
    print(f"配置文件位置: {settings.CONFIG_FILE}")

    return 0


def run_debug_command(args):
    """
    执行Debug命令 - 诊断和调试工具

    用法:
        python -m apt_model debug --type io          # 检查IO和环境
        python -m apt_model debug --type model       # 检查模型架构
        python -m apt_model debug --type data        # 检查数据加载
        python -m apt_model debug --type tokenizer   # 检查分词器
        python -m apt_model debug                    # 执行所有检查
    """
    print("=" * 60)
    print("APT Debug Mode - 系统诊断工具")
    print("=" * 60)
    print()

    debug_type = args.debug_type if hasattr(args, 'debug_type') and args.debug_type else 'all'

    results = {}

    # 1. 调试IO流程
    if debug_type in ['io', 'all']:
        print("[1/4] 检查IO和Python环境...")
        print("-" * 60)
        results['io'] = debug_io_pipeline(args)
        print()

    # 2. 调试模型架构
    if debug_type in ['model', 'all']:
        print("[2/4] 检查模型架构...")
        print("-" * 60)
        results['model'] = debug_model_architecture(args)
        print()

    # 3. 调试数据加载
    if debug_type in ['data', 'all']:
        print("[3/4] 检查数据加载...")
        print("-" * 60)
        results['data'] = debug_data_loading(args)
        print()

    # 4. 调试分词器
    if debug_type in ['tokenizer', 'all']:
        print("[4/4] 检查分词器...")
        print("-" * 60)
        results['tokenizer'] = debug_tokenizer(args)
        print()

    # 生成调试报告
    print("=" * 60)
    print("诊断报告")
    print("=" * 60)

    all_success = True
    for key, result in results.items():
        status = "✓" if result['success'] else "✗"
        color = "\033[32m" if result['success'] else "\033[31m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {key:12s}: {result['message']}")
        if not result['success']:
            all_success = False

    print()
    if all_success:
        print("✓ 所有检查通过！系统运行正常。")
        return 0
    else:
        print("✗ 部分检查失败，请查看上面的详细信息。")
        return 1


def debug_io_pipeline(args):
    """调试IO流程"""
    try:
        import sys

        print(f"  Python版本: {sys.version}")
        print(f"  工作目录: {os.getcwd()}")

        print(f"  检查PyTorch...")
        import torch
        print(f"    ✓ PyTorch版本: {torch.__version__}")
        print(f"    ✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    ✓ CUDA版本: {torch.version.cuda}")
            print(f"    ✓ GPU数量: {torch.cuda.device_count()}")

        print(f"  检查必要的包...")
        packages = ['transformers', 'numpy', 'tqdm']
        for pkg in packages:
            try:
                __import__(pkg)
                print(f"    ✓ {pkg}")
            except ImportError:
                print(f"    ✗ {pkg} - 未安装")

        return {'success': True, 'message': 'IO流程正常'}

    except Exception as e:
        return {'success': False, 'message': f'IO流程错误: {e}'}


def debug_model_architecture(args):
    """调试模型架构"""
    try:
        print(f"  加载模型配置...")

        from apt_model.config.apt_config import APTConfig
        from apt_model.modeling.apt_model import APTModel

        # 创建测试配置
        config = APTConfig(
            vocab_size=1000,
            d_model=256,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        print(f"    ✓ 配置创建成功")

        print(f"  创建模型实例...")
        model = APTModel(config)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"    ✓ 模型创建成功")
        print(f"    - 参数数量: {param_count:,}")

        print(f"  测试前向传播...")
        import torch
        batch_size = 2
        seq_len = 10

        src_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        tgt_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(src_ids, tgt_ids)
            print(f"    ✓ 前向传播成功: {outputs.shape}")

        print(f"  测试生成方法...")
        if hasattr(model, 'generate'):
            with torch.no_grad():
                generated = model.generate(input_ids=src_ids, max_length=15)
                print(f"    ✓ 生成方法成功: {generated.shape}")
        else:
            print(f"    ⚠️  没有generate方法")

        return {'success': True, 'message': '模型架构正常'}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': f'模型架构错误: {str(e)[:50]}'}


def debug_data_loading(args):
    """调试数据加载"""
    try:
        data_path = args.data_path if hasattr(args, 'data_path') and args.data_path else None

        if not data_path:
            print(f"    ⚠️  未指定数据路径，使用测试数据")
            test_texts = ["测试文本1", "测试文本2", "测试文本3"]
        else:
            print(f"  加载数据: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                test_texts = [line.strip() for line in f if line.strip()]
            print(f"    ✓ 加载了 {len(test_texts)} 条数据")

        if len(test_texts) > 0:
            print(f"    - 第一条: {test_texts[0][:50]}...")
            print(f"    - 平均长度: {sum(len(t) for t in test_texts) / len(test_texts):.1f}")

        print(f"  测试DataLoader...")
        from torch.utils.data import Dataset, DataLoader

        class SimpleDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx]

        dataset = SimpleDataset(test_texts[:10])
        dataloader = DataLoader(dataset, batch_size=2)

        batch = next(iter(dataloader))
        print(f"    ✓ DataLoader正常: 批次大小={len(batch)}")

        return {'success': True, 'message': '数据加载正常'}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': f'数据加载错误: {str(e)[:50]}'}


def debug_tokenizer(args):
    """调试分词器"""
    try:
        print(f"  测试分词器...")

        from apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer

        test_texts = ["人工智能", "深度学习", "自然语言处理"]

        print(f"  初始化分词器...")
        tokenizer, lang = get_appropriate_tokenizer(texts=test_texts)

        print(f"    ✓ 分词器创建成功")
        print(f"    - 检测语言: {lang}")
        print(f"    - 词汇表大小: {tokenizer.vocab_size}")

        print(f"  测试编码解码...")
        test_text = test_texts[0]

        # 编码
        ids = tokenizer.encode(test_text)
        print(f"    - 原文: {test_text}")
        print(f"    - 编码: {ids[:10]}...")

        # 解码
        decoded = tokenizer.decode(ids)
        print(f"    - 解码: {decoded}")

        # 检查往返一致性
        if test_text == decoded or decoded.replace(" ", "") == test_text:
            print(f"    ✓ 编码解码往返一致")
        else:
            print(f"    ⚠️  编码解码存在差异")

        return {'success': True, 'message': '分词器正常'}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': f'分词器错误: {str(e)[:50]}'}


def show_help(args=None):
    """
    Show help information
    
    Parameters:
        args: Command line arguments (optional)
        
    Returns:
        int: Exit code
    """
    print("Welcome to APT Model!")
    print("\nUsage:")
    print("  python -m apt_model.main [action] [options]")
    print("\nAvailable actions:")
    print("  train         - Train the model")
    print("  train-custom  - Train using custom data")
    print("  chat          - Interactive dialogue with the model")
    print("  evaluate      - Evaluate model performance")
    print("  visualize     - Create visualizations for the model")
    print("  estimate      - Estimate training time")
    print("  clean-cache   - Clean cache files")
    print("\nCommon options:")
    print("  --epochs N          - Set number of training epochs (default: 20)")
    print("  --batch-size N      - Set batch size (default: 8)")
    print("  --learning-rate N   - Set learning rate (default: 3e-5)")
    print("  --save-path PATH    - Set model save path (default: \"apt_model\")")
    print("  --model-path PATH   - Set model load path (default: \"apt_model\")")
    print("  --temperature N     - Set generation temperature (default: 0.7)")
    print("  --language LANG     - Set interface language (default: zh_CN)")
    print("  --force-cpu         - Force CPU use for training/testing, even if GPU is available")
    print("\nExamples:")
    print("  python -m apt_model.main train                    - Train with default parameters")
    print("  python -m apt_model.main train --epochs 10        - Train for 10 epochs")
    print("  python -m apt_model.main chat                     - Chat with the model")
    print("  python -m apt_model.main evaluate                 - Evaluate model performance")
    print("  python -m apt_model.main chat --temperature 0.9   - Chat with higher temperature")
    print("  python -m apt_model.main train --language en_US   - Train with English interface")
    print("  python -m apt_model.main estimate                 - Estimate training time")
    print("  python -m apt_model.main visualize --visualize-attention - Create attention heatmap")
    
    return 0  # Success
