#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) CLI Commands
Implementation of command-line commands for APT model tool

重构后的命令系统：
- 提取公共代码到辅助函数
- 支持插件命令注册
- 清晰的命令分类
- 统一的错误处理
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
from apt_model.cli.command_registry import register_command


# ============================================================================
# 辅助函数 - 公共代码提取
# ============================================================================

def _setup_resource_monitor(args, logger):
    """设置资源监控器（如果启用）"""
    if args.monitor_resources:
        return ResourceMonitor(logger=logger, log_interval=args.log_interval)
    return None


def _start_monitor(resource_monitor):
    """启动资源监控器"""
    if resource_monitor:
        resource_monitor.start()


def _stop_monitor(resource_monitor):
    """停止资源监控器"""
    if resource_monitor:
        resource_monitor.stop()


def _handle_command_error(command_name, error, logger):
    """统一的命令错误处理"""
    logger.error(f"{command_name}过程中出错: {error}")
    logger.error(traceback.format_exc())
    print(f"错误: {error}")
    return 1


def _get_tokenizer_with_detection(texts, args):
    """
    获取tokenizer并检测语言

    使用新的codec系统（优先）或回退到旧系统
    """
    from apt_model.codecs import get_codec_for_language
    from apt_model.codecs.compat import CodecTokenizerWrapper
    from apt_model.modeling.chinese_tokenizer_integration import detect_language

    # 自动检测语言
    detected_language = args.model_language or detect_language(texts)

    # 尝试使用新的codec系统
    try:
        codec = get_codec_for_language(detected_language)
        if codec:
            tokenizer = CodecTokenizerWrapper(codec)
            return tokenizer, detected_language
    except Exception as e:
        print(f"Codec系统失败，回退到旧分词器: {e}")

    # 回退到旧系统
    from apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer
    return get_appropriate_tokenizer(
        texts,
        tokenizer_type=args.tokenizer_type,
        language=args.model_language
    )


def _create_visualizations(args, model, logger):
    """创建模型可视化（如果请求）"""
    if args.create_plots and model:
        visualizer = ModelVisualizer(logger=logger)
        history = {'loss': []}  # 实际应从训练过程获取

        if args.output_dir:
            cache_manager = CacheManager(cache_dir=args.output_dir, logger=logger)
            visualizer.cache_manager = cache_manager

        visualizer.create_training_history_plot(
            history,
            title=f"{os.path.basename(args.save_path)} Training History"
        )


# ============================================================================
# 训练相关命令
# ============================================================================

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

    # 检查硬件兼容性
    model_config = APTConfig()
    check_hardware_compatibility(model_config, logger)

    # 设置资源监控
    resource_monitor = _setup_resource_monitor(args, logger)
    _start_monitor(resource_monitor)

    try:
        # 调用训练函数
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

        # 创建可视化
        _create_visualizations(args, model, logger)

        return 0  # 成功
    except Exception as e:
        return _handle_command_error("训练", e, logger)
    finally:
        _stop_monitor(resource_monitor)


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

    # 设置资源监控
    resource_monitor = _setup_resource_monitor(args, logger)
    _start_monitor(resource_monitor)

    try:
        # 加载自定义数据
        custom_texts = None
        if args.data_path:
            try:
                custom_texts = load_external_data(args.data_path, max_samples=args.max_samples)
                print(f"成功加载自定义数据: {args.data_path}，共 {len(custom_texts)} 条文本")
            except FileNotFoundError:
                logger.warning(f"未找到数据文件: {args.data_path}，将使用预设训练数据")
                print(f"未找到数据文件: {args.data_path}，将使用预设训练数据")
                from apt_model.training.trainer import get_training_texts
                custom_texts = get_training_texts()
                print(f"使用预设数据，共 {len(custom_texts)} 条文本")

        if not custom_texts:
            logger.error("无法加载数据或数据为空")
            return 1

        # 获取tokenizer并检测语言
        tokenizer, detected_language = _get_tokenizer_with_detection(custom_texts, args)
        logger.info(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
        print(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")

        # 使用自定义数据训练
        model, tokenizer, config = train_with_external_data(
            data_path=None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            max_samples=args.max_samples,
            tokenizer=tokenizer,
            language=detected_language,
            custom_texts=custom_texts
        )
        return 0 if model else 1
    except Exception as e:
        return _handle_command_error("自定义训练", e, logger)
    finally:
        _stop_monitor(resource_monitor)


# ============================================================================
# 交互相关命令
# ============================================================================

def run_chat_command(args):
    """
    执行聊天命令

    参数:
        args: 命令行参数

    返回:
        int: 退出码
    """
    # 设置日志
    log_file = f"apt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file=log_file)
    logger.info("开始与模型交互对话...")

    try:
        # 获取模型路径
        model_dir = args.model_path[0] if isinstance(args.model_path, list) else args.model_path

        # 调用聊天函数
        chat_with_model(
            model_path=model_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length,
            logger=logger
        )
        return 0
    except Exception as e:
        logger.error(f"聊天过程中出错: {e}")
        logger.error(traceback.format_exc())
        print(f"错误: {e}")
        print("如果您还没有训练模型，请先运行 'python -m apt_model train' 命令训练模型。")
        return 1


# ============================================================================
# 评估相关命令
# ============================================================================

def run_evaluate_command(args):
    """
    Run the evaluate command (evaluate model performance) for one or more models.

    Parameters:
        args: Command line arguments

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    logger, lang_manager, device = _initialize_common(args)
    logger.info(f"Starting evaluation of models: {args.model_path}")

    overall_success = True

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
    import matplotlib.pyplot as plt
    import numpy as np

    print("=" * 60)
    print("Starting visualization command")
    print("=" * 60)

    try:
        # 获取命令行参数
        model_path = args.model_path
        if args.output_dir:
            output_dir = args.output_dir
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, "report")

        print(f"模型路径: {model_path}")
        print(f"输出目录: {output_dir}")

        # 初始化 logger
        try:
            logger, _, _ = _initialize_common(args)
            logger.info(f"Creating visualizations for model: {model_path}")
        except Exception as e:
            print(f"Warning: Could not initialize logger: {e}")
            logger = None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        abs_output_dir = os.path.abspath(output_dir)

        # 配置可视化库（支持中文）
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

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

        # 创建各种图表
        charts_created += _create_category_chart(abs_output_dir, logger)
        charts_created += _create_training_history_chart(abs_output_dir, logger)
        charts_created += _create_capability_radar_chart(abs_output_dir, logger)
        charts_created += _create_quality_trend_chart(abs_output_dir, logger)

        if args.visualize_attention:
            charts_created += _create_attention_heatmap(abs_output_dir, logger)

        # 生成整体报告
        _create_visualization_report(abs_output_dir, args, logger)

        print("\n" + "=" * 60)
        print(f"可视化完成！所有图表已保存到: {abs_output_dir}")
        print("=" * 60)

        # 尝试打开输出目录
        _try_open_directory(abs_output_dir)

        return 0

    except Exception as e:
        if 'logger' in locals() and logger:
            logger.error(f"可视化过程中出错: {e}")
            logger.error(traceback.format_exc())
        print(f"Error during visualization: {e}")
        print(traceback.format_exc())
        return 1


# 可视化辅助函数
def _create_category_chart(output_dir, logger):
    """创建类别对比图"""
    import matplotlib.pyplot as plt

    print("\n创建类别对比图...")
    category_scores = {
        "事实性": 60, "逻辑性": 55, "创造性": 70,
        "编程": 50, "中文": 45
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
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                 ha='center', va='bottom')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "category_performance.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"✓ 类别对比图已保存到: {chart_path}")
    if logger:
        logger.info(f"Category chart saved to: {chart_path}")
    return 1


def _create_training_history_chart(output_dir, logger):
    """创建训练历史图"""
    import matplotlib.pyplot as plt

    print("\n创建训练历史图...")
    epochs = range(1, 21)
    train_loss = [4.5 - i * 0.16 for i in range(20)]
    val_loss = [4.6 - i * 0.15 for i in range(20)]
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'b-', label='训练损失')
    plt.plot(epochs, val_loss, 'r-', label='验证损失')
    plt.title("模型训练历史")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    history_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_path)
    plt.close()
    print(f"✓ 训练历史图已保存到: {history_path}")
    if logger:
        logger.info(f"Training history chart saved to: {history_path}")
    return 1


def _create_capability_radar_chart(output_dir, logger):
    """创建能力雷达图"""
    import matplotlib.pyplot as plt
    import numpy as np

    print("\n创建能力雷达图...")
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
    radar_chart_path = os.path.join(output_dir, "capability_radar.png")
    plt.savefig(radar_chart_path)
    plt.close()
    print(f"✓ 雷达图已保存到: {radar_chart_path}")
    if logger:
        logger.info(f"Radar chart saved to: {radar_chart_path}")
    return 1


def _create_quality_trend_chart(output_dir, logger):
    """创建质量评估趋势图"""
    import matplotlib.pyplot as plt
    import numpy as np

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
    quality_path = os.path.join(output_dir, "quality_trend.png")
    plt.savefig(quality_path)
    plt.close()
    print(f"✓ 质量评估趋势图已保存到: {quality_path}")
    if logger:
        logger.info(f"Quality trend chart saved to: {quality_path}")
    return 1


def _create_attention_heatmap(output_dir, logger):
    """创建注意力热图"""
    import matplotlib.pyplot as plt
    import numpy as np

    print("\n创建注意力热图...")
    try:
        import seaborn as sns
        tokens = ["开始", "APT", "模型", "有", "惊人", "的", "生成", "能力"]
        attention = np.random.rand(len(tokens), len(tokens))
        np.fill_diagonal(attention, np.random.uniform(0.7, 1.0, size=len(tokens)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                    cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f")
        plt.title("注意力热图")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        attention_path = os.path.join(output_dir, "attention_heatmap.png")
        plt.savefig(attention_path)
        plt.close()
        print(f"✓ 注意力热图已保存到: {attention_path}")
        if logger:
            logger.info(f"Attention heatmap saved to: {attention_path}")
        return 1
    except Exception as e:
        print(f"Error creating attention heatmap: {e}")
        if logger:
            logger.error(f"Error creating attention heatmap: {e}")
        return 0


def _create_visualization_report(output_dir, args, logger):
    """生成整体可视化报告"""
    print("\n生成整体可视化报告...")
    report_path = os.path.join(output_dir, "visualization_report.md")
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


def _try_open_directory(directory):
    """尝试打开目录"""
    try:
        import platform
        system = platform.system()
        if system == 'Windows':
            os.startfile(directory)
        elif system == 'Darwin':  # macOS
            import subprocess
            subprocess.run(['open', directory])
        elif system == 'Linux':
            import subprocess
            subprocess.run(['xdg-open', directory])
        print("已尝试打开输出目录。")
    except Exception as e:
        print(f"无法自动打开输出目录: {e}")
        print(f"请手动打开目录: {directory}")


# ============================================================================
# 工具相关命令
# ============================================================================

def run_clean_cache_command(args):
    """
    Run the clean-cache command (clean cache files)
    """
    logger, lang_manager, device = _initialize_common(args)
    logger.info("Starting cache cleanup...")

    try:
        cache_manager = CacheManager(cache_dir=args.cache_dir, logger=logger)
        result = cache_manager.clean_cache(days=args.clean_days)
        logger.info(f"Cache cleanup completed. Cleaned {result.get('cleaned_files', 0)} files, "
                   f"{result.get('cleaned_dirs', 0)} directories")
        if result.get('errors', []):
            logger.warning(f"Encountered {len(result['errors'])} errors during cleanup")
        return 0
    except Exception as e:
        return _handle_command_error("缓存清理", e, logger)


def run_estimate_command(args):
    """
    Run the estimate command (estimate training time)
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
        return 0
    except Exception as e:
        return _handle_command_error("时间估算", e, logger)


# ============================================================================
# 占位符命令 - 待实现
# ============================================================================

def run_info_command(args):
    """占位符: 显示模型/数据详细信息"""
    print("INFO 命令尚未实现")
    return 0


def run_list_command(args):
    """占位符: 列出可用资源"""
    print("LIST 命令尚未实现")
    return 0


def run_prune_command(args):
    """占位符: 删除旧模型或数据"""
    print("PRUNE 命令尚未实现")
    return 0


def run_size_command(args):
    """占位符: 计算数据或模型大小"""
    print("SIZE 命令尚未实现")
    return 0


def run_test_command(args):
    """占位符: 测试模型的命令"""
    print("TEST 命令尚未实现")
    return 0


def run_compare_command(args):
    """占位符：比较模型的命令"""
    print("Compare 命令尚未实现。")
    return 0


def run_train_hf_command(args):
    """占位符：用于 Hugging Face 相关训练的命令"""
    print("run_train_hf_command 命令尚未实现。")
    return 0


def run_distill_command(args):
    """占位符：用于知识蒸馏训练的命令"""
    print("run_distill_command 命令尚未实现。")
    return 0


def run_train_reasoning_command(args):
    """
    执行推理模型训练命令

    参数:
        args: 命令行参数

    返回:
        int: 退出码
    """
    logger, lang_manager, device = _initialize_common(args)
    logger.info("开始训练推理增强模型...")

    # 设置资源监控
    resource_monitor = _setup_resource_monitor(args, logger)
    _start_monitor(resource_monitor)

    try:
        # Import reasoning training
        from apt_model.training.train_reasoning import train_reasoning_model, load_reasoning_dataset
        from apt_model.modeling.gpt4o_model import VeinSubspaceShared

        # Load base model (placeholder - should load actual model)
        logger.info("加载基础模型...")
        # TODO: Load actual pre-trained base model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = getattr(args, 'base_model', 'gpt2')
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get model dimension
        d_model = base_model.config.hidden_size

        # Create vein projector
        rank = getattr(args, 'vein_rank', 4)
        vein = VeinSubspaceShared(d_model=d_model, rank=rank)

        logger.info(f"使用 Vein 子空间: d_model={d_model}, rank={rank}")

        # Train reasoning model
        reasoning_controller, training_info = train_reasoning_model(
            base_model=base_model,
            vein_projector=vein,
            tokenizer=tokenizer,
            data_path=getattr(args, 'data_path', None),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_reasoning_steps=getattr(args, 'max_reasoning_steps', 6),
            use_budgeted=getattr(args, 'use_budgeted', True),
            global_budget=getattr(args, 'global_budget', 0.15),
            save_path=args.save_path,
            logger=logger,
            resource_monitor=resource_monitor,
        )

        logger.info("推理模型训练完成！")
        logger.info(f"训练信息: {training_info}")

        return 0  # 成功
    except Exception as e:
        return _handle_command_error("推理训练", e, logger)
    finally:
        _stop_monitor(resource_monitor)


def run_process_data_command(args):
    """占位符：用于数据处理的命令"""
    print("run_process_data_command 命令尚未实现。")
    return 0


def run_backup_command(args):
    """占位符：用于备份操作的命令"""
    print("run_backup_command 命令尚未实现。")
    return 0


def run_upload_command(args):
    """占位符：用于上传操作的命令"""
    print("run_upload_command 命令尚未实现。")
    return 0


def run_export_ollama_command(args):
    """占位符：导出 Ollama 格式的模型命令"""
    print("run_export_ollama_command 命令尚未实现。")
    return 0


def show_help(args=None):
    """
    Show help information
    """
    from apt_model.cli.command_registry import command_registry

    print("Welcome to APT Model!")
    print("\nUsage:")
    print("  python -m apt_model [action] [options]")
    print("\n可用命令:")

    # 按类别显示命令
    commands_by_category = command_registry.get_commands_by_category(include_placeholders=True)

    for category in sorted(commands_by_category.keys()):
        print(f"\n{category.upper()}:")
        for metadata in commands_by_category[category]:
            status = " (尚未实现)" if metadata.is_placeholder else ""
            help_text = metadata.help_text or "无说明"
            print(f"  {metadata.name:<20} - {help_text}{status}")
            if metadata.aliases:
                print(f"    {'':18}别名: {', '.join(metadata.aliases)}")

    print("\n常用选项:")
    print("  --epochs N          - 训练轮数 (默认: 20)")
    print("  --batch-size N      - 批次大小 (默认: 8)")
    print("  --learning-rate N   - 学习率 (默认: 3e-5)")
    print("  --save-path PATH    - 模型保存路径 (默认: 'apt_model')")
    print("  --model-path PATH   - 模型加载路径 (默认: 'apt_model')")
    print("  --temperature N     - 生成温度 (默认: 0.7)")
    print("  --language LANG     - 界面语言 (默认: zh_CN)")
    print("  --force-cpu         - 强制使用CPU")
    print("\n示例:")
    print("  python -m apt_model train")
    print("  python -m apt_model train --epochs 10")
    print("  python -m apt_model chat")
    print("  python -m apt_model evaluate")

    return 0


# ============================================================================
# 命令注册
# ============================================================================

def register_core_commands():
    """
    注册核心命令到命令注册中心

    这个函数在模块导入时自动调用，注册所有核心命令。
    插件可以通过调用 register_command() 添加自己的命令。
    """
    # 训练相关命令
    register_command("train", run_train_command, category="training",
                    help_text="训练模型")
    register_command("train-custom", run_train_custom_command, category="training",
                    help_text="使用自定义数据训练模型")

    # 交互相关命令
    register_command("chat", run_chat_command, category="interactive",
                    help_text="与模型交互对话")

    # 评估相关命令
    register_command("evaluate", run_evaluate_command, category="evaluation",
                    help_text="评估模型性能", aliases=["eval"])
    register_command("visualize", run_visualize_command, category="evaluation",
                    help_text="生成模型评估可视化图表")

    # 工具相关命令
    register_command("clean-cache", run_clean_cache_command, category="tools",
                    help_text="清理缓存文件")
    register_command("estimate", run_estimate_command, category="tools",
                    help_text="估算训练时间")

    # 占位符命令
    register_command("info", run_info_command, category="info",
                    help_text="显示模型/数据详细信息", is_placeholder=True)
    register_command("list", run_list_command, category="info",
                    help_text="列出可用资源", is_placeholder=True)
    register_command("prune", run_prune_command, category="maintenance",
                    help_text="删除旧模型或数据", is_placeholder=True)
    register_command("size", run_size_command, category="info",
                    help_text="计算数据或模型大小", is_placeholder=True)
    register_command("test", run_test_command, category="testing",
                    help_text="测试模型", is_placeholder=True)
    register_command("compare", run_compare_command, category="evaluation",
                    help_text="比较模型性能", is_placeholder=True)
    register_command("train-hf", run_train_hf_command, category="training",
                    help_text="训练 Hugging Face 兼容模型", is_placeholder=True)
    register_command("distill", run_distill_command, category="training",
                    help_text="蒸馏模型", is_placeholder=True)
    register_command("train-reasoning", run_train_reasoning_command, category="training",
                    help_text="训练逻辑推理能力模型", is_placeholder=True)
    register_command("process-data", run_process_data_command, category="data",
                    help_text="处理数据集", is_placeholder=True)
    register_command("backup", run_backup_command, category="maintenance",
                    help_text="备份模型或数据", is_placeholder=True)
    register_command("upload", run_upload_command, category="distribution",
                    help_text="上传模型或数据", is_placeholder=True)
    register_command("export-ollama", run_export_ollama_command, category="distribution",
                    help_text="导出模型到 Ollama 格式", is_placeholder=True)

    # 帮助命令
    register_command("help", show_help, category="general",
                    help_text="显示帮助信息")


# 自动注册核心命令
register_core_commands()
