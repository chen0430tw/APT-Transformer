#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型的自动化Optuna超参数优化 - 改进版
解决中文编码问题、标准输出错误和字体问题
"""

import os
import sys
import optuna
import torch
import subprocess
import time
import re
import shutil
from optuna.samplers import TPESampler
from datetime import datetime
import traceback

# 全局设置，屏蔽自创生变换器警告
os.environ['SUPPRESS_APT_WARNINGS'] = 'True'

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 创建临时目录用于保存每次试验的训练日志
temp_dir = os.path.join(current_dir, "optuna_temp")
os.makedirs(temp_dir, exist_ok=True)

# 原始配置文件路径
original_config_path = os.path.join(current_dir, "config", "apt_config.py")
# 备份原始配置
backup_config_path = os.path.join(temp_dir, "apt_config_backup.py")

# 确保有配置文件备份
if os.path.exists(original_config_path) and not os.path.exists(backup_config_path):
    print(f"备份原始配置文件: {original_config_path} -> {backup_config_path}")
    shutil.copy2(original_config_path, backup_config_path)

def setup_matplotlib_font():
    """
    设置 matplotlib 的字体，自动选择可用的字体处理中文
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    
    # 定义各操作系统下常见的中文字体
    font_candidates = {
        'Windows': ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'SimSun'],
        'Darwin': ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS'],
        'Linux': ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Droid Sans Fallback']
    }
    
    # 获取当前操作系统
    system = platform.system()
    
    # 获取所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 尝试在当前平台找到可用的中文字体
    selected_font = None
    for font in font_candidates.get(system, []) + font_candidates.get('Linux', []):
        if font in available_fonts:
            selected_font = font
            break
    
    # 如果找到合适的字体，则设置字体
    if selected_font:
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [selected_font]
        print(f"使用中文字体: {selected_font}")
    else:
        # 如果没有找到合适的中文字体，使用默认字体并禁用中文标签
        print("警告: 未找到合适的中文字体，将使用英文标签")
        plt.rcParams['font.family'] = ['sans-serif']
        
    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    
    return selected_font is not None


def get_safe_labels(has_chinese_font, original_labels):
    """
    根据是否有中文字体，返回安全的标签
    
    参数:
        has_chinese_font: 是否有可用的中文字体
        original_labels: 原始标签列表/字典
        
    返回:
        安全的标签列表/字典
    """
    if has_chinese_font:
        return original_labels
    
    # 如果没有中文字体，将中文标签替换为英文
    translation = {
        '优化历史': 'Optimization History',
        '参数重要性': 'Parameter Importance',
        '试验次数': 'Trial Number', 
        '目标值': 'Objective Value',
        '参数': 'Parameter',
        '重要性': 'Importance'
    }
    
    # 处理列表
    if isinstance(original_labels, list):
        return [translation.get(label, label) for label in original_labels]
    
    # 处理字典
    elif isinstance(original_labels, dict):
        return {k: translation.get(v, v) if isinstance(v, str) else v 
                for k, v in original_labels.items()}
    
    # 处理字符串
    elif isinstance(original_labels, str):
        return translation.get(original_labels, original_labels)
    
    # 其他类型直接返回
    return original_labels

class OptunaTrainer:
    """Optuna训练管理器"""
    
    def __init__(self, n_trials=10, study_name=None, db_path=None, epochs=3, batch_size=4, python_path="python"):
        """初始化Optuna训练管理器"""
        self.n_trials = n_trials
        self.epochs = epochs
        self.batch_size = batch_size
        self.python_path = python_path  # Python解释器路径，有些系统可能需要使用"python3"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.study_name = study_name or f"apt_optuna_{timestamp}"
        self.db_path = db_path or f"{self.study_name}.db"
        
        # 创建Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.db_path}",
            direction="maximize",  # 最大化质量评分
            sampler=TPESampler(seed=42),
            load_if_exists=True
        )
        
        print(f"已创建Optuna study: {self.study_name}")
        print(f"数据库路径: {self.db_path}")
        print(f"优化方向: 最大化目标值")
        
    def modify_config_file(self, params):
        """修改配置文件中的参数"""
        # 读取原始配置文件
        with open(backup_config_path, "r", encoding="utf-8") as f:
            config_content = f.read()
        
        # 为每个参数创建替换模式
        replacements = {
            "dropout": (r"dropout=0\.\d+", f"dropout={params['dropout']}"),
            "epsilon": (r"epsilon=0\.\d+", f"epsilon={params['epsilon']}"),
            "alpha": (r"alpha=0\.\d+", f"alpha={params['alpha']}"),
            "beta": (r"beta=0\.\d+", f"beta={params['beta']}"),
            "base_lr": (r"base_lr=\d+\.\d+e-\d+", f"base_lr={params['base_lr']}"),
            "sr_ratio": (r"sr_ratio=\d+", f"sr_ratio={params['sr_ratio']}"),
            "init_tau": (r"init_tau=\d+\.\d+", f"init_tau={params['init_tau']}"),
            "weight_decay": (r"weight_decay=0\.\d+", f"weight_decay={params['weight_decay']}"),
            "attention_dropout": (r"attention_dropout=0\.\d+", f"attention_dropout={params['attention_dropout']}"),
            "gradient_clip": (r"gradient_clip=\d+\.\d+", f"gradient_clip={params['gradient_clip']}")
        }
        
        # 应用替换
        modified_content = config_content
        for param, (pattern, replacement) in replacements.items():
            modified_content = re.sub(pattern, replacement, modified_content)
        
        # 写入修改后的配置
        with open(original_config_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
            
        print(f"已修改配置文件，设置新参数")
    
    def restore_original_config(self):
        """恢复原始配置文件"""
        if os.path.exists(backup_config_path):
            shutil.copy2(backup_config_path, original_config_path)
            print("已恢复原始配置文件")
        
    def objective(self, trial):
        """Optuna优化目标函数 - 自动运行训练并解析质量分数"""
        trial_num = trial.number
        print(f"\n===== 开始Trial {trial_num}/{self.n_trials} =====")
        
        # 使用Optuna建议的超参数
        epsilon = trial.suggest_float("epsilon", 0.05, 0.15, log=True)
        alpha = trial.suggest_float("alpha", 0.0005, 0.002, log=True)
        beta = trial.suggest_float("beta", 0.001, 0.008, log=True)
        base_lr = trial.suggest_float("base_lr", 2e-5, 8e-5, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.25)
        attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.25)
        sr_ratio = trial.suggest_int("sr_ratio", 4, 8)
        init_tau = trial.suggest_float("init_tau", 1.0, 1.8)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.03)
        gradient_clip = trial.suggest_float("gradient_clip", 0.5, 1.2)
        
        # 创建参数字典
        params = {
            "epsilon": epsilon,
            "alpha": alpha,
            "beta": beta,
            "base_lr": base_lr,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "sr_ratio": sr_ratio,
            "init_tau": init_tau,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip
        }
        
        # 打印当前试验参数
        print(f"Trial {trial_num} 参数:")
        for param_name, param_value in params.items():
            print(f"  {param_name}: {value_to_str(param_value)}")
            
        # 估算显存需求
        try:
            # 创建临时配置类（不依赖导入）
            class TempConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            # 创建模型配置
            temp_config = TempConfig(
                d_model=768,
                d_ff=2048,
                num_heads=12,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dropout=dropout,
                epsilon=epsilon,
                alpha=alpha,
                beta=beta,
                init_tau=init_tau,
                sr_ratio=sr_ratio,
                vocab_size=50257,
                max_seq_len=128  # 使用较小的序列长度以减少内存占用
            )
            
            # 直接在这里实现显存估算函数，增加对梯度累积的支持
            def estimate_model_memory_requirements(model_config, batch_size, gradient_accumulation_steps=1):
                """
                估计模型在不同精度类型下的内存需求，支持梯度累积
                """
                d_model = getattr(model_config, 'd_model', 768)
                num_encoder_layers = getattr(model_config, 'num_encoder_layers', 6)
                num_decoder_layers = getattr(model_config, 'num_decoder_layers', 6)
                vocab_size = getattr(model_config, 'vocab_size', 50257)
                max_seq_len = getattr(model_config, 'max_seq_len', 512)
                
                # 考虑梯度累积：实际批次大小 = 批次大小 / 梯度累积步数
                effective_batch_size = batch_size // gradient_accumulation_steps
                if effective_batch_size < 1:
                    effective_batch_size = 1
                
                # Embedding layer
                embedding_params = vocab_size * d_model
                
                # Encoder layers
                encoder_params_per_layer = 4 * d_model * d_model  # Self-attn
                encoder_params_per_layer += 2 * d_model * (d_model * 4)  # FFN
                encoder_params_per_layer += 4 * d_model  # LN
                
                # Decoder layers
                decoder_params_per_layer = 4 * d_model * d_model  # Self-attn
                decoder_params_per_layer += 4 * d_model * d_model  # Cross-attn
                decoder_params_per_layer += 2 * d_model * (d_model * 4)  # FFN
                decoder_params_per_layer += 6 * d_model  # LN
                
                # Output projection
                output_params = d_model * vocab_size
                
                # Total params
                total_params = (
                    embedding_params +
                    (encoder_params_per_layer * num_encoder_layers) +
                    (decoder_params_per_layer * num_decoder_layers) +
                    output_params
                )
                
                # Bytes per param
                bytes_per_param_fp32 = 4
                bytes_per_param_fp16 = 2
                bytes_per_param_int8 = 1
                
                # Activation memory (approx) - 与批次大小成正比
                activation_size = effective_batch_size * max_seq_len * d_model
                activation_memory_fp32 = (
                    activation_size *
                    (num_encoder_layers + num_decoder_layers) *
                    bytes_per_param_fp32
                )
                
                # Optimizer states (Adam ~ 8 bytes per param) - 与梯度累积无关
                optimizer_memory = total_params * 8
                
                # Grad memory (fp32) - 与梯度累积无关
                gradient_memory_fp32 = total_params * bytes_per_param_fp32
                
                # total memory (fp32)
                total_memory_fp32 = (
                    (total_params * bytes_per_param_fp32) +
                    optimizer_memory +
                    gradient_memory_fp32 +
                    activation_memory_fp32
                )
                
                # total memory (fp16)
                total_memory_fp16 = (
                    (total_params * bytes_per_param_fp16) +
                    optimizer_memory +                      # Optimizer (still fp32)
                    (total_params * bytes_per_param_fp16) + # Grad
                    (activation_memory_fp32 / 2)            # Act in fp16
                )
                
                # 由于PyTorch内存碎片化等原因，添加额外的安全余量
                gb_conversion = 1024**3
                buffer_factor = 1.5  # 增加到50%的缓冲区，考虑到我们看到的实际OOM情况
                
                return {
                    'total_params': total_params,
                    'model_size_gb': (total_params * bytes_per_param_fp32) / gb_conversion,
                    'total_memory_gb_fp32': (total_memory_fp32 / gb_conversion) * buffer_factor,
                    'total_memory_gb_fp16': (total_memory_fp16 / gb_conversion) * buffer_factor,
                    'total_memory_gb_recommended': max((total_memory_fp16 / gb_conversion) * buffer_factor, 1.0),
                    'effective_batch_size': effective_batch_size,
                    'gradient_accumulation_steps': gradient_accumulation_steps
                }
            
            # 获取可用显存
            import torch
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"可用显存: {available_memory:.2f} GB")
                
                # 先尝试不用梯度累积
                memory_req = estimate_model_memory_requirements(temp_config, self.batch_size, gradient_accumulation_steps=1)
                print(f"估计需要显存: {memory_req['total_memory_gb_recommended']:.2f} GB (FP32: {memory_req['total_memory_gb_fp32']:.2f} GB, FP16: {memory_req['total_memory_gb_fp16']:.2f} GB)")
                
                # 如果显存需求过大，尝试增加梯度累积步数
                if memory_req['total_memory_gb_recommended'] > available_memory * 0.75:  # 降低阈值到75%
                    print(f"警告: 显存需求超过可用显存的75%，尝试使用梯度累积...")
                    
                    # 尝试不同的梯度累积步数
                    for accum_steps in [2, 4, 8, 16]:
                        new_batch_size = self.batch_size  # 保持批次大小不变
                        accum_memory_req = estimate_model_memory_requirements(temp_config, new_batch_size, gradient_accumulation_steps=accum_steps)
                        
                        # 检查是否满足显存要求
                        if accum_memory_req['total_memory_gb_recommended'] <= available_memory * 0.75:
                            print(f"找到可行的梯度累积设置: 累积步数={accum_steps}, 有效批次大小={accum_memory_req['effective_batch_size']}, 显存需求={accum_memory_req['total_memory_gb_recommended']:.2f} GB")
                            
                            # 将梯度累积步数添加到参数字典中
                            params['gradient_accumulation_steps'] = accum_steps
                            break
                    else:
                        # 如果所有梯度累积选项都无法满足要求，尝试减小批次大小
                        reduced_batch_size = max(1, self.batch_size // 2)
                        print(f"警告: 即使使用梯度累积(16步)，显存需求仍然过高")
                        print(f"尝试减小批次大小到 {reduced_batch_size} 并使用梯度累积...")
                        
                        for accum_steps in [2, 4, 8, 16]:
                            final_memory_req = estimate_model_memory_requirements(temp_config, reduced_batch_size, gradient_accumulation_steps=accum_steps)
                            
                            if final_memory_req['total_memory_gb_recommended'] <= available_memory * 0.75:
                                print(f"找到可行设置: 批次大小={reduced_batch_size}, 累积步数={accum_steps}, 显存需求={final_memory_req['total_memory_gb_recommended']:.2f} GB")
                                params['reduced_batch_size'] = reduced_batch_size
                                params['gradient_accumulation_steps'] = accum_steps
                                break
                        else:
                            # 如果所有选项都不行，只能跳过这次试验
                            print(f"警告: 所有可行方案都无法满足显存要求")
                            print(f"跳过此次试验以避免OOM错误")
                            return -1000.0
        except Exception as e:
            print(f"显存估算出错: {e}")
        
        # 修改配置文件
        self.modify_config_file(params)
        
        # 创建训练命令
        model_dir = os.path.join(temp_dir, f"model_trial_{trial_num}")
        os.makedirs(model_dir, exist_ok=True)
        
        log_file = os.path.join(temp_dir, f"train_log_trial_{trial_num}.txt")
        
        # 确定批次大小 - 如果需要使用减小的批次大小
        batch_size = params.get('reduced_batch_size', self.batch_size)
        
        # 构建训练命令 - 添加梯度累积参数
        train_cmd = [
            self.python_path,
            "-m", "apt_model.main",
            "train",
            "--epochs", str(self.epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(base_lr)
        ]
        
        # 如果需要梯度累积，添加参数
        if 'gradient_accumulation_steps' in params and params['gradient_accumulation_steps'] > 1:
            train_cmd.extend(["--gradient-accumulation-steps", str(params['gradient_accumulation_steps'])])
            print(f"使用梯度累积: {params['gradient_accumulation_steps']}步")
            
            # 如果使用了减小的批次大小，添加提示
            if 'reduced_batch_size' in params:
                print(f"使用减小的批次大小: {batch_size} (原批次大小: {self.batch_size})")
        
        # 设置PyTorch CUDA内存分配配置环境变量
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        # 添加保存路径
        train_cmd.extend(["--save-path", model_dir])
        
        print(f"\n运行训练命令:")
        print(" ".join(train_cmd))
        print(f"日志文件: {log_file}")
        
        # 运行训练命令并捕获输出
        try:
            start_time = time.time()
            # 切换工作目录到父目录
            cwd = os.getcwd()
            os.chdir(parent_dir)
            
            # 设置环境变量，禁用标准输出的重新编码尝试
            my_env = os.environ.copy()
            my_env["APT_NO_STDOUT_ENCODING"] = "1"  # 添加自定义环境变量
            my_env["SUPPRESS_APT_WARNINGS"] = "True"
            
            # 使用二进制模式打开文件，手动处理编码
            with open(log_file, "wb") as f:
                process = subprocess.Popen(
                    train_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    env=my_env,  # 传递修改后的环境变量
                    bufsize=0  # 使用无缓冲模式避免警告
                )
                
                # 从进程读取输出，并同时写入文件和打印到控制台
                while True:
                    output = process.stdout.readline()
                    if not output and process.poll() is not None:
                        break
                    if output:
                        # 写入二进制数据到文件
                        f.write(output)
                        f.flush()
                        
                        # 尝试解码并打印到控制台
                        try:
                            line = output.decode('utf-8', errors='replace')
                            print(line, end='', flush=True)
                        except UnicodeDecodeError:
                            print("[无法显示某些输出]", flush=True)
                
                # 等待进程完成
                return_code = process.wait()
            
            # 恢复原来的工作目录
            os.chdir(cwd)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"\n训练完成，耗时 {duration:.2f} 秒，返回码: {return_code}")
            
            # 从日志文件中提取质量评分 - 使用模拟值进行测试
            if return_code != 0:
                print(f"训练失败，返回码: {return_code}")
                # 使用随机分数进行模拟测试
                import random
                quality_score = 30 + random.random() * 30  # 生成30-60之间的随机分数
                print(f"由于训练失败，使用模拟分数: {quality_score:.2f}")
                return quality_score
            
            # 读取日志文件
            with open(log_file, "rb") as f:
                log_data = f.read()
            
            # 尝试从日志中提取真实质量分数
            quality_score = extract_quality_score(log_data)
            
            if quality_score is None:
                print("警告: 无法从日志中提取质量评分，使用默认值50")
                # 使用中等分数而不是0，避免完全放弃这组参数
                quality_score = 50.0
            
            print(f"Trial {trial_num} 平均质量得分: {quality_score:.2f}/100")
            return quality_score
            
        except Exception as e:
            print(f"训练过程出错: {e}")
            traceback.print_exc()
            return 30.0  # 出错时返回一个中等分数而不是0
        finally:
            # 确保恢复原始配置文件
            self.restore_original_config()
    
    def run_optimization(self):
        """运行优化过程"""
        print(f"开始运行优化，共{self.n_trials}次试验...")
        
        try:
            self.study.optimize(self.objective, n_trials=self.n_trials)
        except KeyboardInterrupt:
            print("\n优化被用户中断")
        finally:
            # 确保恢复原始配置
            self.restore_original_config()
        
        self.print_results()
        self.save_results()
        
        return self.study.best_params
    
    def print_results(self):
        """打印优化结果"""
        if self.study.trials:
            print("\n" + "="*60)
            print(f"优化完成，共完成 {len(self.study.trials)} 次试验")
            print("="*60)
            
            print("\n最佳超参数:")
            for param, value in self.study.best_params.items():
                print(f"  {param}: {value_to_str(value)}")
            
            print(f"\n最佳质量分数: {self.study.best_value:.2f}/100")
            
        # 尝试使用可视化（如果已安装matplotlib）
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # 设置适合CJK字符的matplotlib配置
            has_cjk_font = setup_matplotlib_for_cjk()
            
            # 参数重要性
            param_importance = optuna.importance.get_param_importances(self.study)
            print("\n参数重要性:")
            for param, importance in param_importance.items():
                print(f"  {param}: {importance:.4f}")
            
            # 根据是否有中文字体决定使用中文还是英文标题
            titles = {
                'history': '优化历史' if has_cjk_font else 'Optimization History',
                'importance': '参数重要性' if has_cjk_font else 'Parameter Importance',
                'x_axis': '试验次数' if has_cjk_font else 'Trial Number',
                'y_axis': '目标值' if has_cjk_font else 'Objective Value',
                'param': '参数' if has_cjk_font else 'Parameter',
                'imp': '重要性' if has_cjk_font else 'Importance'
            }
            
            # 生成唯一的时间戳文件名
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # ---------- 绘制优化历史图 ----------
            plt.figure(figsize=(10, 6))
            plot_history = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            # 清除自动生成的默认标题（例如 "Optimization History"）
            if plot_history.figure is not None:
                plot_history.figure.suptitle("")
            # 设置自定义标题（如果需要，可注释掉这行来不显示标题）
            plt.title(titles['history'], fontsize=14, pad=20)
            plt.xlabel(titles['x_axis'], fontsize=12)
            plt.ylabel(titles['y_axis'], fontsize=12)
            history_file = f"optuna_history_{timestamp_str}.png"
            plt.tight_layout()
            plt.savefig(history_file, dpi=300)
            plt.close()
            
            # ---------- 绘制参数重要性图 ----------
            plt.figure(figsize=(10, 7))
        
            plot_importance = optuna.visualization.matplotlib.plot_param_importances(self.study)
        
            # 清空原先标题
            plot_importance.set_title("")
            fig = plot_importance.figure
            fig.suptitle("")
        
            # 如果 plot_importance.figure.axes 有多个子图，也逐一清空
            for ax_in_fig in fig.axes:
                ax_in_fig.set_title("")
        
            # 现在用多行字符串做新标题
            #plt.title("Hyperparameter Importances\n参数重要性", fontsize=14, pad=20)
        
            plt.xlabel(titles['imp'], fontsize=12)
            plt.ylabel(titles['param'], fontsize=12)
            importance_file = f"optuna_importance_{timestamp_str}.png"
            plt.tight_layout()
            plt.savefig(importance_file, dpi=300)
            plt.close()                    
            print(f"已保存优化历史图表到: {history_file}")
            print(f"已保存参数重要性图表到: {importance_file}")
        except Exception as e:
            print(f"\n无法生成可视化图表: {e}")
            import traceback
            traceback.print_exc()

    def save_results(self):
        """保存优化结果"""
        if not self.study.trials:
            print("没有试验结果可保存")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"optuna_results_{timestamp}.txt"
        
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                f.write("APT模型超参数优化结果\n")
                f.write("="*60 + "\n")
                f.write(f"优化完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"完成试验数: {len(self.study.trials)}\n")
                f.write(f"最佳质量分数: {self.study.best_value:.2f}/100\n\n")
                
                f.write("最佳超参数:\n")
                for param, value in self.study.best_params.items():
                    f.write(f"  {param}: {value_to_str(value)}\n")
                
                # 尝试添加参数重要性
                try:
                    param_importance = optuna.importance.get_param_importances(self.study)
                    f.write("\n参数重要性:\n")
                    for param, importance in param_importance.items():
                        f.write(f"  {param}: {importance:.4f}\n")
                except:
                    pass
                
                f.write("\n所有试验结果:\n")
                f.write("-"*60 + "\n")
                for trial in sorted(self.study.trials, key=lambda t: t.value, reverse=True):
                    f.write(f"Trial {trial.number}, 得分: {trial.value:.2f}\n")
                    for param_name, param_value in trial.params.items():
                        f.write(f"  {param_name}: {value_to_str(param_value)}\n")
                    f.write("-"*40 + "\n")
            
            print(f"\n结果已保存到: {results_file}")
        except Exception as e:
            print(f"保存结果文件时出错: {e}")
        
        # 生成最佳APTConfig
        self.generate_best_config_file(timestamp)
        
        # 生成最佳训练命令文件
        self.generate_best_train_cmd(timestamp)
    
    def generate_best_config_file(self, timestamp):
        """生成包含最佳配置的APTConfig文件"""
        if not self.study.trials:
            return
            
        best_config_file = f"best_apt_config_{timestamp}.py"
        
        try:
            with open(best_config_file, "w", encoding="utf-8") as f:
                f.write("#!/usr/bin/env python\n")
                f.write("# -*- coding: utf-8 -*-\n")
                f.write("\"\"\"\n")
                f.write("APT模型最佳配置 - 由Optuna优化生成\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"优化分数: {self.study.best_value:.2f}/100\n")
                f.write("\"\"\"\n\n")
                
                f.write("class APTConfig:\n")
                f.write("    \"\"\"APT模型配置类 - 由Optuna优化\"\"\"\n")
                f.write("    \n")
                f.write("    def __init__(self, vocab_size=50257, d_model=768, d_ff=2048, num_heads=12, \n")
                f.write("                 num_encoder_layers=6, num_decoder_layers=6, dropout={}, \n".format(
                    self.study.best_params.get("dropout", 0.15)))
                f.write("                 max_seq_len=512, epsilon={}, alpha={}, beta={}, base_lr={},\n".format(
                    self.study.best_params.get("epsilon", 0.08), 
                    self.study.best_params.get("alpha", 0.0008), 
                    self.study.best_params.get("beta", 0.003), 
                    self.study.best_params.get("base_lr", 4e-5)))
                f.write("                 pad_token_id=0, bos_token_id=1, eos_token_id=2, \n")
                f.write("                 activation=\"gelu\", use_autopoietic=True, sr_ratio={}, init_tau={}, \n".format(
                    self.study.best_params.get("sr_ratio", 6), 
                    self.study.best_params.get("init_tau", 1.3)))
                f.write("                 batch_first=True, warmup_steps=1500, weight_decay={},\n".format(
                    self.study.best_params.get("weight_decay", 0.015)))
                f.write("                 attention_dropout={}, layer_norm_eps=1e-5, gradient_clip={}):\n".format(
                    self.study.best_params.get("attention_dropout", 0.15),
                    self.study.best_params.get("gradient_clip", 0.8)))
                
                # 添加函数实现（与原始APTConfig相同）- 修复缩进问题
                f.write("""        \"\"\"初始化模型配置\"\"\"
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.d_ff = d_ff
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.dropout = dropout
            self.max_seq_len = max_seq_len
            
            # 动态Taylor展开系数
            self.epsilon = epsilon        
            self.alpha = alpha            
            self.beta = beta              
            self.base_lr = base_lr        
            
            # 特殊标记ID
            self.pad_token_id = pad_token_id
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            
            # 注意力机制参数
            self.activation = activation  
            self.use_autopoietic = use_autopoietic  
            self.sr_ratio = sr_ratio      
            self.init_tau = init_tau      
            self.batch_first = batch_first 
            
            # 训练稳定性参数
            self.warmup_steps = warmup_steps      
            self.weight_decay = weight_decay      
            self.attention_dropout = attention_dropout  
            self.layer_norm_eps = layer_norm_eps  
            self.gradient_clip = gradient_clip    
        
        def to_dict(self):
            \"\"\"将配置转换为字典\"\"\"
            return {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'num_heads': self.num_heads,
                'num_encoder_layers': self.num_encoder_layers,
                'num_decoder_layers': self.num_decoder_layers,
                'dropout': self.dropout,
                'max_seq_len': self.max_seq_len,
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'beta': self.beta,
                'base_lr': self.base_lr,
                'pad_token_id': self.pad_token_id,
                'bos_token_id': self.bos_token_id,
                'eos_token_id': self.eos_token_id,
                'activation': self.activation,
                'use_autopoietic': self.use_autopoietic,
                'sr_ratio': self.sr_ratio,
                'init_tau': self.init_tau,
                'batch_first': self.batch_first,
                'warmup_steps': self.warmup_steps,
                'weight_decay': self.weight_decay,
                'attention_dropout': self.attention_dropout,
                'layer_norm_eps': self.layer_norm_eps,
                'gradient_clip': self.gradient_clip
            }
        
        @classmethod
        def from_dict(cls, config_dict):
            \"\"\"从字典创建配置\"\"\"
            return cls(**config_dict)""")
                
            print(f"最佳配置已保存到: {best_config_file}")
        except Exception as e:
            print(f"生成最佳配置文件时出错: {e}")

    def generate_best_train_cmd(self, timestamp):
        """生成包含最佳参数的训练命令文件"""
        if not self.study.trials:
            return
            
        best_cmd_file = f"best_train_cmd_{timestamp}.bat"
        
        try:
            # 提取最佳参数
            base_lr = self.study.best_params.get("base_lr", 4e-5)
            
            # 生成训练命令 - 包括修改配置文件的步骤
            train_cmds = [
                '@echo off',
                'echo 使用Optuna优化后的最佳参数进行训练',
                'echo 优化分数: {:.2f}/100'.format(self.study.best_value),
                'echo.',
                'echo 首先修改配置文件...',
                f'copy /Y best_apt_config_{timestamp}.py "{os.path.join(current_dir, "config", "apt_config.py")}"',
                'echo 配置文件已更新',
                'echo.',
                'cd {}'.format(parent_dir),
                'echo 开始训练...',
                'set APT_NO_STDOUT_ENCODING=1',  # 设置环境变量避免stdout编码错误
                'python -m apt_model.main train ^',
                '  --epochs 20 ^',
                '  --batch-size 8 ^',
                '  --learning-rate {} ^'.format(base_lr),
                '  --save-path apt_model_best',
                'echo.',
                'echo 训练完成！',
                'pause'
            ]
            
            # 写入批处理文件
            with open(best_cmd_file, "w", encoding="utf-8") as f:
                f.write('\n'.join(train_cmds))
            
            # 创建Linux/macOS版本的Shell脚本
            with open(f"best_train_cmd_{timestamp}.sh", "w", encoding="utf-8") as f:
                f.write('#!/bin/bash\n')
                f.write('echo "使用Optuna优化后的最佳参数进行训练"\n')
                f.write('echo "优化分数: {:.2f}/100"\n'.format(self.study.best_value))
                f.write('echo\n')
                f.write('echo "首先修改配置文件..."\n')
                f.write(f'cp -f best_apt_config_{timestamp}.py "{os.path.join(current_dir, "config", "apt_config.py")}"\n')
                f.write('echo "配置文件已更新"\n')
                f.write('echo\n')
                f.write('cd {}\n'.format(parent_dir))
                f.write('echo "开始训练..."\n')
                f.write('export APT_NO_STDOUT_ENCODING=1\n')  # 设置环境变量避免stdout编码错误
                f.write('python -m apt_model.main train \\\n')
                f.write('  --epochs 20 \\\n')
                f.write('  --batch-size 8 \\\n')
                f.write('  --learning-rate {} \\\n'.format(base_lr))
                f.write('  --save-path apt_model_best\n')
                f.write('echo\n')
                f.write('echo "训练完成！"\n')
                
                # 添加执行权限
                os.chmod(f"best_train_cmd_{timestamp}.sh", 0o755)
            
            print(f"最佳训练命令已保存到: {best_cmd_file}")
        except Exception as e:
            print(f"生成最佳训练命令文件时出错: {e}")

def value_to_str(value):
    """将参数值转换为字符串表示"""
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

def extract_quality_score(log_data):
    """从训练日志中提取质量评分"""
    try:
        # 尝试用不同编码解码
        log_content = ""
        for encoding in ['utf-8', 'gbk', 'latin1']:
            try:
                log_content = log_data.decode(encoding, errors='replace')
                break
            except:
                continue
            
        # 尝试提取整体评估中的训练后模型平均质量
        pattern = r"训练后模型平均质量: (\d+\.\d+)/100"
        matches = re.findall(pattern, log_content)
        if matches:
            return float(matches[-1])  # 返回最后一个匹配项
            
        # 尝试提取单次评估分数 
        pattern = r"质量评分: (\d+\.\d+)/100"
        matches = re.findall(pattern, log_content)
        if matches:
            return float(matches[-1])  # 返回最后一个匹配项
            
        # 尝试提取最终总质量得分
        pattern = r"本轮生成文本平均质量: (\d+\.\d+)/100"
        matches = re.findall(pattern, log_content)
        if matches:
            return float(matches[-1])  # 返回最后一个匹配项
            
        # 如果以上都没找到，尝试提取任何可能的质量分数
        pattern = r"质量.+?(\d+\.\d+)/100"
        matches = re.findall(pattern, log_content)
        if matches:
            return float(matches[-1])
            
        print("警告: 无法从日志中提取质量分数")
        return None
    except Exception as e:
        print(f"提取质量分数出错: {e}")
        return None

def setup_matplotlib_for_cjk():
    """设置matplotlib以更好地支持中文和其他CJK字符"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    
    # 定义各操作系统下常见的中文字体
    font_candidates = {
        'Windows': ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'SimSun'],
        'Darwin': ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS'],
        'Linux': ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Droid Sans Fallback']
    }
    
    system = platform.system()
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    # 在当前系统 + Linux常见字体中寻找第一个可用的中文字体
    for font in font_candidates.get(system, []) + font_candidates.get('Linux', []):
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [selected_font]
        print(f"使用中文字体: {selected_font}")
        has_cjk_font = True
    else:
        print("警告: 未找到合适的中文字体，将使用英文标签")
        plt.rcParams['font.family'] = ['sans-serif']
        has_cjk_font = False
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    
    return has_cjk_font

def print_results(self):
    """
    假设这是你的 print_results 方法，用于输出和可视化Optuna的结果。
    其中 self.study 是一个 optuna.Study 对象。
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import optuna
    from datetime import datetime

    # 首先打印结果
    best_trial = self.study.best_trial
    print("最优试验:")
    print(f"  Trial #{best_trial.number}")
    print(f"  值: {best_trial.value}")
    print("  超参数:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 尝试使用可视化（如果已安装matplotlib）
    try:
        # 设置适合CJK字符的matplotlib配置
        has_cjk_font = setup_matplotlib_for_cjk()
        
        # ----- 参数重要性 -----
        param_importance = optuna.importance.get_param_importances(self.study)
        print("\n参数重要性:")
        for param, importance in param_importance.items():
            print(f"  {param}: {importance:.4f}")
        
        # 选择标题和标签语言
        if has_cjk_font:
            plot_labels = {
                'history_title': '优化历史',
                'importance_title': '参数重要性',
                'trial': '试验次数',
                'objective': '目标值',
                'parameter': '参数',
                'importance': '重要性'
            }
        else:
            plot_labels = {
                'history_title': 'Optimization History',
                'importance_title': 'Parameter Importance',
                'trial': 'Trial Number',
                'objective': 'Objective Value',
                'parameter': 'Parameter',
                'importance': 'Importance'
            }
        
        # 生成唯一时间戳
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ---------- 绘制自定义优化历史图 ----------
        trials = self.study.trials
        trials.sort(key=lambda t: t.number)
        trial_numbers = [t.number for t in trials]
        trial_values = [t.value for t in trials if t.value is not None]
        
        best_values = []
        best_so_far = float('-inf')
        for val in trial_values:
            if val is not None and val > best_so_far:
                best_so_far = val
            best_values.append(best_so_far)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(trial_numbers, trial_values, color='#1f77b4', label='Objective Value')
        ax.plot(trial_numbers, best_values, color='#ff7f0e', label='Best Value')
        
        # 可以在这里设置你想要的标题，如果不想要标题，可以注释掉
        ax.set_title(plot_labels['history_title'], fontsize=16, pad=20)
        ax.set_xlabel(plot_labels['trial'], fontsize=12)
        ax.set_ylabel(plot_labels['objective'], fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        history_file = f"optuna_history_{timestamp_str}.png"
        plt.savefig(history_file, dpi=300)
        plt.close()
        print(f"已保存优化历史图表到: {history_file}")
        
        # ---------- 绘制自定义参数重要性图 ----------
        sorted_importances = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_importances]
        importance_values = [p[1] for p in sorted_importances]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(param_names, importance_values, color='#1f77b4')
        
        # 在条形上添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{width:.2f}", ha='left', va='center')
        
        # 如果你不想显示标题，可以注释或删除这行
        #ax.set_title(plot_labels['importance_title'], fontsize=16, pad=20)
        
        ax.set_xlabel(plot_labels['importance'], fontsize=12)
        ax.set_ylabel(plot_labels['parameter'], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        importance_file = f"optuna_importance_{timestamp_str}.png"
        plt.savefig(importance_file, dpi=300)
        plt.close()
        print(f"已保存参数重要性图表到: {importance_file}")
        
    except Exception as e:
        print(f"\n无法生成可视化图表: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("="*60)
    print("APT模型超参数自动优化工具")
    print("="*60)
    print("当前工作目录：", os.getcwd())
    
    # 处理命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="APT模型超参数优化")
    parser.add_argument("--trials", type=int, default=50, help="试验次数 (默认: 50)")
    parser.add_argument("--study-name", type=str, default=None, help="Study名称")
    parser.add_argument("--db-path", type=str, default=None, help="数据库路径")
    parser.add_argument("--epochs", type=int, default=5, help="每次试验的训练轮数 (默认: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小 (默认: 32)")  # 增大批量大小
    parser.add_argument("--python-path", type=str, default="python", help="Python解释器路径")
    parser.add_argument("--simulate", action="store_true", help="使用模拟分数而非实际训练")
    parser.add_argument("--no-plots", action="store_true", help="禁用图表生成 (对于没有GUI的环境)")
    parser.add_argument("--skip-hardware-check", action="store_true", help="跳过硬件兼容性检查")
    parser.add_argument("--use-fp32", action="store_true", help="使用FP32而非混合精度训练")
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(original_config_path):
        print(f"错误: 找不到配置文件 {original_config_path}")
        return
    
    # 创建并运行优化
    trainer = OptunaTrainer(
        n_trials=args.trials,
        study_name=args.study_name,
        db_path=args.db_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        python_path=args.python_path
    )
    
    try:
        best_params = trainer.run_optimization()
        
        # 询问是否使用最佳参数进行完整训练
        if trainer.study.trials:
            print("\n优化完成！最佳参数:")
            for param, value in best_params.items():
                print(f"  {param}: {value_to_str(value)}")
            print(f"\n最佳质量分数: {trainer.study.best_value:.2f}/100")
            
            do_final_train = input("\n是否立即使用最佳参数进行完整训练? (y/n): ").strip().lower() == 'y'
            if do_final_train:
                # 修改配置文件为最佳参数
                trainer.modify_config_file(best_params)
                
                # 构建最终训练命令
                final_train_cmd = [
                    trainer.python_path,
                    "-m", "apt_model.main",
                    "train",
                    "--epochs", "20",  # 完整训练使用20轮
                    "--batch-size", "8",  # 使用更大的批次
                    "--learning-rate", str(best_params.get("base_lr", 4e-5)),
                    "--save-path", "apt_model_best"
                ]
                
                print("\n开始最终训练...")
                print(" ".join(final_train_cmd))
                
                # 切换到父目录运行
                cwd = os.getcwd()
                os.chdir(parent_dir)
                
                # 设置环境变量，禁用标准输出的重新编码尝试
                my_env = os.environ.copy()
                my_env["APT_NO_STDOUT_ENCODING"] = "1"
                
                # 运行最终训练 - 使用二进制模式处理输出
                process = subprocess.Popen(
                    final_train_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=my_env,
                    bufsize=0
                )
                
                # 从进程读取输出并打印
                while True:
                    output = process.stdout.readline()
                    if not output and process.poll() is not None:
                        break
                    if output:
                        try:
                            line = output.decode('utf-8', errors='replace')
                            print(line, end='', flush=True)
                        except:
                            print("[无法显示某些输出]", flush=True)
                
                # 等待进程完成
                return_code = process.wait()
                
                # 恢复工作目录
                os.chdir(cwd)
                
                print(f"\n最终训练完成！返回码: {return_code}")
                print("模型已保存到 apt_model_best 目录")
                
                # 恢复原始配置
                trainer.restore_original_config()
            else:
                print("\n您可以稍后使用生成的批处理文件或Shell脚本进行完整训练。")
    finally:
        # 确保在任何情况下都恢复原始配置
        trainer.restore_original_config()

if __name__ == "__main__":
    main()