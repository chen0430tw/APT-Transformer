#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Optuna优化APT模型超参数
适配APT项目正确结构
"""

import os
import sys
import optuna
from optuna.samplers import TPESampler
import torch
from datetime import datetime
import traceback

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # 添加父目录，确保能导入apt_model

# 尝试导入APT模型所需模块 - 由于脚本位于apt_model内部，使用相对路径导入
try:
    # 位于apt_model内部，使用相对导入
    from config.apt_config import APTConfig
    from utils import set_seed, get_device
    from modeling.apt_model import APTLargeModel
    from generation.generator import generate_natural_text
    from generation.evaluator import evaluate_text_quality
    from training.trainer import train_model, get_training_texts, _test_generation_after_epoch
    print("已成功导入所有模块")
    all_imports_successful = True
except ImportError as e:
    print(f"无法导入模块: {e}")
    all_imports_successful = False

# 如果导入失败，使用简化版函数
if not all_imports_successful:
    def set_seed(seed):
        """设置随机种子"""
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"已设置随机种子: {seed}")
        
    def get_device(force_cpu=False):
        """获取设备"""
        if not force_cpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("使用CPU设备")
        return device

class OptunaTrainer:
    """Optuna训练管理器"""
    
    def __init__(self, n_trials=10, study_name=None, db_path=None):
        """初始化Optuna训练管理器"""
        self.n_trials = n_trials
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.study_name = study_name or f"apt_optuna_{timestamp}"
        self.db_path = db_path or f"{self.study_name}.db"
        
        # 设置随机种子
        set_seed(42)
        
        # 获取设备
        self.device = get_device()
        
        # 创建Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{self.db_path}",
            direction="maximize",
            sampler=TPESampler(seed=42),
            load_if_exists=True
        )
        
        print(f"已创建Optuna study: {self.study_name}")
        print(f"数据库路径: {self.db_path}")
        print(f"优化方向: 最大化目标值")
        
    def objective(self, trial):
        """优化目标函数"""
        print(f"\n开始Trial {trial.number}/{self.n_trials}...")
        
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
        
        # 打印当前试验参数
        print(f"Trial {trial.number} 参数:")
        for param_name, param_value in trial.params.items():
            print(f"  {param_name}: {param_value}")
        
        # 将当前参数保存到临时配置文件，虽然不使用，但保留作为记录
        config_file = f"trial_{trial.number}_config.py"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env python\n")
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("\"\"\"临时APT配置 - Trial {}\"\"\"\n\n".format(trial.number))
            f.write("class APTConfig:\n")
            f.write("    \"\"\"APT模型配置类 - Optuna试验\"\"\"\n")
            f.write("    \n")
            f.write("    def __init__(self, vocab_size=50257, d_model=768, d_ff=2048, num_heads=12, \n")
            f.write("                 num_encoder_layers=6, num_decoder_layers=6, dropout={}, \n".format(dropout))
            f.write("                 max_seq_len=512, epsilon={}, alpha={}, beta={}, base_lr={},\n".format(
                epsilon, alpha, beta, base_lr))
            f.write("                 pad_token_id=0, bos_token_id=1, eos_token_id=2, \n")
            f.write("                 activation=\"gelu\", use_autopoietic=True, sr_ratio={}, init_tau={}, \n".format(
                sr_ratio, init_tau))
            f.write("                 batch_first=True, warmup_steps=1500, weight_decay={},\n".format(weight_decay))
            f.write("                 attention_dropout={}, layer_norm_eps=1e-5, gradient_clip={}):\n".format(
                attention_dropout, gradient_clip))
            
            # 添加实现代码
            f.write("""        \"\"\"初始化模型配置\"\"\"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.base_lr = base_lr
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.activation = activation
        self.use_autopoietic = use_autopoietic
        self.sr_ratio = sr_ratio
        self.init_tau = init_tau
        self.batch_first = batch_first
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
            
        print(f"已保存Trial {trial.number}配置到临时文件: {config_file}")
        
        # 生成与main.py兼容的训练命令
        # 注意将参数名称转换为命令行格式（例如，base_lr -> --learning-rate）
        train_cmd = [
            f"cd {parent_dir}",  # 返回上一级目录
            f"python -m apt_model.main train",
            f"  --epochs 3",
            f"  --batch-size 4",
            f"  --learning-rate {base_lr}",
            f"  --dropout {dropout}",
            f"  --epsilon {epsilon}",
            f"  --alpha {alpha}",
            f"  --beta {beta}",
            f"  --sr-ratio {sr_ratio}",
            f"  --init-tau {init_tau}",
            f"  --weight-decay {weight_decay}",
            f"  --attention-dropout {attention_dropout}",
            f"  --gradient-clip {gradient_clip}",
            f"  --save-path apt_model_trial_{trial.number}"
        ]
        
        # 由于导入问题，默认使用手动模式
        manual_mode = True
        
        if manual_mode:
            print(f"\n请使用如下命令进行训练:")
            print("\n".join(train_cmd))
            print("\n或者使用您适合的其他方式进行训练")
            
            score = float(input("\n请输入训练和评估后的质量分数 (0-100): ").strip())
            return score
        
        # 自动训练模式 - 仅当导入成功时可用，但我们不使用
        return 0.0
    
    def run_optimization(self):
        """运行优化过程"""
        print(f"开始运行优化，共{self.n_trials}次试验...")
        
        try:
            self.study.optimize(self.objective, n_trials=self.n_trials)
        except KeyboardInterrupt:
            print("\n优化被用户中断")
        
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
                print(f"  {param}: {value}")
            
            print(f"\n最佳质量分数: {self.study.best_value:.2f}/100")
            
            # 尝试使用可视化（如果已安装matplotlib）
            try:
                import matplotlib.pyplot as plt
                
                # 参数重要性
                param_importance = optuna.importance.get_param_importances(self.study)
                print("\n参数重要性:")
                for param, importance in param_importance.items():
                    print(f"  {param}: {importance:.4f}")
            except:
                print("\n无法生成参数重要性（需要安装 matplotlib）")
        else:
            print("\n没有完成的试验")
    
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
                    f.write(f"  {param}: {value}\n")
                
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
                        f.write(f"  {param_name}: {param_value}\n")
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
                
                # 添加函数实现（与原始APTConfig相同）
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
        """生成包含最佳参数的训练命令"""
        if not self.study.trials:
            return
            
        best_cmd_file = f"best_train_cmd_{timestamp}.bat"
        
        try:
            # 提取最佳参数
            best_params = self.study.best_params
            base_lr = best_params.get("base_lr", 4e-5)
            dropout = best_params.get("dropout", 0.15)
            epsilon = best_params.get("epsilon", 0.08)
            alpha = best_params.get("alpha", 0.0008)
            beta = best_params.get("beta", 0.003)
            sr_ratio = best_params.get("sr_ratio", 6)
            init_tau = best_params.get("init_tau", 1.3)
            weight_decay = best_params.get("weight_decay", 0.015)
            attention_dropout = best_params.get("attention_dropout", 0.15)
            gradient_clip = best_params.get("gradient_clip", 0.8)
            
            # 生成训练命令
            train_cmds = [
                '@echo off',
                'echo 使用Optuna优化后的最佳参数进行训练',
                'echo 优化分数: {:.2f}/100'.format(self.study.best_value),
                'echo.',
                'cd {}'.format(parent_dir),
                'python -m apt_model.main train ^',
                '  --epochs 20 ^',
                '  --batch-size 8 ^',
                '  --learning-rate {} ^'.format(base_lr),
                '  --dropout {} ^'.format(dropout),
                '  --epsilon {} ^'.format(epsilon),
                '  --alpha {} ^'.format(alpha),
                '  --beta {} ^'.format(beta),
                '  --sr-ratio {} ^'.format(sr_ratio),
                '  --init-tau {} ^'.format(init_tau),
                '  --weight-decay {} ^'.format(weight_decay),
                '  --attention-dropout {} ^'.format(attention_dropout),
                '  --gradient-clip {} ^'.format(gradient_clip),
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
                f.write('cd {}\n'.format(parent_dir))
                f.write('python -m apt_model.main train \\\n')
                f.write('  --epochs 20 \\\n')
                f.write('  --batch-size 8 \\\n')
                f.write('  --learning-rate {} \\\n'.format(base_lr))
                f.write('  --dropout {} \\\n'.format(dropout))
                f.write('  --epsilon {} \\\n'.format(epsilon))
                f.write('  --alpha {} \\\n'.format(alpha))
                f.write('  --beta {} \\\n'.format(beta))
                f.write('  --sr-ratio {} \\\n'.format(sr_ratio))
                f.write('  --init-tau {} \\\n'.format(init_tau))
                f.write('  --weight-decay {} \\\n'.format(weight_decay))
                f.write('  --attention-dropout {} \\\n'.format(attention_dropout))
                f.write('  --gradient-clip {} \\\n'.format(gradient_clip))
                f.write('  --save-path apt_model_best\n')
                f.write('echo\n')
                f.write('echo "训练完成！"\n')
            
            print(f"最佳训练命令已保存到: {best_cmd_file}")
        except Exception as e:
            print(f"生成最佳训练命令文件时出错: {e}")

def main():
    """主函数"""
    print("="*60)
    print("APT模型超参数优化工具")
    print("="*60)
    print("当前工作目录：", os.getcwd())
    
    # 处理命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="APT模型超参数优化")
    parser.add_argument("--trials", type=int, default=10, help="试验次数 (默认: 10)")
    parser.add_argument("--study-name", type=str, default=None, help="Study名称")
    parser.add_argument("--db-path", type=str, default=None, help="数据库路径")
    args = parser.parse_args()
    
    if not all_imports_successful:
        print("由于导入问题，将使用手动评估模式")
    
    # 创建并运行优化
    trainer = OptunaTrainer(
        n_trials=args.trials,
        study_name=args.study_name,
        db_path=args.db_path
    )
    
    best_params = trainer.run_optimization()
    
    # 显示最佳参数的训练命令
    if trainer.study.trials:
        print("\n以下是使用最佳参数的APT训练命令:")
        print(f"cd {parent_dir}")
        print(f"python -m apt_model.main train \\")
        print(f"  --epochs 20 \\")
        print(f"  --batch-size 8 \\")
        print(f"  --learning-rate {best_params.get('base_lr', 4e-5)} \\")
        print(f"  --dropout {best_params.get('dropout', 0.15)} \\")
        print(f"  --epsilon {best_params.get('epsilon', 0.08)} \\")
        print(f"  --alpha {best_params.get('alpha', 0.0008)} \\")
        print(f"  --beta {best_params.get('beta', 0.003)} \\")
        print(f"  --sr-ratio {best_params.get('sr_ratio', 6)} \\")
        print(f"  --init-tau {best_params.get('init_tau', 1.3)} \\")
        print(f"  --weight-decay {best_params.get('weight_decay', 0.015)} \\")
        print(f"  --attention-dropout {best_params.get('attention_dropout', 0.15)} \\")
        print(f"  --gradient-clip {best_params.get('gradient_clip', 0.8)}")
        
        print("\n您也可以直接使用生成的批处理文件或Shell脚本进行训练。")

if __name__ == "__main__":
    main()