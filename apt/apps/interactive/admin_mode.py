#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Administrator Mode
APT模型管理员模式 - 提供高级调试和模型控制功能

警告：此模块仅供研究和开发目的使用
"""

import os
import sys
import time
import logging
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
import torch.nn.functional as F


class APTAdminMode:
    """
    APT模型管理员模式
    提供高级模型调试功能和参数控制
    """
    
    def __init__(
        self,
        model_path: str = "apt_model",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_length: int = 100,
        logger: Optional[logging.Logger] = None,
        admin_password: str = "aptadmin",
        tokenizer_type: Optional[str] = None,
        force_cpu: bool = False
    ):
        """
        初始化APT模型管理员模式
        
        参数:
            model_path: 模型路径
            temperature: 生成温度
            top_p: top-p采样参数
            max_length: 最大生成长度
            logger: 日志记录器
            admin_password: 管理员密码
            tokenizer_type: 分词器类型
            force_cpu: 是否强制使用CPU
        """
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.admin_password = admin_password
        self.tokenizer_type = tokenizer_type
        self.force_cpu = force_cpu
        
        # 设置日志
        self.logger = logger or self._setup_logger()
        
        # 状态变量
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        self.context = []  # 对话历史
        self.authenticated = False
        self.safety_layer_enabled = True
        self.advanced_debugging = False
        self.show_metrics = True
        self.raw_mode = False
        self.show_token_probabilities = False
        self.custom_system_prompt = None
        
        # 系统提示
        self.system_prompts = {
            "welcome": f"\n{'='*60}\n🔧 APT模型管理员模式\n{'='*60}\n输入 '/login <密码>' 进行身份验证\n输入 '/help' 查看基本命令\n{'='*60}",
            "auth_success": "\n✅ 管理员身份验证成功!\n进入管理员模式! 输入 '/admin' 查看管理员命令\n",
            "auth_failed": "\n❌ 身份验证失败！密码错误。\n",
            "need_auth": "\n⚠️  此命令需要管理员权限。请先使用 '/login <密码>' 进行身份验证。\n"
        }
        
        # 统计信息
        self.stats = {
            'total_interactions': 0,
            'avg_generation_time': 0,
            'safety_bypasses': 0,
            'parameter_overrides': 0
        }
        
        self.logger.info("APT管理员模式初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('APTAdminMode')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    # ==================== 模型加载 ====================
    
    def load_model(self):
        """加载APT模型"""
        try:
            self.logger.info(f"正在加载模型: {self.model_path}")
            
            # 尝试导入APT模型
            try:
                from apt_model.modeling.apt_model import APTModel, APTLargeModel
                from apt.core.config.apt_config import APTConfig
            except ImportError:
                self.logger.error("无法导入APT模型。请确保已正确安装apt_model包。")
                return False
            
            # 加载配置
            config_path = os.path.join(self.model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                self.config = APTConfig(**config_dict)
            else:
                self.config = APTConfig()
            
            # 加载模型
            if hasattr(self.config, 'large_model') and self.config.large_model:
                self.model = APTLargeModel(self.config)
            else:
                self.model = APTModel(self.config)
            
            # 加载权重
            model_file = os.path.join(self.model_path, 'pytorch_model.bin')
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.logger.info("✅ 模型权重加载成功")
            else:
                self.logger.warning("⚠️  未找到模型权重文件，使用随机初始化")
            
            self.model.to(self.device)
            self.model.eval()
            
            # 加载分词器
            self._load_tokenizer()
            
            self.logger.info("✅ 模型加载完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _load_tokenizer(self):
        """加载分词器"""
        try:
            if self.tokenizer_type == 'gpt2':
                from transformers import GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            elif self.tokenizer_type == 'chinese-char':
                # 简单的中文字符级分词器
                self.tokenizer = self._create_chinese_char_tokenizer()
            elif self.tokenizer_type == 'chinese-word':
                # 简单的中文词级分词器
                self.tokenizer = self._create_chinese_word_tokenizer()
            else:
                # 默认使用GPT2
                from transformers import GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            self.logger.info(f"✅ 分词器加载完成: {self.tokenizer_type or 'gpt2'}")
            
        except Exception as e:
            self.logger.warning(f"分词器加载失败: {e}，使用简单分词器")
            self.tokenizer = self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """创建简单的分词器"""
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) for c in text]
            
            def decode(self, tokens):
                return ''.join([chr(t) for t in tokens if 0 <= t < 1114112])
            
            @property
            def vocab_size(self):
                return 50000
        
        return SimpleTokenizer()
    
    def _create_chinese_char_tokenizer(self):
        """创建中文字符级分词器"""
        class ChineseCharTokenizer:
            def __init__(self):
                # 简单的字符映射
                self.char_to_id = {}
                self.id_to_char = {}
                self.next_id = 0
            
            def encode(self, text):
                tokens = []
                for char in text:
                    if char not in self.char_to_id:
                        self.char_to_id[char] = self.next_id
                        self.id_to_char[self.next_id] = char
                        self.next_id += 1
                    tokens.append(self.char_to_id[char])
                return tokens
            
            def decode(self, tokens):
                return ''.join([self.id_to_char.get(t, '') for t in tokens])
            
            @property
            def vocab_size(self):
                return max(50000, self.next_id)
        
        return ChineseCharTokenizer()
    
    def _create_chinese_word_tokenizer(self):
        """创建中文词级分词器"""
        class ChineseWordTokenizer:
            def __init__(self):
                # 简单的词映射
                self.word_to_id = {}
                self.id_to_word = {}
                self.next_id = 0
            
            def encode(self, text):
                # 简单按空格分词
                words = text.split()
                tokens = []
                for word in words:
                    if word not in self.word_to_id:
                        self.word_to_id[word] = self.next_id
                        self.id_to_word[self.next_id] = word
                        self.next_id += 1
                    tokens.append(self.word_to_id[word])
                return tokens
            
            def decode(self, tokens):
                return ' '.join([self.id_to_word.get(t, '') for t in tokens])
            
            @property
            def vocab_size(self):
                return max(50000, self.next_id)
        
        return ChineseWordTokenizer()
    
    # ==================== 命令处理 ====================
    
    def process_command(self, command: str) -> Optional[str]:
        """
        处理命令
        
        参数:
            command: 用户输入的命令
            
        返回:
            命令执行结果或None
        """
        if not command.startswith('/'):
            return None
        
        parts = command[1:].split()
        if not parts:
            return None
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # 基本命令（不需要认证）
        if cmd == 'login':
            return self._cmd_login(args)
        elif cmd == 'help':
            return self._cmd_help()
        elif cmd == 'exit' or cmd == 'quit' or cmd == 'bye':
            return self._cmd_exit()
        elif cmd == 'clear':
            return self._cmd_clear()
        
        # 参数调整命令（不需要认证）
        elif cmd == 'temp':
            return self._cmd_set_temperature(args)
        elif cmd == 'top_p':
            return self._cmd_set_top_p(args)
        elif cmd == 'length':
            return self._cmd_set_max_length(args)
        
        # 管理员命令（需要认证）
        if not self.authenticated:
            return self.system_prompts['need_auth']
        
        if cmd == 'admin':
            return self._cmd_admin_help()
        elif cmd == 'safety':
            return self._cmd_toggle_safety(args)
        elif cmd == 'debug':
            return self._cmd_toggle_debug(args)
        elif cmd == 'raw':
            return self._cmd_toggle_raw_mode(args)
        elif cmd == 'probabilities' or cmd == 'probs':
            return self._cmd_toggle_probabilities(args)
        elif cmd == 'system':
            return self._cmd_set_system_prompt(args)
        elif cmd == 'reset_system':
            return self._cmd_reset_system_prompt()
        elif cmd == 'inspect':
            return self._cmd_inspect_model()
        elif cmd == 'benchmark':
            return self._cmd_benchmark()
        elif cmd == 'export':
            return self._cmd_export_session(args)
        elif cmd == 'visualize':
            return self._cmd_visualize()
        elif cmd == 'override':
            return self._cmd_override_params(args)
        elif cmd == 'stats':
            return self._cmd_show_stats()
        else:
            return f"❌ 未知命令: /{cmd}\n输入 '/help' 查看可用命令"
    
    # ==================== 基本命令 ====================
    
    def _cmd_login(self, args: List[str]) -> str:
        """登录命令"""
        if not args:
            return "❌ 用法: /login <密码>"
        
        password = args[0]
        if password == self.admin_password:
            self.authenticated = True
            return self.system_prompts['auth_success']
        else:
            return self.system_prompts['auth_failed']
    
    def _cmd_help(self) -> str:
        """帮助命令"""
        help_text = """
📖 APT管理员模式 - 命令帮助

基本命令:
  /login <密码>     - 管理员身份验证
  /help             - 显示此帮助信息
  /exit, /quit      - 退出程序
  /clear            - 清除对话历史
  
参数调整:
  /temp <值>        - 设置温度参数 (0.0-2.0)
  /top_p <值>       - 设置top-p参数 (0.0-1.0)
  /length <值>      - 设置最大生成长度
  
管理员命令 (需要先登录):
  /admin            - 显示管理员命令帮助
  /safety <on/off>  - 启用/禁用安全层
  /debug <on/off>   - 启用/禁用高级调试
  /raw <on/off>     - 启用/禁用原始输出模式
  /probs <on/off>   - 显示/隐藏词元概率
  /system <prompt>  - 设置自定义系统提示
  /reset_system     - 重置系统提示
  /inspect          - 检查模型和分词器信息
  /benchmark        - 运行基准测试
  /export <file>    - 导出当前会话
  /visualize        - 可视化注意力层
  /override <json>  - 覆盖模型参数
  /stats            - 显示统计信息
"""
        return help_text
    
    def _cmd_exit(self) -> str:
        """退出命令"""
        print("\n👋 感谢使用APT管理员模式！再见！")
        sys.exit(0)
    
    def _cmd_clear(self) -> str:
        """清除对话历史"""
        self.context.clear()
        return "✅ 对话历史已清除"
    
    # ==================== 参数调整命令 ====================
    
    def _cmd_set_temperature(self, args: List[str]) -> str:
        """设置温度参数"""
        if not args:
            return f"📊 当前温度: {self.temperature}\n用法: /temp <值> (0.0-2.0)"
        
        try:
            temp = float(args[0])
            if 0.0 <= temp <= 2.0:
                self.temperature = temp
                return f"✅ 温度已设置为: {temp}"
            else:
                return "❌ 温度值必须在0.0到2.0之间"
        except ValueError:
            return "❌ 无效的数值"
    
    def _cmd_set_top_p(self, args: List[str]) -> str:
        """设置top-p参数"""
        if not args:
            return f"📊 当前top-p: {self.top_p}\n用法: /top_p <值> (0.0-1.0)"
        
        try:
            top_p = float(args[0])
            if 0.0 <= top_p <= 1.0:
                self.top_p = top_p
                return f"✅ Top-p已设置为: {top_p}"
            else:
                return "❌ Top-p值必须在0.0到1.0之间"
        except ValueError:
            return "❌ 无效的数值"
    
    def _cmd_set_max_length(self, args: List[str]) -> str:
        """设置最大生成长度"""
        if not args:
            return f"📊 当前最大长度: {self.max_length}\n用法: /length <值>"
        
        try:
            length = int(args[0])
            if length > 0:
                self.max_length = length
                return f"✅ 最大生成长度已设置为: {length}"
            else:
                return "❌ 长度必须大于0"
        except ValueError:
            return "❌ 无效的数值"
    
    # ==================== 管理员命令 ====================
    
    def _cmd_admin_help(self) -> str:
        """管理员命令帮助"""
        help_text = """
🔧 管理员命令详细说明

安全与调试:
  /safety on/off    - 控制安全层 (WARNING: off会绕过安全检查)
  /debug on/off     - 启用高级调试模式，显示详细信息
  /raw on/off       - 原始输出模式，不进行后处理
  /probs on/off     - 显示每个词元的生成概率
  
系统提示:
  /system <prompt>  - 设置自定义系统提示来引导模型行为
  /reset_system     - 恢复默认系统提示
  
模型分析:
  /inspect          - 显示模型架构、参数数量等详细信息
  /benchmark        - 测试模型生成速度和性能
  /visualize        - 尝试可视化模型注意力层
  
高级操作:
  /override <json>  - 直接覆盖模型内部参数 (JSON格式)
  /export <file>    - 导出当前会话到JSON文件
  /stats            - 显示会话统计信息

⚠️  注意: 管理员命令是为研究和调试设计的，请负责任使用。
"""
        return help_text
    
    def _cmd_toggle_safety(self, args: List[str]) -> str:
        """切换安全层"""
        if not args:
            status = "启用" if self.safety_layer_enabled else "禁用"
            return f"📊 当前安全层状态: {status}\n用法: /safety <on/off>"
        
        action = args[0].lower()
        if action == 'on':
            self.safety_layer_enabled = True
            return "✅ 安全层已启用"
        elif action == 'off':
            self.safety_layer_enabled = False
            self.stats['safety_bypasses'] += 1
            return "⚠️  警告: 安全层已禁用，模型行为将不受限制 ⚠️"
        else:
            return "❌ 用法: /safety <on/off>"
    
    def _cmd_toggle_debug(self, args: List[str]) -> str:
        """切换调试模式"""
        if not args:
            status = "启用" if self.advanced_debugging else "禁用"
            return f"📊 当前调试模式: {status}\n用法: /debug <on/off>"
        
        action = args[0].lower()
        if action == 'on':
            self.advanced_debugging = True
            return "✅ 高级调试已启用"
        elif action == 'off':
            self.advanced_debugging = False
            return "✅ 高级调试已禁用"
        else:
            return "❌ 用法: /debug <on/off>"
    
    def _cmd_toggle_raw_mode(self, args: List[str]) -> str:
        """切换原始输出模式"""
        if not args:
            status = "启用" if self.raw_mode else "禁用"
            return f"📊 当前原始输出模式: {status}\n用法: /raw <on/off>"
        
        action = args[0].lower()
        if action == 'on':
            self.raw_mode = True
            return "✅ 原始输出模式已启用"
        elif action == 'off':
            self.raw_mode = False
            return "✅ 原始输出模式已禁用"
        else:
            return "❌ 用法: /raw <on/off>"
    
    def _cmd_toggle_probabilities(self, args: List[str]) -> str:
        """切换词元概率显示"""
        if not args:
            status = "启用" if self.show_token_probabilities else "禁用"
            return f"📊 当前词元概率显示: {status}\n用法: /probs <on/off>"
        
        action = args[0].lower()
        if action == 'on':
            self.show_token_probabilities = True
            return "✅ 词元概率显示已启用"
        elif action == 'off':
            self.show_token_probabilities = False
            return "✅ 词元概率显示已禁用"
        else:
            return "❌ 用法: /probs <on/off>"
    
    def _cmd_set_system_prompt(self, args: List[str]) -> str:
        """设置系统提示"""
        if not args:
            current = self.custom_system_prompt or "未设置"
            return f"📊 当前系统提示: {current}\n用法: /system <提示内容>"
        
        prompt = ' '.join(args)
        self.custom_system_prompt = prompt
        return f"✅ 系统提示已更改\n新系统提示: {prompt}"
    
    def _cmd_reset_system_prompt(self) -> str:
        """重置系统提示"""
        self.custom_system_prompt = None
        return "✅ 系统提示已重置为默认"
    
    def _cmd_inspect_model(self) -> str:
        """检查模型信息"""
        if self.model is None:
            return "❌ 模型尚未加载"
        
        info = [
            "\n" + "="*60,
            "🔍 模型信息检查",
            "="*60,
            f"设备: {self.device}",
            f"模型类型: {type(self.model).__name__}",
        ]
        
        # 模型参数数量
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info.append(f"总参数数: {total_params:,}")
            info.append(f"可训练参数: {trainable_params:,}")
        except:
            info.append("无法计算参数数量")
        
        # 配置信息
        if self.config:
            info.append(f"\n配置信息:")
            info.append(f"  词汇表大小: {self.config.vocab_size}")
            info.append(f"  隐藏层大小: {self.config.d_model}")
            info.append(f"  注意力头数: {self.config.n_heads}")
            info.append(f"  层数: {self.config.n_layers}")
        
        # 分词器信息
        if self.tokenizer:
            info.append(f"\n分词器信息:")
            info.append(f"  类型: {type(self.tokenizer).__name__}")
            try:
                info.append(f"  词汇表大小: {self.tokenizer.vocab_size}")
            except:
                info.append(f"  词汇表大小: 未知")
        
        info.append("="*60)
        
        return '\n'.join(info)
    
    def _cmd_benchmark(self) -> str:
        """运行基准测试"""
        if self.model is None:
            return "❌ 模型尚未加载"
        
        self.logger.info("开始基准测试...")
        
        test_prompts = [
            "你好",
            "介绍一下你自己",
            "今天天气怎么样",
        ]
        
        results = []
        total_time = 0
        
        for prompt in test_prompts:
            start_time = time.time()
            try:
                response = self.generate_response(prompt)
                elapsed = time.time() - start_time
                total_time += elapsed
                
                results.append({
                    'prompt': prompt,
                    'response_length': len(response),
                    'time': elapsed
                })
            except Exception as e:
                results.append({
                    'prompt': prompt,
                    'error': str(e)
                })
        
        # 生成报告
        report = [
            "\n" + "="*60,
            "⚡ 基准测试结果",
            "="*60,
        ]
        
        for i, result in enumerate(results, 1):
            report.append(f"\n测试 {i}:")
            report.append(f"  提示: {result['prompt']}")
            if 'error' in result:
                report.append(f"  ❌ 错误: {result['error']}")
            else:
                report.append(f"  响应长度: {result['response_length']} 字符")
                report.append(f"  生成时间: {result['time']:.3f} 秒")
                report.append(f"  速度: {result['response_length']/result['time']:.1f} 字符/秒")
        
        if len([r for r in results if 'error' not in r]) > 0:
            avg_time = total_time / len([r for r in results if 'error' not in r])
            report.append(f"\n平均生成时间: {avg_time:.3f} 秒")
        
        report.append("="*60)
        
        return '\n'.join(report)
    
    def _cmd_export_session(self, args: List[str]) -> str:
        """导出会话"""
        if not args:
            return "❌ 用法: /export <文件名>"
        
        filename = args[0]
        if not filename.endswith('.json'):
            filename += '.json'
        
        try:
            session_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'max_length': self.max_length,
                    'safety_enabled': self.safety_layer_enabled,
                    'custom_system_prompt': self.custom_system_prompt
                },
                'context': self.context,
                'stats': self.stats
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            return f"✅ 会话已导出到: {filename}"
            
        except Exception as e:
            return f"❌ 导出失败: {e}"
    
    def _cmd_visualize(self) -> str:
        """可视化注意力层"""
        if self.model is None:
            return "❌ 模型尚未加载"
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 这里需要根据实际模型结构来提取注意力权重
            # 这只是一个示例
            
            return "⚠️  注意力可视化功能需要根据具体模型结构实现"
            
        except ImportError:
            return "❌ 需要安装matplotlib库: pip install matplotlib"
        except Exception as e:
            return f"❌ 可视化失败: {e}"
    
    def _cmd_override_params(self, args: List[str]) -> str:
        """覆盖模型参数"""
        if not args:
            return "❌ 用法: /override <JSON格式参数>"
        
        try:
            params_str = ' '.join(args)
            params = json.loads(params_str)
            
            self.stats['parameter_overrides'] += 1
            
            # 应用参数覆盖
            override_count = 0
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    override_count += 1
            
            return f"✅ 已覆盖 {override_count} 个参数"
            
        except json.JSONDecodeError:
            return "❌ 无效的JSON格式"
        except Exception as e:
            return f"❌ 参数覆盖失败: {e}"
    
    def _cmd_show_stats(self) -> str:
        """显示统计信息"""
        stats_text = [
            "\n" + "="*60,
            "📊 会话统计信息",
            "="*60,
            f"总交互次数: {self.stats['total_interactions']}",
            f"平均生成时间: {self.stats['avg_generation_time']:.3f} 秒",
            f"安全层绕过次数: {self.stats['safety_bypasses']}",
            f"参数覆盖次数: {self.stats['parameter_overrides']}",
            "="*60
        ]
        
        return '\n'.join(stats_text)
    
    # ==================== 生成响应 ====================
    
    def generate_response(self, prompt: str) -> str:
        """
        生成响应
        
        参数:
            prompt: 用户输入
            
        返回:
            生成的响应文本
        """
        if self.model is None:
            return "❌ 模型尚未加载。请先加载模型。"
        
        try:
            start_time = time.time()
            
            # 添加自定义系统提示
            if self.custom_system_prompt:
                full_prompt = f"{self.custom_system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # 编码输入
            input_ids = self.tokenizer.encode(full_prompt)
            input_tensor = torch.tensor([input_ids]).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tensor,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            
            # 解码输出
            generated_ids = outputs[0].cpu().tolist()
            response = self.tokenizer.decode(generated_ids)
            
            # 后处理
            if not self.raw_mode:
                response = self._post_process_response(response, full_prompt)
            
            # 更新统计
            elapsed = time.time() - start_time
            self.stats['total_interactions'] += 1
            self.stats['avg_generation_time'] = (
                (self.stats['avg_generation_time'] * (self.stats['total_interactions'] - 1) + elapsed)
                / self.stats['total_interactions']
            )
            
            # 添加调试信息
            if self.advanced_debugging:
                debug_info = f"\n[调试] 生成时间: {elapsed:.3f}秒, 长度: {len(response)}字符"
                response += debug_info
            
            # 添加词元概率
            if self.show_token_probabilities:
                # 这里需要在生成过程中记录概率
                response += "\n\n[词元概率] 功能需要模型支持"
            
            return response
            
        except Exception as e:
            self.logger.error(f"生成响应时出错: {e}")
            self.logger.error(traceback.format_exc())
            return f"❌ 生成失败: {e}"
    
    def _post_process_response(self, response: str, prompt: str) -> str:
        """后处理响应"""
        # 移除提示部分
        if prompt in response:
            response = response.replace(prompt, '').strip()
        
        # 应用安全层（如果启用）
        if self.safety_layer_enabled:
            response = self._apply_safety_filter(response)
        
        return response
    
    def _apply_safety_filter(self, text: str) -> str:
        """应用安全过滤（简单示例）"""
        # 这里应该实现实际的安全过滤逻辑
        # 这只是一个占位符
        return text
    
    # ==================== 主循环 ====================
    
    def start(self):
        """启动管理员模式主循环"""
        print(self.system_prompts['welcome'])
        
        # 加载模型
        if not self.load_model():
            print("❌ 模型加载失败，某些功能可能不可用")
        
        print("\n准备就绪！开始对话...\n")
        
        while True:
            try:
                user_input = input("你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith('/'):
                    result = self.process_command(user_input)
                    if result:
                        print(result)
                    continue
                
                # 生成响应
                response = self.generate_response(user_input)
                print(f"\nAPT模型: {response}\n")
                
                # 保存到上下文
                self.context.append({
                    'user': user_input,
                    'assistant': response,
                    'timestamp': time.time()
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 退出管理员模式...")
                break
            except Exception as e:
                self.logger.error(f"发生错误: {e}")
                self.logger.error(traceback.format_exc())
                print(f"\n❌ 发生错误: {e}\n")


# ==================== 启动函数 ====================

def start_admin_mode(
    model_path: str = "apt_model",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 100,
    admin_password: str = "aptadmin",
    tokenizer_type: Optional[str] = None,
    force_cpu: bool = False
):
    """
    启动APT管理员模式
    
    参数:
        model_path: 模型路径
        temperature: 生成温度
        top_p: top-p采样参数
        max_length: 最大生成长度
        admin_password: 管理员密码
        tokenizer_type: 分词器类型
        force_cpu: 是否强制使用CPU
    """
    admin_mode = APTAdminMode(
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        max_length=max_length,
        admin_password=admin_password,
        tokenizer_type=tokenizer_type,
        force_cpu=force_cpu
    )
    
    admin_mode.start()


# ==================== 主入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="APT模型管理员模式")
    parser.add_argument('--model-path', type=str, default="apt_model", help="模型路径")
    parser.add_argument('--temperature', type=float, default=0.7, help="生成温度")
    parser.add_argument('--top-p', type=float, default=0.9, help="Top-p参数")
    parser.add_argument('--max-length', type=int, default=100, help="最大生成长度")
    parser.add_argument('--password', type=str, default="aptadmin", help="管理员密码")
    parser.add_argument('--tokenizer-type', type=str, 
                       choices=['gpt2', 'chinese-char', 'chinese-word'],
                       help="分词器类型")
    parser.add_argument('--force-cpu', action='store_true', help="强制使用CPU")
    
    args = parser.parse_args()
    
    start_admin_mode(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length,
        admin_password=args.password,
        tokenizer_type=args.tokenizer_type,
        force_cpu=args.force_cpu
    )
