#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chat functionality for the APT Model (自生成变换器)
Provides interactive command-line chat with trained models.
"""

import os
import time
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union

import torch

from ..utils.logging_utils import setup_logging
from apt_model.generation.generator import generate_natural_text, safe_decode
from ..generation.evaluator import evaluate_text_quality


def chat_with_model(
    model_path: str = "apt_model",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 50,
    logger: Optional[logging.Logger] = None,
    keep_history: bool = True,
    max_history_length: int = 6,
    show_metrics: bool = True,
    custom_prompts: Optional[Dict[str, str]] = None,
    tokenizer_type: Optional[str] = None,
    force_cpu: bool = False
) -> None:
    """
    Start an interactive command-line chat session with a trained APT model.

    Args:
        model_path (str): Path to the trained model directory.
        temperature (float): Temperature parameter for text generation. Controls randomness.
        top_p (float): Top-p parameter for text generation. Controls diversity.
        max_length (int): Maximum length of generated text.
        logger (logging.Logger, optional): Logger for logging chat information.
        keep_history (bool): Whether to keep chat history for context.
        max_history_length (int): Maximum number of recent exchanges to keep as context.
        show_metrics (bool): Whether to show quality metrics for generated responses.
        custom_prompts (Dict[str, str], optional): Custom system prompts for the model.
        tokenizer_type (str, optional): 指定分词器类型 ('gpt2', 'chinese-char', 'chinese-word')
        force_cpu (bool): 强制使用CPU进行推理，避免CUDA错误
    """
    # Log session start
    if logger:
        logger.info(f"Starting chat session with model: {model_path}")
        logger.info(
            f"Parameters: temperature={temperature}, top_p={top_p}, max_length={max_length}")

    print(f"正在加载模型: {model_path}")

    # 检查模型路径可能的位置
    possible_model_paths = [
        model_path,
        os.path.join(model_path, "model"),
        os.path.join(model_path, "model.pt"),
        f"{model_path}_best_quality"
    ]

    model_found = False
    for path in possible_model_paths:
        if os.path.exists(path):
            if os.path.isdir(path) and (
                os.path.exists(
                    os.path.join(
                        path,
                        "model.pt")) or os.path.exists(
                    os.path.join(
                    path,
                    "pytorch_model.bin"))):
                model_path = path
                model_found = True
                break
            elif os.path.isfile(path) and path.endswith((".pt", ".bin")):
                model_path = path
                model_found = True
                break

    if not model_found:
        print(f"\n无法找到有效的模型文件。已尝试以下路径:")
        for path in possible_model_paths:
            print(f"- {path}")

        print("\n请尝试以下解决方案:")
        print("1. 使用 --model-path 参数指定确切的模型文件路径")
        print("2. 重新训练模型: python -m apt_model train-custom --save-path apt_model_new")
        return

    try:
        # Import within function to avoid circular imports
        from ..training.checkpoint import load_model

        # 设置设备
        device = "cpu" if force_cpu else (
            "cuda" if torch.cuda.is_available() else "cpu")
        if force_cpu:
            print("已启用CPU模式，避免可能的CUDA错误")

        # 尝试加载模型，如果指定了分词器类型则使用指定的分词器
        if tokenizer_type:
            try:
                from apt.core.modeling.chinese_tokenizer_integration import get_tokenizer
                # 先加载模型
                model, _, config = load_model(
                    model_path, load_tokenizer=False, device=device)

                # 指定分词器类型
                tokenizer = get_tokenizer(tokenizer_type=tokenizer_type)
                print(f"使用指定的{tokenizer_type}分词器")
            except Exception as e:
                if logger:
                    logger.error(f"指定分词器加载失败: {e}")
                    logger.debug(traceback.format_exc())
                print(f"指定分词器加载失败，尝试使用默认分词器")
                # 回退到标准加载方式
                model, tokenizer, config = load_model(
                    model_path, device=device)
        else:
            # 尝试检测保存的模型使用的是哪种类型的分词器
            try:
                from apt.core.modeling.chinese_tokenizer_integration import load_tokenizer
                # 首先检查是否有保存的分词器配置
                tokenizer_dir = os.path.join(
                    os.path.dirname(model_path), "tokenizer")
                tokenizer_config_path = os.path.join(
                    tokenizer_dir, "tokenizer_config.json")

                if os.path.exists(tokenizer_config_path):
                    import json
                    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    if config.get("type") == "chinese":
                        # 使用中文分词器
                        tokenizer = load_tokenizer(tokenizer_dir)
                        model, _, model_config = load_model(
                            model_path, load_tokenizer=False, device=device)
                        print(f"检测到中文分词器，类型: {config.get('mode', 'char')}")
                    else:
                        # 标准加载
                        model, tokenizer, config = load_model(
                            model_path, device=device)
                else:
                    # 标准加载
                    model, tokenizer, config = load_model(
                        model_path, device=device)
            except Exception as e:
                if logger:
                    logger.error(f"分词器检测失败: {e}")
                    logger.debug(traceback.format_exc())
                print("分词器自动检测失败，尝试使用默认加载方式")
                # 标准加载
                model, tokenizer, config = load_model(
                    model_path, device=device)

        model.eval()
        print(f"模型加载成功! 使用设备: {next(model.parameters()).device}")
    except Exception as e:
        # 详细错误信息只记录到日志
        if logger:
            logger.error(f"加载模型失败: {e}")
            logger.error(traceback.format_exc())

        # 用户界面只显示友好的提示
        print("\n模型加载失败。可能的原因:")
        print("- 模型文件结构不完整或损坏")
        print("- 模型版本不兼容")
        print("- 内存不足")

        if "CUDA" in str(e) or "cuda" in str(e):
            print("\n检测到CUDA错误! 建议尝试:")
            print("- 使用 --force-cpu 参数强制使用CPU模式")
            print("  python -m apt_model chat --model-path apt_model --force-cpu")

        print("\n其他解决方案:")
        print("1. 确认模型路径是否正确")
        print("2. 重新训练模型: python -m apt_model train-custom --save-path apt_model_new")
        return

    # System prompts configuration
    system_prompts = {
        "welcome": f"\n{'='*60}\n欢迎与APT模型对话! (输入'exit'或'quit'退出)\n模型参数: 温度={temperature}, top_p={top_p}, 最大生成长度={max_length}\n{'='*60}",
        "loading": "正在思考...",
        "error": "生成回复时出错",
        "farewell": "再见!",
        "tip_quality_low": "\n安柏: 训练...还不够...",
        "commands_help": """
可用命令:
  /help               - 显示帮助信息
  /temp <value>       - 设置温度参数 (0.1-1.5)
  /top_p <value>      - 设置top_p参数 (0.1-1.0)
  /length <value>     - 设置最大生成长度
  /clear              - 清除对话历史
  /save <filename>    - 保存对话历史到文件
  /metrics <on/off>   - 开启/关闭质量评估显示
  /exit or /quit      - 退出对话
        """}

    # Update with custom prompts if provided
    if custom_prompts:
        system_prompts.update(custom_prompts)

    # Print welcome message
    print(system_prompts["welcome"])

    # Initialize chat context
    context = []  # Store chat history

    # Main chat loop
    while True:
        # Get user input
        user_input = input("\n你: ")

        # Check for special commands
        if user_input.lower() in ['/bye', '/退出', '/exit', '/quit']:
            print(system_prompts["farewell"])
            break

        # Process commands
        if user_input.startswith('/'):
            process_command(user_input,
                            context,
                            {"temperature": temperature,
                             "top_p": top_p,
                             "max_length": max_length,
                             "show_metrics": show_metrics},
                            system_prompts)
            continue

        # Add to chat history
        context.append(f"User: {user_input}")

        # Prepare model input
        if keep_history and len(context) > 1:
            # Only use the most recent exchanges as context
            recent_context = context[-min(max_history_length, len(context)):]
            prompt = "\n".join(recent_context)
        else:
            prompt = user_input

        # Generate reply
        try:
            with torch.no_grad():
                print(system_prompts["loading"])
                start_time = time.time()

                # Generate text
                response, output_ids, curr_temperature, curr_top_p = generate_natural_text(
                    model, tokenizer, prompt, max_steps=max_length, temperature=temperature, top_p=top_p)

                end_time = time.time()

                # Extract model reply portion
                cleaned_response = clean_response(response, prompt)

                # Evaluate reply quality
                quality_score, quality_feedback = evaluate_text_quality(
                    cleaned_response)

                # Display reply
                print(f"\nAPT模型: {cleaned_response}")

                # Show metrics if enabled
                if show_metrics:
                    print(
                        f"\n[生成时间: {end_time - start_time:.2f}秒, 质量评分: {quality_score}/100 - {quality_feedback}]")

                # Add to chat history
                context.append(f"APT: {cleaned_response}")

                # If quality is poor, display a suggestion
                if quality_score < 40:
                    print(system_prompts["tip_quality_low"])

        except Exception as e:
            # 简化错误信息并提供解决建议
            if "CUDA" in str(e) or "cuda" in str(e):
                print("生成过程中遇到CUDA错误，建议尝试使用CPU模式:")
                print("python -m apt_model chat --force-cpu")
            else:
                newline = '\n'
                error_msg = f"{system_prompts['error']}: {str(e).split(newline)[0]}"
                print(error_msg)

            # 详细记录到日志
            if logger:
                logger.error(f"生成回复时出错: {e}")
                logger.error(traceback.format_exc())


def process_command(
    command: str,
    context: List[str],
    settings: Dict[str, Any],
    system_prompts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Process a special command in the chat interface.

    Args:
        command (str): The command to process.
        context (List[str]): The current chat context.
        settings (Dict[str, Any]): Current settings.
        system_prompts (Dict[str, str]): System prompt messages.

    Returns:
        Dict[str, Any]: Updated settings.
    """
    # Split the command and arguments
    parts = command.split()
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    # Process based on command
    if cmd in ['/help', '/?']:
        print(system_prompts["commands_help"])

    elif cmd == '/temp':
        if args and args[0]:
            try:
                temp = float(args[0])
                if 0.1 <= temp <= 1.5:
                    settings["temperature"] = temp
                    print(f"温度参数已设置为: {temp}")
                else:
                    print("温度参数应在 0.1 到 1.5 之间")
            except ValueError:
                print("无效的温度参数")
        else:
            print(f"当前温度参数: {settings['temperature']}")

    elif cmd == '/top_p':
        if args and args[0]:
            try:
                top_p = float(args[0])
                if 0.1 <= top_p <= 1.0:
                    settings["top_p"] = top_p
                    print(f"Top-p参数已设置为: {top_p}")
                else:
                    print("Top-p参数应在 0.1 到 1.0 之间")
            except ValueError:
                print("无效的Top-p参数")
        else:
            print(f"当前Top-p参数: {settings['top_p']}")

    elif cmd == '/length':
        if args and args[0]:
            try:
                length = int(args[0])
                if 10 <= length <= 500:
                    settings["max_length"] = length
                    print(f"最大生成长度已设置为: {length}")
                else:
                    print("最大生成长度应在 10 到 500 之间")
            except ValueError:
                print("无效的长度参数")
        else:
            print(f"当前最大生成长度: {settings['max_length']}")

    elif cmd == '/clear':
        context.clear()
        print("对话历史已清除")

    elif cmd == '/save':
        if args and args[0]:
            filename = args[0]
            if not filename.endswith('.txt'):
                filename += '.txt'

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    for line in context:
                        f.write(f"{line}\n")
                print(f"对话历史已保存到: {filename}")
            except Exception as e:
                print(f"保存对话历史时出错: {e}")
        else:
            print("请指定保存文件名，例如: /save dialogue")

    elif cmd == '/metrics':
        if args and args[0]:
            if args[0].lower() in ['on', 'true', '1', 'yes', 'y']:
                settings["show_metrics"] = True
                print("质量评估显示已开启")
            elif args[0].lower() in ['off', 'false', '0', 'no', 'n']:
                settings["show_metrics"] = False
                print("质量评估显示已关闭")
            else:
                print("无效的参数，使用 'on' 或 'off'")
        else:
            current = "开启" if settings["show_metrics"] else "关闭"
            print(f"当前质量评估显示: {current}")

    elif cmd in ['/exit', '/quit']:
        pass  # This is handled in the main loop

    else:
        print(f"未知命令: {cmd}")
        print("输入 /help 查看可用命令")

    return settings


def clean_response(response: str, prompt: str) -> str:
    """
    Clean the generated response by removing any duplicate prompt content and fixing formatting.

    Args:
        response (str): The raw generated response.
        prompt (str): The input prompt used for generation.

    Returns:
        str: Cleaned response text.
    """
    # Remove the prompt if it appears at the beginning of the response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    # Extract model response if it contains "User:" markers
    if "User: " in response:
        # 查找最后一个"User:"的位置
        parts = response.split("User: ")
        if len(parts) > 1:
            # 提取最后一个"User:"之后的文本
            last_user_part = parts[-1]
            # 检查这部分中是否包含"APT:"或类似标记
            if "APT:" in last_user_part:
                # 如果包含，只保留"APT:"之后的部分
                response = last_user_part.split("APT:")[-1].strip()
            else:
                # 否则使用整个最后部分
                response = last_user_part.strip()

    # Clean up any further occurrences of prompt segments
    for line in prompt.split('\n'):
        if line.startswith("User: ") and line in response:
            response = response.replace(line, "").strip()

    # 删除特殊标记
    special_tokens = ["<|endoftext|>", "<pad>", "<eos>", "<bos>"]
    for token in special_tokens:
        response = response.replace(token, "")

    return response.strip()

# Optional advanced features that could be implemented:
# 1. Chat history persistence between sessions
# 2. Multiple personality modes
# 3. Integration with web interface
# 4. Voice input/output capabilities using text-to-speech and speech-to-text
# 5. Structured conversation modes (e.g., interview, storytelling)


if __name__ == "__main__":
    # This allows running the chat module directly
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive chat with APT model")
    parser.add_argument(
        '--model-path',
        type=str,
        default="apt_model",
        help="Path to model directory")
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Temperature parameter (0.1-1.5)")
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help="Top-p parameter (0.1-1.0)")
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help="Maximum generation length")
    parser.add_argument(
        '--no-history',
        action='store_true',
        help="Don't use chat history for context")
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help="Don't show quality metrics")
    parser.add_argument('--tokenizer-type', type=str, default=None,
                        choices=['gpt2', 'chinese-char', 'chinese-word'],
                        help="Tokenizer type to use")
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help="强制使用CPU进行推理（避免CUDA错误）")

    args = parser.parse_args()

    chat_with_model(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length,
        keep_history=not args.no_history,
        show_metrics=not args.no_metrics,
        tokenizer_type=args.tokenizer_type,
        force_cpu=args.force_cpu
    )