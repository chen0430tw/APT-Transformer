#!/usr/bin/env python
"""convert_checkpoint.py — 将 APT .pt 检查点转换为 HuggingFace 标准格式

用法:
    python -m apt.model.hf_compat.convert_checkpoint \\
        --checkpoint path/to/checkpoint_step_1000.pt \\
        --model-type gpt4o \\
        --output-dir path/to/hf_model/ \\
        --tokenizer path/to/tokenizer.json

输出目录包含:
    config.json             — PretrainedConfig (vLLM/HF 自动识别)
    model.safetensors       — 模型权重 (安全格式)
    tokenizer.json          — 分词器 (如果提供)
    tokenizer_config.json   — 分词器配置 + chat_template
    generation_config.json  — 默认采样参数
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

import torch


# 模型类型 -> (Config 类, Model 类) 的映射
MODEL_REGISTRY = {
    "gpt4o": ("apt.model.hf_compat.configs.GPT4oConfig",
              "apt.model.hf_compat.modeling_gpt4o.GPT4oForCausalLM"),
    "gpto3": ("apt.model.hf_compat.configs.GPTo3Config",
              "apt.model.hf_compat.modeling_gpto3.GPTo3ForCausalLM"),
    "gpt5":  ("apt.model.hf_compat.configs.GPT5Config",
              "apt.model.hf_compat.modeling_gpt5.GPT5ForCausalLM"),
    "claude4": ("apt.model.hf_compat.configs.Claude4Config",
                "apt.model.hf_compat.modeling_claude4.Claude4ForCausalLM"),
    "apt":   ("apt.model.hf_compat.configs.APTConfig",
              "apt.model.hf_compat.modeling_apt.APTForCausalLM"),
    "apt-seq2seq": ("apt.model.hf_compat.configs.APTConfig",
                    "apt.model.hf_compat.modeling_apt.APTForSeq2SeqLM"),
}


def _import_class(dotted_path: str):
    """从 dotted path 动态导入类"""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """加载 .pt 检查点，自动处理不同格式"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        # 标准格式: {"model_state_dict": ..., "model_config": ..., ...}
        if "model_state_dict" in ckpt:
            return ckpt
        # 某些旧格式直接就是 state_dict
        if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
            return {"model_state_dict": ckpt}

    raise ValueError(
        f"无法识别的检查点格式。期望 dict 包含 'model_state_dict' 键，"
        f"实际类型: {type(ckpt)}, 键: {list(ckpt.keys())[:10] if isinstance(ckpt, dict) else 'N/A'}"
    )


def build_config(model_type: str, ckpt: Dict[str, Any], overrides: Dict[str, Any]) -> Any:
    """从检查点信息构建 HF Config"""
    config_cls_path = MODEL_REGISTRY[model_type][0]
    config_cls = _import_class(config_cls_path)

    # 尝试从检查点中提取模型配置
    config_kwargs = {}
    model_config = ckpt.get("model_config", {})
    if isinstance(model_config, dict):
        config_kwargs.update(model_config)
    elif hasattr(model_config, "__dict__"):
        config_kwargs.update(model_config.__dict__)

    # 覆盖 vocab_size
    if "tokenizer_vocab_size" in ckpt:
        config_kwargs.setdefault("vocab_size", ckpt["tokenizer_vocab_size"])

    # 用户覆盖优先
    config_kwargs.update(overrides)

    return config_cls(**config_kwargs)


def remap_state_dict(state_dict: Dict[str, torch.Tensor], model_type: str) -> Dict[str, torch.Tensor]:
    """重映射 state_dict 键名

    原始模型直接保存的 state_dict 键名可能不带 "model." 前缀，
    而 HF wrapper 的键名是 "model.xxx"。
    """
    new_sd = {}
    for k, v in state_dict.items():
        # 已经有 model. 前缀的保持不变
        if k.startswith("model."):
            new_sd[k] = v
        else:
            new_sd[f"model.{k}"] = v
    return new_sd


def save_generation_config(output_dir: str, model_type: str, eos_token_id: int = 3):
    """保存 generation_config.json"""
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": 2,
        "eos_token_id": eos_token_id,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 2048,
        "repetition_penalty": 1.1,
        "transformers_version": "4.40.0",
    }
    path = os.path.join(output_dir, "generation_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)
    print(f"  generation_config.json -> {path}")


def convert(
    checkpoint_path: str,
    model_type: str,
    output_dir: str,
    tokenizer_path: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    safe_serialization: bool = True,
):
    """执行完整转换"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"转换 {checkpoint_path} -> {output_dir}")
    print(f"  模型类型: {model_type}")

    # 1. 加载检查点
    print("  [1/5] 加载检查点...")
    ckpt = load_checkpoint(checkpoint_path)
    state_dict = ckpt["model_state_dict"]
    print(f"        state_dict 键数: {len(state_dict)}")

    # 2. 构建配置
    print("  [2/5] 构建 HF Config...")
    config = build_config(model_type, ckpt, config_overrides or {})
    config.save_pretrained(output_dir)
    print(f"        config.json -> {output_dir}/config.json")

    # 3. 创建模型并加载权重
    print("  [3/5] 创建 HF 模型并加载权重...")
    model_cls_path = MODEL_REGISTRY[model_type][1]
    model_cls = _import_class(model_cls_path)
    model = model_cls(config)

    # 尝试直接加载，如果失败则重映射键名
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # 可能需要加 "model." 前缀
        # 注意: 当 checkpoint 使用无前缀键名 (如 token_embedding.weight)
        # 而 HF wrapper 期望前缀键名 (如 model.token_embedding.weight) 时，
        # missing 和 unexpected 都会非空，所以这里只检查 missing
        remapped = remap_state_dict(state_dict, model_type)
        missing2, unexpected2 = model.load_state_dict(remapped, strict=False)
        if len(missing2) < len(missing):
            missing, unexpected = missing2, unexpected2
            state_dict = remapped

    if missing:
        print(f"        警告: {len(missing)} 个缺失键 (使用随机初始化)")
        for k in missing[:5]:
            print(f"          - {k}")
        if len(missing) > 5:
            print(f"          ... 和 {len(missing) - 5} 个更多")

    if unexpected:
        print(f"        警告: {len(unexpected)} 个意外键 (已忽略)")
        for k in unexpected[:5]:
            print(f"          - {k}")
        if len(unexpected) > 5:
            print(f"          ... 和 {len(unexpected) - 5} 个更多")

    # 4. 保存为 safetensors (或 .bin)
    print("  [4/5] 保存模型权重...")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    fmt = "safetensors" if safe_serialization else "pytorch_bin"
    print(f"        权重格式: {fmt}")

    # 5. 保存 tokenizer (如果提供)
    print("  [5/5] 保存 tokenizer 和 generation_config...")
    if tokenizer_path:
        from apt.model.hf_compat.tokenization_apt import APTTokenizer
        tokenizer = APTTokenizer(tokenizer_file=tokenizer_path)
        tokenizer.save_pretrained(output_dir)
        print(f"        tokenizer -> {output_dir}/tokenizer.json")

    # generation_config.json
    eos_id = getattr(config, "eos_token_id", 3)
    save_generation_config(output_dir, model_type, eos_token_id=eos_id)

    # 汇总
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n转换完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  参数总量: {total_params:,}")
    print(f"  模型类型: {config.model_type}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="将 APT .pt 检查点转换为 HuggingFace 标准格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # GPT4o 模型
  python -m apt.model.hf_compat.convert_checkpoint \\
      --checkpoint output/checkpoint_step_1000.pt \\
      --model-type gpt4o \\
      --output-dir hf_models/gpt4o/ \\
      --tokenizer output/tokenizer.json

  # Claude4 模型 (自定义参数)
  python -m apt.model.hf_compat.convert_checkpoint \\
      --checkpoint ckpt.pt \\
      --model-type claude4 \\
      --output-dir hf_models/claude4/ \\
      --config-override vocab_size=65536 d_model=512

支持的模型类型: gpt4o, gpto3, gpt5, claude4, apt, apt-seq2seq
        """,
    )
    parser.add_argument("--checkpoint", required=True, help=".pt 检查点文件路径")
    parser.add_argument("--model-type", required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="模型类型")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json 路径 (可选)")
    parser.add_argument("--no-safetensors", action="store_true",
                        help="使用 pytorch_model.bin 格式 (默认 safetensors)")
    parser.add_argument("--config-override", nargs="*", default=[],
                        help="覆盖 config 参数, 格式: key=value")

    args = parser.parse_args()

    # 解析 config 覆盖
    overrides = {}
    for item in args.config_override:
        if "=" not in item:
            parser.error(f"无效的 config-override 格式: {item} (期望 key=value)")
        k, v = item.split("=", 1)
        # 尝试解析为 int/float/bool
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        else:
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
        overrides[k] = v

    convert(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        config_overrides=overrides,
        safe_serialization=not args.no_safetensors,
    )


if __name__ == "__main__":
    main()
