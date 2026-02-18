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
    # 支持新版 ("model_config") 和旧版 ("config") 两种格式
    config_kwargs = {}
    model_config = ckpt.get("model_config") or ckpt.get("config", {})
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


def _is_legacy_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
    """检测是否为旧版架构的 checkpoint

    旧版特征: 使用分离的 q_proj/k_proj/v_proj 和 linear1/linear2
    新版特征: 使用融合的 qkv_proj 和 ffn.dense_ffn
    """
    has_separate_qkv = any(
        ".q_proj." in k or ".k_proj." in k or ".v_proj." in k
        for k in state_dict.keys()
    )
    has_fused_qkv = any(".qkv_proj." in k for k in state_dict.keys())
    return has_separate_qkv and not has_fused_qkv


def _remap_legacy_apt_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """将旧版 APT checkpoint 的 state_dict 转换为新版格式

    处理以下变更:
    1. q_proj + k_proj + v_proj → qkv_proj (权重拼接)
    2. linear1/linear2 → ffn.dense_ffn.0/ffn.dense_ffn.3 (路径重命名)
    3. sr_conv1/sr_conv2/sr_layernorm → 丢弃 (架构不兼容, auto_u/auto_v/auto_gate 随机初始化)
    """
    import re

    new_sd = {}
    # 收集 q/k/v 权重用于后续合并
    qkv_buffer: Dict[str, Dict[str, torch.Tensor]] = {}
    skipped_keys = []

    for k, v in state_dict.items():
        # --- 1. 丢弃旧版 sr_conv (与新版 auto_u/auto_v/auto_gate 架构不兼容) ---
        if ".sr_conv1." in k or ".sr_conv2." in k or ".sr_layernorm." in k:
            skipped_keys.append(k)
            continue

        # --- 2. 收集分离的 q/k/v 投影用于合并 ---
        m = re.match(r"(.+\.(?:self_attn|multihead_attn))\.(q_proj|k_proj|v_proj)\.(weight|bias)", k)
        if m:
            attn_prefix = m.group(1)   # e.g. "encoder_layers.0.self_attn"
            proj_type = m.group(2)      # "q_proj", "k_proj", "v_proj"
            param_type = m.group(3)     # "weight" or "bias"
            buf_key = f"{attn_prefix}.{param_type}"
            if buf_key not in qkv_buffer:
                qkv_buffer[buf_key] = {}
            qkv_buffer[buf_key][proj_type] = v
            continue

        # --- 3. FFN 路径重命名: linear1 → ffn.dense_ffn.0, linear2 → ffn.dense_ffn.3 ---
        if ".linear1." in k:
            new_k = k.replace(".linear1.", ".ffn.dense_ffn.0.")
            new_sd[new_k] = v
            continue
        if ".linear2." in k:
            new_k = k.replace(".linear2.", ".ffn.dense_ffn.3.")
            new_sd[new_k] = v
            continue

        # --- 4. 其他 key 保持不变 ---
        new_sd[k] = v

    # --- 合并 q/k/v → qkv_proj ---
    for buf_key, projections in qkv_buffer.items():
        # buf_key 格式: "encoder_layers.0.self_attn.weight"
        parts = buf_key.rsplit(".", 1)
        attn_prefix = parts[0]  # "encoder_layers.0.self_attn"
        param_type = parts[1]    # "weight" or "bias"

        if all(p in projections for p in ("q_proj", "k_proj", "v_proj")):
            # 拼接: [q, k, v] → qkv (dim 0)
            merged = torch.cat([
                projections["q_proj"],
                projections["k_proj"],
                projections["v_proj"],
            ], dim=0)
            new_k = f"{attn_prefix}.qkv_proj.{param_type}"
            new_sd[new_k] = merged
        else:
            # 不完整, 保留原始 key
            for proj_type, tensor in projections.items():
                new_sd[f"{attn_prefix}.{proj_type}.{param_type}"] = tensor

    if skipped_keys:
        print(f"        旧版兼容: 跳过 {len(skipped_keys)} 个 sr_conv 键 (新版使用 auto_u/auto_v/auto_gate)")

    return new_sd


def remap_state_dict(state_dict: Dict[str, torch.Tensor], model_type: str) -> Dict[str, torch.Tensor]:
    """重映射 state_dict 键名

    处理两层映射:
    1. 旧版架构兼容 (分离 q/k/v → 融合 qkv, linear1/2 → ffn, sr_conv → 丢弃)
    2. 添加 "model." 前缀 (原始 APT → HF wrapper)
    """
    # 先处理旧版架构映射
    if _is_legacy_checkpoint(state_dict):
        print("        检测到旧版 APT checkpoint, 执行架构键名映射...")
        state_dict = _remap_legacy_apt_state_dict(state_dict)

    # 再添加 model. 前缀
    new_sd = {}
    for k, v in state_dict.items():
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
        "transformers_version": __import__("transformers").__version__,
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
    overrides = dict(config_overrides or {})
    # 旧版 checkpoint 自动设置兼容参数
    if _is_legacy_checkpoint(state_dict):
        overrides.setdefault("use_rmsnorm", False)   # 旧版用 LayerNorm
        overrides.setdefault("use_swiglu", False)     # 旧版用标准 FFN
        print("        旧版 checkpoint: 自动设置 use_rmsnorm=False, use_swiglu=False")
    config = build_config(model_type, ckpt, overrides)
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
