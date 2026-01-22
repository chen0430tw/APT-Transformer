#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APX Capability Detectors

Auto-detect model capabilities from config and structure.
Supports: MoE, RAG, RLHF, Safety, Quantization, TVA/VFT
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_config(src_repo: Path) -> Optional[Dict[str, Any]]:
    """
    Load model config.json if available.

    Args:
        src_repo: Source model directory

    Returns:
        Config dictionary or None if not found
    """
    config_path = src_repo / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def detect_moe(src_repo: Path, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect MoE (Mixture of Experts) capability.

    Indicators:
    - config contains "num_experts" or "num_local_experts"
    - config has "expert" in architecture name
    - Files like moe_*.py or expert_*.py exist

    Args:
        src_repo: Source model directory
        config: Optional pre-loaded config

    Returns:
        True if MoE detected
    """
    if config is None:
        config = load_config(src_repo)

    if config:
        # Check config fields
        if "num_experts" in config or "num_local_experts" in config:
            return True

        # Check architecture name
        arch = config.get("architectures", [])
        if any("expert" in a.lower() or "moe" in a.lower() for a in arch):
            return True

        # Check model_type
        model_type = config.get("model_type", "")
        if "moe" in model_type.lower() or "expert" in model_type.lower():
            return True

    # Check for MoE-related files
    moe_patterns = ["*moe*.py", "*expert*.py", "*router*.py"]
    for pattern in moe_patterns:
        if list(src_repo.glob(pattern)):
            return True

    return False


def detect_rag(src_repo: Path, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect RAG (Retrieval-Augmented Generation) capability.

    Indicators:
    - config contains "retriever" or "knowledge_base"
    - Files like rag_*.py, retriever_*.py, or knowledge_*.py exist
    - Has vector store or embedding files

    Args:
        src_repo: Source model directory
        config: Optional pre-loaded config

    Returns:
        True if RAG detected
    """
    if config is None:
        config = load_config(src_repo)

    if config:
        # Check config fields
        rag_keys = ["retriever", "knowledge_base", "retrieval", "rag"]
        if any(key in config for key in rag_keys):
            return True

        # Check architecture
        arch = config.get("architectures", [])
        if any("rag" in a.lower() or "retriev" in a.lower() for a in arch):
            return True

    # Check for RAG-related files
    rag_patterns = [
        "*rag*.py",
        "*retriev*.py",
        "*knowledge*.py",
        "*vector_store*.py",
        "*embedding*.py",
    ]
    for pattern in rag_patterns:
        if list(src_repo.glob(pattern)):
            return True

    return False


def detect_rl(src_repo: Path, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect RL (Reinforcement Learning) capability (RLHF, PPO, DPO).

    Indicators:
    - config contains "ppo", "dpo", "rlhf", "reward_model"
    - Files like rl_*.py, ppo_*.py, reward_*.py exist
    - Has reward model files

    Args:
        src_repo: Source model directory
        config: Optional pre-loaded config

    Returns:
        True if RL training detected
    """
    if config is None:
        config = load_config(src_repo)

    if config:
        # Check config fields
        rl_keys = ["ppo", "dpo", "rlhf", "reward_model", "value_head"]
        if any(key in config for key in rl_keys):
            return True

        # Check model_type
        model_type = config.get("model_type", "")
        rl_indicators = ["reward", "rlhf", "ppo", "dpo"]
        if any(ind in model_type.lower() for ind in rl_indicators):
            return True

    # Check for RL-related files
    rl_patterns = [
        "*rl*.py",
        "*ppo*.py",
        "*dpo*.py",
        "*rlhf*.py",
        "*reward*.py",
        "*value_head*.py",
    ]
    for pattern in rl_patterns:
        if list(src_repo.glob(pattern)):
            return True

    return False


def detect_safety(src_repo: Path, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect safety/moderation capability.

    Indicators:
    - config contains "safety", "moderation", "toxicity"
    - Files like safety_*.py, moderator_*.py exist
    - Has safety classifier or filter

    Args:
        src_repo: Source model directory
        config: Optional pre-loaded config

    Returns:
        True if safety features detected
    """
    if config is None:
        config = load_config(src_repo)

    if config:
        # Check config fields
        safety_keys = [
            "safety",
            "moderation",
            "toxicity",
            "content_filter",
            "safety_checker",
        ]
        if any(key in config for key in safety_keys):
            return True

        # Check architecture
        arch = config.get("architectures", [])
        if any("safety" in a.lower() or "moderat" in a.lower() for a in arch):
            return True

    # Check for safety-related files
    safety_patterns = [
        "*safety*.py",
        "*moderat*.py",
        "*toxicity*.py",
        "*content_filter*.py",
    ]
    for pattern in safety_patterns:
        if list(src_repo.glob(pattern)):
            return True

    return False


def detect_quant_distill(
    src_repo: Path, config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Detect quantization or distillation.

    Indicators:
    - config contains "quantization", "bits", "distill"
    - Weight files have quantization markers (4bit, 8bit, GPTQ, AWQ)
    - Has distillation-related files

    Args:
        src_repo: Source model directory
        config: Optional pre-loaded config

    Returns:
        True if quantization/distillation detected
    """
    if config is None:
        config = load_config(src_repo)

    if config:
        # Check config fields
        quant_keys = [
            "quantization",
            "quantization_config",
            "bits",
            "load_in_4bit",
            "load_in_8bit",
            "gptq",
            "awq",
            "distillation",
        ]
        if any(key in config for key in quant_keys):
            return True

        # Check model_type or architecture
        model_type = config.get("model_type", "")
        arch = config.get("architectures", [])
        quant_indicators = ["gptq", "awq", "4bit", "8bit", "distill"]
        if any(ind in model_type.lower() for ind in quant_indicators):
            return True
        if any(
            any(ind in a.lower() for ind in quant_indicators) for a in arch
        ):
            return True

    # Check for quantized weight files
    weight_files = list(src_repo.glob("*.safetensors")) + list(
        src_repo.glob("*.bin")
    )
    for wf in weight_files:
        name_lower = wf.name.lower()
        if any(
            marker in name_lower
            for marker in ["4bit", "8bit", "gptq", "awq", "int4", "int8"]
        ):
            return True

    # Check for distillation files
    distill_patterns = ["*distill*.py", "*teacher*.py", "*student*.py"]
    for pattern in distill_patterns:
        if list(src_repo.glob(pattern)):
            return True

    return False


def detect_tva_vft(src_repo: Path, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect TVA (Tri-Vein Attention) or VFT (Vein-Flow Transformer) capability.

    Indicators:
    - config contains "vein", "tva", "vft", "subspace"
    - Files like vein_*.py, tva_*.py, vft_*.py exist
    - Has VeinSubspace or TriVeinAttention modules

    Args:
        src_repo: Source model directory
        config: Optional pre-loaded config

    Returns:
        True if TVA/VFT detected
    """
    if config is None:
        config = load_config(src_repo)

    if config:
        # Check config fields
        vein_keys = [
            "vein",
            "tva",
            "vft",
            "subspace",
            "vein_rank",
            "tri_vein",
            "vein_subspace",
        ]
        if any(key in config for key in vein_keys):
            return True

        # Check architecture
        arch = config.get("architectures", [])
        if any(
            "vein" in a.lower() or "tva" in a.lower() or "vft" in a.lower()
            for a in arch
        ):
            return True

    # Check for TVA/VFT-related files
    vein_patterns = [
        "*vein*.py",
        "*tva*.py",
        "*vft*.py",
        "*subspace*.py",
        "*tri_vein*.py",
    ]
    for pattern in vein_patterns:
        if list(src_repo.glob(pattern)):
            return True

    return False


def detect_capabilities(src_repo: Path) -> List[str]:
    """
    Auto-detect all capabilities of a model.

    Args:
        src_repo: Source model directory

    Returns:
        List of detected capability names
    """
    capabilities = []

    # Load config once
    config = load_config(src_repo)

    # Run all detectors
    detectors = {
        "moe": detect_moe,
        "rag": detect_rag,
        "rl": detect_rl,
        "safety": detect_safety,
        "quantization": detect_quant_distill,
        "tva": detect_tva_vft,
    }

    for cap_name, detector_fn in detectors.items():
        if detector_fn(src_repo, config):
            capabilities.append(cap_name)

    return capabilities
