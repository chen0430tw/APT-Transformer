#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Capability to Plugin Mapping

Defines the relationship between model capabilities and recommended plugins.
"""

from typing import List, Tuple, Dict


# Capability → Plugin Names Mapping
CAPABILITY_PLUGIN_MAP = {
    # MoE (Mixture of Experts) models
    "moe": [
        "route_optimizer",      # Route optimization plugin
    ],

    # RAG (Retrieval-Augmented Generation) models
    "rag": [
        # RAG-specific plugins to be added
    ],

    # RL (Reinforcement Learning) trained models
    "rl": [
        "grpo",                 # Group Relative Policy Optimization
    ],

    # Safety/Moderation models
    "safety": [
        # Safety auditing plugins to be added
    ],

    # Quantized/Distilled models
    "quantization": [
        "model_distillation",   # Distillation plugin
        "model_pruning",        # Pruning plugin
    ],

    # TVA/VFT (Tri-Vein Attention / Vein-Flow Transformer) models
    "tva": [
        # TVA-specific optimization plugins to be added
    ],
}


# Reverse Mapping: Plugin → Capability Requirements
PLUGIN_CAPABILITY_REQUIREMENTS = {
    "route_optimizer": {
        "required": ["moe"],    # Requires MoE capability
        "optional": [],
    },
    "grpo": {
        "required": ["rl"],     # Requires RL capability
        "optional": [],
    },
    "model_distillation": {
        "required": [],
        "optional": ["quantization"],  # Works better with quantization
    },
    "model_pruning": {
        "required": [],
        "optional": ["quantization"],
    },
    "eqi_reporter": {
        "required": [],
        "optional": [],
    },
    "self_consistency": {
        "required": [],
        "optional": [],
    },
    "beam_search_reasoning": {
        "required": [],
        "optional": [],
    },
    "program_aided_reasoning": {
        "required": [],
        "optional": [],
    },
}


def get_recommended_plugins(capabilities: List[str]) -> List[str]:
    """
    Get recommended plugins based on model capabilities.

    Args:
        capabilities: List of model capabilities

    Returns:
        List of recommended plugin names (deduplicated)

    Example:
        >>> get_recommended_plugins(["moe", "tva"])
        ['route_optimizer']
    """
    plugins = []
    for cap in capabilities:
        if cap in CAPABILITY_PLUGIN_MAP:
            plugins.extend(CAPABILITY_PLUGIN_MAP[cap])

    # Remove duplicates while preserving order
    seen = set()
    unique_plugins = []
    for p in plugins:
        if p not in seen:
            seen.add(p)
            unique_plugins.append(p)

    return unique_plugins


def check_plugin_requirements(
    plugin_name: str,
    capabilities: List[str]
) -> Tuple[bool, str]:
    """
    Check if model capabilities satisfy plugin requirements.

    Args:
        plugin_name: Plugin name
        capabilities: Model capabilities

    Returns:
        (is_satisfied, reason)

    Example:
        >>> check_plugin_requirements("route_optimizer", ["moe"])
        (True, "Requirements satisfied")
        >>> check_plugin_requirements("route_optimizer", ["rag"])
        (False, "Missing required capability: moe")
    """
    if plugin_name not in PLUGIN_CAPABILITY_REQUIREMENTS:
        return True, "No specific requirements"

    reqs = PLUGIN_CAPABILITY_REQUIREMENTS[plugin_name]

    # Check required capabilities
    for required_cap in reqs["required"]:
        if required_cap not in capabilities:
            return False, f"Missing required capability: {required_cap}"

    return True, "Requirements satisfied"


def get_plugin_score(
    plugin_name: str,
    capabilities: List[str]
) -> float:
    """
    Calculate plugin relevance score based on capabilities.

    Args:
        plugin_name: Plugin name
        capabilities: Model capabilities

    Returns:
        Score from 0.0 to 1.0 (higher = more relevant)

    Score calculation:
    - 1.0: All required + all optional capabilities present
    - 0.5-0.99: Required present, some optional present
    - 0.0: Missing required capabilities
    """
    if plugin_name not in PLUGIN_CAPABILITY_REQUIREMENTS:
        return 0.5  # Neutral score for plugins without requirements

    reqs = PLUGIN_CAPABILITY_REQUIREMENTS[plugin_name]
    required = reqs["required"]
    optional = reqs["optional"]

    # Check required capabilities
    if required:
        missing_required = [r for r in required if r not in capabilities]
        if missing_required:
            return 0.0  # Cannot use this plugin

    # Calculate score based on optional capabilities
    if not optional:
        return 1.0  # All requirements met, no optional

    matched_optional = sum(1 for opt in optional if opt in capabilities)
    optional_score = matched_optional / len(optional)

    # Combined score: 0.5 base + 0.5 * optional_score
    return 0.5 + 0.5 * optional_score


def register_capability_mapping(capability: str, plugins: List[str]):
    """
    Register new capability to plugin mapping.

    Args:
        capability: Capability name
        plugins: List of plugin names

    Example:
        >>> register_capability_mapping("custom_feature", ["custom_plugin"])
    """
    if capability in CAPABILITY_PLUGIN_MAP:
        # Extend existing mapping
        existing = CAPABILITY_PLUGIN_MAP[capability]
        for p in plugins:
            if p not in existing:
                existing.append(p)
    else:
        CAPABILITY_PLUGIN_MAP[capability] = list(plugins)


def register_plugin_requirements(
    plugin_name: str,
    required: List[str] = None,
    optional: List[str] = None
):
    """
    Register plugin capability requirements.

    Args:
        plugin_name: Plugin name
        required: Required capabilities
        optional: Optional capabilities

    Example:
        >>> register_plugin_requirements(
        ...     "my_plugin",
        ...     required=["moe"],
        ...     optional=["tva"]
        ... )
    """
    PLUGIN_CAPABILITY_REQUIREMENTS[plugin_name] = {
        "required": required or [],
        "optional": optional or [],
    }
