#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APX Model Packaging Tools

Provides tools for packaging models into APX format with:
- Model conversion and packaging
- Capability detection (MoE, RAG, RLHF, Safety, TVA/VFT, Quantization)
- Adapter generation (HuggingFace, custom)
- Full and thin packaging modes

APX Format:
    .apx (ZIP archive) containing:
    - apx.yaml: Manifest with entrypoints, artifacts, capabilities
    - model/adapters/: Adapter code for loading model
    - artifacts/: Model files (config, weights, tokenizer)
    - tests/: Optional smoke tests
"""

from apt_model.tools.apx.converter import (
    pack_apx,
    detect_framework,
    APXPackagingError,
)
from apt_model.tools.apx.detectors import (
    detect_capabilities,
    detect_moe,
    detect_rag,
    detect_rl,
    detect_safety,
    detect_quant_distill,
    detect_tva_vft,
)
from apt_model.tools.apx.adapters import (
    AdapterType,
    get_adapter_code,
)

__all__ = [
    # Main packaging function
    'pack_apx',
    'detect_framework',
    'APXPackagingError',

    # Capability detection
    'detect_capabilities',
    'detect_moe',
    'detect_rag',
    'detect_rl',
    'detect_safety',
    'detect_quant_distill',
    'detect_tva_vft',

    # Adapter management
    'AdapterType',
    'get_adapter_code',
]

__version__ = "1.0.0"
