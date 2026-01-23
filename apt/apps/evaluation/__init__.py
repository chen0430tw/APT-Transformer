#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Evaluation Module
Provides functionality for evaluating model performance and comparing models

This module now provides a unified evaluation API that consolidates:
- Text quality evaluation
- Code quality evaluation
- Chinese text evaluation
- Model performance evaluation
- Multi-model comparison

Recommended: Use the UnifiedEvaluator class or convenience functions from .unified
Legacy: Original ModelEvaluator and ModelComparison classes are still available
"""

# === Unified API (Recommended) ===
from .unified import (
    UnifiedEvaluator,
    evaluate_text_quality,
    evaluate_code_quality,
    evaluate_chinese_quality,
    quick_evaluate
)

# === Legacy API (Backward Compatibility) ===
from .model_evaluator import ModelEvaluator, evaluate_model
from .comparison import compare_models, ModelComparison

# Define module exports
__all__ = [
    # Unified API (recommended)
    'UnifiedEvaluator',
    'evaluate_text_quality',
    'evaluate_code_quality',
    'evaluate_chinese_quality',
    'quick_evaluate',

    # Legacy API (backward compatibility)
    'ModelEvaluator',
    'evaluate_model',
    'compare_models',
    'ModelComparison',
]