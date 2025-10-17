#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Evaluation Module
Provides functionality for evaluating model performance and comparing models
"""

# Export the main evaluation classes and functions
from .model_evaluator import ModelEvaluator, evaluate_model
from .comparison import compare_models, ModelComparison

# Define module exports
__all__ = [
    'ModelEvaluator',
    'evaluate_model',
    'compare_models',
    'ModelComparison'
]