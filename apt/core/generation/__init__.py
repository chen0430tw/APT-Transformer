#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) - Generation Module

This module includes functionality for text generation and quality evaluation
of the generated text from APT models.
"""

from .generator import generate_natural_text
from .evaluator import evaluate_text_quality

__all__ = [
    'generate_natural_text',
    'evaluate_text_quality',
]