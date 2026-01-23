#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Unified Evaluation Module

This module provides a comprehensive, unified evaluation framework integrating:
- Text quality evaluation
- Code quality evaluation
- Chinese text evaluation
- Model performance evaluation with built-in test sets
- Multi-model comparison and ranking

Consolidates functionality from:
- evaluation/model_evaluator.py
- generation/evaluator.py
- evaluation/comparison.py
"""

import re
import math
import logging
from typing import Tuple, Dict, List, Optional, Union, Any, Callable
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import existing components for integration
from .model_evaluator import ModelEvaluator as _ModelEvaluator
from .comparison import ModelComparison as _ModelComparison
from ..generation.evaluator import (
    TextQualityEvaluator as _TextQualityEvaluator,
    CodeQualityEvaluator as _CodeQualityEvaluator,
    ChineseTextEvaluator as _ChineseTextEvaluator
)


class UnifiedEvaluator:
    """
    Unified evaluation framework for APT models.

    This class provides a single entry point for all evaluation needs:
    - Quality metrics for text, code, and Chinese text
    - Model performance evaluation on standard benchmarks
    - Multi-model comparison and ranking

    Example:
        >>> evaluator = UnifiedEvaluator(logger=my_logger)
        >>>
        >>> # Evaluate text quality
        >>> score, feedback = evaluator.evaluate_text("Generated text here")
        >>>
        >>> # Evaluate a model
        >>> evaluator.add_model("my_model", generator_function)
        >>> results = evaluator.evaluate_model(eval_sets=["general", "reasoning"])
        >>>
        >>> # Compare models
        >>> evaluator.add_model("model_a", generator_a)
        >>> evaluator.add_model("model_b", generator_b)
        >>> comparison = evaluator.compare_models()
    """

    def __init__(self, logger: Optional[logging.Logger] = None, output_dir: Optional[str] = None):
        """
        Initialize the unified evaluator.

        Args:
            logger: Optional logger instance for logging evaluation progress
            output_dir: Optional directory to save evaluation results and visualizations
        """
        self.logger = logger or logging.getLogger('apt_model.evaluation.unified')
        self.output_dir = output_dir

        # Initialize specialized evaluators
        self.text_evaluator = _TextQualityEvaluator(use_external_metrics=True)
        self.code_evaluator = _CodeQualityEvaluator()
        self.chinese_evaluator = _ChineseTextEvaluator()
        self.model_evaluator = _ModelEvaluator(logger=logger)

        # Model comparison will be initialized on demand
        self._model_comparison = None

        # Track models for evaluation
        self.models = {}

    # ==================== Quality Evaluation ====================

    def evaluate_text(
        self,
        text: str,
        reference: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Evaluate the quality of generated text.

        Args:
            text: The text to evaluate
            reference: Optional reference text to compare against
            context: Optional context or prompt that generated the text

        Returns:
            Tuple of (quality_score, feedback_message)
            - quality_score: Float between 0-100
            - feedback_message: Human-readable assessment

        Example:
            >>> score, feedback = evaluator.evaluate_text(
            ...     text="The quick brown fox jumps over the lazy dog.",
            ...     reference="A sentence about a fox and a dog."
            ... )
            >>> print(f"Score: {score:.1f}/100 - {feedback}")
        """
        return self.text_evaluator.evaluate_text_quality(text, reference, context)

    def evaluate_code(
        self,
        code: str,
        language: Optional[str] = None,
        reference: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Evaluate the quality of generated code.

        Args:
            code: The code to evaluate
            language: Programming language (auto-detected if None)
            reference: Optional reference code to compare against

        Returns:
            Tuple of (quality_score, feedback_message)

        Example:
            >>> score, feedback = evaluator.evaluate_code(
            ...     code="def factorial(n):\\n    return 1 if n == 0 else n * factorial(n-1)",
            ...     language="python"
            ... )
        """
        return self.code_evaluator.evaluate_code_quality(code, language, reference)

    def evaluate_chinese(
        self,
        text: str,
        reference: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Evaluate Chinese text quality.

        Args:
            text: Chinese text to evaluate
            reference: Optional reference text for comparison

        Returns:
            Tuple of (quality_score, feedback_message)

        Example:
            >>> score, feedback = evaluator.evaluate_chinese(
            ...     text="人工智能是计算机科学的一个分支。",
            ...     reference="人工智能是模拟人类智能的技术。"
            ... )
        """
        return self.chinese_evaluator.evaluate_chinese_text(text, reference)

    def auto_evaluate(
        self,
        text: str,
        text_type: str = "auto",
        reference: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Automatically detect text type and evaluate accordingly.

        Args:
            text: Text to evaluate
            text_type: Type of text ("auto", "text", "code", "chinese")
            reference: Optional reference for comparison
            context: Optional context

        Returns:
            Tuple of (quality_score, feedback_message)
        """
        if text_type == "auto":
            # Auto-detect text type
            if re.search(r'def\s+\w+\s*\(|function\s+\w+\s*\(|class\s+\w+', text):
                text_type = "code"
            elif re.search(r'[\u4e00-\u9fff]', text) and len(re.findall(r'[\u4e00-\u9fff]', text)) > len(text) * 0.3:
                text_type = "chinese"
            else:
                text_type = "text"

        # Evaluate based on detected type
        if text_type == "code":
            return self.evaluate_code(text, reference=reference)
        elif text_type == "chinese":
            return self.evaluate_chinese(text, reference=reference)
        else:
            return self.evaluate_text(text, reference=reference, context=context)

    # ==================== Model Evaluation ====================

    def add_model(
        self,
        model_name: str,
        generator_fn: Callable[[str], str]
    ) -> 'UnifiedEvaluator':
        """
        Add a model for evaluation.

        Args:
            model_name: Name to identify the model
            generator_fn: Function that takes a prompt (str) and returns generated text (str)

        Returns:
            self for method chaining

        Example:
            >>> def my_generator(prompt):
            ...     return model.generate(prompt)
            >>> evaluator.add_model("my_model", my_generator)
        """
        self.models[model_name] = generator_fn
        self.model_evaluator.add_model(model_name, generator_fn)

        if self.logger:
            self.logger.info(f"Added model '{model_name}' for evaluation")

        return self

    def evaluate_model(
        self,
        eval_sets: Optional[List[str]] = None,
        num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all added models on standard evaluation sets.

        Args:
            eval_sets: List of evaluation set names to use
                      Available: "general", "reasoning", "coding", "creative", "chinese"
                      None means use all available sets
            num_samples: Number of samples to evaluate from each set
                        None means use all samples

        Returns:
            Dictionary containing evaluation results for all models

        Example:
            >>> evaluator.add_model("model_a", generator_a)
            >>> results = evaluator.evaluate_model(eval_sets=["general", "reasoning"], num_samples=5)
            >>> print(results["model_a"]["overall"]["average_score"])
        """
        if not self.models:
            if self.logger:
                self.logger.warning("No models added for evaluation")
            return {}

        return self.model_evaluator.evaluate(eval_set_names=eval_sets, num_samples=num_samples)

    def add_custom_eval_set(
        self,
        name: str,
        evaluation_set: List[Dict[str, str]]
    ) -> 'UnifiedEvaluator':
        """
        Add a custom evaluation set.

        Args:
            name: Name for the evaluation set
            evaluation_set: List of evaluation items, each containing:
                - "prompt": The input prompt
                - "reference": Expected or reference output (optional)
                - "category": Category label (optional)
                - "difficulty": Difficulty level (optional)

        Returns:
            self for method chaining

        Example:
            >>> custom_set = [
            ...     {"prompt": "What is 2+2?", "reference": "4", "category": "math", "difficulty": "easy"},
            ...     {"prompt": "Explain quantum mechanics", "category": "physics", "difficulty": "hard"}
            ... ]
            >>> evaluator.add_custom_eval_set("my_tests", custom_set)
        """
        self.model_evaluator.add_custom_evaluation_set(name, evaluation_set)
        return self

    def get_best_model(self) -> Optional[str]:
        """
        Get the name of the best performing model from evaluation results.

        Returns:
            Name of the best model, or None if no evaluations have been run
        """
        return self.model_evaluator.get_best_model()

    def print_summary(self) -> None:
        """
        Print a summary of evaluation results to console.

        Example:
            >>> evaluator.evaluate_model()
            >>> evaluator.print_summary()
        """
        self.model_evaluator.print_summary()

    def export_results(self, output_path: str) -> bool:
        """
        Export evaluation results to a JSON file.

        Args:
            output_path: Path to save the results

        Returns:
            True if successful, False otherwise
        """
        return self.model_evaluator.export_results(output_path)

    # ==================== Model Comparison ====================

    def compare_models(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        eval_sets: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        custom_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models and generate comparison reports.

        Args:
            model_paths: Dictionary mapping model names to model checkpoint paths
                        If None, uses models added via add_model()
            eval_sets: List of evaluation set names
            num_samples: Number of samples per evaluation set
            custom_prompts: List of custom prompts to test on all models

        Returns:
            Dictionary containing comparison results, rankings, and statistics

        Example:
            >>> comparison = evaluator.compare_models(
            ...     model_paths={"model_v1": "/path/to/v1", "model_v2": "/path/to/v2"},
            ...     eval_sets=["general", "reasoning"],
            ...     custom_prompts=["Tell me a story", "Explain AI"]
            ... )
            >>> print(comparison["summary"]["rankings"])
        """
        if model_paths:
            # Use ModelComparison for checkpoint-based comparison
            if self._model_comparison is None:
                self._model_comparison = _ModelComparison(
                    logger=self.logger,
                    output_dir=self.output_dir
                )

            # Add models
            for model_name, model_path in model_paths.items():
                self._model_comparison.add_model(model_name, model_path)

            # Run comparison
            return self._model_comparison.compare(
                eval_sets=eval_sets,
                num_samples=num_samples,
                prompts=custom_prompts
            )
        else:
            # Use in-memory models
            if not self.models:
                if self.logger:
                    self.logger.warning("No models added for comparison")
                return {}

            # Evaluate all models
            results = self.evaluate_model(eval_sets=eval_sets, num_samples=num_samples)

            # Generate comparison summary
            comparison = {
                "models": list(self.models.keys()),
                "results": results,
                "rankings": self._generate_rankings(results),
                "summary": {}
            }

            return comparison

    def _generate_rankings(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate model rankings from evaluation results."""
        rankings = []

        for model_name, model_results in results.items():
            if "overall" in model_results:
                overall_score = model_results["overall"].get("average_score", 0.0)
                rankings.append({
                    "model": model_name,
                    "score": overall_score
                })

        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)

        return rankings

    # ==================== Utility Methods ====================

    def get_available_eval_sets(self) -> List[str]:
        """
        Get list of available built-in evaluation sets.

        Returns:
            List of evaluation set names
        """
        return list(self.model_evaluator.evaluation_sets.keys())

    def get_eval_set_info(self, set_name: str) -> Dict[str, Any]:
        """
        Get information about an evaluation set.

        Args:
            set_name: Name of the evaluation set

        Returns:
            Dictionary with set information including sample count and categories
        """
        if set_name not in self.model_evaluator.evaluation_sets:
            return {}

        eval_set = self.model_evaluator.evaluation_sets[set_name]

        # Collect statistics
        categories = defaultdict(int)
        difficulties = defaultdict(int)

        for item in eval_set:
            category = item.get("category", "unknown")
            difficulty = item.get("difficulty", "unknown")
            categories[category] += 1
            difficulties[difficulty] += 1

        return {
            "name": set_name,
            "total_samples": len(eval_set),
            "categories": dict(categories),
            "difficulties": dict(difficulties)
        }

    def list_eval_sets(self) -> None:
        """Print information about all available evaluation sets."""
        print("\n" + "="*60)
        print("Available Evaluation Sets".center(60))
        print("="*60 + "\n")

        for set_name in self.get_available_eval_sets():
            info = self.get_eval_set_info(set_name)
            print(f"📋 {set_name}")
            print(f"   Samples: {info['total_samples']}")
            print(f"   Categories: {', '.join(info['categories'].keys())}")
            print(f"   Difficulties: {', '.join(info['difficulties'].keys())}")
            print()


# ==================== Convenience Functions ====================

def evaluate_text_quality(
    text: str,
    reference: Optional[str] = None,
    context: Optional[str] = None
) -> Tuple[float, str]:
    """
    Quick function to evaluate text quality without creating an evaluator instance.

    Args:
        text: Text to evaluate
        reference: Optional reference text
        context: Optional context

    Returns:
        Tuple of (quality_score, feedback_message)
    """
    evaluator = UnifiedEvaluator()
    return evaluator.evaluate_text(text, reference, context)


def evaluate_code_quality(
    code: str,
    language: Optional[str] = None
) -> Tuple[float, str]:
    """
    Quick function to evaluate code quality without creating an evaluator instance.

    Args:
        code: Code to evaluate
        language: Programming language

    Returns:
        Tuple of (quality_score, feedback_message)
    """
    evaluator = UnifiedEvaluator()
    return evaluator.evaluate_code(code, language)


def evaluate_chinese_quality(
    text: str,
    reference: Optional[str] = None
) -> Tuple[float, str]:
    """
    Quick function to evaluate Chinese text quality without creating an evaluator instance.

    Args:
        text: Chinese text to evaluate
        reference: Optional reference text

    Returns:
        Tuple of (quality_score, feedback_message)
    """
    evaluator = UnifiedEvaluator()
    return evaluator.evaluate_chinese(text, reference)


def quick_evaluate(
    text: str,
    text_type: str = "auto",
    reference: Optional[str] = None,
    context: Optional[str] = None
) -> Tuple[float, str]:
    """
    Quick auto-detecting evaluation function.

    Args:
        text: Text to evaluate
        text_type: Type hint ("auto", "text", "code", "chinese")
        reference: Optional reference
        context: Optional context

    Returns:
        Tuple of (quality_score, feedback_message)
    """
    evaluator = UnifiedEvaluator()
    return evaluator.auto_evaluate(text, text_type, reference, context)


# Export all public APIs
__all__ = [
    # Main class
    'UnifiedEvaluator',
    # Convenience functions
    'evaluate_text_quality',
    'evaluate_code_quality',
    'evaluate_chinese_quality',
    'quick_evaluate',
]
