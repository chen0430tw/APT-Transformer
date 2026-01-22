#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Evaluator Module

This module provides a comprehensive evaluation framework for APT models,
with support for various evaluation sets and metrics.
"""

import os
import re
import logging
import traceback
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from collections import defaultdict

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from apt.apt_model.utils.visualization import ModelVisualizer


class ModelEvaluator:
    """
    Model Evaluator
    
    Provides a unified evaluation framework for testing and comparing
    the performance of different models.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize model evaluator
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.metrics = {}
        self.results = {}
        self.visualizer = ModelVisualizer(logger=logger)
        
        # Initialize built-in evaluation sets
        self.evaluation_sets = {
            "general": self._get_general_evaluation_set(),
            "reasoning": self._get_reasoning_evaluation_set(),
            "coding": self._get_coding_evaluation_set(),
            "creative": self._get_creative_evaluation_set(),
            "chinese": self._get_chinese_evaluation_set()
        }
    
    def _get_general_evaluation_set(self) -> List[Dict[str, str]]:
        """Get general evaluation set"""
        return [
            {
                "prompt": "What is artificial intelligence?",
                "reference": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. It encompasses areas like machine learning, natural language processing, computer vision, and robotics.",
                "category": "factual",
                "difficulty": "easy"
            },
            {
                "prompt": "Explain the concept of climate change.",
                "reference": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels, which produces heat-trapping gases.",
                "category": "factual",
                "difficulty": "medium"
            },
            {
                "prompt": "Describe the structure of an atom.",
                "reference": "An atom consists of a central nucleus containing protons and neutrons, surrounded by electrons that orbit in shells. The nucleus is positively charged due to protons, while electrons are negatively charged, creating a balanced atom when their numbers are equal.",
                "category": "factual",
                "difficulty": "medium"
            },
            {
                "prompt": "What is the capital of France?",
                "reference": "Paris is the capital of France.",
                "category": "factual",
                "difficulty": "easy"
            },
            {
                "prompt": "Explain how the internet works.",
                "reference": "The internet is a global network of interconnected computers that communicate using a standardized set of protocols, primarily TCP/IP. Data is broken into packets, routed through various network devices, and reassembled at the destination. Web browsers interpret HTML and other web technologies to display content from web servers.",
                "category": "factual",
                "difficulty": "hard"
            }
        ]
    
    def _get_reasoning_evaluation_set(self) -> List[Dict[str, str]]:
        """Get reasoning evaluation set"""
        return [
            {
                "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "reference": "If 5 machines can make 5 widgets in 5 minutes, then 1 machine can make 1 widget in 5 minutes. Therefore, 100 machines can make 100 widgets in 5 minutes.",
                "category": "logical",
                "difficulty": "medium"
            },
            {
                "prompt": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
                "reference": "Let's say the ball costs x dollars. Then the bat costs x + 1 dollars. We know that together they cost $1.10, so x + (x + 1) = 1.10. This means 2x + 1 = 1.10, so 2x = 0.10, which means x = 0.05. Therefore, the ball costs 5 cents.",
                "category": "logical",
                "difficulty": "medium"
            },
            {
                "prompt": "If Mary is twice as old as Ann was when Mary was as old as Ann is now, and Mary is 30, how old is Ann?",
                "reference": "Let's denote Ann's current age as A and Mary's current age as M. We know that M = 30. We're told that Mary is twice as old as Ann was when Mary was as old as Ann is now. When Mary was as old as Ann is now, Mary's age was A. At that time, Ann's age was some value, let's call it X. We know that Mary (30) is twice as old as X. So 30 = 2X, meaning X = 15. Now, we know that when Mary was A years old, Ann was X = 15 years old. Let's calculate how long ago that was. It was (30 - A) years ago. So Ann's age now is: 15 + (30 - A) = 45 - A. Since Ann's current age is A, we have: A = 45 - A, which means 2A = 45, so A = 22.5. Therefore, Ann is currently 20 years old.",
                "category": "logical",
                "difficulty": "hard"
            },
            {
                "prompt": "If all Zorks are Morks, and some Morks are Porks, can we conclude that some Zorks are definitely Porks?",
                "reference": "No, we cannot conclude that some Zorks are definitely Porks. All we know is that all Zorks are Morks, and some Morks are Porks. It's possible that the Zorks are entirely in the subset of Morks that are not Porks.",
                "category": "logical",
                "difficulty": "medium"
            },
            {
                "prompt": "A train travels from point A to point B at a speed of 60 km/h and returns from B to A at a speed of 40 km/h. What is the average speed for the entire journey?",
                "reference": "To find the average speed, we need to use the formula: average speed = total distance / total time. Let's say the distance between A and B is d km. For the first part of the journey (A to B), time taken = d/60 hours. For the second part (B to A), time taken = d/40 hours. Total distance = 2d km. Total time = d/60 + d/40 = (2d/120) + (3d/120) = 5d/120 = d/24 hours. Average speed = total distance / total time = 2d / (d/24) = 48 km/h.",
                "category": "logical",
                "difficulty": "hard"
            }
        ]
    
    def _get_coding_evaluation_set(self) -> List[Dict[str, str]]:
        """Get coding evaluation set"""
        return [
            {
                "prompt": "Write a Python function to find the factorial of a number.",
                "reference": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
                "category": "python",
                "difficulty": "easy"
            },
            {
                "prompt": "Write a Python function to check if a string is a palindrome.",
                "reference": "def is_palindrome(s):\n    # Remove non-alphanumeric characters and convert to lowercase\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]",
                "category": "python",
                "difficulty": "easy"
            },
            {
                "prompt": "Write a Python function to find the maximum subarray sum using Kadane's algorithm.",
                "reference": "def max_subarray_sum(arr):\n    max_so_far = float('-inf')\n    max_ending_here = 0\n    \n    for num in arr:\n        max_ending_here = max(num, max_ending_here + num)\n        max_so_far = max(max_so_far, max_ending_here)\n    \n    return max_so_far",
                "category": "python",
                "difficulty": "medium"
            },
            {
                "prompt": "Write a JavaScript function to flatten a nested array of arbitrary depth.",
                "reference": "function flattenArray(arr) {\n  return arr.reduce((flat, item) => {\n    return flat.concat(Array.isArray(item) ? flattenArray(item) : item);\n  }, []);\n}",
                "category": "javascript",
                "difficulty": "medium"
            },
            {
                "prompt": "Write a SQL query to find the second highest salary from an employee table.",
                "reference": "SELECT MAX(salary) AS second_highest_salary\nFROM employee\nWHERE salary < (SELECT MAX(salary) FROM employee);",
                "category": "sql",
                "difficulty": "medium"
            }
        ]
    
    def _get_creative_evaluation_set(self) -> List[Dict[str, str]]:
        """Get creative evaluation set"""
        return [
            {
                "prompt": "Write a short story about a robot discovering emotions.",
                "reference": "",  # Creative content has no standard answer
                "category": "story",
                "difficulty": "medium"
            },
            {
                "prompt": "Write a poem about the changing seasons.",
                "reference": "",
                "category": "poem",
                "difficulty": "medium"
            },
            {
                "prompt": "Create a fictional dialogue between a time traveler and a historical figure.",
                "reference": "",
                "category": "dialogue",
                "difficulty": "hard"
            },
            {
                "prompt": "Write a marketing description for a fictional product that lets people temporarily speak with animals.",
                "reference": "",
                "category": "marketing",
                "difficulty": "medium"
            },
            {
                "prompt": "Write a short scene showing a character overcoming their greatest fear.",
                "reference": "",
                "category": "scene",
                "difficulty": "medium"
            }
        ]
    
    def _get_chinese_evaluation_set(self) -> List[Dict[str, str]]:
        """Get Chinese evaluation set"""
        return [
            {
                "prompt": "请解释人工智能是什么？",
                "reference": "人工智能（AI）是模拟人类智能的计算机系统，这些系统被编程为像人类一样思考和学习。它包括机器学习、自然语言处理、计算机视觉和机器人技术等领域。",
                "category": "factual",
                "difficulty": "easy"
            },
            {
                "prompt": "描述一下什么是区块链技术。",
                "reference": "区块链是一种分布式数据库技术，它允许在不依赖中央权威的情况下安全地记录交易和管理数据。它使用加密技术将交易组织成块，然后添加到链中，创建一个不可篡改的交易记录。",
                "category": "factual",
                "difficulty": "medium"
            },
            {
                "prompt": "写一首关于春天的短诗。",
                "reference": "",
                "category": "creative",
                "difficulty": "medium"
            },
            {
                "prompt": "解释一下中国的传统节日春节的由来和习俗。",
                "reference": "春节是中国最重要的传统节日，起源于祭祀祖先和祈求丰收的活动。习俗包括贴春联、放鞭炮、团圆饭、发红包、舞龙舞狮等，象征着辞旧迎新和祈求来年好运。",
                "category": "cultural",
                "difficulty": "medium"
            },
            {
                "prompt": "如果一个数列的第一项是1，第二项是1，从第三项开始每一项等于前两项之和，那么这个数列的第10项是多少？",
                "reference": "这个数列是斐波那契数列。斐波那契数列的前几项是：1, 1, 2, 3, 5, 8, 13, 21, 34, 55。所以第10项是55。",
                "category": "reasoning",
                "difficulty": "medium"
            }
        ]
    
    def add_custom_evaluation_set(self, name: str, evaluation_set: List[Dict[str, str]]) -> None:
        """
        Add a custom evaluation set
        
        Args:
            name: Name of the evaluation set
            evaluation_set: List of evaluation items
        """
        self.evaluation_sets[name] = evaluation_set
        if self.logger:
            self.logger.info(f"Added custom evaluation set '{name}' with {len(evaluation_set)} items")
    
    def add_model(self, model_name: str, model_generator_fn: Callable[[str], str]) -> 'ModelEvaluator':
        """
        Add a model to be evaluated
        
        Args:
            model_name: Name of the model
            model_generator_fn: Function that generates text responses given a prompt
            
        Returns:
            self: Supports method chaining
        """
        self.metrics[model_name] = {}
        self.results[model_name] = {}
        
        # Save the generator function
        self.metrics[model_name]["generator"] = model_generator_fn
        
        if self.logger:
            self.logger.info(f"Added model '{model_name}' for evaluation")
        
        return self
    
    def evaluate(self, eval_set_names: Optional[List[str]] = None, 
                num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate all added models
        
        Args:
            eval_set_names: List of evaluation set names to use, None means all
            num_samples: Number of samples to evaluate from each set, None means all
            
        Returns:
            dict: Evaluation results
        """
        if not self.metrics:
            if self.logger:
                self.logger.warning("No models added, cannot evaluate")
            return {}
        
        # Determine which evaluation sets to use
        if eval_set_names is None:
            eval_set_names = list(self.evaluation_sets.keys())
        
        # Evaluate each model
        for model_name, model_metrics in self.metrics.items():
            if "generator" not in model_metrics:
                if self.logger:
                    self.logger.warning(f"Model '{model_name}' does not have a generator function, skipping")
                continue
            
            generator_fn = model_metrics["generator"]
            
            if self.logger:
                self.logger.info(f"Starting evaluation for model: {model_name}")
            
            # Evaluate on each dataset
            for set_name in eval_set_names:
                if set_name not in self.evaluation_sets:
                    if self.logger:
                        self.logger.warning(f"Evaluation set '{set_name}' does not exist, skipping")
                    continue
                
                eval_set = self.evaluation_sets[set_name]
                
                # Limit number of samples if specified
                if num_samples is not None and num_samples < len(eval_set):
                    eval_samples = eval_set[:num_samples]
                else:
                    eval_samples = eval_set
                
                if self.logger:
                    self.logger.info(f"Evaluation set '{set_name}': {len(eval_samples)} samples")
                
                # Perform evaluation
                results = self._evaluate_samples(model_name, generator_fn, eval_samples, set_name)
                
                # Store results
                self.results[model_name][set_name] = results
        
        # Calculate overall scores
        self._calculate_overall_scores()
        
        # Generate evaluation reports and visualizations
        self._generate_evaluation_reports()
        
        return self.results
    
    def _evaluate_samples(self, model_name: str, generator_fn: Callable[[str], str], 
                         samples: List[Dict[str, str]], set_name: str) -> Dict[str, Any]:
        """
        Evaluate a specific model on a set of samples
        
        Args:
            model_name: Name of the model
            generator_fn: Generator function
            samples: List of samples to evaluate
            set_name: Name of the evaluation set
            
        Returns:
            dict: Evaluation results
        """
        results = {
            "samples": [],
            "scores": {},
            "average_score": 0.0
        }
        
        total_score = 0.0
        total_samples = len(samples)
        
        for i, sample in enumerate(samples):
            prompt = sample.get("prompt", "")
            reference = sample.get("reference", "")
            category = sample.get("category", "general")
            difficulty = sample.get("difficulty", "medium")
            
            # Generate response
            try:
                if self.logger:
                    self.logger.info(f"[{model_name}] Generating sample {i+1}/{total_samples} ({set_name}, {category})")
                
                response = generator_fn(prompt)
                
                # Score the response
                score, feedback = self._score_response(response, reference, category, set_name)
                
                # Create sample result
                sample_result = {
                    "prompt": prompt,
                    "response": response,
                    "reference": reference,
                    "category": category,
                    "difficulty": difficulty,
                    "score": score,
                    "feedback": feedback
                }
                
                # Add to results
                results["samples"].append(sample_result)
                total_score += score
                
                # Update category scores
                if category not in results["scores"]:
                    results["scores"][category] = {"total": 0, "count": 0, "average": 0}
                
                results["scores"][category]["total"] += score
                results["scores"][category]["count"] += 1
                
                if self.logger:
                    self.logger.info(f"[{model_name}] Sample {i+1} score: {score}/100 - {feedback}")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"[{model_name}] Error evaluating sample {i+1}: {e}")
                    self.logger.error(traceback.format_exc())
                
                # Add error sample
                sample_result = {
                    "prompt": prompt,
                    "response": f"Error: {str(e)}",
                    "reference": reference,
                    "category": category,
                    "difficulty": difficulty,
                    "score": 0,
                    "feedback": f"Generation failed: {str(e)}"
                }
                
                # Add to results
                results["samples"].append(sample_result)
        
        # Calculate average score
        if total_samples > 0:
            results["average_score"] = total_score / total_samples
        
        # Calculate average scores for each category
        for category, data in results["scores"].items():
            if data["count"] > 0:
                data["average"] = data["total"] / data["count"]
        
        if self.logger:
            self.logger.info(f"[{model_name}] '{set_name}' evaluation complete, average score: {results['average_score']:.2f}/100")
        
        return results
    
    def _score_response(self, response: str, reference: str, category: str, set_name: str) -> Tuple[float, str]:
        """
        Score a model response
        
        Args:
            response: Model generated response
            reference: Reference answer
            category: Sample category
            set_name: Evaluation set name
            
        Returns:
            tuple: (score, feedback)
        """
        # Choose appropriate scoring method based on set and category
        if set_name == "coding":
            return self._score_code(response, reference, category)
        elif set_name == "reasoning":
            return self._score_reasoning(response, reference, category)
        elif set_name == "creative":
            return self._score_creative(response, category)
        elif set_name == "chinese":
            return self._score_chinese(response, reference, category)
        else:
            # General scoring
            return self._score_general(response, reference, category)
    
    def _score_general(self, response: str, reference: str, category: str) -> Tuple[float, str]:
        """General scoring function"""
        # Basic validity check
        if not response or len(response.strip()) < 5:
            return 0.0, "Response too short or empty"
        
        try:
            # Use TF-IDF and cosine similarity for content similarity
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([reference, response])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                # Fallback for empty or incompatible texts
                similarity = 0.0
            
            # Convert to score (0-100)
            score = min(100, max(0, similarity * 100))
            
            # Add length reward/penalty
            ref_len = len(reference.split())
            resp_len = len(response.split())
            
            length_ratio = min(resp_len / max(1, ref_len), max(1, ref_len) / max(1, resp_len))
            length_bonus = (length_ratio - 0.5) * 20  # Higher bonus for length ratio closer to 1
            
            score = min(100, max(0, score + length_bonus))
            
            # Generate feedback
            if score >= 80:
                feedback = "Excellent response, highly similar to reference answer"
            elif score >= 60:
                feedback = "Good response, captures most key information"
            elif score >= 40:
                feedback = "Acceptable response, contains some relevant information"
            else:
                feedback = "Poor response, differs significantly from reference answer"
            
            return score, feedback
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in general scoring: {e}")
                self.logger.error(traceback.format_exc())
            return 30.0, f"Scoring error: {e}"
    
    def _score_code(self, response: str, reference: str, category: str) -> Tuple[float, str]:
        """Code scoring function"""
        # Check if response contains a code block
        has_code_block = bool(re.search(r'```\w*\n[\s\S]*?\n```', response))
        
        # Extract code
        if has_code_block:
            code_match = re.search(r'```(?:\w*)\n([\s\S]*?)\n```', response)
            code = code_match.group(1) if code_match else ""
        else:
            # Try to use the whole response as code
            code = response
        
        # Basic validity check
        if not code or len(code.strip()) < 5:
            return 10.0, "No valid code found"
        
        # Simplified scoring: check for key elements
        # For Python, check function definitions, control structures, etc.
        if category == "python":
            key_elements = [
                ("function definition", r"def\s+\w+\s*\("),
                ("return statement", r"return\s+"),
                ("conditional", r"if\s+.*:"),
                ("loop", r"for\s+.*:|while\s+.*:"),
                ("list operation", r"\[\s*.*\s*\]"),
                ("dictionary operation", r"\{\s*.*:\s*.*\s*\}")
            ]
            
            matches = 0
            total = len(key_elements)
            matched_elements = []
            
            for name, pattern in key_elements:
                if re.search(pattern, code):
                    matches += 1
                    matched_elements.append(name)
            
            # Calculate syntax correctness (simplified)
            if "def" in code and ":" not in code:
                syntax_score = 30  # Function definition syntax error
            elif "for" in code and ":" not in code:
                syntax_score = 40  # Loop syntax error
            elif "if" in code and ":" not in code:
                syntax_score = 40  # Conditional syntax error
            else:
                syntax_score = 80  # Assume basic syntax is correct
            
            # Functionality completeness
            completeness_score = (matches / total) * 100 if total > 0 else 50
            
            # Final score
            score = (syntax_score * 0.6) + (completeness_score * 0.4)
            
            # Generate feedback
            feedback = f"Code contains: {', '.join(matched_elements)}. "
            
            if score >= 80:
                feedback += "High quality code with correct syntax and complete functionality."
            elif score >= 60:
                feedback += "Good code that mostly implements the required functionality."
            elif score >= 40:
                feedback += "Average code with some issues."
            else:
                feedback += "Poor quality code with significant issues."
            
            return score, feedback
        
        # For other programming languages, use general scoring method
        return self._score_general(code, reference, category)
    
    def _score_reasoning(self, response: str, reference: str, category: str) -> Tuple[float, str]:
        """Reasoning scoring function"""
        # Check for reasoning step markers
        reasoning_markers = [
            "首先", "其次", "然后", "最后", "因此", "所以", "因为", 
            "但是", "另外", "此外", "分析", "考虑", "假设", "推导",
            "一步一步", "思考", "推理", "结论", "总结", "实际上",
            "First", "Second", "Then", "Finally", "Therefore", "Because",
            "However", "Additionally", "Furthermore", "Let's analyze",
            "Consider", "Assume", "Step by step", "Think through",
            "Reasoning", "Conclusion", "In summary", "In fact"
        ]
        
        # Count reasoning markers
        marker_count = sum(1 for marker in reasoning_markers if marker in response)
        
        # Steps clarity score
        if marker_count >= 5:
            steps_score = 100
        elif marker_count >= 3:
            steps_score = 80
        elif marker_count >= 1:
            steps_score = 60
        else:
            steps_score = 40
        
        # Extract conclusion
        conclusion_markers = ["因此", "所以", "总结", "结论", "综上所述", "Therefore", "Thus", "In conclusion", "To sum up"]
        conclusion = ""
        
        for marker in conclusion_markers:
            if marker in response:
                parts = response.split(marker)
                if len(parts) > 1:
                    conclusion = parts[-1].strip()
                    break
        
        # If no explicit conclusion marker, try using the last paragraph
        if not conclusion:
            paragraphs = response.split('\n\n')
            if paragraphs:
                conclusion = paragraphs[-1].strip()
        
        # Conclusion correctness (compared to reference)
        if conclusion and reference:
            # Use simplified similarity calculation
            words_in_common = set(conclusion.lower().split()) & set(reference.lower().split())
            conclusion_score = min(100, len(words_in_common) * 10)
        else:
            conclusion_score = 50  # Can't determine conclusion correctness
        
        # Final score
        score = (steps_score * 0.6) + (conclusion_score * 0.4)
        
        # Generate feedback
        feedback = f"Contains {marker_count} reasoning markers"
        
        if score >= 80:
            feedback += ", very clear reasoning process with accurate conclusion."
        elif score >= 60:
            feedback += ", reasonably clear reasoning process with mostly correct conclusion."
        elif score >= 40:
            feedback += ", somewhat unclear reasoning process with questionable conclusion."
        else:
            feedback += ", unclear reasoning process with potentially incorrect conclusion."
        
        return score, feedback
    
    def _score_creative(self, response: str, category: str) -> Tuple[float, str]:
        """Creative content scoring function"""
        if not response or len(response.strip()) < 10:
            return 0.0, "Response too short or empty"
        
        # For creative content, assess dimensions including:
        # 1. Length and structure
        # 2. Vocabulary diversity
        # 3. Emotional and vivid language
        
        # Length and structure
        words = response.split()
        word_count = len(words)
        
        if word_count < 20:
            length_score = 30
        elif word_count < 50:
            length_score = 60
        elif word_count < 100:
            length_score = 80
        else:
            length_score = 100
        
        # Structure score (check paragraphs)
        paragraphs = response.split('\n\n')
        if len(paragraphs) >= 3:
            structure_score = 100
        elif len(paragraphs) == 2:
            structure_score = 80
        else:
            structure_score = 60
        
        # Vocabulary diversity
        unique_words = set(word.lower() for word in words)
        if word_count > 0:
            diversity = len(unique_words) / word_count
            if diversity > 0.7:
                diversity_score = 100
            elif diversity > 0.5:
                diversity_score = 80
            elif diversity > 0.3:
                diversity_score = 60
            else:
                diversity_score = 40
        else:
            diversity_score = 0
        
        # Overall score
        score = (length_score * 0.3) + (structure_score * 0.3) + (diversity_score * 0.4)
        
        # Generate feedback
        feedback = f"Word count: {word_count}, Paragraphs: {len(paragraphs)}, Vocabulary diversity: {diversity:.2f}"
        
        if score >= 80:
            feedback += ". Excellent creative expression with good structure and rich vocabulary."
        elif score >= 60:
            feedback += ". Good creative expression with reasonable structure."
        elif score >= 40:
            feedback += ". Average creative expression with room for improvement."
        else:
            feedback += ". Poor creative expression, needs significant improvement."
        
        return score, feedback
    
    def _score_chinese(self, response: str, reference: Optional[str], category: str, set_name: Optional[str] = None) -> Tuple[float, str]:
            """中文评分函数"""
            # 中文评分与通用评分类似，但需要特别考虑中文特性
        
            # 基本有效性检查
            if not response or len(response.strip()) < 5:
                return 0, "回应过短或为空"
        
            # 检查是否包含中文
            if not re.search('[\u4e00-\u9fff]', response):
                return 30, "回应不包含中文"
        
            # 中文自然语言处理评分
            # 检查语句完整性（以标点符号计算）
            punctuations = ['。', '！', '？', '；', '，']
            sentence_ends = sum(response.count(p) for p in punctuations)
        
            if sentence_ends >= 3:
                structure_score = 100
            elif sentence_ends >= 2:
                structure_score = 80
            elif sentence_ends >= 1:
                structure_score = 60
            else:
                structure_score = 40
        
            # 与参考答案比较（如果有）
            if reference:
                # 计算关键词重叠
                ref_chars = set(reference)
                resp_chars = set(response)
                common_chars = ref_chars & resp_chars
            
                if len(ref_chars) > 0:
                    similarity_score = min(100, (len(common_chars) / len(ref_chars)) * 100)
                else:
                    similarity_score = 50
            else:
                similarity_score = 70  # 没有参考答案，给予中等分数
        
            # 最终分数
            score = (structure_score * 0.4) + (similarity_score * 0.6)
        
            # 生成反馈
            feedback = f"句子结构评分:{structure_score}, 内容相关性:{similarity_score:.1f}"
        
            if score >= 80:
                feedback += "。中文表达很好，内容相关性高。"
            elif score >= 60:
                feedback += "。中文表达良好，内容基本相关。"
            elif score >= 40:
                feedback += "。中文表达一般，内容相关性有限。"
            else:
                feedback += "。中文表达较差，与主题关联不大。"
        
            return score, feedback

    def _calculate_overall_scores(self) -> None:
        """
        Calculate overall scores across all evaluation sets
        """
        for model_name, model_results in self.results.items():
            total_score = 0.0
            total_samples = 0
            category_scores = defaultdict(lambda: {"total": 0.0, "count": 0})
            
            # Collect scores from all evaluation sets
            for set_name, set_results in model_results.items():
                avg_score = set_results.get("average_score", 0.0)
                total_score += avg_score * len(set_results.get("samples", []))
                total_samples += len(set_results.get("samples", []))
                
                # Collect category scores
                for sample in set_results.get("samples", []):
                    category = sample.get("category", "general")
                    score = sample.get("score", 0.0)
                    
                    category_scores[category]["total"] += score
                    category_scores[category]["count"] += 1
            
            # Calculate overall average
            overall_average = total_score / total_samples if total_samples > 0 else 0.0
            
            # Calculate category averages
            category_averages = {}
            for category, data in category_scores.items():
                if data["count"] > 0:
                    category_averages[category] = data["total"] / data["count"]
            
            # Store overall scores
            self.results[model_name]["overall"] = {
                "average_score": overall_average,
                "total_samples": total_samples,
                "category_scores": category_averages
            }
            
            if self.logger:
                self.logger.info(f"Overall score for '{model_name}': {overall_average:.2f}/100")
    
    def _generate_evaluation_reports(self) -> None:
        """
        Generate evaluation reports and visualizations
        """
        # Skip if no visualizer or no results
        if not hasattr(self, 'visualizer') or not self.results:
            return
        
        try:
            # Generate model comparison chart
            model_scores = {model: results.get("overall", {}).get("average_score", 0.0) 
                           for model, results in self.results.items()}
            
            self.visualizer.plot_model_comparison(model_scores)
            
            # Generate category comparison for each model
            for model_name, model_results in self.results.items():
                overall = model_results.get("overall", {})
                category_scores = overall.get("category_scores", {})
                
                if category_scores:
                    self.visualizer.plot_category_comparison(model_name, category_scores)
                    
                # Generate detailed performance report
                self._generate_detailed_report(model_name, model_results)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating evaluation reports: {e}")
                self.logger.error(traceback.format_exc())
    
    def _generate_detailed_report(self, model_name: str, model_results: Dict[str, Any]) -> None:
        """
        Generate a detailed performance report for a model
        
        Args:
            model_name: Name of the model
            model_results: Evaluation results for the model
        """
        try:
            # Skip if no overall results
            if "overall" not in model_results:
                return
            
            # Create report text
            report = f"# Performance Report for {model_name}\n\n"
            
            # Overall score
            overall = model_results["overall"]
            report += f"## Overall Performance\n\n"
            report += f"- **Average Score**: {overall['average_score']:.2f}/100\n"
            report += f"- **Total Samples**: {overall['total_samples']}\n\n"
            
            # Category scores
            report += f"## Performance by Category\n\n"
            for category, score in overall.get("category_scores", {}).items():
                report += f"- **{category.capitalize()}**: {score:.2f}/100\n"
            report += "\n"
            
            # Evaluation set scores
            report += f"## Performance by Evaluation Set\n\n"
            for set_name, set_results in model_results.items():
                if set_name == "overall":
                    continue
                    
                avg_score = set_results.get("average_score", 0.0)
                num_samples = len(set_results.get("samples", []))
                
                report += f"### {set_name.capitalize()} Evaluation Set\n\n"
                report += f"- **Average Score**: {avg_score:.2f}/100\n"
                report += f"- **Samples**: {num_samples}\n\n"
                
                # Category breakdown for this set
                category_scores = set_results.get("scores", {})
                if category_scores:
                    report += "#### Category Breakdown\n\n"
                    for category, data in category_scores.items():
                        avg = data.get("average", 0.0)
                        count = data.get("count", 0)
                        report += f"- **{category.capitalize()}**: {avg:.2f}/100 ({count} samples)\n"
                report += "\n"
            
            # Sample highlights 
            report += f"## Sample Highlights\n\n"
            
            # Find best and worst samples
            all_samples = []
            for set_results in model_results.values():
                if isinstance(set_results, dict) and "samples" in set_results:
                    all_samples.extend(set_results["samples"])
            
            # Sort by score
            all_samples.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Best samples (top 3)
            if all_samples:
                report += "### Best Samples\n\n"
                for i, sample in enumerate(all_samples[:3]):
                    report += f"**Sample {i+1}** (Score: {sample.get('score', 0):.2f}/100)\n\n"
                    report += f"- **Prompt**: {sample.get('prompt', '')}\n"
                    report += f"- **Response**: {sample.get('response', '')[:200]}...\n"
                    report += f"- **Feedback**: {sample.get('feedback', '')}\n\n"
                
                # Worst samples (bottom 3)
                report += "### Worst Samples\n\n"
                for i, sample in enumerate(all_samples[-3:]):
                    report += f"**Sample {i+1}** (Score: {sample.get('score', 0):.2f}/100)\n\n"
                    report += f"- **Prompt**: {sample.get('prompt', '')}\n"
                    report += f"- **Response**: {sample.get('response', '')[:200]}...\n"
                    report += f"- **Feedback**: {sample.get('feedback', '')}\n\n"
            
            # Visualize the report
            self.visualizer.save_report(model_name, report)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating detailed report for {model_name}: {e}")
                self.logger.error(traceback.format_exc())
    
    def get_best_model(self) -> Optional[str]:
        """
        Get the name of the best performing model
        
        Returns:
            str or None: Name of the best model, or None if no models evaluated
        """
        if not self.results:
            return None
            
        best_model = None
        best_score = -1
        
        for model_name, model_results in self.results.items():
            if "overall" in model_results:
                score = model_results["overall"].get("average_score", 0.0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model
    
    def print_summary(self) -> None:
        """
        Print a summary of evaluation results
        """
        if not self.results:
            print("No evaluation results available.")
            return
            
        print("\n" + "="*60)
        print("Evaluation Summary".center(60))
        print("="*60 + "\n")
        
        # Overall scores
        print("Overall Model Performance:")
        print("-" * 60)
        print(f"{'Model':<30} {'Score':<10} {'Samples':<10}")
        print("-" * 60)
        
        for model_name, model_results in self.results.items():
            if "overall" in model_results:
                overall = model_results["overall"]
                score = overall.get("average_score", 0.0)
                samples = overall.get("total_samples", 0)
                print(f"{model_name:<30} {score:>8.2f}/100 {samples:>10}")
        
        print("\n")
        
        # Best model
        best_model = self.get_best_model()
        if best_model:
            best_score = self.results[best_model]["overall"].get("average_score", 0.0)
            print(f"Best Model: {best_model} (Score: {best_score:.2f}/100)")
        
        print("\n" + "="*60)
    
    def export_results(self, output_path: str) -> bool:
        """
        Export evaluation results to a file
        
        Args:
            output_path: Path to export the results
            
        Returns:
            bool: Success or failure
        """
        try:
            import json
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
                
            if self.logger:
                self.logger.info(f"Exported evaluation results to {output_path}")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error exporting results: {e}")
                self.logger.error(traceback.format_exc())
            return False


def safe_decode(tokenizer, token_ids, skip_special_tokens=True):
    """
    安全的解码函数，处理中文和其他特殊字符
    
    Args:
        tokenizer: 分词器
        token_ids: 要解码的token ID列表
        skip_special_tokens: 是否跳过特殊标记
    
    Returns:
        str: 解码后的文本
    """
    try:
        # 首先尝试标准解码
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except KeyError as e:
        # 如果是ChineseTokenizer，使用自定义解码
        if hasattr(tokenizer, 'decoder'):
            return custom_decode(tokenizer, token_ids, skip_special_tokens)
        # 否则尝试逐个解码
        text = ""
        for token_id in token_ids.tolist():
            try:
                if skip_special_tokens and hasattr(tokenizer, 'all_special_ids') and token_id in tokenizer.all_special_ids:
                    continue
                token = tokenizer.convert_ids_to_tokens(token_id)
                text += token
            except:
                # 如果解码失败，添加占位符
                text += "[?]"
        return text


def custom_decode(tokenizer, token_ids, skip_special_tokens=True):
    """
    为ChineseTokenizer自定义的解码函数
    
    Args:
        tokenizer: 分词器
        token_ids: 要解码的token ID列表
        skip_special_tokens: 是否跳过特殊标记
    
    Returns:
        str: 解码后的文本
    """
    if hasattr(tokenizer, 'decoder'):
        # 对于自定义的ChineseTokenizer
        text = ""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        for token_id in token_ids:
            if skip_special_tokens and hasattr(tokenizer, 'special_tokens_map') and \
               token_id in tokenizer.special_tokens_map.values():
                continue
                
            if token_id in tokenizer.decoder:
                text += tokenizer.decoder[token_id]
            else:
                # 未知ID，添加占位符
                text += "[UNK]"
        return text
    else:
        # 对于标准的tokenizer，使用一个简单的备用方法
        return " ".join([str(id) for id in token_ids.tolist()])


def evaluate_model(model_path: str,
                   output_dir: str = None,
                   eval_sets: list = None,
                   num_samples: int = None,
                   logger=None,
                   force_cpu: bool = False) -> dict:
    """
    评估模型性能
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        eval_sets: 要评估的集合列表
        num_samples: 每个集合中要评估的样本数
        logger: 日志记录器
        force_cpu: 是否强制使用CPU
        
    Returns:
        dict: 评估结果
    """
    try:
        # 设置设备
        device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        if force_cpu and logger:
            logger.info("强制使用CPU进行评估")
        
        # 加载模型
        from apt.apt_model.training.checkpoint import load_model
        model, tokenizer, config = load_model(model_path, device=device)
        model.eval()
        
        # 创建评估器
        evaluator = ModelEvaluator(logger=logger)
        
        # 添加模型到评估器
        # 使用安全的生成函数
        def generator_fn(prompt: str) -> str:
            try:
                device = next(model.parameters()).device
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                # 生成的最大长度为当前长度 + 50
                generated_ids = model.generate(input_ids, max_length=input_ids.size(1) + 50)
                return safe_decode(tokenizer, generated_ids[0], skip_special_tokens=True)
            except Exception as e:
                if logger:
                    logger.warning(f"生成过程中出现错误: {e}")
                    logger.debug(traceback.format_exc())
                # 返回错误信息
                return f"生成错误: {str(e)}"
        
        evaluator.add_model("model", generator_fn)
        
        results = evaluator.evaluate(eval_set_names=eval_sets, num_samples=num_samples)
        
        if output_dir:
            import os, json
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "evaluation_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            if logger:
                logger.info(f"评估报告已保存至 {report_path}")
        
        return results
    except Exception as e:
        if logger:
            logger.error(f"评估模型时出错: {e}")
            logger.error(traceback.format_exc())
        print(f"评估模型时出错: {e}")
        return {}