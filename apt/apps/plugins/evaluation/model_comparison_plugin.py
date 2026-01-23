#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Model Comparison Module
Provides functionality for comparing performance across different models
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from ..utils.visualization import ModelVisualizer
from .model_evaluator import ModelEvaluator, evaluate_model
from ..utils.cache_manager import CacheManager
from apt.apt_model.training.checkpoint import load_model


class ModelComparison:
    """
    Model comparison class for evaluating multiple APT models against each other
    """
    
    def __init__(self, logger=None, output_dir=None, cache_manager=None):
        """
        Initialize the model comparison tool
        
        Args:
            logger: Optional logger instance
            output_dir: Directory to save comparison results
            cache_manager: Optional cache manager instance
        """
        self.logger = logger or logging.getLogger('apt_comparison')
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        self.cache_manager = cache_manager
        self.models = {}
        self.results = {}
        self.evaluator = ModelEvaluator(logger=logger)
        self.visualizer = ModelVisualizer(cache_manager=cache_manager, logger=logger)
    
    def add_model(self, model_name: str, model_path: str) -> bool:
        """
        Add a model to the comparison
        
        Args:
            model_name: Name to identify the model in results
            model_path: Path to the saved model
            
        Returns:
            bool: Whether the model was successfully added
        """
        try:
            if model_name in self.models:
                self.logger.warning(f"Model '{model_name}' already exists, overwriting")
            
            self.models[model_name] = model_path
            self.logger.info(f"Added model '{model_name}' from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add model '{model_name}': {e}")
            return False
    
    def compare(self, eval_sets=None, num_samples=None, prompts=None) -> Dict[str, Any]:
        """
        Compare all added models on standard evaluation sets and custom prompts
        
        Args:
            eval_sets: List of evaluation set names to use
            num_samples: Number of samples to use from each evaluation set
            prompts: List of custom prompts to test
            
        Returns:
            dict: Comparison results
        """
        if not self.models:
            self.logger.warning("No models added for comparison")
            return {}
        
        self.logger.info(f"Starting comparison of {len(self.models)} models")
        
        # Prepare results structure
        comparison_results = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "models": list(self.models.keys()),
                "eval_sets": eval_sets
            },
            "performance": {},
            "prompts": {},
            "summary": {}
        }
        
        # Evaluate each model using the evaluator
        for model_name, model_path in self.models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Get model evaluation results
            eval_results = evaluate_model(
                model_path=model_path,
                output_dir=os.path.join(self.output_dir, model_name) if self.output_dir else None,
                eval_sets=eval_sets,
                num_samples=num_samples,
                logger=self.logger
            )
            
            # Store results
            comparison_results["performance"][model_name] = eval_results
            
            # If custom prompts provided, test them too
            if prompts:
                prompt_results = self._evaluate_custom_prompts(model_name, model_path, prompts)
                comparison_results["prompts"][model_name] = prompt_results
        
        # Generate comparative statistics and summary
        self._generate_summary(comparison_results)
        
        # Create visualization
        self._create_visualizations(comparison_results)
        
        # Save results
        self._save_results(comparison_results)
        
        self.results = comparison_results
        return comparison_results
    
    def _evaluate_custom_prompts(self, model_name: str, model_path: str, 
                               prompts: List[str]) -> Dict[str, Any]:
        """Evaluate a model on custom prompts"""
        model, tokenizer, config = load_model(model_path)
        
        results = {}
        
        for i, prompt in enumerate(prompts):
            try:
                # Generate text
                generated_text, _, temperature, top_p = self._generate_text(model, tokenizer, prompt)
                
                # Evaluate quality
                quality_score, quality_feedback = self._evaluate_quality(generated_text)
                
                results[f"prompt_{i+1}"] = {
                    "prompt": prompt,
                    "response": generated_text,
                    "quality_score": quality_score,
                    "quality_feedback": quality_feedback,
                    "temperature": temperature,
                    "top_p": top_p
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating prompt {i+1} for model {model_name}: {e}")
                results[f"prompt_{i+1}"] = {
                    "prompt": prompt,
                    "error": str(e)
                }
        
        return results
    
    def _generate_text(self, model, tokenizer, prompt: str, 
                     max_steps=50, temperature=0.7, top_p=0.9) -> Tuple[str, Any, float, float]:
        """Generate text from a model"""
        from ..generation.generator import generate_natural_text
        
        return generate_natural_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_steps=max_steps,
            temperature=temperature,
            top_p=top_p
        )
    
    def _evaluate_quality(self, text: str) -> Tuple[float, str]:
        """Evaluate text quality"""
        from ..generation.evaluator import evaluate_text_quality
        
        return evaluate_text_quality(text)
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """Generate summary statistics for the comparison"""
        summary = {}
        
        # For each performance metric/category, rank models
        if "performance" in results:
            performance = results["performance"]
            models = list(performance.keys())
            
            # Collect overall scores
            overall_scores = {}
            for model_name in models:
                try:
                    # Extract the main evaluation score for each model
                    overall_scores[model_name] = performance[model_name].get("overall", {}).get("average", 0)
                except (KeyError, AttributeError):
                    overall_scores[model_name] = 0
            
            # Rank models by overall score
            ranked_models = sorted(
                [(name, score) for name, score in overall_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            summary["rankings"] = {
                "overall": [{"model": name, "score": score} for name, score in ranked_models]
            }
            
            # Find best model per evaluation set if available
            if any("by_set" in performance.get(model, {}).get("overall", {}) for model in models):
                best_by_set = {}
                
                for model_name in models:
                    if "overall" not in performance.get(model_name, {}):
                        continue
                        
                    by_set = performance[model_name]["overall"].get("by_set", {})
                    
                    for set_name, score in by_set.items():
                        if set_name not in best_by_set or score > best_by_set[set_name]["score"]:
                            best_by_set[set_name] = {
                                "model": model_name,
                                "score": score
                            }
                
                summary["best_by_set"] = best_by_set
        
        # Analyze custom prompt results if available
        if "prompts" in results and results["prompts"]:
            prompt_scores = {}
            
            for model_name, prompt_results in results["prompts"].items():
                scores = []
                
                for prompt_key, prompt_data in prompt_results.items():
                    if "quality_score" in prompt_data:
                        scores.append(prompt_data["quality_score"])
                
                if scores:
                    prompt_scores[model_name] = {
                        "average": sum(scores) / len(scores) if scores else 0,
                        "min": min(scores) if scores else 0,
                        "max": max(scores) if scores else 0
                    }
            
            # Rank by average prompt score
            prompt_ranked = sorted(
                [(name, data["average"]) for name, data in prompt_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            summary["prompt_rankings"] = [{"model": name, "score": score} for name, score in prompt_ranked]
            summary["prompt_scores"] = prompt_scores
        
        # Add summary to results
        results["summary"] = summary
    
    def _create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create comparative visualizations"""
        if not self.output_dir:
            return
        
        try:
            # Create overall score comparison chart
            if "summary" in results and "rankings" in results["summary"]:
                rankings = results["summary"]["rankings"]["overall"]
                model_names = [r["model"] for r in rankings]
                scores = [r["score"] for r in rankings]
                
                self.visualizer.create_comparison_bar_chart(
                    {model: {"average": score} for model, score in zip(model_names, scores)},
                    "average",
                    output_path=os.path.join(self.output_dir, "overall_comparison.png"),
                    title="Model Overall Performance Comparison"
                )
            
            # Create evaluation set comparison radar charts
            if any("by_set" in results.get("performance", {}).get(model, {}).get("overall", {}) 
                  for model in results.get("performance", {})):
                
                for model_name, performance in results.get("performance", {}).items():
                    if "overall" in performance and "by_set" in performance["overall"]:
                        by_set = performance["overall"]["by_set"]
                        
                        if by_set:
                            self.visualizer.create_evaluation_radar_chart(
                                by_set,
                                output_path=os.path.join(self.output_dir, f"{model_name}_radar.png"),
                                title=f"{model_name} Performance by Category"
                            )
            
            # Create prompt quality comparison
            if "prompt_scores" in results.get("summary", {}):
                prompt_scores = results["summary"]["prompt_scores"]
                
                if prompt_scores:
                    # Get list of models
                    models = list(prompt_scores.keys())
                    # Get average scores for each model
                    avg_scores = [prompt_scores[m]["average"] for m in models]
                    # Get min and max scores
                    min_scores = [prompt_scores[m]["min"] for m in models]
                    max_scores = [prompt_scores[m]["max"] for m in models]
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(models))
                    width = 0.7
                    
                    bars = ax.bar(x, avg_scores, width, label='Average Score')
                    
                    # Add error bars for min/max
                    ax.errorbar(x, avg_scores, 
                              yerr=[
                                  [avg - min_val for avg, min_val in zip(avg_scores, min_scores)],
                                  [max_val - avg for avg, max_val in zip(avg_scores, max_scores)]
                              ],
                              fmt='none', ecolor='black', capsize=5)
                    
                    ax.set_xlabel('Models')
                    ax.set_ylabel('Quality Score')
                    ax.set_title('Custom Prompt Response Quality')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models, rotation=45, ha='right')
                    ax.set_ylim(0, 100)
                    
                    # Add data labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.1f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "prompt_quality_comparison.png"))
                    plt.close()
        
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save comparison results to output directory"""
        if not self.output_dir:
            return
        
        try:
            # Save full results as JSON
            results_path = os.path.join(self.output_dir, "comparison_results.json")
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved comparison results to {results_path}")
            
            # Also create a summary markdown file
            self._create_summary_report(results)
            
        except Exception as e:
            self.logger.error(f"Error saving comparison results: {e}")
    
    def _create_summary_report(self, results: Dict[str, Any]) -> None:
        """Create a summary report in markdown format"""
        try:
            report_path = os.path.join(self.output_dir, "comparison_summary.md")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# APT模型比较报告\n\n")
                f.write(f"生成时间: {results['metadata']['timestamp']}\n\n")
                
                # Models compared
                f.write(f"## 比较的模型\n\n")
                for model_name in results['metadata']['models']:
                    f.write(f"- {model_name}\n")
                
                # Overall rankings
                if "summary" in results and "rankings" in results["summary"]:
                    f.write(f"\n## 总体排名\n\n")
                    f.write(f"| 排名 | 模型名称 | 总分 |\n")
                    f.write(f"|------|---------|------|\n")
                    
                    for i, ranking in enumerate(results["summary"]["rankings"]["overall"]):
                        f.write(f"| {i+1} | {ranking['model']} | {ranking['score']:.2f} |\n")
                
                # Best by category
                if "summary" in results and "best_by_set" in results["summary"]:
                    f.write(f"\n## 各类别最佳模型\n\n")
                    f.write(f"| 评估类别 | 最佳模型 | 得分 |\n")
                    f.write(f"|----------|----------|------|\n")
                    
                    for set_name, data in results["summary"]["best_by_set"].items():
                        f.write(f"| {set_name} | {data['model']} | {data['score']:.2f} |\n")
                
                # Prompt comparison
                if "summary" in results and "prompt_rankings" in results["summary"]:
                    f.write(f"\n## 自定义提示测试排名\n\n")
                    f.write(f"| 排名 | 模型名称 | 平均质量得分 |\n")
                    f.write(f"|------|---------|------------|\n")
                    
                    for i, ranking in enumerate(results["summary"]["prompt_rankings"]):
                        f.write(f"| {i+1} | {ranking['model']} | {ranking['score']:.2f} |\n")
                
                # Images
                f.write(f"\n## 可视化图表\n\n")
                f.write(f"![整体性能比较](./overall_comparison.png)\n\n")
                f.write(f"![提示质量比较](./prompt_quality_comparison.png)\n\n")
                
                for model_name in results['metadata']['models']:
                    radar_path = f"{model_name}_radar.png"
                    if os.path.exists(os.path.join(self.output_dir, radar_path)):
                        f.write(f"![{model_name} 雷达图](./{radar_path})\n\n")
            
            self.logger.info(f"Created summary report at {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")

def compare_models(model_paths: Dict[str, str], output_dir: Optional[str] = None, 
                 eval_sets: Optional[List[str]] = None, 
                 num_samples: Optional[int] = None,
                 custom_prompts: Optional[List[str]] = None,
                 logger=None) -> Dict[str, Any]:
    """
    Compare multiple models and generate a comparison report
    
    Args:
        model_paths: Dictionary mapping model names to model paths
        output_dir: Directory to save comparison results
        eval_sets: List of evaluation set names to use
        num_samples: Number of samples to use from each evaluation set
        custom_prompts: List of custom prompts to test on all models
        logger: Optional logger instance
        
    Returns:
        dict: Comparison results
    """
    # Create model comparison instance
    comparison = ModelComparison(logger=logger, output_dir=output_dir)
    
    # Add all models
    for model_name, model_path in model_paths.items():
        comparison.add_model(model_name, model_path)
    
    # Run comparison
    results = comparison.compare(
        eval_sets=eval_sets,
        num_samples=num_samples,
        prompts=custom_prompts
    )
    
    return results