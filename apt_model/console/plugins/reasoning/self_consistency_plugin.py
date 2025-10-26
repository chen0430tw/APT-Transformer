#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-Consistency Reasoning Plugin

Implements self-consistency decoding for improved reasoning reliability.

Key idea: Generate multiple diverse reasoning paths, then select the most
consistent answer through majority voting.

Reference:
    Wang et al., "Self-Consistency Improves Chain of Thought Reasoning
    in Language Models" (2022)

Features:
- Multiple independent reasoning paths with temperature sampling
- Automatic answer extraction from reasoning chains
- Majority voting for answer selection
- Confidence scoring based on vote distribution
- Support for both greedy and sampled generation
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)

logger = logging.getLogger(__name__)


class SelfConsistencyPlugin(PluginBase):
    """
    Self-Consistency Reasoning Plugin

    Generates multiple reasoning paths and selects the most consistent answer.

    Workflow:
    1. Generate N diverse reasoning paths (temperature sampling)
    2. Extract answer from each path
    3. Vote on answers using majority rule
    4. Return most consistent answer with confidence score

    Example:
        Question: "What is 15% of 80?"

        Path 1 (temp=0.7):
            "15% = 0.15
             0.15 × 80 = 12
             Answer: 12"

        Path 2 (temp=0.8):
            "We need to calculate 15/100 × 80
             = 0.15 × 80
             = 12
             Answer: 12"

        Path 3 (temp=0.9):
            "15% of 80 = (15/100) × 80
             = 12
             Answer: 12"

        Vote: {12: 3}
        Selected: 12 (confidence: 100%)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Self-Consistency Plugin

        Args:
            config: Plugin configuration
        """
        super().__init__(config)

        # Number of reasoning paths to generate
        self.num_paths = config.get('num_paths', 5) if config else 5

        # Temperature for diverse sampling
        self.temperature = config.get('temperature', 0.7) if config else 0.7

        # Answer extraction patterns
        self.answer_patterns = config.get('answer_patterns', [
            r'[Aa]nswer:\s*(.+)',
            r'答案[:：]\s*(.+)',
            r'最终答案[:：]\s*(.+)',
            r'因此[:：]\s*(.+)',
            r'所以[:：]\s*(.+)',
        ]) if config else [
            r'[Aa]nswer:\s*(.+)',
            r'答案[:：]\s*(.+)',
            r'最终答案[:：]\s*(.+)',
            r'因此[:：]\s*(.+)',
            r'所以[:：]\s*(.+)',
        ]

        # Metrics
        self.metrics = {
            'total_inferences': 0,
            'total_paths_generated': 0,
            'avg_confidence': 0.0,
            'unanimous_votes': 0,  # All paths agree
        }

    def get_manifest(self) -> PluginManifest:
        """
        Get plugin manifest

        Returns:
            Plugin manifest
        """
        return PluginManifest(
            name="self_consistency",
            version="1.0.0",
            description="Self-consistency reasoning with multi-path sampling and majority voting",
            author="APT Team",
            priority=PluginPriority.SC_DECODE,  # 280 (Reasoning tier)
            blocking=True,  # Need to wait for all paths
            events=[
                PluginEvent.ON_INFERENCE_START,
                PluginEvent.ON_DECODE,
                PluginEvent.ON_STEP_EVAL,
            ],
            requires=[
                "core:model",
            ],
            conflicts=[],
            capabilities=[
                PluginCapability.READ_STATE,
                PluginCapability.WRITE_METRICS,
            ],
            resources={
                "cpu_ms": 50.0 * 5,  # Multiple paths
                "gpu_ms": 100.0 * 5,
                "io_mb": 2.0
            },
            rate_limit={
                "steps": 1  # Run on every inference
            },
            sandbox=True,
            fail_limit=5,
            s_default=0.6,  # High utility for reasoning improvement
            eta=1.3
        )

    def on_inference_start(self, context: Dict[str, Any]):
        """
        Inference start event handler

        Args:
            context: Event context
        """
        data = context.get('data', {})

        # Check if self-consistency is enabled for this inference
        if data.get('use_self_consistency', False):
            logger.debug(f"[Self-Consistency] Enabled for current inference")
            self.set_context('enabled', True)
        else:
            self.set_context('enabled', False)

    def on_decode(self, context: Dict[str, Any]):
        """
        Decode event handler - main logic

        Args:
            context: Event context
        """
        if not self.get_context('enabled', default=False):
            return

        step = context.get('step', 0)
        data = context.get('data', {})

        # Get model and tokenizer from context
        model = data.get('model')
        tokenizer = data.get('tokenizer')
        input_text = data.get('input_text')

        if not all([model, tokenizer, input_text]):
            logger.warning(f"[Self-Consistency] Missing required data at step {step}")
            return

        # Generate multiple reasoning paths
        paths = self._generate_paths(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            num_paths=self.num_paths,
            temperature=self.temperature
        )

        # Extract answers from each path
        answers = [self._extract_answer(path) for path in paths]

        # Vote on answers
        answer, confidence, vote_distribution = self._vote_on_answers(answers)

        # Update metrics
        self.metrics['total_inferences'] += 1
        self.metrics['total_paths_generated'] += len(paths)
        self.metrics['avg_confidence'] = (
            (self.metrics['avg_confidence'] * (self.metrics['total_inferences'] - 1) +
             confidence) / self.metrics['total_inferences']
        )

        if confidence == 1.0:
            self.metrics['unanimous_votes'] += 1

        # Store results in context
        self.set_context('paths', paths)
        self.set_context('answers', answers)
        self.set_context('selected_answer', answer)
        self.set_context('confidence', confidence)
        self.set_context('vote_distribution', vote_distribution)

        # Write to public data
        data['self_consistency_result'] = {
            'answer': answer,
            'confidence': confidence,
            'paths_count': len(paths),
            'vote_distribution': vote_distribution,
        }

        logger.info(f"[Self-Consistency] Step {step}: Selected answer '{answer}' "
                   f"with confidence {confidence:.2%}")

    def on_step_eval(self, context: Dict[str, Any]):
        """
        Step evaluation event handler

        Args:
            context: Event context
        """
        if not self.get_context('enabled', default=False):
            return

        step = context.get('step', 0)
        data = context.get('data', {})

        # Write metrics
        if 'metrics' not in data:
            data['metrics'] = {}

        data['metrics']['sc_confidence'] = self.get_context('confidence', default=0.0)
        data['metrics']['sc_paths'] = self.num_paths
        data['metrics']['sc_avg_confidence'] = self.metrics['avg_confidence']

    def _generate_paths(
        self,
        model,
        tokenizer,
        input_text: str,
        num_paths: int,
        temperature: float
    ) -> List[str]:
        """
        Generate multiple diverse reasoning paths

        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Input question/prompt
            num_paths: Number of paths to generate
            temperature: Sampling temperature

        Returns:
            List of generated reasoning paths
        """
        paths = []

        for i in range(num_paths):
            # In real implementation, this would call model.generate()
            # with temperature sampling
            # For now, return placeholder
            path = f"Path {i+1}: [Generated reasoning with temperature={temperature}]"
            paths.append(path)

            logger.debug(f"[Self-Consistency] Generated path {i+1}/{num_paths}")

        return paths

    def _extract_answer(self, reasoning_path: str) -> str:
        """
        Extract final answer from reasoning path

        Args:
            reasoning_path: Complete reasoning chain

        Returns:
            Extracted answer string
        """
        # Try each pattern
        for pattern in self.answer_patterns:
            match = re.search(pattern, reasoning_path, re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                return self._normalize_answer(answer)

        # If no pattern matches, use last non-empty line
        lines = [line.strip() for line in reasoning_path.split('\n') if line.strip()]
        if lines:
            return self._normalize_answer(lines[-1])

        return ""

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer
        """
        # Remove punctuation
        answer = re.sub(r'[,.!?;。，！？；]', '', answer)

        # Lowercase
        answer = answer.lower().strip()

        # Remove extra whitespace
        answer = ' '.join(answer.split())

        return answer

    def _vote_on_answers(self, answers: List[str]) -> Tuple[str, float, Dict[str, int]]:
        """
        Vote on answers using majority rule

        Args:
            answers: List of extracted answers

        Returns:
            Tuple of:
                - Selected answer (most common)
                - Confidence (fraction of votes)
                - Vote distribution (dict of answer -> count)
        """
        # Count votes
        vote_counts = Counter(answers)

        # Get most common answer
        if not vote_counts:
            return "", 0.0, {}

        most_common = vote_counts.most_common(1)[0]
        selected_answer = most_common[0]
        vote_count = most_common[1]

        # Calculate confidence
        total_votes = len(answers)
        confidence = vote_count / total_votes if total_votes > 0 else 0.0

        # Convert to regular dict
        vote_distribution = dict(vote_counts)

        return selected_answer, confidence, vote_distribution

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get plugin statistics

        Returns:
            Statistics dictionary
        """
        return {
            'total_inferences': self.metrics['total_inferences'],
            'total_paths_generated': self.metrics['total_paths_generated'],
            'avg_confidence': self.metrics['avg_confidence'],
            'unanimous_votes': self.metrics['unanimous_votes'],
            'unanimous_rate': (
                self.metrics['unanimous_votes'] / self.metrics['total_inferences']
                if self.metrics['total_inferences'] > 0 else 0.0
            ),
        }

    def cleanup(self):
        """Cleanup resources"""
        logger.info("[Self-Consistency] Plugin cleanup")
        logger.info(f"[Self-Consistency] Statistics: {self.get_statistics()}")
