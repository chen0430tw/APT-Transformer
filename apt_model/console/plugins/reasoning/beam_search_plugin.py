#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beam Search Reasoning Plugin

Implements beam search over reasoning paths for systematic exploration.

Key idea: Maintain k candidate reasoning paths, expand and score at each step,
and select the best path based on cumulative scores.

Features:
- Beam width control (k candidates)
- Path scoring with log probabilities
- Length normalization
- Early stopping when top paths converge
- Support for diverse beam search
"""

import logging
import torch
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)

logger = logging.getLogger(__name__)


@dataclass
class BeamPath:
    """
    Represents one path in the beam

    Attributes:
        tokens: List of token IDs in this path
        score: Cumulative log probability
        hidden_states: Hidden states at current position
        completed: Whether this path has reached end token
    """
    tokens: List[int]
    score: float
    hidden_states: Optional[torch.Tensor] = None
    completed: bool = False

    def normalized_score(self, alpha: float = 0.6) -> float:
        """
        Length-normalized score

        Args:
            alpha: Length penalty parameter (0.0 = no penalty, 1.0 = full penalty)

        Returns:
            Normalized score
        """
        length_penalty = ((5 + len(self.tokens)) / 6) ** alpha
        return self.score / length_penalty


class BeamSearchReasoningPlugin(PluginBase):
    """
    Beam Search Reasoning Plugin

    Maintains k candidate reasoning paths and selects the best one.

    Workflow:
    1. Start with k=beam_width initial candidates
    2. For each step:
       - Expand each candidate by generating next tokens
       - Score all expanded candidates
       - Keep top-k candidates
    3. Return highest-scoring completed path

    Example:
        Question: "What is the capital of France?"

        Beam width = 3

        Step 1:
          Path 1: "The capital" (score: -2.1)
          Path 2: "France's capital" (score: -2.3)
          Path 3: "Paris is" (score: -2.5)

        Step 2:
          Path 1: "The capital of France is Paris" (score: -4.2) ✓
          Path 2: "France's capital city is Paris" (score: -4.8)
          Path 3: "Paris is the capital" (score: -5.1)

        Selected: Path 1 (highest score)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Beam Search Reasoning Plugin

        Args:
            config: Plugin configuration
        """
        super().__init__(config)

        # Beam width (number of candidates)
        self.beam_width = config.get('beam_width', 4) if config else 4

        # Length penalty (alpha parameter)
        self.length_penalty = config.get('length_penalty', 0.6) if config else 0.6

        # Maximum reasoning steps
        self.max_steps = config.get('max_steps', 50) if config else 50

        # Diversity penalty (encourage diverse beams)
        self.diversity_penalty = config.get('diversity_penalty', 0.0) if config else 0.0

        # Early stopping when top beams converge
        self.early_stopping = config.get('early_stopping', True) if config else True

        # Metrics
        self.metrics = {
            'total_inferences': 0,
            'avg_beam_steps': 0.0,
            'avg_final_score': 0.0,
            'early_stops': 0,
        }

    def get_manifest(self) -> PluginManifest:
        """
        Get plugin manifest

        Returns:
            Plugin manifest
        """
        return PluginManifest(
            name="beam_search_reasoning",
            version="1.0.0",
            description="Beam search over reasoning paths with scoring and pruning",
            author="APT Team",
            priority=PluginPriority.BEAM_SEARCH,  # 300 (Reasoning tier)
            blocking=True,  # Need to complete beam search
            events=[
                PluginEvent.ON_INFERENCE_START,
                PluginEvent.ON_DECODE,
                PluginEvent.ON_STEP_END,
            ],
            requires=[
                "core:model",
            ],
            conflicts=[
                "plugin:self_consistency",  # Don't use both simultaneously
            ],
            capabilities=[
                PluginCapability.READ_STATE,
                PluginCapability.WRITE_STATE,
                PluginCapability.WRITE_METRICS,
            ],
            resources={
                "cpu_ms": 30.0 * 4,  # Beam width
                "gpu_ms": 80.0 * 4,
                "io_mb": 5.0
            },
            rate_limit={
                "steps": 1  # Run on every inference
            },
            sandbox=True,
            fail_limit=5,
            s_default=0.5,  # Medium utility
            eta=1.2
        )

    def on_inference_start(self, context: Dict[str, Any]):
        """
        Inference start event handler

        Args:
            context: Event context
        """
        data = context.get('data', {})

        # Check if beam search is enabled for this inference
        if data.get('use_beam_search', False):
            logger.debug(f"[Beam Search] Enabled with beam_width={self.beam_width}")
            self.set_context('enabled', True)
            self.set_context('beams', [])
        else:
            self.set_context('enabled', False)

    def on_decode(self, context: Dict[str, Any]):
        """
        Decode event handler - main beam search logic

        Args:
            context: Event context
        """
        if not self.get_context('enabled', default=False):
            return

        step = context.get('step', 0)
        data = context.get('data', {})

        # Get model and tokenizer
        model = data.get('model')
        tokenizer = data.get('tokenizer')
        input_ids = data.get('input_ids')

        if not all([model, tokenizer, input_ids]):
            logger.warning(f"[Beam Search] Missing required data at step {step}")
            return

        # Initialize beams on first step
        if step == 0:
            beams = self._initialize_beams(input_ids)
            self.set_context('beams', beams)

        # Perform beam search
        best_path, final_score, num_steps = self._beam_search(
            model=model,
            tokenizer=tokenizer,
            initial_beams=self.get_context('beams', default=[]),
            max_steps=self.max_steps
        )

        # Update metrics
        self.metrics['total_inferences'] += 1
        self.metrics['avg_beam_steps'] = (
            (self.metrics['avg_beam_steps'] * (self.metrics['total_inferences'] - 1) +
             num_steps) / self.metrics['total_inferences']
        )
        self.metrics['avg_final_score'] = (
            (self.metrics['avg_final_score'] * (self.metrics['total_inferences'] - 1) +
             final_score) / self.metrics['total_inferences']
        )

        # Store results
        self.set_context('best_path', best_path)
        self.set_context('final_score', final_score)
        self.set_context('num_steps', num_steps)

        # Write to public data
        data['beam_search_result'] = {
            'path': best_path.tokens if best_path else [],
            'score': final_score,
            'num_steps': num_steps,
            'beam_width': self.beam_width,
        }

        logger.info(f"[Beam Search] Completed in {num_steps} steps, "
                   f"score: {final_score:.4f}")

    def on_step_end(self, context: Dict[str, Any]):
        """
        Step end event handler

        Args:
            context: Event context
        """
        if not self.get_context('enabled', default=False):
            return

        data = context.get('data', {})

        # Write metrics
        if 'metrics' not in data:
            data['metrics'] = {}

        data['metrics']['beam_width'] = self.beam_width
        data['metrics']['beam_steps'] = self.get_context('num_steps', default=0)
        data['metrics']['beam_score'] = self.get_context('final_score', default=0.0)

    def _initialize_beams(self, input_ids: torch.Tensor) -> List[BeamPath]:
        """
        Initialize beam candidates

        Args:
            input_ids: Input token IDs

        Returns:
            List of initial beam paths
        """
        # Start with one beam containing the input
        initial_beam = BeamPath(
            tokens=input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids,
            score=0.0,
            hidden_states=None,
            completed=False
        )

        return [initial_beam]

    def _beam_search(
        self,
        model,
        tokenizer,
        initial_beams: List[BeamPath],
        max_steps: int
    ) -> Tuple[Optional[BeamPath], float, int]:
        """
        Perform beam search

        Args:
            model: Language model
            tokenizer: Tokenizer
            initial_beams: Initial beam candidates
            max_steps: Maximum steps

        Returns:
            Tuple of:
                - Best path
                - Final score
                - Number of steps taken
        """
        beams = initial_beams
        eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None

        for step in range(max_steps):
            # Expand all beams
            candidates = []

            for beam in beams:
                if beam.completed:
                    # Keep completed beams as-is
                    candidates.append(beam)
                    continue

                # Generate next tokens for this beam
                # In real implementation, this would use model.forward()
                # For now, create placeholder expansions
                next_tokens = self._get_top_k_tokens(beam, k=self.beam_width)

                for token_id, token_score in next_tokens:
                    new_beam = BeamPath(
                        tokens=beam.tokens + [token_id],
                        score=beam.score + token_score,
                        hidden_states=None,
                        completed=(token_id == eos_token_id)
                    )
                    candidates.append(new_beam)

            # Apply diversity penalty if enabled
            if self.diversity_penalty > 0:
                candidates = self._apply_diversity_penalty(candidates)

            # Select top-k candidates by normalized score
            candidates.sort(key=lambda b: b.normalized_score(self.length_penalty), reverse=True)
            beams = candidates[:self.beam_width]

            # Early stopping check
            if self.early_stopping and all(b.completed for b in beams):
                logger.debug(f"[Beam Search] Early stopping at step {step+1}")
                self.metrics['early_stops'] += 1
                break

            logger.debug(f"[Beam Search] Step {step+1}: {len(beams)} beams, "
                        f"best score: {beams[0].normalized_score(self.length_penalty):.4f}")

        # Return best beam
        if beams:
            best_beam = beams[0]
            return best_beam, best_beam.normalized_score(self.length_penalty), step + 1
        else:
            return None, 0.0, 0

    def _get_top_k_tokens(self, beam: BeamPath, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k next token candidates

        Args:
            beam: Current beam
            k: Number of candidates

        Returns:
            List of (token_id, log_prob) tuples
        """
        # Placeholder implementation
        # In real implementation, this would:
        # 1. Forward through model with beam.tokens
        # 2. Get logits for next token
        # 3. Convert to log probs
        # 4. Return top-k
        return [(i, -0.5) for i in range(k)]

    def _apply_diversity_penalty(self, beams: List[BeamPath]) -> List[BeamPath]:
        """
        Apply diversity penalty to encourage diverse beams

        Args:
            beams: Beam candidates

        Returns:
            Beams with diversity penalty applied
        """
        # Group beams by their most recent tokens
        token_groups = {}
        for beam in beams:
            if beam.tokens:
                last_token = beam.tokens[-1]
                if last_token not in token_groups:
                    token_groups[last_token] = []
                token_groups[last_token].append(beam)

        # Apply penalty to duplicate recent tokens
        for token, group in token_groups.items():
            if len(group) > 1:
                for i, beam in enumerate(group):
                    # Penalize duplicates (keep first occurrence)
                    if i > 0:
                        beam.score -= self.diversity_penalty

        return beams

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get plugin statistics

        Returns:
            Statistics dictionary
        """
        return {
            'total_inferences': self.metrics['total_inferences'],
            'avg_beam_steps': self.metrics['avg_beam_steps'],
            'avg_final_score': self.metrics['avg_final_score'],
            'early_stops': self.metrics['early_stops'],
            'early_stop_rate': (
                self.metrics['early_stops'] / self.metrics['total_inferences']
                if self.metrics['total_inferences'] > 0 else 0.0
            ),
        }

    def cleanup(self):
        """Cleanup resources"""
        logger.info("[Beam Search] Plugin cleanup")
        logger.info(f"[Beam Search] Statistics: {self.get_statistics()}")
