#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runtime Decoder Module

Provides runtime decoding strategies including:
- Structured reasoning (o3-style)
- Chain-of-Thought (CoT)
- Self-consistency reasoning
- Tree-of-Thoughts (ToT)
- Adaptive halting mechanisms
- Expert routing in vein subspace
"""

from apt.core.runtime.decoder.halting import (
    HaltingUnit,
    MultiCriteriaHalting,
    BudgetedHalting,
)

from apt.core.runtime.decoder.routing import (
    ExpertRouter,
    MiniExpert,
    MoELayer,
    SwitchRouter,
)

from apt.core.runtime.decoder.structured_reasoner import (
    StructuredReasoner,
    ChainOfThoughtReasoner,
    SelfConsistencyReasoner,
    TreeOfThoughtsReasoner,
)

from apt.core.runtime.decoder.reasoning_controller import (
    ReasoningController,
    BudgetedReasoningController,
    AdaptiveBudgetController,
)

__all__ = [
    # Halting
    'HaltingUnit',
    'MultiCriteriaHalting',
    'BudgetedHalting',

    # Routing
    'ExpertRouter',
    'MiniExpert',
    'MoELayer',
    'SwitchRouter',

    # Reasoners
    'StructuredReasoner',
    'ChainOfThoughtReasoner',
    'SelfConsistencyReasoner',
    'TreeOfThoughtsReasoner',

    # Controllers
    'ReasoningController',
    'BudgetedReasoningController',
    'AdaptiveBudgetController',
]
