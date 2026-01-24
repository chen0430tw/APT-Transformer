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

try:
    from apt.core.runtime.decoder.halting import (
        HaltingUnit,
        MultiCriteriaHalting,
        BudgetedHalting,
    )
except ImportError:
    HaltingUnit = None
    MultiCriteriaHalting = None
    BudgetedHalting = None

try:
    from apt.core.runtime.decoder.routing import (
        ExpertRouter,
        MiniExpert,
        MoELayer,
        SwitchRouter,
    )
except ImportError:
    ExpertRouter = None
    MiniExpert = None
    MoELayer = None
    SwitchRouter = None

try:
    from apt.core.runtime.decoder.structured_reasoner import (
        StructuredReasoner,
        ChainOfThoughtReasoner,
        SelfConsistencyReasoner,
        TreeOfThoughtsReasoner,
    )
except ImportError:
    StructuredReasoner = None
    ChainOfThoughtReasoner = None
    SelfConsistencyReasoner = None
    TreeOfThoughtsReasoner = None

try:
    from apt.core.runtime.decoder.reasoning_controller import (
        ReasoningController,
        BudgetedReasoningController,
        AdaptiveBudgetController,
    )
except ImportError:
    ReasoningController = None
    BudgetedReasoningController = None
    AdaptiveBudgetController = None

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
