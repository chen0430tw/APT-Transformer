#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reasoning Plugins for APT Console

Advanced reasoning techniques implemented as plugins:
- Self-Consistency: Multi-path sampling with majority voting
- Beam Search Reasoning: Beam search over reasoning paths
- Program-Aided Reasoning: Code generation and execution
- Tree-of-Thought: Tree-structured reasoning exploration
- Least-to-Most: Problem decomposition and progressive solving
"""

try:
    from apt.apps.console.plugins.reasoning.self_consistency_plugin import SelfConsistencyPlugin
except ImportError:
    pass
try:
    from apt.apps.console.plugins.reasoning.beam_search_plugin import BeamSearchReasoningPlugin
except ImportError:
    pass
try:
    from apt.apps.console.plugins.reasoning.program_aided_plugin import ProgramAidedReasoningPlugin
except ImportError:
    pass

__all__ = [
    'SelfConsistencyPlugin',
    'BeamSearchReasoningPlugin',
    'ProgramAidedReasoningPlugin',
]
