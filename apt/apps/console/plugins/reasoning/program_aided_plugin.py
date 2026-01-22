#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Program-Aided Reasoning Plugin

Implements program-aided reasoning (PAL/PoT) for precise computation.

Key idea: Generate executable Python code to solve problems, combining
neural reasoning with symbolic computation.

Reference:
    Gao et al., "PAL: Program-aided Language Models" (2022)
    Chen et al., "Program of Thoughts" (2022)

Features:
- Python code generation from natural language
- Sandboxed code execution
- Result validation and error handling
- Support for mathematical and logical reasoning
- Whitelist of safe functions
"""

import logging
import re
import ast
import sys
from io import StringIO
from typing import Dict, Any, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr

from apt.apps.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)

logger = logging.getLogger(__name__)


class ProgramAidedReasoningPlugin(PluginBase):
    """
    Program-Aided Reasoning Plugin

    Generates and executes Python code for precise problem solving.

    Workflow:
    1. Convert question to Python code generation prompt
    2. Generate Python code with model
    3. Validate code for safety
    4. Execute code in sandboxed environment
    5. Return execution result as answer

    Example:
        Question: "What is 15% of 80?"

        Generated code:
        ```python
        # Calculate 15% of 80
        percentage = 15
        number = 80
        result = (percentage / 100) * number
        print(result)
        ```

        Execution output: "12.0"
        Answer: 12.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Program-Aided Reasoning Plugin

        Args:
            config: Plugin configuration
        """
        super().__init__(config)

        # Code execution timeout (seconds)
        self.timeout = config.get('timeout', 5.0) if config else 5.0

        # Maximum code length
        self.max_code_length = config.get('max_code_length', 1000) if config else 1000

        # Whitelist of allowed modules/functions
        self.allowed_modules = config.get('allowed_modules', [
            'math', 'statistics', 'datetime', 'itertools', 'collections'
        ]) if config else ['math', 'statistics', 'datetime', 'itertools', 'collections']

        # Blacklist of dangerous functions
        self.forbidden_keywords = config.get('forbidden_keywords', [
            'import os', 'import sys', 'import subprocess', 'eval', 'exec',
            'open(', 'file(', '__import__', 'compile', 'globals', 'locals'
        ]) if config else [
            'import os', 'import sys', 'import subprocess', 'eval', 'exec',
            'open(', 'file(', '__import__', 'compile', 'globals', 'locals'
        ]

        # Code generation prompt template
        self.code_prompt_template = config.get('code_prompt_template',
            "# Question: {question}\n# Write Python code to solve this problem:\n"
        ) if config else "# Question: {question}\n# Write Python code to solve this problem:\n"

        # Metrics
        self.metrics = {
            'total_attempts': 0,
            'successful_executions': 0,
            'code_errors': 0,
            'validation_failures': 0,
            'timeouts': 0,
        }

    def get_manifest(self) -> PluginManifest:
        """
        Get plugin manifest

        Returns:
            Plugin manifest
        """
        return PluginManifest(
            name="program_aided_reasoning",
            version="1.0.0",
            description="Program-aided reasoning with code generation and sandboxed execution",
            author="APT Team",
            priority=PluginPriority.PROG_REASON,  # 320 (Reasoning tier)
            blocking=True,  # Need to execute code
            events=[
                PluginEvent.ON_INFERENCE_START,
                PluginEvent.ON_DECODE,
                PluginEvent.ON_STEP_END,
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
                "cpu_ms": 100.0,  # Code execution overhead
                "gpu_ms": 50.0,
                "io_mb": 1.0
            },
            rate_limit={
                "steps": 1
            },
            sandbox=True,  # Critical for code execution
            fail_limit=3,  # Lower tolerance for failures
            s_default=0.7,  # High utility for precise computation
            eta=1.4
        )

    def on_inference_start(self, context: Dict[str, Any]):
        """
        Inference start event handler

        Args:
            context: Event context
        """
        data = context.get('data', {})

        # Check if program-aided reasoning is enabled
        if data.get('use_program_aided', False):
            logger.debug(f"[Program-Aided] Enabled for current inference")
            self.set_context('enabled', True)
        else:
            self.set_context('enabled', False)

    def on_decode(self, context: Dict[str, Any]):
        """
        Decode event handler - main PAL logic

        Args:
            context: Event context
        """
        if not self.get_context('enabled', default=False):
            return

        step = context.get('step', 0)
        data = context.get('data', {})

        # Get model and question
        model = data.get('model')
        tokenizer = data.get('tokenizer')
        question = data.get('input_text') or data.get('question')

        if not all([model, tokenizer, question]):
            logger.warning(f"[Program-Aided] Missing required data at step {step}")
            return

        self.metrics['total_attempts'] += 1

        # Generate code
        code = self._generate_code(model, tokenizer, question)

        if not code:
            logger.warning(f"[Program-Aided] Failed to generate code")
            return

        self.set_context('generated_code', code)

        # Validate code for safety
        is_safe, validation_error = self._validate_code(code)

        if not is_safe:
            logger.warning(f"[Program-Aided] Code validation failed: {validation_error}")
            self.metrics['validation_failures'] += 1
            data['program_aided_result'] = {
                'success': False,
                'error': f'Validation failed: {validation_error}'
            }
            return

        # Execute code
        result, execution_error = self._execute_code(code)

        if execution_error:
            logger.warning(f"[Program-Aided] Code execution failed: {execution_error}")
            self.metrics['code_errors'] += 1
            data['program_aided_result'] = {
                'success': False,
                'error': f'Execution failed: {execution_error}',
                'code': code
            }
            return

        # Success
        self.metrics['successful_executions'] += 1
        self.set_context('result', result)

        # Write to public data
        data['program_aided_result'] = {
            'success': True,
            'result': result,
            'code': code
        }

        logger.info(f"[Program-Aided] Step {step}: Executed code successfully, result: {result}")

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

        data['metrics']['pal_success_rate'] = (
            self.metrics['successful_executions'] / self.metrics['total_attempts']
            if self.metrics['total_attempts'] > 0 else 0.0
        )
        data['metrics']['pal_attempts'] = self.metrics['total_attempts']

    def _generate_code(self, model, tokenizer, question: str) -> str:
        """
        Generate Python code from question

        Args:
            model: Language model
            tokenizer: Tokenizer
            question: Input question

        Returns:
            Generated Python code
        """
        # Create code generation prompt
        prompt = self.code_prompt_template.format(question=question)

        # In real implementation, this would call model.generate()
        # For now, return placeholder
        code = f"""# Generated code for: {question}
# Placeholder implementation
result = 42
print(result)
"""

        logger.debug(f"[Program-Aided] Generated code:\n{code}")
        return code

    def _validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code for safety

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check code length
        if len(code) > self.max_code_length:
            return False, f"Code exceeds maximum length ({self.max_code_length} chars)"

        # Check for forbidden keywords
        for keyword in self.forbidden_keywords:
            if keyword in code:
                return False, f"Forbidden keyword found: {keyword}"

        # Try to parse as valid Python
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        # Additional AST-based validation
        try:
            tree = ast.parse(code)
            validator = CodeValidator(self.allowed_modules)
            validator.visit(tree)
            if validator.errors:
                return False, "; ".join(validator.errors)
        except Exception as e:
            return False, f"AST validation error: {str(e)}"

        return True, None

    def _execute_code(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute code in sandboxed environment

        Args:
            code: Python code to execute

        Returns:
            Tuple of (output, error_message)
        """
        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # Limited globals (only safe modules)
        safe_globals = {'__builtins__': {}}

        # Add allowed modules
        for module_name in self.allowed_modules:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                pass

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals)

            # Get output
            output = stdout_capture.getvalue().strip()
            error_output = stderr_capture.getvalue().strip()

            if error_output:
                return None, error_output

            return output, None

        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get plugin statistics

        Returns:
            Statistics dictionary
        """
        return {
            'total_attempts': self.metrics['total_attempts'],
            'successful_executions': self.metrics['successful_executions'],
            'code_errors': self.metrics['code_errors'],
            'validation_failures': self.metrics['validation_failures'],
            'timeouts': self.metrics['timeouts'],
            'success_rate': (
                self.metrics['successful_executions'] / self.metrics['total_attempts']
                if self.metrics['total_attempts'] > 0 else 0.0
            ),
        }

    def cleanup(self):
        """Cleanup resources"""
        logger.info("[Program-Aided] Plugin cleanup")
        logger.info(f"[Program-Aided] Statistics: {self.get_statistics()}")


class CodeValidator(ast.NodeVisitor):
    """
    AST visitor to validate code safety

    Checks:
    - No imports of forbidden modules
    - No dangerous function calls
    - No attribute access to private/dunder methods
    """

    def __init__(self, allowed_modules: List[str]):
        """
        Initialize validator

        Args:
            allowed_modules: List of allowed module names
        """
        self.allowed_modules = set(allowed_modules)
        self.errors = []

    def visit_Import(self, node):
        """Check import statements"""
        for alias in node.names:
            if alias.name not in self.allowed_modules:
                self.errors.append(f"Import of forbidden module: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check from...import statements"""
        if node.module and node.module not in self.allowed_modules:
            self.errors.append(f"Import from forbidden module: {node.module}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Check attribute access"""
        if isinstance(node.attr, str) and node.attr.startswith('_'):
            self.errors.append(f"Access to private attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node):
        """Check function calls"""
        # Check for dangerous built-in functions
        if isinstance(node.func, ast.Name):
            dangerous_funcs = {'eval', 'exec', 'compile', '__import__', 'globals', 'locals'}
            if node.func.id in dangerous_funcs:
                self.errors.append(f"Call to forbidden function: {node.func.id}")
        self.generic_visit(node)
