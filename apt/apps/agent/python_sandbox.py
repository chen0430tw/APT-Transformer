#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全的 Python 代码沙盒执行器

参考业界最佳实践：
- E2B (Firecracker microVM isolation)
- Modal Sandboxes (gVisor isolation)
- langchain-sandbox (Pyodide + Deno)
- RestrictedPython (AST-level restrictions)

安全机制：
1. RestrictedPython: AST 级别的语法限制
2. 资源限制: CPU、内存、时间限制
3. 禁止危险操作: 文件系统、网络、危险模块
4. 输出截断: 防止输出过大
5. 异常捕获: 安全的错误处理

技术来源：
- RestrictedPython: https://github.com/zopefoundation/RestrictedPython
- PyOdide: https://pyodide.org/
- E2B: https://e2b.dev/
"""

import re
import ast
import sys
import io
import time
import signal
import resource
import threading
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import logging

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

class SandboxConfig:
    """沙盒配置"""

    def __init__(
        self,
        # 资源限制
        max_execution_time: float = 5.0,  # 秒
        max_memory_mb: int = 100,          # MB
        max_output_size: int = 10000,      # 字符

        # 安全限制
        allow_imports: bool = False,       # 是否允许 import
        allowed_modules: Optional[list] = None,  # 允许的模块白名单
        allow_file_ops: bool = False,      # 是否允许文件操作
        allow_network: bool = False,       # 是否允许网络操作

        # 其他
        enable_builtins: bool = True,      # 是否启用内置函数
        restricted_builtins: Optional[list] = None,  # 受限的内置函数
    ):
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size

        self.allow_imports = allow_imports
        self.allowed_modules = allowed_modules or ["math", "random", "datetime", "json", "re"]
        self.allow_file_ops = allow_file_ops
        self.allow_network = allow_network

        self.enable_builtins = enable_builtins
        self.restricted_builtins = restricted_builtins or [
            "open", "exec", "eval", "compile", "__import__",
            "file", "input", "raw_input", "reload", "execfile"
        ]


# ==================== AST 检查器 ====================

class SecurityASTChecker(ast.NodeVisitor):
    """AST 安全检查器"""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.errors = []

    def check(self, code: str) -> Tuple[bool, list]:
        """
        检查代码安全性

        Returns:
            (is_safe, errors)
        """
        try:
            tree = ast.parse(code)
            self.visit(tree)
            return len(self.errors) == 0, self.errors
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

    def visit_Import(self, node):
        """检查 import 语句"""
        if not self.config.allow_imports:
            self.errors.append("Import statements are not allowed")
        else:
            # 检查模块白名单
            for alias in node.names:
                if alias.name not in self.config.allowed_modules:
                    self.errors.append(f"Module '{alias.name}' is not in allowed list")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """检查 from ... import 语句"""
        if not self.config.allow_imports:
            self.errors.append("Import statements are not allowed")
        elif node.module not in self.config.allowed_modules:
            self.errors.append(f"Module '{node.module}' is not in allowed list")
        self.generic_visit(node)

    def visit_Call(self, node):
        """检查函数调用"""
        # 检查危险的内置函数
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.config.restricted_builtins:
                self.errors.append(f"Function '{func_name}' is restricted")

        # 检查文件操作
        if not self.config.allow_file_ops and isinstance(node.func, ast.Name):
            if node.func.id in ["open", "file"]:
                self.errors.append("File operations are not allowed")

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """检查属性访问"""
        # 检查危险的属性（如 __import__, __builtins__）
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            self.errors.append(f"Access to special attribute '{node.attr}' is restricted")

        self.generic_visit(node)


# ==================== 超时处理 ====================

class TimeoutException(Exception):
    """超时异常"""
    pass


@contextmanager
def time_limit(seconds: float):
    """
    时间限制上下文管理器（仅 Unix）

    注意：在 Windows 上不可用，会退化为普通计时
    """
    if sys.platform == "win32":
        # Windows 不支持 signal.SIGALRM，使用线程计时
        timer = None

        def timeout_handler():
            raise TimeoutException(f"Execution timed out after {seconds} seconds")

        try:
            timer = threading.Timer(seconds, timeout_handler)
            timer.start()
            yield
        finally:
            if timer:
                timer.cancel()
    else:
        # Unix: 使用信号
        def signal_handler(signum, frame):
            raise TimeoutException(f"Execution timed out after {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)

        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


# ==================== 资源限制 ====================

@contextmanager
def resource_limit(max_memory_mb: int):
    """
    资源限制上下文管理器（仅 Unix）

    注意：在 Windows 上不可用
    """
    if sys.platform == "win32":
        # Windows 不支持 resource 模块
        yield
        return

    try:
        # 设置内存限制
        max_memory_bytes = max_memory_mb * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)

        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard))

        yield

    finally:
        # 恢复限制
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


# ==================== 安全的全局命名空间 ====================

def create_safe_globals(config: SandboxConfig) -> Dict[str, Any]:
    """
    创建安全的全局命名空间

    只包含安全的内置函数和模块
    """
    safe_builtins = {
        # 基础类型
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,

        # 常用函数
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'sorted': sorted,
        'reversed': reversed,
        'any': any,
        'all': all,

        # 类型检查
        'isinstance': isinstance,
        'type': type,

        # 字符串操作
        'print': print,
        'repr': repr,

        # 其他
        'True': True,
        'False': False,
        'None': None,
    }

    # 如果启用内置函数
    if config.enable_builtins:
        # 移除受限的内置函数
        for name in config.restricted_builtins:
            safe_builtins.pop(name, None)

    # 添加允许的模块
    if config.allow_imports:
        for module_name in config.allowed_modules:
            try:
                safe_builtins[module_name] = __import__(module_name)
            except ImportError:
                logger.warning(f"[Sandbox] Cannot import module: {module_name}")

    return {
        '__builtins__': safe_builtins,
        '__name__': '__sandbox__',
        '__doc__': None,
    }


# ==================== Python 沙盒 ====================

class PythonSandbox:
    """
    安全的 Python 代码沙盒

    多层安全机制：
    1. AST 静态检查
    2. 受限的全局命名空间
    3. 时间和内存限制
    4. 输出截断
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.checker = SecurityASTChecker(self.config)

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        执行 Python 代码

        Args:
            code: Python 代码字符串
            context: 额外的上下文变量

        Returns:
            (success, result, error)
        """
        # 1. 静态安全检查
        is_safe, errors = self.checker.check(code)
        if not is_safe:
            error_msg = "Security check failed:\n" + "\n".join(f"  - {e}" for e in errors)
            return False, None, error_msg

        # 2. 创建安全的全局命名空间
        safe_globals = create_safe_globals(self.config)

        # 添加上下文变量
        if context:
            safe_globals.update(context)

        # 3. 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                with time_limit(self.config.max_execution_time):
                    with resource_limit(self.config.max_memory_mb):
                        # 执行代码
                        exec_globals = safe_globals.copy()
                        exec(code, exec_globals)

                        # 尝试获取返回值（查找最后一个表达式）
                        result = exec_globals.get('__result__', None)

            # 获取输出
            stdout_str = stdout_capture.getvalue()
            stderr_str = stderr_capture.getvalue()

            # 截断输出
            if len(stdout_str) > self.config.max_output_size:
                stdout_str = stdout_str[:self.config.max_output_size] + "\n... (output truncated)"

            # 合并输出和结果
            if stdout_str or stderr_str:
                result = {
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "result": result
                }

            return True, result, None

        except TimeoutException as e:
            return False, None, str(e)

        except MemoryError:
            return False, None, f"Memory limit exceeded ({self.config.max_memory_mb} MB)"

        except Exception as e:
            # 捕获所有其他异常
            error = f"{type(e).__name__}: {str(e)}"
            return False, None, error

        finally:
            stdout_capture.close()
            stderr_capture.close()

    def execute_expression(self, expression: str) -> Tuple[bool, Any, Optional[str]]:
        """
        执行单个表达式（更安全）

        Args:
            expression: Python 表达式

        Returns:
            (success, result, error)
        """
        # 包装为赋值语句
        code = f"__result__ = {expression}"
        return self.execute(code)


# ==================== 工具集成 ====================

from .tool_system import Tool, ToolDefinition, ToolParameter, ToolType


class PythonCodeExecutorTool(Tool):
    """Python 代码执行工具（集成到工具系统）"""

    def __init__(
        self,
        name: str = "python_executor",
        config: Optional[SandboxConfig] = None
    ):
        # 工具定义
        definition = ToolDefinition(
            name=name,
            description="Execute Python code safely in a sandboxed environment",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                    required=True
                ),
                ToolParameter(
                    name="mode",
                    type="string",
                    description="Execution mode: 'script' or 'expression'",
                    required=False,
                    default="script",
                    enum=["script", "expression"]
                )
            ],
            tool_type=ToolType.COMPUTATION,
            timeout=10.0,
            allow_parallel=True,
            cacheable=False  # 代码执行不缓存
        )

        super().__init__(definition)

        self.sandbox = PythonSandbox(config)

    async def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """执行 Python 代码"""
        code = arguments["code"]
        mode = arguments.get("mode", "script")

        if mode == "expression":
            success, result, error = self.sandbox.execute_expression(code)
        else:
            success, result, error = self.sandbox.execute(code, context)

        if success:
            return {
                "success": True,
                "result": result,
                "mode": mode
            }
        else:
            return {
                "success": False,
                "error": error,
                "mode": mode
            }


# ==================== 示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Python Sandbox Demo ===\n")

    # 创建沙盒
    sandbox = PythonSandbox()

    # 测试案例
    test_cases = [
        ("计算表达式", "result = 3 + 5 * 2\nprint(f'Result: {result}')\n__result__ = result"),
        ("列表操作", "data = [1, 2, 3, 4, 5]\nsum_val = sum(data)\n__result__ = sum_val"),
        ("字符串处理", "text = 'Hello, World!'\n__result__ = text.upper()"),
        ("数学计算", "import math\n__result__ = math.sqrt(16)"),  # 如果允许 import
        ("危险操作 - 文件", "open('/etc/passwd', 'r')"),  # 应该被拒绝
        ("危险操作 - eval", "eval('1+1')"),  # 应该被拒绝
    ]

    for i, (desc, code) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {desc}")
        print(f"{'='*60}")
        print(f"Code:\n{code}\n")

        success, result, error = sandbox.execute(code)

        if success:
            print(f"✓ Success")
            print(f"Result: {result}")
        else:
            print(f"✗ Failed")
            print(f"Error: {error}")

    # 测试超时
    print(f"\n{'='*60}")
    print("Test: Timeout")
    print(f"{'='*60}")

    timeout_code = """
import time
time.sleep(10)  # 应该超时
"""

    success, result, error = sandbox.execute(timeout_code)
    print(f"Success: {success}")
    print(f"Error: {error}")

    print("\n=== Demo Complete ===")
