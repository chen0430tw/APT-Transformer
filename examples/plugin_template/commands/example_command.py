"""
Example CLI Command

展示如何为插件创建CLI命令
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExampleCommand:
    """
    示例CLI命令类

    插件可以提供CLI命令来扩展APT-CLI的功能
    """

    def __init__(self):
        """初始化命令"""
        self.name = "example-cmd"

    def execute(self, *args, **kwargs) -> Any:
        """
        执行命令

        这是主要的执行入口点

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            命令执行结果

        Example:
            >>> cmd = ExampleCommand()
            >>> result = cmd.execute(message="Hello")
            >>> print(result)
            {'status': 'success', 'message': 'Hello from example command!'}
        """
        logger.info("Example command executed")

        # 获取参数
        message = kwargs.get('message', 'Hello')
        verbose = kwargs.get('verbose', False)

        if verbose:
            logger.info(f"Message: {message}")
            logger.info(f"Args: {args}")
            logger.info(f"Kwargs: {kwargs}")

        # 执行命令逻辑
        result = self._do_work(message)

        return result

    def _do_work(self, message: str) -> dict:
        """
        执行实际工作

        Args:
            message: 消息文本

        Returns:
            结果字典
        """
        output = f"{message} from example command!"

        return {
            'status': 'success',
            'message': output,
            'timestamp': self._get_timestamp(),
        }

    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()


# 也可以提供函数式命令
def example_function_command(message: str = "Hello") -> dict:
    """
    函数式命令示例

    Args:
        message: 消息文本

    Returns:
        结果字典
    """
    return {
        'status': 'success',
        'message': f"{message} from function command!",
    }
