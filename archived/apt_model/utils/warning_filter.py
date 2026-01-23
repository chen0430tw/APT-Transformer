"""
警告过滤器 - 美化错误和警告消息

功能:
- 过滤第三方库的无关警告
- 简化文件路径显示
- 美化用户可读的警告消息
"""

import warnings
import sys
import os
from pathlib import Path


class CleanWarningFormatter:
    """美化的警告格式化器"""

    @staticmethod
    def format_path(filename):
        """简化路径显示"""
        try:
            # 尝试转换为相对于项目根目录的路径
            project_root = Path.cwd()
            file_path = Path(filename)

            try:
                rel_path = file_path.relative_to(project_root)
                return str(rel_path)
            except ValueError:
                # 如果不在项目目录下，只显示文件名
                return file_path.name
        except:
            return filename


def show_clean_warning(message, category, filename, lineno, file=None, line=None):
    """
    清洁的警告显示函数

    格式: ⚠️ [类别] 消息 (文件:行号)
    """
    # 简化路径
    clean_path = CleanWarningFormatter.format_path(filename)

    # 获取警告类别名称（去掉 Warning 后缀）
    category_name = category.__name__.replace('Warning', '')

    # 构建清洁的警告消息
    if 'apt_model' in filename or 'APT-Transformer' in filename:
        # 自己的代码，显示详细信息
        msg = f"⚠️  [{category_name}] {message}\n"
        msg += f"   位置: {clean_path}:{lineno}\n"
    else:
        # 第三方库，简化显示
        msg = f"⚠️  {message}\n"

    # 写入stderr
    if file is None:
        file = sys.stderr

    try:
        file.write(msg)
    except OSError:
        pass


def setup_warning_filter():
    """
    配置警告过滤器

    过滤规则:
    1. 忽略PyTorch/CUDA的常见警告
    2. 忽略第三方库的DeprecationWarning
    3. 保留APT项目自己的所有警告
    """
    # 设置自定义警告格式
    warnings.showwarning = show_clean_warning

    # 过滤第三方库的警告
    # PyTorch CUDA 警告
    warnings.filterwarnings('ignore', category=UserWarning,
                          message='.*CUDA is not available.*')
    warnings.filterwarnings('ignore', category=UserWarning,
                          message='.*pin_memory.*')
    warnings.filterwarnings('ignore', category=FutureWarning,
                          message='.*torch.cuda.amp.GradScaler.*')

    # 第三方库的 DeprecationWarning
    warnings.filterwarnings('ignore', category=DeprecationWarning,
                          module='.*site-packages.*')

    # 保留APT项目的所有警告（总是显示）
    warnings.filterwarnings('default', module='apt_model.*')
    warnings.filterwarnings('default', module='apt.*')


def filter_third_party_warnings():
    """
    更激进的过滤 - 隐藏所有第三方警告

    使用场景：生产环境或演示时
    """
    # 只显示项目自己的警告
    warnings.filterwarnings('ignore')  # 先忽略所有
    warnings.filterwarnings('default', module='apt_model.*')  # 恢复APT的
    warnings.filterwarnings('default', module='apt.*')


def quiet_mode():
    """
    安静模式 - 只显示错误，不显示警告

    使用场景：脚本运行、自动化测试
    """
    warnings.simplefilter('ignore')


# 便捷函数
def enable_clean_warnings():
    """启用清洁警告模式（推荐）"""
    setup_warning_filter()


def enable_quiet_mode():
    """启用安静模式（无警告）"""
    quiet_mode()


def enable_verbose_mode():
    """启用详细模式（显示所有警告）"""
    warnings.simplefilter('default')
    warnings.showwarning = show_clean_warning


# 自动初始化（在导入时）
# 检查环境变量
warning_mode = os.getenv('APT_WARNING_MODE', 'clean')

if warning_mode == 'clean':
    setup_warning_filter()
elif warning_mode == 'quiet':
    quiet_mode()
elif warning_mode == 'verbose':
    enable_verbose_mode()
elif warning_mode == 'none':
    pass  # 不做任何过滤


if __name__ == '__main__':
    # 测试
    print("测试警告过滤器...\n")

    enable_clean_warnings()

    # 测试各种警告
    warnings.warn("这是一个用户警告", UserWarning)
    warnings.warn("这是一个Future警告", FutureWarning)
    warnings.warn("这是一个Deprecation警告", DeprecationWarning)

    print("\n✅ 警告过滤器测试完成")
