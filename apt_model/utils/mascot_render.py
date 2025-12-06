#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Mascot Renderer (APT 吉祥物渲染器)

使用 chafa.py 在终端渲染兔子吉祥物
灵感来自 Linux Tux 企鹅启动画面
"""

import os
from typing import Optional

# 检查是否安装了 chafa.py
try:
    from chafa import Canvas, CanvasConfig, PixelMode
    from chafa.loader import Loader
    HAS_CHAFA = True
except (ImportError, FileNotFoundError, OSError, Exception):
    # ImportError: chafa.py 未安装
    # FileNotFoundError: Windows 上 ImageMagick 未安装
    # OSError: 其他系统级错误
    # Exception: 其他未预期的错误
    HAS_CHAFA = False


def print_apt_mascot(cols: int = 20, show_banner: bool = True, color_mode: bool = True, print_func=None):
    """
    打印 APT 兔子吉祥物（类似 Linux Tux 小巧 Logo）

    参数:
        cols: 显示宽度（字符数，默认20字符宽，类似Linux企鹅大小）
        show_banner: 是否显示横幅文字
        color_mode: 是否使用彩色模式（默认 True，chafa支持很好的彩色）
        print_func: 自定义输出函数（默认使用print，在logger环境中可传入info_print）

    设计理念:
        - 小巧简洁的 Logo，类似 Linux Tux 企鹅
        - 使用 chafa.py 库实现高质量终端渲染
        - 支持彩色和黑白两种模式
    """
    # 默认使用 print，除非指定了自定义函数
    if print_func is None:
        print_func = print

    # 显示横幅
    if show_banner:
        print_func("\n" + "="*70)
        print_func("  APT - Autopoietic Transformer | 自生成变换器")
        print_func("="*70)

    # 获取兔子图片路径
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mascot_path = os.path.join(script_dir, "docs", "assets", "兔兔伯爵.png")

    if not os.path.exists(mascot_path):
        # 如果找不到图片，显示简单的文字横幅
        if show_banner:
            print_func("  Training Session Starting... | 训练会话启动中...")
            print_func("="*70 + "\n")
        return

    # 检查 chafa.py 是否安装
    if not HAS_CHAFA:
        print_func("  🐰 提示: 安装以下依赖可以显示精美的兔子吉祥物:")
        print_func("     • Linux/Mac: pip install chafa.py")
        print_func("     • Windows: pip install chafa.py + 安装 ImageMagick")
        print_func("       (ImageMagick下载: https://imagemagick.org/script/download.php)")
        if show_banner:
            print_func("="*70)
            print_func("  Training Session Starting... | 训练会话启动中...")
            print_func("="*70 + "\n")
        return

    try:
        # 创建 chafa 配置
        config = CanvasConfig()

        # 加载图片
        image = Loader(mascot_path)

        # 手动设置尺寸以显示完整图片
        # 根据图片宽高比计算合适的终端字符数
        aspect_ratio = image.height / image.width  # 图片高/宽比
        config.width = cols
        # Windows PowerShell 字符宽高比约为 1.2:1，使用较小的除数
        config.height = int(cols * aspect_ratio * 0.8)

        # 使用符号模式避免渲染黑块
        config.pixel_mode = PixelMode.CHAFA_PIXEL_MODE_SYMBOLS

        # 创建画布
        canvas = Canvas(config)

        # 绘制所有像素
        canvas.draw_all_pixels(
            image.pixel_type,
            image.get_pixels(),
            image.width,
            image.height,
            image.rowstride
        )

        # 获取并打印输出
        output = canvas.print()
        print_func(output.decode())
        # 重置终端颜色，避免残留背景色
        print_func("\033[0m")

    except Exception as e:
        # 静默失败，不影响程序运行
        print_func(f"  (无法渲染吉祥物: {e})")

    if show_banner:
        print_func("="*70)
        print_func("  Training Session Starting... | 训练会话启动中...")
        print_func("="*70 + "\n")


if __name__ == "__main__":
    # 测试渲染（小巧 Logo，20 字符宽，类似 Linux 企鹅）
    print_apt_mascot(cols=20, show_banner=True, color_mode=True)
