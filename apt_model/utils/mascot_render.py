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
        # 加载图片
        image = Loader(mascot_path)

        # 创建 chafa 配置
        config = CanvasConfig()

        # 设置目标宽度和一个较大的初始高度
        config.width = cols
        config.height = cols * 3  # 设置较大的初始值，calc_canvas_geometry 会调整

        # 使用符号模式避免渲染黑块
        config.pixel_mode = PixelMode.CHAFA_PIXEL_MODE_SYMBOLS

        # 让 chafa 根据图片自动计算合适的 canvas 尺寸并缩放图片
        # font_ratio = 0.5 表示终端字符高度是宽度的2倍
        config.calc_canvas_geometry(
            image.width,
            image.height,
            0.5  # font_ratio
        )

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
        decoded_output = output.decode()
        # 在每一行末尾添加颜色重置，防止背景色溢出
        lines = decoded_output.split('\n')
        cleaned_lines = [line + '\033[0m' if line.strip() else line for line in lines]
        print_func('\n'.join(cleaned_lines))
        # 最后再次重置，确保完全清除
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
