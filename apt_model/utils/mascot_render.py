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
except ImportError:
    HAS_CHAFA = False


def print_apt_mascot(cols: int = 20, show_banner: bool = True, color_mode: bool = True):
    """
    打印 APT 兔子吉祥物（类似 Linux Tux 小巧 Logo）

    参数:
        cols: 显示宽度（字符数，默认20字符宽，类似Linux企鹅大小）
        show_banner: 是否显示横幅文字
        color_mode: 是否使用彩色模式（默认 True，chafa支持很好的彩色）

    设计理念:
        - 小巧简洁的 Logo，类似 Linux Tux 企鹅
        - 使用 chafa.py 库实现高质量终端渲染
        - 支持彩色和黑白两种模式
    """
    # 显示横幅
    if show_banner:
        print("\n" + "="*70)
        print("  APT - Autopoietic Transformer | 自生成变换器")
        print("="*70)

    # 获取兔子图片路径
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mascot_path = os.path.join(script_dir, "docs", "assets", "兔兔伯爵.png")

    if not os.path.exists(mascot_path):
        # 如果找不到图片，显示简单的文字横幅
        if show_banner:
            print("  Training Session Starting... | 训练会话启动中...")
            print("="*70 + "\n")
        return

    # 检查 chafa.py 是否安装
    if not HAS_CHAFA:
        print("  提示: 安装 chafa.py 可以显示精美的吉祥物图案")
        print("  pip install chafa.py")
        if show_banner:
            print("="*70)
            print("  Training Session Starting... | 训练会话启动中...")
            print("="*70 + "\n")
        return

    try:
        # 终端字体宽高比（通常终端字符高度是宽度的2倍左右）
        FONT_RATIO = 0.5  # width/height

        # 创建 chafa 配置
        config = CanvasConfig()

        # 设置输出宽度
        config.width = cols
        config.height = cols  # 先设置一个初始值，后面会自动计算

        # 加载图片
        image = Loader(mascot_path)

        # 根据图片比例和字体比例自动计算合适的高度
        config.calc_canvas_geometry(
            image.width,
            image.height,
            FONT_RATIO
        )

        # 如果启用彩色模式
        if color_mode:
            # 使用全彩色模式
            config.set_color_space(2)  # CHAFA_COLOR_SPACE_RGB
        else:
            # 使用单色模式
            config.set_color_space(0)  # CHAFA_COLOR_SPACE_NONE

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
        print(output.decode())

    except Exception as e:
        # 静默失败，不影响程序运行
        print(f"  (无法渲染吉祥物: {e})")

    if show_banner:
        print("="*70)
        print("  Training Session Starting... | 训练会话启动中...")
        print("="*70 + "\n")


if __name__ == "__main__":
    # 测试渲染（小巧 Logo，20 字符宽，类似 Linux 企鹅）
    print_apt_mascot(cols=20, show_banner=True, color_mode=True)
