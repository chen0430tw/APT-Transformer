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
        # 先用 PIL 加载并缩放图片
        from PIL import Image as PILImage

        pil_image = PILImage.open(mascot_path)

        # 计算缩放后的尺寸
        aspect_ratio = pil_image.height / pil_image.width
        target_width = cols * 4  # 每个字符对应约4个像素
        target_height = int(target_width * aspect_ratio)

        # 缩放图片
        pil_image = pil_image.resize((target_width, target_height), PILImage.Resampling.LANCZOS)

        # 转换为 RGB（如果是 RGBA）
        if pil_image.mode == 'RGBA':
            # 创建白色背景
            background = PILImage.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 获取像素数据
        pixel_data = pil_image.tobytes()

        # 创建 chafa 配置
        config = CanvasConfig()
        config.width = cols
        config.height = int(cols * aspect_ratio)
        config.pixel_mode = PixelMode.CHAFA_PIXEL_MODE_SYMBOLS

        # 【调试信息】
        print_func(f"[DEBUG] 原图: {PILImage.open(mascot_path).size}")
        print_func(f"[DEBUG] 缩放后: {pil_image.size}")
        print_func(f"[DEBUG] Canvas: {config.width}x{config.height}")

        # 创建画布并绘制
        canvas = Canvas(config)
        from chafa import PixelType
        canvas.draw_all_pixels(
            PixelType.CHAFA_PIXEL_RGB8,
            pixel_data,
            pil_image.width,
            pil_image.height,
            pil_image.width * 3  # rowstride for RGB
        )

        # 获取并打印输出
        output = canvas.print()
        decoded_output = output.decode()
        # 在每一行末尾添加颜色重置，防止背景色溢出
        lines = decoded_output.split('\n')

        # 【调试信息】打印输出统计
        print_func(f"[DEBUG] 输出行数: {len(lines)}")
        print_func(f"[DEBUG] 非空行数: {len([l for l in lines if l.strip()])}")
        print_func("=" * 70)

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
