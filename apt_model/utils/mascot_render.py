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
    try:
        from chafa import TermDb, TermInfo
        HAS_TERMDB = True
    except ImportError:
        HAS_TERMDB = False
    HAS_CHAFA = True
except (ImportError, FileNotFoundError, OSError, Exception):
    # ImportError: chafa.py 未安装
    # FileNotFoundError: Windows 上 ImageMagick 未安装
    # OSError: 其他系统级错误
    # Exception: 其他未预期的错误
    HAS_CHAFA = False
    HAS_TERMDB = False


def print_apt_mascot(cols: int = 35, show_banner: bool = True, color_mode: bool = True, print_func=None, use_sixel: bool = False):
    """
    打印 APT 兔子吉祥物（类似 Linux Tux 小巧 Logo）

    参数:
        cols: 显示宽度（字符数或像素列数，默认35）
        show_banner: 是否显示横幅文字
        color_mode: 是否使用彩色模式（默认 True，chafa支持很好的彩色）
        print_func: 自定义输出函数（默认使用print，在logger环境中可传入info_print）
        use_sixel: 是否使用 Sixel 图形模式（默认 False，使用字符艺术）
                   Sixel 模式可显示完美像素图片（高清像素渲染）
                   支持终端：Windows Terminal (v1.22+), WezTerm, mintty, Konsole, iTerm2等

    设计理念:
        - 小巧简洁的 Logo，类似 Linux Tux 企鹅
        - 使用 chafa.py 库实现高质量终端渲染
        - 支持字符模式（symbols）和像素模式（sixel）
        - 自动计算高度以保持图片比例
    """
    # 默认使用 print，除非指定了自定义函数
    if print_func is None:
        print_func = print

    # Sixel 模式：使用系统 chafa 命令（更可靠）
    if use_sixel:
        import subprocess
        import shutil

        # 获取兔子图片路径
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        mascot_path = os.path.join(script_dir, "docs", "assets", "兔兔伯爵.png")

        if not os.path.exists(mascot_path):
            print_func("  (找不到吉祥物图片)")
            return

        # 检查系统是否有 chafa 命令
        chafa_cmd = shutil.which("chafa")
        if chafa_cmd:
            if show_banner:
                print_func("\n" + "="*70)
                print_func("  APT - Autopoietic Transformer | 自生成变换器")
                print_func("="*70)

            try:
                # 使用系统 chafa 命令渲染 Sixel
                result = subprocess.run(
                    [chafa_cmd, "-f", "sixels", "-s", f"{cols}x", mascot_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print_func(result.stdout)
            except subprocess.CalledProcessError as e:
                print_func(f"  (Sixel 渲染失败: {e})")

            if show_banner:
                print_func("="*70)
                print_func("  Training Session Starting... | 训练会话启动中...")
                print_func("="*70 + "\n")
            return
        else:
            print_func("  (未找到 chafa 命令，切换到字符模式)")
            use_sixel = False  # 回退到字符模式

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
        # 使用 chafa 的原生 Loader 加载图片
        # Loader 会正确处理图片并提供与 draw_all_pixels 兼容的像素数据
        image = Loader(mascot_path)

        # 创建 chafa 配置
        config = CanvasConfig()

        if use_sixel:
            # Sixel 模式：显示完美像素图片
            # 根据图片比例计算高度（像素级）
            calculated_height = int(cols * (image.height / image.width))

            config.width = cols
            config.height = calculated_height
            config.pixel_mode = PixelMode.CHAFA_PIXEL_MODE_SIXELS
            print_func(f"[DEBUG] Sixel 模式: {cols}x{calculated_height} px")
        else:
            # 字符艺术模式
            # 终端字符的纵横比（字符宽度/高度），通常字符高度是宽度的2倍
            FONT_RATIO = 0.5  # width/height

            # 手动计算合适的高度（保持图片比例 × 字符纵横比）
            # height = width * (图片高度/图片宽度) * (字符宽度/字符高度)
            calculated_height = int(cols * (image.height / image.width) * FONT_RATIO)

            config.width = cols
            config.height = calculated_height
            config.pixel_mode = PixelMode.CHAFA_PIXEL_MODE_SYMBOLS

        # 【调试信息】
        print_func(f"[DEBUG] 原图尺寸: {image.width}x{image.height}")
        print_func(f"[DEBUG] Canvas: {config.width}x{config.height}")
        print_func(f"[DEBUG] 像素类型: {image.pixel_type}")
        print_func(f"[DEBUG] Rowstride: {image.rowstride}")

        # 创建画布并绘制
        canvas = Canvas(config)

        # 使用 Loader 提供的原生属性绘制
        # chafa 会自动将原图下采样到 canvas 尺寸
        canvas.draw_all_pixels(
            image.pixel_type,
            image.get_pixels(),  # 使用 get_pixels() 方法获取像素数据
            image.width,
            image.height,
            image.rowstride
        )

        # 获取并打印输出
        # Sixel 模式需要 fallback 参数告诉 chafa 终端支持 Sixel
        if use_sixel and HAS_TERMDB:
            # 创建支持 Sixel 的终端信息
            term_db = TermDb()
            term_info = term_db.detect()
            print_func(f"[DEBUG] 检测到终端: {term_info}")
            output = canvas.print(term_info=term_info)
        elif use_sixel:
            # 没有 TermDb，使用 fallback（假设终端支持 Sixel）
            print_func("[DEBUG] 使用 fallback 模式（假设终端支持 Sixel）")
            # 直接输出，假设终端支持
            output = canvas.print()
        else:
            output = canvas.print()

        # 【调试信息】原始输出
        print_func(f"[DEBUG] 原始输出长度: {len(output)} bytes")
        print_func(f"[DEBUG] 原始输出前100字节: {output[:100]}")

        decoded_output = output.decode()
        print_func(f"[DEBUG] 解码后长度: {len(decoded_output)} chars")

        # 在每一行末尾添加颜色重置，防止背景色溢出
        lines = decoded_output.split('\n')

        # 【调试信息】打印输出统计
        print_func(f"[DEBUG] 输出行数: {len(lines)}")
        print_func(f"[DEBUG] 非空行数: {len([l for l in lines if l.strip()])}")
        print_func("=" * 70)

        # 逐行打印，避免单次 print 输出过长导致截断
        for line in lines:
            if line.strip():
                print_func(line + '\033[0m')  # 每行末尾添加颜色重置
            else:
                print_func(line)

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
    import sys

    # 检查命令行参数
    use_sixel = "--sixel" in sys.argv

    if use_sixel:
        print("使用 Sixel 模式（完美像素图片）")
        print("支持终端：Windows Terminal v1.22+, WezTerm, mintty, Konsole, iTerm2\n")
        # Sixel 模式：35 字符宽度等效，适合终端显示
        print_apt_mascot(cols=35, show_banner=True, color_mode=True, use_sixel=True)
    else:
        print("使用字符艺术模式")
        print("提示：添加 --sixel 参数可切换到 Sixel 高清像素模式\n")
        # 字符模式：35 字符宽，适合终端显示
        print_apt_mascot(cols=35, show_banner=True, color_mode=True, use_sixel=False)
