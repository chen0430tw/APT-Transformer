#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Mascot Renderer (APT 吉祥物渲染器)

使用多种渲染引擎在终端渲染兔子吉祥物
灵感来自 Linux Tux 企鹅启动画面

支持的渲染模式：
- PTPF: 高质量半块彩色渲染（推荐，所有终端）
- Sixel: 完美像素图形（需终端支持）
- 字符艺术: 经典字符画（最大兼容性）
"""

import os
import sys
import shutil
from typing import Optional

# 导入设置管理器用于debug模式控制
try:
    from apt.core.config.settings_manager import settings
    HAS_SETTINGS = True
except ImportError:
    HAS_SETTINGS = False

# 检查是否安装了 PTPF Lite
try:
    # 尝试从scripts目录导入PTPF模块
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scripts_dir = os.path.join(script_dir, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from ptpf_lite import (
        ptpf_render_ansi_hpq_sosa,
        PTPFConfig
    )
    from PIL import Image
    HAS_PTPF = True
except (ImportError, Exception):
    HAS_PTPF = False

# Optional: numpy for low-cols fusion renderer
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


def _is_debug_enabled():
    """检查是否启用debug模式"""
    if HAS_SETTINGS:
        return settings.get_debug_enabled()
    return False


def _render_halfblock_fused_ansi(image, cols: int, frames: int = 4, samples: int = 5, prefilter: bool = True) -> str:
    """Low-cols (e.g. 45) detail-preserving downsample using multi-jitter median fusion.

    This is the 'LOVE crater' trick: when spatial bandwidth is too low, avoid mean-pooling.
    We take several micro-sampled medians (frames) and *average* them into a stable fg/bg.
    Output is a single static ANSI frame (no animation needed).
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy not available")
    from PIL import Image, ImageFilter

    img = image.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, img).convert("RGB")
    if prefilter:
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1, percent=75, threshold=2))
    rgb_u8 = np.array(rgb, dtype=np.uint8)
    H, W, _ = rgb_u8.shape

    # terminal char cells are ~2:1 tall (half-block doubles vertical detail)
    char_aspect = 2.0
    rows = max(1, int(round((H / W) * cols * (1.0 / char_aspect))))
    xs = np.linspace(0, W, cols + 1).astype(int)
    ys = np.linspace(0, H, rows + 1).astype(int)

    def _hash2(i, j, k=0):
        x = (i * 0x1F1F1F1F) ^ (j * 0x5BD1E995) ^ (k * 0x27D4EB2D)
        x = (x ^ (x >> 15)) * 0x85EBCA6B
        x = (x ^ (x >> 13)) * 0xC2B2AE35
        x = x ^ (x >> 16)
        return (x & 0xFFFFFFFF) / 2**32

    def _pick_points(x0, x1, y0, y1, n, salt, i, j, k):
        pts = []
        w = max(1, (x1 - x0) - 1)
        h = max(1, (y1 - y0) - 1)
        for t in range(n):
            r1 = _hash2(i * 131 + t * 17 + salt, j * 313 + t * 29 + salt, k)
            r2 = _hash2(i * 911 + t * 23 + salt, j * 719 + t * 31 + salt, k + 7)
            xx = x0 + int(r1 * w)
            yy = y0 + int(r2 * h)
            pts.append((yy, xx))
        return pts

    lines = []
    for j in range(rows):
        y0, y1 = ys[j], ys[j + 1]
        ym = (y0 + y1) // 2
        segs = []
        for i in range(cols):
            x0, x1 = xs[i], xs[i + 1]
            if x1 <= x0 or y1 <= y0:
                fg = bgc = (0, 0, 0)
            else:
                # fusion across 'frames'
                fg_list = []
                bg_list = []
                for k in range(frames):
                    up_pts = _pick_points(x0, x1, y0, ym, samples, 11, i, j, k)
                    dn_pts = _pick_points(x0, x1, ym, y1, samples, 37, i, j, k)
                    up = np.array([rgb_u8[yy, xx] for yy, xx in up_pts], dtype=np.uint8)
                    dn = np.array([rgb_u8[yy, xx] for yy, xx in dn_pts], dtype=np.uint8)
                    fg_list.append(np.median(up, axis=0))
                    bg_list.append(np.median(dn, axis=0))
                fg = tuple(np.clip(np.mean(fg_list, axis=0), 0, 255).astype(np.uint8).tolist())
                bgc = tuple(np.clip(np.mean(bg_list, axis=0), 0, 255).astype(np.uint8).tolist())

            segs.append(
                f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m"
                f"\x1b[48;2;{bgc[0]};{bgc[1]};{bgc[2]}m▀"
            )
        lines.append("".join(segs) + "\x1b[0m")

    return "\n".join(lines)

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


def print_apt_mascot(cols: int = 50, show_banner: bool = True, color_mode: bool = True, print_func=None,
                     use_sixel: bool = False, use_ptpf: bool = None, use_ascii: bool = False):
    """
    打印 APT 兔子吉祥物（类似 Linux Tux 小巧 Logo）

    参数:
        cols: 显示宽度（字符数或像素列数，默认50，推荐范围45-70）
        show_banner: 是否显示横幅文字
        color_mode: 是否使用彩色模式（默认 True，chafa支持很好的彩色）
        print_func: 自定义输出函数（默认使用print，在logger环境中可传入info_print）
        use_sixel: 强制使用 Sixel 图形模式（默认 False）
                   Sixel 模式可显示完美像素图片（高清像素渲染）
                   支持终端：Windows Terminal (v1.22+), WezTerm, mintty, Konsole, iTerm2等
        use_ptpf: 强制使用 PTPF Lite 高质量半块渲染（默认 None=自动）
                  PTPF 模式使用 HPQ + SOSA 算法，半块字符（█ ▀ ▄）提供2x垂直分辨率
                  支持所有终端，自动结构感知，高质量彩色输出
        use_ascii: 强制使用传统字符艺术模式（默认 False）

    自动模式（默认）：
        优先级: PTPF（如果可用）→ 字符艺术 → 显示安装建议
        这样默认就能看到PTPF的高质量渲染，无需手动指定参数

    强制模式：
        use_sixel=True  → 强制 Sixel（失败则降级到字符艺术）
        use_ptpf=True   → 强制 PTPF（失败则降级到字符艺术）
        use_ascii=True  → 强制字符艺术

    设计理念:
        - 小巧简洁的 Logo，类似 Linux Tux 企鹅
        - 智能选择最佳渲染引擎，展示最佳效果
        - 优雅降级，确保任何环境都能正常显示
    """
    # 默认使用 print，除非指定了自定义函数
    if print_func is None:
        print_func = print

    # 获取兔子图片路径（所有模式都需要）
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mascot_path = os.path.join(script_dir, "docs", "assets", "兔兔伯爵.png")

    if not os.path.exists(mascot_path):
        print_func("  (找不到吉祥物图片)")
        if show_banner:
            print_func("="*70)
            print_func("  Training Session Starting... | 训练会话启动中...")
            print_func("="*70 + "\n")
        return

    # 自动模式：如果没有明确指定模式，默认尝试PTPF
    if use_ptpf is None and not use_sixel and not use_ascii:
        use_ptpf = HAS_PTPF  # 如果PTPF可用就用，否则降级到ASCII

    # PTPF 模式：高质量半块彩色渲染（优先级最高）
    if use_ptpf:
        if not HAS_PTPF:
            print_func("  (PTPF 模块未找到，切换到字符艺术模式)")
            use_ptpf = False
        else:
            if show_banner:
                print_func("\n" + "="*70)
                print_func("  APT - Autopoietic Transformer | 自生成变换器")
                print_func("="*70)

            try:
                # 加载图片
                image = Image.open(mascot_path).convert("RGB")

                # 使用 PTPF Lite 渲染
                # PTPF使用半块字符，适度提升分辨率保持细节但不过大
                # 建议50-70列以获得良好效果
                ptpf_cols = min(cols, 65)  # never exceed terminal width; cap at 65

                cfg = PTPFConfig(
                    cols=ptpf_cols,
                    char_aspect=2.0,
                    blur_k=1,  # 进一步减少模糊，保留更多细节
                    unsharp_amount=1.0,  # 增强锐化，让边缘更清晰
                    sat_k=1.5,  # 进一步增加饱和度，色彩更鲜艳
                    gray_mix=0.05,  # 进一步减少灰度混合，提升色彩纯度
                    sosa_edge_gain=1.4,  # 进一步增强边缘检测
                    sosa_thresh=0.40,  # 略微降低阈值，保留更多细节
                    hole_amp_A=0.010,  # 减少抖动噪声
                    hole_amp_B=0.020,
                )

                # 渲染ANSI输出
                # - cols <= 45: use low-cols fusion with enhanced quality (more samples)
                # - else: use PTPF (HPQ+SOSA high-quality look)
                if cols <= 45:
                    # 低分辨率：使用增强的 fusion 渲染（更多采样点以保持清晰度）
                    ansi_output = _render_halfblock_fused_ansi(image, cols=cols, frames=6, samples=8, prefilter=True)
                else:
                    # 中高分辨率：使用 PTPF 高质量渲染
                    ansi_output = ptpf_render_ansi_hpq_sosa(image, cols=ptpf_cols, mode="auto", cfg=cfg)

                # 打印输出
                print_func(ansi_output)

            except Exception as e:
                print_func(f"  (PTPF 渲染失败: {e})")

            if show_banner:
                print_func("="*70)
                print_func("  Training Session Starting... | 训练会话启动中...")
                print_func("="*70 + "\n")
            return

    # Sixel 模式：使用系统 chafa 命令（更可靠）
    if use_sixel:
        import subprocess

        # 检查系统是否有 chafa 命令（Windows 需要 .exe）
        chafa_cmd = shutil.which("chafa") or shutil.which("chafa.exe")

        # 如果 PATH 中找不到，尝试常见的 winget 安装路径
        if not chafa_cmd:
            import glob
            winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
            winget_paths = [
                # WinGet Links（符号链接目录）
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\chafa.exe"),
                # WinGet Packages 目录（包含完整 Source 后缀 + 版本号子目录）
                # 结构: hpjansson.Chafa_Microsoft.Winget.Source_xxx/chafa-x.x.x-x-xxx/bin/chafa.exe
                os.path.join(winget_base, "hpjansson.Chafa*", "chafa-*", "bin", "chafa.exe"),
                os.path.join(winget_base, "hpjansson.Chafa*", "chafa-*", "chafa.exe"),
                os.path.join(winget_base, "hpjansson.Chafa*", "bin", "chafa.exe"),
                os.path.join(winget_base, "hpjansson.Chafa*", "chafa.exe"),
            ]
            for path_pattern in winget_paths:
                matches = glob.glob(path_pattern)
                if matches:
                    chafa_cmd = matches[0]
                    break

        if _is_debug_enabled():
            print_func(f"[DEBUG] 查找系统 chafa: {chafa_cmd}")
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
            if _is_debug_enabled():
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
        if _is_debug_enabled():
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
            if _is_debug_enabled():
                print_func(f"[DEBUG] 检测到终端: {term_info}")
            output = canvas.print(term_info=term_info)
        elif use_sixel:
            # 没有 TermDb，使用 fallback（假设终端支持 Sixel）
            if _is_debug_enabled():
                print_func("[DEBUG] 使用 fallback 模式（假设终端支持 Sixel）")
            # 直接输出，假设终端支持
            output = canvas.print()
        else:
            output = canvas.print()

        # 【调试信息】原始输出
        if _is_debug_enabled():
            print_func(f"[DEBUG] 原始输出长度: {len(output)} bytes")
            print_func(f"[DEBUG] 原始输出前100字节: {output[:100]}")

        decoded_output = output.decode()

        if _is_debug_enabled():
            print_func(f"[DEBUG] 解码后长度: {len(decoded_output)} chars")

        # 在每一行末尾添加颜色重置，防止背景色溢出
        lines = decoded_output.split('\n')

        # 【调试信息】打印输出统计
        if _is_debug_enabled():
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
    force_ptpf = "--ptpf" in sys.argv
    force_sixel = "--sixel" in sys.argv
    force_ascii = "--ascii" in sys.argv

    if force_sixel:
        print("强制使用 Sixel 模式（完美像素图片）")
        print("支持终端：Windows Terminal v1.22+, WezTerm, mintty, Konsole, iTerm2\n")
        print_apt_mascot(cols=35, show_banner=True, color_mode=True, use_sixel=True)
    elif force_ptpf:
        print("强制使用 PTPF Lite 模式（高质量半块彩色渲染）")
        print("特性：HPQ预处理 + SOSA结构感知 + S-FSO抖动 + 半块字符（2x分辨率）")
        print("支持所有终端，自动优化边缘和填充区域\n")
        print_apt_mascot(cols=35, show_banner=True, color_mode=True, use_ptpf=True)
    elif force_ascii:
        print("强制使用字符艺术模式（传统chafa字符画）")
        print("最大兼容性，支持所有终端\n")
        print_apt_mascot(cols=35, show_banner=True, color_mode=True, use_ascii=True)
    else:
        # 自动模式：优先PTPF，降级到ASCII
        print("自动选择最佳渲染模式...")
        if HAS_PTPF:
            print("✓ 使用 PTPF Lite（高质量半块渲染 + 2x分辨率）\n")
        elif HAS_CHAFA:
            print("✓ 使用字符艺术模式（PTPF不可用，需要: pip install numpy pillow）\n")
        else:
            print("! 未安装渲染库\n")

        print("提示：")
        print("  --sixel : 强制 Sixel 高清像素模式（需终端支持）")
        print("  --ptpf  : 强制 PTPF Lite 高质量半块渲染")
        print("  --ascii : 强制传统字符艺术模式\n")

        # 自动模式：use_ptpf=None会自动选择最佳可用模式
        print_apt_mascot(cols=35, show_banner=True, color_mode=True)
