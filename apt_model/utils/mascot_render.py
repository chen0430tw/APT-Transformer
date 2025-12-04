#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Mascot Renderer (APT 吉祥物渲染器)

基于 PTPF 算法为 APT 项目渲染兔子吉祥物
灵感来自 Linux Tux 企鹅启动画面
"""

import os
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional


# =========================
# 核心算法（来自 ptpf_render.py）
# =========================

def box_blur_float01(rgb01: np.ndarray, k: int = 5) -> np.ndarray:
    """盒式模糊（float01 输入输出）"""
    if k <= 1:
        return np.clip(rgb01, 0, 1).astype(np.float32)
    radius = max(1, k // 2)
    im = Image.fromarray((np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    im = im.filter(ImageFilter.BoxBlur(radius))
    return (np.asarray(im).astype(np.float32) / 255.0)


def unsharp_like(img01: np.ndarray, amount: float = 0.5, inner_k: int = 3) -> np.ndarray:
    """轻度反锐化"""
    blur = box_blur_float01(img01, k=inner_k)
    return np.clip(img01 + amount * (img01 - blur), 0, 1)


def to_gray01(rgb01: np.ndarray) -> np.ndarray:
    """RGB 转灰度"""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)


def sobel_fast(gray01: np.ndarray) -> np.ndarray:
    """快速 Sobel 边缘检测"""
    a = np.pad(gray01, ((1,1),(1,1)), mode='edge')
    gx = (a[1:-1, 2:] - a[1:-1, :-2]) + 2*(a[2:, 2:] - a[2:, :-2]) + (a[:-2, 2:] - a[:-2, :-2])
    gy = (a[2:, 1:-1] - a[:-2, 1:-1]) + 2*(a[2:, 2:] - a[:-2, 2:]) + (a[2:, :-2] - a[:-2, :-2])
    g = np.hypot(gx, gy).astype(np.float32)
    m = float(g.max()) + 1e-8
    return (g / m)


def resize_for_terminal(im: Image.Image, cols: int = 80, char_aspect: float = 2.0) -> Image.Image:
    """等距缩放（保持图片比例）"""
    W, H = im.size
    rows = max(2, int(round(H / W * cols / char_aspect)))
    if rows % 2 == 1:
        rows += 1
    return im.resize((cols, rows), Image.BILINEAR)


# =========================
# ANSI 彩色渲染（卡通色盘）
# =========================

def palette64_cartoon() -> np.ndarray:
    """卡通风格 64 色盘（适合兔子角色）"""
    red   = [(218,70,56),(200,56,48),(175,45,40),(150,36,34)]
    orange= [(255,166,66),(244,140,54),(230,118,45),(210,98,34)]
    skin  = [(250,220,190),(243,204,168),(233,188,154),(222,172,140)]
    cream = [(255,244,220),(252,236,206),(246,228,196),(240,220,186)]
    purple= [(94,58,128),(74,44,108),(58,35,92),(46,28,74)]
    dark  = [(56,44,58),(44,34,46),(30,24,32),(18,14,20)]
    star  = [(255,240,210),(255,226,170),(255,206,130),(255,186,98)]
    pal = np.array(red+orange+skin+cream+purple+dark+star, np.uint8)

    # 补充到 64 色
    if pal.shape[0] < 64:
        lv = np.array([0, 95, 175, 255], np.uint8)
        cube = np.array([(r,g,b) for r in lv for g in lv for b in lv], np.uint8)
        pal = np.vstack([pal, cube[:64-pal.shape[0]]])
    return pal[:64]


def fs_dither_quant(rgb_u8: np.ndarray, pal_u8: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Floyd-Steinberg 抖动量化"""
    h, w, _ = rgb_u8.shape
    pal = pal_u8.astype(np.float32)
    arr = rgb_u8.astype(np.float32).copy()
    out = np.empty_like(rgb_u8, dtype=np.uint8)

    for y in range(h):
        xs = range(w) if (y & 1) == 0 else range(w-1, -1, -1)
        for x in xs:
            c = arr[y, x]
            diff = pal - c
            i = int(np.argmin(np.einsum('ij,ij->i', diff, diff)))
            q = pal[i]
            out[y, x] = q.astype(np.uint8)

            # 误差扩散
            err = (c - q) * (strength / 16.0)
            if x + 1 < w: arr[y, x+1] += err * 7
            if y + 1 < h:
                if x - 1 >= 0: arr[y+1, x-1] += err * 3
                arr[y+1, x]   += err * 5
                if x + 1 < w: arr[y+1, x+1] += err * 1
    return np.clip(out, 0, 255)


def ansi_line_truecolor(row_u8: np.ndarray) -> str:
    """生成 Truecolor ANSI 行（使用全块字符 █）"""
    s, last = [], None
    for r, g, b in row_u8.tolist():
        key = (r, g, b)
        if key != last:
            s.append(f"\x1b[38;2;{r};{g};{b}m")
            last = key
        s.append("█")
    s.append("\x1b[0m")
    return "".join(s)


# =========================
# 吉祥物渲染主函数
# =========================

def render_mascot_ansi(
    image_path: str,
    cols: int = 60,
    char_aspect: float = 2.0,
    smooth_k: int = 5,
    unsharp: float = 0.5,
    edge_mix: float = 0.30,
    dither_strength: float = 0.9
) -> str:
    """
    渲染兔子吉祥物为 ANSI 彩色艺术

    参数:
        image_path: 图片路径
        cols: 字符列数（宽度）
        char_aspect: 字符纵横比
        smooth_k: 平滑核大小
        unsharp: 反锐化强度
        edge_mix: 边缘融合比例
        dither_strength: 抖动强度

    返回:
        ANSI 彩色字符串
    """
    if not os.path.exists(image_path):
        return f"[错误] 找不到图片：{image_path}"

    try:
        # 1) 加载并缩放
        im = Image.open(image_path).convert("RGB")
        imr = resize_for_terminal(im, cols=cols, char_aspect=char_aspect)
        rgb01 = np.asarray(imr, np.uint8).astype(np.float32) / 255.0

        # 2) 湿层处理（平滑 + 反锐化）
        wet = box_blur_float01(rgb01, k=smooth_k)
        wet = unsharp_like(wet, amount=unsharp, inner_k=3)

        # 3) 可选边缘提示
        if edge_mix > 0:
            edge = sobel_fast(to_gray01(wet))[..., None]
            wet = np.clip((1-edge_mix)*wet + edge_mix*np.clip(wet + 0.5*edge, 0, 1), 0, 1)

        # 4) 卡通色盘 + FS 抖动
        pal = palette64_cartoon()
        q = fs_dither_quant(
            (wet*255.0 + 0.5).astype(np.uint8),
            pal_u8=pal.astype(np.uint8),
            strength=dither_strength
        )

        # 5) 生成 ANSI 字符串
        lines = []
        for y in range(q.shape[0]):
            lines.append(ansi_line_truecolor(q[y]))

        return "\n".join(lines)

    except Exception as e:
        return f"[错误] 渲染失败: {e}"


def print_apt_mascot(cols: int = 60, show_banner: bool = True):
    """
    打印 APT 兔子吉祥物（类似 Linux Tux）

    参数:
        cols: 显示宽度（字符数）
        show_banner: 是否显示横幅文字
    """
    # 获取兔子图片路径
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mascot_path = os.path.join(script_dir, "兔兔伯爵.png")

    if not os.path.exists(mascot_path):
        # 如果找不到图片，显示简单的文字横幅
        if show_banner:
            print("\n" + "="*70)
            print(" APT - Autopoietic Transformer")
            print(" 自生成变换器")
            print("="*70 + "\n")
        return

    # 渲染并打印兔子
    ansi_art = render_mascot_ansi(
        mascot_path,
        cols=cols,
        char_aspect=2.0,
        smooth_k=5,
        unsharp=0.5,
        edge_mix=0.30,
        dither_strength=0.9
    )

    if show_banner:
        print("\n" + "="*70)
    print(ansi_art)
    if show_banner:
        print("="*70)
        print(" APT - Autopoietic Transformer | 自生成变换器")
        print(" Training Session Starting... | 训练会话启动中...")
        print("="*70 + "\n")


if __name__ == "__main__":
    # 测试渲染
    print_apt_mascot(cols=60, show_banner=True)
