# ptpf_render.py
# -*- coding: utf-8 -*-
"""
PTPF 渲染器（等距缩放 + 双管线分离）
- 模式：
  --mode ascii  生成黑白 ASCII（.txt + 预览 .png）
  --mode ansi   生成彩色 ANSI（.ans + 预览 .png）
  --mode both   两者都生成（禁止混用，分别独立渲染）
- 等距缩放：按等宽字元纵横比 char_aspect（默认 2.0）计算行数，严格保持原图比例；
           不添加黑边，不拉伸/拉高。
- ASCII 管线：湿层（盒式平滑+轻反锐化）→ 边缘提示融合 → 亮度映射到字符集
- ANSI  管线：湿层（盒式平滑+轻反锐化）→ 可选边缘提示 → 64 色盘量化 + FS 抖动（蛇形）
"""

import argparse, os, sys
from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter

# =========================
# 公共工具：I/O 与等距缩放
# =========================
def load_image(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im

def resize_for_terminal(im: Image.Image, cols: int = 240, char_aspect: float = 2.0, even_rows: bool=True) -> Image.Image:
    """严格等距缩放（按等宽字元纵横比换算行数），不裁切、不加边。"""
    W, H = im.size
    rows = max(2, int(round(H / W * cols / char_aspect)))
    if even_rows and (rows % 2 == 1):
        rows += 1
    return im.resize((cols, rows), Image.BILINEAR)

# =========================
# 公共工具：湿层（平滑 + 轻反锐化）
# =========================
def box_blur_float01(rgb01: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return np.clip(rgb01, 0, 1).astype(np.float32)
    radius = max(1, k // 2)
    im = Image.fromarray((np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    im = im.filter(ImageFilter.BoxBlur(radius))
    return (np.asarray(im).astype(np.float32) / 255.0)

def unsharp_like(img01: np.ndarray, amount: float = 0.5, inner_k: int = 3) -> np.ndarray:
    blur = box_blur_float01(img01, k=inner_k)
    return np.clip(img01 + amount * (img01 - blur), 0, 1)

def to_gray01(rgb01: np.ndarray) -> np.ndarray:
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)

def sobel_fast(gray01: np.ndarray) -> np.ndarray:
    a = np.pad(gray01, ((1,1),(1,1)), mode='edge')
    gx = (a[1:-1, 2:] - a[1:-1, :-2]) + 2*(a[2:, 2:] - a[2:, :-2]) + (a[:-2, 2:] - a[:-2, :-2])
    gy = (a[2:, 1:-1] - a[:-2, 1:-1]) + 2*(a[2:, 2:] - a[:-2, 2:]) + (a[2:, :-2] - a[:-2, :-2])
    g = np.hypot(gx, gy).astype(np.float32)
    m = float(g.max()) + 1e-8
    return (g / m)

# =========================
# 黑白 ASCII 管线（独立）
# =========================
def ascii_render(im: Image.Image,
                 cols: int = 240, char_aspect: float = 2.0,
                 smooth_k: int = 3, unsharp: float = 0.5,
                 edge_mix: float = 0.30, gamma: float = 1.0,
                 charset: str = " .,:;i1tfLC08@",
                 out_txt: str = "ascii_out.txt",
                 out_png: str = "ascii_preview.png") -> Tuple[str, str]:
    # 1) 等距缩放
    imr = resize_for_terminal(im, cols=cols, char_aspect=char_aspect)
    rgb01 = np.asarray(imr, np.uint8).astype(np.float32) / 255.0

    # 2) 湿层
    wet = box_blur_float01(rgb01, k=smooth_k)
    wet = unsharp_like(wet, amount=unsharp, inner_k=3)

    # 3) 边缘提示融合（仅亮度域）
    gray = to_gray01(wet)
    if edge_mix > 0:
        edge = sobel_fast(gray)
        # 亮度 + 边缘的保守融合，避免“刮花”
        gray = np.clip((1-edge_mix)*gray + edge_mix*(gray + 0.5*edge), 0, 1)

    # 4) γ 调整（控制“墨量”）
    if abs(gamma - 1.0) > 1e-6:
        gray = np.clip(gray, 0, 1) ** gamma

    # 5) 归一化
    gmin, gmax = float(gray.min()), float(gray.max())
    gray = (gray - gmin) / (gmax - gmin + 1e-8)

    # 6) 映射字符
    bins = np.linspace(0, 1, num=len(charset)+1)
    idx = np.digitize(gray, bins) - 1
    idx = np.clip(idx, 0, len(charset)-1)
    lines = ["".join(charset[j] for j in row) for row in idx]

    # 7) 保存 .txt
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 8) 预览 PNG（把分档亮度渲成灰图，方便对齐观察）
    prev = (idx.astype(np.float32) / (len(charset)-1 + 1e-8) * 255.0).astype(np.uint8)
    Image.fromarray(prev, mode="L").save(out_png)

    return out_txt, out_png

# =========================
# 彩色 ANSI 管线（独立）
# =========================
def palette64_default() -> np.ndarray:
    """通用 64 色盘：砖红/草绿/天灰/钢蓝/柏油/高光（更适合建筑/自然）"""
    brick = [(95,48,32), (128,64,48), (160,80,60), (192,96,72)]
    grass = [(40,64,40), (56,88,56), (72,112,72), (96,136,96)]
    sky   = [(128,136,152), (144,152,168), (168,176,192), (196,204,216)]
    steel = [(88,96,112), (104,112,128), (120,128,144), (136,144,160)]
    asph  = [(24,28,32), (36,40,44), (56,60,64), (80,84,90)]
    hl    = [(210,210,210), (235,235,235), (250,250,250), (255,255,255)]
    pal = np.array(brick + grass + sky + steel + asph + hl, dtype=np.uint8)
    if pal.shape[0] < 64:
        lv = np.array([0, 95, 175, 255], np.uint8)
        cube = np.array([(r,g,b) for r in lv for g in lv for b in lv], np.uint8)
        pal = np.vstack([pal, cube[:64-pal.shape[0]]])
    return pal[:64]

def palette64_cartoon() -> np.ndarray:
    """卡通类 64 色盘：橘/红/奶白/紫阴影/深描边/高光（适合插画/Q版）"""
    red   = [(218,70,56),(200,56,48),(175,45,40),(150,36,34)]
    orange= [(255,166,66),(244,140,54),(230,118,45),(210,98,34)]
    skin  = [(250,220,190),(243,204,168),(233,188,154),(222,172,140)]
    cream = [(255,244,220),(252,236,206),(246,228,196),(240,220,186)]
    purple= [(94,58,128),(74,44,108),(58,35,92),(46,28,74)]
    dark  = [(56,44,58),(44,34,46),(30,24,32),(18,14,20)]
    star  = [(255,240,210),(255,226,170),(255,206,130),(255,186,98)]
    pal = np.array(red+orange+skin+cream+purple+dark+star, np.uint8)
    if pal.shape[0] < 64:
        lv = np.array([0, 95, 175, 255], np.uint8)
        cube = np.array([(r,g,b) for r in lv for g in lv for b in lv], np.uint8)
        pal = np.vstack([pal, cube[:64-pal.shape[0]]])
    return pal[:64]

def get_palette64(name: str) -> np.ndarray:
    if name == "cartoon":
        return palette64_cartoon().astype(np.float32)
    return palette64_default().astype(np.float32)

def fs_dither_quant(rgb_u8: np.ndarray, pal_u8: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Floyd–Steinberg 抖动（蛇形扫描），返回量化后的 RGB u8 图。"""
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
    """把一行 RGB 转成 Truecolor ANSI 文本（前景色+全块字符 █）。"""
    s, last = [], None
    for r, g, b in row_u8.tolist():
        key = (r, g, b)
        if key != last:
            s.append(f"\x1b[38;2;{r};{g};{b}m")
            last = key
        s.append("█")
    s.append("\x1b[0m")
    return "".join(s)

def ansi_render(im: Image.Image,
                cols: int = 240, char_aspect: float = 2.0,
                smooth_k: int = 5, unsharp: float = 0.5,
                edge_mix: float = 0.30, dither_strength: float = 0.9,
                palette: str = "default",
                out_ans: str = "ansi_out.ans",
                out_png: str = "ansi_preview.png") -> Tuple[str, str]:
    # 1) 等距缩放
    imr = resize_for_terminal(im, cols=cols, char_aspect=char_aspect)
    rgb01 = np.asarray(imr, np.uint8).astype(np.float32) / 255.0

    # 2) 湿层
    wet = box_blur_float01(rgb01, k=smooth_k)
    wet = unsharp_like(wet, amount=unsharp, inner_k=3)

    # 3) 可选边缘提示（在色彩层做轻融合，避免“刮花”）
    if edge_mix > 0:
        edge = sobel_fast(to_gray01(wet))[..., None]  # HxWx1
        wet = np.clip((1-edge_mix)*wet + edge_mix*np.clip(wet + 0.5*edge, 0, 1), 0, 1)

    # 4) 64 色盘 + FS 抖动（蛇形）
    pal = get_palette64(palette)
    q = fs_dither_quant((wet*255.0 + 0.5).astype(np.uint8), pal_u8=pal.astype(np.uint8), strength=dither_strength)

    # 5) 保存 Truecolor ANSI（每格一色的块字符）与 PNG 预览
    with open(out_ans, "w", encoding="utf-8") as f:
        for y in range(q.shape[0]):
            f.write(ansi_line_truecolor(q[y]) + "\n")
    Image.fromarray(q).save(out_png)
    return out_ans, out_png

# =========================
# CLI
# =========================
def build_parser():
    ap = argparse.ArgumentParser(description="PTPF 渲染器（等距缩放 + ASCII/ANSI 双管线）")
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("--mode", choices=["ascii", "ansi", "both"], default="both", help="渲染模式")
    ap.add_argument("--cols", type=int, default=240, help="字符列数（宽度），等距缩放只需改这个")
    ap.add_argument("--char-aspect", type=float, default=2.0, help="等宽字元纵横比（常用≈2.0）")

    # ASCII 参数
    ap.add_argument("--ascii-smooth-k", type=int, default=3)
    ap.add_argument("--ascii-unsharp", type=float, default=0.5)
    ap.add_argument("--ascii-edge-mix", type=float, default=0.30, help="0 关闭边缘提示")
    ap.add_argument("--ascii-gamma", type=float, default=1.0)
    ap.add_argument("--charset", default=" .,:;i1tfLC08@", help="明度单调字符表")
    ap.add_argument("--txt", default="ascii_out.txt")
    ap.add_argument("--ascii-png", default="ascii_preview.png")

    # ANSI 参数
    ap.add_argument("--ansi-smooth-k", type=int, default=5)
    ap.add_argument("--ansi-unsharp", type=float, default=0.5)
    ap.add_argument("--ansi-edge-mix", type=float, default=0.30, help="0 关闭边缘提示")
    ap.add_argument("--dither-strength", type=float, default=0.9)
    ap.add_argument("--palette", choices=["default", "cartoon"], default="default")
    ap.add_argument("--ans", default="ansi_out.ans")
    ap.add_argument("--ansi-png", default="ansi_preview.png")
    return ap

def main():
    args = build_parser().parse_args()
    if not os.path.exists(args.image):
        print(f"[错误] 找不到图片：{args.image}", file=sys.stderr)
        sys.exit(1)

    im = load_image(args.image)

    if args.mode in ("ascii", "both"):
        ascii_render(
            im=im,
            cols=args.cols, char_aspect=args.char_aspect,
            smooth_k=args.ascii_smooth_k, unsharp=args.ascii_unsharp,
            edge_mix=args.ascii_edge_mix, gamma=args.ascii_gamma,
            charset=args.charset,
            out_txt=args.txt, out_png=args.ascii_png,
        )
        print(f"[ASCII] 已输出：{args.txt} / {args.ascii_png}")

    if args.mode in ("ansi", "both"):
        ansi_render(
            im=im,
            cols=args.cols, char_aspect=args.char_aspect,
            smooth_k=args.ansi_smooth_k, unsharp=args.ansi_unsharp,
            edge_mix=args.ansi_edge_mix, dither_strength=args.dither_strength,
            palette=args.palette,
            out_ans=args.ans, out_png=args.ansi_png,
        )
        print(f"[ANSI ] 已输出：{args.ans} / {args.ansi_png}")

if __name__ == "__main__":
    main()
