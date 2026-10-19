"""出圖：PNG／JPEG（8-bit）、TIFF（16-bit）、FITS（32-bit 浮點），可以縮圖（附輸出銳化）、嵌入 sRGB 色彩描述檔、
加文字浮水印。給 IG、Threads 用的話，長邊 2048 或 1600 加一點輸出銳化最清楚。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import ops


@dataclass
class ExportOptions:
    long_side: int = 0             # 0＝原尺寸
    sharpen: float = 0.3           # 縮圖後的輸出銳化（0–1）
    icc: bool = True               # 嵌入 sRGB 描述檔
    watermark: str = ""
    position: str = "br"           # br／bl／tr／tl
    opacity: float = 0.6
    size: float = 0.025            # 字高占長邊的比例
    jpeg_quality: int = 95


def _srgb_profile() -> bytes | None:
    try:
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:  # noqa: BLE001  沒有 littleCMS 就不嵌
        return None


def _resize(p: np.ndarray, long_side: int) -> np.ndarray:
    h, w = p.shape[1:]
    if long_side <= 0 or max(h, w) <= long_side:
        return p
    s = long_side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(c, np.float32), "F")
                                .resize(size, Image.Resampling.LANCZOS)) for c in p])


def _sharpen(p: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return p
    Y = ops.luminance(p) if p.shape[0] >= 3 else p[0]
    detail = Y - ops.blur(Y, 0.8)
    return np.clip(p + detail[None] * amount * 1.5, 0, 1)


def _font(px: int):
    for name in ("msjh.ttc", "msjhbd.ttc", "arial.ttf", "PingFang.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _watermark(p: np.ndarray, o: ExportOptions) -> np.ndarray:
    if not o.watermark.strip():
        return p
    h, w = p.shape[1:]
    px = max(10, int(max(h, w) * o.size))
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    font = _font(px)
    x0, y0, x1, y1 = draw.textbbox((0, 0), o.watermark, font=font)
    tw, th = x1 - x0, y1 - y0
    margin = px
    x = w - tw - margin if o.position in ("br", "tr") else margin
    y = h - th - margin * 1.2 if o.position in ("br", "bl") else margin * 0.6
    draw.text((x - x0, y - y0), o.watermark, fill=255, font=font)
    a = np.asarray(layer, np.float32) / 255.0 * float(o.opacity)
    return p * (1 - a[None]) + a[None]


def finish(display: np.ndarray, o: ExportOptions) -> np.ndarray:
    p = ops.planes(np.clip(display, 0, 1).astype(np.float32))
    resized = _resize(p, o.long_side)
    if resized is not p:
        resized = _sharpen(resized, o.sharpen)
    return _watermark(resized, o)


def save(display: np.ndarray, path: Path, o: ExportOptions, header=None, linear: np.ndarray | None = None) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    tmp = path.with_name(f".{path.stem}.saving{path.suffix}")
    try:
        if suffix in (".fit", ".fits", ".fts"):
            from apu_processing import imageio
            data = linear if linear is not None else display
            imageio.save_fits(tmp, data[0] if data.ndim == 3 and data.shape[0] == 1 else data, header)
        else:
            p = finish(display, o)
            icc = _srgb_profile() if o.icc else None
            if suffix in (".tif", ".tiff"):
                import tifffile
                arr = (np.clip(p, 0, 1) * 65535 + 0.5).astype(np.uint16)
                arr = arr[0] if arr.shape[0] == 1 else np.moveaxis(arr, 0, -1)
                kw = {"iccprofile": icc} if icc else {}
                tifffile.imwrite(tmp, arr, photometric="minisblack" if arr.ndim == 2 else "rgb",
                                 compression="zlib", **kw)
            else:
                arr = (np.clip(p, 0, 1) * 255 + 0.5).astype(np.uint8)
                img = Image.fromarray(arr[0] if arr.shape[0] == 1 else np.moveaxis(arr, 0, -1))
                kw = {"icc_profile": icc} if icc else {}
                if suffix in (".jpg", ".jpeg"):
                    img.save(tmp, "JPEG", quality=o.jpeg_quality, subsampling=0, **kw)
                else:
                    img.save(tmp, "PNG", **kw)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
