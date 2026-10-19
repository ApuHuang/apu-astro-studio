"""遮罩：每個步驟都可以只作用在一部分畫面。使用者選「作用範圍」就好，不用自己做遮罩。

mode：
- all：整張
- bright／dark：亮部／暗部（lo、hi 是 0–1 的亮度範圍；線性影像先自動拉伸再判斷，數字才直覺）
- range：亮度在 lo～hi 之間
- stars／nostars：星點／星點以外
- color：顏色範圍（hue 中心、width 寬度，OKLCh 色相）
- brush：手畫（strokes：[(x, y, 半徑, 加或擦)]，完整尺寸的座標）
共同：feather（羽化，完整尺寸 px）、invert（反轉）。
"""

from __future__ import annotations

import numpy as np
from apu_processing import detail
from scipy import ndimage

from . import ops

MODES = ("all", "bright", "dark", "range", "stars", "nostars", "color", "brush")


def default_mask() -> dict:
    return {"mode": "all", "lo": 0.25, "hi": 1.0, "hue": 30.0, "width": 60.0, "feather": 8.0, "invert": False,
            "strokes": []}


def rasterize(strokes: list, shape: tuple[int, int], scale: float,
              origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """手畫的筆觸 → (H, W) 0–1。strokes 的座標是完整尺寸，scale＝完整尺寸 / 這個解析度；
    origin＝這張影像左上角在整張的位置（預覽區用）。"""
    h, w = shape
    m = np.zeros((h, w), np.float32)
    if not strokes:
        return m
    yy, xx = np.ogrid[:h, :w]
    for x, y, r, add in strokes:
        cx, cy, rr = (x - origin[0]) / scale, (y - origin[1]) / scale, max(0.5, r / scale)
        y0, y1 = max(0, int(cy - rr - 1)), min(h, int(cy + rr + 2))
        x0, x1 = max(0, int(cx - rr - 1)), min(w, int(cx + rr + 2))
        if y0 >= y1 or x0 >= x1:
            continue
        disk = (yy[y0:y1] - cy) ** 2 + (xx[:, x0:x1] - cx) ** 2 <= rr * rr
        m[y0:y1, x0:x1][disk] = 1.0 if add else 0.0
    return m


def star_weight(L: np.ndarray, grow: float = 0.0) -> np.ndarray:
    """星點範圍（含星暈，平滑邊緣）；grow 是再往外擴幾 px。"""
    stars = detail.find_stars(L)
    w = stars.weight
    if grow > 0.5:
        w = ndimage.maximum_filter(w, size=int(2 * grow + 1))
    return w.astype(np.float32)


def build(spec: dict | None, img: np.ndarray, linear: bool, scale: float, cache: dict | None = None,
          ref: np.ndarray | None = None, origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray | None:
    """回傳 (H, W) 的 0–1 遮罩；整張（mode=all 且沒反轉）時回傳 None（不用混合，省時間）。
    ref：預覽區時整張的縮圖（線性影像自動拉伸的參數從整張量）；origin：這塊在整張的位置。"""
    if not spec or (spec.get("mode", "all") == "all" and not spec.get("invert")):
        return None
    p = ops.planes(img)
    mode = spec.get("mode", "all")
    h, w = p.shape[1:]
    if mode == "all":
        m = np.ones((h, w), np.float32)
    elif mode in ("bright", "dark", "range"):
        disp = ops.auto_display(p, ref) if linear else p
        Y = ops.luminance(disp)
        lo, hi = float(spec.get("lo", 0.25)), float(spec.get("hi", 1.0))
        soft = 0.08
        if mode == "bright":
            m = ops.smoothstep(lo - soft, lo + soft, Y)
        elif mode == "dark":
            m = 1 - ops.smoothstep(lo - soft, lo + soft, Y)
        else:
            m = ops.smoothstep(lo - soft, lo + soft, Y) * (1 - ops.smoothstep(hi - soft, hi + soft, Y))
    elif mode in ("stars", "nostars"):
        key = ("stars", h, w)
        if cache is not None and key in cache:
            sw = cache[key]
        else:
            sw = star_weight(ops.luminance(p), 1.0 / max(scale, 1e-6))
            if cache is not None:
                cache[key] = sw
        m = sw if mode == "stars" else 1 - sw
    elif mode == "color":
        if p.shape[0] < 3:
            m = np.ones((h, w), np.float32)
        else:
            disp = ops.auto_display(p, ref) if linear else p
            lab = ops.rgb_to_oklab(disp)
            hue = np.degrees(np.arctan2(lab[2], lab[1])) % 360
            chroma = np.hypot(lab[1], lab[2])
            m = ops.angular_weight(hue, float(spec.get("hue", 30)), float(spec.get("width", 60)), 20.0)
            m *= ops.smoothstep(0.01, 0.04, chroma)
    elif mode == "brush":
        m = rasterize(spec.get("strokes", []), (h, w), scale, origin)
    else:
        m = np.ones((h, w), np.float32)
    feather = float(spec.get("feather", 0)) / max(scale, 1e-6)
    if feather > 0.3:
        m = ops.blur(m.astype(np.float32), feather)
    if spec.get("invert"):
        m = 1 - m
    return np.clip(m, 0, 1).astype(np.float32)


def blend(before: np.ndarray, after: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return after
    a, b = ops.planes(before), ops.planes(after)
    out = a + (b - a) * mask[None]
    return ops.unplanes(out.astype(np.float32), before)
