"""色彩：校色（拉伸前）、調色與曲線（拉伸後）。

調色在 OKLab／OKLCh 裡做：轉色相時亮度不變、加飽和不會讓某些顏色突然變亮，比 HSV 自然。
"""

from __future__ import annotations

import math

import numpy as np
from apu_processing import color as pcolor
from apu_processing import display
from scipy import ndimage

from .. import ops
from . import T, Ctx, Tool, choice, register, section, slider, special, toggle

# ---------------------------------------------------------------------- 校色


class ColorCalibration(Tool):
    kind = "color_cal"
    title = T("校色", "Color Calibration")
    help = T("背景中性化：讓天空三色一樣。白平衡：用很多顆星的平均顏色校正（近似星表校色的結果）。窄帶合成通常不做。",
             "Background neutralization makes the sky neutral. White balance uses the average colour of many stars "
             "(close to catalogue-based calibration). Usually skipped for narrowband.")
    phase = "linear"
    icon = "color"
    maskable = False
    auto_layer = "main"

    def defaults(self) -> dict:
        return {"neutralize": True, "balance": "stars", "strength": 1.0}

    def controls(self) -> list[dict]:
        return [
            toggle("neutralize", T("背景中性化", "Neutralize background")),
            choice("balance", T("白平衡", "White balance"), [("stars", T("依星點顏色", "From star colours")),
                                                           ("none", T("不做", "None"))]),
            slider("strength", T("白平衡強度", "Balance strength"), 0, 1, 0.05, "{:.0%}",
                   show=lambda p: p.get("balance") == "stars"),
        ]

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        if img.shape[0] < 3:
            return img
        out = img
        # 白平衡倍率、天空、飽和星的上限都從整張量（預覽區時用整張的縮圖量）
        whole = ctx.whole(img)
        scale = ctx.whole_scale
        if params.get("balance", "stars") == "stars":
            key = ("balance", whole.shape)
            bal = ctx.cache.get(key)
            if bal is None:
                bal = pcolor.measure_balance(whole)
                ctx.cache[key] = bal
            k = float(params.get("strength", 1.0))
            gains = tuple(1 + (g - 1) * k for g in bal.gains)
            ctx.view["gains"] = gains
            noise = ops.measure_noise(whole, scale)
            tops = [max(float(img[c].max()), float(whole[c].max())) for c in range(img.shape[0])]
            out = _apply_gains(out, gains, noise.sky, tops, ctx.scale)
            if whole is not img:
                whole = _apply_gains(whole, gains, noise.sky, tops, ctx.whole_scale)
            else:
                whole = out
        if params.get("neutralize", True):
            sky = np.asarray(ops.measure_noise(whole, scale).sky, np.float32)
            out = out - (sky - sky.mean())[:, None, None]
        return out.astype(np.float32)


def _apply_gains(image: np.ndarray, gains, sky, tops: list[float], scale: float = 1.0) -> np.ndarray:
    """(x − 天空)·倍率 + 天空；已經飽和的區域另外處理。

    飽和用「局部平頂」判斷（psf.saturation_mask）：WBPP 的 master 每顆飽和星停在不同高度（0.93～0.97），
    跟整張最大值比只抓得到一兩顆，其他的星核乘上不同倍率後變成粉紅色、外圍一圈紅（甜甜圈）。
    飽和區本身沒有顏色資訊（被截掉了）：亮度取三色最大值，顏色——
    - 大的（星系核心）：外圍一圈的顏色（改成白色的話，M31 整個飽和的核心變成一顆藍白色的蛋）
    - 小的（星點）：中性白。光暈的顏色不等於星的顏色（紅光的星點比較大，光暈偏紅）；用外圈的顏色的話，
      紅色星雲裡的亮星核心變成一塊粉紅色的圓盤"""
    from .. import psf
    sky = np.asarray(sky, np.float32)[:, None, None]
    g = np.asarray(gains, np.float32)[:, None, None]
    out = ((image - sky) * g + sky).astype(np.float32)
    near = psf.saturation_mask(image, tops=tops)
    if not near.any() or near.mean() > 0.05:
        return out
    region = ndimage.binary_dilation(near, iterations=2)
    soft = ndimage.gaussian_filter(region.astype(np.float32), 1.5)
    target = out.copy()
    lab, n = ndimage.label(region)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        pad = 6
        y0, y1 = max(0, sl[0].start - pad), min(out.shape[1], sl[0].stop + pad)
        x0, x1 = max(0, sl[1].start - pad), min(out.shape[2], sl[1].stop + pad)
        comp = lab[y0:y1, x0:x1] == i
        patch = out[:, y0:y1, x0:x1]
        col = np.ones(out.shape[0])
        if comp.sum() > 400 / scale ** 2:                    # 大的（星系核心）：外圍一圈的顏色
            ring = ndimage.binary_dilation(comp, iterations=4) & ~region[y0:y1, x0:x1]
            if ring.sum() >= 4:
                col = np.maximum(np.array([np.median(patch[c][ring]) for c in range(out.shape[0])]) - sky[:, 0, 0], 0)
        col = col / col.max() if col.max() > 0 else np.ones(out.shape[0])
        bright = patch.max(axis=0)
        t = sky[:, 0, 0, None, None] + (bright - sky.max())[None] * col[:, None, None]
        target[:, y0:y1, x0:x1][:, comp] = t[:, comp]
    return out + soft[None] * (target - out)


register(ColorCalibration())

# ---------------------------------------------------------------------- 調色

PALETTES = {
    "none": [],
    # SHO 金藍：綠色往金色轉、青色往藍色轉（窄帶最常見的風格）
    "sho_gold": [(142, 80, -70, 1.15, 0.0), (195, 60, 45, 1.2, 0.0)],
    # SHO 橘青：綠色轉橘、加深青色
    "sho_orange": [(142, 80, -85, 1.25, 0.0), (195, 60, 10, 1.3, -0.02)],
    # HOO 紅青：紅色偏暖、青色提亮
    "hoo": [(29, 50, 8, 1.1, 0.0), (195, 70, -5, 1.2, 0.02)],
}


class ColorGrade(Tool):
    kind = "color_grade"
    title = T("調色", "Color Grading")
    help = T("三個色輪分別調暗部、中間調、亮部的色調；自然飽和度只加強原本不飽和的地方；"
             "選擇性色彩可以用滴管在畫面上點一個顏色再調整它。",
             "Wheels tint shadows, midtones and highlights; vibrance boosts only muted colours; "
             "selective colour lets you pick a colour from the image with the eyedropper and adjust it.")
    phase = "nonlinear"
    icon = "palette"

    def defaults(self) -> dict:
        d = {"saturation": 1.0, "vibrance": 0.0, "protect": 0.5, "palette": "none", "scnr": 0.0,
             "sel_on": False, "sel_hue": 142.0, "sel_width": 60.0, "sel_shift": 0.0, "sel_sat": 1.0, "sel_light": 0.0}
        for z in ("sh", "mid", "hi"):
            d.update({f"{z}_h": 0.0, f"{z}_a": 0.0, f"{z}_l": 0.0})
        return d

    def controls(self) -> list[dict]:
        sel = lambda p: bool(p.get("sel_on"))  # noqa: E731
        return [
            special("wheels", "", T("色輪（暗部／中間調／亮部）", "Wheels (shadows / midtones / highlights)")),
            section(T("飽和度", "Saturation")),
            slider("vibrance", T("自然飽和度", "Vibrance"), -1, 1, 0.05, "{:+.2f}",
                   help=T("只加強原本不飽和的顏色，已經很鮮豔的地方不太變。", "Boosts muted colours more than vivid ones.")),
            slider("saturation", T("飽和度", "Saturation"), 0, 2, 0.05, "{:.0%}"),
            slider("protect", T("保護背景", "Protect background"), 0, 1, 0.05, "{:.0%}",
                   help=T("暗的背景少加飽和，避免把背景噪聲加成彩色雜點。", "Adds less saturation in the dark sky so "
                                                                     "noise doesn't turn into colour speckles.")),
            section(T("窄帶色盤", "Narrowband palette")),
            choice("palette", T("色盤", "Palette"), [("none", T("不套用", "None")),
                                                    ("sho_gold", T("SHO 金藍", "SHO gold/blue")),
                                                    ("sho_orange", T("SHO 橘青", "SHO orange/teal")),
                                                    ("hoo", T("HOO 紅青", "HOO red/teal"))]),
            slider("scnr", T("去綠", "Remove green"), 0, 1, 0.05, "{:.0%}"),
            section(T("選擇性色彩", "Selective colour")),
            toggle("sel_on", T("啟用", "Enable")),
            special("eyedropper", "sel_hue", T("滴管：在畫面上點一個顏色", "Eyedropper: click a colour in the image"),
                    show=sel),
            slider("sel_hue", T("色相", "Hue"), 0, 360, 1, "{:.0f}°", show=sel),
            slider("sel_width", T("範圍", "Range"), 10, 180, 5, "{:.0f}°", show=sel),
            slider("sel_shift", T("色相偏移", "Hue shift"), -90, 90, 1, "{:+.0f}°", show=sel),
            slider("sel_sat", T("飽和度", "Saturation"), 0, 2, 0.05, "{:.0%}", show=sel),
            slider("sel_light", T("明暗", "Lightness"), -0.2, 0.2, 0.01, "{:+.2f}", show=sel),
        ]

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        p = np.clip(img, 0, 1)
        if p.shape[0] < 3:
            # 單色：只有色輪的明暗有意義
            Y = p[0]
            out = Y.copy()
            for z, w in self._zones(Y):
                out += w * float(params.get(f"{z}_l", 0)) * 0.15
            return np.clip(out, 0, 1)[None].astype(np.float32)
        ops_list = list(PALETTES.get(params.get("palette", "none"), []))
        if params.get("sel_on"):
            ops_list.append((float(params.get("sel_hue", 142)), float(params.get("sel_width", 60)),
                             float(params.get("sel_shift", 0)), float(params.get("sel_sat", 1)),
                             float(params.get("sel_light", 0))))
        sat, vib = float(params.get("saturation", 1)), float(params.get("vibrance", 0))
        if ops_list or abs(sat - 1) > 1e-3 or abs(vib) > 1e-3:
            lab = ops.rgb_to_oklab(p)
            Lk, a, b = lab[0], lab[1], lab[2]
            C = np.hypot(a, b)
            H = np.degrees(np.arctan2(b, a)) % 360
            for center, width, shift, csc, light in ops_list:
                w = ops.angular_weight(H, center, width, width * 0.5) * ops.smoothstep(0.005, 0.03, C)
                H = (H + shift * w) % 360
                C = C * (1 + (csc - 1) * w)
                Lk = Lk + light * w
                ctx.check()
            if abs(sat - 1) > 1e-3 or abs(vib) > 1e-3:
                protect = float(params.get("protect", 0.5))
                Y = ops.luminance(p)
                guard = 1 - protect * (1 - ops.smoothstep(0.08, 0.35, Y))
                boost = (sat - 1) + vib * (1 - ops.smoothstep(0.0, 0.18, C))
                C = C * np.maximum(0, 1 + boost * guard)
            hr = np.radians(H)
            p = ops.oklab_to_rgb(np.stack([Lk, C * np.cos(hr), C * np.sin(hr)]))
        # 色輪：依亮度分三區各加一個色偏與明暗
        Y = ops.luminance(p)
        add = np.zeros_like(p)
        for z, w in self._zones(Y):
            amt, light = float(params.get(f"{z}_a", 0)), float(params.get(f"{z}_l", 0))
            if abs(amt) > 1e-4:
                add += w[None] * (ops.hue_rgb(float(params.get(f"{z}_h", 0)))[:, None, None] * amt * 0.12)
            if abs(light) > 1e-4:
                add += w[None] * light * 0.15
        p = p + add
        g = float(params.get("scnr", 0))
        if g > 1e-4:
            neutral = (p[0] + p[2]) * 0.5
            p[1] = p[1] - np.maximum(p[1] - neutral, 0) * g
        return np.clip(p, 0, 1).astype(np.float32)

    @staticmethod
    def _zones(Y: np.ndarray):
        sh = np.clip(1 - Y / 0.55, 0, 1) ** 2
        hi = np.clip((Y - 0.45) / 0.55, 0, 1) ** 2
        mid = np.clip(1 - sh - hi, 0, 1)
        return (("sh", sh), ("mid", mid), ("hi", hi))


register(ColorGrade())

# ---------------------------------------------------------------------- 曲線


def identity_curves() -> dict:
    return {k: [[0.0, 0.0], [1.0, 1.0]] for k in ("master", "red", "green", "blue")}


class Curves(Tool):
    kind = "curves"
    title = T("曲線", "Curves")
    help = T("主曲線調明暗與對比，紅綠藍各自的曲線微調色調。雙擊加點、雙擊點刪除。",
             "The master curve sets brightness and contrast; the red, green and blue curves fine-tune colour. "
             "Double-click to add or remove points.")
    phase = "nonlinear"
    icon = "curves"

    def defaults(self) -> dict:
        return {"curves": identity_curves()}

    def controls(self) -> list[dict]:
        return [special("curves", "curves", T("曲線", "Curves"))]

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        c = params.get("curves") or identity_curves()
        adj = display.DisplayAdjustments(curves=display.ToneCurves(**{k: [list(map(float, q)) for q in v]
                                                                       for k, v in c.items()}))
        if adj.is_neutral():
            return img
        src = np.clip(img, 0, 1)
        out = display.apply(src[0] if src.shape[0] == 1 else src, adj)
        return ops.planes(np.asarray(out, np.float32))


register(Curves())

__all__ = ["ColorCalibration", "ColorGrade", "Curves", "math"]
