"""後製用的數學：亮度、色彩空間（sRGB ↔ OKLab）、拉伸曲線（GHS、arcsinh、MTF）、顯示用的自動拉伸。

影像形狀一律 (C, H, W)（單色 C=1），float32。
"""

from __future__ import annotations

import math

import numpy as np

from . import backend


def planes(img: np.ndarray) -> np.ndarray:
    return img[None] if img.ndim == 2 else img


def unplanes(p: np.ndarray, like: np.ndarray) -> np.ndarray:
    return p[0] if like.ndim == 2 else p


def luminance(p: np.ndarray) -> np.ndarray:
    """(C, H, W) → (H, W)。彩色用 Rec.709 權重（跟 Processing 的飽和度同一套）。"""
    if p.shape[0] >= 3:
        return (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).astype(np.float32)
    return p[0]


def robust_stats(a: np.ndarray, sample: int = 400_000) -> tuple[float, float]:
    """(中位數, MAD×1.4826)，抽樣算。"""
    flat = a.ravel()
    if flat.size > sample:
        flat = flat[:: flat.size // sample]
    flat = flat[np.isfinite(flat)]
    if not flat.size:
        return 0.0, 0.0
    med = float(np.median(flat))
    return med, float(1.4826 * np.median(np.abs(flat - med)))


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    if e1 <= e0:
        return (x >= e0).astype(np.float32)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return backend.gaussian(img, sigma)


# ---------------------------------------------------------------------- 顯示用的自動拉伸（線性影像看得到東西）


def stf_params(p: np.ndarray, target: float = 0.25, clip: float = 2.8) -> list[tuple[float, float, float]]:
    """每個色版的 (黑點, 範圍, midtone)，類似 PixInsight 的 STF。"""
    out = []
    for ch in p:
        med, mad = robust_stats(ch)
        top = float(np.percentile(ch[:: max(1, ch.shape[0] // 400), :: max(1, ch.shape[1] // 400)], 99.99))
        low = med - clip * mad
        span = max(top - low, 1e-6)
        m_n = float(np.clip((med - low) / span, 1e-6, 1 - 1e-6))
        out.append((low, span, m_n * (1 - target) / (m_n - 2 * target * m_n + target)))
    return out


def mtf(m: float, x: np.ndarray) -> np.ndarray:
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def apply_stf(p: np.ndarray, params: list[tuple[float, float, float]], linked: bool = False) -> np.ndarray:
    out = np.empty_like(p, dtype=np.float32)
    if linked:
        low = float(np.mean([q[0] for q in params]))
        span = float(np.mean([q[1] for q in params]))
        m = float(np.mean([q[2] for q in params]))
        params = [(low, span, m)] * len(p)
    for c, (low, span, m) in enumerate(params):
        x = np.clip((p[c] - low) / span, 0, 1)
        out[c] = np.clip(mtf(m, x), 0, 1)
    return out


def pixel_noise(plane: np.ndarray) -> float:
    """單一像素的噪聲 σ：相鄰像素差的 MAD（不受星點、星雲影響），抽樣算。"""
    a = plane[:: 2, :: 2] if plane.size > 4_000_000 else plane
    d = (a[1::2, :] - a[:-1:2, :])[:, ::2]
    return float(np.median(np.abs(d - np.median(d)))) * 1.4826 / np.sqrt(2)


def measure_noise(img: np.ndarray, scale: float):
    """apu_processing.stretch.measure 的包裝。Processing 0.2.1 在天空區塊（64／倍率）是奇數且 ≥16 px 時
    reshape 會失敗（例如快速預覽縮小 3 倍 → 21 px，M31 4920×3276 就遇到）；稍微調整倍率讓區塊變成偶數
    （噪聲尺度的換算差不到 3%）。已寫進 docs/handoff-to-processing.md。"""
    from apu_processing import stretch as pstretch
    tile = max(8, int(round(64 / scale)))
    if tile >= 16 and tile % 2:
        scale = 64.0 / (tile + 1)
    return pstretch.measure(img, scale)


def auto_display(p: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """線性影像 → 看得到的 0–1 顯示（只為了顯示，不改資料）。三色各自拉伸（像 PixInsight 的 STF 不連動）：
    彩色相機校色前綠色特別強，連動的話整張偏綠，看不出光害梯度。ref：拉伸參數從這張量（預覽區用整張的縮圖）。"""
    return apply_stf(p, stf_params(p if ref is None else ref), linked=False)


# ---------------------------------------------------------------------- GHS（Generalized Hyperbolic Stretch）


def _ghs_t(x: np.ndarray, D: float, b: float) -> np.ndarray:
    if D <= 0:
        return x
    if b == -1:
        return np.log1p(D * x)
    if b < 0:
        return (1 - np.power(np.maximum(1 - b * D * x, 1e-12), (b + 1) / b)) / (D * (b + 1))
    if b == 0:
        return 1 - np.exp(-D * x)
    return 1 - np.power(1 + b * D * x, -1 / b)


def _ghs_dt(x: float, D: float, b: float) -> float:
    if D <= 0:
        return 1.0
    if b == -1:
        return D / (1 + D * x)
    if b < 0:
        return max(1 - b * D * x, 1e-12) ** (1 / b)
    if b == 0:
        return D * math.exp(-D * x)
    return (1 + b * D * x) ** (-(1 + b) / b)


def ghs(x: np.ndarray, ln_d1: float, b: float, sp: float, lp: float = 0.0, hp: float = 1.0) -> np.ndarray:
    """GHS 拉伸（Cranfield & Payne）：ln_d1 = ln(D+1) 是拉伸量，b 是局部強度，sp 是對稱點（最常放在背景峰值），
    lp 以下、hp 以上改成直線（保護暗部／亮部）。x 是 0–1，輸出正規化到 0–1。"""
    D = math.expm1(max(0.0, ln_d1))
    if D <= 1e-9:
        return np.clip(x, 0, 1).astype(np.float32)
    sp = float(np.clip(sp, 0.0, 1.0))
    lp = float(np.clip(lp, 0.0, sp))
    hp = float(np.clip(hp, sp, 1.0))
    x = np.clip(x, 0.0, 1.0).astype(np.float64)

    def raw(v: np.ndarray) -> np.ndarray:
        y = np.empty_like(v)
        a = v < lp
        bseg = (v >= lp) & (v < sp)
        c = (v >= sp) & (v < hp)
        d = v >= hp
        y[bseg] = -_ghs_t(sp - v[bseg], D, b)
        y[c] = _ghs_t(v[c] - sp, D, b)
        y[a] = -_ghs_t(np.array(sp - lp), D, b) - _ghs_dt(sp - lp, D, b) * (lp - v[a])
        y[d] = _ghs_t(np.array(hp - sp), D, b) + _ghs_dt(hp - sp, D, b) * (v[d] - hp)
        return y

    ends = raw(np.array([0.0, 1.0]))
    y = (raw(x) - ends[0]) / max(ends[1] - ends[0], 1e-12)
    return np.clip(y, 0, 1).astype(np.float32)


def arcsinh_stretch(p: np.ndarray, beta: float, black: list[float]) -> np.ndarray:
    """保色的 arcsinh 拉伸：用亮度決定倍率，三色乘同一個倍率（星點不會拉到發白，顏色留著）。"""
    lin = np.stack([np.maximum(p[c] - np.float32(black[c]), 0) for c in range(len(p))])
    L = lin.mean(axis=0)
    beta = max(1e-3, float(beta))
    scale = np.where(L > 1e-9, np.arcsinh(beta * L) / (np.maximum(L, 1e-9) * np.arcsinh(beta)), 0)
    out = lin * scale[None].astype(np.float32)
    # 超過 1 的部分按比例壓回（保留色相），不要單色截斷
    peak = out.max(axis=0)
    over = peak > 1
    if over.any():
        out[:, over] /= peak[over]
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------- 色彩空間


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055).astype(np.float32)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1i = np.linalg.inv(_M1).astype(np.float32)
_M2i = np.linalg.inv(_M2).astype(np.float32)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """顯示值（sRGB 編碼，0–1）(3, H, W) → OKLab (3, H, W)。"""
    lin = srgb_to_linear(rgb)
    lms = np.tensordot(_M1, lin, axes=1)
    lms = np.cbrt(np.maximum(lms, 0))
    return np.tensordot(_M2, lms, axes=1).astype(np.float32)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lms = np.tensordot(_M2i, lab, axes=1)
    lin = np.tensordot(_M1i, lms ** 3, axes=1)
    return linear_to_srgb(lin)


def oklab_hue(rgb_pixel: tuple[float, float, float]) -> tuple[float, float]:
    """一個像素的 OKLCh（色相角度、彩度），給滴管用。"""
    lab = rgb_to_oklab(np.array(rgb_pixel, np.float32).reshape(3, 1, 1))[:, 0, 0]
    return float(math.degrees(math.atan2(lab[2], lab[1])) % 360), float(math.hypot(lab[1], lab[2]))


def hue_rgb(hue_deg: float) -> np.ndarray:
    """OKLCh 色相方向在 RGB 上的單位偏移（三色平均 0），給色輪用。"""
    h = math.radians(hue_deg)
    lab = np.array([0.7, 0.1 * math.cos(h), 0.1 * math.sin(h)], np.float32).reshape(3, 1, 1)
    gray = np.array([0.7, 0.0, 0.0], np.float32).reshape(3, 1, 1)
    d = (oklab_to_rgb(lab) - oklab_to_rgb(gray))[:, 0, 0]
    d -= d.mean()
    n = float(np.abs(d).max()) or 1.0
    return (d / n).astype(np.float32)


def angular_weight(hue: np.ndarray, center: float, width: float, feather: float) -> np.ndarray:
    """色相離 center 在 ±width/2 內為 1，再往外 feather 度平滑降到 0。"""
    d = np.abs(((hue - center + 180) % 360) - 180)
    half = width / 2
    return (1 - smoothstep(half, half + max(feather, 1e-3), d)).astype(np.float32)
