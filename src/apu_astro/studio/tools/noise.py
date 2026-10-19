"""降噪：多尺度收縮（APU Processing 的 starlet 降噪），亮度與色彩噪聲分開調。拉伸前、拉伸後都可以用。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from apu_processing import denoise as pdenoise

from .. import ops
from . import T, Ctx, State, Tool, record_denoise_ratio, register, slider


class Denoise(Tool):
    kind = "denoise"
    title = T("降噪", "Noise Reduction")
    help = T("依畫面量到的噪聲大小自動決定門檻，強的結構幾乎不受影響。亮度噪聲（顆粒）與色彩噪聲（彩色雜點）分開調。"
             "去星後只對無星圖做效果最好。",
             "Thresholds come from the measured noise, so strong structure is barely touched. Luminance (grain) "
             "and colour noise are set separately. Works best on the starless layer.")
    icon = "denoise"

    def defaults(self) -> dict:
        return {"amount": 0.5, "luminance": 1.0, "chroma": 1.0}

    def controls(self) -> list[dict]:
        return [
            slider("amount", T("強度", "Amount"), 0, 1, 0.05, "{:.0%}"),
            slider("luminance", T("亮度噪聲", "Luminance noise"), 0.3, 2.5, 0.05, "{:.2f}×"),
            slider("chroma", T("色彩噪聲", "Colour noise"), 0.3, 2.5, 0.05, "{:.2f}×"),
        ]

    def run(self, state: State, params: dict, layer: str, mask: dict | None, ctx: Ctx) -> State:
        out = super().run(state, params, layer, mask, ctx)
        if float(params.get("amount", 0.5)) > 0:
            record_denoise_ratio(state, out, self.target_layers(state, layer), ctx)
        return out

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        amount = float(params.get("amount", 0.5))
        if amount <= 0:
            return img
        base = pdenoise.DenoiseSettings()
        s = replace(base, amount=amount,
                    k_luminance=tuple(k * float(params.get("luminance", 1.0)) for k in base.k_luminance),
                    k_chroma=tuple(k * float(params.get("chroma", 1.0)) for k in base.k_chroma))
        src = img[0] if img.shape[0] == 1 else img
        out = pdenoise.reduce(src, s, ctx.cancel)
        out = np.asarray(out, np.float32)[None] if img.shape[0] == 1 else np.asarray(out, np.float32)
        return _limit_change(img, out, ctx.scale)


CHANGE_LIMIT = 4.0   # 降噪每個像素最多改幾倍當地噪聲（見 _limit_change）


def _limit_change(before: np.ndarray, after: np.ndarray, scale: float) -> np.ndarray:
    """降噪只准改「噪聲那麼大」的量：每個像素最多 ±CHANGE_LIMIT 倍當地噪聲（亮的地方照光子噪聲放大，
    跟 AI 降噪的 s＝√(1＋亮度/16) 一樣）。Processing 的多尺度降噪在亮星外圍 3～4 px 把星光削掉一截
    （LBN576：12σ → 0.3σ，旁邊又墊高 3σ），看起來每顆亮星一圈黑環。以前只對無星圖做所以沒看過；
    不拔星之後作用在整張上就很明顯。改超過幾倍噪聲的已經是在改結構，不是去噪。"""
    from .stars import _background
    out = np.empty_like(after)
    for c in range(before.shape[0]):
        sig = ops.pixel_noise(before[c])
        level = np.maximum(before[c] - _background(before[c]), 0) / max(sig, 1e-12)
        lim = CHANGE_LIMIT * sig * np.sqrt(1.0 + level / 16.0)
        out[c] = before[c] + np.clip(after[c] - before[c], -lim, lim)
    return out


register(Denoise())
