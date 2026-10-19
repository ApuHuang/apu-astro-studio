"""拉伸：線性 → 非線性。

- 自動：APU Processing 用 17 組參考圖校準的拉伸（依剩下的噪聲決定拉多深），最省事
- GHS：廣義雙曲線拉伸，可以指定要強化哪一段亮度（對稱點）、保護暗部與亮部
- 保色（arcsinh）：三色乘同一個倍率，星點拉亮也不會變白、變大（星點層用這個）
- STF：PixInsight 螢幕拉伸那種「中間調＋黑點」
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from apu_processing import stretch as pstretch

from .. import ops
from . import DENOISE_SIGMA_FACTOR, T, Ctx, Tool, choice, register, slider, toggle


class Stretch(Tool):
    kind = "stretch"
    title = T("拉伸", "Stretch")
    help = T("把線性的影像拉成看得到的亮度。無星圖用「自動」或「GHS」，星點層用「保色」。拉伸之後才能調色、做曲線。",
             "Turns linear data into a visible image. Use Auto or GHS for the starless layer and Colour-preserving "
             "for the stars layer. Colour grading and curves come after stretching.")
    icon = "stretch"
    maskable = False

    def defaults(self) -> dict:
        return {"method": "auto", "strength": 0.5, "background": 0.13, "linked": True, "bg_color": 0.5,
                "ghs_d": 6.0, "ghs_b": 3.0, "ghs_sp_auto": True, "ghs_sp": 0.0, "ghs_lp": 0.0, "ghs_hp": 1.0,
                "ghs_color": "channels", "beta": 60.0, "stf_target": 0.25}

    def controls(self) -> list[dict]:
        m = lambda name: (lambda p: p.get("method", "auto") == name)  # noqa: E731
        return [
            choice("method", T("方法", "Method"), [("auto", T("自動", "Auto")), ("ghs", T("GHS", "GHS")),
                                                  ("arcsinh", T("保色", "Colour-preserving")),
                                                  ("stf", T("STF", "STF"))]),
            slider("strength", T("拉伸強度", "Strength"), 0, 1, 0.01, "{:.0%}", show=m("auto")),
            slider("background", T("天空亮度", "Sky brightness"), 0.02, 0.30, 0.005, "{:.1%}", show=m("auto")),
            slider("bg_color", T("背景保留色彩", "Keep background colour"), 0, 1, 0.05, "{:.0%}", show=m("auto")),
            toggle("linked", T("三色連動", "Linked channels"), show=lambda p: p.get("method") in ("auto", "stf")),
            slider("ghs_d", T("拉伸量 ln(D+1)", "Stretch ln(D+1)"), 0, 12, 0.1, "{:.1f}", show=m("ghs")),
            slider("ghs_b", T("局部強度 b", "Local intensity b"), -5, 15, 0.1, "{:.1f}", show=m("ghs"),
                   help=T("數字大：對稱點附近集中拉開、其他地方較少；數字小：整體比較平均。",
                          "Higher concentrates contrast around the symmetry point; lower spreads it out.")),
            toggle("ghs_sp_auto", T("對稱點放在背景峰值", "Symmetry point at background peak"), show=m("ghs")),
            slider("ghs_sp", T("對稱點", "Symmetry point"), 0, 1, 0.001, "{:.3f}",
                   show=lambda p: p.get("method") == "ghs" and not p.get("ghs_sp_auto", True)),
            slider("ghs_lp", T("暗部保護", "Shadow protection"), 0, 1, 0.001, "{:.3f}", show=m("ghs")),
            slider("ghs_hp", T("亮部保護", "Highlight protection"), 0, 1, 0.005, "{:.3f}", show=m("ghs")),
            choice("ghs_color", T("色彩", "Colour"), [("channels", T("各色版", "Per channel")),
                                                     ("preserve", T("保色", "Preserve colour"))], show=m("ghs")),
            slider("beta", T("拉伸量", "Stretch"), 1, 400, 1, "{:.0f}", show=m("arcsinh")),
            slider("stf_target", T("目標背景", "Target background"), 0.05, 0.4, 0.01, "{:.2f}", show=m("stf")),
        ]

    def out_phase(self, linear: bool) -> str:
        return "nonlinear"

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        method = params.get("method", "auto")
        p = img
        whole = ctx.whole(img)       # 拉伸參數從整張量（預覽區時用整張的縮圖）
        if method == "auto":
            noise = ops.measure_noise(whole[0] if whole.shape[0] == 1 else whole, ctx.whole_scale)
            # 降噪的影響抵銷掉（見 tools/__init__.py 的 DENOISE_SIGMA_FACTOR）：開不開 AI、降噪開多強，
            # 拉伸的深淺都一樣，只是乾不乾淨
            notes = (ctx.ref.notes if ctx.ref is not None else ctx.notes).get(ctx.layer, {})
            rl, rc = notes.get("denoise_ratio", (1.0, [1.0] * len(noise.channel_sigma)))
            sl = noise.sigma * rl * DENOISE_SIGMA_FACTOR
            sc = tuple(v * r * DENOISE_SIGMA_FACTOR for v, r in zip(noise.channel_sigma, rc))
            # 暗處微弱訊號的量是「差值／噪聲」，跟著換成同一個噪聲（不然背景色、黑點會偏）
            noise = dataclasses.replace(noise, sigma=sl, channel_sigma=sc,
                                        faint_amount=noise.faint_amount * noise.sigma / max(sl, 1e-12))
            s = pstretch.StretchSettings(background=float(params.get("background", 0.13)),
                                         strength=float(params.get("strength", 0.5)),
                                         linked=bool(params.get("linked", True)),
                                         background_color=float(params.get("bg_color", 0.5)))
            out = pstretch.apply(p[0] if p.shape[0] == 1 else p, s, noise)
            return ops.planes(np.asarray(out, np.float32))
        if method == "stf":
            return ops.apply_stf(p, ops.stf_params(whole, float(params.get("stf_target", 0.25))),
                                 linked=bool(params.get("linked", True)))
        L = ops.luminance(whole)
        med, mad = ops.robust_stats(L)
        black = max(0.0, med - 2.8 * mad)
        if method == "arcsinh":
            chans = [max(0.0, ops.robust_stats(c)[0] - 2.8 * ops.robust_stats(c)[1]) for c in whole]
            # 拉伸量滑桿：1～400 → β 1～2e5（線性資料的星點層數值很小）
            beta = 10 ** (math.log10(max(1.0, float(params.get("beta", 60)))) * 1.3 + 0.5)
            return ops.arcsinh_stretch(p, beta, chans)
        # GHS
        Lw = np.clip((L - black) / max(1 - black, 1e-6), 0, 1)
        sp = float(np.median(Lw[:: 4, :: 4])) if params.get("ghs_sp_auto", True) else float(params.get("ghs_sp", 0))
        ctx.view["ghs_sp"] = sp
        x = np.clip((p - black) / max(1 - black, 1e-6), 0, 1)
        Ln = np.clip((ops.luminance(p) - black) / max(1 - black, 1e-6), 0, 1)
        args = (float(params.get("ghs_d", 6.0)), float(params.get("ghs_b", 3.0)), sp,
                min(float(params.get("ghs_lp", 0.0)), sp), max(float(params.get("ghs_hp", 1.0)), sp))
        if params.get("ghs_color") == "preserve" and p.shape[0] >= 3:
            Ly = ops.ghs(Ln, *args)
            ratio = np.where(Ln > 1e-7, Ly / np.maximum(Ln, 1e-7), 0)
            out = x * ratio[None]
            peak = out.max(axis=0)
            over = peak > 1
            if over.any():
                out[:, over] /= peak[over]
            return np.clip(out, 0, 1).astype(np.float32)
        return np.stack([ops.ghs(c, *args) for c in x])


register(Stretch())
