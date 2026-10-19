"""細節：反摺積（拉伸前）與銳化／局部對比（拉伸後）。"""

from __future__ import annotations

import numpy as np
from apu_processing import denoise as pdenoise
from apu_processing import detail as pdetail
from scipy import ndimage

from .. import backend, ops, psf
from . import T, Ctx, Tool, register, section, slider, toggle

# ---------------------------------------------------------------------- 反摺積


class Deconvolution(Tool):
    kind = "deconv"
    title = T("反摺積", "Deconvolution")
    help = T("從畫面裡的星點量出模糊的形狀，再把它還原（Richardson–Lucy）。畫面分區量星點的橢圓（傾斜、像差讓角落的星"
             "拉長），每區用自己的形狀還原，星點會變圓、全畫面一樣大。飽和星核不參與計算，不會長出甜甜圈；"
             "暗的背景自動保護，不會把噪聲變尖。要在拉伸前、去星前做。",
             "Measures the shape of the blur from the stars and reverses it (Richardson–Lucy). The star shape is "
             "measured per zone (tilt or aberrations elongate corner stars) and each zone is restored with its own "
             "shape, so stars become round and even across the field. Saturated cores are left out (no donuts) and "
             "the faint background is protected. Use before stretching and before star removal.")
    phase = "linear"
    icon = "deconv"
    auto_layer = "main"

    def defaults(self) -> dict:
        return {"iterations": 20, "psf": 0.0, "round": True, "stars": 0.6, "nebula": 0.5, "guard": 0.6,
                "protect": True}

    def controls(self) -> list[dict]:
        return [
            slider("iterations", T("強度（迭代次數）", "Strength (iterations)"), 0, 60, 1, "{:.0f}"),
            slider("psf", T("模糊大小（0＝自動量測）", "Blur size (0 = measured)"), 0, 8, 0.1,
                   "{:.1f} px", help=T("星點核心的半高寬。自動時從畫面裡的星量（分區、含橢圓）。",
                                       "Star core FWHM; measured from the image per zone (including elongation) when 0.")),
            toggle("round", T("讓星點變圓", "Make stars round"),
                   help=T("用量到的橢圓還原，拉長的星（傾斜、像差、追蹤）變圓。關掉的話只用圓形的模糊（舊的做法）。",
                          "Restores with the measured elliptical shape so elongated stars become round. Off uses a "
                          "round blur only (previous behaviour).")),
            section(T("強度分配", "Amounts")),
            slider("stars", T("星點", "Stars"), 0, 1, 0.05, "{:.0%}"),
            slider("nebula", T("星雲", "Nebula"), 0, 1, 0.05, "{:.0%}"),
            slider("guard", T("暗處保護", "Faint-area guard"), 0, 1, 0.05, "{:.0%}",
                   help=T("訊號弱（接近背景）的地方少做，噪聲不會被變尖。", "Applies less where the signal is near the "
                                                                     "background so noise isn't sharpened.")),
            toggle("protect", T("保護大星與飽和星", "Protect big and saturated stars")),
        ]

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        iters = int(params.get("iterations", 20))
        if iters <= 0:
            return img
        L = img.mean(axis=0) if img.shape[0] > 1 else img[0]
        key = ("stars", L.shape)
        stars = ctx.cache.get(key)
        if stars is None:
            stars = pdetail.find_stars(L)
            ctx.cache[key] = stars
        ctx.check()
        round_at, sharp_at = self._kernels(L, stars, params, ctx)
        # 飽和星核：不參與比對（往外擴 2 px），結果也保持原值
        whole = ctx.whole(img)
        tops = [max(float(img[c].max()), float(whole[c].max())) for c in range(img.shape[0])]
        sat = psf.saturation_mask(img, tops=tops)
        valid = ~ndimage.binary_dilation(sat, iterations=2) if sat.any() else None
        sky, sig = ops.robust_stats(L)
        if ctx.ref is not None:      # 預覽區：天空從整張量（這一塊可能整片都是星雲）
            sky = ops.robust_stats(ops.luminance(ctx.whole(img)) if img.shape[0] > 1 else ctx.whole(img)[0])[0]
        pedestal = sky - 5 * max(sig, 1e-9)
        tile, pad = max(256, int(1024 / ctx.scale)), max(24, int(64 / ctx.scale))
        W = stars.weight.astype(np.float32)
        nebula = float(params.get("nebula", 0.5))
        guard = float(params.get("guard", 0.6))
        faint = np.ones_like(L)
        if guard > 0:
            snr = (ops.blur(L, 1.5) - sky) / max(sig * 0.6, 1e-9)
            faint = (1 - guard) + guard * ops.smoothstep(1.0, 5.0, snr)
        keep = 1 - ops.blur((~valid).astype(np.float32), 1.0) if valid is not None else 1.0
        L0 = L
        if round_at is not None:
            # 第一段：變圓。核只有長軸方向多出來的那部分，星點上全部套用（只把長軸縮到跟短軸一樣，星不會變小）；
            # 星雲照星雲強度、暗處照暗處保護（不然背景噪聲會被拉成一條條）
            x = np.maximum(L - pedestal, 1e-6).astype(np.float32)
            R = psf.tiled_rl(x, round_at, 20, valid, tile=tile, pad=pad, cancel=ctx.cancel) + pedestal
            ctx.check()
            a_round = (W + (1 - W) * nebula * faint) * keep
            L = L + (R - L) * a_round
        # 第二段：銳化（圓形核，短軸的 0.8 倍）
        x = np.maximum(L - pedestal, 1e-6).astype(np.float32)
        D = psf.tiled_rl(x, sharp_at, iters, valid, tile=tile, pad=pad, cancel=ctx.cancel) + pedestal
        ctx.check()
        amount = (W * float(params.get("stars", 0.6)) + (1 - W) * nebula) * faint * keep
        if params.get("protect", True) and len(stars.y):
            lab, n = ndimage.label(stars.mask)
            cy = np.clip(np.round(stars.y).astype(int), 0, L.shape[0] - 1)
            cx = np.clip(np.round(stars.x).astype(int), 0, L.shape[1] - 1)
            big = (stars.radius > 6.0 / max(ctx.scale, 1e-6)) | stars.saturated | sat[cy, cx]
            if big.any() and n:
                bad = np.zeros(n + 1, bool)
                bad[lab[cy[big], cx[big]]] = True
                bad[0] = False
                region = ndimage.binary_dilation(bad[lab], iterations=2)
                amount *= 1 - ops.blur(region.astype(np.float32), 2.0)
        newL = L + (D - L) * amount
        # 防振鈴：不准比局部背景低 1σ 以上（原圖本來就比較暗的地方不准更暗）。反摺積會在星點旁邊挖出一圈暗環
        # （疊圖時的插值振鈴本來就有一點，LBN576 校色後 −2σ，反摺積後變 −4σ），AI 降噪後很明顯，
        # 去星時補的地方也會被暗環拉暗變成暗洞。星點內部、星雲照常銳化（都比背景亮）
        from .stars import _background
        floor = np.minimum(L0, _background(L0) - sig)
        newL = np.maximum(newL, floor)
        delta = newL - L0
        if img.shape[0] == 1:
            return (img + delta[None]).astype(np.float32)
        # 彩色：三色按原本的比例分到亮度的變化（星色不變）；接近背景的地方三色同加
        ratio = np.where(L0[None] > sky + 3 * sig, np.clip(img / np.maximum(L0[None], 1e-9), 0, 3), 1.0)
        return (img + delta[None] * ratio).astype(np.float32)

    def _kernels(self, L: np.ndarray, stars, params: dict, ctx: Ctx):
        """回傳 (round_at, sharp_at)：位置 → 這個解析度用的核。round_at 是「變圓」那段（沒有就 None）。"""
        manual = float(params.get("psf", 0.0)) / ctx.scale
        if manual > 0.05:
            k = psf.elliptical_kernel(np.eye(2) * max(0.3, manual / 2.3548) ** 2)
            ctx.view["psf_fwhm_full"] = manual * ctx.scale
            return None, (lambda y, x: k)
        provider = ctx.meta.get("source_psf")
        grid = provider() if provider is not None else None
        if grid is None or grid.overall is None:
            key = ("fwhm", L.shape)
            fwhm = ctx.cache.get(key) or pdetail.measure_core_fwhm(L, stars) * pdetail.PSF_CORE_LIMIT
            ctx.cache[key] = fwhm
            ctx.view["psf_fwhm_full"] = fwhm * ctx.scale
            k = psf.elliptical_kernel(np.eye(2) * max(0.3, fwhm / 2.3548) ** 2)
            return None, (lambda y, x: k)
        oy, ox = ctx.origin[1], ctx.origin[0]
        s2 = ctx.scale ** 2
        covs: dict = {}

        def cov(y: float, x: float) -> np.ndarray:
            key = (round(y), round(x))
            if key not in covs:
                c = psf.cov_in_frame(grid, oy + y * ctx.scale, ox + x * ctx.scale, ctx.frame_ops)
                covs[key] = (grid.overall.cov() if c is None else c) / s2
            return covs[key]
        cells = [c for row in grid.cells for c in row if c is not None]
        ctx.view["psf_fwhm_full"] = round(float(np.median([c.fwhm_minor for c in cells])) * pdetail.PSF_CORE_LIMIT, 2)
        ctx.view["psf_ellipticity_max"] = round(max(c.ellipticity for c in cells), 2)
        round_at = (lambda y, x: psf.round_kernel(cov(y, x))) if params.get("round", True) else None
        return round_at, (lambda y, x: psf.sharpen_kernel(cov(y, x), pdetail.PSF_CORE_LIMIT))


register(Deconvolution())

# ---------------------------------------------------------------------- 銳化（局部對比）


def _garrote(w: np.ndarray, t: float) -> np.ndarray:
    t2 = t * t
    return np.where(np.abs(w) > t, w - t2 / np.where(w == 0, 1, w), 0).astype(np.float32)


class Sharpen(Tool):
    kind = "sharpen"
    title = T("銳化與局部對比", "Sharpen & Local Contrast")
    help = T("細節：1～2 px 的細結構；結構：4～16 px 的星雲層次；高光壓縮：壓住太亮的核心（例如獵戶座中心）。"
             "只加強超過噪聲門檻的結構，噪聲不會被放大。",
             "Detail boosts 1–2 px structure, Structure boosts 4–16 px nebula layers, and Highlight compression "
             "tames over-bright cores. Only structure above the noise threshold is boosted.")
    phase = "nonlinear"
    icon = "sharpen"

    def defaults(self) -> dict:
        return {"detail": 0.4, "structure": 0.3, "hdr": 0.0, "threshold": 2.0, "protect_stars": True}

    def controls(self) -> list[dict]:
        return [
            slider("detail", T("細節", "Detail"), 0, 2, 0.05, "{:.2f}"),
            slider("structure", T("結構", "Structure"), 0, 2, 0.05, "{:.2f}"),
            slider("hdr", T("高光壓縮", "Highlight compression"), 0, 1, 0.05, "{:.0%}"),
            slider("threshold", T("噪聲門檻", "Noise threshold"), 0, 5, 0.1, "{:.1f} σ"),
            toggle("protect_stars", T("不加強星點", "Leave stars alone")),
        ]

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        p = np.clip(img, 0, 1) if not linear else img
        L = ops.luminance(p) if p.shape[0] >= 3 else p[0]
        scales = 5
        coeffs, smooth = pdenoise.starlet(L, scales)
        ctx.check()
        gains = [float(params.get("detail", 0.4))] * 2 + [float(params.get("structure", 0.3))] * 3
        k = float(params.get("threshold", 2.0))
        boost = np.zeros_like(L)
        for j, w in enumerate(coeffs[:scales]):
            g = gains[j] if j < len(gains) else 0.0
            if g <= 0:
                continue
            _m, s = ops.robust_stats(w)
            boost += g * _garrote(w, k * s)
        if params.get("protect_stars", True):
            key = ("starmask", L.shape)
            sw = ctx.cache.get(key)
            if sw is None:
                sw = pdetail.find_stars(L).weight.astype(np.float32)
                ctx.cache[key] = sw
            boost *= 1 - sw
        hdr = float(params.get("hdr", 0.0))
        newL = L + boost
        if hdr > 0:
            base = smooth + coeffs[-1] if len(coeffs) > scales else smooth
            if ctx.ref is not None:
                wl = ctx.whole(p)
                q = float(np.quantile((wl.mean(axis=0) if wl.shape[0] > 1 else wl[0])[:: 2, :: 2], 0.75))
            else:
                q = float(np.quantile(base[:: 4, :: 4], 0.75))
            newL = newL - hdr * 0.8 * np.maximum(base - q, 0)
        if p.shape[0] == 1:
            return np.clip(newL, 0, 1)[None].astype(np.float32) if not linear else newL[None].astype(np.float32)
        ratio = np.where(L > 1e-4, newL / np.maximum(L, 1e-4), 1.0)
        out = p * ratio[None]
        return (np.clip(out, 0, 1) if not linear else out).astype(np.float32)


register(Sharpen())
