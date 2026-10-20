"""AI 降噪的品質檢查（模擬：master 當答案，加上新的噪聲）。

- 天空：AI 輸出跟答案的誤差（噪聲 σ 單位；加噪的輸入大約是 1）
- 斑駁：誤差模糊 4 px 後還剩多少（只修細顆粒、留下一塊塊的低頻噪聲時會偏大）
- 暗星：答案裡比較暗的星（峰值 3～15σ），AI 輸出的光度／答案；小孔徑（星點範圍）與 15×15 大孔徑
  （大孔徑也小於 1 ＝ 星被當噪聲吃掉；只有小孔徑小於 1 ＝ 被抹開）
- 星色（彩色）：顏色比例跟答案差多少，AI vs 加噪的輸入；「變差」＝比輸入差 1.5 倍以上的星的比例
- 並排圖：輸入｜AI｜答案（暗星多的區域，放大 3 倍）

python tools/ai/star_check.py [--model 路徑] [--out 資料夾]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from apu_processing import detail as pdetail
from apu_processing import imageio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import artifact_audit as A  # noqa: E402

from apu_astro.studio import io  # noqa: E402
from apu_astro.studio.tools import ai  # noqa: E402

TARGETS = ("NGC6888_Ha", "M16_OIII", "NGC2244_OSC", "M31_OSC", "IC434_OSC")


def aperture(img: np.ndarray, y: int, x: int, r: int) -> float:
    bg = float(np.median(img[y - 12:y + 13, x - 12:x + 13]))
    return float((img[y - r:y + r + 1, x - r:x + r + 1] - bg).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="output/ai/quality")
    ap.add_argument("--protect", action="store_true")
    ap.add_argument("--coarse", action="store_true", help="跟工作室一樣加去斑那層和色彩降噪")
    ap.add_argument("--processing", action="store_true", help="改用 Processing 的降噪（強度 85%）當對照")
    ap.add_argument("--rgb-model", default=None, help="彩色模型（三色一起看）的 ONNX；給了就用它")
    ap.add_argument("--rho", type=float, default=0.0,
                    help="色版之間噪聲的相關（實拍 OSC 0.4～0.8；0＝各色獨立，跟以前的數字可比）")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS), help="只跑這幾張（預設全部）")
    a = ap.parse_args()
    if a.model:
        ai.MODEL = Path(a.model)
    if a.rgb_model:
        ai.MODEL_RGB = Path(a.rgb_model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for name in a.targets:
        img, _ = io.load_image(A.find(A.MASTERS[name]))
        p = img if img.ndim == 3 else img[None]
        C, h, w = p.shape
        y0, x0 = (h - 1600) // 2, (w - 1600) // 2
        p = p[:, y0:y0 + 1600, x0:x0 + 1600]
        units = [ai.sky_sigma(p[c]) for c in range(C)]
        truth = [(p[c] - s) / g / 4.0 for c, (s, g) in enumerate(units)]
        noisy = []
        def unit_noise() -> np.ndarray:
            n = ndimage.gaussian_filter(rng.standard_normal(truth[0].shape).astype(np.float32), 0.6)
            return n / n.std()
        shared = unit_noise() if a.rho > 0 else 0.0
        for t in truth:
            n = np.sqrt(1 - a.rho) * unit_noise() + np.sqrt(a.rho) * shared
            noisy.append(t + np.sqrt(1 + np.maximum(t, 0) / 50) * n)
        est = [ai.sky_sigma(x)[1] for x in noisy]
        if a.processing:
            from apu_processing import denoise as pdenoise
            stack = np.stack([x * 1e-3 + 0.05 for x in noisy])
            d = np.asarray(pdenoise.reduce(stack if C >= 3 else stack[0], pdenoise.DenoiseSettings(amount=0.85)),
                           np.float32)
            d = d if d.ndim == 3 else d[None]
            outs = [(d[c] - 0.05) / 1e-3 for c in range(C)]
        else:
            _xs, outs, _ = ai.denoise_planes([x / e for x, e in zip(noisy, est)], protect_stars=a.protect,
                                             coarse=ai.COARSE if a.coarse else (),
                                             chroma=ai.CHROMA_SIGMA if a.coarse else 0.0)
            outs = [o * e for o, e in zip(outs, est)]
        T, N, O = np.mean(truth, axis=0), np.mean(noisy, axis=0), np.mean(outs, axis=0)
        sky = T < 2
        sd = np.sqrt(1 + np.maximum(T, 0) / 50)
        res = {"sky_in": float(np.sqrt(np.mean(((N - T) / sd)[sky] ** 2))),
               "sky_ai": float(np.sqrt(np.mean(((O - T) / sd)[sky] ** 2))),
               "mottle_in": float(np.sqrt(np.mean(ndimage.gaussian_filter(N - T, 4)[sky] ** 2))),
               "mottle_ai": float(np.sqrt(np.mean(ndimage.gaussian_filter(O - T, 4)[sky] ** 2)))}
        if C >= 3:   # 天空的色彩誤差（各色減三色平均、模糊 2 px）：彩色小點、色斑
            def chroma(v: list[np.ndarray]) -> float:
                e = np.stack(v) - np.mean(v, axis=0) - (np.stack(truth) - T)
                return float(np.sqrt(np.mean(np.stack([ndimage.gaussian_filter(c, 2) for c in e])[:, sky] ** 2)))
            res["chroma_in"], res["chroma_ai"] = chroma(noisy), chroma(outs)
        # 單點：各色版天空誤差超過 1.5σ 的像素（每百萬）；AI 後誤差大約 0.3σ，超過 1.5 的就是留下來的單點
        res["speck_in"], res["speck_ai"] = (float(np.mean([np.mean(np.abs(v - t)[sky] > 1.5) for v, t in zip(vs, truth)])) * 1e6
                                            for vs in (noisy, outs))
        stars = pdetail.find_stars(T, 3.0)
        small, big, ein, eai = [], [], [], []
        dense = np.zeros_like(T)
        for y, x, r in zip(stars.y, stars.x, stars.radius):
            yy, xx = int(round(y)), int(round(x))
            if not (16 <= yy < 1600 - 16 and 16 <= xx < 1600 - 16):
                continue
            rr = max(2, int(round(r * 1.5)))
            peak = float(T[yy, xx] - np.median(T[yy - 12:yy + 13, xx - 12:xx + 13]))
            if 3 <= peak < 15:
                ft_s, ft_b = aperture(T, yy, xx, rr), aperture(T, yy, xx, 7)
                if ft_s > 0 and ft_b > 0:
                    small.append(aperture(O, yy, xx, rr) / ft_s)
                    big.append(aperture(O, yy, xx, 7) / ft_b)
                    dense[yy, xx] += 1
            if C >= 3 and peak >= 15:
                ft = np.array([aperture(t, yy, xx, rr) for t in truth])
                fn = np.array([aperture(v, yy, xx, rr) for v in noisy])
                fa = np.array([aperture(v, yy, xx, rr) for v in outs])
                if min(ft[1], fn[1], fa[1]) > 0:
                    ct = ft[[0, 2]] / ft[1]
                    ein.append(np.abs((fn[[0, 2]] / fn[1]) / ct - 1).max())
                    eai.append(np.abs((fa[[0, 2]] / fa[1]) / ct - 1).max())
        line = (f"{name:12s} sky {res['sky_in']:.2f}→{res['sky_ai']:.2f}  mottle {res['mottle_in']:.3f}→"
                f"{res['mottle_ai']:.3f}")
        line += f"  specks/Mpx {res['speck_in']:.0f}→{res['speck_ai']:.0f}"
        if "chroma_in" in res:
            line += f"  chroma {res['chroma_in']:.3f}→{res['chroma_ai']:.3f}"
        if small:
            line += (f"  faint stars {len(small)}: flux small {np.median(small):.2f} (p10 {np.quantile(small, .1):.2f})"
                     f" big {np.median(big):.2f} (p10 {np.quantile(big, .1):.2f})")
        if ein:
            ein, eai = np.array(ein), np.array(eai)
            line += (f"  colour err {np.median(ein) * 100:.1f}%→{np.median(eai) * 100:.1f}%"
                     f" worse {np.mean(eai > ein * 1.5 + 0.01) * 100:.1f}%")
        print(line, flush=True)
        # 並排圖：暗星最多的 300×200
        d = ndimage.uniform_filter(dense, (200, 300))
        cy, cx = np.unravel_index(int(np.argmax(d[100:-100, 150:-150])), d[100:-100, 150:-150].shape)
        sl = (slice(cy, cy + 200), slice(cx, cx + 300))

        def disp(v: np.ndarray) -> np.ndarray:
            return np.clip(np.arcsinh(np.maximum(v, -3) / 1.5) / np.arcsinh(40 / 1.5), 0, 1)
        row = np.concatenate([disp(N[sl]), disp(O[sl]), disp(T[sl])], 1)
        row = np.repeat(np.repeat(row, 3, 0), 3, 1)
        imageio.save_display(out / f"{name}.png", np.repeat(row[None], 3, 0).astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
