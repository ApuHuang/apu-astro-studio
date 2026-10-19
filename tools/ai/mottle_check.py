"""斑駁（大顆的低頻噪聲）檢查：用實拍的驗證組（M81：A_4＝4 張 sub 疊的，truth＝另外 20 張的平均）比較幾種降噪。

- sky：天空的誤差（A_4 的噪聲 σ 單位；truth 本身還有約 0.45 的噪聲，誤差不會比這個低）
- m4／m8：誤差模糊 4／8 px、再扣掉 32 px 以上的部分後還剩多少（斑駁；A_4 跟 truth 是不同的 sub，
  背景梯度本來就不一樣，不扣的話每種方法都一樣大）
- bright：最亮 15% 的誤差（星系、星點有沒有被改壞；seeing 不同，數字本來就大，看相對變化）
- faint：truth 裡 3～15σ 的暗星，結果的光度／不降噪的光度（小孔徑；seeing 不同，所以不跟 truth 比）
- chroma：天空的色彩誤差（各色版減三色平均、模糊 2 px；彩色小點、色斑）
- 並排圖：同一個拉伸（用 truth 量的參數），天空區域放大 2 倍

python tools/ai/mottle_check.py [--out output/ai/mottle] [--variants raw ai ai+dn coarse tool]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from apu_processing import detail as pdetail
from apu_processing import imageio
from scipy import ndimage

from apu_astro.studio import ops
from apu_astro.studio.tools import Ctx, ai, get

DATA = Path("output/ai/n2n/M81_300.00s")


def block_mean(x: np.ndarray, f: int) -> np.ndarray:
    h, w = (x.shape[0] // f) * f, (x.shape[1] // f) * f
    return x[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def upsample(c: np.ndarray, f: int, shape: tuple[int, int]) -> np.ndarray:
    """block_mean 的反向（雙線性，像素中心對齊）。"""
    h, w = shape
    yy = np.clip((np.arange(h) + 0.5) / f - 0.5, 0, c.shape[0] - 1)
    xx = np.clip((np.arange(w) + 0.5) / f - 0.5, 0, c.shape[1] - 1)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(c, [Y, X], order=1).astype(np.float32)


def multiscale(planes: list[np.ndarray], factors: tuple[int, ...], amount: float = 0.85) -> list[np.ndarray]:
    """雛形：完整解析度的 AI 之後，縮小 f 倍再跑一次，只取修正量放大加回去（亮處照完整解析度的亮度淡出）。"""
    _fixed, cur, _ = ai.denoise_planes(planes, amount)
    for f in factors:
        w_full = ai.bright_fade(np.mean(cur, axis=0))
        nxt = []
        for x in cur:
            d = block_mean(x, f)
            sky, sig = ai.sky_sigma(d)
            pred = ai.denoise_plane((d - sky) / sig) * sig + sky
            nxt.append(x + upsample(pred - d, f, x.shape) * w_full * amount)
        cur = nxt
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ai/mottle")
    ap.add_argument("--variants", nargs="*", default=["raw", "ai", "ai+dn", "ms2", "ms24"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = np.load(DATA / "truth.npy").astype(np.float32)
    A = np.load(DATA / "A_4.npy").astype(np.float32)
    C, h, w = A.shape
    box = (slice(h // 2 - 700, h // 2 + 700), slice(w // 2 - 1000, w // 2 + 1000))
    A, T = A[:, box[0], box[1]], truth[:, box[0], box[1]]
    units = [ai.sky_sigma(A[c]) for c in range(C)]
    sig = np.array([g for _, g in units], np.float32)[:, None, None]
    Tl = (T / sig).mean(0)
    sky = Tl < np.quantile(Tl, 0.5)
    bright = Tl > np.quantile(Tl, 0.85)
    stars = pdetail.find_stars(Tl, 5.0)
    bgT = ndimage.median_filter(Tl, 25)
    faint = [(int(round(y)), int(round(x)), max(2, int(round(r * 1.5)))) for y, x, r in zip(stars.y, stars.x, stars.radius)
             if 16 <= y < Tl.shape[0] - 16 and 16 <= x < Tl.shape[1] - 16 and 3 <= Tl[int(round(y)), int(round(x))] - bgT[int(round(y)), int(round(x))] < 15]

    def flux(img: np.ndarray, y: int, x: int, r: int) -> float:
        bg = float(np.median(img[y - 12:y + 13, x - 12:x + 13]))
        return float((img[y - r:y + r + 1, x - r:x + r + 1] - bg).sum())

    results = {}
    for name in a.variants:
        t0 = time.time()
        xs = [(A[c] - s) / g for c, (s, g) in enumerate(units)]
        if name == "raw":
            o = xs
        elif name == "ai":
            o = ai.denoise_planes(xs, 0.85)[1]
        elif name in ("ai+dn", "tool+dn"):
            o = ai.denoise_planes(xs, 0.85, coarse=ai.COARSE if name == "tool+dn" else ())[1]
            img = np.stack([v * g + s for v, (s, g) in zip(o, units)])
            img = get("denoise").apply(img, {"amount": 0.5, "luminance": 1.0, "chroma": 1.0}, Ctx(1.0), True)
            o = [(img[c] - s) / g for c, (s, g) in enumerate(units)]
        elif name == "tool":         # 工作室的 AI 降噪（含去斑那層與色彩降噪）
            o = ai.denoise_planes(xs, 0.85, coarse=ai.COARSE, chroma=ai.CHROMA_SIGMA)[1]
        elif name == "coarse":       # 只有去斑那層
            o = ai.denoise_planes(xs, 0.85, coarse=ai.COARSE)[1]
        elif name.startswith("ms"):
            o = multiscale(xs, tuple(int(c) for c in name[2:]))
        else:
            raise SystemExit(f"unknown variant {name}")
        res = np.stack([v * g + s for v, (s, g) in zip(o, units)])
        results[name] = res
        err = ((res - T) / sig).mean(0)
        big = ndimage.gaussian_filter(err, 32)
        line = (f"{name:6s} {time.time() - t0:5.1f}s  sky {np.sqrt(np.mean((err - big)[sky] ** 2)):.3f}"
                f"  m4 {np.sqrt(np.mean((ndimage.gaussian_filter(err, 4) - big)[sky] ** 2)):.3f}"
                f"  m8 {np.sqrt(np.mean((ndimage.gaussian_filter(err, 8) - big)[sky] ** 2)):.3f}"
                f"  bright {np.sqrt(np.mean(err[bright] ** 2)):.3f}")
        if C >= 3:   # 色彩誤差（各色版減三色平均，模糊 2 px）：彩色小點、色斑
            ce = (res - res.mean(0)) / sig - (T - T.mean(0)) / sig
            line += f"  chroma {np.sqrt(np.mean(np.stack([ndimage.gaussian_filter(e, 2) for e in ce])[:, sky] ** 2)):.3f}"
        L = (res / sig).mean(0)
        if name == "raw":
            raw_flux = [flux(L, y, x, r) for y, x, r in faint]
        fr = [flux(L, y, x, r) / f0 for (y, x, r), f0 in zip(faint, raw_flux) if f0 > 0]
        fr = [v for v in fr if np.isfinite(v)]
        if fr:
            line += f"  faint {len(fr)}: {np.median(fr):.2f} (p10 {np.quantile(fr, .1):.2f})"
        print(line, flush=True)
    # 並排圖：天空最多的 200×300，同一個拉伸（背景各自對齊：A_4 跟 truth 的天空亮度不一樣）
    d = ndimage.uniform_filter(sky.astype(np.float32), (200, 300))
    cy, cx = np.unravel_index(int(np.argmax(d[100:-100, 150:-150])), d[100:-100, 150:-150].shape)
    sl = (slice(cy, cy + 200), slice(cx, cx + 300))

    def disp(v: np.ndarray) -> np.ndarray:
        v = v[:, sl[0], sl[1]]
        return np.clip(np.arcsinh((v - np.median(v, axis=(1, 2), keepdims=True)) / (sig * 0.25) + 1.5)
                       / np.arcsinh(150), 0, 1)
    tiles = [disp(r) for r in list(results.values()) + [T]]
    strip = np.concatenate(tiles, axis=2)
    strip = np.repeat(np.repeat(strip, 2, 1), 2, 2)
    imageio.save_display(out / "M81_sky.png", strip.astype(np.float32))
    print("tiles:", " | ".join(list(results) + ["truth"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
