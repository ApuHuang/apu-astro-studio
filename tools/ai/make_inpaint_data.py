"""去星「仿製填滿」的訓練資料：master 的亮度切小塊，附上星點遮罩（跟去星一樣的找法）和「不能挖洞」的範圍。

訓練時在不能挖洞以外的地方挖洞（形狀、大小照星點遮罩），原圖就是答案（自監督，不需要任何人的無星圖）；
原本的星點也遮起來（跟實際去星時一樣），但不算損失。
- 亮度＝各色版平均（去星時 AI 只補亮度，顏色用平滑內插），換成天空 σ 單位
- 星點：pdetail.find_stars（5σ）→ recenter_stars → star_like → star_footprints（外擴 2 px），跟 StarSplit 一樣
- 不能挖洞：星點遮罩外擴 4 px、比局部背景亮 5σ 的地方外擴 3 px（沒找到的暗星、星暈）
- 評估用的影像（EVAL_MASTERS）另外存成驗證組 val_*.npy

python tools/ai/make_inpaint_data.py [--out output/ai/inpaint] [--crops 64]
（用 .venv：要讀 XISF、用去星的函式）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from apu_processing import detail as pdetail
from scipy import ndimage

from apu_astro.studio import io
from apu_astro.studio.tools import ai
from apu_astro.studio.tools import stars as tstars

CROP = 320
EVAL_MASTERS = ("LBN576", "IC434", "M31", "M16", "NGC2244", "NGC6888", "M81")


def prepare(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img, _ = io.load_image(path)
    p = img if img.ndim == 3 else img[None]
    L = np.nan_to_num(np.mean(p, axis=0).astype(np.float32))
    sky, sig = ai.sky_sigma(L)
    Ln = (L - sky) / sig
    st = pdetail.find_stars(Ln, 5.0)
    st = tstars.recenter_stars(Ln, st, 1.0)
    st = tstars.keep_stars(st, tstars.star_like(Ln, st, 1.0))
    mask, radii = tstars.star_footprints(Ln, st, 1.0, 2.0)
    small = Ln[::4, ::4]
    bg = ndimage.zoom(ndimage.median_filter(small, 9), (Ln.shape[0] / small.shape[0], Ln.shape[1] / small.shape[1]),
                      order=1)[:Ln.shape[0], :Ln.shape[1]]
    keepout = ndimage.binary_dilation(mask, iterations=4) | ndimage.binary_dilation(Ln - bg > 5.0, iterations=3)
    return Ln, mask, keepout, radii


def crops(Ln, mask, keepout, n: int, rng: np.random.Generator):
    H, W = Ln.shape
    bright = ndimage.uniform_filter(Ln[::8, ::8], 5)
    ys, xs = np.nonzero(bright > np.quantile(bright, 0.8))
    out = []
    for i in range(n):
        if i < n * 0.3 and len(ys):                      # 三成挑亮的地方（星雲），補洞最難的是有結構的背景
            j = int(rng.integers(len(ys)))
            y0 = int(np.clip(ys[j] * 8 - CROP // 2, 0, H - CROP))
            x0 = int(np.clip(xs[j] * 8 - CROP // 2, 0, W - CROP))
        else:
            y0, x0 = int(rng.integers(0, H - CROP)), int(rng.integers(0, W - CROP))
        sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        out.append((Ln[sl], mask[sl], keepout[sl]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ai/inpaint")
    ap.add_argument("--crops", type=int, default=64)
    ap.add_argument("--synth-meta", default="output/ai/synth/meta.json", help="master 清單（跟降噪的模擬配對同一份）")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    masters = json.loads(Path(a.synth_meta).read_text(encoding="utf-8"))
    index, all_radii = [], []
    for m in masters:
        val = m["holdout"] or any(t.lower() in m["path"].lower() for t in EVAL_MASTERS)
        name = ("val_" if val else "") + m["name"].replace("val_", "")
        if (out / f"{name}.npz").is_file():
            index.append({"name": name, "val": val})
            continue
        t0 = time.time()
        try:
            Ln, mask, keepout, radii = prepare(m["path"])
        except Exception as e:  # noqa: BLE001  單張讀不到不中斷
            print(f"{name}: failed {e}", flush=True)
            continue
        cs = crops(Ln, mask, keepout, a.crops, rng)
        np.savez(out / f"{name}.npz", L=np.clip(np.stack([c[0] for c in cs]), -6e4, 6e4).astype(np.float16),
                 mask=np.stack([c[1] for c in cs]), keepout=np.stack([c[2] for c in cs]))
        if not val:
            all_radii.append(radii)
        index.append({"name": name, "val": val})
        print(f"{name:32s} {len(radii):6d} stars, free {np.mean(~keepout) * 100:4.0f}%  {time.time() - t0:.0f}s",
              flush=True)
    if all_radii:
        np.save(out / "radii.npy", np.concatenate(all_radii).astype(np.float32))
    (out / "meta.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
