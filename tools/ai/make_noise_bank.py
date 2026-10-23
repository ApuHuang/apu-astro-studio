"""真實噪聲庫：從實拍配對（output/ai/n2n 的 A_k／B_k）切出純噪聲，給訓練加到 master 上做「有準確答案」的配對。

A_k 跟 B_k 是同一片天空、不同 sub 疊的，相減後星雲、星點抵銷，(A−B)/√2 就是 k 張疊圖的噪聲：
相鄰像素的相關（去馬賽克、對齊插值）、斑駁、暗點、疊圖軟體的質感都是真的（模擬噪聲只有模糊過的高斯）。
- 避開亮星：兩半的星點有細微差異（seeing、對齊），相減會留殘影（約峰值的 3%）。用「比局部背景亮多少」判斷
  （扣掉約 70 px 的中位數背景），不用整張的天空：有梯度、大片星雲的圖，或張數多（噪聲小、換算成 σ 起伏很大）的時候，
  用整張天空當基準幾乎切不到（A7C、2242 一塊都沒有）。背景亮一點沒關係：每塊各自換成自己的噪聲單位。
  張數多的層級噪聲小，星點換算成 σ 都很亮，256 px 的塊裡幾乎一定有亮星 → 不整塊丟掉，星點的位置（比局部背景亮 6σ、
  外擴 3 px）換成同一張別處的噪聲（噪聲各處統計一樣，看不出接縫），星點佔一塊 MAX_STAR_FRAC 以內才用
- 每塊各自換成相鄰差 σ＝1 的單位（推論時就是這樣量噪聲的）
- 不切有「沒資料」（0）的地方：Photons 疊的圖旋轉、中天翻轉後有效範圍不是方的，配對只裁到外框，四個角還有 0；
  切到的話那塊噪聲量出來接近 0，換算後數值爆掉（第十四版第一次訓練損失 10¹¹）
- 驗證組（holdout）不切

python tools/ai/make_noise_bank.py [--data output/ai/n2n] [--out output/ai/noise_bank] [--tiles 24]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from train_n2n import robust_sky_sigma  # noqa: E402

TILE = 256
STAR_LEVEL = 6.0       # 比局部背景亮這麼多倍噪聲的算星點（相減會留殘影），外擴 3 px 後換成別處的噪聲
MAX_STAR_FRAC = 0.15   # 星點佔一塊最多這個比例


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/ai/n2n")
    ap.add_argument("--out", default="output/ai/noise_bank")
    ap.add_argument("--tiles", type=int, default=24, help="每組、每個張數、每個色版最多切幾塊")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    index = []
    for meta_path in sorted(Path(a.data).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("holdout") or meta.get("deep"):
            continue
        folder = meta_path.parent
        tiles, info = [], []
        for k in meta["levels"]:
            A = np.load(folder / f"A_{k}.npy", mmap_mode="r")
            B = np.load(folder / f"B_{k}.npy", mmap_mode="r")
            C, H, W = A.shape
            for c in range(C):
                a_c = np.asarray(A[c], np.float32)
                b_c = np.asarray(B[c], np.float32)
                sky, sig = robust_sky_sigma(a_c)
                level = ((a_c + b_c) / 2 - sky) / sig
                bg = ndimage.median_filter(level[::8, ::8], 9)          # 局部背景（約 70 px）
                bg = ndimage.zoom(bg, (H / bg.shape[0], W / bg.shape[1]), order=1)[:H, :W]
                star = ndimage.binary_dilation(level - bg > STAR_LEVEL, iterations=3)
                empty = (a_c == 0) | (b_c == 0)                      # 沒資料的角落
                noise = (a_c - b_c) / np.sqrt(2)
                got, tries = 0, 0
                while got < a.tiles and tries < a.tiles * 40:
                    tries += 1
                    y0, x0 = int(rng.integers(0, H - TILE)), int(rng.integers(0, W - TILE))
                    m = star[y0:y0 + TILE, x0:x0 + TILE].copy()
                    if m.mean() > MAX_STAR_FRAC or empty[y0:y0 + TILE, x0:x0 + TILE].any():
                        continue
                    n = noise[y0:y0 + TILE, x0:x0 + TILE].copy()
                    for _ in range(6):                 # 星點的位置換成別處（也不是星點）的噪聲
                        if not m.any():
                            break
                        y1, x1 = int(rng.integers(0, H - TILE)), int(rng.integers(0, W - TILE))
                        m2 = star[y1:y1 + TILE, x1:x1 + TILE] | empty[y1:y1 + TILE, x1:x1 + TILE]
                        use = m & ~m2
                        n[use] = noise[y1:y1 + TILE, x1:x1 + TILE][use]
                        m &= ~use
                    if m.any():                        # 剩下極少數：從這塊不是星點的像素抽
                        pool = n[~star[y0:y0 + TILE, x0:x0 + TILE]]
                        n[m] = rng.choice(pool, int(m.sum()))
                    s = robust_sky_sigma(n)[1]
                    n = (n - np.median(n)) / s
                    if not np.isfinite(n).all():
                        continue
                    tiles.append(n.astype(np.float16))
                    info.append([int(k), int(c)])
                    got += 1
        if not tiles:
            print(f"{folder.name}: no sky tiles")
            continue
        np.save(out / f"{folder.name}.npy", np.stack(tiles))
        index.append({"name": folder.name, "count": len(tiles), "stacker": meta.get("stacker", "wbpp"),
                      "channels": meta["shape"][0], "tiles": info})
        print(f"{folder.name:28s} {len(tiles):5d} tiles", flush=True)
    (out / "meta.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print("total", sum(i["count"] for i in index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
