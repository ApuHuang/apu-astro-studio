"""準備 AI 降噪（Noise2Noise）的訓練資料。

同一組對齊好的 sub（WBPP 的 registered/*_r.xisf）隨機分成不重疊的兩半 H1、H2：
A_k ＝ H1 前 k 張的平均，B_k ＝ H2 前 k 張的平均 → 同一片天空、兩份獨立的噪聲，不需要「乾淨的答案」。
k 取 4、8、16、一半，涵蓋不同噪聲程度。每張 sub 只讀一次（同時加進所有包含它的子集合）。
驗證組另外存 truth＝其餘所有 sub 的平均（幾乎沒有噪聲），用來客觀比較降噪結果。

python tools/ai/make_n2n.py --out output/ai/n2n [--holdout M81]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from apu_astro.studio import xisf

SETS = [
    r"D:\Astro\Fuji\registered\Light_BIN-1_7752x5178_EXPOSURE-20.00s_FILTER-NoFilter_RGB",
    r"D:\Astro\VDB152\test\registered\Light_BIN-1_5544x3684_EXPOSURE-300.00s_FILTER-H_mono",
    r"E:\AstroPhotography\NGC6960\Takahashi130&183mc\WBPP\registered\Light_BIN-1_5496x3672_EXPOSURE-300.00s_FILTER-NoFilter_RGB",
    r"E:\AstroPhotography\RedCat\WBPP\registered\Light_BIN-1_6024x4024_EXPOSURE-120.00s_FILTER-NoFilter_RGB",
    r"E:\AstroPhotography\RedCat\WBPP\registered\Light_BIN-1_6024x4024_EXPOSURE-174.00s_FILTER-NoFilter_RGB",
    r"E:\AstroPhotography\RedCat\WBPP\registered\Light_BIN-1_6024x4024_EXPOSURE-302.00s_FILTER-NoFilter_RGB",
    r"E:\AstroPhotography\SH2-119\ALLWBPP\registered\Light_BIN-1_6024x4024_EXPOSURE-151.00s_FILTER-NoFilter_RGB",
    r"E:\AstroPhotography\SH2-119\ALLWBPP\registered\Light_BIN-1_6024x4024_EXPOSURE-302.00s_FILTER-NoFilter_RGB",
    r"E:\AstroPhotography\深空檔\M81\WBPP\registered\Light_BIN-1_5496x3672_EXPOSURE-300.00s_FILTER-NoFilter_RGB",
]
LEVELS = (4, 8, 16)


def set_name(folder: Path) -> str:
    parts = folder.parts
    target = next((p for p in reversed(parts[:-1]) if p not in ("registered", "WBPP", "ALLWBPP", "test")), parts[-2])
    exp = re.search(r"EXPOSURE-([\d.]+)s", folder.name)
    return re.sub(r"[^\w.-]+", "_", f"{target}_{exp.group(1) if exp else ''}s")


def make(folder: Path, out: Path, holdout: bool, seed: int = 7, deep: bool = False) -> None:
    """deep：深答案配對（A_4／A_8／A_16 用前 16 張，truth＝其餘全部，拿來訓練；至少 40 張）。"""
    files = sorted(folder.glob("*_r.xisf"))
    if len(files) < (2 * max(LEVELS) + 8 if deep else 16):
        print(f"skip {folder} ({len(files)} subs)")
        return
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(files)))
    half = len(files) // 2
    h1, h2 = order[:half], order[half:2 * half]
    levels = [k for k in LEVELS if k < half] + [half]
    if deep:
        levels = list(LEVELS)
        h1, h2 = order[:max(LEVELS)], []
    name = set_name(folder)
    dst = out / name
    if (dst / "meta.json").is_file():
        print(f"done already: {name}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    sums: dict[str, np.ndarray] = {}
    valid = None
    t0 = time.time()
    truth_from = (order[max(LEVELS):] if deep else h1[levels[0]:] + h2 + order[2 * half:]) if holdout or deep else []
    for n, idx in enumerate(order):
        data = xisf.to_unit(xisf.read(files[idx]))
        p = data if data.ndim == 3 else data[None]
        ok = np.all(p != 0, axis=0)
        valid = ok if valid is None else (valid & ok)
        targets = []
        if idx in h1:
            pos = h1.index(idx)
            targets += [f"A_{k}" for k in levels if pos < k]
        if idx in h2:
            pos = h2.index(idx)
            targets += [f"B_{k}" for k in levels if pos < k]
        if (holdout or deep) and idx in truth_from:
            targets.append("truth")
        for t in targets:
            if t not in sums:
                sums[t] = np.zeros(p.shape, np.float32)
            sums[t] += p
        print(f"\r{name}: {n + 1}/{len(files)} {time.time() - t0:.0f}s", end="", flush=True)
    print()
    ys, xs = np.nonzero(valid)
    y0, y1, x0, x1 = ys.min() + 8, ys.max() - 8, xs.min() + 8, xs.max() - 8
    counts = {f"A_{k}": k for k in levels} | ({} if deep else {f"B_{k}": k for k in levels})
    if holdout or deep:
        counts["truth"] = len(truth_from)
    for t, s in sums.items():
        arr = (s[:, y0:y1, x0:x1] / counts[t]).astype(np.float16)
        np.save(dst / f"{t}.npy", arr)
    meta = {"source": str(folder), "subs": len(files), "levels": levels, "shape": list(arr.shape),
            "holdout": holdout, "deep": deep, "crop": [int(y0), int(y1), int(x0), int(x1)], "counts": counts}
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {name} {arr.shape} levels={levels} in {time.time() - t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ai/n2n")
    ap.add_argument("--holdout", default="M81")
    ap.add_argument("--deep", action="store_true", help="做深答案配對（張數 40 以上的組；驗證組照舊）")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        folder = Path(s)
        if not folder.is_dir():
            print(f"missing {folder}")
            continue
        holdout = a.holdout.lower() in str(folder).lower()
        if a.deep and holdout:
            continue
        make(folder, out, holdout=holdout, deep=a.deep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
