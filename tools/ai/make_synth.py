"""AI 降噪的第二種訓練資料：現成的 master 當底，訓練時再加兩份獨立的模擬噪聲（見 train_n2n.py 的 Synth）。

真實配對（make_n2n.py）要 WBPP 留下 registered 檔，只有 9 組、單色只有一組；master 有幾十個，
涵蓋單色窄帶、星系、各種星雲。這裡只做「底」：每個 master 換成天空噪聲 σ 單位、切成 256×256 的小塊
（40% 挑亮的地方），存成 float16。噪聲在訓練時才加，每次都不一樣。

測試用的目標（NGC2244、NGC6888、M81）不拿來訓練；NGC6888 Ha 與 NGC2244 另外存成模擬噪聲的驗證組
（底圖就是答案，可以直接量單色窄帶與彩色相機的效果）。

python tools/ai/make_synth.py --out output/ai/synth
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from apu_astro.studio import xisf

ROOTS = [Path(r"D:\Astro"), Path(r"E:\AstroPhotography")]
SKIP = ("autocrop", "drizzle", "fastintegration", "2022ztf", "c2023", "小米銀河", "屋頂派")
TEST = ("ngc2244", "ngc6888", "m81")
VALIDATION = {
    "val_NGC6888_Ha": r"E:\AstroPhotography\1130803 NGC6888 ASI183MC\183m\WBPP\master"
                      r"\masterLight_BIN-1_5544x3684_EXPOSURE-300.00s_FILTER-Ha_mono_autocrop.xisf",
    "val_NGC2244_OSC": r"D:\Astro\NGC2244\WBPP\master"
                       r"\masterLight_BIN-1_5496x3672_EXPOSURE-300.00s_FILTER-NoFilter_RGB.xisf",
}
CROP = 256
PER_MASTER = 48            # 每個 master 每個色版的小塊數
PER_TARGET = 3             # 同一個目標最多幾個 master
BRIGHT_SHARE = 0.4


def sky_sigma(a: np.ndarray) -> tuple[float, float]:
    """跟 train_n2n.robust_sky_sigma、studio/tools/ai.sky_sigma 一樣的定義。"""
    s = a[::4, ::4].astype(np.float32)
    sky = float(np.median(s))
    d = (a[1::2, ::2].astype(np.float32) - a[:-1:2, ::2].astype(np.float32))[::2, ::2]
    return sky, max(float(np.median(np.abs(d - np.median(d)))) * 1.4826 / math.sqrt(2), 1e-9)


def target_of(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 4 and parts[-2].lower() == "master":
        return parts[-4]
    return parts[-2]


def find_masters() -> list[Path]:
    found = []
    for root in ROOTS:
        for p in root.rglob("masterLight*.xisf"):
            low = str(p).lower()
            if any(k in low for k in SKIP) or any(k in low for k in TEST):
                continue
            found.append(p)
    by_target: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(found):
        by_target[target_of(p)].append(p)
    out = []
    for _t, paths in sorted(by_target.items()):
        # 同一個目標：合成好的彩色優先，再來是單色；分開的 R/G/B 只取一個
        paths.sort(key=lambda p: ("__" in p.name, "_mono" not in p.name and "_RGB" not in p.name, p.name))
        seen_split = False
        keep = []
        for p in paths:
            if "__" in p.name:
                if seen_split:
                    continue
                seen_split = True
            keep.append(p)
        out += keep[:PER_TARGET]
    return out


def crops_of(path: Path, rng: np.random.Generator) -> np.ndarray | None:
    data = xisf.to_unit(xisf.read(path))
    p = data if data.ndim == 3 else data[None]
    C, h, w = p.shape
    if h < CROP * 2 or w < CROP * 2:
        return None
    norm = np.empty_like(p, dtype=np.float32)
    for c in range(C):
        sky, sig = sky_sigma(p[c])
        norm[c] = (p[c] - sky) / sig
    valid = np.all(p != 0, axis=0)               # 疊圖邊緣（0）不要
    L = norm.mean(axis=0)
    sub = 8
    Ls, vs = L[::sub, ::sub], valid[::sub, ::sub]
    m = CROP // 2 // sub + 1
    inner = np.zeros_like(vs)
    inner[m:-m, m:-m] = True
    bright = np.argwhere((Ls > 15) & vs & inner) * sub
    anywhere = np.argwhere(vs & inner) * sub
    out = []
    for _ in range(PER_MASTER):
        pts = bright if (len(bright) and rng.random() < BRIGHT_SHARE) else anywhere
        cy, cx = pts[int(rng.integers(len(pts)))]
        y0 = int(np.clip(cy - CROP // 2, 0, h - CROP))
        x0 = int(np.clip(cx - CROP // 2, 0, w - CROP))
        if not valid[y0:y0 + CROP, x0:x0 + CROP].all():
            continue
        c = int(rng.integers(C))
        out.append(norm[c, y0:y0 + CROP, x0:x0 + CROP])
    return np.stack(out).astype(np.float16) if out else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ai/synth")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    masters = find_masters()
    print(f"{len(masters)} masters", flush=True)
    meta = []
    jobs = [(f"m{i:02d}_" + re.sub(r"[^\w.-]+", "_", target_of(p))[:40], p, False) for i, p in enumerate(masters)]
    jobs += [(name, Path(p), True) for name, p in VALIDATION.items()]
    for name, path, holdout in jobs:
        dst = out / f"{name}.npy"
        if dst.is_file():
            arr = np.load(dst, mmap_mode="r")
            meta.append({"name": name, "path": str(path), "n": int(arr.shape[0]), "holdout": holdout})
            continue
        t0 = time.time()
        try:
            arr = crops_of(path, rng)
        except Exception as exc:  # noqa: BLE001  讀不了的檔案跳過
            print(f"skip {path}: {exc}", flush=True)
            continue
        if arr is None:
            print(f"skip {path}: too small", flush=True)
            continue
        np.save(dst, arr)
        meta.append({"name": name, "path": str(path), "n": int(arr.shape[0]), "holdout": holdout})
        print(f"{name}: {arr.shape[0]} crops {time.time() - t0:.1f}s", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", sum(m["n"] for m in meta if not m["holdout"]), "training crops", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
