"""比較降噪：驗證組（沒參加訓練）的 A_k 分別「不降噪／Processing 的降噪（各種強度，取最好的）／AI 降噪」，
跟 truth（其餘所有 sub 的平均，幾乎沒有噪聲）比 RMSE（噪聲 σ 單位）；天空與亮處分開算，另外存並排的裁切圖。

.venv\\Scripts\\python tools/ai/compare.py --data output/ai/n2n --out output/ai/compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from apu_processing import denoise as pdenoise
from apu_processing import imageio
from apu_processing import stretch as pstretch

from apu_astro.studio.tools import ai


def rmse(a: np.ndarray, b: np.ndarray, m: np.ndarray | None = None) -> float:
    d = (a - b) ** 2
    return float(np.sqrt(d[m].mean() if m is not None else d.mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/ai/n2n")
    ap.add_argument("--out", default="output/ai/compare")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for meta_path in Path(a.data).glob("*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("holdout"):
            continue
        folder = meta_path.parent
        truth = np.load(folder / "truth.npy").astype(np.float32)
        C, h, w = truth.shape
        box = (slice(h // 2 - 500, h // 2 + 500), slice(w // 2 - 750, w // 2 + 750))
        t = truth[:, box[0], box[1]]
        bright = t.mean(0) > np.quantile(t.mean(0), 0.85)
        sky = t.mean(0) < np.quantile(t.mean(0), 0.5)
        for k in meta["levels"][:3]:
            A = np.load(folder / f"A_{k}.npy").astype(np.float32)[:, box[0], box[1]]
            sig = np.array([ai.sky_sigma(A[c])[1] for c in range(C)], np.float32)[:, None, None]
            res = {"raw": A}
            best = None
            for amount in (0.3, 0.5, 0.7, 0.85, 1.0):
                s = pdenoise.DenoiseSettings(amount=amount)
                d = np.asarray(pdenoise.reduce(A, s), np.float32)
                e = rmse(d / sig, t / sig)
                if best is None or e < best[0]:
                    best = (e, amount, d)
            res[f"processing {best[1]:.0%}"] = best[2]
            if ai.available():
                d = np.empty_like(A)
                for c in range(C):
                    sk, sg = ai.sky_sigma(A[c])
                    d[c] = ai.denoise_plane((A[c] - sk) / sg) * sg + sk
                res["ai"] = d
            row = {}
            for name, img in res.items():
                row[name] = {"all": round(rmse(img / sig, t / sig), 3),
                             "sky": round(rmse(img / sig, t / sig, np.broadcast_to(sky, img.shape)), 3),
                             "bright": round(rmse(img / sig, t / sig, np.broadcast_to(bright, img.shape)), 3)}
            report[f"{folder.name} k={k}"] = row
            print(f"{folder.name} k={k}:", json.dumps(row, ensure_ascii=False), flush=True)
            # 並排圖：同一個拉伸（用 truth 量的參數）
            noise = pstretch.measure(t)
            tiles = [pstretch.apply(img, pstretch.StretchSettings(), noise) for img in list(res.values()) + [t]]
            crop = (slice(300, 700), slice(500, 1000))
            strip = np.concatenate([np.asarray(x)[:, crop[0], crop[1]] for x in tiles], axis=2)
            imageio.save_display(out / f"{folder.name}_k{k}.png", strip)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
