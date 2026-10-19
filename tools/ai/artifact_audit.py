"""AI 降噪的人造瑕疵檢查：多張不同類型的 master 整塊跑一次（跟後製工作室一樣的處理：補暗點 → AI → 亮處淡出），
自動量六種瑕疵，另外存 1:1 的裁切圖給人看。

1. 大範圍改動：修正量模糊 16 px 後的大小（應該接近 0；整片變亮、變暗、變色都會在這裡）
2. 頂到上限的比例：修正量接近 ±LIMIT·s（模型在那裡沒把握）
3. 分塊接縫：輸入錯開 256 px 再算一次，跟原本的差
4. 新的離群點：AI 之後才出現、比周圍（5×5 中位數）差 4σ 以上的點（原本不是離群點）
5. 彩色相機的色斑：三色修正量的差，模糊 8 px 後的大小
6. 裁切圖：輸入｜輸出｜修正量（±3σ），每張 master 取天空、暗淡星雲、亮處各一塊

python tools/ai/artifact_audit.py --out output/ai/audit [--model 路徑]
數字的單位都是天空噪聲 σ。門檻（超過就標 !）：大範圍 0.15、上限 0.5%、接縫 0.3、新離群點 0.01%、色斑 0.15。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from apu_processing import imageio
from scipy import ndimage

from apu_astro.studio import io
from apu_astro.studio.tools import ai

MASTERS = {
    "NGC2244_OSC": r"D:\Astro\NGC2244\WBPP\master\masterLight_BIN-1_5496x3672_EXPOSURE-300.00s_FILTER-NoFilter_RGB.xisf",
    "NGC6888_Ha": r"E:\AstroPhotography\1130803 NGC6888 ASI183MC\183m\WBPP\master"
                  r"\masterLight_BIN-1_5544x3684_EXPOSURE-300.00s_FILTER-Ha_mono_autocrop.xisf",
    "M81_OSC": r"E:\AstroPhotography\深空檔\M81\WBPP\master"
               r"\masterLight_BIN-1_5496x3672_EXPOSURE-300.00s_FILTER-NoFilter_RGB.xisf",
    "M16_OIII": r"E:\AstroPhotography\深空檔\M16-SHO\masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_FILTER-O_mono.xisf",
    "M16_SII": r"E:\AstroPhotography\深空檔\M16-SHO\masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_FILTER-S_mono.xisf",
    "M31_OSC": r"E:\AstroPhotography\M31\M31 WBPP\master"
               r"\masterLight_BIN-1_4920x3276_EXPOSURE-300.00s_FILTER-NoFilter_RGB.xisf",
    "Fuji_wide": r"D:\Astro\Fuji\master\masterLight_BIN-1_7752x5178_EXPOSURE-20.00s_FILTER-NoFilter_RGB.xisf",
    "IC434_OSC": r"E:\AstroPhotography\IC434\WBPP\master"
                 r"\masterLight_BIN-1_6024x4024_EXPOSURE-302.00s_FILTER-NoFilter_RGB.xisf",
}
CROP = 3000
LIMITS = {"lowfreq": 0.15, "saturated_pct": 0.5, "seam": 0.3, "new_outliers_pct": 0.01, "chroma": 0.15}
MODEL_LIMIT = 6.0


def find(path: str) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    hits = list(p.parent.parent.rglob(p.name)) if p.parent.parent.exists() else []
    return hits[0] if hits else None





def stretch(v: np.ndarray) -> np.ndarray:
    return np.clip(np.arcsinh(np.maximum(v, -3) / 2) / np.arcsinh(80 / 2), 0, 1)


COARSE: tuple[int, ...] = ()   # --coarse：跟工作室一樣加縮小一半那層


def audit(name: str, path: Path, out: Path) -> dict:
    img, _header = io.load_image(path)
    p = img if img.ndim == 3 else img[None]
    C, h, w = p.shape
    y0, x0 = max(0, (h - CROP) // 2), max(0, (w - CROP) // 2)
    p = p[:, y0:y0 + CROP, x0:x0 + CROP]
    rep = {}
    lowfreq, sat, seam, created, where = [], [], [], [], []
    lf_maps = []
    planes = []
    for c in range(C):
        sky, sig = ai.sky_sigma(p[c])
        planes.append((p[c] - sky) / sig)
    xs, outs, _n = ai.denoise_planes(planes, coarse=COARSE)   # 跟後製工作室的 AI 降噪一樣（強度 100%）
    for xf, o in zip(xs, outs):
        fix = o - xf
        lf = ndimage.gaussian_filter(fix, 16)
        lf_maps.append(lf)
        iy, ix = np.unravel_index(int(np.argmax(np.abs(lf))), lf.shape)
        lowfreq.append(float(abs(lf[iy, ix])))
        where.append({"y": int(iy), "x": int(ix), "edge_px": int(min(iy, ix, lf.shape[0] - 1 - iy, lf.shape[1] - 1 - ix)),
                      "level": round(float(ndimage.uniform_filter(xf, 32)[iy, ix]), 1),
                      "max_nearby": round(float(xf[max(0, iy - 48):iy + 48, max(0, ix - 48):ix + 48].max()), 0)})
        s = np.sqrt(1 + ndimage.uniform_filter(np.maximum(xf, 0), 5) / 16)
        pred_raw = ai.denoise_plane(xf)
        sat.append(float((np.abs(xf - pred_raw) > 0.95 * MODEL_LIMIT * s).mean() * 100))
        # 接縫：錯開 256 px 再算；錯開後的圖左上是新的影像邊界，比較時排除四周 160 px
        shifted = ai.denoise_plane(xf[256:, 256:])
        dd = np.abs(shifted - pred_raw[256:, 256:])[160:-160, 160:-160]
        seam.append(float(np.quantile(dd, 0.99999)))
        d_in = xf - ndimage.median_filter(xf, 5)
        d_out = o - ndimage.median_filter(o, 5)
        created.append(float(((np.abs(d_out) > 4) & (np.abs(d_in) < 2)).mean() * 100))
    rep["lowfreq"] = round(max(lowfreq), 3)
    rep["saturated_pct"] = round(max(sat), 3)
    rep["seam"] = round(max(seam), 3)
    rep["new_outliers_pct"] = round(max(created), 4)
    rep["lowfreq_at"] = where[int(np.argmax(lowfreq))]
    lfm = np.max(np.abs(np.stack(lf_maps)), axis=0)[::8, ::8]
    imageio.save_display(out / f"{name}_lowfreq.png", np.repeat(np.clip(lfm / 0.3, 0, 1)[None], 3, 0).astype(np.float32))
    if C >= 3:
        fixes = np.stack([o - x for o, x in zip(outs, xs)])
        chroma = fixes - fixes.mean(axis=0, keepdims=True)
        rep["chroma"] = round(float(max(np.abs(ndimage.gaussian_filter(ch, 8)).max() for ch in chroma)), 3)
    rep["flags"] = [k for k, lim in LIMITS.items() if rep.get(k, 0) > lim]
    # 裁切圖：天空（最暗）、暗淡星雲（中等）、亮處（最亮）的 400×400
    L = np.mean(xs, axis=0)
    blocks = ndimage.uniform_filter(L, 200)[200:-200:50, 200:-200:50]
    picks = {}
    order = np.argsort(blocks.ravel())
    for label, q in (("sky", 0.05), ("faint", 0.6), ("bright", 0.995)):
        i = order[int(q * (len(order) - 1))]
        by, bx = np.unravel_index(i, blocks.shape)
        picks[label] = (200 + by * 50 - 200, 200 + bx * 50 - 200)
    rows = []
    for label, (cy, cx) in picks.items():
        sl = (slice(cy, cy + 400), slice(cx, cx + 400))
        inp = np.stack([stretch(x[sl]) for x in xs])
        res = np.stack([stretch(o[sl]) for o in outs])
        fx = np.stack([np.clip((o[sl] - x[sl]) / 6 + 0.5, 0, 1) for o, x in zip(outs, xs)])
        rows.append(np.concatenate([inp, res, fx], axis=2))
    sheet = np.concatenate(rows, axis=1).astype(np.float32)
    imageio.save_display(out / f"{name}.png", sheet if sheet.shape[0] == 3 else np.repeat(sheet[:1], 3, 0))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/ai/audit")
    ap.add_argument("--model", default=None)
    ap.add_argument("--coarse", action="store_true", help="加縮小一半再跑的那層（ai.COARSE）")
    a = ap.parse_args()
    if a.model:
        ai.MODEL = Path(a.model)
    if a.coarse:
        global COARSE
        COARSE = ai.COARSE
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, path in MASTERS.items():
        p = find(path)
        if p is None:
            print(f"{name}: not found", flush=True)
            continue
        t0 = time.time()
        rep = audit(name, p, out)
        report[name] = rep
        flag = ("  ! " + ", ".join(rep["flags"])) if rep["flags"] else ""
        print(f"{name:12s} {time.time() - t0:5.0f}s  " + "  ".join(f"{k} {v}" for k, v in rep.items()
                                                               if k not in ("flags", "lowfreq_at"))
              + f"  lowfreq@{rep['lowfreq_at']}"
              + flag, flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    bad = [n for n, r in report.items() if r["flags"]]
    print("PASS" if not bad else f"FLAGGED: {bad}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
