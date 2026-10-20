"""檢查 Photons 配到的 dark／flat 能不能用（做實拍配對之前；使用者：「有些 dark 可能過期」「flat 也要看一下配對」）。

同一組 light 抽幾張，用 Photons 的校正（calibrate：扣 dark 或 bias、除 flat、單張熱像素）做三種：
完整、不扣 dark（改扣 bias，沒有就不扣）、不除 flat，比較：
- cold：比同色 3×3 中位數暗 6σ 的單點（每百萬像素）。過期的 dark 在已經不熱的位置扣太多，留下一個個洞；
  沒有抖動（dither）的話洞在每張的同一個天空位置，配對的兩半都有 → 模型學到「孤立的暗點是真的」
- hot：校正後還留著的亮單點（亮 6σ）
- noise：相鄰同色像素差的噪聲（dark 錯的話會變大）
- bg：天空亮度的起伏（64 px 方塊取 25% 分位數、扣掉平面後的標準差，ADU）；flat 對的話會變小，dark 不合
  （溫度不同的 amp glow、熱訊號）會變大。用絕對值：不扣 dark／bias 時天空含相機的黑階，用百分比的話分母變大、
  看起來比較平（A7C 一開始就被這個騙了）
- dither：各張對參考的位移（px）；太小的話感光元件上固定的瑕疵在兩半是同一個位置

python tools/ai/calib_check.py --inputs "light 資料夾" "dark 資料夾" "flat 資料夾" [--object X] [--exposure 300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from apu_photons import backend
from apu_photons.calibration import COLD_FLAT, MasterBuilder, Masters, _plane_views, calibrate, median3
from apu_photons.imageio import load_image
from apu_photons.ingest import expand_inputs, ingest
from apu_photons.register import register
from apu_photons.stars import detect

sys.path.insert(0, str(Path(__file__).parent))
from make_n2n_photons import choose_lights  # noqa: E402

SAMPLES = 3


def cfa_stats(data: np.ndarray, bayer: str | None) -> dict:
    cold = hot = 0
    sig = []
    for sy, sx in _plane_views(data.shape, bayer):
        plane = data[sy, sx]
        d = plane - median3(plane)
        s = 1.4826 * float(np.median(np.abs(d[::3, ::3])))
        sig.append(s)
        cold += int((d < -6 * s).sum())
        hot += int((d > 6 * s).sum())
    lum = data.reshape(data.shape[0] // 2, 2, data.shape[1] // 2, 2).mean(axis=(1, 3)) if bayer else data
    b = 32 if bayer else 64
    h, w = (lum.shape[0] // b) * b, (lum.shape[1] // b) * b
    blocks = np.quantile(lum[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(h // b, w // b, -1),
                         0.25, axis=2)
    yy, xx = np.mgrid[:blocks.shape[0], :blocks.shape[1]]
    A = np.column_stack([np.ones(blocks.size), yy.ravel(), xx.ravel()])
    coef, *_ = np.linalg.lstsq(A, blocks.ravel(), rcond=None)
    resid = blocks.ravel() - A @ coef
    level = float(np.median(blocks))
    return {"cold": cold / data.size * 1e6, "hot": hot / data.size * 1e6, "noise": float(np.mean(sig)),
            "bg": float(np.std(resid)), "level": level}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--object", default=None)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--cache", default="output/ai/photons_cache/_check")
    ap.add_argument("--flat-any-night", action="store_true", help="沒有同一晚的 flat 時用日期最近那晚的（Photons 的選項）")
    a = ap.parse_args()
    paths = expand_inputs([Path(p) for p in a.inputs])
    key, lights, calib, _trains = choose_lights(paths, a.object, a.exposure)
    warnings: list = []
    project = ingest(lights + calib, warnings, flat_any_night=a.flat_any_night)
    be = backend.select("auto")
    builder = MasterBuilder(project.calibration_sets, Path(a.cache), warnings, log=lambda m: None, be=be, workers=4)
    by_group: dict[str, list] = {}
    for f in project.lights:
        by_group.setdefault(f.calib["group"], []).append(f)
    print(f"{key}: {len(lights)} lights in {len(by_group)} calibration group(s)")
    for gk, frames in by_group.items():
        entry = project.calib_groups[gk]
        sets = {k: entry.get(k) for k in ("dark", "bias", "flat")}
        print(f"  group {gk}: {len(frames)} lights, dark={sets['dark']} bias={sets['bias']} flat={sets['flat']}")
        for k in ("dark", "bias", "flat"):
            sid = sets[k]
            if sid:
                cs = project.calibration_sets[sid]
                dates = sorted({(f.date_obs or "")[:10] for f in cs.frames})
                temps = sorted({f.header.get("CCD-TEMP") for f in cs.frames if f.header.get("CCD-TEMP") is not None})
                print(f"    {k}: {len(cs.frames)} files, dates {dates[:1]}..{dates[-1:]}, temp {temps[:1]}..{temps[-1:]}")
        ldates = sorted({(f.date_obs or "")[:10] for f in frames})
        ltemps = sorted({f.header.get("CCD-TEMP") for f in frames if f.header.get("CCD-TEMP") is not None})
        print(f"    lights: dates {ldates[:1]}..{ldates[-1:]}, temp {ltemps[:1]}..{ltemps[-1:]}")
        bayer, shape = frames[0].bayer, frames[0].shape
        full = builder.masters(entry, bayer, shape)
        flat_bad = full.flat < COLD_FLAT if full.flat is not None else None
        variants = {"full": full}
        if full.dark is not None:
            variants["no dark"] = Masters(bias=full.bias, dark=None, flat=full.flat, bad=flat_bad)
        if full.flat is not None:
            bad = None if full.bad is None else (full.bad & ~flat_bad if flat_bad is not None else full.bad)
            variants["no flat"] = Masters(bias=full.bias, dark=full.dark, flat=None, bad=bad)
        if len(variants) == 1:
            variants["(only bias/none)"] = full
        idx = np.linspace(0, len(frames) - 1, min(SAMPLES, len(frames))).round().astype(int)
        res: dict[str, list] = {k: [] for k in variants}
        for i in idx:
            raw, _ = load_image(frames[i].path)
            for name, m in variants.items():
                cal, _n = calibrate(raw, m, bayer)
                res[name].append(cfa_stats(cal, bayer))
        for name, rows in res.items():
            avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            print(f"    {name:9s} cold {avg['cold']:7.1f}/Mpx  hot {avg['hot']:6.1f}/Mpx  noise {avg['noise']:8.2f}"
                  f"  bg {avg['bg']:8.2f} (sky {avg['level']:.0f})")
        # 抖動：抽樣的幾張對第一張的位移
        stars = []
        for i in idx:
            raw, _ = load_image(frames[i].path)
            cal, _n = calibrate(raw, full, bayer)
            stars.append(detect(cal, bayer))
        ref = np.column_stack([stars[0].x, stars[0].y])
        shifts = []
        for st in stars[1:]:
            try:
                m = register(np.column_stack([st.x, st.y]), ref)[0]
                shifts.append(float(np.hypot(m[0, 2], m[1, 2])))
            except Exception:  # noqa: BLE001
                shifts.append(float("nan"))
        print(f"    shift vs first sample: {', '.join(f'{s:.1f}' for s in shifts)} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
