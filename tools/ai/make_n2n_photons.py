"""用使用者的原始檔（light＋校正檔）做 Noise2Noise 實拍配對：Photons 疊圖流程，只換掉最後的整合。

make_n2n.py 只能用 WBPP 留下的 registered sub，那些已經用完了（剩下的都不到 16 張）。這裡直接用 Photons 當函式庫：
校正（dark／flat／bias、單張熱像素）、找星、對齊、正規化都照 Photons 的做法，到「整合」那一步時改成：
所有 frame 照 Photons 的方式（warp_band：去馬賽克＋對齊＋正規化）搬到參考網格，隨機分成兩半，
A_k＝前一半的前 k 張平均、B_k＝後一半的前 k 張平均（跟 make_n2n.py 同樣的格式），存好就停（不輸出 master）。
不改 Photons 的程式（三個專案獨立開發），只在這個腳本裡把 engine._integrate 換掉。

好處：整合版的使用者也是用 Photons 疊圖，這些配對的噪聲質感（雙線性去馬賽克、三次插值對齊）跟他們實際的一樣。

挑片：light 先用 APU Pick 的量測與評分（預設設定）挑過，淘汰的不用（只在記憶體裡判斷，不搬檔、不寫報表）。
校正檔：先跑 tools/ai/calib_check.py 看 dark／flat 配得對不對（過期的 dark 會留下一個個暗洞，沒抖動的話兩半同一個
位置，模型會學到孤立的暗點是真的）；不能用就加 --no-dark／--no-flat。

python tools/ai/make_n2n_photons.py --name NGC1893_300s --inputs "E:/.../NGC 1893/Light" "E:/.../NGC 1893/Flat"
（資料夾不往下找；同一個資料夾裡有多個目標、曝光時用 --object／--exposure 挑，預設挑張數最多的那組）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from apu_photons import engine
from apu_photons.ingest import expand_inputs, ingest
from apu_photons.integrate import read_patch, warp_band

LEVELS = (4, 8, 16)
MIN_SUBS = 16
BAND = 1024


class PairsDone(engine.Cancelled):
    """配對存好了，停止 Photons 後面的步驟（繼承 Cancelled：run() 不會把它當錯誤寫進紀錄檔）。"""


def _exposure_key(e: float | None) -> int:
    """曝光秒數分組：60 秒以上照 10 秒進位（相機記成 300／301／302、119.7／119.8 的是同一組）。"""
    e = float(e or 0)
    return int(round(e / 10) * 10) if e >= 60 else int(round(e))


def _object(f) -> str:
    h = f.header or {}
    return str(h.get("OBJECT") or f.path.parent.name).strip()


def choose_lights(paths: list[Path], obj: str | None, exposure: float | None, one_train: bool = False):
    """自己先 ingest 一次：挑出要用的 light（同一個目標、同一台相機、同一個曝光、同一個濾鏡），其餘的校正檔照收。"""
    warnings: list = []
    project = ingest(paths, warnings)
    groups = defaultdict(list)
    for f in project.lights:
        cam = (f.camera or "").split("|")[0]
        groups[(_object(f), cam, _exposure_key(f.exposure), f.filter or "")].append(f)
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  group {k}: {len(v)} lights")
    cands = [(k, v) for k, v in groups.items()
             if (obj is None or k[0].casefold() == obj.casefold()) and (exposure is None or k[2] == _exposure_key(exposure))]
    if not cands:
        raise SystemExit("找不到符合的 light")
    key, lights = max(cands, key=lambda kv: len(kv[1]))
    if one_train:            # 只用張數最多的光學系統（變焦鏡頭焦距不同時 Photons 合併後對不齊，例如 2242 的 102／105 mm）
        counts = defaultdict(int)
        for f in lights:
            counts[f.train] += 1
        best = max(counts, key=counts.get)
        lights = [f for f in lights if f.train == best]
    calib = [f.path for f in project.files if f.kind not in ("light", "unknown")]
    trains = sorted({f.train for f in lights if f.train})
    return key, [f.path for f in lights], calib, trains


def pick(lights: list[Path]) -> list[Path]:
    """APU Pick 的預設評分挑片（照 filter／曝光／gain 分組），回傳留下的。"""
    from astro_light_selector.grouping import DEFAULT_GROUP_KEYS
    from astro_light_selector.pipeline import decide, measure_files
    frames = measure_files(lights, workers=max(1, (os.cpu_count() or 2) - 1))
    sel = decide(frames, DEFAULT_GROUP_KEYS, "score")
    keep = [Path(d.metrics.file) for d in sel.decisions if d.keep]
    for d in sel.decisions:
        if not d.keep:
            print(f"    pick rejects {Path(d.metrics.file).name}: {', '.join(str(r) for r in d.reasons)}")
    return keep


def make_pairs(frames, bayer, shape, be, dst: Path, meta: dict, seed: int = 7, holdout: bool = False,
               deep: bool = False) -> None:
    """holdout：驗證組。A_4／A_8／A_16 用前 16 張，truth＝其餘全部的平均（跟 A 不重疊，幾乎沒噪聲）；不存 B。
    deep：切法跟驗證組一樣，但拿來訓練（答案是其餘全部的平均，噪聲只剩兩半互當答案時的 20～40%）。"""
    holdout_like = holdout or deep
    jobs = engine._jobs(frames)
    n = len(jobs)
    if n < MIN_SUBS or (holdout_like and n < 2 * max(LEVELS) + 8):
        raise SystemExit(f"只有 {n} 張通過對齊，不夠用")
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(n)]
    if holdout_like:
        levels = list(LEVELS)
        h1, h2 = order[:max(LEVELS)], order[max(LEVELS):]
    else:
        half = n // 2
        h1, h2 = order[:half], order[half:2 * half]
        levels = [k for k in LEVELS if k < half] + [half]
    h, w = shape
    nch = 3 if bayer else 1
    sums: dict[str, np.ndarray] = {}
    valid = np.ones((h, w), bool)
    unit = None
    t0 = time.time()
    for done, idx in enumerate(h1 + h2):
        job = jobs[idx]
        cal = np.load(job.cal_path, mmap_mode="r")
        img = np.empty((nch, h, w), np.float32)
        for r0 in range(0, h, BAND):
            r1 = min(h, r0 + BAND)
            out = warp_band(be, job, read_patch(job, cal, r0, r1, w), job.pattern(bayer), r0, r1, w, shape)
            img[:, r0:r1] = be.to_numpy(out)
        be.free()
        ok = np.all(np.isfinite(img), axis=0)
        valid &= ok
        if unit is None:                       # 換成 0～1 左右（存 float16；ADU 會超過 float16 的上限）
            unit = 1.0 / max(1.0, float(np.nanmax(img)))
        img = np.nan_to_num(img) * unit
        if idx in h1:
            targets = [f"A_{k}" for k in levels if h1.index(idx) < k]
        elif holdout_like:
            targets = ["truth"]
        else:
            targets = [f"B_{k}" for k in levels if h2.index(idx) < k]
        for t in targets:
            if t not in sums:
                sums[t] = np.zeros((nch, h, w), np.float32)
            sums[t] += img
        print(f"\r  pairs: {done + 1}/{len(h1) + len(h2)} {time.time() - t0:.0f}s", end="", flush=True)
    print()
    ys, xs = np.nonzero(valid)
    y0, y1, x0, x1 = ys.min() + 8, ys.max() - 8, xs.min() + 8, xs.max() - 8
    counts = {f"A_{k}": k for k in levels} | ({"truth": len(h2)} if holdout_like else {f"B_{k}": k for k in levels})
    dst.mkdir(parents=True, exist_ok=True)
    arr = None
    for t, s in sums.items():
        arr = (s[:, y0:y1, x0:x1] / counts[t]).astype(np.float16)
        np.save(dst / f"{t}.npy", arr)
    meta.update({"subs": n, "levels": levels, "shape": list(arr.shape), "holdout": holdout, "deep": deep,
                 "crop": [int(y0), int(y1), int(x0), int(x1)], "counts": counts, "stacker": "photons",
                 "frames": [str(frames[i].path) for i in h1 + h2]})
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  saved {dst.name} {arr.shape} levels={levels} in {time.time() - t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="輸出資料夾名稱（例如 NGC1893_300s）")
    ap.add_argument("--inputs", nargs="+", required=True, help="light 與校正檔的資料夾（不往下找）")
    ap.add_argument("--object", default=None, help="只用 OBJECT（或資料夾名稱）是這個的 light")
    ap.add_argument("--exposure", type=float, default=None, help="只用這個曝光秒數的 light")
    ap.add_argument("--out", default="output/ai/n2n")
    ap.add_argument("--cache", default="output/ai/photons_cache", help="Photons 校正後的暫存（做完可以刪）")
    ap.add_argument("--no-dark", action="store_true", help="不用 dark（calib_check 看出過期、扣太多時）")
    ap.add_argument("--no-flat", action="store_true", help="不用 flat（calib_check 看出配錯、變更不平時）")
    ap.add_argument("--no-pick", action="store_true", help="不先用 APU Pick 挑片")
    ap.add_argument("--holdout", action="store_true", help="做成驗證組（A 跟其餘全部的平均 truth，不參加訓練）")
    ap.add_argument("--deep", action="store_true", help="深答案配對（切法同驗證組，拿來訓練；至少 40 張）")
    ap.add_argument("--keep-cache", action="store_true", help="做完不刪 Photons 校正後的暫存")
    ap.add_argument("--one-train", action="store_true", help="只用張數最多的光學系統（焦距不同的不合併）")
    ap.add_argument("--flat-any-night", action="store_true",
                    help="沒有同一晚的 flat 時用日期最近那晚的（calib_check 看過那張 flat 對其他晚也有用時）")
    a = ap.parse_args()
    dst = Path(a.out) / a.name
    if (dst / "meta.json").is_file():
        print(f"done already: {a.name}")
        return 0
    paths = expand_inputs([Path(p) for p in a.inputs])
    key, lights, calib, trains = choose_lights(paths, a.object, a.exposure, a.one_train)
    print(f"{a.name}: {key} {len(lights)} lights, {len(calib)} calibration files, trains {trains}")
    if a.no_dark or a.no_flat:
        drop = ({"dark"} if a.no_dark else set()) | ({"flat", "flat_dark"} if a.no_flat else set())
        project = ingest(calib, [])
        calib = [f.path for f in project.files if f.kind not in drop and f.kind != "unknown"]
        print(f"  without {sorted(drop)}: {len(calib)} calibration files")
    n_all = len(lights)
    if not a.no_pick:
        lights = pick(lights)
        print(f"  pick keeps {len(lights)}/{n_all}")
    meta = {"source": [str(p) for p in a.inputs], "object": key[0], "camera": key[1], "exposure": key[2],
            "filter": key[3], "picked": None if a.no_pick else [len(lights), n_all], "no_dark": a.no_dark,
            "no_flat": a.no_flat, "flat_any_night": a.flat_any_night}

    def integrate_hook(frames, bayer, shape, s, gdir, be, workers, log, progress=None):
        make_pairs(frames, bayer, shape, be, dst, meta, holdout=a.holdout, deep=a.deep)
        raise PairsDone()

    engine._integrate = integrate_hook
    s = engine.Settings(merge_trains=[trains] if len(trains) > 1 else [], keep_rejection_maps=False,
                        target=a.name, flat_any_night=a.flat_any_night)
    cache = Path(a.cache) / a.name
    try:
        engine.run(lights + calib, cache / "out", s, cache=cache, echo=lambda m: None)
    except PairsDone:
        pass
    if (dst / "meta.json").is_file() and not a.keep_cache:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)    # 校正後的每張 80 MB，20 組加起來超過 100 GB
    return 0 if (dst / "meta.json").is_file() else 1


if __name__ == "__main__":
    sys.exit(main())
