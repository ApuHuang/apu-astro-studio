"""契約：疊圖頁（pages/stack.py）用到的 APU Photons 函式與常數，用跟疊圖頁一樣的呼叫方式。"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from apu_photons import backend
from apu_photons import engine as photons_engine
from apu_photons import settings as photons_settings
from apu_photons.cache import cache_size, clear_cache, default_cache, format_size
from apu_photons.engine import STAGES, Cancelled, Result, Settings, run
from apu_photons.i18n import set_language
from apu_photons.imageio import IMAGE_SUFFIXES, TOP_DOWN, row_order
from apu_photons.ingest import SUB_KINDS, expand_inputs, filter_names, ingest, unused_text
from apu_photons.model import KINDS, UNKNOWN, Project
from apu_photons.preview import apply_stretch, stretch_params
from apu_photons.recipe import RecipeError, load_recipe
from astropy.io import fits

from .data import star_field, write_frames
from .snap import REL, Display, check

# 跨平台（macOS ARM 對 Windows x86）的數值容許誤差（使用者 2026-11-21）：離散的決定兩邊都要完全一樣，
# 數值在 Windows 用 1e-5；macOS 的 master 每個像素差 ≤ 峰值的 1e-3，背景、權重、變換、FWHM 相對 2e-3。
# 原因：ARM 與 x86 浮點最後幾位不同，被 Background2D 的 sigma clipping、FWHM 疊代（收斂門檻 1e-3）放大
MAC = sys.platform == "darwin"
NUM_REL = 2e-3 if MAC else REL
MASTER_FLOOR = 1e-3 if MAC else 1e-7

# 疊圖頁 _plan 傳給 ingest 的參數（預設值）
PLAN_ARGS = dict(kinds={}, filter_aliases={}, temp_tolerance=2.0, split_nights=True, flat_any_night=False,
                 overrides={}, merge_trains=None, file_calib={}, name=None)


@pytest.fixture(scope="module")
def frames(tmp_path_factory) -> tuple[Path, list[Path], list[Path]]:
    folder = tmp_path_factory.mktemp("photons") / "data"
    lights, darks = write_frames(folder)
    return folder, lights, darks


@pytest.fixture(scope="module")
def stacked(frames, tmp_path_factory):
    """疊圖頁 _start 的呼叫方式（CPU、單一行程，結果才每台機器一樣）。"""
    folder, lights, darks = frames
    out = folder.parent / "out"
    # 不自動裁邊：裁邊依覆蓋率門檻決定，對齊的浮點差一點就差一列（macOS 雲端 115 列 vs Windows 116 列），輸出尺寸要固定
    s = Settings(gpu="cpu", workers=1, crop_common=False)
    s.reference = None
    stages: list[str] = []
    diag: dict = {"inputs": [p.name for p in lights + darks]}
    real_rank, real_reg = photons_engine._rank_frames, photons_engine.register_many

    def rank(frs):
        out_ = real_rank(frs)
        diag["rank_inputs"] = [{"name": f.name, "fwhm": float(f.stars.fwhm), "n_stars": len(f.stars),
                                "background_per_s": float(f.stars.background / (f.exposure or 1.0))} for f in frs]
        diag["ranked"] = [f.name for f in out_]
        return out_

    def reg(tasks, workers, progress=None):
        out_ = real_reg(tasks, workers, progress)
        diag["register"] = [{"ok": r is not None, "reason": None if why is None else str(why),
                             "lengths": [len(x) for x in r if hasattr(x, "__len__") and not isinstance(x, str)]
                             if isinstance(r, tuple) else None} for r, why in out_]
        return out_

    photons_engine._rank_frames, photons_engine.register_many = rank, reg
    try:
        res = run(lights + darks, out, s, echo=lambda _line: None,
                  progress=lambda stage, d, t: stages.append(stage) if not stages or stages[-1] != stage else None,
                  cancel=threading.Event())
    finally:
        photons_engine._rank_frames, photons_engine.register_many = real_rank, real_reg
    _collect_diag(res, diag)
    _save_diag(res, diag)
    return res, stages, folder.parent, diag


def _collect_diag(res, diag: dict) -> None:
    """疊圖過程的決定與量測（兩層比對用；雲端也存成 artifact）。"""
    diag["platform"] = {"machine": platform.machine(), "system": platform.system(), "python": sys.version,
                        "numpy": np.__version__}
    diag["files_order"] = [f.name for f in res.project.files]
    diag["references"] = {ag.id: ag.reference.name if ag.reference else None for ag in res.project.align_groups}
    diag["frames"] = [{"name": f.name, "kind": f.kind, "accepted": f.accepted, "status": f.status,
                       "transform": None if f.transform is None else np.asarray(f.transform).tolist(),
                       "residual_px": f.registration_residual_px, "n_stars": None if f.stars is None else len(f.stars),
                       "weight": f.weight, "rejected_pixel_fraction": f.rejected_pixel_fraction,
                       "rejected_pixels": None if f.rejected_pixel_fraction is None or not f.shape
                       else round(f.rejected_pixel_fraction * f.shape[0] * f.shape[1]),
                       "fwhm": None if f.stars is None else float(f.stars.fwhm),
                       "background": None if f.stars is None else float(f.stars.background),
                       "norm": f.norm and {k: v for k, v in f.norm.items() if k in ("scale", "offset")}}
                      for f in res.project.files if f.kind == "light"]


def _save_diag(res, diag: dict) -> None:
    """APU_CONTRACT_ARTIFACTS 有設時（雲端），把疊圖的過程與結果存下來給 workflow 上傳，比較不同平台為什麼結果不同。"""
    dest = os.environ.get("APU_CONTRACT_ARTIFACTS")
    if not dest:
        return
    dest = Path(dest) / "photons_run"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "diag.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for o in res.outputs:
        shutil.copy(o.output, dest / o.output.name)
    if res.log and Path(res.log).is_file():
        shutil.copy(res.log, dest / Path(res.log).name)


def test_constants():
    check("photons_constants", {"image_suffixes": sorted(IMAGE_SUFFIXES), "top_down": TOP_DOWN, "kinds": list(KINDS),
                                "unknown": UNKNOWN, "sub_kinds": list(SUB_KINDS), "stages": list(STAGES),
                                "cancelled_is_exception": issubclass(Cancelled, Exception),
                                "recipe_error_is_exception": issubclass(RecipeError, Exception)})


def test_settings_fields():
    """疊圖頁用關鍵字參數建 Settings：欄位名稱與預設值是契約。"""
    s = Settings()
    check("photons_settings_defaults", {k: getattr(s, k) for k in sorted(s.__dataclass_fields__)})


def test_row_order():
    out = {}
    for name, hdr in (("top", {"ROWORDER": "TOP-DOWN"}), ("bottom", {"ROWORDER": "BOTTOM-UP"}), ("none", {})):
        out[name] = row_order(fits.Header(hdr))
    check("photons_row_order", out)


def test_master_display():
    """load_master_display：每個色版各自 stretch_params（隔點抽樣）＋ apply_stretch。"""
    img = star_field(200, 260, seed=5)
    step = 1
    disp = np.stack([apply_stretch(c, stretch_params(np.asarray(c[::step, ::step], np.float32))) for c in img])
    check("photons_master_display", {"params": [stretch_params(c) for c in img], "display": Display(disp),
                                     "float": np.asarray(disp, np.float32)})


def test_backend():
    be = backend.select("cpu")
    check("photons_backend", {"cpu_gpu": bool(be.gpu), "has_device": hasattr(be, "device"),
                              "workers_is_int": isinstance(backend.default_workers(), int)
                              and backend.default_workers() >= 1})


def test_settings_file(tmp_path, monkeypatch):
    """save_settings 合併寫入，但 params 這一個鍵是整包覆寫（疊圖頁存參數前要先讀回來）。"""
    monkeypatch.setattr(photons_settings, "settings_path", lambda: tmp_path / "photons.json")
    photons_settings.save_settings(list_mode="tree", params={"a": 1, "b": 2})
    photons_settings.save_settings(last_dir="X")
    photons_settings.save_settings(params={"a": 3})
    check("photons_settings_file", photons_settings.load_settings())


def test_expand_inputs(frames):
    folder, lights, darks = frames
    (folder / "notes.txt").write_text("x")
    got = expand_inputs([folder, lights[0]])
    check("photons_expand_inputs", [p.name for p in got])


def test_ingest(frames):
    folder, lights, darks = frames
    set_language("en")
    warnings: list = []
    project = ingest(lights + darks, warnings, **PLAN_ARGS)
    assert isinstance(project, Project)
    check("photons_ingest", {
        "name": project.name,
        "files": [(f.path.name, f.kind, f.filter, f.accepted, str(f.reject_reason or ""), f.night, f.temp)
                  for f in project.files],
        "lights": len(project.lights),
        "trains": {k: (t.bayer, t.shape) for k, t in project.trains.items()},
        "align_groups": len(project.align_groups),
        "calibration_sets": {k: (s.kind, len(s.frames), bool(s.is_master), sorted(s.used_by),
                                 "" if s.used_by else str(unused_text(s)))
                             for k, s in project.calibration_sets.items()},
        "calib_groups": sorted(project.calib_groups),
        "filter_names": filter_names(project),
        "warnings": [str(w) for w in warnings],
    })


def test_run_decisions(stacked):
    """第一層：離散的決定，每個平台都要完全一樣（輸入順序、參考張、各張偵測／配對到的星數、排除的像素數、收了哪幾張）。"""
    res, _stages, _root, diag = stacked
    frames = diag["frames"]
    check("photons_run_decisions", {
        "files_order": diag["files_order"],
        "references": diag["references"],
        "detected_stars": {f["name"]: f["n_stars"] for f in frames},
        "matched_stars": [r["lengths"][1] if r["ok"] else None for r in diag["register"]],
        "rejected_pixels": {f["name"]: f["rejected_pixels"] for f in frames},
        "accepted": {f["name"]: f["accepted"] for f in frames},
    })


def test_run_numbers(stacked):
    """第二層之一：各張的浮點量測（背景、權重、對齊變換、FWHM）。"""
    res, _stages, _root, diag = stacked
    check("photons_run_numbers", {f["name"]: {"fwhm": f["fwhm"], "background": f["background"], "weight": f["weight"],
                                              "transform": None if f["transform"] is None
                                              else np.asarray(f["transform"], np.float64)}
                                  for f in diag["frames"]}, rel=NUM_REL)


def test_run(stacked):
    """第二層之二：輸出的 master（macOS 每個像素差 ≤ 峰值的 1e-3）與輸出清單。"""
    res, stages, root, _diag = stacked
    assert isinstance(res, Result)
    masters = {}
    for o in res.outputs:
        with fits.open(o.output) as h:
            masters[o.output.name] = np.asarray(h[0].data, np.float32)
    check("photons_run", {
        "stages": stages,
        "outputs": [(o.align, o.filter, o.output.relative_to(root).as_posix(),
                     o.coverage.relative_to(root).as_posix() if o.coverage else None,
                     len(o.frames), sorted(o.channels)) for o in res.outputs],
        "recipe": res.recipe.relative_to(root).as_posix(),
        "masters": masters,
        "accepted": sorted(f.path.name for o in res.outputs for f in o.frames),
    }, floor=MASTER_FLOOR)


def test_load_recipe(stacked):
    res, _stages, root, _diag = stacked
    out = {}
    for reproduce in (False, True):
        rec = load_recipe(res.recipe, reproduce=reproduce, base=Settings(gpu="cpu", workers=1))
        s = rec.settings
        out[str(reproduce)] = {"rejection": s.rejection, "low": s.low, "high": s.high, "weighting": s.weighting,
                               "clamp": s.clamp, "distortion": s.distortion, "gpu": s.gpu, "workers": s.workers,
                               "warnings": [str(w) for w in rec.warnings]}
    bad = root / "bad.recipe.json"
    bad.write_text(json.dumps({"x": 1}))
    try:
        load_recipe(bad)
        out["bad"] = "no error"
    except RecipeError:
        out["bad"] = "RecipeError"
    check("photons_load_recipe", out)


def test_cache(tmp_path):
    out = tmp_path / "out"
    c = default_cache(out)
    before = cache_size(c)
    (c / "tmp").mkdir(parents=True)
    (c / "tmp" / "x.npy").write_bytes(b"0" * 3000)       # Photons 自己的子資料夾才算
    size = cache_size(c)
    freed = clear_cache(c)
    check("photons_cache", {"path": c.relative_to(tmp_path).as_posix(), "before": before, "size": size,
                            "freed_positive": freed > 0, "after": cache_size(c),
                            "format": [format_size(n) for n in (0, 512, 3000, 5 * 2 ** 20, 7 * 2 ** 30)]})
