"""契約測試的快照：Studio 實際呼叫的組件函式，用固定的小輸入跑一次，結果跟存好的快照比。組件行為一變就失敗。

容許誤差（使用者 2026-11-19 訂的）：
- 拉伸後的顯示影像（用 Display() 包起來的 0–1 影像）：換成 8-bit 後每個像素相差 ≤ 1
- 線性值（其他浮點數、浮點陣列）：相對誤差 ≤ 1e-5（跨平台／不同 CPU 超過的項目放寬到 1e-4，check 的 rel 參數，
  只放寬那一項；不維護 macOS 專用的快照）；接近 0 的值，下限是整個陣列最大值的 1e-7（check 的 floor 參數）
  （不然相對誤差在 0 附近沒有意義：0.0 對 1e-12 會是無限大）
- 整數、布林、字串、None：完全一樣

更新快照（確定組件的改變是要的之後）：
    set APU_UPDATE_SNAPSHOTS=1 && .venv\\Scripts\\python -m pytest tests/contract
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
from pathlib import Path

import numpy as np

SNAP_DIR = Path(__file__).with_name("snapshots")
UPDATE = os.environ.get("APU_UPDATE_SNAPSHOTS") == "1"
REL = 1e-5
FLOOR = 1e-7
_rel = [REL]          # 這一次 check 用的相對誤差（個別項目可以放寬）
_floor = [FLOOR]      # 這一次 check 的下限（陣列最大值的倍數）


class Display:
    """0–1 的顯示影像：用 8-bit 差 ≤ 1 比。"""

    def __init__(self, image):
        self.image = np.asarray(image, np.float32)


class _Norm:
    def __init__(self, root: Path | None):
        self.root = str(root.resolve()) if root is not None else None
        self.arrays: dict[str, np.ndarray] = {}

    def path(self, p) -> str:
        s = str(Path(p).resolve()) if Path(p).is_absolute() else str(p)
        if self.root and s.startswith(self.root):
            s = "<tmp>" + s[len(self.root):]
        return s.replace("\\", "/")

    def text(self, s: str) -> str:
        if self.root:
            s = s.replace(self.root, "<tmp>").replace(self.root.replace("\\", "/"), "<tmp>")
        return s.replace("\\", "/") if "<tmp>" in s else s

    def __call__(self, obj, key: str = "v"):
        if isinstance(obj, Display):
            a = np.clip(np.round(obj.image * 255), 0, 255).astype(np.uint8)
            self.arrays[key] = a
            return {"__display__": key}
        if isinstance(obj, np.ndarray):
            self.arrays[key] = obj
            return {"__array__": key}
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if math.isfinite(v) else {"__float__": repr(v)}
        if obj is None:
            return None
        if isinstance(obj, enum.Enum):
            return {"__enum__": f"{type(obj).__name__}.{obj.name}"}
        if isinstance(obj, Path):
            return {"__path__": self.path(obj)}
        if isinstance(obj, str):
            return self.text(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {"__dataclass__": type(obj).__name__,
                    **{f.name: self(getattr(obj, f.name), f"{key}.{f.name}") for f in dataclasses.fields(obj)}}
        if isinstance(obj, dict):
            return {"__dict__": [[self(k, f"{key}.k{i}"), self(v, f"{key}.{i}")] for i, (k, v) in
                                 enumerate(sorted(obj.items(), key=lambda kv: repr(kv[0])))]}
        if isinstance(obj, (list, tuple)):
            return [self(v, f"{key}.{i}") for i, v in enumerate(obj)]
        if isinstance(obj, (set, frozenset)):
            return {"__set__": sorted(repr(v) for v in obj)}
        # 其他物件（例如 Msg）：比顯示的文字
        return {"__str__": self.text(str(obj))}


def _close(a: float, b: float, peak: float = 0.0) -> bool:
    return abs(a - b) <= max(_rel[0] * abs(b), _floor[0] * peak, 1e-12)


def _compare(got, want, ga: dict, wa: dict, where: str, errors: list[str]) -> None:
    if isinstance(want, dict) and "__display__" in want:
        if not (isinstance(got, dict) and "__display__" in got):
            errors.append(f"{where}: 不是顯示影像")
            return
        g, w = ga[got["__display__"]], wa[want["__display__"]]
        if g.shape != w.shape:
            errors.append(f"{where}: 形狀 {g.shape} ≠ {w.shape}")
            return
        d = np.abs(g.astype(np.int16) - w.astype(np.int16))
        if d.max() > 1:
            errors.append(f"{where}: 8-bit 最大差 {int(d.max())}（{int((d > 1).sum())} 個像素差 > 1）")
        return
    if isinstance(want, dict) and "__array__" in want:
        if not (isinstance(got, dict) and "__array__" in got):
            errors.append(f"{where}: 不是陣列")
            return
        g, w = ga[got["__array__"]], wa[want["__array__"]]
        if g.shape != w.shape or g.dtype.kind != w.dtype.kind:
            errors.append(f"{where}: 形狀／型別 {g.shape} {g.dtype} ≠ {w.shape} {w.dtype}")
            return
        if w.dtype.kind == "f":
            g64, w64 = g.astype(np.float64), w.astype(np.float64)
            same_nan = np.array_equal(np.isnan(g64), np.isnan(w64))
            ok = np.isfinite(w64)
            peak = float(np.max(np.abs(w64[ok]))) if ok.any() else 0.0
            tol = np.maximum(_rel[0] * np.abs(w64), _floor[0] * peak)
            bad = ok & ~(np.abs(g64 - w64) <= tol)
            if not same_nan or bad.any():
                diff = np.abs(g64 - w64)[ok]
                rel = float(np.max(diff / np.maximum(np.abs(w64[ok]), FLOOR * peak + 1e-300)))
                errors.append(f"{where}: {int(bad.sum())} 個值超過容許誤差（最大相對差 {rel:.3g}、"
                              f"最大差是峰值的 {float(diff.max()) / (peak or 1.0):.3g}）")
        elif not np.array_equal(g, w):
            errors.append(f"{where}: {int((g != w).sum())} 個值不同")
        return
    if isinstance(want, float):
        if not isinstance(got, (int, float)) or isinstance(got, bool) or not _close(float(got), want):
            errors.append(f"{where}: {got!r} ≠ {want!r}")
        return
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            errors.append(f"{where}: 長度 {len(got) if isinstance(got, list) else type(got).__name__} ≠ {len(want)}")
            return
        for i, (g, w) in enumerate(zip(got, want)):
            _compare(g, w, ga, wa, f"{where}[{i}]", errors)
        return
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            errors.append(f"{where}: 欄位 {sorted(got) if isinstance(got, dict) else got!r} ≠ {sorted(want)}")
            return
        for k in want:
            _compare(got[k], want[k], ga, wa, f"{where}.{k}", errors)
        return
    if got != want:
        errors.append(f"{where}: {got!r} ≠ {want!r}")


def check(name: str, value, root: Path | None = None, rel: float = REL, floor: float = FLOOR) -> None:
    """跟 snapshots/<name>.json（＋ .npz）比；APU_UPDATE_SNAPSHOTS=1 時改寫快照。
    rel：相對誤差；floor：陣列的下限＝陣列最大值 × floor（floor=1e-3 就是「每個值差 ≤ 峰值的 1e-3」）。"""
    norm = _Norm(root)
    tree = norm(value)
    js, nz = SNAP_DIR / f"{name}.json", SNAP_DIR / f"{name}.npz"
    if UPDATE:
        SNAP_DIR.mkdir(exist_ok=True)
        js.write_text(json.dumps(tree, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        if norm.arrays:
            np.savez_compressed(nz, **norm.arrays)
        elif nz.exists():
            nz.unlink()
        return
    assert js.exists(), f"沒有快照 {js.name}：第一次要用 APU_UPDATE_SNAPSHOTS=1 產生"
    want = json.loads(js.read_text(encoding="utf-8"))
    wa = dict(np.load(nz, allow_pickle=False)) if nz.exists() else {}
    got = json.loads(json.dumps(tree))       # 跟存檔一樣走一次 JSON（tuple → list）
    errors: list[str] = []
    _rel[0], _floor[0] = rel, floor
    try:
        _compare(got, want, norm.arrays, wa, name, errors)
    finally:
        _rel[0], _floor[0] = REL, FLOOR
    assert not errors, "組件行為跟快照不同：\n" + "\n".join(errors[:20])
