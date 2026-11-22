"""步驟鏈：來源 → 一串步驟（可以停用、調順序、回頭改任何一步）。

- 每一步記住輸入、參數與結果。改了第 k 步，只重算 k 之後；前面的結果沿用
- 解析度：inter（互動層：快速層再縮小 INTER_FACTOR 倍，拖滑桿時）、quick（縮小，放手時）、full（完整，停手後背景算）、
  region（預覽區：只算框起來的那一塊，完整解析度）
- 整張統計（背景模型、白平衡、天空、拉伸參數、遮罩與顯示的自動拉伸）：完整解析度一律自己量（出圖跟 0.1.x 一樣），
  量到的值依「這一步、這個輸入」存起來（_stats 的 "full"），回灌給快速層、互動層、預覽區；完整解析度還沒算過時，
  預覽用快速層量的（"quick"；互動層、預覽區從快速層同一步的輸入量，ctx.ref）。拖滑桿時統計不用重量
- 去星：完整解析度算過後，結果縮小存起來（_split_full），去星和上游的參數都沒變（輸入鍵一樣）時快速層、互動層直接沿用；
  沿用時快速層、互動層的快取鍵加上記號，下游自動重算
- 去星把一層分成 starless／stars 兩層，後面的步驟各自選要作用在哪一層，合成星點再合回一層
- 完整解析度的結果很大（2000 萬像素彩色一層約 240 MB），超過記憶體上限就丟掉最久沒用的，要用時從最近的結果重算

引擎不 import Qt；介面在背景執行緒呼叫 evaluate()。
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from apu_processing import api as papi

from . import io, masks, ops
from .tools import Cancelled, Ctx, State, get
from .tools import geometry as tgeometry

QUICK_LONG_SIDE = 1800
INTER_FACTOR = 2          # 互動層＝快速層再縮小幾倍（整數，面積平均）
INTER_MIN_QUICK = 800     # 快速層長邊不到這麼大就不用互動層（小圖快速層本來就快；縮太小降噪等工具會出錯）
REGION_PAD = 128          # 預覽區四周多算的寬度（反摺積、降噪、補星點在邊緣要有鄰近的像素）
REGION_MAX = 2600         # 預覽區最大邊長
UI_ONLY = {"view", "brush"}


@dataclass
class Step:
    kind: str
    params: dict
    enabled: bool = True
    layer: str = "auto"
    mask: dict = field(default_factory=masks.default_mask)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = ""                 # 使用者自訂的名稱（例如「第二次降噪」）

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params, "enabled": self.enabled, "layer": self.layer,
                "mask": self.mask, "id": self.id, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> Step:
        tool = get(d["kind"])
        params = {**tool.defaults(), **(d.get("params") or {})}
        return cls(d["kind"], params, bool(d.get("enabled", True)), d.get("layer", "auto"),
                   {**masks.default_mask(), **(d.get("mask") or {})}, d.get("id") or uuid.uuid4().hex[:8],
                   d.get("label", ""))

    def key_part(self) -> str:
        # 只影響介面的參數（檢視切換、筆刷大小）不算：改了不用重算
        params = {k: v for k, v in self.params.items() if k not in UI_ONLY}
        mask = self.mask if self.mask.get("mode", "all") != "all" or self.mask.get("invert") else {}
        return json.dumps([self.kind, params, self.enabled, self.layer, mask], sort_keys=True, default=str)


def new_step(kind: str, layer: str = "auto", **params) -> Step:
    tool = get(kind)
    return Step(kind, {**tool.defaults(), **params}, layer=layer)


def default_steps(color: bool, narrowband: str | None = None) -> list[Step]:
    """使用者的流程：去光梯度 → 校色 → 反摺積 →（AI 降噪）→ 去星 → 降噪 → 分開拉伸 → 調色 → 銳化 → 再降噪 → 合成星點。
    2026-10-19 使用者試過幾種之後，覺得 10/18 最早放 AI 的這版效果最好，改回來（中間試過：不拔星、合併拉伸、拉伸後的
    步驟先關著、AI 去斑與色彩降噪；那些工具／選項都還在，只是預設不用）。
    AI 降噪在有模型時才放：反摺積之後（反摺積要原本的噪聲判斷哪裡該做）、去星之前（去星在乾淨的影像上找星比較準）。
    放了 AI 之後拉伸深淺、去星的範圍都不變（降噪的影響會被抵銷，見 tools/__init__.py 的 DENOISE_SIGMA_FACTOR）。"""
    from .tools import ai as tai
    steps = [new_step("geometry", auto_edges=True), new_step("gradient")]
    if color:
        cal = new_step("color_cal")
        cal.enabled = narrowband is None
        steps.append(cal)
    steps.append(new_step("deconv"))
    if tai.available():
        steps.append(new_step("ai_denoise"))
    steps += [
        new_step("star_split"),
        new_step("denoise", "starless", amount=0.5),
        new_step("stretch", "starless", method="auto"),
        new_step("stretch", "stars", method="arcsinh"),
    ]
    if color:
        palette = {"SHO": "sho_gold", "HOO": "hoo"}.get(narrowband or "", "none")
        steps.append(new_step("color_grade", "starless", palette=palette, vibrance=0.2))
    steps += [
        new_step("sharpen", "starless"),
        new_step("denoise", "starless", amount=0.3, luminance=0.8, chroma=1.2),
        new_step("star_merge", saturation=0.8 if narrowband else 1.2, demagenta=0.8 if narrowband else 0.0),
    ]
    return steps


# ---------------------------------------------------------------------- 來源


@dataclass
class Source:
    paths: list[Path]
    coverages: list[Path | None] = field(default_factory=list)
    compose: dict | None = None          # 多張單色：{"preset": "SHO", "channels": [[role, source, strength], …]}
    image: np.ndarray | None = None      # (C, H, W) 完整尺寸、線性
    header: object = None
    filters: list[str] = field(default_factory=list)
    composer: object = None
    coverage_mask: np.ndarray | None = None
    channel_shift: dict = field(default_factory=dict)   # 三色對齊：各色版最大位移（px）
    psf_stars: tuple | None = None       # 開檔時挑好的星 (y, x, 半徑)：反摺積量星點形狀時直接用

    @property
    def color(self) -> bool:
        return self.image is not None and self.image.shape[0] >= 3

    @property
    def narrowband(self) -> str | None:
        if self.compose and self.compose.get("preset") in ("HOO", "SHO"):
            return self.compose["preset"]
        return None

    def title(self) -> str:
        if self.compose:
            return f"{self.compose.get('preset')} · " + " + ".join(p.stem for p in self.paths)
        return self.paths[0].name if self.paths else ""

    def to_dict(self) -> dict:
        return {"paths": [str(p) for p in self.paths], "coverages": [str(c) if c else None for c in self.coverages],
                "compose": self.compose}


def open_source(paths: list[Path], compose: dict | None = None, coverages: list | None = None) -> Source:
    """開檔：一張就直接用；好幾張單色就合成（HOO／SHO／RGB）；疊圖紀錄開它主要對齊組的 master。"""
    paths = [Path(p) for p in paths]
    records = [p for p in paths if papi.is_recipe(p)]
    if records:
        project = papi.load_recipe(records[0])
        _group, outputs = project.main_group()
        paths = [o.path for o in outputs]
        coverages = [o.coverage for o in outputs]
    coverages = list(coverages or [None] * len(paths))
    if len(paths) == 1:
        image, header = io.load_image(paths[0])
        src = Source(paths, coverages, None, ops.planes(image), header,
                     [str(header.get("FILTER", "")).strip()] if header is not None else [])
    else:
        planes, headers = [], []
        for p in paths:
            img, h = io.load_image(p)
            if img.ndim != 2:
                img = img.mean(axis=0)
            planes.append(img)
            headers.append(h)
        composer = papi.Composer(planes, [p.name for p in paths])
        filters = [str(h.get("FILTER", "")).strip() for h in headers]
        settings = settings_from_dict(compose, filters) if compose else papi.ComposeSettings.recommended(filters)
        rgb = composer.compose(settings)
        src = Source(paths, coverages, settings_to_dict(settings), np.ascontiguousarray(rgb, np.float32),
                     headers[0], filters, composer)
    fractions = []
    for p, c in zip(paths, coverages):
        found = Path(c) if c else papi.find_coverage(p)
        try:
            fractions.append(papi.load_coverage_fraction(found) if found else None)
        except (OSError, ValueError):
            fractions.append(None)
    cov = papi.combine_coverage(fractions, src.image.shape[-2:])
    if cov is not None:
        good = papi.coverage_good_mask(cov)
        src.coverage_mask = None if good.all() else good
    _align(src)
    return src


def _align(src: Source) -> None:
    """彩色：各色版對齊綠色（第二個）色版。橫向色差、大氣色散、多張單色之間的對齊誤差會讓亮星一邊紅一邊青，
    拉伸、加飽和度後很明顯（NGC2244 紅色版角落偏 0.9 px）。讀檔時就做，之後每一步、每種解析度都一致。"""
    if src.image is None or src.image.ndim != 3 or src.image.shape[0] < 3:
        return
    from . import psf
    src.image, info = psf.align_channels(src.image)
    src.psf_stars = info.pop("stars", None)
    src.channel_shift = info


def settings_to_dict(s) -> dict:
    return {"preset": s.preset, "channels": [[c.role, c.source, c.strength] for c in s.channels]}


def settings_from_dict(d: dict, filters: list[str]):
    s = papi.ComposeSettings.for_preset(d.get("preset", "HOO"), filters)
    for i, (_role, source, strength) in enumerate(d.get("channels", [])):
        if i < len(s.channels):
            s = s.with_source(i, int(source)).with_strength(i, float(strength))
    return s


def recompose(src: Source, compose: dict) -> None:
    if src.composer is None:
        return
    settings = settings_from_dict(compose, src.filters)
    src.compose = settings_to_dict(settings)
    src.image = np.ascontiguousarray(src.composer.compose(settings), np.float32)
    _align(src)


# ---------------------------------------------------------------------- 計算


def _downsample(p: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return p
    C, h, w = p.shape
    h2, w2 = h // f * f, w // f * f
    return p[:, :h2, :w2].reshape(C, h2 // f, f, w2 // f, f).mean(axis=(2, 4)).astype(np.float32)


@dataclass
class Result:
    state: State
    views: dict
    index: int                # 算到第幾步（-1 是來源）
    res: str


class Chain:
    def __init__(self, memory_budget: int = 8 * 2 ** 30):
        self.source: Source | None = None
        self.steps: list[Step] = []
        self.memory_budget = memory_budget
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[str, int], tuple[str, State, dict]] = OrderedDict()
        self._analysis: dict[tuple[str, str], tuple[str, dict]] = {}
        self._source_states: dict[str, State] = {}
        # (步驟 id, 輸入鍵, "full"／"quick") → 那個解析度量的整張統計
        self._stats: OrderedDict[tuple[str, str, str], dict] = OrderedDict()
        # (第幾步的輸入鍵, 顯示的層) → ("full"／"quick", 顯示用自動拉伸的參數)
        self._disp_memo: OrderedDict[tuple, tuple] = OrderedDict()
        self._split_full: dict[str, tuple[str, State]] = {}       # 去星步驟 id → (輸入鍵, 完整解析度結果縮到快速層)
        self._full_keys: OrderedDict[str, None] = OrderedDict()    # 完整解析度算過的步驟鍵
        self.source_revision = 0
        self.region: tuple[int, int, int, int] | None = None    # 預覽區（來源影像的座標 y0, x0, y1, x1）
        self._source_psf = None                                    # 原圖的星點形狀（反摺積用，量一次）
        self._spike_angle: object = "unset"                        # 原圖的繞射芒方向（去星用，量一次）

    # ---------- 設定

    def set_source(self, src: Source) -> None:
        with self._lock:
            self.source = src
            self.source_revision += 1
            self._source_psf = None
            self._spike_angle = "unset"
            self._cache.clear()
            self._analysis.clear()
            self._source_states.clear()
            self._stats.clear()
            self._disp_memo.clear()
            self._split_full.clear()
            self._full_keys.clear()

    def source_changed(self) -> None:
        """來源影像換了（例如合成的強度）：全部重算。"""
        self.set_source(self.source)

    def quick_factor(self) -> int:
        if self.source is None or self.source.image is None:
            return 1
        longest = max(self.source.image.shape[-2:])
        return max(1, int(np.ceil(longest / QUICK_LONG_SIDE)))

    def inter_ok(self) -> bool:
        """這張圖要不要用互動層（快速層夠大才用）。"""
        if self.source is None or self.source.image is None:
            return False
        longest = max(self.source.image.shape[-2:]) // self.quick_factor()
        return longest >= INTER_MIN_QUICK

    def scale(self, res: str) -> float:
        if res == "quick":
            return float(self.quick_factor())
        if res == "inter":
            return float(self.quick_factor() * INTER_FACTOR)
        return 1.0

    # ---------- 預覽區

    def set_region(self, box: tuple[int, int, int, int] | None) -> None:
        if box is not None and self.source is not None:
            H, W = self.source.image.shape[-2:]
            y0, x0, y1, x1 = (int(round(v)) for v in box)
            y0, x0 = max(0, min(H - 16, y0)), max(0, min(W - 16, x0))
            y1, x1 = max(y0 + 16, min(H, y1, y0 + REGION_MAX)), max(x0 + 16, min(W, x1, x0 + REGION_MAX))
            box = (y0, x0, y1, x1)
        with self._lock:
            self.region = box
            for key in [k for k in self._source_states if k.startswith("region:")]:
                del self._source_states[key]
            for key in [k for k in self._cache if k[0].startswith("region:")]:
                del self._cache[key]

    def region_res(self) -> str | None:
        return None if self.region is None else "region:" + ",".join(str(v) for v in self.region)

    def map_to_source(self, box: tuple[float, float, float, float], index: int,
                      steps: list[Step] | None = None) -> tuple[int, int, int, int] | None:
        """第 index 步結果上的框 (y0, x0, y1, x1)（完整尺寸）→ 來源影像上的框（反推前面的裁切與旋轉）。
        自動裁邊要用快速預覽量到的邊；還沒算過就回傳 None。"""
        done = self._geometry_ops(index, steps)
        if done is None:
            return None
        b = tuple(float(v) for v in box)
        for op, cin in reversed(done):
            b = tgeometry.box_inverse(op, b, cin)
        y0, x0, y1, x1 = b
        return (int(min(y0, y1)), int(min(x0, x1)), int(max(y0, y1)), int(max(x0, x1)))

    def map_from_source(self, box: tuple, index: int, steps: list[Step] | None = None) -> tuple | None:
        """來源影像上的框 → 第 index 步結果（快速預覽的座標）上的框。"""
        done = self._geometry_ops(index, steps)
        if done is None:
            return None
        b = tuple(float(v) for v in box)
        for op, cin in done:
            b = tgeometry.box_forward(op, b, cin)
        return b

    def _geometry_ops(self, index: int, steps: list[Step] | None = None) -> list[tuple[tuple, tuple]] | None:
        """第 0～index 步的裁切與旋轉拆成的基本動作（各自的輸入大小）；自動裁邊用快速預覽量到的邊。"""
        if self.source is None:
            return None
        steps = steps if steps is not None else self.steps
        canvas = tuple(self.source.image.shape[-2:])
        done: list[tuple[tuple, tuple]] = []
        for i in range(0, min(index, len(steps) - 1) + 1):
            s = steps[i]
            if not s.enabled or s.kind != "geometry":
                continue
            edges = None
            if s.params.get("auto_edges"):
                hit = self._cache.get(("quick", i))
                if hit is None:
                    return None
                edges = hit[2].get("edges_full")
            for op in tgeometry.plan(s.params, canvas, edges):
                done.append((op, canvas))
                canvas = tgeometry.canvas_after(op, canvas)
        return done

    def snapshot(self) -> list[Step]:
        """背景計算用的步驟副本（介面同時在改參數也不會互相干擾）。"""
        return copy.deepcopy(self.steps)

    # ---------- 快取

    def _base_keys(self, steps: list[Step]) -> list[str]:
        """每一步的鍵＝這一步與上游所有參數的 hash（跟解析度無關）。"""
        h = hashlib.sha1(f"src{self.source_revision}".encode())
        out = []
        for s in steps:
            h = hashlib.sha1(h.digest() + s.key_part().encode())
            out.append(h.hexdigest())
        return out

    def _keys(self, steps: list[Step], res: str = "full") -> list[str]:
        """快取用的鍵：快速層、互動層在去星沿用完整解析度的結果時，從那一步起加記號（下游跟自己算的分開快取）。"""
        base = self._base_keys(steps)
        if res not in ("quick", "inter") or not self._split_full:
            return base
        h = hashlib.sha1(f"src{self.source_revision}".encode())
        out = []
        for i, s in enumerate(steps):
            part = s.key_part().encode()
            if self._split_reuse(s, base[i]) is not None:
                part += b"|full-split"
            h = hashlib.sha1(h.digest() + part)
            out.append(h.hexdigest())
        return out

    def _split_reuse(self, step: Step, key: str) -> State | None:
        """這一步是去星、而且完整解析度算過同樣參數（含上游）的：回傳縮到快速層的結果。"""
        if not step.enabled or step.kind != "star_split":
            return None
        hit = self._split_full.get(step.id)
        return hit[1] if hit is not None and hit[0] == key else None

    def full_ready(self, index: int, steps: list[Step] | None = None) -> bool:
        """第 index 步（含上游）用目前的參數算過完整解析度：預覽用的是完整解析度的統計。"""
        steps = steps if steps is not None else self.steps
        if self.source is None or index < 0 or index >= len(steps):
            return False
        with self._lock:
            return self._base_keys(steps)[index] in self._full_keys

    def base_key(self, index: int, steps: list[Step] | None = None) -> str:
        steps = steps if steps is not None else self.steps
        return self._base_keys(steps)[index] if 0 <= index < len(steps) else f"src{self.source_revision}"

    def _source_state(self, res: str) -> State:
        st = self._source_states.get(res)
        if st is None:
            if res.startswith("region:"):
                y0, x0, y1, x1 = (int(v) for v in res.split(":")[1].split(","))
                H, W = self.source.image.shape[-2:]
                b = (max(0, y0 - REGION_PAD), max(0, x0 - REGION_PAD), min(H, y1 + REGION_PAD),
                     min(W, x1 + REGION_PAD))
                p = np.ascontiguousarray(ops.planes(self.source.image)[:, b[0]:b[2], b[1]:b[3]])
                st = State({"main": p}, {"main": "linear"}, region={"box": b, "inner": (y0, x0, y1, x1),
                                                                    "canvas": (H, W)})
            elif res == "inter":
                st = State({"main": _downsample(self._source_state("quick").layers["main"], INTER_FACTOR)},
                           {"main": "linear"})
            else:
                p = _downsample(self.source.image, self.quick_factor()) if res == "quick" else self.source.image
                st = State({"main": p}, {"main": "linear"})
            self._source_states[res] = st
        return st

    def _evict(self) -> None:
        small = ("quick", "inter")
        total = sum(st.nbytes() for (res, _i), (_k, st, _v) in self._cache.items() if res not in small)
        for key in list(self._cache):
            if total <= self.memory_budget:
                break
            if key[0] in small:
                continue
            total -= self._cache[key][1].nbytes()
            del self._cache[key]

    def cached(self, index: int, res: str, steps: list[Step] | None = None) -> Result | None:
        """不用算就有的結果（介面切換步驟時立刻顯示）。"""
        steps = steps if steps is not None else self.steps
        with self._lock:
            if self.source is None:
                return None
            if index < 0:
                return Result(self._source_state(res), {}, -1, res)
            keys = self._keys(steps, res)
            hit = self._cache.get((res, index))
            if hit is not None and hit[0] == keys[index]:
                self._cache.move_to_end((res, index))
                return Result(hit[1], hit[2], index, res)
        return None

    def evaluate(self, upto: int, res: str, steps: list[Step] | None = None, cancel=lambda: False,
                 progress=lambda f, s: None) -> Result:
        """算到第 upto 步（含），回傳那一步的結果。"""
        steps = steps if steps is not None else self.snapshot()
        upto = min(upto, len(steps) - 1)
        with self._lock:
            keys = self._keys(steps, res)
            base = self._base_keys(steps)
            qkeys = keys if res in ("quick", "inter") else self._keys(steps, "quick")
            start, state, views = -1, self._source_state(res), {}
            for i in range(upto, -1, -1):
                hit = self._cache.get((res, i))
                if hit is not None and hit[0] == keys[i]:
                    start, state, views = i, hit[1], hit[2]
                    self._cache.move_to_end((res, i))
                    break
                if res == "inter":
                    # 互動層：快速層已經算到這一步的話直接縮小 INTER_FACTOR 倍（拖第 k 步時只要算第 k 步）
                    q = self._cache.get(("quick", i))
                    if q is not None and q[0] == keys[i]:
                        start, state, views = i, _downsample_state(q[1], INTER_FACTOR), {}
                        self._cache[(res, i)] = (keys[i], state, views)
                        break
        scale = self.scale(res)
        meta = {"coverage_mask": self.source.coverage_mask if self.source else None,
                "header": self.source.header if self.source else None,
                "source_psf": self.source_psf, "spike_angle": self.source_spike_angle}
        region = res.startswith("region:")
        if region and start < upto:
            # 預覽區：每一步的輸入在快速預覽的樣子（還沒有完整解析度的統計時從這裡量）；通常已經算好在快取裡
            self.evaluate(upto, "quick", steps, cancel)
        for i in range(start + 1, upto + 1):
            if cancel():
                raise Cancelled()
            step = steps[i]
            views = {}
            reuse = self._split_reuse(step, base[i]) if res in ("quick", "inter") else None
            if reuse is not None:
                # 去星：完整解析度算過同樣參數的，直接縮小沿用（快速層抓不到 1～2 px 的暗星）
                state = reuse if res == "quick" else _downsample_state(reuse, INTER_FACTOR)
            elif step.enabled:
                tool = get(step.kind)
                in_key = keys[i - 1] if i > 0 else f"src{self.source_revision}"
                in_base = base[i - 1] if i > 0 else f"src{self.source_revision}"
                akey = (step.id, res)
                with self._lock:
                    prev = self._analysis.get(akey)
                    acache = prev[1] if prev is not None and prev[0] == in_key else {}
                ctx = Ctx(scale, cancel, lambda f, s, i=i: progress((i + f) / (upto + 1), s), acache, meta, views)
                if step.kind in ("deconv", "star_split"):
                    ctx.frame_ops = self._geometry_ops(i - 1, steps) if i > 0 else []
                if res == "full":
                    # 完整解析度：自己量（跟 0.1.x 一樣），量到的存起來回灌給預覽
                    ctx.stats = self._stats_for(step.id, in_base, "full")
                else:
                    ref = None if res == "quick" else self._quick_input(i, qkeys, required=region)
                    if ref is not None:
                        ctx.ref = ref
                        ctx.ref_scale = float(self.quick_factor())
                    ctx.stats = self._stats_for(step.id, in_base, "quick") if res == "quick" or ref is not None else {}
                    full_stats = self._stats_get(step.id, in_base, "full")
                    ctx.stats_read = (full_stats,) if full_stats else ()
                ctx.interactive = res == "inter"
                if region:
                    ctx.region = state.region
                    if not any(st.enabled for st in steps[:i]):
                        ctx.full_input = self.source.image
                progress(i / (upto + 1), step.kind)
                try:
                    state = tool.run(state, step.params, step.layer, step.mask, ctx)
                except papi.Cancelled as exc:
                    raise Cancelled() from exc
                with self._lock:
                    self._analysis[akey] = (in_key, acache)
            with self._lock:
                self._cache[(res, i)] = (keys[i], state, views)
                self._cache.move_to_end((res, i))
                self._evict()
                if res == "full":
                    self._mark_full(base[i])
                    if step.enabled and step.kind == "star_split" and self.source is not None:
                        self._split_full[step.id] = (base[i], _downsample_state(state, self.quick_factor()))
        if res == "full":
            with self._lock:
                for i in range(0, start + 1):
                    self._mark_full(base[i])
        progress(1.0, "")
        return Result(state, views, upto, res)

    def _mark_full(self, key: str) -> None:
        self._full_keys[key] = None
        self._full_keys.move_to_end(key)
        while len(self._full_keys) > 512:
            self._full_keys.popitem(last=False)

    def source_psf(self):
        """原圖（完整解析度）的星點形狀 3×3 區（psf.PSFGrid），第一次用到時量。"""
        with self._lock:
            if self._source_psf is None and self.source is not None:
                from . import psf
                img = ops.planes(self.source.image)
                self._source_psf = psf.measure_grid(img.mean(axis=0), 3, self.source.psf_stars)
            return self._source_psf

    def source_spike_angle(self):
        """原圖（完整解析度）的繞射芒方向（度）或 None，第一次用到時量。"""
        with self._lock:
            if self._spike_angle == "unset" and self.source is not None:
                from apu_processing import api as papi
                from .tools import stars as tstars
                L = ops.planes(self.source.image).mean(axis=0)
                found = papi.find_stars(L, 5.0)
                found = tstars.keep_stars(found, tstars.star_like(L, found, 1.0))
                self._spike_angle = tstars.estimate_spike_angle(L, found, 1.0)
            return None if self._spike_angle == "unset" else self._spike_angle

    def _quick_input(self, index: int, keys: list[str], required: bool = True) -> State | None:
        """第 index 步在快速預覽的輸入（前一步的結果）。required=False 時還沒算好就回傳 None。"""
        if index == 0:
            return self._source_state("quick")
        with self._lock:
            hit = self._cache.get(("quick", index - 1))
        if hit is None or hit[0] != keys[index - 1]:
            if not required:
                return None
            raise RuntimeError("quick result missing")
        return hit[1]

    def _stats_get(self, step_id: str, in_key: str, source: str) -> dict | None:
        with self._lock:
            return self._stats.get((step_id, in_key, source))

    def _stats_for(self, step_id: str, in_key: str, source: str) -> dict:
        """這一步、這個輸入在某個解析度（"full"／"quick"）量的整張統計；輸入變了就換一份。"""
        with self._lock:
            key = (step_id, in_key, source)
            d = self._stats.get(key)
            if d is None:
                d = self._stats[key] = {}
                while len(self._stats) > 256:
                    self._stats.popitem(last=False)
            self._stats.move_to_end(key)
            return d

    def quick_state(self, index: int, steps: list[Step] | None = None) -> State | None:
        """第 index 步在快速層的結果（顯示的自動拉伸參數從這裡量）；還沒算好時 None。"""
        steps = steps if steps is not None else self.steps
        with self._lock:
            if self.source is None:
                return None
            if index < 0:
                return self._source_state("quick")
            hit = self._cache.get(("quick", index))
            if hit is None or hit[0] != self._keys(steps, "quick")[index]:
                return None
            return hit[1]

    def display(self, result: Result, layer: str = "combined") -> np.ndarray:
        """結果的顯示影像。線性的層的自動拉伸參數：完整解析度量自己的（跟 0.1.x 一樣）並記住；
        快速層、互動層、預覽區用記住的完整解析度參數，還沒有的話用快速層量的。"""
        key = (self.base_key(result.index), layer)
        with self._lock:
            full = self._disp_memo.get(key + ("full",))
            if result.res == "full" and full is None:
                full = self._disp_memo[key + ("full",)] = {}
            quick = self._disp_memo.get(key + ("quick",))
            if result.res != "full" and quick is None:
                quick = self._disp_memo[key + ("quick",)] = {}
            while len(self._disp_memo) > 128:
                self._disp_memo.popitem(last=False)
        if result.res == "full":
            return display_of(result.state, layer, None, full)
        ref = None if result.res == "quick" else self.quick_state(result.index)
        return display_of(result.state, layer, ref, quick, full)

    def region_display(self, result: Result, layer: str = "combined") -> tuple[np.ndarray, tuple[int, int]] | None:
        """預覽區結果的顯示影像（只有使用者框的範圍）與左上角在這一步座標的位置 (x, y)。
        線性的層用整張的自動拉伸參數，顏色亮度才會跟周圍一樣。"""
        reg = result.state.region
        if reg is None:
            return None
        disp = self.display(result, layer)
        by0, bx0 = reg["box"][0], reg["box"][1]
        iy0, ix0, iy1, ix1 = reg["inner"]
        crop = disp[:, iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
        # 位置用快速預覽的座標（自動裁邊在縮圖上量的邊可能差幾個像素），疊在快速預覽上才對得齊
        pos = (ix0, iy0)
        if result.res.startswith("region:"):
            src_inner = tuple(int(v) for v in result.res.split(":")[1].split(","))
            fb = self.map_from_source(src_inner, result.index)
            if fb is not None:
                pos = (int(round(min(fb[1], fb[3]))), int(round(min(fb[0], fb[2]))))
        return crop, pos

    # ---------- 專案檔

    def to_dict(self) -> dict:
        return {"format": "apu-astro-studio/1", "source": self.source.to_dict() if self.source else None,
                "steps": [s.to_dict() for s in self.steps]}


def display_of(state: State, layer: str = "combined", ref: State | None = None,
               memo: dict | None = None, memo_read: dict | None = None) -> np.ndarray:
    """給畫面看的 0–1 影像：線性的層自動拉伸；分離星點時可以看無星圖、星點層或合起來的樣子。
    ref：自動拉伸的參數從這個狀態量（預覽區、互動層用快速層）。memo：記住量過的拉伸參數（同一個 ref）；
    memo_read：優先用的參數（完整解析度量的），沒有才自己量、記在 memo。"""
    def auto(img: np.ndarray, w: np.ndarray | None, name: str) -> np.ndarray:
        if memo_read and name in memo_read:
            return ops.apply_stf(img, memo_read[name], linked=False)
        if memo is None:
            return ops.auto_display(img, w)
        if name not in memo:
            memo[name] = ops.stf_params(img if w is None else w)
        return ops.apply_stf(img, memo[name], linked=False)

    def whole(name: str) -> np.ndarray | None:
        return ops.planes(ref.layers[name]) if ref is not None and name in ref.layers else None

    def disp(name: str) -> np.ndarray:
        img = ops.planes(state.layers[name])
        if state.phase.get(name) != "linear":
            return np.clip(img, 0, 1)
        if name == "stars":
            # 星點層大部分是 0，自動拉伸會失控；用跟「合成星點」預設一樣的保色拉伸
            return ops.arcsinh_stretch(img, 3000.0, [0.0] * len(img))
        return auto(img, whole(name), name)

    if not state.separated:
        return disp("main")
    if layer == "starless":
        return disp("starless")
    if layer == "stars":
        return disp("stars")
    if state.phase.get("starless") == "linear" and state.phase.get("stars") == "linear":
        w = None
        if ref is not None and ref.separated:
            w = ops.planes(ref.layers["starless"]) + ops.planes(ref.layers["stars"])
        return auto(ops.planes(state.layers["starless"]) + ops.planes(state.layers["stars"]), w, "combined")
    a, b = disp("starless"), disp("stars")
    return (1 - (1 - a) * (1 - b)).astype(np.float32)


def _downsample_state(st: State, f: int) -> State:
    """整個狀態（各層）面積平均縮小 f 倍（互動層從快速層的結果做）。"""
    out = st.copy()
    out.warnings = list(st.warnings)
    out.layers = {k: _downsample(ops.planes(v), f) for k, v in st.layers.items()}
    return out


def save_project(chain: Chain, path: Path) -> None:
    path.write_text(json.dumps(chain.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


def load_project(path: Path) -> tuple[dict, list[Step]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = [Step.from_dict(d) for d in data.get("steps", [])]
    return data.get("source") or {}, steps
