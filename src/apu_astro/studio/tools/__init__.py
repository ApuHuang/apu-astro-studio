"""後製工具：每個工具是一個步驟（去光梯度、校色、反摺積、去星、降噪、拉伸、調色、銳化、合成星點…）。

工具只描述「參數、介面控制項、怎麼算」，不碰 Qt。介面依 controls() 自動產生右側面板。
文字用 T(中文, 英文)，顯示時才依目前語言取。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ... import i18n
from .. import masks, ops


class T(str):
    """雙語文字：str(T) 依目前語言。"""

    def __new__(cls, zh: str, en: str):
        obj = super().__new__(cls, zh)
        obj.zh, obj.en = zh, en
        return obj

    def __str__(self) -> str:
        return self.zh if i18n.get_language() == "zh" else self.en

    def text(self) -> str:
        return str(self)


@dataclass
class State:
    """步驟之間傳的東西：圖層（分離星點前只有 main，之後是 starless／stars）與各圖層是線性還是已拉伸。"""
    layers: dict[str, np.ndarray]
    phase: dict[str, str]
    warnings: list = field(default_factory=list)
    # 預覽區：只算畫面的一塊（完整解析度）。{"box": 這塊在整張的位置 (y0, x0, y1, x1)（含邊界緩衝），
    # "inner": 使用者框的範圍, "canvas": 整張的 (H, W)}，都是完整尺寸的像素、這一步的座標
    region: dict | None = None
    # 各層往下傳的附註（例如 AI 降噪把噪聲降了幾倍，自動拉伸要乘回去；去星分層時跟著複製）
    notes: dict = field(default_factory=dict)

    def copy(self) -> State:
        return State(dict(self.layers), dict(self.phase), [], self.region,
                     {k: dict(v) for k, v in self.notes.items()})

    @property
    def separated(self) -> bool:
        return "starless" in self.layers

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.layers.values())


@dataclass
class Ctx:
    scale: float                                  # 完整尺寸的幾個像素＝這個解析度的一個像素
    cancel: Callable[[], bool] = lambda: False
    progress: Callable[[float, str], None] = lambda f, s: None
    cache: dict = field(default_factory=dict)     # 這個步驟在這個解析度的分析結果（輸入沒變就重用）
    meta: dict = field(default_factory=dict)      # 整條鏈共用：header、覆蓋率、是不是窄帶合成
    view: dict = field(default_factory=dict)      # 給介面看的東西（例如背景模型）
    # 預覽區：ref＝同一步的輸入在快速預覽（整張縮圖）的樣子。需要整張統計的工具（背景模型、拉伸、白平衡…）
    # 從 ref 量，套到這一塊上，結果才會跟整張算的一樣
    ref: State | None = None
    ref_scale: float = 1.0
    region: dict | None = None
    layer: str = "main"
    full_input: np.ndarray | None = None          # 預覽區且這一步的輸入就是來源時：整張完整解析度的輸入
    notes: dict = field(default_factory=dict)     # 輸入狀態的各層附註（State.notes）
    frame_ops: list | None = None                 # 前面的裁切／旋轉／翻轉（chain._geometry_ops），換算回原圖座標用

    def check(self) -> None:
        if self.cancel():
            raise Cancelled()

    def whole(self, img: np.ndarray) -> np.ndarray:
        """量整張統計用的影像：預覽區時是整張的縮圖，平常就是自己。"""
        if self.ref is not None and self.layer in self.ref.layers:
            return ops.planes(self.ref.layers[self.layer])
        return img

    @property
    def whole_scale(self) -> float:
        return self.ref_scale if self.ref is not None else self.scale

    @property
    def origin(self) -> tuple[float, float]:
        """這塊影像左上角在整張的位置 (x, y)（完整尺寸的像素）。"""
        if self.region is None:
            return 0.0, 0.0
        return float(self.region["box"][1]), float(self.region["box"][0])


class Cancelled(Exception):
    pass


class RegionOutside(Exception):
    """預覽區整塊被裁切掉了。"""


LAYER_CHOICES = [("auto", T("自動", "Auto")), ("main", T("整張", "Whole image")),
                 ("starless", T("無星圖", "Starless")), ("stars", T("星點層", "Stars"))]


# 自動拉伸的深淺依噪聲決定。降噪會讓量到的噪聲變小、拉伸變重；為了讓「開不開降噪、用不用 AI」都不改變拉伸的深淺，
# 每個（線性階段的）降噪步驟記下自己讓 8 px 噪聲降了幾倍（State.notes 的 denoise_ratio，累乘），
# 自動拉伸用「量到的噪聲 × denoise_ratio × DENOISE_SIGMA_FACTOR」。
# 係數＝建議流程的 Processing 降噪（無星圖、強度 50%）讓 8 px 噪聲降的倍數的倒數：
# NGC2244、NGC6888、M31、IC434、M81、M16 量到 1.86～1.97 倍，取 1.92 → 建議流程的外觀幾乎不變。
# （只記「降噪前」的噪聲不行：去星也會讓量到的噪聲變小約 1.4 倍，暗星原本算在裡面）
DENOISE_SIGMA_FACTOR = 1 / 1.92


def record_denoise_ratio(state: State, out: State, layers: list[str], ctx: Ctx) -> None:
    """降噪步驟呼叫：記下這一步讓噪聲降了幾倍（跟前面的降噪累乘）。只算線性的層。"""
    for name in layers:
        if state.phase.get(name, "linear") != "linear":
            continue
        before, after = ops.planes(state.layers[name]), ops.planes(out.layers[name])
        nb = ops.measure_noise(before[0] if before.shape[0] == 1 else before, ctx.scale)
        na = ops.measure_noise(after[0] if after.shape[0] == 1 else after, ctx.scale)
        rl = nb.sigma / max(na.sigma, 1e-12)
        rc = [b / max(a, 1e-12) for b, a in zip(nb.channel_sigma, na.channel_sigma)]
        old = out.notes.get(name, {}).get("denoise_ratio")
        if old is not None:
            rl, rc = rl * old[0], [r * o for r, o in zip(rc, old[1])]
        out.notes.setdefault(name, {})["denoise_ratio"] = (float(rl), [float(r) for r in rc])


class Tool:
    kind = ""
    title = T("", "")
    help = T("", "")
    phase = "any"            # linear：拉伸前才能用；nonlinear：拉伸後；any
    icon = "process"
    maskable = True          # 可以選作用範圍
    auto_layer = "starless"  # 分離星點後「自動」作用在哪一層

    def defaults(self) -> dict:
        return {}

    def controls(self) -> list[dict]:
        return []

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        """(C, H, W) → (C, H, W)。"""
        return img

    def target_layers(self, state: State, layer: str) -> list[str]:
        if not state.separated:
            return ["main"]
        if layer == "auto":
            return [self.auto_layer] if self.auto_layer != "both" else ["starless", "stars"]
        if layer == "main":
            return ["starless", "stars"]
        return [layer] if layer in state.layers else ["starless"]

    def run(self, state: State, params: dict, layer: str, mask: dict | None, ctx: Ctx) -> State:
        out = state.copy()
        ctx.notes = state.notes
        for name in self.target_layers(state, layer):
            ctx.layer = name
            img = state.layers[name]
            linear = state.phase.get(name, "linear") == "linear"
            if self.phase == "linear" and not linear:
                out.warnings.append(T(f"「{self.title.zh}」通常在拉伸前做（這一層已經拉伸）",
                                      f"{self.title.en} is meant for linear data (this layer is stretched)"))
            elif self.phase == "nonlinear" and linear:
                out.warnings.append(T(f"「{self.title.zh}」通常在拉伸後做（這一層還是線性）",
                                      f"{self.title.en} is meant for stretched data (this layer is linear)"))
            p = ops.planes(img)
            res = ops.planes(self.apply(p, params, ctx, linear))
            m = masks.build(mask, p, linear, ctx.scale, ctx.cache, ref=ctx.whole(p) if ctx.ref is not None else None,
                            origin=ctx.origin) if self.maskable else None
            res = ops.planes(masks.blend(p, res, m))
            out.phase[name] = self.out_phase(linear)
            if out.phase[name] != "linear":
                # 拉伸後的值是 0–1：銳化、降噪會稍微超出（以前最後一步合成星點會夾回來，現在不拔星就沒有了）
                res = np.clip(res, 0.0, 1.0)
            out.layers[name] = np.ascontiguousarray(res, np.float32)
            ctx.check()
        return out

    def out_phase(self, linear: bool) -> str:
        return "linear" if linear else "nonlinear"


# ---------------------------------------------------------------------- 控制項


def slider(key: str, label: T, lo: float, hi: float, step: float = 0.01, fmt: str = "{:.2f}", show=None,
           help: T | None = None) -> dict:
    return {"type": "slider", "key": key, "label": label, "lo": lo, "hi": hi, "step": step, "fmt": fmt,
            "show": show, "help": help}


def toggle(key: str, label: T, show=None, help: T | None = None) -> dict:
    return {"type": "toggle", "key": key, "label": label, "show": show, "help": help}


def choice(key: str, label: T, options: list[tuple[str, T]], show=None, help: T | None = None) -> dict:
    return {"type": "choice", "key": key, "label": label, "options": options, "show": show, "help": help}


def section(label: T, show=None) -> dict:
    return {"type": "section", "label": label, "show": show}


def special(kind: str, key: str = "", label: T | None = None, show=None) -> dict:
    """介面另外處理的控制項：brush（筆刷）、eyedropper（滴管）、wheels（色輪）、curves（曲線）、view（檢視切換）、crop。"""
    return {"type": kind, "key": key, "label": label, "show": show}


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    REGISTRY[tool.kind] = tool
    return tool


def get(kind: str) -> Tool:
    _load()
    return REGISTRY[kind]


def all_tools() -> list[Tool]:
    _load()
    return list(REGISTRY.values())


_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import ai, color, detail, geometry, gradient, noise, stars, stretch  # noqa: F401
