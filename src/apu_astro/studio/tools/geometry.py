"""裁切與旋轉：90° 旋轉、翻轉、矩形裁切（在畫面上拖曳畫框）。所有圖層一起做。"""

from __future__ import annotations

import numpy as np

from .. import ops
from . import T, Ctx, RegionOutside, State, Tool, register, slider, special, toggle

# ---------------------------------------------------------------------- 座標換算（預覽區用）
# 一步裁切與旋轉拆成幾個基本動作：("crop", y0, x0, y1, x1)、("flip_h",)、("flip_v",)、("rot",)（逆時針 90°）。
# 框都是 (y0, x0, y1, x1)，完整尺寸的像素；canvas 是整張的 (H, W)。


def plan(params: dict, canvas: tuple[int, int], edges: tuple[int, int, int, int] | None) -> list[tuple]:
    """這一步在完整尺寸上的基本動作。edges：自動裁邊量到的 (上, 下, 左, 右)（完整尺寸）。"""
    H, W = canvas
    out: list[tuple] = []
    if params.get("auto_edges") and edges:
        top, bottom, left, right = edges
        if top + bottom < H - 8 and left + right < W - 8 and any(edges):
            out.append(("crop", top, left, H - bottom, W - right))
            H, W = H - top - bottom, W - left - right
    box = params.get("box")
    if box:
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = int(max(0, min(W - 1, round(x0)))), int(max(1, min(W, round(x1))))
        y0, y1 = int(max(0, min(H - 1, round(y0)))), int(max(1, min(H, round(y1))))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out.append(("crop", y0, x0, y1, x1))
            H, W = y1 - y0, x1 - x0
    if params.get("flip_h"):
        out.append(("flip_h",))
    if params.get("flip_v"):
        out.append(("flip_v",))
    out += [("rot",)] * (int(params.get("turns", 0)) % 4)
    return out


def canvas_after(op: tuple, canvas: tuple[int, int]) -> tuple[int, int]:
    if op[0] == "crop":
        return op[3] - op[1], op[4] - op[2]
    if op[0] == "rot":
        return canvas[1], canvas[0]
    return canvas


def box_forward(op: tuple, box: tuple, canvas: tuple[int, int]) -> tuple:
    y0, x0, y1, x1 = box
    H, W = canvas
    if op[0] == "crop":
        _k, cy0, cx0, cy1, cx1 = op
        return (max(y0, cy0) - cy0, max(x0, cx0) - cx0, min(y1, cy1) - cy0, min(x1, cx1) - cx0)
    if op[0] == "flip_h":
        return (y0, W - x1, y1, W - x0)
    if op[0] == "flip_v":
        return (H - y1, x0, H - y0, x1)
    return (W - x1, y0, W - x0, y1)                      # rot


def box_inverse(op: tuple, box: tuple, canvas_in: tuple[int, int]) -> tuple:
    """box 是這個動作之後的座標，回傳動作之前的座標（canvas_in 是動作之前的整張大小）。"""
    y0, x0, y1, x1 = box
    H, W = canvas_in
    if op[0] == "crop":
        _k, cy0, cx0, _cy1, _cx1 = op
        return (y0 + cy0, x0 + cx0, y1 + cy0, x1 + cx0)
    if op[0] == "flip_h":
        return (y0, W - x1, y1, W - x0)
    if op[0] == "flip_v":
        return (H - y1, x0, H - y0, x1)
    return (x0, W - y1, x1, W - y0)                      # rot：之後的 (W-x1, y0, W-x0, y1)


def region_apply(op: tuple, p: np.ndarray, region: dict) -> tuple[np.ndarray, dict]:
    """對預覽區那一塊做一個基本動作，回傳 (新的那一塊, 新的位置)。整塊都被裁掉時丟 RegionOutside。"""
    box, inner, canvas = region["box"], region["inner"], region["canvas"]
    if op[0] == "crop":
        nb = box_forward(op, box, canvas)
        if nb[2] - nb[0] < 4 or nb[3] - nb[1] < 4:
            raise RegionOutside()
        _k, cy0, cx0, _cy1, _cx1 = op
        p = p[:, nb[0] + cy0 - box[0]:nb[2] + cy0 - box[0], nb[1] + cx0 - box[1]:nb[3] + cx0 - box[1]]
        ni = box_forward(op, inner, canvas)
        ni = (max(ni[0], nb[0]), max(ni[1], nb[1]), min(ni[2], nb[2]), min(ni[3], nb[3]))
        if ni[2] - ni[0] < 4 or ni[3] - ni[1] < 4:
            raise RegionOutside()
    else:
        nb, ni = box_forward(op, box, canvas), box_forward(op, inner, canvas)
        if op[0] == "flip_h":
            p = p[:, :, ::-1]
        elif op[0] == "flip_v":
            p = p[:, ::-1, :]
        else:
            p = np.rot90(p, 1, axes=(1, 2))
    return p, {"box": nb, "inner": ni, "canvas": canvas_after(op, canvas)}


class Geometry(Tool):
    kind = "geometry"
    title = T("裁切與旋轉", "Crop & Rotate")
    help = T("在畫面上拖曳畫出裁切框；疊圖邊緣覆蓋不足的地方建議裁掉，去光梯度會更準。",
             "Drag on the image to draw the crop. Cropping the poorly covered stacking edges helps gradient removal.")
    icon = "crop"
    maskable = False
    auto_layer = "main"

    def defaults(self) -> dict:
        return {"auto_edges": False, "turns": 0, "flip_h": False, "flip_v": False, "box": None}

    def controls(self) -> list[dict]:
        return [
            toggle("auto_edges", T("自動裁掉疊圖邊緣", "Auto-crop stacking edges"),
                   help=T("找出四邊疊圖時對不齊、覆蓋不足的那幾排（顏色或亮度明顯不同、或是 0）並裁掉。",
                          "Finds the rows and columns along the edges that are misaligned or poorly covered "
                          "(off in colour or brightness, or zero) and crops them.")),
            special("crop", "box", T("在畫面上拖曳畫裁切框", "Drag on the image to set the crop")),
            slider("turns", T("旋轉（每格 90°，逆時針）", "Rotate (90° steps, counter-clockwise)"), 0, 3, 1, "{:.0f}"),
            toggle("flip_h", T("左右翻轉", "Flip horizontally")),
            toggle("flip_v", T("上下翻轉", "Flip vertically")),
        ]

    @staticmethod
    def detect_edges(p: np.ndarray) -> tuple[int, int, int, int]:
        """四邊各要裁掉幾排。每一排（欄）跟往內 10～20 排的鄰近區域比：亮度、色彩比例、噪聲差太多就算壞邊。
        跟鄰近比而不是跟中央比：光害梯度、延伸到邊緣的星雲只會慢慢變，不會被誤判。"""
        C, h, w = p.shape
        L = p.mean(axis=0)
        _m, noise = (lambda a: (float(np.median(a)), float(np.median(np.abs(a - np.median(a)))) * 1.4826))(
            (L[h // 4: 3 * h // 4: 2, w // 4: 3 * w // 4: 2] - L[h // 4 + 1: 3 * h // 4 + 1: 2,
                                                                 w // 4: 3 * w // 4: 2]).ravel())
        noise = max(noise / np.sqrt(2), 1e-9)
        # 每個色版的天空與噪聲（找零星的異常低值：WBPP 邊緣常有一兩成的像素是 0，合成後變成很大的負值）
        cs = []
        for c in range(C):
            core = p[c, h // 4: 3 * h // 4: 3, w // 4: 3 * w // 4: 3]
            cm = float(np.median(core))
            cd = p[c, h // 4: 3 * h // 4: 2, w // 4: 3 * w // 4: 2] - p[c, h // 4 + 1: 3 * h // 4 + 1: 2,
                                                                         w // 4: 3 * w // 4: 2]
            cs.append((cm, max(float(np.median(np.abs(cd - np.median(cd)))) * 1.4826 / np.sqrt(2), 1e-9)))

        def outliers(line: np.ndarray, ref: np.ndarray) -> float:
            """這一排裡異常低（≤0 或比鄰近區域的中位數低 15σ 以上）的像素比例，取各色版最大的。"""
            out = 0.0
            for c in range(C):
                rm = float(np.median(ref[c]))
                out = max(out, float(((line[c] <= 0) | (line[c] < rm - 15 * cs[c][1])).mean()))
            return out

        def stats(lines: np.ndarray) -> tuple[float, list[float], float]:
            # lines：(C, k, n)；回傳 (亮度中位數, 各色比例, 噪聲)
            Ls = lines.mean(axis=0)
            m = float(np.median(Ls))
            cm = [float(np.median(lines[c])) for c in range(C)]
            tot = sum(cm) or 1e-9
            d = np.diff(Ls, axis=-1)
            return m, [c / tot for c in cm], float(np.median(np.abs(d))) * 1.4826 / np.sqrt(2)

        def side(get: callable, n_max: int, length: int) -> int:
            k = 0
            while k < n_max:
                line = get(k, k + 1)
                ref = get(k + 10, k + 20)
                m, r, nz = stats(line)
                mr, rr, nzr = stats(ref)
                tol = 6 * 1.25 * noise / np.sqrt(length) + 0.02 * abs(mr)
                bad = (not np.isfinite(m) or np.all(line == 0) or abs(m - mr) > tol or outliers(line, ref) > 0.002
                       or (C >= 3 and max(abs(a - b) for a, b in zip(r, rr)) > 0.05)
                       or nz < 0.3 * nzr or nz > 3 * nzr)
                if not bad:
                    break
                k += 1
            return k + (2 if k else 0)   # 多裁兩排，過渡區常常也不乾淨

        lim_h, lim_w = h // 8, w // 8
        top = side(lambda a, b: p[:, a:b, :], lim_h, w)
        bottom = side(lambda a, b: p[:, h - b:h - a, :], lim_h, w)
        left = side(lambda a, b: p[:, :, a:b].transpose(0, 2, 1), lim_w, h)
        right = side(lambda a, b: p[:, :, w - b:w - a].transpose(0, 2, 1), lim_w, h)
        return top, bottom, left, right

    @staticmethod
    def transform(p: np.ndarray, params: dict, scale: float, edges: tuple | None = None) -> np.ndarray:
        if params.get("auto_edges"):
            top, bottom, left, right = edges if edges is not None else Geometry.detect_edges(p)
            h, w = p.shape[1:]
            if top + bottom < h - 8 and left + right < w - 8:
                p = p[:, top:h - bottom, left:w - right]
        box = params.get("box")
        if box:
            x0, y0, x1, y1 = (float(v) / scale for v in box)
            h, w = p.shape[1:]
            x0, x1 = int(max(0, min(w - 1, round(x0)))), int(max(1, min(w, round(x1))))
            y0, y1 = int(max(0, min(h - 1, round(y0)))), int(max(1, min(h, round(y1))))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                p = p[:, y0:y1, x0:x1]
        if params.get("flip_h"):
            p = p[:, :, ::-1]
        if params.get("flip_v"):
            p = p[:, ::-1, :]
        t = int(params.get("turns", 0)) % 4
        if t:
            p = np.rot90(p, t, axes=(1, 2))
        return np.ascontiguousarray(p, np.float32)

    def run(self, state: State, params: dict, layer: str, mask: dict | None, ctx: Ctx) -> State:
        out = state.copy()
        first = next(iter(state.layers))
        if state.region is None:
            edges = None
            if params.get("auto_edges"):
                # 所有圖層用同一組邊（照第一層量）；記下完整尺寸的值，預覽區換算座標用
                edges = self.detect_edges(ops.planes(state.layers[first]))
                ctx.view["edges_full"] = tuple(int(round(v * ctx.scale)) for v in edges)
            for name, img in state.layers.items():
                out.layers[name] = self.transform(ops.planes(img), params, ctx.scale, edges)
            return out
        # 預覽區：自動裁邊照整張的縮圖量，換成完整尺寸的基本動作套在這一塊上
        edges = None
        if params.get("auto_edges"):
            key = ("edges",)
            if key not in ctx.cache:
                if ctx.full_input is not None:      # 輸入就是來源：跟完整解析度一樣量，位置才會完全對齊
                    ctx.cache[key] = self.detect_edges(ops.planes(ctx.full_input))
                elif ctx.ref is not None:
                    e = self.detect_edges(ops.planes(ctx.ref.layers[first]))
                    ctx.cache[key] = tuple(int(round(v * ctx.ref_scale)) for v in e)
            edges = ctx.cache.get(key)
        steps = plan(params, state.region["canvas"], edges)
        region = state.region
        for name, img in state.layers.items():
            p, region = ops.planes(img), state.region
            for op in steps:
                p, region = region_apply(op, p, region)
            out.layers[name] = np.ascontiguousarray(p, np.float32)
        out.region = region
        return out


register(Geometry())
