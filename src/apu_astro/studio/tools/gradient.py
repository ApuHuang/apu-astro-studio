"""去光梯度：在天空取樣、擬合平滑的背景面、扣掉（或除掉）。

比 Processing 多的：
- 背景面可以選「多項式」（穩、適合單純的光害）或「彈性曲面」（薄板樣條，貼得住不規則、多方向的光害）
- 筆刷塗掉不要取樣的地方（星雲、銀河）；軟體也會自動排除比背景亮的格子
- 可以看「背景模型」與「被扣掉的部分」，確認星雲沒有被當成光害
- 相減（光害）或相除（殘留的暗角）
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre
from scipy import ndimage
from scipy.interpolate import RBFInterpolator, RectBivariateSpline

from .. import masks, ops
from . import T, Ctx, Tool, choice, register, section, slider, special


def _cells(plane: np.ndarray, cell: int) -> np.ndarray:
    h, w = plane.shape
    rows, cols = max(1, h // cell), max(1, w // cell)
    x = plane[:rows * cell, :cols * cell]
    sub = max(1, cell // 8)
    if sub > 1:
        x = x.reshape(rows * cell // sub, sub, cols * cell // sub, sub).mean(axis=(1, 3))
    k = cell // sub
    blocks = x.reshape(rows, k, cols, k).transpose(0, 2, 1, 3).reshape(rows, cols, k * k)
    return np.median(blocks, axis=2).astype(np.float64)


def _coords(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n * 2 - 1


def _poly_fit(values: np.ndarray, mask: np.ndarray, degree: int) -> np.ndarray:
    rows, cols = values.shape
    yy, xx = np.meshgrid(_coords(rows), _coords(cols), indexing="ij")
    V = legendre.legvander2d(yy[mask], xx[mask], [degree, degree])
    keep = np.array([i + j <= degree for i in range(degree + 1) for j in range(degree + 1)])
    coef = np.zeros((degree + 1) * (degree + 1))
    sol, *_ = np.linalg.lstsq(V[:, keep], values[mask], rcond=None)
    coef[keep] = sol
    return coef.reshape(degree + 1, degree + 1)


def _poly_grid(coef: np.ndarray, rows: int, cols: int) -> np.ndarray:
    yy, xx = np.meshgrid(_coords(rows), _coords(cols), indexing="ij")
    return legendre.legval2d(yy, xx, coef)


def _poly_full(coef: np.ndarray, h: int, w: int, cell: int) -> np.ndarray:
    ys = (np.arange(h) + 0.5) / (h // cell * cell) * 2 - 1
    xs = (np.arange(w) + 0.5) / (w // cell * cell) * 2 - 1
    return (legendre.legvander(ys, coef.shape[0] - 1) @ coef @ legendre.legvander(xs, coef.shape[1] - 1).T
            ).astype(np.float32)


def _rbf(values: np.ndarray, mask: np.ndarray, smooth: float) -> np.ndarray:
    """薄板樣條：smooth 0（貼緊取樣點）～1（很平滑）。回傳格子上的值。"""
    rows, cols = values.shape
    yy, xx = np.meshgrid(_coords(rows) * rows / max(rows, cols), _coords(cols) * cols / max(rows, cols),
                         indexing="ij")
    pts = np.column_stack([yy[mask], xx[mask]])
    v = values[mask]
    scale = float(np.std(v)) or 1.0
    # 平滑量跟取樣點數成正比，滑桿用對數感受
    s = (10 ** (6 * smooth - 4)) * len(v)
    f = RBFInterpolator(pts, (v - v.mean()) / scale, kernel="thin_plate_spline", smoothing=s, degree=1)
    grid = np.column_stack([yy.ravel(), xx.ravel()])
    return f(grid).reshape(rows, cols) * scale + v.mean()


def _upsample(grid: np.ndarray, h: int, w: int, cell: int) -> np.ndarray:
    """格子中心的值 → 每個像素（三次樣條插值，格子外延伸）。"""
    rows, cols = grid.shape
    yy = np.clip((np.arange(h) + 0.5) / cell - 0.5, 0, rows - 1)
    xx = np.clip((np.arange(w) + 0.5) / cell - 0.5, 0, cols - 1)
    k = min(3, rows - 1, cols - 1)
    if k < 1:
        return np.full((h, w), float(grid.mean()), np.float32)
    # 張量積樣條：可分離，整張 2000 萬像素也只要一下
    spline = RectBivariateSpline(np.arange(rows), np.arange(cols), grid, kx=k, ky=k)
    return spline(yy, xx).astype(np.float32)


class Gradient(Tool):
    kind = "gradient"
    title = T("去光梯度", "Gradient Removal")
    help = T("在天空取樣、擬合平滑的背景面再扣掉。用筆刷塗掉星雲與銀河不要取樣；看「被扣掉的部分」確認沒有扣到星雲。",
             "Samples the sky, fits a smooth background and removes it. Paint over nebulae so they aren't sampled; "
             "check “Removed” to make sure no nebula was taken out.")
    phase = "linear"
    icon = "gradient"
    maskable = False
    auto_layer = "main"

    def defaults(self) -> dict:
        return {"method": "poly", "degree": 4, "smooth": 0.5, "cell": 64, "protect": 2.0, "mode": "subtract",
                "exclude": [], "brush": 150, "view": "result"}

    def controls(self) -> list[dict]:
        rbf = lambda p: p.get("method") == "rbf"  # noqa: E731
        return [
            special("view", "view", T("顯示", "Show")),
            choice("method", T("背景面", "Model"), [("poly", T("多項式", "Polynomial")),
                                                   ("rbf", T("彈性曲面", "Flexible"))],
                   help=T("多項式：單純的光害（一個方向變亮）最穩。彈性曲面：不規則、多方向的光害（Bortle 5 以上常見）。",
                          "Polynomial: steady for simple gradients. Flexible (thin-plate spline): follows irregular, "
                          "multi-directional light pollution.")),
            slider("degree", T("階數", "Degree"), 1, 6, 1, "{:.0f}", show=lambda p: not rbf(p)),
            slider("smooth", T("平滑度", "Smoothness"), 0, 1, 0.05, "{:.2f}", show=rbf),
            slider("protect", T("星雲保護", "Nebula protection"), 0.8, 4.0, 0.1, "{:.1f}",
                   help=T("比背景面亮多少就不當成天空。數字小＝排除得多（保護暗淡星雲），太小會取不到樣。",
                          "How much brighter than the model a cell may be and still count as sky. Lower protects faint "
                          "nebulosity more.")),
            slider("cell", T("取樣格大小", "Sample size"), 24, 160, 8, "{:.0f} px"),
            choice("mode", T("方式", "Correction"), [("subtract", T("相減（光害）", "Subtract (light pollution)")),
                                                    ("divide", T("相除（暗角）", "Divide (vignetting)"))]),
            section(T("排除區域", "Exclusion")),
            special("brush", "exclude", T("用筆刷塗掉不要取樣的地方（右鍵擦掉）", "Paint areas not to sample (right-drag erases)")),
            slider("brush", T("筆刷大小", "Brush size"), 20, 600, 10, "{:.0f} px"),
        ]

    def apply(self, img: np.ndarray, params: dict, ctx: Ctx, linear: bool) -> np.ndarray:
        C, h, w = img.shape
        if ctx.ref is not None:
            # 預覽區：背景面用整張（縮圖）擬合，再取這一塊的位置
            whole = ctx.whole(img)
            surfaces_w = self._fit(whole, params, ctx.whole_scale, ctx, view=False)
            if surfaces_w is None:
                return img
            f = ctx.whole_scale
            ox, oy = ctx.origin
            ys = np.clip((oy + np.arange(h) + 0.5) / f - 0.5, 0, surfaces_w.shape[1] - 1)
            xs = np.clip((ox + np.arange(w) + 0.5) / f - 0.5, 0, surfaces_w.shape[2] - 1)
            surfaces = np.stack([RectBivariateSpline(np.arange(sw.shape[0]), np.arange(sw.shape[1]), sw, kx=1, ky=1)(
                ys, xs) for sw in surfaces_w]).astype(np.float32)
            med = np.median(surfaces_w.reshape(C, -1), axis=1)[:, None, None]
        else:
            surfaces = self._fit(img, params, ctx.scale, ctx, view=True)
            if surfaces is None:
                return img
            med = np.median(surfaces.reshape(C, -1), axis=1)[:, None, None]
        if params.get("mode") == "divide":
            return (img / np.maximum(surfaces / np.maximum(med, 1e-9), 1e-3)).astype(np.float32)
        removed = surfaces - med
        if ctx.ref is None:
            ctx.view["removed"] = removed
        return (img - removed).astype(np.float32)

    def _fit(self, img: np.ndarray, params: dict, scale: float, ctx: Ctx, view: bool) -> np.ndarray | None:
        """取樣、擬合，回傳跟 img 一樣大的背景面 (C, h, w)；取不到樣時 None。"""
        C, h, w = img.shape
        cell = int(max(6, round(float(params.get("cell", 64)) / scale)))
        cell = int(max(6, min(cell, min(h, w) // 12)))   # 小影像也至少切成 12×12 格
        cell -= cell % max(1, cell // 8)                  # 要是 _cells 內部小格的整數倍（例如 21 → 20）
        cells = np.stack([_cells(c, cell) for c in img])
        L = cells.mean(axis=0)
        rows, cols = L.shape
        valid = np.ones((rows, cols), bool)
        my, mx = max(1, rows // 50), max(1, cols // 50)
        valid[:my] = valid[-my:] = False
        valid[:, :mx] = valid[:, -mx:] = False
        valid &= np.all(cells != 0, axis=0)
        excl = masks.rasterize(params.get("exclude", []), (h, w), scale)
        if excl.any():
            frac = excl[:rows * cell, :cols * cell].reshape(rows, cell, cols, cell).mean(axis=(1, 3))
            valid &= frac < 0.2
        cov = ctx.meta.get("coverage_mask")
        if cov is not None:
            cm = cov if cov.shape == (h, w) else ndimage.zoom(cov.astype(np.float32),
                                                               (h / cov.shape[0], w / cov.shape[1]), order=0) > 0.5
            frac = cm[:rows * cell, :cols * cell].reshape(rows, cell, cols, cell).mean(axis=(1, 3))
            if (valid & (frac > 0.99)).sum() >= 24:
                valid &= frac > 0.99
        if valid.sum() < 12:
            if view:
                ctx.view["model"] = None
            return None
        degree = int(params.get("degree", 4))
        protect = float(params.get("protect", 2.0))
        mask = valid & (L <= np.quantile(L[valid], 0.5))
        rbf = params.get("method") == "rbf"
        for _ in range(8):
            ctx.check()
            surf = _rbf(L, mask, float(params.get("smooth", 0.5))) if rbf else _poly_grid(_poly_fit(L, mask, degree),
                                                                                          rows, cols)
            resid = L - surf
            r = resid[mask]
            s = float(1.4826 * np.median(np.abs(r - np.median(r)))) or 1e-12
            new = valid & (resid < protect * s) & (resid > -4 * s)
            if new.sum() < 12 or np.array_equal(new, mask):
                break
            mask = new
        surfaces = np.empty((C, h, w), np.float32)
        for c in range(C):
            ctx.check()
            if rbf:
                surfaces[c] = _upsample(_rbf(cells[c], mask, float(params.get("smooth", 0.5))), h, w, cell)
            else:
                surfaces[c] = _poly_full(_poly_fit(cells[c], mask, degree), h, w, cell)
        if view:
            ctx.view["model"] = surfaces
            ctx.view["sky_cells"] = (mask, cell)
        return surfaces


register(Gradient())

__all__ = ["Gradient", "ops"]
