"""星點形狀（點擴散函數，PSF）：量測、飽和星核、橢圓高斯核。

反摺積要知道星點實際的形狀才能讓星點變圓：同一張圖裡，中央的星可能是圓的、角落被像差拉長，
或整張都朝同一個方向拉長（追蹤、風）。這裡把畫面分成 N×N 區，每區用亮、沒飽和、旁邊沒有別的星的星
量「加權二階矩」（自適應高斯窗，修正掉窗的影響），得到橢圓高斯的長軸、短軸、方向。

飽和星核：WBPP 疊出來的 master 每顆飽和星停在不同的高度（0.93～0.97），不是整張的最大值，
所以用「接近該色版上限、而且周圍一圈都差不多平」判斷，不跟整張最大值比。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------- 飽和


def saturation_mask(planes: np.ndarray, level: float = 0.88, tops: list[float] | None = None) -> np.ndarray:
    """飽和的平頂星核（任一色版）：值 ≥ level × 該色版上限，而且 3×3 最小值也 ≥ (level − 0.03) × 上限。
    往外擴 1 px（平頂邊緣那圈也是被截過的）。tops：各色版的上限（預覽區只有一塊時用整張的）。"""
    p = planes if planes.ndim == 3 else planes[None]
    out = np.zeros(p.shape[1:], bool)
    for i, c in enumerate(p):
        top = float(c.max()) if tops is None else float(tops[i])
        if top <= 0:
            continue
        cand = c >= level * top
        if not cand.any():
            continue
        flat = ndimage.minimum_filter(c, size=3) >= (level - 0.03) * top
        out |= cand & flat
    return ndimage.binary_dilation(out, iterations=1) if out.any() else out


# ---------------------------------------------------------------------- 量測


@dataclass
class PSFCell:
    sigma_major: float      # px
    sigma_minor: float
    angle: float            # 長軸方向，度（x 軸往 y 軸轉，影像座標 y 向下）
    n: int                  # 用了幾顆星

    @property
    def fwhm_major(self) -> float:
        return 2.3548 * self.sigma_major

    @property
    def fwhm_minor(self) -> float:
        return 2.3548 * self.sigma_minor

    @property
    def ellipticity(self) -> float:
        return 1.0 - self.sigma_minor / max(self.sigma_major, 1e-9)

    def cov(self) -> np.ndarray:
        t = math.radians(self.angle)
        r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        return r @ np.diag([self.sigma_major ** 2, self.sigma_minor ** 2]) @ r.T


@dataclass
class PSFGrid:
    cells: list[list[PSFCell | None]]     # [row][col]
    shape: tuple[int, int]                # 量測時的影像大小 (H, W)
    overall: PSFCell | None               # 整張的中位數

    def at(self, row: int, col: int) -> PSFCell | None:
        c = self.cells[row][col]
        return c if c is not None else self.overall


def _moments(cut: np.ndarray, sigma_w: float) -> np.ndarray | None:
    """自適應高斯窗的二階矩 → 修正掉窗的影響後的共變異數 (2×2)。cut 已扣背景、中心是星心。"""
    r = cut.shape[0] // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    cy = cx = 0.0
    cov_w = np.eye(2) * sigma_w ** 2
    m = None
    for _ in range(5):
        inv = np.linalg.inv(cov_w)
        dy, dx = yy - cy, xx - cx
        w = np.exp(-0.5 * (inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy))
        f = np.maximum(cut, 0) * w
        s = f.sum()
        if s <= 0:
            return None
        cx, cy = float((f * xx).sum() / s), float((f * yy).sum() / s)
        dy, dx = yy - cy, xx - cx
        m = np.array([[(f * dx * dx).sum(), (f * dx * dy).sum()], [(f * dx * dy).sum(), (f * dy * dy).sum()]]) / s
        cov_w = 2 * m                                       # 窗跟著星的形狀調（自適應）
        if abs(cx) > r / 2 or abs(cy) > r / 2:
            return None
    # 觀測到的加權矩 M＝(C⁻¹＋W⁻¹)⁻¹ → C＝(M⁻¹ − W⁻¹)⁻¹
    try:
        inv_c = np.linalg.inv(m) - np.linalg.inv(cov_w)
        c = np.linalg.inv(inv_c)
    except np.linalg.LinAlgError:
        return None
    ev = np.linalg.eigvalsh(c)
    if ev[0] <= 0.05 or ev[1] > 400:
        return None
    return c


def measure_stars(L: np.ndarray, max_stars: int = 3000,
                  positions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """挑亮、沒飽和、孤立的星，回傳 (y, x, 共變異數 (n,2,2))。positions＝(y, x, 半徑)：已經挑好的星
    （開檔對齊三色時挑過），只重量形狀（找星要 2～3 秒）。"""
    h, w = L.shape
    if positions is not None:
        out_c, keep = [], []
        for i, (y, x, rad) in enumerate(zip(*positions)):
            r = int(max(6, min(16, round(rad * 3))))
            y, x = int(round(y)), int(round(x))
            if y < r + 2 or x < r + 2 or y >= h - r - 2 or x >= w - r - 2:
                continue
            cut = L[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
            ring = np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]])
            c = _moments(cut - np.median(ring), max(1.0, rad * 0.8))
            if c is not None:
                out_c.append(c)
                keep.append(i)
        ys, xs = positions[0][keep], positions[1][keep]
        return ys.astype(float), xs.astype(float), np.array(out_c).reshape(-1, 2, 2)
    from apu_processing import detail as pdetail
    stars = pdetail.find_stars(L, 8.0)
    if len(stars.y) == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 2, 2))
    sat = saturation_mask(L)
    bg = ndimage.median_filter(L[::4, ::4], size=9)
    sample = (L[::4, ::4] - bg)
    noise = 1.4826 * float(np.median(np.abs(sample - np.median(sample)))) or 1e-9
    ys, xs = stars.y, stars.x
    # 孤立：最近的鄰居夠遠
    pts = np.column_stack([ys, xs])
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2)
    nearest = dist[:, 1] if len(pts) > 1 else np.full(len(pts), 1e9)
    order = np.argsort(-stars.radius)
    out_y, out_x, out_c, out_r = [], [], [], []
    for i in order:
        r = int(max(6, min(16, round(stars.radius[i] * 3))))
        y, x = int(round(ys[i])), int(round(xs[i]))
        if y < r + 2 or x < r + 2 or y >= h - r - 2 or x >= w - r - 2 or nearest[i] < 2 * r:
            continue
        if sat[y - 2:y + 3, x - 2:x + 3].any():
            continue
        cut = L[y - r:y + r + 1, x - r:x + r + 1].astype(np.float64)
        ring = np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]])
        cut = cut - np.median(ring)
        if cut.max() < 30 * noise:
            continue
        c = _moments(cut, max(1.0, stars.radius[i] * 0.8))
        if c is None:
            continue
        out_y.append(y)
        out_x.append(x)
        out_c.append(c)
        out_r.append(stars.radius[i])
        if len(out_y) >= max_stars:
            break
    measure_stars.last_radius = np.array(out_r, float)
    return np.array(out_y, float), np.array(out_x, float), np.array(out_c).reshape(-1, 2, 2)


def _cell(covs: np.ndarray) -> PSFCell | None:
    if len(covs) < 5:
        return None
    c = np.median(covs, axis=0)                       # 各分量取中位數，壓掉異常的星
    ev, vec = np.linalg.eigh(c)
    major = vec[:, 1]
    angle = math.degrees(math.atan2(major[1], major[0])) % 180
    return PSFCell(math.sqrt(max(ev[1], 1e-6)), math.sqrt(max(ev[0], 1e-6)), angle, len(covs))


def measure_grid(L: np.ndarray, n: int = 3, positions=None) -> PSFGrid:
    h, w = L.shape
    ys, xs, covs = measure_stars(L, positions=positions)
    cells: list[list[PSFCell | None]] = []
    for r in range(n):
        row = []
        for c in range(n):
            sel = (ys >= r * h / n) & (ys < (r + 1) * h / n) & (xs >= c * w / n) & (xs < (c + 1) * w / n)
            row.append(_cell(covs[sel]))
        cells.append(row)
    return PSFGrid(cells, (h, w), _cell(covs))


# ---------------------------------------------------------------------- 核


def elliptical_kernel(cov: np.ndarray, radius: int | None = None) -> np.ndarray:
    """共變異數 cov 的橢圓高斯核（總和 1，點對稱）。"""
    ev = np.linalg.eigvalsh(cov)
    if radius is None:
        radius = int(math.ceil(3.5 * math.sqrt(max(ev[1], 0.04))))
    radius = max(1, min(radius, 25))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    inv = np.linalg.inv(cov)
    k = np.exp(-0.5 * (inv[0, 0] * xx * xx + 2 * inv[0, 1] * xx * yy + inv[1, 1] * yy * yy))
    return (k / k.sum()).astype(np.float32)


# ---------------------------------------------------------------------- 位置與座標


def cov_at(grid: PSFGrid, y: float, x: float) -> np.ndarray | None:
    """量測影像座標 (y, x) 的共變異數：各區中心之間雙線性內插（缺的區用整張的）。"""
    if grid.overall is None:
        return None
    h, w = grid.shape
    n = len(grid.cells)
    fy = np.clip(y / h * n - 0.5, 0, n - 1)
    fx = np.clip(x / w * n - 0.5, 0, n - 1)
    r0, c0 = int(np.floor(fy)), int(np.floor(fx))
    r1, c1 = min(r0 + 1, n - 1), min(c0 + 1, n - 1)
    ty, tx = fy - r0, fx - c0
    out = np.zeros((2, 2))
    for r, c, wgt in ((r0, c0, (1 - ty) * (1 - tx)), (r0, c1, (1 - ty) * tx), (r1, c0, ty * (1 - tx)), (r1, c1, ty * tx)):
        out += wgt * grid.at(r, c).cov()
    return out


_JACOBIAN = {"flip_h": np.array([[-1.0, 0.0], [0.0, 1.0]]), "flip_v": np.array([[1.0, 0.0], [0.0, -1.0]]),
             "rot": np.array([[0.0, 1.0], [-1.0, 0.0]])}


def cov_in_frame(grid: PSFGrid, y: float, x: float, frame_ops: list | None) -> np.ndarray | None:
    """目前這一步的座標 (y, x)（完整尺寸）的星點共變異數 (x, y 順序)。frame_ops：前面裁切／旋轉／翻轉拆成的
    基本動作 [(動作, 動作前的大小)…]（chain._geometry_ops）；先把位置反推回原圖量，再把橢圓跟著轉回來。"""
    from .tools import geometry as tgeometry
    sy, sx = float(y), float(x)
    if frame_ops:
        b = (sy, sx, sy + 1, sx + 1)
        for op, cin in reversed(frame_ops):
            b = tgeometry.box_inverse(op, b, cin)
        sy, sx = min(b[0], b[2]), min(b[1], b[3])
    cov = cov_at(grid, sy, sx)
    if cov is None:
        return None
    for op, _cin in frame_ops or []:
        j = _JACOBIAN.get(op[0])
        if j is not None:
            cov = j @ cov @ j.T
    return cov


def round_kernel(cov: np.ndarray) -> np.ndarray:
    """「變圓」用的核：只有長軸方向多出來的那部分（變異數 λmax − λmin），還原後長軸跟短軸一樣、星不會變小。
    NGC2244 右側的星橢圓率 0.19～0.23 → 0.01～0.07（20 次迭代）。"""
    ev, vec = np.linalg.eigh(cov)
    k = vec @ np.diag([0.09, max(ev[1] - ev[0], 0.09)]) @ vec.T
    return elliptical_kernel(k)


def sharpen_kernel(cov: np.ndarray, core_limit: float = 0.8) -> np.ndarray:
    """銳化用的圓形核：短軸 σ 的 core_limit 倍（跟 Processing 的反摺積一樣保守，不會振盪）。"""
    ev = np.linalg.eigvalsh(cov)
    return elliptical_kernel(np.eye(2) * max(0.09, core_limit ** 2 * ev[0]))


def tiled_rl(x: np.ndarray, kernel_at, iterations: int, valid: np.ndarray | None = None, tile: int = 1024,
             pad: int = 64, cancel=lambda: False) -> np.ndarray:
    """分塊 Richardson–Lucy：每塊用自己位置的核（kernel_at(中心 y, 中心 x) → 核），重疊 pad px 漸變混合。"""
    from . import backend
    h, w = x.shape
    ny, nx = max(1, round(h / tile)), max(1, round(w / tile))
    if ny == 1 and nx == 1:
        return backend.richardson_lucy_kernel(x, kernel_at(h / 2, w / 2), iterations, valid, cancel)
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    acc = np.zeros((h, w), np.float64)
    wsum = np.zeros((h, w), np.float64)
    for i in range(ny):
        for j in range(nx):
            if cancel():
                return x
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            py0, py1, px0, px1 = max(0, y0 - pad), min(h, y1 + pad), max(0, x0 - pad), min(w, x1 + pad)
            k = kernel_at((y0 + y1) / 2, (x0 + x1) / 2)
            v = None if valid is None else valid[py0:py1, px0:px1]
            est = backend.richardson_lucy_kernel(x[py0:py1, px0:px1], k, iterations, v, cancel)
            # 權重：塊內 1，往重疊區外緣線性降到 0（影像邊緣不降）
            def ramp(n0, n1, a, b, lo_edge, hi_edge):
                r = np.ones(n1 - n0)
                if not lo_edge:
                    r[:2 * (a - n0)] = np.linspace(0, 1, 2 * (a - n0)) if a > n0 else 1
                if not hi_edge:
                    m = 2 * (n1 - b)
                    if m > 0:
                        r[-m:] = np.minimum(r[-m:], np.linspace(1, 0, m))
                return np.maximum(r, 1e-3)
            wy = ramp(py0, py1, y0, y1, y0 == 0, y1 == h)
            wx = ramp(px0, px1, x0, x1, x0 == 0, x1 == w)
            wt = np.outer(wy, wx)
            acc[py0:py1, px0:px1] += est * wt
            wsum[py0:py1, px0:px1] += wt
    return (acc / np.maximum(wsum, 1e-9)).astype(np.float32)


# ---------------------------------------------------------------------- 三色對齊（橫向色差）


def _centroid(plane: np.ndarray, y: float, x: float, r: int = 6) -> tuple[float, float] | None:
    """高斯窗加權的重心，窗跟著移到上一次的重心再算（固定的窗會把重心往窗中心拉，量到的位移只有六成）。"""
    yi, xi = int(round(y)), int(round(x))
    cut = plane[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float64)
    if cut.shape != (2 * r + 1, 2 * r + 1):
        return None
    ring = np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]])
    cut = np.maximum(cut - np.median(ring), 0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    cy = cx = 0.0
    for _ in range(8):
        w = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0) * cut
        s = w.sum()
        if s <= 0:
            return None
        ny, nx = float((w * yy).sum() / s), float((w * xx).sum() / s)
        if abs(ny - cy) < 1e-3 and abs(nx - cx) < 1e-3:
            cy, cx = ny, nx
            break
        cy, cx = ny, nx
        if abs(cy) > r / 2 or abs(cx) > r / 2:
            return None
    return yi + cy, xi + cx


def channel_offsets(planes: np.ndarray, ref: int = 1, stars=None) -> list[np.ndarray | None]:
    """各色版相對參考色版的星點位置差，用線性模型 d＝a0＋a1·x'＋a2·y'（x', y' 是 −0.5～0.5 的座標）描述，
    回傳每個色版的係數 [[dy 三個], [dx 三個]]（參考色版是 None）。星太少時全部 None。stars＝(y, x)。"""
    C, h, w = planes.shape
    ys, xs = stars if stars is not None else measure_stars(planes.mean(axis=0), 1500)[:2]
    out: list[np.ndarray | None] = [None] * C
    if len(ys) < 30:
        return out
    ref_c = [(_centroid(planes[ref], y, x)) for y, x in zip(ys, xs)]
    for c in range(C):
        if c == ref:
            continue
        rows, dys, dxs = [], [], []
        for (y, x), rc in zip(zip(ys, xs), ref_c):
            cc = _centroid(planes[c], y, x)
            if rc is None or cc is None:
                continue
            rows.append([1.0, x / w - 0.5, y / h - 0.5])
            dys.append(cc[0] - rc[0])
            dxs.append(cc[1] - rc[1])
        if len(rows) < 30:
            continue
        A = np.array(rows)
        coef = []
        for d in (np.array(dys), np.array(dxs)):
            keep = np.ones(len(d), bool)
            for _ in range(4):                               # 去掉離群的星（雙星、有東西擋到）
                sol, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
                res = d - A @ sol
                mad = 1.4826 * np.median(np.abs(res[keep])) + 1e-6
                keep = np.abs(res) < 3 * mad
            coef.append(sol)
        out[c] = np.array(coef)
    return out


def align_channels(planes: np.ndarray, ref: int = 1, min_shift: float = 0.08) -> tuple[np.ndarray, dict]:
    """把各色版對齊參考色版（次像素、三次內插）。最大位移小於 min_shift px 就不動。回傳 (影像, 資訊)。
    NGC2244（OSC）紅色版相對綠色版放射狀往外偏 0.2～0.3 px（橫向色差）：拉伸、加飽和度後亮星一邊紅一邊青。"""
    if planes.ndim != 3 or planes.shape[0] < 3:
        return planes, {}
    C, h, w = planes.shape
    ys, xs, _c = measure_stars(planes.mean(axis=0), 1500)
    radius = measure_stars.last_radius
    coefs = channel_offsets(planes, ref, (ys, xs))
    out = planes
    info: dict = {"stars": (ys, xs, radius)}
    corners = np.array([[1, -0.5, -0.5], [1, 0.5, -0.5], [1, -0.5, 0.5], [1, 0.5, 0.5], [1, 0.0, 0.0]])
    for c, coef in enumerate(coefs):
        if coef is None:
            continue
        shift = float(np.max(np.hypot(corners @ coef[0], corners @ coef[1])))
        info[c] = round(shift, 2)
        if shift < min_shift:
            continue
        if out is planes:
            out = planes.copy()
        yy = (np.arange(h, dtype=np.float32) / h - 0.5)[:, None]
        xx = (np.arange(w, dtype=np.float32) / w - 0.5)[None, :]
        dy = (coef[0][0] + coef[0][1] * xx + coef[0][2] * yy).astype(np.float32)
        dx = (coef[1][0] + coef[1][1] * xx + coef[1][2] * yy).astype(np.float32)
        out[c] = _warp(planes[c], dy, dx)
    return out, info


def _warp(plane: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """在 (y+dy, x+dx) 取樣（三次內插）。有 N 卡用 GPU。"""
    from . import backend
    h, w = plane.shape
    cp = backend.gpu()
    if cp is not None:
        try:
            import cupyx.scipy.ndimage as cnd
            gy = cp.arange(h, dtype=cp.float32)[:, None] + cp.asarray(dy)
            gx = cp.arange(w, dtype=cp.float32)[None, :] + cp.asarray(dx)
            coords = cp.stack([cp.broadcast_to(gy, (h, w)), cp.broadcast_to(gx, (h, w))])
            res = cnd.map_coordinates(cp.asarray(plane, dtype=cp.float32), coords, order=3, mode="nearest")
            out = cp.asnumpy(res).astype(np.float32)
            del coords, res
            cp.get_default_memory_pool().free_all_blocks()
            return out
        except Exception:  # noqa: BLE001  GPU 不行就用 CPU
            pass
    gy = np.arange(h, dtype=np.float32)[:, None] + dy
    gx = np.arange(w, dtype=np.float32)[None, :] + dx
    return ndimage.map_coordinates(plane, [np.broadcast_to(gy, (h, w)), np.broadcast_to(gx, (h, w))],
                                   order=3, mode="nearest").astype(np.float32)


measure_stars.last_radius = np.zeros(0)
