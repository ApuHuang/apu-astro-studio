"""運算後端：有 NVIDIA 顯示卡（CuPy）就用 GPU 做最重的部分（反摺積的模糊、大範圍卷積），沒有就用 CPU（SciPy）。

只包最常用的幾個函式；呼叫端拿 numpy 陣列進、numpy 陣列出，不用管資料在哪裡。
"""

from __future__ import annotations

import threading

import numpy as np
from scipy import ndimage

_lock = threading.Lock()
_cupy = None
_checked = False
enabled = True          # 介面上的「使用 GPU」開關


def gpu():
    """可用的 CuPy 模組，或 None。第一次呼叫時偵測（不在匯入時載入 CUDA，啟動才快）。"""
    global _cupy, _checked
    if not enabled:
        return None
    with _lock:
        if not _checked:
            _checked = True
            try:
                import cupy
                cupy.cuda.runtime.getDeviceCount()
                cupy.zeros(1).sum()
                _cupy = cupy
            except Exception:  # noqa: BLE001  沒有 GPU、驅動或 CUDA：用 CPU
                _cupy = None
    return _cupy


def device_name() -> str | None:
    cp = gpu()
    if cp is None:
        return None
    try:
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:  # noqa: BLE001
        return "GPU"


# 可分離卷積的 CUDA 核心：半徑與權重當參數傳進去，只要編譯一次（cupyx 的 gaussian_filter 每換一個大小就要
# 重新編譯，拖滑桿時會卡）；也用不到 cuFFT（打包時刪掉了）。邊界照 SciPy 的 reflect（d c b a | a b c d）。
_KERNEL_SRC = r"""
extern "C" __global__ void conv_h(const float* src, float* dst, const float* w, int r, int h, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= h) return;
    float acc = 0.0f;
    const float* row = src + (long long)y * width;
    for (int k = -r; k <= r; ++k) {
        int xx = x + k;
        if (xx < 0) xx = -xx - 1;
        if (xx >= width) xx = 2 * width - xx - 1;
        xx = min(max(xx, 0), width - 1);
        acc += w[k + r] * row[xx];
    }
    dst[(long long)y * width + x] = acc;
}
extern "C" __global__ void conv_v(const float* src, float* dst, const float* w, int r, int h, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= h) return;
    float acc = 0.0f;
    for (int k = -r; k <= r; ++k) {
        int yy = y + k;
        if (yy < 0) yy = -yy - 1;
        if (yy >= h) yy = 2 * h - yy - 1;
        yy = min(max(yy, 0), h - 1);
        acc += w[k + r] * src[(long long)yy * width + x];
    }
    dst[(long long)y * width + x] = acc;
}
extern "C" __global__ void conv_2d(const float* src, float* dst, const float* k, int r, int h, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= h) return;
    int n = 2 * r + 1;
    float acc = 0.0f;
    for (int j = -r; j <= r; ++j) {
        int yy = y + j;
        if (yy < 0) yy = -yy - 1;
        if (yy >= h) yy = 2 * h - yy - 1;
        yy = min(max(yy, 0), h - 1);
        const float* row = src + (long long)yy * width;
        const float* kr = k + (j + r) * n;
        for (int i = -r; i <= r; ++i) {
            int xx = x + i;
            if (xx < 0) xx = -xx - 1;
            if (xx >= width) xx = 2 * width - xx - 1;
            xx = min(max(xx, 0), width - 1);
            acc += kr[i + r] * row[xx];
        }
    }
    dst[(long long)y * width + x] = acc;
}
"""
_kernels = None


def _gpu_kernels(cp):
    global _kernels
    if _kernels is None:
        mod = cp.RawModule(code=_KERNEL_SRC)
        _kernels = (mod.get_function("conv_h"), mod.get_function("conv_v"), mod.get_function("conv_2d"))
    return _kernels


def _weights(sigma: float) -> np.ndarray:
    r = max(1, int(3.5 * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return (w / w.sum()).astype(np.float32)


def _gpu_gaussian(cp, x, sigma: float, w_dev=None):
    """顯示卡上的 2D 高斯模糊（x 是 cupy float32 的 2D 陣列）。"""
    conv_h, conv_v, _c2 = _gpu_kernels(cp)
    if w_dev is None:
        w_dev = cp.asarray(_weights(sigma))
    r = (int(w_dev.size) - 1) // 2
    h, width = x.shape
    tmp = cp.empty_like(x)
    out = cp.empty_like(x)
    block = (32, 8)
    grid = ((width + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])
    conv_h(grid, block, (x, tmp, w_dev, np.int32(r), np.int32(h), np.int32(width)))
    conv_v(grid, block, (tmp, out, w_dev, np.int32(r), np.int32(h), np.int32(width)))
    return out


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """2D 高斯模糊（單一平面）。"""
    if sigma <= 0:
        return img
    cp = gpu()
    if cp is not None and img.size > 1_000_000:
        x = cp.asarray(np.ascontiguousarray(img, np.float32))
        return cp.asnumpy(_gpu_gaussian(cp, x, sigma))
    return ndimage.gaussian_filter(img.astype(np.float32, copy=False), sigma, mode="reflect", truncate=3.5)


def richardson_lucy(L: np.ndarray, sigma: float, iterations: int, cancel=lambda: False) -> np.ndarray:
    """高斯 PSF 的 Richardson–Lucy 反摺積（輸入要是正值）。GPU 時整個迴圈留在顯示卡上。"""
    cp = gpu()
    if cp is not None:
        x = cp.asarray(np.ascontiguousarray(L, np.float32))
        w_dev = cp.asarray(_weights(sigma))
        est = x.copy()
        for _ in range(iterations):
            if cancel():
                break
            conv = _gpu_gaussian(cp, est, sigma, w_dev)
            ratio = x / cp.maximum(conv, 1e-12)
            est *= _gpu_gaussian(cp, ratio, sigma, w_dev)
        out = cp.asnumpy(est)
        del x, est
        cp.get_default_memory_pool().free_all_blocks()
        return out
    est = L.astype(np.float32, copy=True)
    for _ in range(iterations):
        if cancel():
            break
        conv = ndimage.gaussian_filter(est, sigma, mode="reflect", truncate=3.5)
        ratio = L / np.maximum(conv, 1e-12)
        est *= ndimage.gaussian_filter(ratio, sigma, mode="reflect", truncate=3.5)
    return est


def _gpu_conv2d(cp, x, k_dev, r: int):
    """顯示卡上的 2D 卷積（核是點對稱的，卷積＝相關；邊界 reflect）。"""
    _h, _v, conv_2d = _gpu_kernels(cp)
    h, width = x.shape
    out = cp.empty_like(x)
    block = (32, 8)
    grid = ((width + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])
    conv_2d(grid, block, (x, out, k_dev, np.int32(r), np.int32(h), np.int32(width)))
    return out


def _cpu_conv2d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    r = k.shape[0] // 2
    pad = np.pad(x, r, mode="symmetric")
    return fftconvolve(pad, k, mode="valid").astype(np.float32)


def richardson_lucy_kernel(L: np.ndarray, kernel: np.ndarray, iterations: int, valid: np.ndarray | None = None,
                           cancel=lambda: False) -> np.ndarray:
    """任意點對稱核的 Richardson–Lucy（輸入要是正值）。valid＝False 的像素（飽和星核）當成未知：
    不拿來比對（不然演算法會想把被截平的頂「還原」，在周圍振盪出甜甜圈），估計值由周圍決定。
    est ← est · Kᵀ(m·x／K·est) ／ Kᵀm"""
    k = np.ascontiguousarray(kernel, np.float32)
    r = k.shape[0] // 2
    m = None if valid is None else valid.astype(np.float32)
    cp = gpu()
    if cp is not None:
        x = cp.asarray(np.ascontiguousarray(L, np.float32))
        k_dev = cp.asarray(k)
        est = x.copy()
        if m is not None:
            m_dev = cp.asarray(m)
            den = cp.maximum(_gpu_conv2d(cp, m_dev, k_dev, r), 1e-6)
            x = x * m_dev
        for _ in range(iterations):
            if cancel():
                break
            conv = _gpu_conv2d(cp, est, k_dev, r)
            ratio = x / cp.maximum(conv, 1e-12)
            upd = _gpu_conv2d(cp, ratio, k_dev, r)
            est *= upd / den if m is not None else upd
        out = cp.asnumpy(est)
        del x, est
        cp.get_default_memory_pool().free_all_blocks()
        return out
    x = L.astype(np.float32, copy=True)
    est = x.copy()
    den = None
    if m is not None:
        den = np.maximum(_cpu_conv2d(m, k), 1e-6)
        x = x * m
    for _ in range(iterations):
        if cancel():
            break
        ratio = x / np.maximum(_cpu_conv2d(est, k), 1e-12)
        upd = _cpu_conv2d(ratio, k)
        est *= upd / den if den is not None else upd
    return est


# ---------------------------------------------------------------------- 補洞（去星）


def _avg4(xp, u):
    """上下左右四個鄰居的平均（邊緣複製）。"""
    p = xp.pad(u, 1, mode="edge")
    return 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])


def _jacobi(xp, u, v, known, iterations: int):
    for _ in range(iterations):
        u = xp.where(known, v, _avg4(xp, u))
    return u


def harmonic_fill(planes: np.ndarray, mask: np.ndarray, cancel=lambda: None) -> np.ndarray:
    """遮罩內用周圍補（調和內插：解拉普拉斯方程，補出來的是邊界的平滑延伸）。多重網格：一路縮小一半到
    小於 16 px，最小的一層解完，放大一層當初值、再迭代修細節。

    原本用 Processing 的正規化卷積（由小範圍到大範圍取周圍平均）：大的洞中間是用很遠的地方平均出來的，
    亮星的星暈補完後中間一塊比外圈暗（LBN576 的亮星在無星圖上一圈暗盤＋亮環）。
    cancel：每一層呼叫一次（丟出例外就中斷）。"""
    if not mask.any():
        return planes.copy()
    cp = gpu()
    xp = cp if cp is not None else np
    out = np.empty_like(planes, dtype=np.float32)
    km = xp.asarray(~mask)
    # 每一層的已知像素：2×2 裡至少兩個已知
    knowns = [km]
    while min(knowns[-1].shape) > 16 and not bool(knowns[-1].all()):
        k = knowns[-1].astype(xp.float32)
        k = xp.pad(k, ((0, k.shape[0] % 2), (0, k.shape[1] % 2)))
        s = k[0::2, 0::2] + k[1::2, 0::2] + k[0::2, 1::2] + k[1::2, 1::2]
        knowns.append(s >= 2)
    iters = [24, 32] + [48] * len(knowns)
    for c in range(planes.shape[0]):
        cancel()
        values = [xp.asarray(planes[c], dtype=xp.float32)]
        for lvl in range(1, len(knowns)):
            k = knowns[lvl - 1].astype(xp.float32)
            vk = values[-1] * k
            ph, pw = k.shape[0] % 2, k.shape[1] % 2
            k = xp.pad(k, ((0, ph), (0, pw)))
            vk = xp.pad(vk, ((0, ph), (0, pw)))
            sk = k[0::2, 0::2] + k[1::2, 0::2] + k[0::2, 1::2] + k[1::2, 1::2]
            sv = vk[0::2, 0::2] + vk[1::2, 0::2] + vk[0::2, 1::2] + vk[1::2, 1::2]
            values.append(sv / xp.maximum(sk, 1e-6))
        k = knowns[-1]
        v = values[-1]
        fill = v[k].mean() if bool(k.any()) else values[0][km].mean()
        sol = _jacobi(xp, xp.where(k, v, fill), v, k, 400)
        for lvl in range(len(knowns) - 2, -1, -1):
            k = knowns[lvl]
            h, w = k.shape
            up = xp.repeat(xp.repeat(sol, 2, axis=0), 2, axis=1)[:h, :w]
            sol = _jacobi(xp, xp.where(k, values[lvl], up), values[lvl], k, iters[lvl])
        out[c] = cp.asnumpy(sol) if cp is not None else sol
        del values, sol
    if cp is not None:
        del km, knowns
        cp.get_default_memory_pool().free_all_blocks()
    return out
