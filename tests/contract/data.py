"""契約測試用的固定小輸入（亂數種子固定，每台機器產生的都一樣）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def star_field(h: int = 96, w: int = 128, channels: int = 3, n: int = 40, seed: int = 7, sky: float = 0.05,
               noise: float = 0.002, gradient: float = 0.02, shift: tuple[float, float] = (0.0, 0.0),
               fwhm: float = 2.6, elong: float = 1.0, angle: float = 30.0) -> np.ndarray:
    """(C, H, W) float32 線性影像：天空＋線性光害梯度＋高斯星點（亮度跨三個數量級、有幾顆飽和到 1）＋噪聲。
    elong：長軸／短軸，長軸方向 angle 度（不放 0 度：角度接近 0 時相對誤差沒有意義）。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ys, xs = rng.uniform(4, h - 4, n), rng.uniform(4, w - 4, n)
    amps = 10 ** rng.uniform(-2.0, 0.6, n)
    colors = rng.uniform(0.7, 1.3, (n, channels))
    sx = fwhm / 2.3548
    c, s_ = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    img = np.zeros((channels, h, w))
    for y, x, a, col in zip(ys, xs, amps, colors):
        dx, dy = xx - x - shift[1], yy - y - shift[0]
        u, v = dx * c + dy * s_, -dx * s_ + dy * c
        g = np.exp(-((u / (sx * elong)) ** 2 + (v / sx) ** 2) / 2)
        img += a * col[:, None, None] * g[None]
    base = sky + gradient * (xx / w)[None] + np.linspace(0, 0.01, channels)[:, None, None]
    img = base + img + rng.normal(0, noise, (channels, h, w))
    return np.clip(img, 0, 1).astype(np.float32)


# light 的星點：橢圓率約 0.3（長短軸比 1/√(1−0.3²)）。正圓的星方向角不穩定：Pick 量的方向角在不同 CPU 上差相對 2e-4
ELONG = 1 / np.sqrt(1 - 0.3 ** 2)


def mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """(3, H, W) → 單一平面的 Bayer 馬賽克（pattern 例如 "RGGB"：左上 2×2 的顏色）。"""
    out = np.zeros(rgb.shape[1:], np.float32)
    for i, c in enumerate(pattern):
        dy, dx = i // 2, i % 2
        out[dy::2, dx::2] = rgb["RGB".index(c), dy::2, dx::2]
    return out


def write_frames(folder: Path, n_light: int = 6, n_dark: int = 3, size=(120, 160),
                 bayer: str | None = None) -> tuple[list[Path], list[Path]]:
    """天文相機的 light（星點稍微位移；最後一張星點拉長、比較糊）與 dark，ADU、16-bit 範圍的 FITS。
    bayer：彩色相機（OSC）的排列（例如 "RGGB"）；None＝單色。彩色時星點有各自的顏色，header 寫 BAYERPAT。"""
    folder.mkdir(parents=True, exist_ok=True)
    h, w = size
    lights, darks = [], []
    for i in range(n_light):
        bad = i == n_light - 1
        img = star_field(h, w, 3 if bayer else 1, n=45, seed=11, sky=0.03, noise=0.0015, gradient=0.005,
                         shift=(0.7 * i, -0.5 * i), fwhm=4.0 if bad else 2.6, elong=1.8 if bad else ELONG)
        img = mosaic(img, bayer) if bayer else img[0]
        rng = np.random.default_rng(100 + i)
        data = (img * 40000 + 800 + rng.normal(0, 6, img.shape)).astype(np.float32)
        hdr = fits.Header()
        hdr.update(IMAGETYP="Light Frame", EXPTIME=120.0, FILTER="L", INSTRUME="TestCam", GAIN=100,
                   OFFSET=10, **{"CCD-TEMP": -10.0}, XBINNING=1, YBINNING=1, XPIXSZ=3.76, FOCALLEN=400.0,
                   OBJECT="Test", ROWORDER="TOP-DOWN", **{"DATE-OBS": f"2026-11-01T20:{10 + 3 * i:02d}:00"})
        if bayer:
            hdr.update(BAYERPAT=bayer, XBAYROFF=0, YBAYROFF=0)
            del hdr["FILTER"]
        p = folder / f"light_{i:03d}.fits"
        fits.PrimaryHDU(data, hdr).writeto(p, overwrite=True)
        lights.append(p)
    for i in range(n_dark):
        rng = np.random.default_rng(500 + i)
        data = (800 + rng.normal(0, 6, (h, w))).astype(np.float32)
        hdr = fits.Header()
        hdr.update(IMAGETYP="Dark Frame", EXPTIME=120.0, INSTRUME="TestCam", GAIN=100, OFFSET=10,
                   **{"CCD-TEMP": -10.0}, XBINNING=1, YBINNING=1, ROWORDER="TOP-DOWN",
                   **{"DATE-OBS": f"2026-11-01T23:{10 + i:02d}:00"})
        p = folder / f"dark_{i:03d}.fits"
        fits.PrimaryHDU(data, hdr).writeto(p, overwrite=True)
        darks.append(p)
    return lights, darks
