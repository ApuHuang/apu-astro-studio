"""契約測試用的固定小輸入（亂數種子固定，每台機器產生的都一樣）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def star_field(h: int = 96, w: int = 128, channels: int = 3, n: int = 40, seed: int = 7, sky: float = 0.05,
               noise: float = 0.002, gradient: float = 0.02, shift: tuple[float, float] = (0.0, 0.0),
               fwhm: float = 2.6, elong: float = 1.0) -> np.ndarray:
    """(C, H, W) float32 線性影像：天空＋線性光害梯度＋高斯星點（亮度跨三個數量級、有幾顆飽和到 1）＋噪聲。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ys, xs = rng.uniform(4, h - 4, n), rng.uniform(4, w - 4, n)
    amps = 10 ** rng.uniform(-2.0, 0.6, n)
    colors = rng.uniform(0.7, 1.3, (n, channels))
    sx = fwhm / 2.3548
    img = np.zeros((channels, h, w))
    for y, x, a, col in zip(ys, xs, amps, colors):
        g = np.exp(-(((xx - x - shift[1]) / (sx * elong)) ** 2 + ((yy - y - shift[0]) / sx) ** 2) / 2)
        img += a * col[:, None, None] * g[None]
    base = sky + gradient * (xx / w)[None] + np.linspace(0, 0.01, channels)[:, None, None]
    img = base + img + rng.normal(0, noise, (channels, h, w))
    return np.clip(img, 0, 1).astype(np.float32)


def write_frames(folder: Path, n_light: int = 6, n_dark: int = 3, size=(120, 160)) -> tuple[list[Path], list[Path]]:
    """單色天文相機的 light（星點稍微位移；最後一張星點拉長、比較糊）與 dark，ADU、16-bit 範圍的 FITS。"""
    folder.mkdir(parents=True, exist_ok=True)
    h, w = size
    lights, darks = [], []
    for i in range(n_light):
        bad = i == n_light - 1
        img = star_field(h, w, 1, n=45, seed=11, sky=0.03, noise=0.0015, gradient=0.005,
                         shift=(0.7 * i, -0.5 * i), fwhm=4.0 if bad else 2.6, elong=1.8 if bad else 1.0)[0]
        rng = np.random.default_rng(100 + i)
        data = (img * 40000 + 800 + rng.normal(0, 6, img.shape)).astype(np.float32)
        hdr = fits.Header()
        hdr.update(IMAGETYP="Light Frame", EXPTIME=120.0, FILTER="L", INSTRUME="TestCam", GAIN=100,
                   OFFSET=10, **{"CCD-TEMP": -10.0}, XBINNING=1, YBINNING=1, XPIXSZ=3.76, FOCALLEN=400.0,
                   OBJECT="Test", ROWORDER="TOP-DOWN", **{"DATE-OBS": f"2026-11-01T20:{10 + 3 * i:02d}:00"})
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
