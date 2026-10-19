"""讀影像：XISF 自己讀，其他（FITS、TIFF、PNG、JPEG）交給 APU Processing 的 imageio。

記憶體裡一律是 0–1 的 float32：單色 (H, W)、彩色 (3, H, W)，第 0 列是畫面最上面。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from apu_processing import imageio
from astropy.io import fits

from . import xisf

IMAGE_SUFFIXES = {".xisf"} | set(imageio.FITS_SUFFIXES) | {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def is_image(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def load_image(path: Path | str) -> tuple[np.ndarray, fits.Header]:
    path = Path(path)
    if path.suffix.lower() == ".xisf":
        img = xisf.read(path)
        data = xisf.to_unit(img)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        return np.ascontiguousarray(data, np.float32), img.header()
    data, header = imageio.load_image(path)
    return np.ascontiguousarray(data, np.float32), header
