"""星點形狀地圖：把畫面分成 N×N 區，每區用亮、沒飽和、旁邊沒有其他星的星量二階矩，
印出半高寬（長軸／短軸）、橢圓率（1 − 短/長）、方向。

python tools/psf_map.py <master...> [--grid 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from apu_astro.studio import io, psf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--grid", type=int, default=3)
    a = ap.parse_args()
    for path in a.paths:
        img, _h = io.load_image(path)
        p = img if img.ndim == 3 else img[None]
        grid = psf.measure_grid(p.mean(axis=0), a.grid)
        print(f"== {Path(path).name[:70]}  ({p.shape[2]}×{p.shape[1]})")
        for row in grid.cells:
            print("   " + " | ".join(
                f"{c.fwhm_major:4.2f}/{c.fwhm_minor:4.2f} e{c.ellipticity:.2f} {c.angle:4.0f}° n{c.n}" if c else "    —" for c in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
