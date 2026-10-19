"""反摺積前後的星點形狀（3×3 區）：python tools/deconv_check.py <master> [--no-round]"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from apu_astro.studio import chain as ch, ops, psf


def show(title: str, grid) -> None:
    print("  " + title)
    for row in grid.cells:
        print("     " + " | ".join(f"{c.fwhm_major:4.2f}/{c.fwhm_minor:4.2f} e{c.ellipticity:.2f}" if c else "   —" for c in row))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--no-round", action="store_true")
    a = ap.parse_args()
    src = ch.open_source([a.path])
    c = ch.Chain(16 * 2 ** 30)
    c.set_source(src)
    c.steps = [s for s in ch.default_steps(src.color, src.narrowband) if s.kind in ("geometry", "gradient", "color_cal", "deconv")]
    d = [s.kind for s in c.steps].index("deconv")
    c.steps[d].params["round"] = not a.no_round
    c.evaluate(d, "quick")
    t = time.time()
    r = c.evaluate(d, "full")
    print(f"deconv full {time.time() - t:.1f}s  view {r.views}")
    before = ops.planes(c.cached(d - 1, "full").state.layers["main"]).mean(axis=0)
    after = ops.planes(r.state.layers["main"]).mean(axis=0)
    show("before", psf.measure_grid(before, 3))
    show("after", psf.measure_grid(after, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
