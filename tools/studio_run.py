"""開發用：不開視窗跑後製工作室的預設流程，印每一步的時間，存最後結果與每一步的縮圖。

python tools/studio_run.py <影像或疊圖紀錄...> --out <資料夾> [--full]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from apu_processing import imageio

from apu_astro.studio import chain as ch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = ch.open_source([Path(p) for p in a.paths])
    print(f"open {time.time() - t0:.1f}s shape={src.image.shape} narrowband={src.narrowband} "
          f"coverage={'yes' if src.coverage_mask is not None else 'no'}")
    c = ch.Chain()
    c.set_source(src)
    c.steps = ch.default_steps(src.color, src.narrowband)
    res = "full" if a.full else "quick"
    print("quick factor", c.quick_factor())
    for i, s in enumerate(c.steps):
        t = time.time()
        r = c.evaluate(i, res)
        disp = ch.display_of(r.state)
        imageio.save_display(out / f"{i:02d}_{s.kind}_{s.layer}.jpg", disp if disp.shape[0] == 3 else disp[0])
        warn = "; ".join(str(w) for w in r.state.warnings)
        extra = {k: v for k, v in r.views.items() if isinstance(v, (int, float, tuple))}
        print(f"{i:2d} {s.kind:12s} {s.layer:9s} {time.time() - t:6.1f}s layers={list(r.state.layers)} "
              f"phase={r.state.phase} {extra} {warn}")
    final = ch.display_of(c.evaluate(len(c.steps) - 1, res).state)
    imageio.save_display(out / "final.png", final if final.shape[0] == 3 else final[0])
    print(f"total {time.time() - t0:.1f}s", float(np.median(final)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
