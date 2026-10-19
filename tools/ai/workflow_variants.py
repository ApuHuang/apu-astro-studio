"""AI 降噪放進流程的幾種做法並排比較（預覽區只算一塊，很快）。

python tools/ai/workflow_variants.py <master> <y0,x0,y1,x1> <輸出.png> [做法...]
做法：base（建議流程、不用 AI）、ai（反摺積後加 AI，有模型時的建議流程）、ai_first（反摺積前加 AI）、
ai_replace（AI 取代線性階段的降噪）。預設關著的步驟都打開。
每一格下方印時間；另外印每種做法最後成品的「暗點數」（比 5×5 中位數暗 0.12 以上的像素）。
"""

from __future__ import annotations

import sys
import time

import numpy as np
from apu_processing import imageio
from scipy import ndimage

from apu_astro.studio import chain as ch


def variant(src: ch.Source, name: str) -> list[ch.Step]:
    steps = [s for s in ch.default_steps(src.color, src.narrowband) if s.kind != "ai_denoise"]
    for s in steps:
        s.enabled = True
    kinds = [s.kind for s in steps]
    after = kinds.index("deconv") + 1
    if name == "ai":
        steps.insert(after, ch.new_step("ai_denoise"))
    elif name == "ai_first":
        steps.insert(after - 1, ch.new_step("ai_denoise"))
    elif name == "ai_replace":
        steps[kinds.index("denoise")].enabled = False
        steps.insert(after, ch.new_step("ai_denoise"))
    return steps


def main() -> int:
    path, box, out = sys.argv[1], tuple(int(v) for v in sys.argv[2].split(",")), sys.argv[3]
    names = sys.argv[4:] or ["base", "ai", "ai_first", "ai_replace"]
    src = ch.open_source([path])
    tiles = []
    for name in names:
        c = ch.Chain()
        c.set_source(src)
        c.steps = variant(src, name)
        c.set_region(box)
        t0 = time.time()
        r = c.evaluate(len(c.steps) - 1, c.region_res())
        d = c.region_display(r)[0]
        d = d if d.shape[0] == 3 else np.repeat(d, 3, 0)
        L = d.mean(axis=0)
        dots = int(((L - ndimage.median_filter(L, 5)) < -0.12).sum())
        print(f"{name:11s} {time.time() - t0:5.1f}s  median {float(np.median(L)):.3f}  dots {dots}", flush=True)
        tiles.append(d)
    imageio.save_display(out, np.concatenate(tiles, axis=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
