"""0.2.0 階段 2：互動層、各解析度沿用快速層的統計、拖曳節流、放大時自動預覽區。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from apu_processing import api as papi
from conftest import pump
from test_studio import _star_field, _write_xisf

from apu_astro.studio import chain as ch
from apu_astro.studio import ops
from apu_astro.studio import tools as tools_mod


def _small_chain(monkeypatch, steps=None) -> ch.Chain:
    # 快速層＝縮小 2 倍、互動層＝縮小 4 倍（測試影像 320×240）
    monkeypatch.setattr(ch, "QUICK_LONG_SIDE", 160)
    c = ch.Chain()
    c.set_source(ch.Source([Path("synthetic.fits")], image=_star_field(color=True)))
    c.steps = steps or [ch.new_step("gradient"), ch.new_step("color_cal"), ch.new_step("stretch"),
                        ch.new_step("color_grade", saturation=1.3)]
    return c


def test_cost_declared():
    cheap = {t.kind for t in tools_mod.all_tools() if t.cost == "cheap"}
    assert cheap == {"geometry", "color_cal", "color_grade", "curves", "stretch", "star_merge"}


def test_inter_layer_is_quick_downsampled_and_only_reruns_changed_step(monkeypatch):
    c = _small_chain(monkeypatch)
    last = len(c.steps) - 1
    q = c.evaluate(last, "quick")
    assert c.scale("inter") == 2 * c.scale("quick") == 4
    calls = []
    grad = tools_mod.get("gradient")
    real = grad.apply
    monkeypatch.setattr(grad, "apply", lambda *a, **k: calls.append(1) or real(*a, **k))
    c.steps[2].params["strength"] = 0.8           # 拖拉伸的滑桿
    r = c.evaluate(last, "inter")
    assert r.state.layers["main"].shape == (3, q.state.layers["main"].shape[1] // 2, q.state.layers["main"].shape[2] // 2)
    assert not calls                              # 前面的步驟從快速層縮小，不重算


def test_stats_measured_once_on_quick_and_shared(monkeypatch):
    """拉伸、校色的整張統計只在快速層量一次；互動層、完整解析度、預覽區都沿用（不在自己的解析度重量）。"""
    c = _small_chain(monkeypatch)
    last = len(c.steps) - 1
    shapes = []
    real = papi.measure_stretch_noise
    monkeypatch.setattr(papi, "measure_stretch_noise", lambda img, *a, **k: shapes.append(img.shape[-2:]) or
                        real(img, *a, **k))
    c.evaluate(last, "quick")
    qshape = c.evaluate(-1, "quick").state.layers["main"].shape[-2:]
    n_quick = len(shapes)
    assert n_quick > 0 and all(s == qshape for s in shapes)
    c.evaluate(last, "full")
    c.set_region((40, 60, 160, 220))
    c.evaluate(last, c.region_res())
    assert len(shapes) == n_quick                 # 完整解析度、預覽區沒有重量
    c.steps[2].params["strength"] = 0.8           # 拖拉伸強度：輸入沒變，統計不用重量
    c.evaluate(last, "inter")
    c.evaluate(last, "quick")
    assert len(shapes) == n_quick


def test_region_matches_full_when_quick_is_downsampled(monkeypatch):
    """快速層是縮圖時，預覽區跟完整解析度都用快速層的統計 → 同一塊的結果一樣（以前完整解析度自己量，對不上）。"""
    c = _small_chain(monkeypatch)
    last = len(c.steps) - 1
    full = c.evaluate(last, "full")
    c.set_region(c.map_to_source((40.0, 60.0, 160.0, 220.0), last))
    r = c.evaluate(last, c.region_res())
    disp, (x0, y0) = c.region_display(r)
    ref = c.display(full)[:, y0:y0 + disp.shape[1], x0:x0 + disp.shape[2]]
    assert disp.shape == ref.shape and float(np.abs(disp - ref).max()) < 2e-3


def test_display_of_linear_full_uses_quick_stretch(monkeypatch):
    """線性的結果：完整解析度的顯示拉伸參數從快速層量（畫面不會在快速→完整時跳一下）。"""
    c = _small_chain(monkeypatch, [ch.new_step("gradient")])
    q = c.evaluate(0, "quick")
    f = c.evaluate(0, "full")
    want = ops.apply_stf(f.state.layers["main"], ops.stf_params(q.state.layers["main"]), linked=False)
    assert np.allclose(c.display(f), want)
    assert not np.allclose(c.display(f), ops.auto_display(f.state.layers["main"]))   # 不是自己量的


def test_drag_updates_during_drag_and_layers(app, window, tmp_path, monkeypatch):
    """拖曳中互動層節流更新（不是等停手），放手後快速層，再完整解析度；畫面一直有圖、右下角顯示進度。"""
    p = tmp_path / "m.xisf"
    _write_xisf(p, _star_field())
    window.step.set("process")
    page = window.page("process")
    page.open_paths([p])
    pump(app, 30, lambda: page.chain.source is not None and not page.loading)
    last = len(page.chain.steps) - 1
    pump(app, 90, lambda: page.chain.cached(last, "full") is not None and page._job is None)
    stretch = next(i for i, s in enumerate(page.chain.steps) if s.kind == "stretch")
    page.select(stretch)
    pump(app, 30, lambda: page.chain.cached(stretch, "quick") is not None and page._job is None)
    reqs = []
    real = page._request
    monkeypatch.setattr(page, "_request", lambda res: reqs.append(res) or real(res))
    shown = []
    real_set = page.image.set_image
    monkeypatch.setattr(page.image, "set_image", lambda img, *a, **k: shown.append(img is not None) or
                        real_set(img, *a, **k))
    step = page.chain.steps[stretch]
    t0 = time.time()
    v = 0.5
    while time.time() - t0 < 0.6:                 # 0.6 秒的拖曳，每 16 ms 一個事件
        v += 0.01
        step.params["strength"] = v
        page.schedule()
        pump(app, 0.016)
    assert reqs.count("inter") >= 2 and "quick" not in reqs     # 拖曳中就有更新、還沒放手
    assert all(shown)                                          # 畫面從沒清空
    pump(app, 30, lambda: page.chain.cached(stretch, "quick") is not None and page._job is None)
    assert "quick" in reqs and page.image.has_image and page.image.busy_text is None


def test_auto_region_when_zoomed_in(app, window, tmp_path):
    p = tmp_path / "m.xisf"
    _write_xisf(p, _star_field(480, 640))
    window.step.set("process")
    page = window.page("process")
    page.open_paths([p])
    pump(app, 30, lambda: page.chain.source is not None and not page.loading)
    last = len(page.chain.steps) - 1
    pump(app, 90, lambda: page.chain.cached(last, "quick") is not None and page._job is None)
    page.image.resize(400, 300)
    page.image.set_zoom(2.0 / page.image.devicePixelRatioF())
    pump(app, 30, lambda: page.chain.region is not None)
    assert page._region_auto and not page.region_btn.isChecked()
    pump(app, 60, lambda: page.image.inset is not None and page._job is None)
    page.image.fit()
    pump(app, 5, lambda: page.chain.region is None)
    assert page.chain.region is None and not page._region_auto
    if page._job:
        page._job.cancelled.set()
