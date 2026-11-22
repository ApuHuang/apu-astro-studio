"""0.2.0 階段 2：互動層、完整解析度的統計回灌預覽、去星沿用、拖曳節流、放大時自動預覽區。"""

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


def _saturated_field(h: int, w: int) -> np.ndarray:
    """星場加幾顆飽和（截成平頂）的亮星：校色、反摺積的飽和處理才會走到。"""
    img = _star_field(h, w, color=True)
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in ((h * 0.3, w * 0.25), (h * 0.7, w * 0.6), (h * 0.5, w * 0.85)):
        img = img + 4.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))[None]
    return np.minimum(img, 0.95).astype(np.float32)


def _small_chain(monkeypatch, steps=None, size=(240, 320), image=None) -> ch.Chain:
    # 快速層＝縮小 2 倍、互動層＝縮小 4 倍（測試影像預設 320×240）
    monkeypatch.setattr(ch, "QUICK_LONG_SIDE", size[1] // 2)
    monkeypatch.setattr(ch, "INTER_MIN_QUICK", 0)
    c = ch.Chain()
    c.set_source(ch.Source([Path("synthetic.fits")], image=_star_field(*size, color=True) if image is None else image))
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


def _measure_log(monkeypatch) -> list:
    shapes = []
    real = papi.measure_stretch_noise
    monkeypatch.setattr(papi, "measure_stretch_noise", lambda img, *a, **k: shapes.append(img.shape[-2:]) or
                        real(img, *a, **k))
    return shapes


def test_full_resolution_measures_its_own_stats(monkeypatch):
    """完整解析度一律自己量（出圖跟 0.1.x 一樣）：先算過快速層、互動層、預覽區，結果也跟只算完整解析度的逐位元相同。"""
    c = _small_chain(monkeypatch)
    last = len(c.steps) - 1
    c.evaluate(last, "quick")
    c.steps[2].params["strength"] = 0.8
    c.evaluate(last, "inter")
    c.set_region((40, 60, 160, 220))
    c.evaluate(last, c.region_res())
    shapes = _measure_log(monkeypatch)
    full = c.evaluate(last, "full")
    assert shapes and all(s == (240, 320) for s in shapes)        # 在完整解析度上量
    fresh = _small_chain(monkeypatch)
    fresh.steps = c.steps
    alone = fresh.evaluate(last, "full")
    assert np.array_equal(full.state.layers["main"], alone.state.layers["main"])


def test_full_stats_fed_back_to_previews(monkeypatch):
    """完整解析度算過後，快速層、互動層、預覽區用它量的統計，自己不再量；預覽區跟完整解析度同一塊一樣。"""
    c = _small_chain(monkeypatch)
    last = len(c.steps) - 1
    c.evaluate(last, "quick")
    full = c.evaluate(last, "full")
    assert c.full_ready(last)
    shapes = _measure_log(monkeypatch)
    c._cache = type(c._cache)((k, v) for k, v in c._cache.items() if k[0] != "quick")   # 快速層重算
    c.evaluate(last, "quick")
    c.set_region(c.map_to_source((40.0, 60.0, 160.0, 220.0), last))
    r = c.evaluate(last, c.region_res())
    c.steps[3].params["saturation"] = 1.4         # 拖調色：上游統計都有完整解析度的
    c.evaluate(last, "inter")
    assert shapes == []
    disp, (x0, y0) = c.region_display(r)
    ref = c.display(full)[:, y0:y0 + disp.shape[1], x0:x0 + disp.shape[2]]
    assert disp.shape == ref.shape and float(np.abs(disp - ref).max()) < 2e-3
    assert not c.full_ready(last)                 # 參數改了：這一步還沒有完整解析度


def test_drag_every_step_after_full(monkeypatch):
    """完整解析度算過後，拖每一步（統計有的從完整解析度讀、新的參數在快速層量）都不會出錯；
    回灌的統計不能有跟解析度有關的陣列（2026-11-21：校色的飽和區遮罩放進統計，拖強度時形狀對不上）。"""
    c = _small_chain(monkeypatch, [ch.new_step("gradient"), ch.new_step("color_cal"), ch.new_step("deconv"),
                                   ch.new_step("star_split"), ch.new_step("denoise", "starless"),
                                   ch.new_step("stretch", "starless", method="auto"),
                                   ch.new_step("stretch", "stars", method="arcsinh"),
                                   ch.new_step("color_grade", "starless"), ch.new_step("star_merge")],
                     size=(640, 960), image=_saturated_field(640, 960))
    last = len(c.steps) - 1
    c.evaluate(last, "quick")
    c.evaluate(last, "full")
    c.evaluate(last, "quick")
    changes = {"gradient": ("smooth", 0.8), "color_cal": ("strength", 0.6), "deconv": ("stars", 0.4),
               "star_split": ("grow", 3.0), "denoise": ("amount", 0.7), "stretch": ("strength", 0.7),
               "color_grade": ("vibrance", 0.4), "star_merge": ("reduce", 0.3)}
    for i, s in enumerate(c.steps):
        key, v = changes[s.kind]
        old = s.params.get(key)
        s.params[key] = v
        for res in ("inter", "quick"):
            r = c.evaluate(i, res)
            assert all(np.isfinite(a).all() for a in r.state.layers.values())
        s.params[key] = old


def test_quick_stats_until_full_then_preview_matches_full(monkeypatch):
    """完整解析度算完之前預覽用快速層量的；之後快速層的結果縮小比，跟完整解析度更接近。"""
    c = _small_chain(monkeypatch)
    last = len(c.steps) - 1
    before = c.display(c.evaluate(last, "quick"))
    full = c.display(c.evaluate(last, "full"))
    c._cache = type(c._cache)((k, v) for k, v in c._cache.items() if k[0] != "quick")
    after = c.display(c.evaluate(last, "quick"))
    fd = ch._downsample(full, 2)
    bias = lambda a: abs(float((a - fd).mean()))   # noqa: E731
    assert bias(after) <= bias(before) + 1e-6


def test_display_full_own_and_previews_use_full_params(monkeypatch):
    """線性的結果：完整解析度的顯示用自己量的參數（跟 0.1.x 一樣）；之後快速層的顯示也用那組參數。"""
    c = _small_chain(monkeypatch, [ch.new_step("gradient")])
    q = c.evaluate(0, "quick")
    before = c.display(q)
    assert np.allclose(before, ops.auto_display(q.state.layers["main"]))     # 還沒有完整解析度：快速層自己的
    f = c.evaluate(0, "full")
    assert np.array_equal(c.display(f), ch.display_of(f.state))
    want = ops.apply_stf(q.state.layers["main"], ops.stf_params(f.state.layers["main"]), linked=False)
    assert np.allclose(c.display(q), want)


def test_star_split_reuses_full_result_until_params_change(monkeypatch):
    c = _small_chain(monkeypatch, [ch.new_step("gradient"), ch.new_step("star_split"),
                                   ch.new_step("stretch", "starless")])
    c.evaluate(2, "quick")
    own = c.evaluate(1, "quick").state
    full = c.evaluate(1, "full").state
    down = ch._downsample_state(full, 2)
    reused = c.evaluate(1, "quick").state
    for k in ("starless", "stars"):
        assert np.array_equal(reused.layers[k], down.layers[k])
        assert not np.array_equal(reused.layers[k], own.layers[k])
    inter = c.evaluate(2, "inter").state
    assert inter.layers["starless"].shape[-2:] == (60, 80)
    c.steps[2].params["strength"] = 0.7           # 下游改了：去星還是沿用
    assert np.array_equal(c.evaluate(1, "quick").state.layers["stars"], down.layers["stars"])
    smooth = c.steps[0].params.get("smooth")
    c.steps[0].params["smooth"] = 0.9             # 上游改了：快速層自己算
    assert c._split_reuse(c.steps[1], c._base_keys(c.steps)[1]) is None
    assert not np.array_equal(c.evaluate(1, "quick").state.layers["stars"], down.layers["stars"])
    c.steps[0].params["smooth"] = smooth
    assert c._split_reuse(c.steps[1], c._base_keys(c.steps)[1]) is not None
    c.steps[1].params["grow"] = float(c.steps[1].params.get("grow", 1.0)) + 1   # 去星自己改了：一樣失效
    assert c._split_reuse(c.steps[1], c._base_keys(c.steps)[1]) is None


def test_inter_needs_big_enough_quick(monkeypatch):
    c = _small_chain(monkeypatch)
    assert c.inter_ok()
    monkeypatch.setattr(ch, "INTER_MIN_QUICK", 800)
    assert not c.inter_ok()          # 快速層 160 px：拖曳中直接用快速層


def _drag(app, page, monkeypatch, stretch):
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
    return reqs, shown


def test_small_image_drag_uses_quick(app, window, tmp_path, monkeypatch):
    """小圖（快速層長邊不到 INTER_MIN_QUICK）：拖曳中直接節流算快速層。"""
    p = tmp_path / "m.xisf"
    _write_xisf(p, _star_field())
    window.step.set("process")
    page = window.page("process")
    page.open_paths([p])
    pump(app, 30, lambda: page.chain.source is not None and not page.loading)
    stretch = next(i for i, s in enumerate(page.chain.steps) if s.kind == "stretch")
    page.select(stretch)
    pump(app, 60, lambda: page.chain.cached(stretch, "quick") is not None and page._job is None)
    reqs, shown = _drag(app, page, monkeypatch, stretch)
    assert "inter" not in reqs and reqs.count("quick") >= 2 and all(shown)
    pump(app, 30, lambda: page._job is None)


def test_drag_updates_during_drag_and_layers(app, window, tmp_path, monkeypatch):
    """拖曳中互動層節流更新（不是等停手），放手後快速層，再完整解析度；畫面一直有圖、右下角顯示進度。"""
    monkeypatch.setattr(ch, "INTER_MIN_QUICK", 0)
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
    reqs, shown = _drag(app, page, monkeypatch, stretch)
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
