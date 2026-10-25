"""後製工作室：XISF 讀檔、步驟鏈的快取（只重算改到的之後）、去星分支、各工具、出圖、介面的基本操作。"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from conftest import pump

from apu_astro.studio import chain as ch
from apu_astro.studio import export, masks, ops, xisf
from apu_astro.studio.tools import Ctx, get


def _star_field(h=240, w=320, color=True, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    sky = 0.01 + 0.004 * (xx / w) + 0.002 * (yy / h)            # 光害梯度
    neb = 0.006 * np.exp(-(((xx - w * 0.5) / (w * 0.2)) ** 2 + ((yy - h * 0.5) / (h * 0.25)) ** 2))
    img = sky + neb
    for _ in range(60):
        cy, cx, f = rng.uniform(5, h - 5), rng.uniform(5, w - 5), rng.uniform(0.02, 0.4)
        img = img + f * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3 ** 2))
    img = img + rng.normal(0, 0.0008, img.shape)
    if color:
        return np.stack([img * 1.05, img * 1.25, img * 0.9]).astype(np.float32)
    return img[None].astype(np.float32)


def _write_xisf(path: Path, data: np.ndarray, compress: bool = False) -> None:
    """測試用的最小 XISF（Planar、Float32、可選 zlib＋shuffle）。"""
    c, h, w = data.shape
    raw = np.ascontiguousarray(data, "<f4").tobytes()
    comp_attr = ""
    if compress:
        n = len(raw) // 4
        shuffled = np.frombuffer(raw, np.uint8).reshape(n, 4).T.tobytes()
        payload = zlib.compress(shuffled)
        comp_attr = f' compression="zlib+sh:{len(raw)}:4"'
    else:
        payload = raw
    def header(pos: int) -> bytes:
        xml = (f'<?xml version="1.0" encoding="UTF-8"?><xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">'
               f'<Image geometry="{w}:{h}:{c}" sampleFormat="Float32" bounds="0:1" '
               f'colorSpace="{"RGB" if c == 3 else "Gray"}" location="attachment:{pos}:{len(payload)}"{comp_attr}>'
               f'<FITSKeyword name="FILTER" value="\'Ha\'" comment="filter"/>'
               f'<FITSKeyword name="FOCALLEN" value="430.5" comment=""/></Image></xisf>').encode()
        return xml
    pos = 4096
    xml = header(pos)
    blob = b"XISF0100" + struct.pack("<II", len(xml), 0) + xml
    blob += b"\0" * (pos - len(blob))
    path.write_bytes(blob + payload)


@pytest.mark.parametrize("compress", [False, True])
def test_xisf_roundtrip(tmp_path, compress):
    data = _star_field(40, 60)
    p = tmp_path / "x.xisf"
    _write_xisf(p, data, compress)
    img = xisf.read(p)
    assert img.data.shape == (3, 40, 60)
    np.testing.assert_allclose(img.data, data, atol=1e-7)
    h = img.header()
    assert h["FILTER"] == "Ha" and abs(h["FOCALLEN"] - 430.5) < 1e-6


def _chain(color=True) -> ch.Chain:
    src = ch.Source([Path("synthetic.fits")], image=_star_field(color=color))
    c = ch.Chain()
    c.set_source(src)
    c.steps = ch.default_steps(color)
    return c


def test_default_workflow_runs():
    c = _chain()
    kinds = [s.kind for s in c.steps]
    # 10/18 的流程：去星 → 分開拉伸 → 合成星點，全部打開
    assert kinds.index("star_split") < kinds.index("stretch") < kinds.index("star_merge") == len(kinds) - 1
    assert all(s.enabled for s in c.steps)
    r = c.evaluate(len(c.steps) - 1, "full")
    assert list(r.state.layers) == ["main"] and r.state.phase["main"] == "nonlinear"
    out = r.state.layers["main"]
    assert out.shape == (3, 240, 320) and np.isfinite(out).all() and 0 <= out.min() and out.max() <= 1


def test_star_split_recombines():
    # 去星不在預設流程裡，但舊專案檔可能有：無星圖＋星點層 ≈ 原圖（星點範圍外完全一樣）
    c = _chain()
    c.steps = [ch.new_step("geometry", auto_edges=True), ch.new_step("star_split")]
    mid = c.evaluate(1, "full").state
    assert set(mid.layers) == {"starless", "stars"}
    before = c.evaluate(0, "full").state.layers["main"]
    recon = mid.layers["starless"] + mid.layers["stars"]
    assert float(np.median(np.abs(recon - before))) < 2e-3


def test_cache_recomputes_only_after_changed_step(monkeypatch):
    c = _chain(color=False)
    for s in c.steps:                    # 拉伸以後的步驟預設關著
        s.enabled = True
    c.evaluate(len(c.steps) - 1, "quick")
    calls = []
    for kind in ("gradient", "deconv", "sharpen"):
        tool = get(kind)
        orig = tool.apply

        def spy(img, params, ctx, linear, orig=orig, kind=kind):
            calls.append(kind)
            return orig(img, params, ctx, linear)
        monkeypatch.setattr(tool, "apply", spy)
    sharpen = next(s for s in c.steps if s.kind == "sharpen")
    sharpen.params["detail"] = 1.2
    c.evaluate(len(c.steps) - 1, "quick")
    assert calls == ["sharpen"]          # 前面的去光梯度、反摺積都沿用快取
    calls.clear()
    c.steps[0].params["view"] = "model"  # 只影響畫面的參數：不用重算
    c.evaluate(len(c.steps) - 1, "quick")
    assert calls == []


def test_ghs_is_monotonic_and_normalized():
    x = np.linspace(0, 1, 2001, dtype=np.float32)
    for b in (-1.0, -0.5, 0.0, 1.0, 4.0):
        y = ops.ghs(x, 6.0, b, 0.05, 0.01, 0.9)
        assert abs(float(y[0])) < 1e-6 and abs(float(y[-1]) - 1) < 1e-6
        assert np.all(np.diff(y) >= -1e-6)


def test_gradient_removes_slope_and_keeps_level():
    img = _star_field(color=False)
    tool = get("gradient")
    out = tool.apply(img, tool.defaults(), Ctx(1.0), True)
    cols = np.arange(320)

    def slope(a):  # 上方天空每一欄的中位數，對欄位做直線擬合；星雲左右對稱，不影響斜率
        return np.polyfit(cols, np.median(a[0, 5:50], axis=0), 1)[0]
    assert abs(slope(out)) < 0.2 * abs(slope(img))
    assert abs(float(np.median(out)) - float(np.median(img))) < 2e-3


def test_masks_and_color_tools():
    img = _star_field()
    disp = ops.auto_display(img)
    m = masks.build({**masks.default_mask(), "mode": "bright", "lo": 0.5, "feather": 0}, disp, False, 1.0)
    assert m.shape == (240, 320) and m.max() > 0.9 and m.min() < 0.1
    strokes = [[50, 50, 10, True], [52, 52, 4, False]]
    r = masks.rasterize(strokes, (100, 100), 1.0)
    assert r[50, 45] == 1 and r[52, 52] == 0
    grade = get("color_grade")
    p = {**grade.defaults(), "sel_on": True, "sel_hue": 30, "sel_shift": 40, "mid_a": 0.5, "mid_h": 250}
    out = grade.apply(disp, p, Ctx(1.0), False)
    assert out.shape == disp.shape and np.isfinite(out).all()
    assert not np.allclose(out, disp)


def test_export_png_tiff_resize_watermark(tmp_path):
    disp = ops.auto_display(_star_field())
    o = export.ExportOptions(long_side=160, watermark="APU", sharpen=0.3)
    export.save(disp, tmp_path / "a.png", o)
    export.save(disp, tmp_path / "a.tif", export.ExportOptions())
    from PIL import Image
    with Image.open(tmp_path / "a.png") as im:
        assert max(im.size) == 160
    import tifffile
    t = tifffile.imread(tmp_path / "a.tif")
    assert t.dtype == np.uint16 and t.shape == (240, 320, 3)


def test_project_roundtrip(tmp_path):
    c = _chain()
    c.steps[0].params["exclude"] = [[10, 10, 5, True]]
    c.steps[2].mask = {**masks.default_mask(), "mode": "dark"}
    path = tmp_path / "p.apuastro"
    ch.save_project(c, path)
    src, steps = ch.load_project(path)
    assert [s.kind for s in steps] == [s.kind for s in c.steps]
    assert steps[0].params["exclude"] == [[10, 10, 5, True]] and steps[2].mask["mode"] == "dark"
    assert src["paths"] == ["synthetic.fits"]


def test_studio_page_opens_and_edits(app, window, tmp_path):
    p = tmp_path / "m.xisf"
    _write_xisf(p, _star_field())
    window.step.set("process")
    page = window.page("process")
    page.open_paths([p])
    pump(app, 30, lambda: page.chain.source is not None and not page.loading)
    last = len(page.chain.steps) - 1
    pump(app, 60, lambda: page.chain.cached(last, "quick") is not None and page._job is None)
    assert page.image.has_image
    page.steps_panel.select(0)                   # 去光梯度
    pump(app, 0.3)
    assert page.selected == 0
    page._set_tool("brush_exclude")
    page._stroke(100, 100, 30, True)
    page._stroke_done()
    assert page.chain.steps[0].params["exclude"] == [[100.0, 100.0, 30.0, True]]
    pump(app, 30, lambda: page.chain.cached(0, "quick") is not None and page._job is None)
    page.add_step("curves")
    assert page.chain.steps[1].kind == "curves"
    page.delete_step()
    assert all(s.kind != "curves" for s in page.chain.steps)
    page._set_tool(None)
    page.view_mode.set("compare")
    pump(app, 0.3)
    page.view_mode.set("result")
    # 預覽區：框一塊 → 只算那一塊，疊在畫面上；再按一次回到整張
    page.select(len(page.chain.steps) - 1)
    pump(app, 30, lambda: page.chain.cached(page.selected, "quick") is not None and page._job is None)
    page._toggle_region()
    assert page._tool_mode == "region" and page.region_btn.isChecked()
    page.image.set_crop_box((60.0, 50.0, 200.0, 150.0))
    assert page.chain.region is not None and page._tool_mode is None
    pump(app, 60, lambda: page.image.inset is not None and page._job is None)
    assert page.image.inset.shape[1:] == (100, 140) and page.image.inset_pos == (60.0, 50.0)
    page._toggle_region()
    assert page.chain.region is None and page.image.inset is None and not page.region_btn.isChecked()
    for job in [page._job] if page._job else []:
        job.cancelled.set()


def test_region_matches_full_resolution_crop():
    """預覽區（只算一塊）跟整張算的同一塊一樣；裁切、旋轉、翻轉之後位置也對得上。"""
    src = ch.Source([Path("synthetic.fits")], image=_star_field(color=True))
    c = ch.Chain()
    c.set_source(src)
    c.steps = [ch.new_step("geometry", box=[20, 10, 300, 230], turns=1, flip_h=True), ch.new_step("gradient"),
               ch.new_step("color_cal"), ch.new_step("stretch"), ch.new_step("color_grade", saturation=1.3)]
    last = len(c.steps) - 1
    full = c.evaluate(last, "full")
    c.evaluate(last, "quick")
    # 在最後一步的畫面上框一塊 → 換回來源座標 → 再換回來要一樣
    box = (40.0, 30.0, 160.0, 150.0)
    src_box = c.map_to_source(box, last)
    back = c.map_from_source(src_box, last)
    assert np.allclose(back, box)
    c.set_region(src_box)
    r = c.evaluate(last, c.region_res())
    disp, (x0, y0) = c.region_display(r)
    ref = ch.display_of(full.state)[:, y0:y0 + disp.shape[1], x0:x0 + disp.shape[2]]
    assert (x0, y0) == (30, 40) and disp.shape == ref.shape == (3, 120, 120)
    assert float(np.abs(disp - ref).max()) < 2e-3


def test_region_cropped_away_raises():
    from apu_astro.studio.tools import RegionOutside
    src = ch.Source([Path("synthetic.fits")], image=_star_field(color=False))
    c = ch.Chain()
    c.set_source(src)
    c.steps = [ch.new_step("geometry", box=[0, 0, 100, 100])]
    c.evaluate(0, "quick")
    c.set_region((150, 200, 220, 300))
    with pytest.raises(RegionOutside):
        c.evaluate(0, c.region_res())


@pytest.mark.parametrize("scale", [1.0, 3.0, 2.6])
def test_gradient_any_cell_size(scale):
    """取樣格不是內部小格整數倍時不能出錯（M31 4920×3276 縮小 3 倍 → 取樣格 21 px 曾經當掉）。"""
    from apu_astro.studio.tools import Ctx, get
    img = _star_field(1092, 1640)
    out = get("gradient").apply(img, get("gradient").defaults(), Ctx(scale), True)
    assert out.shape == img.shape and np.isfinite(out).all()


@pytest.mark.skipif(not __import__("apu_astro.studio.tools.ai", fromlist=["ai"]).available(), reason="沒有 AI 模型")
def test_ai_denoise_no_seams_no_lowfreq_and_keeps_bright():
    """AI 降噪的人造瑕疵：錯開分塊位置結果一樣（沒有接縫）、修正量沒有大範圍的成分、亮的地方幾乎不動。"""
    from scipy import ndimage

    from apu_astro.studio.tools import ai
    img = _star_field(900, 1400, color=False)[0]
    sky, sig = ai.sky_sigma(img)
    x = (img - sky) / sig
    a = ai.denoise_plane(x)
    b = ai.denoise_plane(x[200:, 200:])
    assert float(np.abs(b - a[200:, 200:])[160:-160, 160:-160].max()) < 1e-3
    _fixed, (out,), _n = ai.denoise_planes([x])
    fix = out - _fixed[0]
    assert float(np.abs(ndimage.gaussian_filter(fix, 16)).max()) < 0.15
    assert float(np.std(fix[x < 2])) > 0.3                        # 天空真的有降噪
    bright = x > 200
    assert float(np.abs(fix[bright]).max()) < 0.05 * float(x[bright].min())


def test_ai_coarse_pass_removes_mottle_keeps_dark_lane():
    """縮小一半那層：去掉斑駁（4～8 px 的低頻噪聲），但暗帶（中尺度的結構）不能被填亮（NGC2244 曾經從 −0.8σ 變 −0.5σ）。"""
    from scipy import ndimage

    from apu_astro.studio.tools import ai
    rng = np.random.default_rng(5)
    h, w = 768, 768
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    truth = 2.0 + 1.5 * np.exp(-((yy - 300) ** 2 + (xx - 400) ** 2) / (2 * 150.0 ** 2))
    lane = np.abs(xx - 0.4 * yy - 300) < 6                       # 斜的暗帶，寬 12 px
    truth = truth - 1.2 * ndimage.gaussian_filter(lane.astype(np.float32), 2.0)
    white = rng.standard_normal((h, w)).astype(np.float32)
    mottle = ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 3.0)
    mottle *= 0.35 / mottle.std()
    x = (truth + white + mottle).astype(np.float32)
    sky, sig = ai.sky_sigma(x)
    xs = [(x - sky) / sig]
    base = ai.denoise_planes(xs)[1][0] * sig + sky
    coarse = ai.denoise_planes(xs, coarse=ai.COARSE)[1][0] * sig + sky
    flat = ~ndimage.binary_dilation(lane, iterations=30)
    m_base = float(np.std(ndimage.gaussian_filter(base - truth, 4)[flat]))
    m_coarse = float(np.std(ndimage.gaussian_filter(coarse - truth, 4)[flat]))
    assert m_coarse < 0.9 * m_base                                 # 斑駁有減少
    core = ndimage.binary_erosion(lane, iterations=3)
    lane_base = float(ndimage.gaussian_filter(base, 3)[core].mean())
    lane_coarse = float(ndimage.gaussian_filter(coarse, 3)[core].mean())
    assert abs(lane_coarse - lane_base) < 0.1                      # 暗帶的深淺不變


def test_denoise_does_not_change_stretch_strength():
    """降噪（線性階段）不改變自動拉伸的深淺：有沒有降噪，拉伸後的天空與亮處差不多。"""
    src = ch.Source([Path("synthetic.fits")], image=_star_field(color=True))
    medians = []
    for steps in ([ch.new_step("stretch")], [ch.new_step("denoise", amount=0.8), ch.new_step("stretch")]):
        c = ch.Chain()
        c.set_source(src)
        c.steps = steps
        out = c.evaluate(len(steps) - 1, "full").state.layers["main"]
        medians.append((float(np.median(out)), float(np.quantile(out, 0.99))))
    assert abs(medians[0][0] - medians[1][0]) < 0.03 and abs(medians[0][1] - medians[1][1]) < 0.05


def _elliptical_field(h=600, w=800, cov=((4.0, 1.2), (1.2, 2.6)), shift_r=(0.0, 0.0), sat_level=None, seed=1,
                      noise=0.0015):
    """合成星場：橢圓高斯星點（共變異數 cov，x/y 順序）；紅色版可以整體位移；可以把亮星截平成飽和。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    inv = np.linalg.inv(np.array(cov))
    planes = np.full((3, h, w), 0.02)
    for _ in range(120):
        cy, cx, f = rng.uniform(20, h - 20), rng.uniform(20, w - 20), rng.uniform(0.05, 0.6)
        for c in range(3):
            dy, dx = yy - cy - (shift_r[0] if c == 0 else 0), xx - cx - (shift_r[1] if c == 0 else 0)
            q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
            planes[c] += f * np.exp(-0.5 * q)
    planes += rng.normal(0, noise, planes.shape)
    if sat_level is not None:
        planes = np.minimum(planes, sat_level)
    return planes.astype(np.float32)


def test_psf_measure_and_saturation():
    from apu_astro.studio import psf
    img = _elliptical_field()
    grid = psf.measure_grid(img.mean(axis=0), 2)
    c = grid.overall
    ev, vec = np.linalg.eigh(np.array([[4.0, 1.2], [1.2, 2.6]]))
    true_e = 1 - np.sqrt(ev[0] / ev[1])
    true_angle = np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) % 180
    assert abs(c.ellipticity - true_e) < 0.05 and abs(c.angle - true_angle) < 8
    sat = _elliptical_field(sat_level=0.3)              # 截平在 0.3（不是整張最大值以外的任何值）
    m = psf.saturation_mask(sat)
    assert m.any() and m.mean() < 0.05


def test_channel_alignment_fixes_offset():
    from apu_astro.studio import psf
    img = _elliptical_field(cov=((2.5, 0.0), (0.0, 2.5)), shift_r=(0.4, 0.6))
    co = psf.channel_offsets(img)
    assert abs(co[0][0][0] - 0.4) < 0.08 and abs(co[0][1][0] - 0.6) < 0.08
    al, info = psf.align_channels(img)
    co2 = psf.channel_offsets(al)
    assert abs(co2[0][0][0]) < 0.05 and abs(co2[0][1][0]) < 0.05 and info[0] > 0.5


def test_deconvolution_makes_stars_round_without_donuts():
    from apu_astro.studio import psf
    # 噪聲大一點：星心最多 150 倍噪聲，在 GUARD_SNR（300）以下才會變圓；更亮的星不做（會振鈴，見下一個測試）
    img = _elliptical_field(sat_level=0.45, noise=0.003)
    src = ch.Source([Path("synthetic.fits")], image=img)
    c = ch.Chain()
    c.set_source(src)
    c.steps = [ch.new_step("deconv")]
    out = c.evaluate(0, "full").state.layers["main"]
    before = psf.measure_grid(img.mean(axis=0), 1).overall
    after = psf.measure_grid(out.mean(axis=0), 1).overall
    assert after.ellipticity < before.ellipticity * 0.6
    # 飽和的平頂星核：反摺積後不能比周圍暗（甜甜圈）
    m = psf.saturation_mask(img)
    lab, n = ndimage_label(m)
    for i in range(1, n + 1):
        core = lab == i
        assert float(out.mean(axis=0)[core].min()) >= 0.95 * float(img.mean(axis=0)[core].min())


def test_deconvolution_leaves_bright_stars_without_ring():
    """很亮的星（星心比背景亮幾千倍噪聲）：變圓那段會振鈴，長軸方向星核外先掉下去再亮起來一圈，拉伸後就是甜甜圈
    （LBN576）。用 Moffat 形狀的橢圓星（高斯星跟反摺積的模型一樣，不會振鈴）：沿長軸往外不准變亮超過 3 倍噪聲。"""
    from apu_processing import detail as pdetail
    rng = np.random.default_rng(4)
    h = w = 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full((h, w), 0.02)

    def moffat(cy, cx, f):
        return f * (1 + ((xx - cx) / 3.5) ** 2 + ((yy - cy) / 2.0) ** 2) ** -2.5
    for _ in range(12):
        img += moffat(rng.uniform(30, h - 30), rng.uniform(30, w - 30), rng.uniform(0.05, 0.8))
    for _ in range(80):                                # 暗星（量星點形狀用）
        img += moffat(rng.uniform(10, h - 10), rng.uniform(10, w - 10), rng.uniform(0.005, 0.04))
    noise = 0.0002
    img = np.stack([img + rng.normal(0, noise, img.shape)] * 3).astype(np.float32)
    src = ch.Source([Path("synthetic.fits")], image=img)
    c = ch.Chain()
    c.set_source(src)
    c.steps = [ch.new_step("deconv")]
    out = c.evaluate(0, "full").state.layers["main"].mean(axis=0)
    L = img.mean(axis=0)
    st = pdetail.find_stars(L)
    order = np.argsort(-L[np.round(st.y).astype(int), np.round(st.x).astype(int)])
    checked = 0
    for i in order[:6]:
        y, x = int(round(st.y[i])), int(round(st.x[i]))
        if not (20 <= y < h - 20 and 20 <= x < w - 20):
            continue
        prof = out[y, x:x + 14]                        # 長軸（x 方向）
        assert float(np.max(np.diff(prof))) < 3 * noise, prof
        checked += 1
    assert checked >= 3


def ndimage_label(m):
    from scipy import ndimage
    return ndimage.label(m)


def _star_split(img: np.ndarray, scale: float):
    from apu_astro.studio.tools import State
    tool = get("star_split")
    ctx = Ctx(scale)
    state = State(layers={"main": img}, phase={"main": "linear"})
    return tool.run(state, tool.defaults(), "main", None, ctx), ctx


@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_star_split_at_quick_resolution_keeps_most_of_the_sky(scale):
    # 快速預覽（一個像素＝完整尺寸 4 px）：曾經外擴次數算成 0（scipy 的 0 次＝擴到填滿）整張被當成星，
    # 尖銳度門檻又讓所有的星不及格
    img = _star_field(h=480, w=640, seed=3)
    out, ctx = _star_split(img, scale)
    stars = out.layers["stars"].mean(axis=0)
    assert ctx.view["stars_found"] >= 20
    assert (stars > 0).mean() < 0.4
    # 星的位置在無星圖上不再是亮點
    sl = out.layers["starless"].mean(axis=0)
    assert float(np.percentile(sl, 99.9)) < float(np.percentile(img.mean(axis=0), 99.9)) * 0.5


def test_detect_stars_splits_close_pairs():
    # 門檻低了以後靠很近的兩顆星連成一條，find_stars 當成星雲纖維整塊丟掉（LBN576 一顆 300σ 的星留在無星圖上）；
    # mask 也要一塊一顆，recenter_stars 才會真的重新定位、拆開
    from scipy import ndimage

    from apu_astro.studio.tools import stars as tstars
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    L = np.full((h, w), 0.02, np.float32)
    for cy, cx in ((60, 60), (60, 65.5), (60, 71), (140, 140)):     # 三顆一串＋一顆單獨的
        L += 0.2 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3 ** 2))
    L += np.random.default_rng(0).normal(0, 0.0005, L.shape).astype(np.float32)
    st = tstars.detect_stars(L, 5.0)
    assert ndimage.label(st.mask)[1] == len(st.y)
    st = tstars.recenter_stars(L, st, 1.0)
    for cy, cx in ((60, 60), (60, 65.5), (60, 71), (140, 140)):
        assert float(np.min(np.hypot(st.y - cy, st.x - cx))) < 1.0


@pytest.mark.parametrize("nebula", [0.0, 0.001])
def test_star_split_removes_wings(nebula):
    # 星點模型：星核挖掉補、外圍扣模型。星的位置（8 px 內）要跟真正的背景差不到 1σ，而且洞不能大。
    # 以前整顆挖掉：範圍小（亮星只到 6 px），邊上還有星光、洞整個偏亮（平坦背景中位數 +0.28σ、6% 超過 1σ）；
    # 範圍大的話密集星場一半被補平
    rng = np.random.default_rng(5)
    h, w = 400, 400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    noise = 0.0005
    bg = 0.02 + nebula * np.sin(xx / 20.0) * np.cos(yy / 27.0)        # 星雲的紋路
    img = bg + rng.normal(0, noise, (h, w)).astype(np.float32)
    truth = img.copy()
    pos = []
    for _ in range(160):
        cy, cx = rng.uniform(20, h - 20), rng.uniform(20, w - 20)
        f = noise * 10 ** rng.uniform(np.log10(8), np.log10(1500))
        img += f * (1 + ((yy - cy) ** 2 + (xx - cx) ** 2) / 2.2 ** 2) ** -2.5      # Moffat：外圍比高斯寬
        pos.append((cy, cx, f))
    out, ctx = _star_split(np.stack([img] * 3).astype(np.float32), 1.0)
    sl = out.layers["starless"].mean(axis=0)
    assert np.asarray(ctx.view["holes"]).mean() < 0.2
    res = np.array([float((sl - truth)[np.hypot(yy - cy, xx - cx) < 8].mean()) / noise
                    for cy, cx, f in pos if f >= 20 * noise])        # 暗星在找星門檻邊緣，不算
    assert np.mean(np.abs(res) > 1) <= (0.03 if nebula == 0 else 0.1)
    if nebula == 0:
        assert abs(float(np.median(res))) < 0.2


def test_harmonic_fill_has_no_dark_centre():
    # 大洞中間不能比邊界暗（以前的補法中間用遠處平均，亮星的星暈補完一圈暗盤＋亮環）
    from apu_astro.studio import backend
    h, w = 400, 400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(yy - 200, xx - 200)
    img = (0.1 + 0.05 * 60 / np.maximum(r, 60))[None].astype(np.float32)
    mask = r < 80
    out = backend.harmonic_fill(img, mask)
    edge = float(img[0][(r > 80) & (r < 82)].mean())
    assert abs(float(out[0, 200, 200]) - edge) < 0.003
    assert np.array_equal(out[0][~mask], img[0][~mask])


def test_flat_tops_finds_saturated_star_below_global_max():
    # 飽和星停在 0.5，別處有更亮的 0.9：還是要認得出平頂（LBN576 預覽區的亮星因此沒被去掉）
    from apu_astro.studio.tools import stars as tstars
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    L = 0.02 + np.minimum(2.0 * np.exp(-((yy - 60) ** 2 + (xx - 60) ** 2) / (2 * 4.0 ** 2)), 0.5)
    L += 0.9 * np.exp(-((yy - 150) ** 2 + (xx - 150) ** 2) / (2 * 1.2 ** 2))
    L += np.random.default_rng(0).normal(0, 0.001, L.shape)
    hp = (L - tstars._background(L)).astype(np.float32)
    flat = tstars.flat_tops(hp, 1.0)
    assert flat[60, 60] and not flat[150, 150]


def test_deconvolution_does_not_dig_dark_rings():
    # 反摺積不准讓星點旁邊比局部背景暗 1σ 以上（暗環在 AI 降噪後很明顯，去星時變成暗洞）
    img = _elliptical_field(cov=((2.5, 0.0), (0.0, 2.5)))
    src = ch.Source([Path("synthetic.fits")], image=img)
    c = ch.Chain()
    c.set_source(src)
    c.steps = [ch.new_step("deconv")]
    out = c.evaluate(0, "full").state.layers["main"].mean(axis=0)
    L = img.mean(axis=0)
    sky, sig = ops.robust_stats(L)
    assert float(out.min()) >= min(float(L.min()), sky - 1.5 * sig)
