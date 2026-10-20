"""元件：滑桿直接輸入（夾到範圍、對齊步進）、影像檢視的座標換算與裁切框。"""

from __future__ import annotations

import numpy as np
from apu_astro.ui.imageview import ImageView
from apu_astro.ui.var import Var
from apu_astro.ui.widgets import ParameterSlider, Segmented, Switch
from conftest import pump
from PySide6.QtCore import QPointF


def test_var_notifies_and_unsubscribes(app):
    v = Var(1)
    seen = []
    unsub = v.subscribe(seen.append)
    v.set(2)
    v.set(2)
    unsub()
    v.set(3)
    assert seen == [2]


def test_slider_typed_value_is_clamped_and_snapped(app):
    var = Var(50.0)
    s = ParameterSlider("x", var, 0, 100, lambda v: f"{v:.0f}%", step=5)
    s._start_edit()
    s.edit.setText("123%")
    s._commit_edit()
    assert var.get() == 100
    s._start_edit()
    s.edit.setText("42")
    s._commit_edit()
    assert var.get() == 40
    assert s.slider.value() == 8
    var.set(65.0)
    assert s.value.text() == "65%"


def test_switch_and_segmented_follow_var(app):
    b = Var(False)
    sw = Switch(b)
    b.set(True)
    pump(app, 0.3)
    assert sw._offset > 0.9
    choice = Var("a")
    seg = Segmented([("a", "A"), ("b", "B")], choice)
    seg.resize(seg.sizeHint())
    assert seg._index() == 0
    choice.set("b")
    assert seg._index() == 1


def test_imageview_coordinates_and_crop(app):
    view = ImageView()
    view.resize(400, 300)
    view.show()
    img = np.random.default_rng(0).random((3, 600, 800), dtype=np.float32)
    view.set_image(img)
    pump(app, 0.1)
    assert abs(view.fit_zoom() - 0.5) < 1e-6
    x, y = view.to_image(QPointF(200, 150))
    assert abs(x - 400) < 1e-6 and abs(y - 300) < 1e-6
    view.set_zoom(1.0, QPointF(100, 100))
    x2, y2 = view.to_image(QPointF(100, 100))
    assert abs(x2 - 200) < 1e-6 and abs(y2 - 200) < 1e-6
    view.grab()  # 實際畫一次，不能出錯
    view.set_crop_mode(True)
    view.set_crop_box((10, 20, 110, 220))
    assert view.crop_box == (10, 20, 110, 220)
    view.set_compare(img * 0.5, ("a", "b"))
    view.grab()
    # 快速預覽（縮圖）換成完整結果時，以完整尺寸為準的位置不變
    cx, cy, z = view.cx, view.cy, view.zoom
    view.set_image(img[:, ::4, ::4], keep_view=True, pixel_scale=4)
    assert (view.cx, view.cy, view.zoom) == (cx, cy, z)
    # 自動裁邊在快速預覽上會差幾個像素（149×199 × 4 對 600×800）：還是同一張，縮放與位置不變
    view.set_image(np.ascontiguousarray(img[:, :149, :199]), keep_view=True, pixel_scale=4)
    assert (view.cx, view.cy, view.zoom) == (cx, cy, z)
    view.set_image(img, keep_view=True)
    assert (view.cx, view.cy, view.zoom) == (cx, cy, z)
    # 真的換了大小（例如裁掉一大塊）才回到整張
    view.set_image(img[:, :300, :400], keep_view=True)
    assert view.zoom is None
