"""0.2.0 階段 5：預設集（標準／阿噗流程／自訂）、去星後的分支分組、4 個階段、專案檔 /2、狀態標籤、設定移到頁首。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import pump
from test_studio import _chain, _star_field, _write_xisf

from apu_astro.settings import load_settings, save_settings
from apu_astro.studio import chain as ch
from apu_astro.studio import presets


def _kinds(steps, idx):
    return [(steps[i].kind, steps[i].layer) for i in idx]


def test_standard_and_apu_use_the_same_steps():
    """標準只是介面摺疊：步驟跟阿噗流程（＝0.1.x 的建議流程）完全一樣，成品不變。"""
    a = presets.template(presets.STANDARD, True, None)
    b = presets.template(presets.APU, True, None)
    assert presets.same_steps(a, b) and presets.same_steps(a, ch.default_steps(True))
    assert presets.view_of(presets.STANDARD) == "simple" and presets.view_of(presets.APU) == "full"


def test_branch_groups_and_merge_point():
    steps = ch.default_steps(True)
    br = presets.branch_of(steps)
    split = next(i for i, s in enumerate(steps) if s.kind == "star_split")
    assert br[split] == "split" and br[-1] == "merge" and all(b is None for b in br[:split])
    rows = presets.display_order(steps)
    headers = [r[1] for r in rows if r[0] == "header"]
    assert headers == ["starless", "stars"]
    starless = [r[1] for r in rows if r[0] == "step" and r[2] == "starless"]
    stars = [r[1] for r in rows if r[0] == "step" and r[2] == "stars"]
    assert [steps[i].kind for i in stars] == ["stretch"]
    assert [steps[i].kind for i in starless] == ["denoise", "stretch", "color_grade", "sharpen", "denoise"]
    assert sorted(r[1] for r in rows if r[0] == "step") == list(range(len(steps)))    # 每一步都出現一次


def test_move_within_branch_only():
    steps = ch.default_steps(True)
    stars_stretch = next(i for i, s in enumerate(steps) if s.kind == "stretch" and s.layer == "stars")
    assert presets.neighbour_in_branch(steps, stars_stretch, -1) is None       # 星點組只有一步
    assert presets.neighbour_in_branch(steps, stars_stretch, 1) is None
    starless_stretch = next(i for i, s in enumerate(steps) if s.kind == "stretch" and s.layer == "starless")
    grade = next(i for i, s in enumerate(steps) if s.kind == "color_grade")
    assert presets.neighbour_in_branch(steps, starless_stretch, 1) == grade    # 跳過中間的星點組
    assert presets.neighbour_in_branch(steps, 0, -1) is None


def test_stages():
    steps = ch.default_steps(True)
    st = presets.stage_of(steps)
    by = {k: [steps[i].kind for i in range(len(steps)) if st[i] == k] for k in range(4)}
    assert by[0][:3] == ["geometry", "gradient", "color_cal"] and "deconv" in by[0]
    assert by[1] == ["star_split", "denoise", "stretch", "stretch"]
    assert by[2] == ["color_grade", "sharpen", "denoise"] and by[3] == ["star_merge"]
    for s in steps:
        if s.kind in presets.KEY_PARAMS or s.kind == "stretch":
            keys = {c.get("key") for c in __import__("apu_astro.studio.tools", fromlist=["get"]).get(s.kind).controls()}
            assert set(presets.key_params(s)) <= keys


def test_project_v2_roundtrip_and_v1_compat(tmp_path):
    c = _chain()
    path = tmp_path / "p.apuastro"
    ch.save_project(c, path, {"preset": presets.STANDARD, "view": "simple"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "apu-astro-studio/2" and data["preset"] == "standard"
    src, steps, meta = ch.load_project(path)
    assert presets.same_steps(steps, c.steps) and meta == {"preset": "standard", "view": "simple"}
    # 0.1.x 存的 /1：步驟照讀，沒有預設集，用完整清單
    v1 = {"format": "apu-astro-studio/1", "source": data["source"], "steps": data["steps"]}
    old = tmp_path / "old.apuastro"
    old.write_text(json.dumps(v1), encoding="utf-8")
    src1, steps1, meta1 = ch.load_project(old)
    assert presets.same_steps(steps1, c.steps) and meta1 == {"preset": None, "view": "full"}
    new = tmp_path / "new.apuastro"
    new.write_text(json.dumps({**v1, "format": "apu-astro-studio/9"}), encoding="utf-8")
    with pytest.raises(ch.ProjectFormatError):
        ch.load_project(new)


def test_custom_preset_dicts_roundtrip():
    steps = ch.default_steps(True)
    steps[1].params["smooth"] = 0.9
    steps[3].enabled = False
    back = presets.steps_from_dicts(presets.steps_to_dicts(steps))
    assert presets.same_steps(back, steps) and {s.id for s in back}.isdisjoint({s.id for s in steps})


def _open(app, window, tmp_path, preset):
    save_settings(studio_preset=preset)
    p = tmp_path / "m.xisf"
    _write_xisf(p, _star_field())
    window.step.set("process")
    page = window.page("process")
    page.open_paths([p])
    pump(app, 30, lambda: page.chain.source is not None and not page.loading)
    last = len(page.chain.steps) - 1
    pump(app, 60, lambda: page.chain.cached(last, "quick") is not None and page._job is None)
    return page


def test_standard_view_stages_and_key_sliders(app, window, tmp_path):
    page = _open(app, window, tmp_path, presets.STANDARD)
    assert page.view_kind == "simple" and page.steps_panel.view == "simple"
    rows = [page.steps_panel.list.item(r).data(0x0100) for r in range(page.steps_panel.list.count())]
    assert [r for r in rows if r[0] == "stage"] == [("stage", k) for k in range(4)]
    page.steps_panel.select_stage(1)
    pump(app, 0.3)
    assert page.stage == 1 and page.selected == presets.stage_steps(page.chain.steps, 1)[-1]
    # 狀態標籤：階段名稱、到第幾步、解析度
    pump(app, 30, lambda: page.chain.cached(page.selected, "quick") is not None and page._job is None)
    page.refresh_view()
    assert "去星與拉伸" in page.image.label_text and "預覽" in page.image.label_text
    # 展開細步驟：階段下面列出步驟，可以選單一步
    page._expand_stage(1)
    split = next(i for i, s in enumerate(page.chain.steps) if s.kind == "star_split")
    page.steps_panel.select(split)
    pump(app, 0.3)
    assert page.selected == split and page.stage is None
    if page._job:
        page._job.cancelled.set()


def test_preset_switch_save_delete(app, window, tmp_path, monkeypatch):
    page = _open(app, window, tmp_path, presets.STANDARD)
    page.chain.steps[1].params["smooth"] = 0.95          # 改過參數：換預設集要先問
    asked = []
    monkeypatch.setattr(page, "_confirm", lambda text: asked.append(text) or False)
    page._preset_action("choose:" + presets.APU)
    assert asked and page.preset == presets.STANDARD and page.chain.steps[1].params["smooth"] == 0.95
    monkeypatch.setattr(page, "_confirm", lambda text: True)
    page._preset_action("choose:" + presets.APU)
    assert page.preset == presets.APU and page.view_kind == "full"
    assert presets.same_steps(page.chain.steps, ch.default_steps(True))
    assert load_settings().get("studio_preset") == presets.APU
    # 存成自訂預設集
    page.chain.steps[1].params["smooth"] = 0.7
    monkeypatch.setattr(page, "_ask_name", lambda: "我的流程")
    page._preset_action("save")
    assert page.preset == presets.CUSTOM + "我的流程"
    assert load_settings()["studio_presets"]["我的流程"]["steps"][1]["params"]["smooth"] == 0.7
    page._preset_action("choose:" + presets.STANDARD)
    page._preset_action("choose:" + presets.CUSTOM + "我的流程")
    assert page.chain.steps[1].params["smooth"] == 0.7 and page.view_kind == "full"
    page._preset_action("delete:我的流程")
    assert "我的流程" not in load_settings().get("studio_presets", {}) and page.preset is None
    if page._job:
        page._job.cancelled.set()


def test_full_view_groups_and_settings_in_header(app, window, tmp_path):
    page = _open(app, window, tmp_path, presets.APU)
    rows = [page.steps_panel.list.item(r).data(0x0100) for r in range(page.steps_panel.list.count())]
    assert ("header", "starless") in rows and ("header", "stars") in rows
    # 「預覽與效能」不在步驟面板，在頁首的齒輪
    from PySide6.QtWidgets import QAbstractButton, QLabel
    for i in (-1, 0, len(page.chain.steps) - 1):
        page.select(i)
        pump(app, 0.1)
        texts = [w.text() for w in page.inspector.widget().findChildren(QLabel)] +                 [w.text() for w in page.inspector.widget().findChildren(QAbstractButton)]
        assert texts and not any("預覽與效能" in x or "使用 GPU" in x for x in texts)
    page._fill_settings_menu()
    names = [a.text() for a in page.settings_menu.actions()]
    assert any("完整解析度" in n for n in names) and any("GPU" in n for n in names)
    # 星點分支的拉伸不能往上移到無星圖組
    stars = next(i for i, s in enumerate(page.chain.steps) if s.kind == "stretch" and s.layer == "stars")
    page.steps_panel.select(stars)
    pump(app, 0.2)
    assert not page.steps_panel.up_btn.isEnabled()
    before = [s.id for s in page.chain.steps]
    page.move_step(-1)
    assert [s.id for s in page.chain.steps] == before
    # 無星圖的拉伸往下：跟調色交換（中間的星點組不動）
    sl = next(i for i, s in enumerate(page.chain.steps) if s.kind == "stretch" and s.layer == "starless")
    grade = next(i for i, s in enumerate(page.chain.steps) if s.kind == "color_grade")
    page.steps_panel.select(sl)
    pump(app, 0.2)
    page.move_step(1)
    assert page.chain.steps[grade].kind == "stretch" and page.chain.steps[sl].kind == "color_grade"
    if page._job:
        page._job.cancelled.set()


def test_save_project_writes_preset(app, window, tmp_path, monkeypatch):
    page = _open(app, window, tmp_path, presets.STANDARD)
    out = tmp_path / "x.apuastro"
    monkeypatch.setattr("apu_astro.pages.studio.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), ""))
    page.ask_save_project()
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["format"] == "apu-astro-studio/2" and data["preset"] == "standard" and data["view"] == "simple"
    if page._job:
        page._job.cancelled.set()
