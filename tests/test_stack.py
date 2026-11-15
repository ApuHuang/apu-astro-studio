"""疊圖頁：Photons 0.3 的內插限幅／畸變修正、共用設定不互相洗掉、載入紀錄完全重現、清除快取。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from apu_photons import settings as photons_settings
from apu_photons.engine import Settings

from apu_astro.pages import stack as stack_mod


def test_saving_params_keeps_what_standalone_saved(app, window):
    """params 是整包覆寫的：獨立版存的參數（包括這裡不認識的）不能被整合版洗掉。"""
    photons_settings.save_settings(params={"distortion": True, "clamp": False, "from_a_newer_photons": 7})
    try:
        page = window.page("stack")
        assert page.distortion.get() is True and page.clamp.get() is False
        page.weights.set(not page.weights.get())             # 隨便改一個參數 → 存檔
        p = photons_settings.load_settings()["params"]
        assert p["distortion"] is True and p["clamp"] is False and p["from_a_newer_photons"] == 7
        s = page.settings()
        assert s.distortion is True and s.clamp is False
        page.clamp.set(True)
        assert photons_settings.load_settings()["params"]["clamp"] is True and page.settings().clamp is True
    finally:
        photons_settings.save_settings(params={})


def test_recipe_reproduce_carries_clamp_and_distortion(app, window, monkeypatch, tmp_path):
    """0.2 的紀錄完全重現時 Photons 把限幅關掉（那時沒有）：頁面要跟著關，不能又用回預設。"""
    page = window.page("stack")
    assert page.clamp.get() is True and page.distortion.get() is False
    rec = SimpleNamespace(settings=Settings(clamp=False, distortion=True), files=[], warnings=[])
    monkeypatch.setattr(stack_mod, "load_recipe", lambda *a, **k: rec)
    try:
        page.load_recipe_file(tmp_path / "x.recipe.json", reproduce=True)
        s = page.settings()
        assert s.clamp is False and s.distortion is True
    finally:
        photons_settings.save_settings(params={})


def test_cache_size_and_clear(app, window, monkeypatch, tmp_path):
    page = window.page("stack")
    out = tmp_path / "out"
    work = out / ".photons_cache" / "calibrated"
    work.mkdir(parents=True)
    (work / "a.npy").write_bytes(b"\0" * 300_000)
    (out / "master.fits").write_bytes(b"keep")
    page.output_dir.set(str(out))
    page.refresh()
    assert page.cache_btn.isEnabled() and page._cache_bytes == 300_000
    monkeypatch.setattr(stack_mod, "ask_yes_no", lambda *a, **k: False)
    page._clear_cache()
    assert (work / "a.npy").exists()                          # 按「否」不刪
    monkeypatch.setattr(stack_mod, "ask_yes_no", lambda *a, **k: True)
    page._clear_cache()
    assert not (work / "a.npy").exists() and (out / "master.fits").exists()
    assert not page.cache_btn.isEnabled() and page._cache_bytes == 0
    assert Path(out).is_dir()
