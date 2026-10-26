"""外殼：翻譯完整、換步驟、換語言（三個套件一起換）、拖放分派。"""

from __future__ import annotations

from apu_astro import i18n
from conftest import pump


def test_catalogs_have_same_keys():
    assert set(i18n._ZH) == set(i18n._EN)


def test_set_language_switches_all_packages():
    from apu_photons import i18n as ph
    from apu_processing import i18n as pr
    from astro_light_selector import i18n as pk
    i18n.set_language("en")
    try:
        assert pk.get_language() == ph.get_language() == pr.get_language() == "en"
        assert i18n.tr("step.pick") == "Pick"
    finally:
        i18n.set_language("zh")
    assert pk.get_language() == "zh"


def test_steps_and_lazy_pages(app, window):
    assert set(window.pages) == {window.step.get()}
    for key in ("pick", "stack", "process"):
        window.step.set(key)
        pump(app, 0.3)
        assert window.stack.currentWidget() is window.pages[key]
    assert set(window.pages) == {"pick", "stack", "process"}


def test_language_rebuilds_pages(app, window):
    for key in ("pick", "stack", "process"):
        window.page(key)
    window.lang_var.set("en")
    pump(app, 0.3)
    try:
        assert window.pages["pick"].measure_btn.text() == "Measure"
        assert window.pages["stack"].stack_btn.text() == "Stack"
        assert window.title.text() == "Nothing open"
    finally:
        window.lang_var.set("zh")
        pump(app, 0.2)
    assert window.pages["pick"].measure_btn.text() == "量測"


def test_startup_recipe_goes_to_process(app, window, tmp_path, monkeypatch):
    opened = []
    page = window.page("process")
    monkeypatch.setattr(page, "handle_drop", lambda paths: opened.extend(paths))
    window.step.set("pick")
    recipe = tmp_path / "M42.recipe.json"
    window.open_startup([recipe])
    assert window.step.get() == "process" and opened == [recipe]


def test_studio_buttons_enabled_after_open(app, window, tmp_path):
    # 0.1.3／0.1.4：開好圖後出圖、專案、檢視切換還是停用，要切換語言（整頁重建）才能按
    import numpy as np
    from astropy.io import fits
    img = tmp_path / "x.fits"
    rng = np.random.default_rng(0)
    fits.writeto(img, (0.1 + 0.01 * rng.standard_normal((3, 120, 160))).astype(np.float32))
    window.step.set("process")
    page = window.page("process")
    assert not page.export_btn.isEnabled()
    page.open_paths([img])
    pump(app, 30.0, lambda: page.chain.source is not None and not page.loading)
    assert page.export_btn.isEnabled() and page.project_btn.isEnabled() and page.view_seg.isEnabled()


def test_close_without_work(app, window):
    assert not any(p.is_busy() for p in window.pages.values())
    assert window.close()
