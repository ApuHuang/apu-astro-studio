"""挑片：讀 APU Pick 的範例報表，結果與 Pick 相同；保留的片送去疊圖。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import pump

SAMPLES = Path(__file__).resolve().parents[2] / "astro-light-selector" / "samples"


@pytest.fixture
def samples(tmp_path: Path) -> Path:
    if not SAMPLES.is_dir():
        pytest.skip("APU Pick 的範例資料不在旁邊")
    folder = tmp_path / "lights"
    folder.mkdir()
    for f in SAMPLES.glob("*.fits"):
        shutil.copy2(f, folder / f.name)
    shutil.copy2(SAMPLES / "selection_report.csv", folder / "selection_report.csv")
    return folder


def test_open_folder_reads_report(app, window, samples):
    page = window.page("pick")
    page.open_folder(samples)
    pump(app, 0.5)
    sel = page.selection
    assert sel is not None and len(sel.decisions) == 11
    rejected = {Path(d.metrics.file).name for d in sel.decisions if not d.keep}
    assert {"bad_cloud.fits", "bad_seeing.fits", "bad_trailing.fits"} <= rejected
    assert page.table.rowCount() == 11
    assert page.chart.ordered and len(page.chart.ordered) == 11
    assert page.rail_text() == f"{sel.n_keep}/11"


def test_override_from_keyboard(app, window, samples):
    page = window.page("pick")
    page.open_folder(samples)
    pump(app, 0.3)
    page.view.set("list")
    row = next(r for r in range(page.table.rowCount()) if page.table.item(r, 0).text() == "light_000.fits")
    page.table.selectRow(row)
    page._set_override("reject")
    assert page.overrides == {"light_000.fits": "reject"}
    d = next(d for d in page.selection.decisions if Path(d.metrics.file).name == "light_000.fits")
    assert not d.keep and d.override == "reject"
    report = (samples / "selection_report.csv").read_text(encoding="utf-8-sig")
    assert "manual" in report or "reject" in report


def test_stop_keeps_measured_frames_and_measure_new(app, window, samples, monkeypatch):
    """Pick 0.9：按停止時量好的留著；「量測新檔案」只量還沒有結果的。"""
    from apu_astro.pages import pick as pick_mod
    page = window.page("pick")
    page.open_folder(samples)
    pump(app, 0.3)
    done, rest = page.frames[:4], page.frames[4:]
    page.frames = []
    page._handle(("cancelled", done))                       # 停止時引擎交回已量好的 4 張
    assert len(page.frames) == 4 and page.selection is not None and page.new_btn.isEnabled()
    started = []
    monkeypatch.setattr(page, "_run_measure", lambda files, home, kind: started.append((len(files), kind)))
    page._measure_new()
    assert started == [(7, "added")]                        # 只量剩下的 7 張
    page._handle(("added", rest))
    assert len(page.frames) == 11
    shown = []
    monkeypatch.setattr(pick_mod, "info", lambda _p, text: shown.append(text))
    started.clear()
    page._measure_new()
    assert not started and shown                            # 都量過了：只提示，不開工作


def test_threshold_and_grouping_are_remembered(app, window, samples):
    from astro_light_selector import settings as pick_settings

    from apu_astro.pages.pick import PickPage
    page = window.page("pick")
    page.open_folder(samples)
    pump(app, 0.3)
    try:
        page.method.set("keep_best")
        page.keep_best.set(60.0)
        page.grouping["night"].set(True)
        page._apply()
        st = pick_settings.load_settings()
        assert st["threshold"]["method"] == "keep_best" and st["threshold"]["keep_best"] == 60.0
        assert "night" in st["group_keys"]
        again = PickPage(window)                            # 下次開啟：照上次的
        assert again.method.get() == "keep_best" and again.keep_best.get() == 60.0 and again.grouping["night"].get()
        again.deleteLater()
    finally:
        pick_settings.save_settings(threshold=None, group_keys=None)


def _select(app, page, name: str) -> None:
    page.view.set("list")
    row = next(r for r in range(page.table.rowCount()) if page.table.item(r, 0).text() == name)
    page.table.selectRow(row)
    page._load_preview()
    pump(app, 20, lambda: page._preview_key in page._previews)


def test_info_histogram_and_field_panes(app, window, samples):
    """清單右邊的資訊、直方圖、像場（Pick 0.6～0.9）：文字來自 Pick，圖用 Qt 畫。"""
    from astro_light_selector import settings as pick_settings
    page = window.page("pick")
    page.open_folder(samples)
    pump(app, 0.3)
    _select(app, page, "light_000.fits")
    try:
        page.pane.set("info")
        text = page.info_text.toPlainText()
        assert "FWHM" in text and len(text) > 200
        page.pane.set("hist")
        assert page.hist_view.preview is not None and page.hist_view.preview.hist is not None
        assert page.hist_stats.toPlainText().strip()
        for stretch in ("linear", "stretched"):
            page.hist_stretch.set(stretch)
            assert page.hist_view.grab().width() > 0                    # 兩種座標都畫得出來（不丟例外）
        page.pane.set("field")
        key = page._preview_key
        pump(app, 60, lambda: not page._field_running and key in page._field_cache)
        result = page._field_cache[key]
        assert not isinstance(result, str), result
        assert page.field_plot.result is result and page.field_text.toPlainText().strip()
        for view in ("quad", "grid"):
            page.field_view.set(view)
            assert page.field_plot.grab().width() > 0
        page.field_mode.set("average")                                  # 分數最高的幾張保留片一起量
        pump(app, 120, lambda: not page._field_running and page.field_plot.result is not None)
        assert page.field_plot.result.n_frames > 1
        assert pick_settings.load_settings()["preview_pane"] == "field"
    finally:
        pick_settings.save_settings(preview_pane="preview", field_mode="single", field_view="grid",
                                    hist_stretch="stretched")


def test_pick_detail_loads_without_matplotlib(monkeypatch):
    """打包版沒有 matplotlib：Pick 的 detail 靠空殼載入，載完空殼要拿掉（別的套件 import matplotlib 仍然要失敗）。"""
    import builtins
    import sys

    from apu_astro.pages import pick_detail
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "matplotlib" or k.startswith("matplotlib.") or k in ("astro_light_selector.detail",
                                                                          "astro_light_selector.plot")}
    real_import = builtins.__import__

    def no_matplotlib(name, *a, **k):
        if (name == "matplotlib" or name.startswith("matplotlib.")) and name not in sys.modules:
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    monkeypatch.setattr(pick_detail, "_detail", None)
    try:
        detail = pick_detail.pick_detail()
        assert detail.histogram_stats(None) == [] and callable(detail.info_lines)
        assert "matplotlib" not in sys.modules and "matplotlib.figure" not in sys.modules
    finally:
        monkeypatch.undo()
        for k in ("astro_light_selector.detail", "astro_light_selector.plot"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        pick_detail._detail = None


def test_organize_and_undo(app, window, samples, monkeypatch):
    """整理檔案（Pick 0.6）：依曝光搬進子資料夾，量測結果跟著換位置；復原整理搬回原位。"""
    from apu_astro.pages import pick as pick_mod
    page = window.page("pick")
    page.open_folder(samples)
    pump(app, 0.3)
    monkeypatch.setattr(pick_mod, "ask_yes_no", lambda *a, **k: True)
    said = []
    monkeypatch.setattr(pick_mod, "info", lambda _p, text: said.append(text))
    for k, var in page.organize_by.items():
        var.set(k == "exposure")
    before = sorted(p.name for p in samples.glob("*.fits"))
    page._organize()
    assert not said, said
    moved = sorted(p.name for p in samples.rglob("*.fits") if p.parent != samples)
    assert moved == before and not list(samples.glob("*.fits"))          # 全部進了子資料夾
    assert len(page.frames) == 11 and all(Path(f.file).exists() for f in page.frames)
    assert page.recursive.get() and page.selection is not None and len(page.selection.decisions) == 11
    page._undo_organize()
    assert sorted(p.name for p in samples.glob("*.fits")) == before
    assert all(Path(f.file).parent == samples for f in page.frames)


def test_send_kept_frames_to_stack(app, window, samples):
    page = window.page("pick")
    page.open_folder(samples)
    pump(app, 0.3)
    kept = page.kept_paths()
    assert kept and all(p.exists() for p in kept)
    page.send_to_stack()
    pump(app, 0.2)
    assert window.step.get() == "stack"
    stack = window.page("stack")
    assert {p.name for p in stack.files} == {p.name for p in kept}
    pump(app, 20, lambda: not stack._planning)
    assert stack.plan is not None and len(stack.plan.lights) == len(kept)
