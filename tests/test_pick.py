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
