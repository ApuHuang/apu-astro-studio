"""新版本提醒：版本比較、挑這台電腦的 zip、一天查一次、頂部列的下載鍵（都不連網）。"""

from __future__ import annotations

from apu_astro import update
from apu_astro.settings import load_settings, save_settings

TAG = update.platform_tag()
RELEASE = {
    "tag_name": "v9.0.0",
    "html_url": "https://github.com/ApuHuang/apu-astro-studio/releases/tag/v9.0.0",
    "assets": [{"name": "APUAstro-9.0.0-other.zip", "browser_download_url": "https://example.com/other.zip"},
               {"name": f"APUAstro-9.0.0-{TAG}.zip", "browser_download_url": "https://example.com/mine.zip"},
               "junk"],
}


def test_newer_picks_this_platform():
    info = update.newer(RELEASE, "0.1.6")
    assert info == update.Update("9.0.0", "https://example.com/mine.zip", f"APUAstro-9.0.0-{TAG}.zip")
    assert update.newer(RELEASE, "9.0.0") is None and update.newer(RELEASE, "10.0") is None
    assert update.newer({"tag_name": "v9.0.0rc1"}, "0.1.6") is None      # 預覽版不算
    assert update.newer(None) is None and update.newer({"tag_name": "latest"}) is None
    # 沒有這台電腦的 zip（或連結不是 https）：給 Release 頁面
    page = update.newer({**RELEASE, "assets": [{"name": f"x-{TAG}.zip", "browser_download_url": "http://x"}]}, "0.1")
    assert page.url == RELEASE["html_url"] and page.file == ""


def test_checks_once_a_day():
    assert update.due({}) and update.cached({}) is None
    update.remember(RELEASE, now=1000.0)
    st = load_settings()
    assert update.cached(st)["tag_name"] == "v9.0.0"
    assert not update.due(st, now=1000.0 + 3600) and update.due(st, now=1000.0 + 25 * 3600)
    assert update.due(st, now=500.0)                                    # 時鐘倒退也重查
    save_settings(update=None)


def test_badge_shows_cached_release_without_network(app, window, monkeypatch):
    import time
    called = []
    monkeypatch.setattr(update, "fetch_latest", lambda *a, **k: called.append(1) or RELEASE)
    assert window.update_badge.isHidden()
    update.remember(RELEASE, now=time.time())
    try:
        window.check_for_updates()
        assert not window.update_badge.isHidden() and "9.0.0" in window.update_badge.text()
        assert window.update_info.url == "https://example.com/mine.zip" and not called   # 一天內不連網
        save_settings(check_updates=False)
        window.check_for_updates()
        assert window.update_badge.isHidden()
    finally:
        save_settings(update=None, check_updates=True)
