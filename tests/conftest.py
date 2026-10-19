"""測試共用：Qt 用 offscreen（不開視窗），設定檔寫到暫存資料夾，三個套件的設定也不動使用者的。"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
_TMP = Path(tempfile.mkdtemp(prefix="apu_astro_tests_"))
os.environ["APU_ASTRO_SETTINGS"] = str(_TMP / "astro.json")
os.environ["APU_PHOTONS_SETTINGS"] = str(_TMP / "photons.json")
os.environ["APU_PROCESSING_SETTINGS"] = str(_TMP / "processing.json")

import pytest  # noqa: E402
from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# APU Pick 的設定沒有環境變數可以改：測試時把它的設定資料夾指到暫存資料夾
from astro_light_selector import settings as _pick_settings  # noqa: E402

_pick_settings.config_dir = lambda: _TMP / "pick"


@pytest.fixture(scope="session")
def app() -> QApplication:
    from apu_astro.i18n import set_language
    from apu_astro.theme import apply_theme
    qapp = QApplication.instance() or QApplication([])
    set_language("zh")
    apply_theme(qapp)
    return qapp


def pump(app: QApplication, seconds: float = 0.2, until=None) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 30)
        if until is not None and until():
            return
        time.sleep(0.01)


@pytest.fixture
def window(app):
    from apu_astro.window import MainWindow
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    pump(app, 0.2)
    yield win
    for page in win.pages.values():
        page.close_page()
    win.hide()
    win.deleteLater()
    pump(app, 0.1)
