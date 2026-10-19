"""整合版自己的設定（語言、面板收合、視窗大小、上次在哪一步），存在系統的應用程式設定資料夾。

各步驟的處理參數（挑片的權重、疊圖的參數、後製的設定）照舊用各自套件的設定檔，跟獨立版共用：
同一個人在獨立版調好的參數，整合版打開就是一樣的。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAME = "APU Astro"


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_NAME


def settings_path() -> Path:
    override = os.environ.get("APU_ASTRO_SETTINGS")  # 測試用
    return Path(override) if override else config_dir() / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(**values: object) -> None:
    """合併寫入；存不進去就算了，只是下次要重選。"""
    data = load_settings()
    data.update(values)
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
