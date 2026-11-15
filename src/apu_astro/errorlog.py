"""沒預期到的錯誤：記到設定資料夾的 error.log，並跳出對話框（做法照 APU Pick 0.9）。

背景工作（量測、疊圖、後製）的錯誤各頁面自己會顯示；這裡接的是介面本身出錯（按鍵的處理、畫面更新）：
Qt 預設只印到主控台，打包成沒有主控台的 exe 後使用者只會覺得「按了沒反應」。這個模組不 import Qt。
"""

from __future__ import annotations

import datetime
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable

from . import __version__
from .settings import settings_path

MAX_BYTES = 512 * 1024      # 超過就只留後半（不會越寫越大）
_showing = False            # 對話框開著時又出錯：只記錄，不再疊一個對話框


def log_path() -> Path:
    return settings_path().with_name("error.log")


def write(text: str) -> Path | None:
    """加一筆到 error.log；寫不進去回傳 None。"""
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            old = path.read_text(encoding="utf-8", errors="replace")
            path.write_text(old[len(old) // 2:], encoding="utf-8")
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"--- {stamp}  APU Astro {__version__}  {sys.platform}\n{text.rstrip()}\n\n")
        return path
    except OSError:
        return None


def install(show: Callable[[str, Path | None], None] | None = None) -> None:
    """接手主執行緒與背景執行緒沒人處理的例外。show(錯誤最後一行, 紀錄檔)：在主執行緒顯示給使用者。"""

    def hook(kind, value, tb) -> None:
        global _showing
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return
        text = "".join(traceback.format_exception(kind, value, tb))
        path = write(text)
        try:
            if sys.__stderr__ is not None:
                sys.__stderr__.write(text)
        except Exception:  # noqa: BLE001
            pass
        if show is not None and not _showing and threading.current_thread() is threading.main_thread():
            _showing = True
            try:
                show(f"{kind.__name__}: {value}", path)
            except Exception:  # noqa: BLE001  顯示錯誤時又出錯：已經記下來了，不要再往外丟
                pass
            finally:
                _showing = False

    sys.excepthook = hook
    threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)
