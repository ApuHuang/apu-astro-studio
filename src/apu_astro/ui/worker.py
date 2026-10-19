"""背景工作：在 Python 執行緒跑，結果用 Qt 訊號送回主執行緒（queued connection）。

用法同 Pick／Photons／Processing 的 events 佇列：背景執行緒呼叫 bridge.post(("kind", ...))，
主執行緒的 handler 收到一個 tuple。頁面已經關掉時 post 會安靜失敗。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal


class Bridge(QObject):
    event = Signal(object)

    def __init__(self, handler: Callable[[tuple], None], parent: QObject | None = None):
        super().__init__(parent)
        self.event.connect(handler)

    def post(self, event: tuple) -> None:
        try:
            self.event.emit(event)
        except RuntimeError:  # 接收的頁面已經刪掉
            pass


def run_thread(fn: Callable[[], None], name: str = "apu-astro-worker") -> threading.Thread:
    t = threading.Thread(target=fn, name=name, daemon=True)
    t.start()
    return t
