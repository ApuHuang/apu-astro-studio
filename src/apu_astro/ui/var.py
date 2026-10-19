"""可觀察的值：取代 Tk 的 StringVar／DoubleVar／BooleanVar。

頁面的狀態放在 Var 裡、元件綁上去；換語言時整個介面重建，Var（狀態）留著，新元件再綁上去就好。
元件被刪掉時（owner 的 destroyed）自動解除訂閱。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject


class Var:
    def __init__(self, value: Any = None):
        self._value = value
        self._subs: list[Callable[[Any], None]] = []

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        if value == self._value and type(value) is type(self._value):
            return
        self._value = value
        for cb in list(self._subs):
            try:
                cb(value)
            except RuntimeError:  # 元件的 C++ 物件已經刪掉（重建介面的空檔）
                self._discard(cb)

    def subscribe(self, cb: Callable[[Any], None], owner: QObject | None = None) -> Callable[[], None]:
        """value 改了就呼叫 cb(value)。owner 被刪掉時自動取消。回傳取消訂閱的函式。"""
        self._subs.append(cb)

        def unsubscribe(*_args: object) -> None:
            self._discard(cb)

        if owner is not None:
            owner.destroyed.connect(unsubscribe)
        return unsubscribe

    def _discard(self, cb: Callable[[Any], None]) -> None:
        try:
            self._subs.remove(cb)
        except ValueError:
            pass
