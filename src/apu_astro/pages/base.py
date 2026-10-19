"""三個步驟頁面的共同骨架：頁首（左邊分頁切換、右邊動作按鈕）、內容區、右側面板，狀態交給外殼的狀態列。

頁面的狀態（資料、設定）放在頁面物件上；介面元件在 build_ui() 裡建立，換語言時整個 rebuild()，狀態不動。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from ..theme import D
from ..ui.widgets import Inspector, PanelGroup

if TYPE_CHECKING:
    from ..window import MainWindow


class Page(QWidget):
    key = ""
    status_changed = Signal()   # 狀態列的文字、進度、摘要
    title_changed = Signal()    # 頂部列中間的名稱
    rail_changed = Signal()     # 左側步驟欄的小字

    def __init__(self, shell: MainWindow):
        super().__init__()
        self.setObjectName("Page")
        self.shell = shell
        self._status_fn: Callable[[], str] = lambda: ""
        self._summary_fn: Callable[[], str] = lambda: ""
        self.progress: float | None = None     # 0–1；None＝沒有工作在跑
        self.has_result = False                # 狀態列的小圓點是藍色
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._ui: QWidget | None = None

    # ------------------------------------------------------------------ 給子類別覆寫

    def build_ui(self) -> QWidget:
        raise NotImplementedError

    def refresh(self) -> None:
        """依目前狀態更新介面。"""

    def doc_title(self) -> str:
        return ""

    def rail_text(self) -> str:
        return ""

    def is_busy(self) -> bool:
        return False

    def close_page(self) -> None:
        """取消背景工作（視窗關閉前）。"""

    def ask_open(self) -> None:
        """Ctrl+O。"""

    def handle_drop(self, paths: list[Path]) -> None:
        """把檔案或資料夾拖進視窗。"""

    def shortcut(self, name: str) -> None:
        """其他快捷鍵（save、fit、actual）。"""

    # ------------------------------------------------------------------ 共用

    def rebuild(self) -> None:
        if self._ui is not None:
            self._ui.hide()
            self._ui.setParent(None)
            self._ui.deleteLater()
        self._ui = self.build_ui()
        self._lay.addWidget(self._ui)
        self.refresh()
        self.title_changed.emit()
        self.rail_changed.emit()
        self.status_changed.emit()

    def status_text(self) -> str:
        return self._status_fn()

    def summary_text(self) -> str:
        return self._summary_fn()

    def set_status(self, fn: Callable[[], str] | str, progress: float | None | bool = False) -> None:
        """狀態存成函式：換語言時用新語言重新產生。progress=False 表示不改進度。"""
        self._status_fn = fn if callable(fn) else (lambda text=fn: text)
        if progress is not False:
            self.progress = progress
        self.status_changed.emit()

    def set_summary(self, fn: Callable[[], str] | str) -> None:
        self._summary_fn = fn if callable(fn) else (lambda text=fn: text)
        self.status_changed.emit()

    def group(self, key: str, title: str, info: str | None = None) -> PanelGroup:
        """右側面板的一組，收合狀態記在整合版的設定裡。"""
        return PanelGroup(f"{self.key}.{key}", title, info, self.shell.panel_state, self.shell.save_panel)

    @staticmethod
    def layout(left: Sequence[QWidget], right: Sequence[QWidget], content: QWidget,
               inspector: Inspector | None) -> QWidget:
        """頁首＋（內容｜右側面板）。"""
        ui = QWidget()
        ui.setObjectName("Body")
        v = QVBoxLayout(ui)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        header = QWidget()
        header.setObjectName("PageHeader")
        header.setFixedHeight(D.header_height)
        header.setStyleSheet(f"QWidget#PageHeader {{ background: {D.chrome}; border-bottom: 1px solid {D.separator}; }}")
        h = QHBoxLayout(header)
        h.setContentsMargins(14, 0, 14, 0)
        h.setSpacing(8)
        for w in left:
            h.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
        h.addStretch(1)
        for w in right:
            h.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addWidget(header)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(content, 1)
        if inspector is not None:
            body.addWidget(inspector)
        v.addLayout(body, 1)
        return ui


class Views(QStackedWidget):
    """內容區的幾個分頁（例如趨勢圖／清單／門檻細節），用頁首的 Segmented 切換。"""

    def __init__(self, pages: dict[str, QWidget]):
        super().__init__()
        self.keys = list(pages)
        for w in pages.values():
            self.addWidget(w)

    def show_key(self, key: str) -> None:
        if key in self.keys:
            self.setCurrentIndex(self.keys.index(key))
