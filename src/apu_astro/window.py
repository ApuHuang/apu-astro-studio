"""主視窗：頂部列（名稱、目前的專案、語言）、左側步驟欄（挑片 › 疊圖 › 後製）、步驟頁面、底部狀態列。

頁面第一次點到才建立（沒用到的步驟不載入）；切換只是換頁，結果與進度都留著。
交棒：挑片 → send_to_stack(檔案) → 疊圖；疊圖 → send_to_process(recipe) → 後製。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QCloseEvent, QKeySequence, QPainter, QPaintEvent, QShortcut
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMainWindow, QProgressBar,
                               QStackedWidget, QVBoxLayout, QWidget)

from . import __version__, icons
from .i18n import APP_NAME, get_language, set_language, tr
from .pages.base import Page
from .settings import load_settings, save_settings
from .theme import D, dark_title_bar, fonts
from .ui.var import Var
from .ui.widgets import Badge, Segmented, ask_yes_no, label

STEPS = ("pick", "stack", "process")


# ---------------------------------------------------------------------- 步驟欄


class StepButton(QWidget):
    clicked = Signal(str)

    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.selected = False
        self.hover = False
        self.sub = ""
        self.progress: float | None = None
        self.setFixedSize(D.rail_width - 12, 68)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr(f"step.{key}.help"))

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)

    def enterEvent(self, _e) -> None:  # noqa: N802
        self.hover = True
        self.update()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        self.hover = False
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.hover and not self.selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#1c1c1e"))
            p.drawRoundedRect(QRectF(self.rect()), 10, 10)
        color = "#ffffff" if self.selected else D.secondary
        pm = icons.pixmap(self.key, color, 22)
        p.drawPixmap(int((self.width() - 22) / 2), 9, pm)
        p.setPen(QColor(color))
        p.setFont(fonts().ui(9, bold=self.selected))
        p.drawText(QRect(0, 33, self.width(), 16), Qt.AlignmentFlag.AlignCenter, tr(f"step.{self.key}"))
        if self.sub:
            p.setPen(QColor(D.accent))
            p.setFont(fonts().ui(7.5))
            p.drawText(QRect(0, 49, self.width(), 14), Qt.AlignmentFlag.AlignCenter,
                       p.fontMetrics().elidedText(self.sub, Qt.TextElideMode.ElideRight, self.width() - 6))
        if self.progress is not None:
            r = QRectF(10, self.height() - 4, self.width() - 20, 3)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(D.control_hover))
            p.drawRoundedRect(r, 1.5, 1.5)
            p.setBrush(QColor(D.accent))
            p.drawRoundedRect(QRectF(r.left(), r.top(), r.width() * max(0.02, min(1.0, self.progress)), r.height()),
                              1.5, 1.5)


class StepRail(QFrame):
    """左側的三個步驟；選中的底色會滑過去。"""

    def __init__(self, current: Var):
        super().__init__()
        self.current = current
        self.setFixedWidth(D.rail_width)
        self.setObjectName("Rail")
        self.setStyleSheet(f"QFrame#Rail {{ background: {D.rail}; border-right: 1px solid {D.separator}; }}")
        self.pill = QFrame(self)
        self.pill.setStyleSheet(f"background: {D.selected_step}; border-radius: 10px;")
        self.anim = QPropertyAnimation(self.pill, b"geometry", self)
        self.anim.setDuration(200)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 10, 6, 10)
        lay.setSpacing(0)
        self.buttons: dict[str, StepButton] = {}
        for i, key in enumerate(STEPS):
            if i:
                line = QFrame()
                line.setFixedSize(1, 14)
                line.setStyleSheet(f"background: {D.separator_strong};")
                lay.addWidget(line, 0, Qt.AlignmentFlag.AlignHCenter)
            b = StepButton(key)
            b.clicked.connect(current.set)
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
            self.buttons[key] = b
        lay.addStretch(1)
        current.subscribe(lambda _v: self._select(animate=True), self)
        self._select(animate=False)

    def _select(self, animate: bool) -> None:
        for key, b in self.buttons.items():
            b.selected = key == self.current.get()
            b.update()
        b = self.buttons.get(self.current.get())
        if b is None:
            return
        target = b.geometry()
        if animate and self.isVisible() and self.pill.geometry().isValid() and self.pill.width() > 0:
            self.anim.stop()
            self.anim.setStartValue(self.pill.geometry())
            self.anim.setEndValue(target)
            self.anim.start()
        else:
            self.pill.setGeometry(target)
        self.pill.lower()

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._select(animate=False)

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self._select(animate=False)

    def set_sub(self, key: str, text: str, progress: float | None) -> None:
        b = self.buttons[key]
        b.sub, b.progress = text, progress
        b.update()

    def retranslate(self) -> None:
        for key, b in self.buttons.items():
            b.setToolTip(tr(f"step.{key}.help"))
            b.update()


# ---------------------------------------------------------------------- 狀態列


class Dot(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.color = D.secondary
        self.setFixedSize(8, 8)

    def set_color(self, color: str) -> None:
        self.color = color
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.color))
        p.drawEllipse(QRectF(1, 1, 6, 6))


class StatusBar(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("StatusBar")
        self.setFixedHeight(D.status_bar_height)
        self.setStyleSheet(f"QFrame#StatusBar {{ background: {D.chrome}; border-top: 1px solid {D.separator}; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(8)
        self.dot = Dot()
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(120)
        self.text = label("", D.secondary)
        self.summary = label("", D.label)
        lay.addWidget(self.dot)
        lay.addWidget(self.bar)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.summary)

    def show_page(self, page: Page | None) -> None:
        if page is None:
            self.text.setText("")
            self.summary.setText("")
            self.bar.hide()
            self.dot.show()
            return
        busy = page.progress is not None
        self.bar.setVisible(busy)
        self.dot.setVisible(not busy)
        if busy:
            self.bar.setValue(int(max(0.0, min(1.0, page.progress)) * 1000))
        self.dot.set_color(D.accent if page.has_result else D.secondary)
        text = page.status_text()
        self.text.setText(self.text.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight,
                                                             max(100, self.text.width())))
        self.text.setToolTip(text)
        self.summary.setText(page.summary_text())


# ---------------------------------------------------------------------- 主視窗


class MainWindow(QMainWindow):
    def __init__(self, open_paths: list[str] | None = None):
        super().__init__()
        st = load_settings()
        self.panel_state: dict[str, bool] = dict(st.get("panel", {}))
        self.lang_var = Var(get_language())
        self.step = Var(st.get("step", "pick") if st.get("step") in STEPS else "pick")
        self.pages: dict[str, Page] = {}
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.setWindowIcon(icons.app_icon())
        self.setAcceptDrops(True)
        self.setMinimumSize(1100, 720)

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_top_bar())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.rail = StepRail(self.step)
        body.addWidget(self.rail)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        v.addLayout(body, 1)
        self.status_bar = StatusBar()
        v.addWidget(self.status_bar)
        self.setCentralWidget(central)

        self.step.subscribe(lambda _v: self._show_step(fade=True), self)
        self.lang_var.subscribe(self._language_clicked, self)
        self._shortcuts()
        self._show_step(fade=False)
        if open_paths:
            self.open_startup([Path(p) for p in open_paths])

    # ------------------------------------------------------------------ 版面

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(D.top_bar_height)
        bar.setStyleSheet(f"QFrame#TopBar {{ background: {D.chrome}; border-bottom: 1px solid {D.separator}; }}")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 14, 0)
        h.setSpacing(8)
        name = label(APP_NAME, D.label, 11, bold=True)
        h.addWidget(name)
        h.addWidget(Badge(f"v{__version__}"), 0, Qt.AlignmentFlag.AlignVCenter)
        h.addStretch(1)
        self.title_icon = QLabel()
        self.title_icon.setPixmap(icons.pixmap("folder", D.secondary, 15))
        h.addWidget(self.title_icon)
        self.title = label("", D.secondary, 9.5)
        h.addWidget(self.title)
        h.addStretch(1)
        self.lang_seg = Segmented([("zh", tr("lang.zh")), ("en", tr("lang.en"))], self.lang_var)
        h.addWidget(self.lang_seg)
        return bar

    def _shortcuts(self) -> None:
        def add(seq, fn) -> None:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)

        add(QKeySequence.StandardKey.Open, lambda: self.current_page().ask_open())
        add(QKeySequence.StandardKey.Save, lambda: self.current_page().shortcut("save"))
        add("Ctrl+0", lambda: self.current_page().shortcut("fit"))
        add("Ctrl+1", lambda: self.current_page().shortcut("actual"))
        for i, key in enumerate(STEPS, start=1):
            add(f"Alt+{i}", lambda k=key: self.step.set(k))

    # ------------------------------------------------------------------ 頁面

    def page(self, key: str) -> Page:
        """取得某一步的頁面，第一次用到才建立。"""
        page = self.pages.get(key)
        if page is None:
            if key == "pick":
                from .pages.pick import PickPage as cls
            elif key == "stack":
                from .pages.stack import StackPage as cls
            else:
                from .pages.studio import StudioPage as cls
            page = cls(self)
            page.status_changed.connect(lambda p=page: self._page_status(p))
            page.title_changed.connect(self._update_title)
            page.rail_changed.connect(lambda p=page: self.rail.set_sub(p.key, p.rail_text(), p.progress))
            page.rebuild()
            self.pages[key] = page
            self.stack.addWidget(page)
        return page

    def current_page(self) -> Page:
        return self.page(self.step.get())

    def _show_step(self, fade: bool) -> None:
        key = self.step.get()
        save_settings(step=key)
        page = self.page(key)
        self.stack.setCurrentWidget(page)
        if fade:
            effect = QGraphicsOpacityEffect(page)
            page.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity", page)
            anim.setDuration(160)
            anim.setStartValue(0.35)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.finished.connect(lambda p=page: p.setGraphicsEffect(None))
            anim.start()
        self._update_title()
        self.status_bar.show_page(page)

    def _page_status(self, page: Page) -> None:
        self.rail.set_sub(page.key, page.rail_text(), page.progress)
        if page is self.pages.get(self.step.get()):
            self.status_bar.show_page(page)

    def _update_title(self) -> None:
        page = self.pages.get(self.step.get())
        text = page.doc_title() if page is not None else ""
        self.title.setText(text or tr("no_project"))
        self.title.setStyleSheet(f"color: {D.label if text else D.secondary}; background: transparent;")
        self.title_icon.setVisible(bool(text))

    # ------------------------------------------------------------------ 交棒

    def send_to_stack(self, paths: list[Path]) -> None:
        page = self.page("stack")
        page.add_files(paths)
        self.step.set("stack")
        page.set_status(lambda n=len(paths): tr("sent.stack", n=n))

    def send_to_process(self, recipe: Path) -> None:
        page = self.page("process")
        page.open_paths([recipe])
        self.step.set("process")

    def open_paths(self, paths: list[Path]) -> None:
        """拖進視窗的檔案：交給目前這一步（疊圖頁收到 recipe 是載入疊圖設定，後製頁收到是開它的 master）。"""
        if paths:
            self.current_page().handle_drop(paths)

    def open_startup(self, paths: list[Path]) -> None:
        """命令列（或拖到 exe 上）的檔案：疊圖紀錄交給後製，其他交給上次停留的那一步。"""
        if any(p.name.endswith(".recipe.json") for p in paths):
            self.step.set("process")
        self.open_paths(paths)

    # ------------------------------------------------------------------ 設定

    def save_panel(self, key: str, expanded: bool) -> None:
        self.panel_state[key] = expanded
        save_settings(panel=self.panel_state)

    def _language_clicked(self, lang: str) -> None:
        if lang == get_language():
            return
        busy = [k for k, p in self.pages.items() if p.is_busy()]
        if busy:  # 執行中不換：重建介面會打斷進度顯示
            self.lang_var.set(get_language())
            return
        set_language(lang)
        save_settings(language=lang)
        self.rail.retranslate()
        for page in self.pages.values():
            page.rebuild()
        self._update_title()
        self.status_bar.show_page(self.current_page())

    # ------------------------------------------------------------------ 拖放、關閉

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        self.open_paths(paths)

    def closeEvent(self, e: QCloseEvent) -> None:  # noqa: N802
        busy = [tr(f"step.{k}") for k, p in self.pages.items() if p.is_busy()]
        if busy and not ask_yes_no(self, tr("close.confirm", steps=tr("close.sep").join(busy))):
            e.ignore()
            return
        save_settings(geometry=[self.width(), self.height()], maximized=self.isMaximized())
        for page in self.pages.values():
            page.close_page()
        e.accept()

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        dark_title_bar(self)


__all__ = ["MainWindow", "STEPS", "QAction"]
