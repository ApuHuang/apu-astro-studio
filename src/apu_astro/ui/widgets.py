"""暗房元件的 Qt 版：名稱與用法對應 Pick／Photons／Processing 的 darkroom.py（PanelGroup、ParameterSlider、
ParameterToggle、Segmented、MetricRow、InfoButton），再加上動畫與「點數字直接輸入」。

元件都綁在 Var 上：Var 改了元件跟著變，使用者操作元件就 set Var。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from PySide6.QtCore import (Property, QEasingCurve, QEvent, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSize,
                            Qt, QTimer, QVariantAnimation, Signal)
from PySide6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)

from ..i18n import tr
from ..theme import D, fonts
from .var import Var

QWIDGETSIZE_MAX = 16777215


def button(text: str, kind: str | None = None, tooltip: str | None = None,
           on_click: Callable[[], None] | None = None) -> QPushButton:
    """kind：None（一般）、"prominent"（藍色主要動作，一個畫面最多一個）、"flat"。"""
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if tooltip:
        b.setToolTip(tooltip)
    if on_click is not None:
        b.clicked.connect(lambda _checked=False: on_click())
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def label(text: str = "", color: str = D.label, size: float = 9, bold: bool = False, mono: bool = False,
          wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setFont(fonts().mono(size) if mono else fonts().ui(size, bold))
    lb.setStyleSheet(f"color: {color}; background: transparent;")
    lb.setWordWrap(wrap)
    return lb


def hline(color: str = D.separator) -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {color}; border: none;")
    return f


def vline(color: str = D.separator, height: int | None = None) -> QFrame:
    f = QFrame()
    f.setFixedWidth(1)
    if height:
        f.setFixedHeight(height)
    f.setStyleSheet(f"background: {color}; border: none;")
    return f


class Badge(QLabel):
    """版本標籤：橘色細框，對應 APU Astro 頂部列的版本。"""

    def __init__(self, text: str, color: str = D.beta):
        super().__init__(text)
        self.setFont(fonts().ui(7, bold=True))
        self.setStyleSheet(f"color: {color}; border: 1px solid {color}; border-radius: 4px; padding: 0 4px;"
                           "background: transparent;")
        self.setFixedHeight(16)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


# ---------------------------------------------------------------------- 開關


class Switch(QWidget):
    """開關（對應 APU Astro 的 ParameterToggle）：滑動動畫；停用時畫成暗色、點了沒反應。"""

    def __init__(self, var: Var, parent: QWidget | None = None):
        super().__init__(parent)
        self.var = var
        self.enabled_ = True
        self.forced_off = False
        self._offset = 1.0 if var.get() else 0.0
        self.setFixedSize(30, 17)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        var.subscribe(self._changed, self)

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, v: float) -> None:
        self._offset = v
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _changed(self, value: Any) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if value and not self.forced_off else 0.0)
        self._anim.start()

    def set_enabled(self, enabled: bool, forced_off: bool = False) -> None:
        """forced_off：功能用不了（例如沒有 GPU）時畫成關，但保留使用者原本的設定。"""
        self.enabled_ = enabled
        self.forced_off = forced_off and not enabled
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)
        self._offset = 1.0 if self.var.get() and not self.forced_off else 0.0
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and self.enabled_:
            self.var.set(not bool(self.var.get()))

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        t = self._offset
        on, off = QColor(D.prominent), QColor(D.control_hover)
        if not self.enabled_:
            on, off = QColor("#1d3550"), QColor(D.card_header)
        track = QColor(int(off.red() + (on.red() - off.red()) * t), int(off.green() + (on.green() - off.green()) * t),
                       int(off.blue() + (on.blue() - off.blue()) * t))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        pad = 2.0
        d = h - 2 * pad
        x = pad + (w - 2 * pad - d) * t
        p.setBrush(QColor("white") if self.enabled_ else QColor("#8a8a8a"))
        p.drawEllipse(QRectF(x, pad, d, d))


class ParameterToggle(QWidget):
    """開關列：標題填滿寬度，開關在右邊；點標題也能切換。"""

    def __init__(self, title: str, var: Var, tooltip: str | None = None):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        self.label = label(title)
        self.label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.switch = Switch(var)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.switch)
        if tooltip:
            self.setToolTip(tooltip)
        self.label.mousePressEvent = lambda _e: self.switch.enabled_ and var.set(not bool(var.get()))

    def set_enabled(self, enabled: bool, forced_off: bool = False) -> None:
        self.switch.set_enabled(enabled, forced_off)
        self.label.setStyleSheet(f"color: {D.label if enabled else D.disabled}; background: transparent;")


# ---------------------------------------------------------------------- 分段切換


class Segmented(QWidget):
    """分段切換（對應 APU Astro 頂部列的「繁中｜EN」）：選中的那一格有滑動的底色。"""

    def __init__(self, options: Sequence[tuple[Any, str]], var: Var, stretch: bool = False,
                 tooltips: dict[Any, str] | None = None):
        super().__init__()
        self.options = list(options)
        self.var = var
        self.stretch = stretch
        self.tooltips = tooltips or {}
        self.setFont(fonts().ui(9))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self._hover = -1
        self._pill = QRectF()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._set_pill)
        self.setSizePolicy(QSizePolicy.Policy.Expanding if stretch else QSizePolicy.Policy.Fixed,
                           QSizePolicy.Policy.Fixed)
        var.subscribe(lambda _v: self._move_pill(animate=True), self)

    def sizeHint(self) -> QSize:  # noqa: N802
        fm = QFontMetrics(self.font())
        widths = [fm.horizontalAdvance(t) + 22 for _, t in self.options]
        return QSize(sum(widths) + 4 + 2 * (len(widths) - 1), fm.height() + 10)

    minimumSizeHint = sizeHint

    def _rects(self) -> list[QRectF]:
        fm = QFontMetrics(self.font())
        n = len(self.options)
        inner = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        if self.stretch:
            w = (inner.width() - 2 * (n - 1)) / max(1, n)
            return [QRectF(inner.left() + i * (w + 2), inner.top(), w, inner.height()) for i in range(n)]
        out, x = [], inner.left()
        for _, t in self.options:
            w = fm.horizontalAdvance(t) + 22
            out.append(QRectF(x, inner.top(), w, inner.height()))
            x += w + 2
        return out

    def _index(self) -> int:
        values = [v for v, _ in self.options]
        return values.index(self.var.get()) if self.var.get() in values else -1

    def _set_pill(self, r: QRectF) -> None:
        self._pill = QRectF(r)
        self.update()

    def _move_pill(self, animate: bool) -> None:
        i = self._index()
        rects = self._rects()
        target = rects[i] if 0 <= i < len(rects) else QRectF()
        if animate and not self._pill.isNull() and not target.isNull() and self.isVisible():
            self._anim.stop()
            self._anim.setStartValue(self._pill)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._set_pill(target)

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._move_pill(animate=False)
        super().resizeEvent(e)

    def showEvent(self, e) -> None:  # noqa: N802
        self._move_pill(animate=False)
        super().showEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        hit = next((i for i, r in enumerate(self._rects()) if r.contains(e.position())), -1)
        if hit != self._hover:
            self._hover = hit
            value = self.options[hit][0] if hit >= 0 else None
            self.setToolTip(self.tooltips.get(value, ""))
            self.update()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if not self.isEnabled():
            return
        for (value, _), r in zip(self.options, self._rects()):
            if r.contains(e.position()):
                self.var.set(value)
                return

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(D.segment_bg))
        p.drawRoundedRect(QRectF(self.rect()), 7, 7)
        rects = self._rects()
        if 0 <= self._hover < len(rects) and self._hover != self._index():
            p.setBrush(QColor("#333336"))
            p.drawRoundedRect(rects[self._hover], 5, 5)
        if not self._pill.isNull():
            p.setBrush(QColor(D.segment_on))
            p.drawRoundedRect(self._pill, 5, 5)
        current = self._index()
        for i, ((_, text), r) in enumerate(zip(self.options, rects)):
            color = "#ffffff" if i == current else ("#aeaeb2" if self.isEnabled() else D.disabled)
            p.setPen(QColor(color))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)


# ---------------------------------------------------------------------- 滑桿


_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


class _ValueLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ParameterSlider(QWidget):
    """滑桿列：標題、目前的值（等寬數字，點一下可以直接輸入）、滑桿。

    fmt：值 → 顯示文字。step：滑桿的最小單位。parse_scale：輸入的數字要乘多少才是 Var 的值（例如顯示 % 時是 1）。
    """

    def __init__(self, title: str, var: Var, lo: float, hi: float, fmt: Callable[[float], str],
                 step: float = 1.0, tooltip: str | None = None, editable: bool = True):
        super().__init__()
        self.var, self.lo, self.hi, self.fmt, self.step = var, lo, hi, fmt, step
        self.n = max(1, round((hi - lo) / step))
        self._editing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.title = label(title, D.secondary)
        self.value = _ValueLabel()
        self.value.setFont(fonts().mono(9))
        self.value.setStyleSheet(f"color: {D.label}; background: transparent;")
        if editable:
            self.value.setCursor(Qt.CursorShape.IBeamCursor)
            self.value.setToolTip(tr("slider.edit_hint"))
            self.value.clicked.connect(self._start_edit)
        self.edit = QLineEdit()
        self.edit.setFont(fonts().mono(9))
        self.edit.setFixedWidth(72)
        self.edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit.hide()
        self.edit.returnPressed.connect(self._commit_edit)
        self.edit.editingFinished.connect(self._commit_edit)
        self.edit.installEventFilter(self)
        head.addWidget(self.title, 1)
        head.addWidget(self.value)
        head.addWidget(self.edit)
        lay.addLayout(head)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.n)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.valueChanged.connect(self._moved)
        lay.addWidget(self.slider)
        if tooltip:
            self.setToolTip(tooltip)
        var.subscribe(lambda _v: self.refresh(), self)
        self.refresh()

    def _to_value(self, i: int) -> float:
        v = self.lo + i * self.step
        return round(v, 6) if self.step != int(self.step) else float(round(v))

    def _moved(self, i: int) -> None:
        v = self._to_value(i)
        cur = self.var.get()
        if isinstance(cur, int) and not isinstance(cur, bool):
            v = int(round(v))
        self.var.set(v)

    def refresh(self) -> None:
        try:
            v = float(self.var.get())
        except (TypeError, ValueError):
            self.value.setText("—")
            return
        self.value.setText(self.fmt(v))
        i = round((min(max(v, self.lo), self.hi) - self.lo) / self.step)
        if self.slider.value() != i:
            self.slider.blockSignals(True)
            self.slider.setValue(i)
            self.slider.blockSignals(False)

    def set_enabled(self, enabled: bool) -> None:
        self.slider.setEnabled(enabled)
        self.value.setStyleSheet(f"color: {D.label if enabled else D.disabled}; background: transparent;")

    # 點數字直接輸入（APU Pick 0.6 規劃的第 1 項）：超出範圍夾到上下限
    def _start_edit(self) -> None:
        if not self.slider.isEnabled():
            return
        self._editing = True
        self.edit.setText(f"{float(self.var.get()):g}")
        self.value.hide()
        self.edit.show()
        self.edit.setFocus()
        self.edit.selectAll()

    def _commit_edit(self) -> None:
        if not self._editing:
            return
        self._editing = False
        m = _NUMBER.search(self.edit.text())
        self.edit.hide()
        self.value.show()
        if m:
            v = float(m.group().replace(",", "."))
            v = min(max(v, self.lo), self.hi)
            v = self.lo + round((v - self.lo) / self.step) * self.step
            cur = self.var.get()
            self.var.set(int(round(v)) if isinstance(cur, int) and not isinstance(cur, bool) else round(v, 6))
        self.refresh()

    def eventFilter(self, obj, e) -> bool:  # noqa: N802
        if obj is self.edit and e.type() == QEvent.Type.KeyPress and e.key() == Qt.Key.Key_Escape:
            self._editing = False
            self.edit.hide()
            self.value.show()
            return True
        return super().eventFilter(obj, e)


class MetricRow(QWidget):
    """唯讀的數值列（對應 APU Astro 的 MetricRow）。"""

    def __init__(self, title: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.addWidget(label(title, D.secondary), 1)
        self.value = label("—", mono=True)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.value)

    def set(self, text: str, color: str = D.label) -> None:
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {color}; background: transparent;")


# ---------------------------------------------------------------------- 說明氣泡


class Popover(QFrame):
    """點 ⓘ 才出現的說明（對應 APU Astro 的 InfoButton popover）；點別的地方或按 Esc 就關。"""

    def __init__(self, text: str, parent: QWidget):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lb = label(text, wrap=True)
        lb.setFixedWidth(360)
        lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(lb)

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(D.separator_strong), 1))
        p.setBrush(QColor(D.card_header))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)

    def show_beside(self, owner: QWidget) -> None:
        """面板在右邊，說明往左邊開。"""
        self.adjustSize()
        top_left = owner.mapToGlobal(QPoint(0, 0))
        x = top_left.x() - self.width() - 8
        y = top_left.y() - 6
        screen = owner.screen().availableGeometry()
        x = max(screen.left() + 4, x)
        y = min(max(screen.top() + 4, y), screen.bottom() - self.height() - 4)
        self.move(x, y)
        self.show()


class InfoButton(QLabel):
    """分組標題上的 ⓘ。"""

    def __init__(self, text: str):
        super().__init__("ⓘ")
        self.text_ = text
        self.setFont(fonts().ui(10))
        self.setStyleSheet(f"color: {D.secondary}; background: transparent;")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("info.tooltip"))

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            Popover(self.text_, self.window()).show_beside(self)
            e.accept()

    def enterEvent(self, _e) -> None:  # noqa: N802
        self.setStyleSheet(f"color: {D.label}; background: transparent;")

    def leaveEvent(self, _e) -> None:  # noqa: N802
        self.setStyleSheet(f"color: {D.secondary}; background: transparent;")


# ---------------------------------------------------------------------- 面板


class _Header(QWidget):
    clicked = Signal()

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class PanelGroup(QFrame):
    """右側面板裡可收合的一組（對應 APU Astro 的 PanelGroup）：圓角卡片，收合有動畫，狀態會記住。

    內容加在 self.body（QVBoxLayout 是 self.lay）。state：{key: 展開與否}；on_toggle(key, expanded) 存檔。
    """

    def __init__(self, key: str, title: str, info: str | None = None, state: dict[str, bool] | None = None,
                 on_toggle: Callable[[str, bool], None] | None = None):
        super().__init__()
        self.key = key
        self.on_toggle = on_toggle
        self.setObjectName("PanelGroup")
        self.setStyleSheet(f"QFrame#PanelGroup {{ background: {D.card}; border-radius: 10px; }}")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = _Header()
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.setFixedHeight(34)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 0, 10, 0)
        hl.setSpacing(6)
        self.chevron = label("", D.secondary, 8)
        self.chevron.setFixedWidth(12)
        hl.addWidget(self.chevron)
        hl.addWidget(label(title, D.label, 9, bold=True), 1)
        if info:
            hl.addWidget(InfoButton(info))
        header.clicked.connect(self.toggle)
        outer.addWidget(header)
        self.body = QWidget()
        self.body.setStyleSheet("background: transparent;")
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(14, 2, 14, 12)
        self.lay.setSpacing(4)
        outer.addWidget(self.body)
        self.expanded = (state or {}).get(key, True)
        self._anim = QPropertyAnimation(self.body, b"maximumHeight", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._anim_done)
        self._apply(animate=False)

    def _apply(self, animate: bool) -> None:
        self.chevron.setText("▾" if self.expanded else "▸")
        if not animate:
            self.body.setVisible(self.expanded)
            self.body.setMaximumHeight(QWIDGETSIZE_MAX)
            return
        self._anim.stop()
        full = self.body.sizeHint().height()
        if self.expanded:
            self.body.setMaximumHeight(0)
            self.body.setVisible(True)
            self._anim.setStartValue(0)
            self._anim.setEndValue(full)
        else:
            self._anim.setStartValue(self.body.height())
            self._anim.setEndValue(0)
        self._anim.start()

    def _anim_done(self) -> None:
        if self.expanded:
            self.body.setMaximumHeight(QWIDGETSIZE_MAX)
        else:
            self.body.setVisible(False)
            self.body.setMaximumHeight(QWIDGETSIZE_MAX)

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self._apply(animate=True)
        if self.on_toggle is not None:
            self.on_toggle(self.key, self.expanded)

    def add(self, widget: QWidget, spacing_before: int = 0) -> QWidget:
        if spacing_before:
            self.lay.addSpacing(spacing_before)
        self.lay.addWidget(widget)
        return widget

    def add_row(self, *widgets: QWidget, stretch_last: bool = False) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 0)
        row.setSpacing(6)
        for i, w in enumerate(widgets):
            row.addWidget(w, 1 if (stretch_last and i == len(widgets) - 1) else 0)
        if not stretch_last:
            row.addStretch(1)
        self.lay.addLayout(row)
        return row


class Inspector(QScrollArea):
    """右側面板：一疊 PanelGroup，放不下就捲動。"""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(D.panel_width)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"QScrollArea {{ background: {D.chrome}; border: none; border-left: 1px solid {D.separator}; }}")
        inner = QWidget()
        inner.setStyleSheet(f"background: {D.chrome};")
        self.lay = QVBoxLayout(inner)
        self.lay.setContentsMargins(10, 10, 10, 10)
        self.lay.setSpacing(8)
        self.lay.addStretch(1)
        self.setWidget(inner)

    def add(self, group: QWidget) -> QWidget:
        self.lay.insertWidget(self.lay.count() - 1, group)
        return group


# ---------------------------------------------------------------------- 歡迎畫面


class Hero(QWidget):
    """還沒有內容時的歡迎畫面：大標題、副標、下一步提示。"""

    def __init__(self, title: str, subtitle: str, hint: str = "", pixmap=None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addStretch(5)
        if pixmap is not None:
            pic = QLabel()
            pic.setPixmap(pixmap)
            pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(pic)
            lay.addSpacing(10)
        t = label(title, D.label, 24, bold=True)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        s = label(subtitle, D.secondary, 11, wrap=True)
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(s)
        lay.addSpacing(20)
        self.hint = label(hint, D.accent, 9.5, wrap=True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.hint)
        lay.addStretch(6)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)


# ---------------------------------------------------------------------- 對話框


def _box(parent: QWidget | None, icon: QMessageBox.Icon, text: str, title: str) -> QMessageBox:
    box = QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    return box


def ask_yes_no(parent: QWidget | None, text: str, title: str = "APU Astro", yes: str | None = None,
               no: str | None = None) -> bool:
    box = _box(parent, QMessageBox.Icon.Question, text, title)
    y = box.addButton(yes or tr("btn.yes"), QMessageBox.ButtonRole.YesRole)
    y.setProperty("kind", "prominent")
    box.addButton(no or tr("btn.no"), QMessageBox.ButtonRole.NoRole)
    box.setDefaultButton(y)
    box.exec()
    return box.clickedButton() is y


def ask_three(parent: QWidget | None, text: str, options: Sequence[str], title: str = "APU Astro") -> int | None:
    """三選一（例如載入 recipe：完全重現／只套用設定／取消）；回傳選項的位置，取消或關掉是 None。"""
    box = _box(parent, QMessageBox.Icon.Question, text, title)
    buttons = []
    for i, text_ in enumerate(options):
        role = QMessageBox.ButtonRole.RejectRole if i == len(options) - 1 else QMessageBox.ButtonRole.AcceptRole
        b = box.addButton(text_, role)
        if i == 0:
            b.setProperty("kind", "prominent")
        buttons.append(b)
    box.exec()
    clicked = box.clickedButton()
    i = buttons.index(clicked) if clicked in buttons else len(options) - 1
    return None if i == len(options) - 1 else i


def info(parent: QWidget | None, text: str, title: str = "APU Astro") -> None:
    _box(parent, QMessageBox.Icon.Information, text, title).exec()


def warn(parent: QWidget | None, text: str, title: str = "APU Astro") -> None:
    _box(parent, QMessageBox.Icon.Warning, text, title).exec()


def error(parent: QWidget | None, text: str, title: str = "APU Astro") -> None:
    _box(parent, QMessageBox.Icon.Critical, text, title).exec()


def short_path(path: str, limit: int = 44) -> str:
    if len(path) <= limit:
        return path
    head = limit // 3
    return f"{path[:head]}…{path[-(limit - head - 1):]}"


def delayed(parent: QWidget, ms: int, fn: Callable[[], None]) -> QTimer:
    """單次、可重設的延遲呼叫（停止操作 ms 毫秒後才執行；跟 Pick／Processing 的 0.3 秒一樣）。"""
    t = QTimer(parent)
    t.setSingleShot(True)
    t.setInterval(ms)
    t.timeout.connect(fn)
    return t
