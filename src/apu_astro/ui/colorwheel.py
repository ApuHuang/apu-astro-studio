"""色輪（調色用）：拖曳圓點選色偏方向（角度＝OKLCh 色相）與量（離中心的距離）；雙擊歸零。下方一條明暗滑桿。

色環上的顏色用 OKLab 算（跟調色工具同一個色相定義），不是 HSV，所以拖到哪個顏色就會偏向那個顏色。
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QConicalGradient, QMouseEvent, QPainter, QPaintEvent, QPen, QRadialGradient
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ..studio import ops
from ..theme import D
from .widgets import label


def _ring_colors(n: int = 36) -> list[QColor]:
    out = []
    for i in range(n + 1):
        h = math.radians(i * 360 / n)
        lab = np.array([0.72, 0.13 * math.cos(h), 0.13 * math.sin(h)], np.float32).reshape(3, 1, 1)
        r, g, b = (np.clip(ops.oklab_to_rgb(lab)[:, 0, 0], 0, 1) * 255).astype(int)
        out.append(QColor(int(r), int(g), int(b)))
    return out


_COLORS: list[QColor] | None = None


class Wheel(QWidget):
    changed = Signal(float, float)      # (色相角度, 量 0–1)

    def __init__(self, hue: float = 0.0, amount: float = 0.0):
        super().__init__()
        self.hue, self.amount = hue, amount
        self.setFixedSize(84, 84)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._drag = False

    def set_value(self, hue: float, amount: float) -> None:
        self.hue, self.amount = hue, amount
        self.update()

    def _center(self) -> tuple[QPointF, float]:
        r = min(self.width(), self.height()) / 2 - 4
        return QPointF(self.width() / 2, self.height() / 2), r

    def _from_pos(self, pos: QPointF) -> None:
        c, r = self._center()
        dx, dy = pos.x() - c.x(), c.y() - pos.y()
        amount = min(1.0, math.hypot(dx, dy) / (r * 0.85))
        hue = math.degrees(math.atan2(dy, dx)) % 360
        self.set_value(hue, amount)
        self.changed.emit(hue, amount)

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._from_pos(e.position())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._drag:
            self._from_pos(e.position())

    def mouseReleaseEvent(self, _e: QMouseEvent) -> None:  # noqa: N802
        self._drag = False

    def mouseDoubleClickEvent(self, _e: QMouseEvent) -> None:  # noqa: N802
        self.set_value(self.hue, 0.0)
        self.changed.emit(self.hue, 0.0)

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        global _COLORS
        if _COLORS is None:
            _COLORS = _ring_colors()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, r = self._center()
        cone = QConicalGradient(c, 0)
        for i, col in enumerate(_COLORS):
            cone.setColorAt(i / (len(_COLORS) - 1), col)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(cone))
        p.drawEllipse(c, r, r)
        fade = QRadialGradient(c, r)
        fade.setColorAt(0.0, QColor(58, 58, 60, 255))
        fade.setColorAt(0.75, QColor(58, 58, 60, 120))
        fade.setColorAt(1.0, QColor(58, 58, 60, 0))
        p.setBrush(QBrush(fade))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(D.separator_strong), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
        p.drawLine(QPointF(c.x() - 4, c.y()), QPointF(c.x() + 4, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - 4), QPointF(c.x(), c.y() + 4))
        a = math.radians(self.hue)
        d = self.amount * r * 0.85
        dot = QPointF(c.x() + d * math.cos(a), c.y() - d * math.sin(a))
        p.setPen(QPen(QColor(0, 0, 0, 180), 3))
        p.drawEllipse(dot, 5, 5)
        p.setPen(QPen(QColor("white"), 1.6))
        p.drawEllipse(dot, 5, 5)


class WheelBox(QWidget):
    """一個色輪＋標題（明暗滑桿由呼叫端另外放）。"""

    def __init__(self, title: str, hue: float, amount: float):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        t = label(title, D.secondary, 8.5)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        self.wheel = Wheel(hue, amount)
        lay.addWidget(self.wheel, 0, Qt.AlignmentFlag.AlignHCenter)


__all__ = ["Wheel", "WheelBox", "QRectF"]
