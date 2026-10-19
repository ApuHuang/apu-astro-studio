"""曲線編輯器（APU Processing 的 CurveEditor 的 Qt 版）：主控／紅／綠／藍四條曲線，背後畫直方圖。

操作與 Processing、Swift 版相同：拖曳控制點；雙擊空白處新增、雙擊控制點刪除（頭尾兩點不能刪）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from apu_processing.display import IDENTITY, ToneCurves, curve_values
from apu_processing.i18n import tr as ptr
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .var import Var
from .widgets import Segmented

CHANNELS = ("master", "red", "green", "blue")
CHANNEL_COLOR = {"master": "#ebebeb", "red": "#ff6961", "green": "#7ee081", "blue": "#6ab0ff"}
HIST_COLOR = {"master": "#4a4a4e", "red": "#5a2e2e", "green": "#2e5a33", "blue": "#2e405a"}


class _Canvas(QWidget):
    def __init__(self, editor: CurveEditor):
        super().__init__()
        self.editor = editor
        self.setFixedHeight(268)
        self.setMouseTracking(True)
        self._drag: int | None = None
        self._hover: int | None = None
        self.pad = 7.0

    def _span(self) -> float:
        return min(self.width(), self.height()) - 2 * self.pad

    def _origin(self) -> QPointF:
        span = self._span()
        return QPointF((self.width() - span) / 2, (self.height() - span) / 2)

    def to_screen(self, x: float, y: float) -> QPointF:
        o, span = self._origin(), self._span()
        return QPointF(o.x() + x * span, o.y() + (1 - y) * span)

    def to_value(self, p: QPointF) -> tuple[float, float]:
        o, span = self._origin(), self._span()
        return min(1.0, max(0.0, (p.x() - o.x()) / span)), min(1.0, max(0.0, 1 - (p.y() - o.y()) / span))

    def hit(self, p: QPointF) -> int | None:
        best, where = None, 10.0
        for i, (x, y) in enumerate(self.editor.points()):
            s = self.to_screen(x, y)
            d = ((s.x() - p.x()) ** 2 + (s.y() - p.y()) ** 2) ** 0.5
            if d < where:
                best, where = i, d
        return best

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = self.hit(e.position())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None:
            h = self.hit(e.position())
            if h != self._hover:
                self._hover = h
                self.setCursor(Qt.CursorShape.SizeAllCursor if h is not None else Qt.CursorShape.CrossCursor)
                self.update()
            return
        pts = [list(p) for p in self.editor.points()]
        order = sorted(range(len(pts)), key=lambda i: pts[i][0])
        rank = order.index(self._drag)
        x, y = self.to_value(e.position())
        lo = pts[order[rank - 1]][0] + 0.01 if rank > 0 else 0.0
        hi = pts[order[rank + 1]][0] - 0.01 if rank < len(order) - 1 else 1.0
        if rank == 0:
            x = 0.0 if pts[self._drag][0] == 0.0 else min(x, hi)
        elif rank == len(order) - 1:
            x = 1.0 if pts[self._drag][0] == 1.0 else max(x, lo)
        else:
            x = min(max(x, lo), hi)
        pts[self._drag] = [x, y]
        self.editor.set_points(pts)

    def mouseReleaseEvent(self, _e: QMouseEvent) -> None:  # noqa: N802
        self._drag = None

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pts = [list(p) for p in self.editor.points()]
        hit = self.hit(e.position())
        if hit is not None:
            order = sorted(range(len(pts)), key=lambda i: pts[i][0])
            if hit not in (order[0], order[-1]):
                del pts[hit]
                self.editor.set_points(pts)
            return
        x, _ = self.to_value(e.position())
        y = float(curve_values(pts, np.array([x]))[0])
        pts.append([x, y])
        self.editor.set_points(sorted(pts))

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        o, span = self._origin(), self._span()
        box = QRectF(o.x(), o.y(), span, span)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#141416"))
        p.drawRoundedRect(box.adjusted(-self.pad + 1, -self.pad + 1, self.pad - 1, self.pad - 1), 8, 8)
        ch = self.editor.channel.get()
        hist = self.editor.hist.get(ch)
        if hist is not None and hist.max() > 0:
            h = np.sqrt(hist / hist.max())
            path = QPainterPath(QPointF(box.left(), box.bottom()))
            n = len(h)
            for i, v in enumerate(h):
                path.lineTo(box.left() + (i + 0.5) / n * span, box.bottom() - v * span)
            path.lineTo(box.right(), box.bottom())
            path.closeSubpath()
            p.setBrush(QColor(HIST_COLOR[ch]))
            p.drawPath(path)
        p.setPen(QPen(QColor("#2c2c2e"), 1))
        for k in (0.25, 0.5, 0.75):
            p.drawLine(QPointF(box.left() + k * span, box.top()), QPointF(box.left() + k * span, box.bottom()))
            p.drawLine(QPointF(box.left(), box.top() + k * span), QPointF(box.right(), box.top() + k * span))
        p.setPen(QPen(QColor("#3a3a3c"), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(box.left(), box.bottom()), QPointF(box.right(), box.top()))
        xs = np.linspace(0, 1, 257)
        ys = curve_values(self.editor.points(), xs)
        path = QPainterPath(self.to_screen(float(xs[0]), float(ys[0])))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self.to_screen(float(x), float(y)))
        color = QColor(CHANNEL_COLOR[ch])
        p.setPen(QPen(color, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        for i, (x, y) in enumerate(self.editor.points()):
            s = self.to_screen(x, y)
            p.setBrush(color if i in (self._hover, self._drag) else QColor("#141416"))
            p.setPen(QPen(color, 2))
            p.drawEllipse(s, 4.5, 4.5)


class CurveEditor(QWidget):
    def __init__(self, on_change: Callable[[ToneCurves], None]):
        super().__init__()
        self.on_change = on_change
        self.curves = ToneCurves()
        self.channel = Var("master")
        self.hist: dict[str, np.ndarray] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(Segmented([(k, ptr(f"gui.curve.{k}")) for k in CHANNELS], self.channel, stretch=True))
        self.canvas = _Canvas(self)
        lay.addWidget(self.canvas)
        self.channel.subscribe(lambda _v: self.canvas.update(), self)

    def points(self) -> list[list[float]]:
        return getattr(self.curves, self.channel.get())

    def set_points(self, pts: list[list[float]]) -> None:
        setattr(self.curves, self.channel.get(), pts)
        self.on_change(self.curves)
        self.canvas.update()

    def set_curves(self, curves: ToneCurves) -> None:
        self.curves = curves
        self.canvas.update()

    def set_histogram(self, display: np.ndarray | None) -> None:
        """display：(C, H, W) 的顯示影像（預覽縮圖就夠）。"""
        self.hist = {}
        if display is not None:
            planes = display if display.ndim == 3 else display[None]
            sub = planes[:, ::2, ::2]
            bins = np.linspace(0, 1, 129)
            self.hist["master"] = np.histogram(sub.mean(axis=0), bins)[0].astype(float)
            for i, k in enumerate(("red", "green", "blue")):
                self.hist[k] = np.histogram(sub[min(i, len(sub) - 1)], bins)[0].astype(float)
        self.canvas.update()

    def reset_channel(self) -> None:
        self.set_points([list(p) for p in IDENTITY])
