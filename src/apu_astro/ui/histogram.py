"""直方圖：目前畫面的 RGB（或亮度），平方根縮放，暗部與亮部截斷時在兩端標紅。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from ..theme import D

COLORS = ("#ff6961", "#7ee081", "#6ab0ff")


class Histogram(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(86)
        self.hist: list[np.ndarray] = []
        self.clip = (0.0, 0.0)

    def set_image(self, display: np.ndarray | None) -> None:
        self.hist = []
        if display is not None:
            p = display if display.ndim == 3 else display[None]
            sub = p[:, :: max(1, p.shape[1] // 400), :: max(1, p.shape[2] // 400)]
            bins = np.linspace(0, 1, 129)
            self.hist = [np.histogram(np.clip(c, 0, 1), bins)[0].astype(float) for c in sub]
            n = sub[0].size
            self.clip = (float((sub <= 0.0005).all(axis=0).sum()) / n, float((sub >= 0.9995).any(axis=0).sum()) / n)
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#141416"))
        p.drawRoundedRect(r, 8, 8)
        if not self.hist:
            return
        top = max(float(np.sqrt(h).max()) for h in self.hist) or 1.0
        inner = r.adjusted(6, 6, -6, -6)
        colors = COLORS if len(self.hist) >= 3 else ("#d0d0d0",)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for h, col in zip(self.hist, colors):
            v = np.sqrt(h) / top
            path = QPainterPath(QPointF(inner.left(), inner.bottom()))
            n = len(v)
            for i, y in enumerate(v):
                path.lineTo(inner.left() + (i + 0.5) / n * inner.width(), inner.bottom() - y * inner.height())
            path.lineTo(inner.right(), inner.bottom())
            path.closeSubpath()
            c = QColor(col)
            c.setAlpha(110)
            p.setBrush(c)
            p.drawPath(path)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setPen(QPen(QColor(D.reject), 3))
        if self.clip[0] > 0.002:
            p.drawLine(QPointF(inner.left(), inner.top()), QPointF(inner.left(), inner.bottom()))
        if self.clip[1] > 0.0005:
            p.drawLine(QPointF(inner.right(), inner.top()), QPointF(inner.right(), inner.bottom()))
