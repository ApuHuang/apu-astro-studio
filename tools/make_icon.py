"""產生整合版的圖示：src/apu_astro/assets/app.ico 與 icon_256.png。

深色圓角方塊，上面三顆依序變亮的星（挑片 → 疊圖 → 後製），最亮那顆帶十字芒。
"""

from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QRadialGradient  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "src" / "apu_astro" / "assets"


def star(p: QPainter, c: QPointF, r: float, color: QColor, spikes: bool) -> None:
    glow = QRadialGradient(c, r * 3.2)
    g = QColor(color)
    g.setAlpha(110)
    glow.setColorAt(0, g)
    glow.setColorAt(1, QColor(0, 0, 0, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(glow))
    p.drawEllipse(c, r * 3.2, r * 3.2)
    if spikes:
        pen = QPen(color, r * 0.28, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        L = r * 3.6
        p.drawLine(QPointF(c.x() - L, c.y()), QPointF(c.x() + L, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - L), QPointF(c.x(), c.y() + L))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("white"))
    p.drawEllipse(c, r * 0.75, r * 0.75)
    p.setBrush(color)
    p.drawEllipse(c, r * 0.5, r * 0.5)


def render(size: int = 256) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 256
    box = QRectF(12 * s, 12 * s, 232 * s, 232 * s)
    path = QPainterPath()
    path.addRoundedRect(box, 52 * s, 52 * s)
    bg = QRadialGradient(QPointF(170 * s, 80 * s), 220 * s)
    bg.setColorAt(0, QColor("#1d2a3a"))
    bg.setColorAt(1, QColor("#0b0b0d"))
    p.fillPath(path, QBrush(bg))
    p.setPen(QPen(QColor("#2c2c2e"), 2 * s))
    p.drawPath(path)
    # 一條弧線把三顆星串起來（三個步驟）
    arc = QPainterPath(QPointF(58 * s, 182 * s))
    arc.cubicTo(QPointF(92 * s, 120 * s), QPointF(140 * s, 96 * s), QPointF(186 * s, 74 * s))
    p.setPen(QPen(QColor(92, 173, 255, 90), 5 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(arc)
    star(p, QPointF(58 * s, 182 * s), 9 * s, QColor("#7f8ea3"), False)
    star(p, QPointF(116 * s, 120 * s), 12 * s, QColor("#5cadff"), False)
    star(p, QPointF(186 * s, 74 * s), 16 * s, QColor("#ffa842"), True)
    p.end()
    return img


def to_pil(img: QImage) -> Image.Image:
    buf = QBuffer()
    buf.open(QBuffer.OpenModeFlag.ReadWrite)
    img.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    ASSETS.mkdir(parents=True, exist_ok=True)
    big = to_pil(render(256))
    big.save(ASSETS / "icon_256.png")
    big.save(ASSETS / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("ok", ASSETS / "app.ico", math.pi > 3 and app is not None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
