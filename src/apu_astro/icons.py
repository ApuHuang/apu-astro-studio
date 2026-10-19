"""介面用的線條圖示（SVG），畫成指定顏色。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_PATHS = {
    # 挑片：漏斗
    "pick": '<path d="M4 4h16v2.2a2 2 0 0 1-.6 1.4L15 12v7l-6 2v-8.5L4.5 7.6A2 2 0 0 1 4 6.2z"/>',
    # 疊圖：三層
    "stack": '<path d="M12 4 4 8l8 4 8-4z"/><path d="m4 12 8 4 8-4"/><path d="m4 16 8 4 8-4"/>',
    # 後製：魔杖
    "process": '<path d="M6 21 21 6l-3-3L3 18z"/><path d="m15 6 3 3"/>'
               '<path d="M9 3a2 2 0 0 0 2 2 2 2 0 0 0-2 2 2 2 0 0 0-2-2 2 2 0 0 0 2-2"/>'
               '<path d="M19 13a2 2 0 0 0 2 2 2 2 0 0 0-2 2 2 2 0 0 0-2-2 2 2 0 0 0 2-2"/>',
    "folder": '<path d="M5 4h4l3 3h7a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m13 18 6-6"/><path d="m13 6 6 6"/>',
    "check": '<path d="m5 12 5 5L20 7"/>',
    # 後製工具
    "source": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 15 5-5 4 4 3-3 6 6"/><circle cx="16" cy="9" r="1.2"/>',
    "gradient": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 15c4-1 6-5 9-6s6 0 9-2"/>',
    "color": '<circle cx="12" cy="12" r="8"/><path d="M12 4v16"/><path d="M12 4a8 8 0 0 1 0 16"/>',
    "deconv": '<circle cx="12" cy="12" r="2"/><circle cx="12" cy="12" r="6" stroke-dasharray="2 2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "stars": '<path d="M12 3l2 5 5 .5-3.8 3.4 1.2 5.1L12 14.3 7.6 17l1.2-5.1L5 8.5l5-.5z"/>',
    "denoise": '<path d="M4 12c2-3 4-3 6 0s4 3 6 0 3-2 4-1"/><path d="M4 7h.01M8 5h.01M15 6h.01M19 5h.01M5 18h.01M10 19h.01M17 18h.01"/>',
    "stretch": '<path d="M4 20c4 0 6-14 16-16"/><path d="M4 20h16M4 20V4"/>',
    "palette": '<path d="M12 3a9 9 0 1 0 0 18c1.5 0 2-1 2-2s-.5-1.5-.5-2.5S14.5 15 16 15h2a3 3 0 0 0 3-3c0-5-4-9-9-9z"/><circle cx="7.5" cy="11" r="1"/><circle cx="10.5" cy="7" r="1"/><circle cx="15" cy="7.5" r="1"/>',
    "sharpen": '<path d="M12 3l9 16H3z"/><path d="M12 9v4"/>',
    "curves": '<path d="M4 20V4h16"/><path d="M4 20C10 20 12 4 20 4"/>',
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "up": '<path d="m6 15 6-6 6 6"/>',
    "down": '<path d="m6 9 6 6 6-6"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
    "brush": '<path d="M18 3l3 3-9 9-3-3z"/><path d="M9 12c-3 0-4 2-4 4s-1 3-2 3c2 1 6 1 8-1 1-1 1-3 1-3"/>',
    "dropper": '<path d="M14 4l6 6M11 7l6 6M17 9 8 18H5v-3l9-9"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
}


def _svg(name: str, color: str, stroke: float) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{_PATHS[name]}</svg>').encode()


def pixmap(name: str, color: str, size: int = 22, stroke: float = 1.7, dpr: float = 2.0) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(_svg(name, color, stroke)))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, size * dpr, size * dpr))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, color: str, size: int = 18) -> QIcon:
    ic = QIcon()
    ic.addPixmap(pixmap(name, color, size))
    return ic


def app_icon() -> QIcon:
    """視窗圖示（tools/make_icon.py 產生）。"""
    ico = Path(__file__).parent / "assets" / "app.ico"
    if ico.is_file():
        return QIcon(str(ico))
    ic = QIcon()
    for s in (16, 32, 64, 256):
        ic.addPixmap(pixmap("process", "#5cadff", s, dpr=1.0))
    return ic


__all__ = ["pixmap", "icon", "app_icon", "QSize"]
