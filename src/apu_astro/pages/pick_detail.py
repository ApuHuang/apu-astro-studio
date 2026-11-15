"""挑片清單右邊的「資訊」「直方圖」「像場」三頁（APU Pick 0.6～0.9 的功能）。

文字內容（拍攝資訊、分數拆解、直方圖的數字、傾斜與場曲）直接用 Pick 的 `detail` 模組組出來，跟獨立版一樣；
圖是這裡用 Qt 畫的：Pick 用 matplotlib 畫，整合版不包 matplotlib（省 30 MB，介面也跟其他頁一致）。
畫法照 Pick 的 `draw_histogram`、`draw_field`。
"""

from __future__ import annotations

import html
import math
import sys
import types

import numpy as np
from astro_light_selector.i18n import tr as ptr
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ..theme import D, fonts

CHANNEL_COLORS = ("#ff6b6b", "#5cd65c", "#5cadff")
_detail = None


def pick_detail():
    """Pick 的 detail 模組。它在模組開頭 import matplotlib（畫圖用），整合版打包時沒有 matplotlib：
    沒有的時候放兩個空殼讓它載得起來，只用裡面不畫圖的函式（info_lines、histogram_stats、field_lines、quad_points）。"""
    global _detail
    if _detail is None:
        try:
            from astro_light_selector import detail
        except ImportError:
            fakes = []
            for name, attrs in (("matplotlib", {}), ("matplotlib.figure", {"Figure": object}),
                                ("astro_light_selector.plot", {"DARK": {}})):
                if name not in sys.modules:
                    mod = types.ModuleType(name)
                    mod.__dict__.update(attrs)
                    sys.modules[name] = mod
                    fakes.append(name)
            sys.modules.pop("astro_light_selector.detail", None)
            try:
                from astro_light_selector import detail
            finally:
                # 空殼用完馬上拿掉：留著的話別的套件 import matplotlib 會成功、以為有裝
                # （photutils 找星時讀 matplotlib.__version__ → AttributeError，0.1.7 打包的冒煙測試抓到）
                for name in fakes:
                    sys.modules.pop(name, None)
        _detail = detail
    return _detail


def lines_html(lines: list[tuple[str, str]], mono: bool = False) -> str:
    """Pick 的 (文字, 標籤) 清單 → HTML。文字裡 Tab 分欄、換行分列；標籤 head 粗體、dim 次要色、bad 紅色。"""
    color = {"head": D.label, "dim": D.secondary, "bad": D.reject, "": D.label}
    rows = []
    for text, tag in lines:
        for part in text.split("\n"):
            if not part.strip():
                continue
            cells = part.split("\t")
            c = color.get(tag, D.label)
            if tag == "head":
                rows.append(f'<tr><td colspan="6" style="color:{c}; font-weight:600; padding-top:9px">'
                            f"{html.escape(part)}</td></tr>")
            elif len(cells) == 1:
                rows.append(f'<tr><td colspan="6" style="color:{c}; padding-top:2px">{html.escape(part)}</td></tr>')
            else:
                first = (f'<td style="color:{D.secondary if not mono else c}; padding-right:14px; white-space:nowrap">'
                         f"{html.escape(cells[0])}</td>")
                # 標籤＋值的兩欄：值那一欄拿走剩下的寬度（不然長的值被擠成很窄的一條、斷成好幾行）
                wide = ' width="100%"' if not mono and len(cells) == 2 else ""
                rest = "".join(f'<td{wide} style="color:{c}; padding-right:14px" align="{"right" if mono else "left"}">'
                               f"{html.escape(x)}</td>" for x in cells[1:])
                rows.append(f"<tr>{first}{rest}</tr>")
    return f'<table cellspacing="0" cellpadding="1"{"" if mono else " width=100%"}>{"".join(rows)}</table>'


class _Plot(QWidget):
    """共用：深色底、沒有資料時中間一行字。"""

    def __init__(self, min_height: int):
        super().__init__()
        self.message = ""
        self.setMinimumHeight(min_height)

    def _begin(self) -> QPainter:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(D.list_bg))
        return p

    def _say(self, p: QPainter, text: str) -> None:
        p.setPen(QColor(D.secondary))
        p.setFont(fonts().ui(8.5))
        p.drawText(QRectF(self.rect()).adjusted(16, 0, -16, 0),
                   int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap), text)


class HistogramView(_Plot):
    """這一張照片的直方圖。拉伸後：0～255、線性座標（像一般修圖軟體）；拉伸前：線性 ADU、兩軸都用對數，
    標出背景（虛線）和飽和（紅線）。彩色可以分 R、G、B 或合成一條亮度。"""

    def __init__(self):
        super().__init__(220)
        self.preview = None
        self.stretched = True
        self.luminance = False

    def set_data(self, preview, stretched: bool, luminance: bool) -> None:
        self.preview, self.stretched, self.luminance = preview, stretched, luminance
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = self._begin()
        pv = self.preview
        if pv is None or pv.hist is None:
            self._say(p, self.message or ptr("gui.hist.loading"))
            return
        left, top, right, bottom = 10.0, 8.0, 10.0, 30.0
        area = QRectF(left, top, self.width() - left - right, self.height() - top - bottom)
        p.setPen(QPen(QColor(D.separator), 1))
        p.drawRect(area)
        color_image = len(pv.hist) == 3 and not self.luminance
        if self.stretched:
            rows = pv.stretched_hist if color_image else [pv.stretched_lum]
            edges = np.linspace(0, 255, 257)
        else:
            rows = pv.hist if color_image else [pv.hist_lum]
            edges = np.asarray(pv.hist_edges, float)
        rows = [np.asarray(r, float) for r in rows]
        colors = CHANNEL_COLORS if color_image else (D.label,)
        if self.stretched:
            x0, x1 = 0.0, 255.0
            fx = lambda v: area.left() + (v - x0) / (x1 - x0) * area.width()  # noqa: E731
            top_count = max(float(r.max()) for r in rows) or 1.0
            fy = lambda c: area.bottom() - min(c / top_count, 1.0) * (area.height() - 4)  # noqa: E731
        else:
            pos = edges[edges > 0]
            x0, x1 = math.log10(float(pos.min())), math.log10(float(edges.max()))
            fx = lambda v: area.left() + (math.log10(max(v, 10 ** x0)) - x0) / max(x1 - x0, 1e-9) * area.width()  # noqa: E731
            top_count = math.log10(max(max(float(r.max()) for r in rows), 10.0))
            fy = lambda c: area.bottom() - math.log10(max(c, 0.8) / 0.8) / (top_count - math.log10(0.8)) \
                * (area.height() - 4)  # noqa: E731
        p.setClipRect(area.adjusted(1, 1, -1, -1))
        for counts, color in zip(rows, colors):
            path = QPainterPath()
            path.moveTo(fx(edges[0]), area.bottom())
            for i, c in enumerate(counts):
                y = fy(float(c))
                path.lineTo(fx(edges[i]), y)
                path.lineTo(fx(edges[i + 1]), y)
            path.lineTo(fx(edges[-1]), area.bottom())
            col = QColor(color)
            if self.stretched:
                fill = QColor(col)
                fill.setAlphaF(0.25 if color_image else 0.35)
                p.fillPath(path, fill)
            p.setPen(QPen(col, 1))
            p.drawPath(path)
        legend: list[tuple[str, QColor, Qt.PenStyle]] = []
        if not self.stretched:
            if pv.background:
                x = fx(float(np.mean(pv.background)))
                p.setPen(QPen(QColor(D.secondary), 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
                legend.append((ptr("gui.hist.background"), QColor(D.secondary), Qt.PenStyle.DashLine))
            if pv.saturation:
                x = fx(float(pv.saturation))
                p.setPen(QPen(QColor(D.reject), 1))
                p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
                legend.append((ptr("gui.hist.saturation"), QColor(D.reject), Qt.PenStyle.SolidLine))
        p.setClipping(False)
        p.setFont(fonts().ui(7.5))
        fm = p.fontMetrics()
        y = area.top() + 6
        for text, col, style in legend:                       # 右上角的圖例
            w = fm.horizontalAdvance(text)
            p.setPen(QPen(col, 1, style))
            p.drawLine(QPointF(area.right() - w - 30, y + fm.height() / 2), QPointF(area.right() - w - 12, y + fm.height() / 2))
            p.setPen(QColor(D.label))
            p.drawText(QPointF(area.right() - w - 8, y + fm.ascent()), text)
            y += fm.height() + 2
        p.setPen(QColor(D.secondary))
        if self.stretched:
            ticks = [0, 64, 128, 192, 255]
        else:
            ticks = [10 ** k for k in range(math.ceil(x0), math.floor(x1) + 1)]
        for t in ticks:
            x = fx(float(t))
            text = f"{t:g}" if t < 10000 else f"1e{round(math.log10(t))}"
            p.drawText(QRectF(x - 30, area.bottom() + 2, 60, 12), int(Qt.AlignmentFlag.AlignHCenter), text)
        p.drawText(QRectF(area.left(), area.bottom() + 15, area.width(), 13), int(Qt.AlignmentFlag.AlignHCenter),
                   ptr("gui.hist.stretched_axis") if self.stretched else ptr("gui.hist.adu"))


class FieldView(_Plot):
    """像場。grid：照片四角、四邊中點、正中央各一塊拼成九宮格，左上角標那一區的星點大小（px），最大的那個角標紅色。
    quad：像場四邊形，虛線是完全平整的樣子，實線是量到的（哪個角星點大就往外凸），短線是星點拉長的方向。"""

    def __init__(self):
        super().__init__(300)
        self.result = None
        self.view = "grid"

    def set_data(self, result, message: str, view: str) -> None:
        self.result, self.message, self.view = result, message, view
        self.update()

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = self._begin()
        if self.result is None:
            self._say(p, self.message)
            return
        if self.view == "quad" or self.result.tiles is None:
            self._quad(p)
        else:
            self._grid(p)

    def _grid(self, p: QPainter) -> None:
        from astro_light_selector.field import CORNERS, GRID
        r = self.result
        tiles = r.tiles
        t = int(tiles[0][0].shape[0])
        gap = max(2, t // 32)
        size = GRID * t + (GRID - 1) * gap
        mosaic = np.full((size, size), 70, dtype=np.uint8)    # 格線的灰
        for row in range(GRID):
            for col in range(GRID):
                y, x = row * (t + gap), col * (t + gap)
                mosaic[y:y + t, x:x + t] = np.asarray(tiles[row][col])[:t, :t]
        img = QImage(mosaic.tobytes(), size, size, size, QImage.Format.Format_Grayscale8)
        side = min(self.width(), self.height()) - 12
        ox, oy = (self.width() - side) / 2, (self.height() - side) / 2
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, side < size)
        p.drawImage(QRectF(ox, oy, side, side), img)
        k = side / size
        worst_cell = CORNERS.get(r.worst_corner) if r.worst_corner else None
        p.setFont(fonts().mono(8.5))
        fm = p.fontMetrics()
        for row, cells in enumerate(r.cells):
            for col, cell in enumerate(cells):
                text = f"{cell.fwhm:.2f}" if cell is not None else "—"
                x = ox + (col * (t + gap) + t * 0.05) * k
                y = oy + (row * (t + gap) + t * 0.05) * k
                box = QRectF(x, y, fm.horizontalAdvance(text) + 6, fm.height() + 2)
                p.fillRect(box, QColor(0, 0, 0, 153))
                p.setPen(QColor(D.reject if (row, col) == worst_cell else D.label if cell is not None else D.secondary))
                p.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    def _quad(self, p: QPainter) -> None:
        from astro_light_selector.field import CORNERS, GRID
        r = self.result
        # 座標跟 Pick 一樣：x −2.15～2.15、y −1.95～1.95（影像方向：上面是照片的上緣），等比例
        k = min(self.width() / 4.3, self.height() / 3.9)
        cx, cy = self.width() / 2, self.height() / 2
        pt = lambda x, y: QPointF(cx + x * k, cy + y * k)  # noqa: E731
        p.setPen(QPen(QColor(D.secondary), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(QPolygonF([pt(-1, -1), pt(1, -1), pt(1, 1), pt(-1, 1)]))
        points = pick_detail().quad_points(r)
        keep = QColor(D.accent)
        if points is not None:
            poly = QPolygonF([pt(*points[n]) for n in ("tl", "tr", "br", "bl")])
            fill = QColor(keep)
            fill.setAlphaF(0.12)
            p.setBrush(fill)
            p.setPen(QPen(keep, 1.8))
            p.drawPolygon(poly)
            p.setBrush(keep)
            for q in poly:
                p.drawEllipse(q, 2.6, 2.6)
        names = {pos: name for name, pos in CORNERS.items()}
        p.setFont(fonts().mono(9))
        for row, cells in enumerate(r.cells):
            for col, cell in enumerate(cells):
                dx, dy = col - GRID // 2, row - GRID // 2
                name = names.get((row, col))
                if name is not None and points is not None:   # 角落：數字放在頂點外側
                    px, py = points[name]
                    tx, ty = px + dx * 0.34, py + dy * 0.24
                elif (dx, dy) == (0, 0) or points is None:
                    tx, ty = dx * 1.34, dy * 1.24
                else:                                         # 四邊：放在那一邊比較凸的頂點外面，才不會壓到線
                    side = [q for n, q in points.items() if (dx and CORNERS[n][1] - GRID // 2 == dx)
                            or (dy and CORNERS[n][0] - GRID // 2 == dy)]
                    tx = dx * (max(1.0, *(abs(q[0]) for q in side)) + 0.36) if dx else 0.0
                    ty = dy * (max(1.0, *(abs(q[1]) for q in side)) + 0.26) if dy else 0.0
                if cell is None:
                    p.setPen(QColor(D.secondary))
                    p.drawText(QRectF(cx + tx * k - 30, cy + ty * k - 9, 60, 18), int(Qt.AlignmentFlag.AlignCenter), "—")
                    continue
                p.setPen(QColor(D.reject if name is not None and name == r.worst_corner else D.label))
                p.drawText(QRectF(cx + tx * k - 34, cy + (ty - 0.09) * k - 9, 68, 18),
                           int(Qt.AlignmentFlag.AlignCenter), f"{cell.fwhm:.2f}")
                half = 0.16 * min(1.0, cell.eccentricity) * cell.alignment   # 星點拉長的方向，越長拉得越多
                ex, ey = math.cos(cell.angle) * half, math.sin(cell.angle) * half
                pen = QPen(QColor(D.secondary), 2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(pt(tx - ex, ty + 0.13 - ey), pt(tx + ex, ty + 0.13 + ey))
