"""影像檢視：滾輪縮放、拖曳平移、雙擊符合視窗、裁切框、左右對照（拖曳中線）、游標位置回報。

每次只處理畫面上看得到的像素（同 APU Processing 的畫布）：從預覽快取（長邊 2560）或完整影像裁出可見範圍
→ 縮到螢幕大小（Retina／高 DPI 用實際的裝置像素，1:1 是真的 1:1）→ 套成品微調 → QImage。
拖曳、縮放進行中先用邏輯像素畫（少 2～4 倍的像素），停手 0.15 秒後再用裝置像素補畫清楚。

座標一律用「完整尺寸的像素」：快速預覽（縮圖，pixel_scale > 1）與完整結果互換時畫面不會跳。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import D, fonts

PREVIEW_LONG_SIDE = 2560
MIN_ZOOM, MAX_ZOOM = 0.02, 16.0
# 新影像的尺寸跟目前的差在這個比例以內就當成同一張（保留縮放與位置）：自動裁疊圖邊緣在快速預覽上量的邊
# 會跟完整解析度差幾個像素（LBN576 加邊緣：2244×1492 對 2246×1496），以前差超過 1 px 就重設，改數值後縮放就跳回整張
SAME_FRAME = 0.01


def area_downsample(image: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """面積平均縮圖（PIL BOX，每個色版分開）；回傳 (縮圖, 原圖 / 縮圖 的倍率)。"""
    planes = image[None] if image.ndim == 2 else image
    h, w = planes.shape[1:]
    scale = long_side / max(h, w)
    if scale >= 1:
        return planes, 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    out = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(p, np.float32), "F")
                               .resize(size, Image.Resampling.BOX)) for p in planes])
    return out, w / size[0]


def _resize(region: np.ndarray, size: tuple[int, int], nearest: bool) -> np.ndarray:
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if region.shape[2] == size[0] and region.shape[1] == size[1]:
        return region
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(ch, np.float32), "F").resize(size, method))
                     for ch in region])


def to_qimage(planes: np.ndarray) -> QImage:
    """(C, h, w) 0–1 → QImage（C=1 灰階、C=3 RGB）。"""
    arr = (np.clip(planes, 0, 1) * 255 + 0.5).astype(np.uint8)
    if arr.shape[0] == 1:
        g = np.ascontiguousarray(arr[0])
        img = QImage(g.data, g.shape[1], g.shape[0], g.strides[0], QImage.Format.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(np.moveaxis(arr[:3], 0, -1))
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
    return img.copy()  # 脫離 numpy 的記憶體


class ImageView(QWidget):
    zoom_changed = Signal(float)
    crop_changed = Signal()
    hovered = Signal(float, float)        # 完整尺寸的座標；離開影像時 (-1, -1)
    stroke = Signal(float, float, float, bool)   # 筆刷：(x, y, 半徑, 加或擦)，完整尺寸的座標
    stroke_done = Signal()
    picked = Signal(float, float)         # 滴管點到的位置

    def __init__(self, empty_text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.empty_text = empty_text
        self.full: np.ndarray | None = None        # (C, H, W) float32，0–1 的顯示影像
        self.preview: np.ndarray | None = None
        self.preview_factor = 1.0
        self.pixel_scale = 1.0                      # 這張圖一個像素＝完整尺寸幾個像素
        self.adjust: Callable[[np.ndarray], np.ndarray] | None = None
        self.compare: np.ndarray | None = None      # 對照模式左邊的圖（同一個幾何，不套微調）
        self.compare_preview: np.ndarray | None = None
        self.compare_factor = 1.0
        self.compare_scale = 1.0
        self.compare_labels = ("", "")
        self.split = 0.5
        self.zoom: float | None = None              # 螢幕（邏輯）像素 / 完整尺寸像素；None＝符合視窗
        self.cx = self.cy = 0.0
        self.crop_mode = False
        self.crop_box: tuple[float, float, float, float] | None = None
        self._drag: tuple[str, QPointF, float, float] | None = None
        self._crop_start: QPointF | None = None
        self._interactive = False
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(150)
        self._idle.timeout.connect(self._end_interaction)
        self._cache: tuple[tuple, QImage, QRectF] | None = None
        self._cache_cmp: tuple[tuple, QImage, QRectF] | None = None
        self._revision = 0
        # 工具：None（拖曳平移）、"brush"（左鍵塗、右鍵擦）、"pick"（滴管）；中鍵一律平移
        self.tool: str | None = None
        self.brush_radius = 60.0                    # 完整尺寸的 px
        self._painting: bool | None = None          # 正在塗（True）或擦（False）
        self._mouse = QPointF(-1, -1)
        self.overlay: QImage | None = None          # 遮罩或排除區域的疊色（低解析度、平滑放大）
        self.overlay_rect = (0.0, 0.0)
        # 預覽區：完整解析度的一小塊，疊在 (x, y)（完整尺寸的座標）；stale＝參數改了、正在重算
        self.inset: np.ndarray | None = None
        self.inset_compare: np.ndarray | None = None
        self.inset_pos = (0.0, 0.0)
        self.inset_stale = False
        self._inset_rev = 0
        self._cache_inset: tuple[tuple, QImage, QRectF] | None = None
        self._cache_inset_cmp: tuple[tuple, QImage, QRectF] | None = None

    # ------------------------------------------------------------------ 工具與疊色

    def set_tool(self, tool: str | None, radius: float | None = None) -> None:
        self.tool = tool
        if radius is not None:
            self.brush_radius = float(radius)
        cursor = {"brush": Qt.CursorShape.BlankCursor, "pick": Qt.CursorShape.CrossCursor}.get(
            tool or "", Qt.CursorShape.ArrowCursor)
        self.setCursor(cursor)
        self.update()

    def set_overlay(self, mask: np.ndarray | None, color: str = "#ff5050", pixel_scale: float = 1.0,
                    alpha: float = 0.45) -> None:
        """mask：(h, w) 0–1；畫成半透明的顏色疊在影像上。"""
        if mask is None:
            self.overlay = None
            self.update()
            return
        m = np.clip(mask, 0, 1)
        h, w = m.shape
        c = QColor(color)
        rgba = np.empty((h, w, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = c.red(), c.green(), c.blue()
        rgba[..., 3] = (m * 255 * alpha).astype(np.uint8)
        img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        self.overlay = img
        self.overlay_rect = (w * pixel_scale, h * pixel_scale)
        self.update()

    def set_inset(self, image: np.ndarray | None, pos: tuple[float, float] = (0.0, 0.0),
                  compare: np.ndarray | None = None, stale: bool = False) -> None:
        """預覽區：image 是完整解析度的一塊 (C, h, w)，左上角在 pos；compare 是對照模式左邊那張的同一塊。"""
        if image is None:
            if self.inset is None:
                return
            self.inset = self.inset_compare = None
        else:
            self.inset = np.ascontiguousarray(image[None] if image.ndim == 2 else image, np.float32)
            self.inset_compare = None if compare is None else np.ascontiguousarray(
                compare[None] if compare.ndim == 2 else compare, np.float32)
            self.inset_pos = (float(pos[0]), float(pos[1]))
        self.inset_stale = stale
        self._inset_rev += 1
        self.update()

    def inset_value(self, fx: float, fy: float) -> np.ndarray | None:
        """游標位置在預覽區裡時，回傳那個像素的值。"""
        if self.inset is None:
            return None
        x, y = int(fx - self.inset_pos[0]), int(fy - self.inset_pos[1])
        if 0 <= y < self.inset.shape[1] and 0 <= x < self.inset.shape[2]:
            return self.inset[:, y, x]
        return None

    # ------------------------------------------------------------------ 內容

    @property
    def has_image(self) -> bool:
        return self.full is not None

    def full_size(self) -> tuple[float, float]:
        """完整尺寸的 (寬, 高)。"""
        if self.full is None:
            return 0.0, 0.0
        return self.full.shape[2] * self.pixel_scale, self.full.shape[1] * self.pixel_scale

    def set_image(self, image: np.ndarray | None, keep_view: bool = True, pixel_scale: float = 1.0) -> None:
        old = self.full_size()
        if image is None:
            self.full = self.preview = None
            self._revision += 1
            self.update()
            return
        planes = image[None] if image.ndim == 2 else image
        self.full = np.ascontiguousarray(planes, np.float32)
        self.pixel_scale = float(pixel_scale)
        self.preview, self.preview_factor = area_downsample(self.full, PREVIEW_LONG_SIDE)
        self._revision += 1
        new = self.full_size()
        same = all(abs(o - n) <= max(1.0, SAME_FRAME * n) for o, n in zip(old, new))
        if not keep_view or old == (0.0, 0.0) or not same:
            self.zoom = None
            self.cx, self.cy = new[0] / 2, new[1] / 2
        else:
            self.cx, self.cy = min(max(self.cx, 0.0), new[0]), min(max(self.cy, 0.0), new[1])
        self._emit_zoom()
        self.update()

    def set_adjust(self, fn: Callable[[np.ndarray], np.ndarray] | None) -> None:
        """成品微調：(C, h, w) 0–1 → 同形狀；只重畫，不重算影像。"""
        self.adjust = fn
        self._revision += 1
        self.update()

    def set_compare(self, image: np.ndarray | None, labels: tuple[str, str] = ("", ""),
                    pixel_scale: float = 1.0) -> None:
        """對照模式：左邊顯示這張（不套微調），右邊是目前的圖；None 關掉。"""
        if image is None:
            self.compare = self.compare_preview = None
        else:
            planes = image[None] if image.ndim == 2 else image
            self.compare = np.ascontiguousarray(planes, np.float32)
            self.compare_preview, self.compare_factor = area_downsample(self.compare, PREVIEW_LONG_SIDE)
            self.compare_scale = float(pixel_scale)
        self.compare_labels = labels
        self._cache_cmp = None
        self.update()

    def set_crop_mode(self, on: bool) -> None:
        self.crop_mode = on
        if not on:
            self.crop_box = None
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_crop_box(self, box: tuple[float, float, float, float] | None) -> None:
        self.crop_box = box
        self.update()
        self.crop_changed.emit()

    # ------------------------------------------------------------------ 縮放

    def fit_zoom(self) -> float:
        w, h = self.full_size()
        if w <= 0 or h <= 0:
            return 1.0
        return max(MIN_ZOOM, min(self.width() / w, self.height() / h))

    def current_zoom(self) -> float:
        return self.fit_zoom() if self.zoom is None else self.zoom

    def fit(self) -> None:
        w, h = self.full_size()
        self.zoom = None
        self.cx, self.cy = w / 2, h / 2
        self._emit_zoom()
        self.update()

    def set_zoom(self, z: float, anchor: QPointF | None = None) -> None:
        """z：螢幕像素 / 完整尺寸像素（1.0 = 100%）；anchor 是保持不動的螢幕位置。"""
        if self.full is None:
            return
        z = min(MAX_ZOOM, max(MIN_ZOOM, z))
        if anchor is not None:
            fx, fy = self.to_image(anchor)
            self.zoom = z
            self.cx = fx - (anchor.x() - self.width() / 2) / z
            self.cy = fy - (anchor.y() - self.height() / 2) / z
        else:
            self.zoom = z
        self._clamp_center()
        self._emit_zoom()
        self._touch()
        self.update()

    def actual_size(self) -> None:
        """100%：一個影像像素＝一個裝置像素（高 DPI 螢幕上也是真的 1:1）。"""
        self.set_zoom(1.0 / self.devicePixelRatioF())

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(self.current_zoom() * self.devicePixelRatioF())

    def _clamp_center(self) -> None:
        w, h = self.full_size()
        self.cx = min(max(self.cx, 0.0), w)
        self.cy = min(max(self.cy, 0.0), h)

    def to_image(self, p: QPointF) -> tuple[float, float]:
        z = self.current_zoom()
        cx, cy = (self.full_size()[0] / 2, self.full_size()[1] / 2) if self.zoom is None else (self.cx, self.cy)
        return cx + (p.x() - self.width() / 2) / z, cy + (p.y() - self.height() / 2) / z

    def to_screen(self, fx: float, fy: float) -> QPointF:
        z = self.current_zoom()
        cx, cy = (self.full_size()[0] / 2, self.full_size()[1] / 2) if self.zoom is None else (self.cx, self.cy)
        return QPointF((fx - cx) * z + self.width() / 2, (fy - cy) * z + self.height() / 2)

    # ------------------------------------------------------------------ 畫

    def _touch(self) -> None:
        self._interactive = True
        self._idle.start()

    def _end_interaction(self) -> None:
        self._interactive = False
        self.update()

    def _render(self, full: np.ndarray, preview: np.ndarray, factor: float, scale: float,
                adjust: Callable[[np.ndarray], np.ndarray] | None,
                origin: tuple[float, float] = (0.0, 0.0)) -> tuple[QImage, QRectF] | None:
        """把可見範圍畫成 QImage，回傳 (影像, 螢幕上的位置)。origin：這張圖左上角的位置（預覽區用）。"""
        z = self.current_zoom()
        ox, oy = origin
        W, H = full.shape[2] * scale, full.shape[1] * scale
        tl = self.to_image(QPointF(0, 0))
        br = self.to_image(QPointF(self.width(), self.height()))
        fx0, fy0 = max(0.0, tl[0] - ox), max(0.0, tl[1] - oy)
        fx1, fy1 = min(W, br[0] - ox), min(H, br[1] - oy)
        if fx1 <= fx0 or fy1 <= fy0:
            return None
        dpr = 1.0 if self._interactive else self.devicePixelRatioF()
        # 每個影像像素在螢幕上占幾個裝置像素；夠小就用預覽快取
        density = z * dpr * scale
        use_preview = density * factor <= 1.0 and preview is not full
        src, k = (preview, scale * factor) if use_preview else (full, scale)
        sh, sw = src.shape[1:]
        ix0, iy0 = int(np.floor(fx0 / k)), int(np.floor(fy0 / k))
        ix1, iy1 = min(sw, int(np.ceil(fx1 / k))), min(sh, int(np.ceil(fy1 / k)))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        a = self.to_screen(ox + ix0 * k, oy + iy0 * k)
        b = self.to_screen(ox + ix1 * k, oy + iy1 * k)
        dest = QRectF(a, b)
        tw, th = max(1, round(dest.width() * dpr)), max(1, round(dest.height() * dpr))
        # 放大超過 1.5 倍用最近鄰（看得出像素與星形）
        region = src[:, iy0:iy1, ix0:ix1]
        nearest = tw > region.shape[2] * 1.5
        out = _resize(region, (tw, th), nearest)
        if adjust is not None:
            out = adjust(out)
        img = to_qimage(out)
        img.setDevicePixelRatio(dpr)
        return img, dest

    def paintEvent(self, _e: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(D.canvas))
        if self.full is None:
            if self.empty_text:
                p.setPen(QColor(D.secondary))
                p.setFont(fonts().ui(11))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty_text)
            return
        z = self.current_zoom()
        key = (self._revision, round(z, 6), round(self.cx, 3), round(self.cy, 3), self.width(), self.height(),
               self._interactive, self.zoom is None)
        if self._cache is None or self._cache[0] != key:
            r = self._render(self.full, self.preview, self.preview_factor, self.pixel_scale, self.adjust)
            self._cache = (key, *r) if r is not None else None
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        if self._cache is not None:
            p.drawImage(self._cache[2], self._cache[1])
        if self.inset is not None:
            ikey = key + (self._inset_rev,)
            if self._cache_inset is None or self._cache_inset[0] != ikey:
                r = self._render(self.inset, self.inset, 1.0, 1.0, self.adjust, self.inset_pos)
                self._cache_inset = (ikey, *r) if r is not None else None
            if self._cache_inset is not None:
                p.drawImage(self._cache_inset[2], self._cache_inset[1])
        if self.compare is not None:
            self._paint_compare(p, key)
        if self.inset is not None:
            self._paint_inset_frame(p)
        if self.overlay is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            ow, oh = self.overlay_rect
            p.drawImage(QRectF(self.to_screen(0, 0), self.to_screen(ow, oh)), self.overlay)
        if self.crop_mode or self.crop_box is not None:
            self._paint_crop(p)
        if self.tool == "brush" and self._mouse.x() >= 0:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = self.brush_radius * self.current_zoom()
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 160), 3))
            p.drawEllipse(self._mouse, r, r)
            p.setPen(QPen(QColor(255, 255, 255, 230), 1.4))
            p.drawEllipse(self._mouse, r, r)

    def _split_x(self) -> float:
        return self.width() * self.split

    def _paint_compare(self, p: QPainter, key: tuple) -> None:
        ckey = key + (id(self.compare),)
        if self._cache_cmp is None or self._cache_cmp[0] != ckey:
            r = self._render(self.compare, self.compare_preview, self.compare_factor, self.compare_scale, None)
            self._cache_cmp = (ckey, *r) if r is not None else None
        sx = self._split_x()
        if self._cache_cmp is not None:
            p.save()
            p.setClipRect(QRectF(0, 0, sx, self.height()))
            p.fillRect(QRectF(0, 0, sx, self.height()), QColor(D.canvas))
            p.drawImage(self._cache_cmp[2], self._cache_cmp[1])
            if self.inset_compare is not None:
                ikey = ckey + (self._inset_rev,)
                if self._cache_inset_cmp is None or self._cache_inset_cmp[0] != ikey:
                    r = self._render(self.inset_compare, self.inset_compare, 1.0, 1.0, None, self.inset_pos)
                    self._cache_inset_cmp = (ikey, *r) if r is not None else None
                if self._cache_inset_cmp is not None:
                    p.drawImage(self._cache_inset_cmp[2], self._cache_inset_cmp[1])
            p.restore()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        p.drawLine(QPointF(sx, 0), QPointF(sx, self.height()))
        cy = self.height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 240))
        p.drawEllipse(QPointF(sx, cy), 13, 13)
        p.setPen(QPen(QColor(D.base), 1.8))
        for d in (-1, 1):
            path = QPainterPath()
            path.moveTo(sx + d * 3, cy - 4.5)
            path.lineTo(sx + d * 7.5, cy)
            path.lineTo(sx + d * 3, cy + 4.5)
            p.drawPath(path)
        f = fonts().ui(9, bold=True)
        p.setFont(f)
        for text, left in ((self.compare_labels[0], True), (self.compare_labels[1], False)):
            if not text:
                continue
            w = p.fontMetrics().horizontalAdvance(text) + 18
            r = QRectF(12 if left else self.width() - w - 12, 12, w, 24)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 160))
            p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor("white"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_inset_frame(self, p: QPainter) -> None:
        """預覽區的外框：白色細線；正在重算時是黃色虛線。"""
        x0, y0 = self.inset_pos
        h, w = self.inset.shape[1:]
        r = QRectF(self.to_screen(x0, y0), self.to_screen(x0 + w, y0 + h))
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.inset_stale:
            p.setPen(QPen(QColor("#ffcc44"), 1.2, Qt.PenStyle.DashLine))
        else:
            p.setPen(QPen(QColor(255, 255, 255, 150), 1.0))
        p.drawRect(r)
        p.restore()

    def _paint_crop(self, p: QPainter) -> None:
        if self.crop_box is None:
            return
        x0, y0, x1, y1 = self.crop_box
        r = QRectF(self.to_screen(x0, y0), self.to_screen(x1, y1)).normalized()
        outside = QPainterPath()
        outside.addRect(QRectF(self.rect()))
        inner = QPainterPath()
        inner.addRect(r)
        p.fillPath(outside.subtracted(inner), QColor(0, 0, 0, 120))
        pen = QPen(QColor("#ffcc44"), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        w, h = abs(x1 - x0), abs(y1 - y0)
        text = f"{w:.0f} × {h:.0f}"
        p.setFont(fonts().mono(9))
        tw = p.fontMetrics().horizontalAdvance(text) + 12
        box = QRectF(r.left(), max(0, r.top() - 22), tw, 20)
        p.fillRect(box, QColor(0, 0, 0, 170))
        p.setPen(QColor("#ffcc44"))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    # ------------------------------------------------------------------ 滑鼠

    def _near_split(self, pos: QPointF) -> bool:
        return self.compare is not None and abs(pos.x() - self._split_x()) <= 10

    def wheelEvent(self, e: QWheelEvent) -> None:  # noqa: N802
        if self.full is None:
            return
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        self.set_zoom(self.current_zoom() * (1.25 ** steps), e.position())

    def _emit_stroke(self, pos: QPointF) -> None:
        fx, fy = self.to_image(pos)
        self.stroke.emit(fx, fy, self.brush_radius, bool(self._painting))

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self.full is None:
            return
        pos = e.position()
        if e.button() == Qt.MouseButton.MiddleButton or (
                e.button() == Qt.MouseButton.LeftButton and self.tool is None and not self.crop_mode
                and not self._near_split(pos)):
            if self.zoom is None:
                self.zoom = self.fit_zoom()
                self.cx, self.cy = self.full_size()[0] / 2, self.full_size()[1] / 2
            self._drag = ("pan", pos, self.cx, self.cy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self.tool == "brush" and e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._painting = e.button() == Qt.MouseButton.LeftButton
            self._emit_stroke(pos)
            return
        if self.tool == "pick" and e.button() == Qt.MouseButton.LeftButton:
            fx, fy = self.to_image(pos)
            w, h = self.full_size()
            if 0 <= fx < w and 0 <= fy < h:
                self.picked.emit(fx, fy)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._near_split(pos):
            self._drag = ("split", pos, self.split, 0.0)
        elif self.crop_mode:
            fx, fy = self.to_image(pos)
            w, h = self.full_size()
            self._crop_start = QPointF(min(max(fx, 0), w), min(max(fy, 0), h))
            self.crop_box = None
        else:
            if self.zoom is None:
                self.zoom = self.fit_zoom()
                self.cx, self.cy = self.full_size()[0] / 2, self.full_size()[1] / 2
            self._drag = ("pan", pos, self.cx, self.cy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pos = e.position()
        self._mouse = pos
        if self.tool == "brush":
            if self._painting is not None and self._drag is None:
                self._emit_stroke(pos)
            self.update()
        if self.full is not None:
            fx, fy = self.to_image(pos)
            w, h = self.full_size()
            self.hovered.emit(fx, fy) if 0 <= fx < w and 0 <= fy < h else self.hovered.emit(-1.0, -1.0)
        if self._drag is not None:
            kind, start, a, b = self._drag
            if kind == "split":
                self.split = min(0.98, max(0.02, pos.x() / max(1, self.width())))
                self.update()
            else:
                z = self.current_zoom()
                self.cx = a - (pos.x() - start.x()) / z
                self.cy = b - (pos.y() - start.y()) / z
                self._clamp_center()
                self._touch()
                self.update()
            return
        if self._crop_start is not None:
            fx, fy = self.to_image(pos)
            w, h = self.full_size()
            fx, fy = min(max(fx, 0), w), min(max(fy, 0), h)
            s = self._crop_start
            self.crop_box = (min(s.x(), fx), min(s.y(), fy), max(s.x(), fx), max(s.y(), fy))
            self.update()
            return
        if not self.crop_mode and self.tool is None:
            self.setCursor(Qt.CursorShape.SplitHCursor if self._near_split(pos) else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._painting is not None:
            self._painting = None
            self.stroke_done.emit()
            return
        if self._drag is not None:
            self._drag = None
            self.set_tool(self.tool)
            if self.crop_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
        if self._crop_start is not None:
            self._crop_start = None
            if self.crop_box is not None:
                x0, y0, x1, y1 = self.crop_box
                if x1 - x0 < 8 or y1 - y0 < 8:
                    self.crop_box = None
            self.update()
            self.crop_changed.emit()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if not self.crop_mode and e.button() == Qt.MouseButton.LeftButton:
            self.fit()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        self._mouse = QPointF(-1, -1)
        self.hovered.emit(-1.0, -1.0)
        self.update()

    def resizeEvent(self, e) -> None:  # noqa: N802
        if self.zoom is None:
            self._emit_zoom()
        super().resizeEvent(e)
