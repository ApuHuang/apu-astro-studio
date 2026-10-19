"""挑片的趨勢圖（pyqtgraph）：各指標隨拍攝順序的變化，保留／淘汰分色；內容與 APU Pick 的 plot.py 相同。

比 matplotlib 版多的：滑過去顯示那一張的數值；點一下跳到清單上的那一張；滾輪縮放、拖曳平移時間軸（各格連動）。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from astro_light_selector.i18n import tr as ptr
from astro_light_selector.selector import Decision
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..theme import D, fonts

PANELS = ["score", "fwhm", "eccentricity", "n_stars", "background", "snr", "altitude"]
KEEP, REJECT, THRESHOLD = D.accent, "#ff6b6b", D.beta

pg.setConfigOptions(antialias=True, background=D.canvas, foreground=D.secondary)


def _value(d: Decision, key: str) -> float:
    v = d.score if key == "score" else getattr(d.metrics, key)
    return float("nan") if v is None else float(v)


def _robust_limits(y: np.ndarray) -> tuple[float, float] | None:
    """用 2%～98% 分位數定 y 軸範圍，讓天亮、雲那幾張極端值不會把整格壓扁（同 Pick）。"""
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return None
    lo, hi = np.percentile(y, [2, 98])
    pad = max(hi - lo, abs(hi) * 0.02, 1e-9) * 0.3
    lo, hi = lo - pad, hi + pad
    if y.min() >= lo and y.max() <= hi:
        return None
    return float(lo), float(hi)


class TrendChart(QWidget):
    clicked = Signal(int)          # 點到的那一張在 self.ordered 裡的位置

    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.layout.setSpacing(2)
        self.glw.ci.setContentsMargins(6, 8, 14, 6)
        lay.addWidget(self.glw)
        self.hover = QLabel(self.glw)
        self.hover.setStyleSheet(f"background: rgba(30,30,32,230); color: {D.label}; border: 1px solid "
                                 f"{D.separator_strong}; border-radius: 6px; padding: 4px 8px;")
        self.hover.setFont(fonts().ui(8.5))
        self.hover.hide()
        self.ordered: list[Decision] = []
        self._group_labels: list[tuple[pg.TextItem, int]] = []
        self.plots: list[pg.PlotItem] = []
        self.cursor_lines: list[pg.InfiniteLine] = []
        self.glw.scene().sigMouseMoved.connect(self._moved)
        self.glw.scene().sigMouseClicked.connect(self._clicked)
        self._tick_font = QFont(fonts().ui_family)
        self._tick_font.setPointSizeF(8)

    def set_data(self, decisions: list[Decision], thresholds: dict[str, float] | None) -> None:
        self.glw.clear()
        self.plots, self.cursor_lines = [], []
        self._group_labels: list[tuple[pg.TextItem, int]] = []
        self.hover.hide()
        # 先依分組、再依時間，同組連在一起（LRGB 輪流拍時才不會切得零零碎碎）
        ordered = sorted(decisions, key=lambda d: (d.group, d.metrics.date_obs or "", d.metrics.file))
        self.ordered = ordered
        if not ordered:
            return
        x = np.arange(len(ordered), dtype=float)
        keep = np.array([d.keep for d in ordered])
        panels = [p for p in PANELS
                  if (p != "score" or any(d.score is not None for d in ordered))
                  and (p != "altitude" or any(d.metrics.altitude is not None for d in ordered))]
        n_keep = int(keep.sum())
        title = ptr("plot.title", n=len(ordered), keep=n_keep, reject=len(ordered) - n_keep)
        self.glw.addLabel(title, row=0, col=0, color=D.label, size="10pt")
        first: pg.PlotItem | None = None
        for row, key in enumerate(panels, start=1):
            p = self.glw.addPlot(row=row, col=0)
            self._style(p, ptr(f"plot.{key}"), last=row == len(panels))
            p.setMouseEnabled(x=True, y=False)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if key == "altitude":
                self._altitude(p, ordered, x, keep)
            else:
                self._metric(p, key, ordered, x, keep, thresholds if key == "score" else None)
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen(QColor(255, 255, 255, 60), width=1))
            line.hide()
            p.addItem(line, ignoreBounds=True)
            self.cursor_lines.append(line)
            self.plots.append(p)
        # 換組的地方畫分隔線，最上面一格標組名
        for i in range(len(ordered)):
            if i > 0 and ordered[i].group != ordered[i - 1].group:
                for p in self.plots:
                    p.addItem(pg.InfiniteLine(pos=i - 0.5, angle=90, pen=pg.mkPen(D.separator_strong, width=1)),
                              ignoreBounds=True)
            if ordered[i].group and (i == 0 or ordered[i].group != ordered[i - 1].group):
                t = pg.TextItem(ordered[i].group, color=D.secondary, anchor=(0, 0))
                t.setFont(self._tick_font)
                self.plots[0].addItem(t, ignoreBounds=True)
                t.setPos(i, self.plots[0].viewRange()[1][1])
                t.setZValue(5)
                self._group_labels.append((t, i))
        if first is not None:
            first.setXRange(-1, len(ordered), padding=0)
            first.sigRangeChanged.connect(self._reposition_labels)
        self._reposition_labels()

    def _reposition_labels(self, *_args) -> None:
        if not self.plots:
            return
        top = self.plots[0].viewRange()[1][1]
        for t, i in self._group_labels:
            try:
                t.setPos(i, top)
            except RuntimeError:
                pass

    def _style(self, p: pg.PlotItem, title: str, last: bool) -> None:
        p.showGrid(x=True, y=True, alpha=0.12)
        p.hideButtons()
        p.getViewBox().setBackgroundColor(QColor("#0b0b0c"))
        for side in ("left", "bottom"):
            ax = p.getAxis(side)
            ax.setPen(pg.mkPen(D.separator_strong))
            ax.setTextPen(pg.mkPen(D.secondary))
            ax.setTickFont(self._tick_font)
        p.getAxis("left").setWidth(70)
        p.getAxis("left").enableAutoSIPrefix(False)
        p.setLabel("left", title, color=D.label, **{"font-size": "9pt"})
        if last:
            p.setLabel("bottom", ptr("plot.xlabel"), color=D.label, **{"font-size": "9pt"})
        else:
            p.getAxis("bottom").setStyle(showValues=False)
            p.getAxis("bottom").setHeight(4)

    def _metric(self, p: pg.PlotItem, key: str, ordered: list[Decision], x: np.ndarray, keep: np.ndarray,
                thresholds: dict[str, float] | None) -> None:
        y = np.array([_value(d, key) for d in ordered])
        lim = _robust_limits(y)
        if lim is not None:
            lo, hi = lim
            over, under = y > hi, y < lo
            for mask, symbol, value in ((over, "t1", hi), (under, "t", lo)):
                for m, color in ((mask & keep, KEEP), (mask & ~keep, REJECT)):
                    if m.any():
                        p.plot(x[m], np.full(m.sum(), value), pen=None, symbol=symbol, symbolSize=9,
                               symbolBrush=color, symbolPen=None)
            y = np.where(over | under, np.nan, y)
            p.setYRange(lo, hi, padding=0.02)
        ok = np.isfinite(y)
        if (keep & ok).any():
            p.plot(x[keep & ok], y[keep & ok], pen=None, symbol="o", symbolSize=6, symbolBrush=KEEP, symbolPen=None)
        if (~keep & ok).any():
            p.plot(x[~keep & ok], y[~keep & ok], pen=None, symbol="x", symbolSize=9, symbolBrush=REJECT,
                   symbolPen=None)
        if thresholds:
            for group, th in thresholds.items():
                idx = [i for i, d in enumerate(ordered) if d.group == group]
                if idx:
                    p.plot([min(idx) - 0.5, max(idx) + 0.5], [th, th],
                           pen=pg.mkPen(THRESHOLD, width=1.3, style=Qt.PenStyle.DashLine))

    def _altitude(self, p: pg.PlotItem, ordered: list[Decision], x: np.ndarray, keep: np.ndarray) -> None:
        """目標仰角（保留／淘汰分色）加上月亮仰角（只畫在地平線上的時候），月相變了就標一次。"""
        alt = np.array([np.nan if d.metrics.altitude is None else d.metrics.altitude for d in ordered])
        moon = np.array([np.nan if d.metrics.moon_alt is None else d.metrics.moon_alt for d in ordered])
        up = moon > 0
        if up.any():
            p.plot(x[up], moon[up], pen=None, symbol="o", symbolSize=4, symbolBrush=D.secondary, symbolPen=None)
        ok = np.isfinite(alt)
        if (keep & ok).any():
            p.plot(x[keep & ok], alt[keep & ok], pen=None, symbol="o", symbolSize=6, symbolBrush=KEEP, symbolPen=None)
        if (~keep & ok).any():
            p.plot(x[~keep & ok], alt[~keep & ok], pen=None, symbol="x", symbolSize=9, symbolBrush=REJECT,
                   symbolPen=None)
        shown = None
        for i, d in enumerate(ordered):
            ph = d.metrics.moon_illum
            if ph is None or (shown is not None and abs(ph - shown) < 0.05):
                continue
            shown = ph
            t = pg.TextItem(ptr("plot.moon_phase", p=ph * 100), color=D.secondary, anchor=(0, 0))
            t.setFont(self._tick_font)
            p.addItem(t, ignoreBounds=True)
            t.setPos(i, 90)
        lo = min(0.0, float(np.nanmin(alt)) - 5) if ok.any() else 0.0
        p.setYRange(lo, 90, padding=0.02)

    # ------------------------------------------------------------------ 滑鼠

    def _index_at(self, scene_pos) -> int | None:
        for p in self.plots:
            if p.sceneBoundingRect().contains(scene_pos):
                v = p.getViewBox().mapSceneToView(scene_pos)
                i = int(round(v.x()))
                if 0 <= i < len(self.ordered) and abs(v.x() - i) <= 0.6:
                    return i
                return None
        return None

    def _moved(self, scene_pos) -> None:
        i = self._index_at(scene_pos)
        if i is None:
            self.hover.hide()
            for line in self.cursor_lines:
                line.hide()
            return
        d = self.ordered[i]
        m = d.metrics
        name = m.file.replace("\\", "/").rsplit("/", 1)[-1]
        verdict = ptr("term.keep") if d.keep else ptr("term.reject")
        parts = [f"#{i + 1}  {name}", f"{verdict}" + (f" · {ptr('plot.score')} {d.score:.1f}" if d.score is not None
                                                      else "")]
        vals = []
        for key, fmt in (("fwhm", ".2f"), ("eccentricity", ".2f"), ("n_stars", "d"), ("snr", ".0f")):
            v = getattr(m, key)
            if v is not None and v == v:
                vals.append(f"{ptr('plot.' + key)} {format(v, fmt)}")
        if vals:
            parts.append(" · ".join(vals))
        self.hover.setText("\n".join(parts))
        self.hover.adjustSize()
        pos = self.glw.mapFromScene(scene_pos)
        x = pos.x() + 16 if pos.x() + 16 + self.hover.width() < self.glw.width() else pos.x() - self.hover.width() - 16
        self.hover.move(int(x), int(max(4, pos.y() - self.hover.height() - 8)))
        self.hover.show()
        self.hover.raise_()
        for line in self.cursor_lines:
            line.setPos(i)
            line.show()

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        i = self._index_at(event.scenePos())
        if i is not None:
            self.clicked.emit(i)


__all__ = ["TrendChart", "QPointF"]
