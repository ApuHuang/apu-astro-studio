"""挑片（APU Pick 的核心）：開資料夾 → 量測 → 調門檻（即時更新）→ 搬移淘汰片／全部還原 → 送去疊圖。

流程、判斷、報表與 APU Pick 0.5 完全相同（直接呼叫 astro_light_selector），介面換成 Qt；
另外做了 0.6 規劃的幾項：滑桿數字可以直接輸入、清單用鍵盤快速挑片（K 保留、X 淘汰、A 改回自動）、
保留片的總曝光、趨勢圖點一下跳到那一張。
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
from astro_light_selector import settings as pick_settings
from astro_light_selector.grouping import EXPOSURE_TOLERANCE
from astro_light_selector.i18n import tr as ptr
from astro_light_selector.i18n import tr_metric
from astro_light_selector.metrics import FrameMetrics, is_image
from astro_light_selector.mover import read_move_log
from astro_light_selector.pipeline import (REJECT_DIR_NAME, REPORT_NAME, Cancelled, Selection, collect_files, decide,
                                           frame_key, measure_files, move_files, restore_files, should_recurse)
from astro_light_selector.preview import Preview, closeup, make_preview, thumbnail
from astro_light_selector.report import (match_frames, read_csv, read_overrides, report_subfolders,
                                         score_summary_lines, write_csv)
from astro_light_selector.scoring import DEFAULT_WEIGHTS, METRICS, ScoreConfig
from PIL import Image
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMenu,
                               QPlainTextEdit, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..i18n import tr
from ..theme import D, fonts
from ..ui.var import Var
from ..ui.widgets import (Hero, Inspector, MetricRow, ParameterSlider, ParameterToggle, Segmented, ask_yes_no, button,
                          delayed, error, info, label, short_path, warn)
from ..ui.worker import Bridge, run_thread
from .base import Page, Views

# 清單欄位：(key, 標題的翻譯代號, 寬度 px, 對齊)
COLUMNS = [
    ("file", "gui.col.file", 300, Qt.AlignmentFlag.AlignLeft),
    ("folder", "gui.col.folder", 150, Qt.AlignmentFlag.AlignLeft),
    ("result", "gui.col.result", 64, Qt.AlignmentFlag.AlignCenter),
    ("score", "gui.col.score", 58, Qt.AlignmentFlag.AlignRight),
    ("fwhm", "gui.col.fwhm", 58, Qt.AlignmentFlag.AlignRight),
    ("ecc", "gui.col.ecc", 70, Qt.AlignmentFlag.AlignRight),
    ("stars", "gui.col.stars", 62, Qt.AlignmentFlag.AlignRight),
    ("bkg", "gui.col.bkg", 74, Qt.AlignmentFlag.AlignRight),
    ("snr", "gui.col.snr", 58, Qt.AlignmentFlag.AlignRight),
    ("alt", "gui.col.alt", 58, Qt.AlignmentFlag.AlignRight),
    ("moon", "gui.col.moon", 70, Qt.AlignmentFlag.AlignRight),
    ("group", "gui.col.group", 170, Qt.AlignmentFlag.AlignLeft),
    ("reason", "gui.col.reason", 360, Qt.AlignmentFlag.AlignLeft),
]
WATCH_INTERVAL_MS = 10_000


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _fmt(v: float | None, spec: str) -> str:
    return "-" if v is None or v != v else format(v, spec)


def _hours(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60}h {m % 60:02d}m"


def pil_to_pixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGB":
        img = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    q = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class _Item(QTableWidgetItem):
    """數字欄照數值排序。"""

    def __init__(self, text: str, key: object):
        super().__init__(text)
        self.key = key

    def __lt__(self, other: QTableWidgetItem) -> bool:
        a, b = self.key, getattr(other, "key", None)
        try:
            return (a is None, a) < (b is None, b)
        except TypeError:
            return str(a) < str(b)


class FrameTable(QTableWidget):
    """清單：鍵盤快速挑片（K 強制保留、X 強制淘汰、A 改回自動）。"""

    def __init__(self, on_key) -> None:
        super().__init__()
        self.on_key = on_key

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        key = {Qt.Key.Key_K: "keep", Qt.Key.Key_X: "reject", Qt.Key.Key_A: None}
        if e.key() in key and not e.modifiers():
            self.on_key(key[e.key()])
            return
        super().keyPressEvent(e)


class ClickLabel(QLabel):
    def __init__(self, on_click) -> None:
        super().__init__()
        self.on_click = on_click

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.on_click(e.position())


class PickPage(Page):
    key = "pick"

    def __init__(self, shell):
        super().__init__(shell)
        st = pick_settings.load_settings()
        self.frames: list[FrameMetrics] = []
        self.selection: Selection | None = None
        self.folder: Path | None = None
        self.reject_dir_override: Path | None = None
        self.overrides: dict[str, str] = {}
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self._started = 0.0
        self._previews: OrderedDict[str, Preview] = OrderedDict()
        self._preview_key: str | None = None
        self._preview_center = (0.5, 0.5)
        self._watch_sizes: dict[str, int] = {}
        self.bridge = Bridge(self._handle, self)

        self.view = Var("plot")
        self.method = Var("auto")
        self.pass_line = Var(80.0)
        self.min_score = Var(85.0)
        self.keep_best = Var(70.0)
        self.grouping = {k: Var(v) for k, v in (("filter", True), ("exposure", True), ("gain", True),
                                              ("night", False), ("folder", False))}
        self.recursive = Var(False)
        self.exposure_tol = Var(float(st.get("exposure_tolerance", EXPOSURE_TOLERANCE)))
        self.workers = Var(default_workers())
        self.watch = Var(False)
        self.only_reject = Var(False)
        saved = st.get("weights") or {}
        self.weights = {m: Var(float(saved.get(m, DEFAULT_WEIGHTS[m] * 100))) for m in METRICS}

        self._apply_timer = delayed(self, 300, self._apply)  # 跟 APU Astro 一樣，停止操作約 0.3 秒後才更新結果
        self._preview_timer = delayed(self, 120, self._load_preview)
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(WATCH_INTERVAL_MS)
        self._watch_timer.timeout.connect(self._watch_tick)
        for var in (self.method, self.pass_line, self.min_score, self.keep_best, self.exposure_tol,
                    *self.grouping.values(), *self.weights.values()):
            var.subscribe(lambda _v: self._schedule_apply())
        self.recursive.subscribe(lambda _v: self._recursive_changed())
        self.only_reject.subscribe(lambda _v: self._fill_table())
        self.watch.subscribe(lambda _v: self._toggle_watch())
        self.view.subscribe(lambda v: self.views.show_key(v) if hasattr(self, "views") else None)
        self._status_fn = lambda: ptr("gui.status.start")
        self._setting_folder = False

    # ------------------------------------------------------------------ 介面

    def build_ui(self) -> QWidget:
        self.views_seg = Segmented([("plot", ptr("gui.tab.plot")), ("list", ptr("gui.tab.list")),
                                    ("details", ptr("gui.tab.details"))], self.view)
        self.open_btn = button(ptr("gui.btn.open"), tooltip=ptr("gui.btn.open.help", shortcut="Ctrl+O"),
                               on_click=self.ask_open)
        self.measure_btn = button(ptr("gui.btn.measure"), tooltip=ptr("gui.btn.measure.help"),
                                  on_click=self._measure_or_stop)
        self.move_btn = button(ptr("gui.btn.move"), tooltip=ptr("gui.btn.move.help"), on_click=self._move)
        self.send_btn = button(tr("btn.send_stack"), "prominent", tr("btn.send_stack.help"), self.send_to_stack)

        content = QStackedWidget()
        self.hero = Hero(tr("pick.welcome.title"), tr("pick.welcome.sub"))
        content.addWidget(self.hero)
        self.chart = None
        self.views = Views({"plot": self._build_plot(), "list": self._build_list(), "details": self._build_details()})
        self.views.show_key(self.view.get())
        content.addWidget(self.views)
        self.content = content
        return self.layout([self.views_seg], [self.open_btn, self.measure_btn, self.move_btn, self.send_btn],
                           content, self._build_inspector())

    def _build_plot(self) -> QWidget:
        from .pick_chart import TrendChart
        self.chart = TrendChart()
        self.chart.clicked.connect(self._chart_clicked)
        return self.chart

    def _build_list(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget()
        bar.setStyleSheet(f"background: {D.base};")
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 6, 12, 6)
        h.addWidget(ParameterToggle(ptr("gui.only_reject"), self.only_reject))
        h.addStretch(1)
        h.addWidget(label(tr("pick.keys.hint"), D.tertiary, 8.5))
        for key, choice in (("gui.btn.force_keep", "keep"), ("gui.btn.force_reject", "reject"),
                            ("gui.btn.clear_override", None)):
            h.addWidget(button(ptr(key), tooltip=ptr("gui.override.help", shortcut="Ctrl"),
                               on_click=lambda c=choice: self._set_override(c)))
        v.addWidget(bar)
        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(0)
        self.table = FrameTable(self._set_override)
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels([ptr(c[1]) for c in COLUMNS])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSortingEnabled(True)
        self.table.setWordWrap(False)
        hh = self.table.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setStretchLastSection(True)
        for i, (_, _, width, _) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, width)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.itemSelectionChanged.connect(self._preview_timer.start)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._row_menu)
        split.addWidget(self.table, 1)
        split.addWidget(self._build_preview_pane())
        v.addLayout(split, 1)
        return w

    def _build_preview_pane(self) -> QWidget:
        pane = QWidget()
        pane.setFixedWidth(430)
        pane.setStyleSheet(f"background: {D.list_bg}; border-left: 1px solid {D.separator};")
        v = QVBoxLayout(pane)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(4)
        self.preview_title = label("", D.label, 9, bold=True)
        self.preview_sky = label("", D.secondary, 8.5, wrap=True)
        self.preview_thumb = ClickLabel(self._move_closeup)
        self.preview_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_thumb.setMinimumHeight(260)
        self.preview_thumb.setCursor(Qt.CursorShape.CrossCursor)
        self.preview_thumb.setStyleSheet(f"color: {D.secondary}; border: none;")
        self.preview_caption = label("", D.secondary, 8.5)
        self.preview_closeup = QLabel()
        self.preview_closeup.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_closeup.setStyleSheet("border: none;")
        for w in (self.preview_title, self.preview_sky, self.preview_thumb, self.preview_caption,
                  self.preview_closeup):
            w.setStyleSheet(w.styleSheet() + "border: none;")
            v.addWidget(w)
        v.addStretch(1)
        self._clear_preview(ptr("gui.preview.hint"))
        return pane

    def _build_details(self) -> QWidget:
        self.details_text = QPlainTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(fonts().ui(9.5))
        return self.details_text

    def _build_inspector(self) -> Inspector:
        ins = Inspector()
        g = ins.add(self.group("result", ptr("gui.group.result"), ptr("gui.group.result.info")))
        self.metrics = {key: g.add(MetricRow(ptr(f"gui.metric.{key}")))
                        for key in ("frames", "kept", "rejected", "manual", "threshold", "ref_fwhm")}
        self.metrics["exposure"] = g.add(MetricRow(tr("pick.total_exposure")))
        self.exposure_detail = g.add(label("", D.secondary, 8.5, wrap=True))

        g = ins.add(self.group("threshold", ptr("gui.group.threshold"), ptr("gui.group.threshold.info")))
        g.add(Segmented([("auto", ptr("gui.method.auto")), ("min_score", ptr("gui.method.min_score")),
                         ("keep_best", ptr("gui.method.keep_best"))], self.method, stretch=True))
        points = lambda v: ptr("gui.value.points", v=v)  # noqa: E731
        self.sliders = {
            "auto": g.add(ParameterSlider(ptr("gui.slider.pass_line"), self.pass_line, 50, 100, points), 6),
            "min_score": g.add(ParameterSlider(ptr("gui.slider.min_score"), self.min_score, 0, 100, points)),
            "keep_best": g.add(ParameterSlider(ptr("gui.slider.keep_best"), self.keep_best, 10, 100,
                                               lambda v: f"{v:.0f}%")),
        }

        g = ins.add(self.group("weights", ptr("gui.group.weights"), ptr("gui.group.weights.info")))
        self.weight_badge = label("", D.accent, 8.5)
        self.reset_weights_btn = button(ptr("gui.btn.reset_weights"), on_click=self._reset_weights)
        g.add_row(self.weight_badge, QWidget(), self.reset_weights_btn, stretch_last=False)
        self.weight_sliders = {}
        for m, key in (("fwhm", "gui.col.fwhm"), ("eccentricity", "gui.col.ecc"), ("n_stars", "gui.col.stars"),
                       ("background", "gui.col.bkg"), ("snr", "gui.col.snr")):
            self.weight_sliders[m] = g.add(ParameterSlider(ptr(key), self.weights[m], 0, 100,
                                                           lambda _v, m=m: self._weight_share_text(m)))
        for var in self.weights.values():
            var.subscribe(lambda _v: self._update_weight_view(), g)

        g = ins.add(self.group("grouping", ptr("gui.group.grouping"), ptr("gui.group.grouping.info")))
        for key in ("filter", "exposure", "gain", "night", "folder"):
            g.add(ParameterToggle(ptr(f"gui.toggle.{key}"), self.grouping[key]))
        g.add(ParameterSlider(ptr("gui.slider.exposure_tolerance"), self.exposure_tol, 0, 30,
                              lambda v: ptr("gui.value.seconds", v=v)), 6)

        g = ins.add(self.group("measure", ptr("gui.group.measure"), ptr("gui.group.measure.info")))
        g.add(ParameterSlider(ptr("gui.slider.workers"), self.workers, 1, max(2, os.cpu_count() or 2),
                              lambda v: f"{v:.0f}"))
        g.add(ParameterToggle(ptr("gui.toggle.recursive"), self.recursive), 4)
        g.add(ParameterToggle(ptr("gui.toggle.watch"), self.watch))
        self.load_btn = button(ptr("gui.btn.load"), on_click=self._load_report)
        g.add_row(self.load_btn)

        g = ins.add(self.group("rejects", ptr("gui.group.rejects"), ptr("gui.group.rejects.info")))
        g.add(label(ptr("gui.reject_folder"), D.secondary, 8.5))
        self.reject_label = g.add(label("", D.label, 8.5, wrap=True))
        self.change_btn = button(ptr("gui.btn.change"), on_click=self._browse_reject)
        self.restore_btn = button(ptr("gui.btn.restore"), on_click=self._restore)
        g.add_row(self.change_btn, self.restore_btn)
        return ins

    # ------------------------------------------------------------------ 外殼

    def doc_title(self) -> str:
        return self.folder.name if self.folder else ""

    def rail_text(self) -> str:
        if self.is_busy() and self.progress is not None:
            return tr("rail.busy", p=round(self.progress * 100))
        if self.selection is not None:
            return f"{self.selection.n_keep}/{len(self.selection.decisions)}"
        return ""

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def close_page(self) -> None:
        self.cancel.set()
        self._watch_timer.stop()

    def ask_open(self) -> None:
        if self.is_busy():
            return
        start = str(self.folder) if self.folder and self.folder.is_dir() else ""
        path = QFileDialog.getExistingDirectory(self, ptr("gui.pick_folder"), start)
        if path:
            self.open_folder(Path(path))

    def handle_drop(self, paths: list[Path]) -> None:
        if paths and not self.is_busy():
            self.open_folder(paths[0])

    # ------------------------------------------------------------------ 狀態

    def _reject_dir(self) -> Path | None:
        if self.reject_dir_override is not None:
            return self.reject_dir_override
        return self.folder / REJECT_DIR_NAME if self.folder else None

    def refresh(self) -> None:
        busy = self.is_busy()
        has_folder = self.folder is not None and self.folder.is_dir()
        self.open_btn.setEnabled(not busy)
        if busy:
            self.measure_btn.setText(ptr("gui.btn.stop"))
            self.measure_btn.setEnabled(True)
        else:
            self.measure_btn.setText(ptr("gui.btn.remeasure") if self.frames else ptr("gui.btn.measure"))
            self.measure_btn.setEnabled(has_folder)
        self.move_btn.setEnabled(not busy and self.selection is not None)
        n_keep = self.selection.n_keep if self.selection is not None else 0
        self.send_btn.setEnabled(not busy and n_keep > 0)
        self.send_btn.setText(tr("btn.send_stack.n", n=n_keep) if n_keep else tr("btn.send_stack"))
        self.load_btn.setEnabled(not busy and has_folder and (self.folder / REPORT_NAME).is_file())
        self.restore_btn.setEnabled(not busy and has_folder)
        self.change_btn.setEnabled(not busy and has_folder and not self.recursive.get())
        reject_dir = self._reject_dir()
        self.reject_label.setText(ptr("gui.reject_each") if self.recursive.get()
                                  else (short_path(str(reject_dir)) if reject_dir else "—"))
        hint = ptr("gui.welcome.measuring") if busy else (ptr("gui.welcome.ready") if has_folder
                                                         else ptr("gui.welcome.hint"))
        self.hero.set_hint(hint + ("" if has_folder else "\n" + tr("drop.hint")))
        has = self.selection is not None
        self.content.setCurrentIndex(1 if has else 0)
        self.views_seg.setVisible(has)
        for key, slider in self.sliders.items():
            slider.setVisible(key == self.method.get())
        self._update_weight_view()
        self.has_result = has
        self.status_changed.emit()
        self.rail_changed.emit()
        if has:
            self._fill_metrics()
        else:
            for row in self.metrics.values():
                row.set("—")
            self.exposure_detail.setText("")

    def _weights_now(self) -> dict[str, float] | None:
        weights = {m: max(0.0, float(var.get())) for m, var in self.weights.items()}
        return weights if sum(weights.values()) > 0 else None

    def _weight_share_text(self, metric: str) -> str:
        weights = self._weights_now()
        return f"{weights[metric] / sum(weights.values()) * 100:.0f}%" if weights else "—"

    def _update_weight_view(self) -> None:
        if not hasattr(self, "weight_sliders"):
            return
        try:
            for slider in self.weight_sliders.values():
                slider.refresh()
            weights = self._weights_now()
            total = sum(weights.values()) if weights else 0
            recommended = weights is not None and all(abs(weights[m] / total - DEFAULT_WEIGHTS[m]) < 0.005
                                                      for m in METRICS)
            self.weight_badge.setText(ptr("gui.weights.recommended") if recommended else ptr("gui.weights.custom"))
            self.weight_badge.setStyleSheet(f"color: {D.accent if recommended else D.beta}; background: transparent;")
            self.reset_weights_btn.setEnabled(not recommended)
        except RuntimeError:
            pass

    def _reset_weights(self) -> None:
        for m, var in self.weights.items():
            var.set(DEFAULT_WEIGHTS[m] * 100)

    # ------------------------------------------------------------------ 資料夾

    def open_folder(self, folder: Path, recursive: bool | None = None) -> None:
        """開啟放 light frames 的資料夾（拖一張影像進來也行）。recursive=None 時自動判斷要不要包含子資料夾。"""
        if folder.is_file() and is_image(folder):
            folder = folder.parent
        if recursive is None:
            report = folder / REPORT_NAME
            recursive = folder.is_dir() and (report_subfolders(report, folder) or should_recurse(folder))
        self._setting_folder = True
        try:
            self.recursive.set(bool(recursive))
        finally:
            self._setting_folder = False
        self.watch.set(False)  # 換資料夾就停止監看
        self.folder = folder
        self.reject_dir_override = None
        self.frames, self.selection = [], None
        self.overrides = read_overrides(folder / REPORT_NAME) if folder.is_dir() else {}
        self._previews.clear()
        self._preview_key = None
        self._clear_preview(ptr("gui.preview.hint"))
        self._fill_table()
        self.details_text.setPlainText("")
        self.progress = None
        if not folder.is_dir():
            self.set_status(lambda: ptr("gui.status.no_folder", folder=folder))
        elif (folder / REPORT_NAME).is_file():
            self._load_report()
        else:
            files, home_of = collect_files(folder, self._reject_dir(), bool(recursive))
            n = len(files)
            k = len({Path(home_of.get(str(p), p)).resolve().parent for p in files})
            if recursive and n:
                self.set_status(lambda: ptr("gui.status.found_dirs", n=n, k=k))
            else:
                self.set_status(lambda: ptr("gui.status.found", n=n) if n else ptr("gui.status.no_fits"))
        self.refresh()
        self.title_changed.emit()

    def _recursive_changed(self) -> None:
        if self._setting_folder or self.folder is None:
            return
        if self.is_busy():
            self._setting_folder = True
            try:
                self.recursive.set(not self.recursive.get())
            finally:
                self._setting_folder = False
            return
        self.open_folder(self.folder, bool(self.recursive.get()))

    def _browse_reject(self) -> None:
        path = QFileDialog.getExistingDirectory(self, ptr("gui.pick_reject"), str(self.folder or ""))
        if path:
            self.reject_dir_override = Path(path)
            self.refresh()

    def _load_report(self) -> None:
        folder, reject_dir = self.folder, self._reject_dir()
        if folder is None or reject_dir is None:
            return
        try:
            frames = read_csv(folder / REPORT_NAME, folder)
            files, home_of = collect_files(folder, reject_dir, bool(self.recursive.get()))
        except Exception as exc:  # noqa: BLE001
            error(self, ptr("gui.err.load", error=exc))
            return
        # 照 Pick 0.7：路徑對不上的（檔案被搬動過）用檔名認回，手動覆寫也換到新位置
        kept, renamed = match_frames(frames, files, home_of, folder)
        n_new, n_renamed = len(files) - len(kept), len(renamed)
        if renamed:
            self.overrides = {renamed.get(k, k): v for k, v in self.overrides.items()}
        self.frames = kept
        self._apply()
        if renamed:
            self._save_report()
        self.set_status(lambda: ptr("gui.status.loaded", n=len(kept))
                        + (ptr("gui.status.loaded_renamed", n=n_renamed) if n_renamed else "")
                        + (ptr("gui.status.loaded_new", n=n_new) if n_new else ""))
        self.refresh()

    # ------------------------------------------------------------------ 量測

    def _measure_or_stop(self) -> None:
        if self.is_busy():
            self.cancel.set()
            self.watch.set(False)
            self.set_status(lambda: ptr("gui.status.stopping"))
        else:
            self._start_measure()

    def _start_measure(self) -> None:
        folder, reject_dir = self.folder, self._reject_dir()
        if folder is None or reject_dir is None or not folder.is_dir():
            warn(self, ptr("gui.warn.pick_folder"))
            return
        files, home_of = collect_files(folder, reject_dir, bool(self.recursive.get()))
        if not files:
            warn(self, ptr("gui.status.no_fits"))
            return
        n, k = len(files), len(home_of)
        self.set_status(lambda: ptr("gui.status.measuring", n=n,
                                    extra=ptr("gui.status.includes_rejected", n=k) if k else ""), 0.0)
        self._run_measure(files, home_of, "measured")

    def _run_measure(self, files: list[Path], home_of: dict[str, str], kind: str) -> None:
        self.cancel.clear()
        self._started = time.monotonic()
        workers = max(1, int(self.workers.get()))
        post, cancel = self.bridge.post, self.cancel

        def work() -> None:
            try:
                frames = measure_files(files, workers, cancel=cancel, home_of=home_of,
                                       progress=lambda i, n, m: post(("progress", i, n, m)))
                post((kind, frames))
            except Cancelled:
                post(("cancelled",))
            except Exception:  # noqa: BLE001
                post(("error", traceback.format_exc()))

        self.progress = 0.0
        self.worker = run_thread(work, "pick-measure")
        self.refresh()

    def _toggle_watch(self) -> None:
        self._watch_sizes = {}
        if self.watch.get():
            n = len(self.frames)
            self.set_status(lambda: ptr("gui.status.watching", n=n))
            self._watch_timer.start()
            self._watch_tick()
        else:
            self._watch_timer.stop()
            if not self.is_busy():
                self.set_status(lambda: ptr("gui.status.watch_off"))

    def _watch_tick(self) -> None:
        """檢查一次資料夾。新檔案的大小要連續兩次一樣（相機軟體寫完了）才量測。"""
        folder, reject_dir = self.folder, self._reject_dir()
        if not self.watch.get() or self.is_busy() or folder is None or reject_dir is None or not folder.is_dir():
            return
        files, home_of = collect_files(folder, reject_dir, bool(self.recursive.get()))
        known = {frame_key(f.file, folder) for f in self.frames}
        sizes, ready = {}, []
        for p in files:
            if frame_key(home_of.get(str(p), p), folder) in known:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            sizes[str(p)] = size
            if size > 0 and self._watch_sizes.get(str(p)) == size:
                ready.append(p)
        self._watch_sizes = sizes
        if ready:
            n = len(ready)
            self.set_status(lambda: ptr("gui.status.watch_measuring", n=n))
            self._run_measure(ready, home_of, "watched")

    @staticmethod
    def _progress_text(i: int, n: int, eta: float, name: str) -> str:
        if eta >= 90:
            left = ptr("gui.status.left_min", m=eta / 60)
        elif i < n:
            left = ptr("gui.status.left_sec", s=eta)
        else:
            left = ""
        return ptr("gui.status.progress", i=i, n=n, left=left, name=name)

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "preview":
            _, key, preview = event
            self._previews[key] = preview
            while len(self._previews) > 8:
                self._previews.popitem(last=False)
            if key == self._preview_key:
                self._show_preview()
            return
        if kind == "preview_error":
            if event[1] == self._preview_key:
                self._clear_preview(ptr("gui.preview.failed", error=event[2]))
            return
        if kind == "progress":
            _, i, n, m = event
            eta = (time.monotonic() - self._started) / max(1, i) * (n - i)
            name = Path(m.file).name
            self.set_status(lambda: self._progress_text(i, n, eta, name), i / max(1, n))
            self.rail_changed.emit()
            return
        self.worker = None
        self.progress = None
        if kind == "watched":
            new = event[1]
            keys = {frame_key(m.file, self.folder) for m in new}
            self.frames = [f for f in self.frames if frame_key(f.file, self.folder) not in keys] + new
            self._apply()
            self._save_report()
            latest = max(new, key=lambda m: (m.date_obs or "", m.file))
            n, name, total = len(new), Path(latest.file).name, len(self.frames)
            self.set_status(lambda: ptr("gui.status.watch_added", n=n, name=name, total=total))
        elif kind == "measured":
            self.frames = event[1]
            self._apply()
            self._save_report()
            took = (time.monotonic() - self._started) / 60
            n, errors = len(self.frames), sum(1 for f in self.frames if f.error)
            self.set_status(lambda: ptr("gui.status.done", n=n, m=took)
                            + (ptr("gui.status.done_errors", n=errors) if errors else "")
                            + ptr("gui.status.done_saved", report=REPORT_NAME))
        elif kind == "cancelled":
            self.set_status(lambda: ptr("gui.status.cancelled"))
        elif kind == "error":
            self.set_status(lambda: ptr("gui.status.failed"))
            error(self, ptr("gui.err.measure", error=event[1][-1500:]))
        self.refresh()

    def _save_report(self) -> None:
        if self.folder is not None and self.selection is not None:
            try:
                write_csv(self.selection.decisions, self.folder / REPORT_NAME, root=self.folder)
            except OSError as exc:
                warn(self, ptr("gui.warn.report_save", error=exc))

    # ------------------------------------------------------------------ 挑片

    def _schedule_apply(self) -> None:
        if hasattr(self, "sliders"):
            for key, slider in self.sliders.items():
                slider.setVisible(key == self.method.get())
        self._apply_timer.start()

    def _config(self) -> ScoreConfig | None:
        method = self.method.get()
        cfg = ScoreConfig(pass_pct=float(self.pass_line.get()) / 100)
        if method == "min_score":
            cfg.min_score = float(self.min_score.get())
        elif method == "keep_best":
            cfg.keep_best = float(self.keep_best.get()) / 100
        return cfg

    def _group_keys(self) -> tuple[str, ...]:
        return tuple(k for k in ("filter", "exposure", "gain", "night", "folder") if self.grouping[k].get())

    def _apply(self) -> None:
        if not self.frames:
            return
        weights = self._weights_now()
        if weights is None:
            self.set_status(lambda: ptr("gui.status.no_weights"))
            return
        cfg = self._config()
        total = sum(weights.values())
        cfg.weights = {m: w / total for m, w in weights.items()}
        tolerance = max(0.0, float(self.exposure_tol.get()))
        pick_settings.save_settings(weights=weights, exposure_tolerance=tolerance)
        self.selection = sel = decide(self.frames, self._group_keys(), "score", cfg, exposure_tolerance=tolerance,
                                      overrides=self.overrides, root=self.folder)
        n, kept = len(sel.decisions), sel.n_keep
        th = sel.thresholds
        if len(th) == 1:
            extra = ptr("gui.summary.one_threshold", t=next(iter(th.values())))
        else:
            extra = ptr("gui.summary.groups", g=len(th)) if th else ""
        self.set_summary(lambda: ptr("gui.summary", n=n, keep=kept, reject=n - kept, extra=extra))
        parts = [f"{tr_metric(m)} {cfg.weights[m] * 100:.0f}%" for m in METRICS]
        lines: list[str] = [ptr("summary.weights", parts=ptr("summary.part_sep").join(parts)), ""]
        for lab, size in sel.group_sizes.items():
            lines.append(ptr("summary.group_header", label=lab or ptr("group.all"), n=size))
            result = sel.results.get(lab)
            lines += score_summary_lines(result) if result else [ptr("summary.no_measurable")]
            lines.append("")
        self.details_text.setPlainText("\n".join(lines))
        self.chart.set_data(sel.decisions, th)
        self._fill_table()
        self.refresh()

    def _fill_metrics(self) -> None:
        sel = self.selection
        n, kept = len(sel.decisions), sel.n_keep
        self.metrics["frames"].set(str(n))
        self.metrics["kept"].set(str(kept))
        self.metrics["rejected"].set(str(n - kept), D.reject if n - kept else D.label)
        self.metrics["manual"].set(str(sum(1 for d in sel.decisions if d.override)))
        results = [r for r in sel.results.values() if r is not None]
        if len(results) == 1:
            self.metrics["threshold"].set(f"{results[0].threshold:.1f}")
            ref = results[0].reference.get("fwhm")
            self.metrics["ref_fwhm"].set(f"{ref:.2f} px" if ref is not None else "—")
        else:
            self.metrics["threshold"].set(ptr("gui.metric.per_group") if results else "—")
            self.metrics["ref_fwhm"].set(ptr("gui.metric.per_group") if results else "—")
        # 保留片的總曝光（0.6 規劃第 5 項）：每組一行
        per: dict[str, float] = {}
        for d in sel.decisions:
            if d.keep and d.metrics.exposure:
                per[d.group] = per.get(d.group, 0.0) + float(d.metrics.exposure)
        total = sum(per.values())
        self.metrics["exposure"].set(_hours(total) if total else "—")
        self.exposure_detail.setText("\n".join(f"{g or ptr('group.all')}：{_hours(s)}" for g, s in per.items())
                                     if len(per) > 1 else "")
        self.exposure_detail.setVisible(len(per) > 1)

    def _fill_table(self) -> None:
        if not hasattr(self, "table"):
            return
        selected = {it.data(Qt.ItemDataRole.UserRole) for it in self.table.selectedItems()}
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.table.setColumnHidden(1, not self.recursive.get())
        if self.selection is None:
            self.table.setSortingEnabled(True)
            return
        rows = sorted(self.selection.decisions, key=lambda d: (d.metrics.date_obs or "", d.metrics.file))
        if self.only_reject.get():
            rows = [d for d in rows if not d.keep]
        self.table.setRowCount(len(rows))
        keep_text, reject_text = ptr("term.keep"), ptr("term.reject")
        reject_color = QColor(D.reject)
        for r, d in enumerate(rows):
            m = d.metrics
            key = frame_key(m.file, self.folder)
            sub = key.rpartition("/")[0]
            result = keep_text if d.keep else reject_text
            if d.override:
                result = f"✎ {result}"
            values = [
                (Path(m.file).name, Path(m.file).name.casefold()), (sub, sub), (result, (d.keep, result)),
                (_fmt(d.score, ".1f"), d.score), (_fmt(m.fwhm, ".2f"), m.fwhm), (_fmt(m.eccentricity, ".2f"),
                                                                                   m.eccentricity),
                (str(m.n_stars), m.n_stars), (_fmt(m.background, ".0f"), m.background), (_fmt(m.snr, ".1f"), m.snr),
                (_fmt(m.altitude, ".0f"), m.altitude), (_fmt(m.moon_alt, ".0f"), m.moon_alt), (d.group, d.group),
                ("; ".join(str(x) for x in d.reasons), ""),
            ]
            for c, (text, sort_key) in enumerate(values):
                if isinstance(sort_key, float) and sort_key != sort_key:
                    sort_key = None
                it = _Item(text, sort_key)
                it.setData(Qt.ItemDataRole.UserRole, key)
                it.setTextAlignment(COLUMNS[c][3] | Qt.AlignmentFlag.AlignVCenter)
                if not d.keep:
                    it.setForeground(reject_color)
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        if selected:
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) in selected:
                    self.table.selectRow(r) if len(selected) == 1 else self.table.item(r, 0).setSelected(True)

    def _selected_keys(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]

    def _row_menu(self, pos: QPoint) -> None:
        if not self._selected_keys():
            return
        menu = QMenu(self)
        for key, choice in (("gui.btn.force_keep", "keep"), ("gui.btn.force_reject", "reject"),
                            ("gui.btn.clear_override", None)):
            act = QAction(ptr(key), menu)
            act.triggered.connect(lambda _c=False, c=choice: self._set_override(c))
            menu.addAction(act)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _set_override(self, choice: str | None) -> None:
        """把清單上選取的片強制保留／強制淘汰（choice=None 改回自動），並馬上存進報表。"""
        keys = self._selected_keys()
        if not keys or self.selection is None:
            return
        for key in keys:
            if choice is None:
                self.overrides.pop(key, None)
            else:
                self.overrides[key] = choice
        current = self.table.currentRow()
        self._apply()
        self._save_report()
        # 鍵盤挑片：處理完移到下一張
        if len(keys) == 1 and 0 <= current < self.table.rowCount() - 1:
            self.table.selectRow(current + 1)

    def _chart_clicked(self, index: int) -> None:
        """趨勢圖點一下：跳到清單上的那一張（0.6 規劃第 6 項）。"""
        if self.chart is None or not (0 <= index < len(self.chart.ordered)):
            return
        key = frame_key(self.chart.ordered[index].metrics.file, self.folder)
        if self.only_reject.get() and self.chart.ordered[index].keep:
            self.only_reject.set(False)
        self.view.set("list")
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == key:
                self.table.selectRow(r)
                self.table.scrollToItem(self.table.item(r, 0), QAbstractItemView.ScrollHint.PositionAtCenter)
                self.table.setFocus()
                break

    # ------------------------------------------------------------------ 預覽

    def _current_path(self, key: str) -> Path | None:
        """這張片現在在哪：還在 light 資料夾，或已經被搬到淘汰片資料夾。"""
        if self.folder is None:
            return None
        home = self.folder / key
        if home.exists():
            return home
        reject_dir = home.parent / REJECT_DIR_NAME if self.recursive.get() else self._reject_dir()
        moved = read_move_log(reject_dir) if reject_dir else {}
        target = moved.get(home.resolve())
        return target if target is not None and target.exists() else None

    def _load_preview(self) -> None:
        keys = self._selected_keys()
        if not keys:
            return
        key = keys[0]
        self._preview_key = key
        self.preview_title.setText(Path(key).name)
        self.preview_sky.setText(self._sky_text(key))
        if key in self._previews:
            self._previews.move_to_end(key)
            self._show_preview()
            return
        path = self._current_path(key)
        if path is None:
            self._clear_preview(ptr("gui.preview.missing"))
            return
        self._clear_preview(ptr("gui.preview.loading"))
        post = self.bridge.post

        def work() -> None:
            try:
                post(("preview", key, make_preview(path)))
            except Exception as exc:  # noqa: BLE001
                post(("preview_error", key, str(exc)))

        run_thread(work, "pick-preview")

    def _sky_text(self, key: str) -> str:
        m = next((d.metrics for d in self.selection.decisions if frame_key(d.metrics.file, self.folder) == key),
                 None) if self.selection else None
        if m is None or m.altitude is None:
            return ""
        illum = (m.moon_illum or 0) * 100
        if m.moon_alt is not None and m.moon_alt > 0:
            return ptr("gui.preview.sky_moon_up", alt=m.altitude, moon=m.moon_alt, illum=illum, sep=m.moon_sep or 0)
        return ptr("gui.preview.sky_moon_down", alt=m.altitude, illum=illum)

    def _clear_preview(self, text: str) -> None:
        self.preview_thumb.setPixmap(QPixmap())
        self.preview_thumb.setText(text)
        self.preview_closeup.setPixmap(QPixmap())
        self.preview_caption.setText("")

    def _show_preview(self) -> None:
        preview = self._previews.get(self._preview_key or "")
        if preview is None:
            return
        dpr = self.devicePixelRatioF()
        width = 400
        thumb = thumbnail(preview, int(width * dpr), int(280 * dpr))
        pm = pil_to_pixmap(thumb)
        pm.setDevicePixelRatio(dpr)
        self.preview_thumb.setText("")
        self.preview_thumb.setPixmap(pm)
        self._thumb_size = (thumb.width / dpr, thumb.height / dpr)
        cells = int(width / 2)
        cu = pil_to_pixmap(closeup(preview, *self._preview_center, cells, 2))
        self.preview_closeup.setPixmap(cu)
        self.preview_caption.setText(ptr("gui.preview.closeup"))

    def _move_closeup(self, pos) -> None:
        if self._preview_key not in self._previews or not hasattr(self, "_thumb_size"):
            return
        w, h = self._thumb_size
        x = (pos.x() - (self.preview_thumb.width() - w) / 2) / w
        y = (pos.y() - (self.preview_thumb.height() - h) / 2) / h
        self._preview_center = (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))
        self._show_preview()

    # ------------------------------------------------------------------ 搬檔

    def _move(self) -> None:
        sel, reject_dir, folder = self.selection, self._reject_dir(), self.folder
        if sel is None or reject_dir is None or folder is None:
            return
        recursive = bool(self.recursive.get())
        out, back = move_files(sel.decisions, reject_dir, recursive, dry_run=True)
        if not out and not back:
            info(self, ptr("gui.move.nothing"))
            return
        lines = []
        if out and recursive:
            counts: dict[str, int] = {}
            for target in out:
                sub = frame_key(target.parent.parent, folder)
                counts[sub] = counts.get(sub, 0) + 1
            shown = sorted(counts.items())[:12]
            lines.append("\n".join([ptr("gui.move.out_each", n=len(out))]
                                   + [ptr("gui.move.dir_line", dir=s if s != "." else ptr("group.top_folder"), n=n)
                                      for s, n in shown]
                                   + ([ptr("gui.move.more_dirs", n=len(counts) - len(shown))]
                                      if len(counts) > len(shown) else [])))
        elif out:
            lines.append(ptr("gui.move.out", n=len(out), dir=reject_dir))
        if back:
            lines.append(ptr("gui.move.back", n=len(back)))
        if not ask_yes_no(self, "\n\n".join(lines) + "\n\n" + ptr("gui.move.confirm")):
            return
        try:
            out, back = move_files(sel.decisions, reject_dir, recursive)
            write_csv(sel.decisions, folder / REPORT_NAME, root=folder)
        except Exception as exc:  # noqa: BLE001
            error(self, ptr("gui.move.error", error=exc))
            return
        n_out, n_back = len(out), len(back)

        def text() -> str:
            parts = []
            if n_out:
                parts.append(ptr("gui.move.done_out", n=n_out))
            if n_back:
                parts.append(ptr("gui.move.done_back", n=n_back))
            where = ptr("gui.move.where_each") if recursive else ptr("gui.move.where", dir=reject_dir)
            return ptr("gui.move.sep").join(parts) + where

        self.set_status(text)

    def _restore(self) -> None:
        reject_dir, folder = self._reject_dir(), self.folder
        if reject_dir is None or folder is None:
            return
        recursive = bool(self.recursive.get())
        n = len(restore_files(folder, reject_dir, recursive, dry_run=True))
        if n == 0:
            info(self, ptr("gui.restore.nothing"))
            return
        question = ptr("gui.restore.confirm_each", n=n) if recursive else ptr("gui.restore.confirm", dir=reject_dir,
                                                                                n=n)
        if not ask_yes_no(self, question):
            return
        try:
            restored = len(restore_files(folder, reject_dir, recursive))
        except Exception as exc:  # noqa: BLE001
            error(self, ptr("gui.restore.error", error=exc))
            return
        self.set_status(lambda: ptr("gui.restore.done", n=restored))

    # ------------------------------------------------------------------ 交棒

    def kept_paths(self) -> list[Path]:
        """保留的片現在的位置（不用先搬檔；已經被搬走的會找到現在的位置）。"""
        if self.selection is None:
            return []
        out = []
        for d in self.selection.decisions:
            if not d.keep:
                continue
            path = self._current_path(frame_key(d.metrics.file, self.folder)) if self.folder else Path(d.metrics.file)
            if path is not None:
                out.append(path)
        return out

    def send_to_stack(self) -> None:
        paths = self.kept_paths()
        if not paths:
            info(self, tr("send.nothing"))
            return
        self.shell.send_to_stack(paths)


__all__ = ["PickPage", "np"]
