"""疊圖（APU Photons 0.2 的核心）：加入檔案（light 與校正檔一起）→ 自動分類、分組、配對校正檔（可以改）
→ 試跑或疊圖 → 在「結果」逐組放大檢視 → 送去後製。

分組、配對、疊圖都直接呼叫 apu_photons（ingest、run），設定與 Photons 獨立版共用同一個設定檔。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from apu_photons import backend
from apu_photons import settings as photons_settings
from apu_photons.engine import Cancelled, Result, Settings, run
from apu_photons.i18n import get_language as photons_language
from apu_photons.i18n import tr as ptr
from apu_photons.imageio import IMAGE_SUFFIXES, TOP_DOWN, row_order
from apu_photons.ingest import SUB_KINDS, expand_inputs, filter_names, ingest, unused_text
from apu_photons.model import KINDS, UNKNOWN, Project
from apu_photons.preview import apply_stretch, stretch_params
from apu_photons.recipe import RecipeError, load_recipe
from astropy.io import fits
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMenu, QPlainTextEdit, QPushButton, QSplitter, QStackedLayout,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..i18n import tr
from ..theme import D, fonts
from ..ui.imageview import ImageView
from ..ui.var import Var
from ..ui.widgets import (Hero, Inspector, MetricRow, ParameterSlider, ParameterToggle, Segmented, ask_three, button,
                          error, info, label, short_path, warn)
from ..ui.worker import Bridge, run_thread
from .base import Page, Views

PREVIEW_FRAMES = 20
ALL_ALIGN = ""  # 輸出設定套用到「所有對齊組」

# 檔案清單欄位：(key, 翻譯代號, 寬度 px, 置右)
COLUMNS = [
    ("kind", "gui.col.kind", 76, False), ("filter", "gui.col.filter", 62, False), ("exp", "gui.col.exp", 56, True),
    ("gain", "gui.col.gain", 50, True), ("temp", "gui.col.temp", 52, True), ("night", "gui.col.night", 92, False),
    ("result", "gui.col.result", 62, False), ("fwhm", "gui.col.fwhm", 56, True), ("stars", "gui.col.stars", 52, True),
    ("weight", "gui.col.weight", 58, True), ("rejected", "gui.col.rejected", 68, True),
    ("reason", "gui.col.reason", 240, False),
]


def _fmt(v, nd: int = 0) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{f:.{nd}f}" if nd else f"{f:g}"


def load_master_display(path: Path) -> np.ndarray:
    """疊圖結果 → 拉伸好的顯示影像 (C, H, W)，方向照 ROWORDER 翻成由上往下（同 Photons 的放大檢視）。"""
    with fits.open(path, memmap=True) as hdul:
        data, header = hdul[0].data, hdul[0].header
        if row_order(header) != TOP_DOWN:
            data = data[..., ::-1, :]
        chans = np.asarray(data, np.float32)
    chans = chans if chans.ndim == 3 else chans[None]
    h, w = chans.shape[-2:]
    step = max(1, max(h, w) // 1000)
    return np.stack([apply_stretch(c, stretch_params(np.asarray(c[::step, ::step], np.float32))) for c in chans])


class StackPage(Page):
    key = "stack"

    def __init__(self, shell):
        super().__init__(shell)
        st = photons_settings.load_settings()
        p = st.get("params", {})
        self.files: list[Path] = []
        self.kinds: dict[str, str] = {}
        self.calib_overrides: dict[str, dict] = {}
        self.file_calib: dict[str, dict] = {}
        self.align_output: dict[str, dict] = {}
        self.filter_aliases: dict[str, str] = dict(st.get("filter_aliases", {}))
        self.plan: Project | None = None
        self.plan_warnings: list = []
        self._plan_token = 0
        self._planning = False
        self.result: Result | None = None
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.log_lines: list[str] = []
        self.gpu_name: str | None = None
        self._gpu_checked = False
        self._stage: tuple[str, int, int] | None = None
        self._stage_started = self._started = self._elapsed = 0.0
        self._reference = None
        self._loading_output = False
        self._expanded: dict[str, bool] = {}
        self._shown_path: Path | None = None
        self._image_token = 0
        self.bridge = Bridge(self._handle, self)

        self.view = Var("files")
        self.output_dir = Var("")
        self.target = Var("")
        self.rejection = Var(p.get("rejection", "winsorized"))
        self.low = Var(float(p.get("low", 4.0)))
        self.high = Var(float(p.get("high", 3.0)))
        self.weights = Var(bool(p.get("weights", True)))
        self.local_norm = Var(bool(p.get("local_norm", False)))
        self.dual_band = Var(bool(p.get("dual_band", True)))
        self.split_nights = Var(bool(p.get("split_nights", True)))
        self.flat_any_night = Var(bool(p.get("flat_any_night", False)))
        self.temp_tol = Var(float(p.get("temp_tolerance", 2.0)))
        self.global_output = {"scale": p.get("scale", "1"), "drizzle": p.get("drizzle", False),
                              "pixfrac": p.get("pixfrac", 0.9), "fill_holes": p.get("fill_holes", True),
                              "crop": p.get("crop", True), "min_coverage": p.get("min_coverage", 0.9)}
        self.align_target = Var(ALL_ALIGN)
        self.scale = Var(self.global_output["scale"])
        self.drizzle = Var(bool(self.global_output["drizzle"]))
        self.pixfrac = Var(float(self.global_output["pixfrac"]))
        self.fill_holes = Var(bool(self.global_output["fill_holes"]))
        self.crop = Var(bool(self.global_output["crop"]))
        self.min_cov = Var(float(self.global_output["min_coverage"]))
        self.merge = Var(False)
        self.grid = Var("")
        self.bits = Var(str(p.get("bits", "32")))
        self.gpu = Var(bool(p.get("gpu", True)))
        self.workers = Var(int(p.get("workers", backend.default_workers())))
        self.memory = Var(int(p.get("memory_gb", 2)))
        self.preview_n = Var(int(p.get("preview_n", PREVIEW_FRAMES)))
        self.list_mode = Var(st.get("list_mode", "groups"))
        self.result_view = Var("drizzle")
        self.result_group = Var(0)

        for var in (self.rejection, self.low, self.high, self.weights, self.local_norm, self.dual_band, self.bits,
                    self.gpu, self.workers, self.memory, self.preview_n):
            var.subscribe(lambda _v: self._params_changed())
        for var in (self.scale, self.drizzle, self.pixfrac, self.fill_holes, self.crop, self.min_cov):
            var.subscribe(lambda _v: self._output_changed())
        for var in (self.split_nights, self.flat_any_night, self.temp_tol, self.merge, self.grid):
            var.subscribe(lambda _v: (self._params_changed(), self._schedule_plan()))
        self.target.subscribe(lambda _v: self.title_changed.emit())
        self.align_target.subscribe(lambda _v: self._load_output_controls())
        self.result_view.subscribe(lambda _v: self._show_preview())
        self.result_group.subscribe(lambda _v: self._show_preview())
        self.list_mode.subscribe(lambda v: (photons_settings.save_settings(list_mode=v), self._fill_files()))
        self.view.subscribe(lambda v: self.views.show_key(v) if hasattr(self, "views") else None)
        self._status_fn = lambda: ptr("gui.status.start")
        run_thread(self._detect_gpu, "photons-gpu")

    # ------------------------------------------------------------------ 介面

    def build_ui(self) -> QWidget:
        self.views_seg = Segmented([("files", ptr("gui.tab.files")), ("calibration", ptr("gui.tab.calibration")),
                                    ("result", ptr("gui.tab.result")), ("log", ptr("gui.tab.log"))], self.view)
        self.add_btn = button(ptr("gui.btn.add_files"), tooltip=ptr("gui.btn.add_files.help", shortcut="Ctrl+O"),
                              on_click=self.ask_add_files)
        self.trial_btn = button(ptr("gui.btn.trial"), tooltip=ptr("gui.btn.trial.help"),
                                on_click=lambda: self._start(trial=True))
        self.stack_btn = button(ptr("gui.btn.stack"), tooltip=ptr("gui.btn.stack.help"), on_click=self._stack_or_stop)
        self.send_btn = button(tr("btn.send_process"), "prominent", tr("btn.send_process.help"), self.send_to_process)
        self.views = Views({"files": self._build_files(), "calibration": self._build_calibration(),
                            "result": self._build_result(), "log": self._build_log()})
        self.views.show_key(self.view.get())
        return self.layout([self.views_seg], [self.add_btn, self.trial_btn, self.stack_btn, self.send_btn],
                           self.views, self._build_inspector())

    def _toolbar(self) -> tuple[QWidget, QHBoxLayout]:
        bar = QWidget()
        bar.setStyleSheet(f"background: {D.base};")
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 6, 12, 6)
        h.setSpacing(6)
        return bar, h

    def _tree(self, first: str, columns: list[tuple[str, str, int, bool]]) -> QTreeWidget:
        t = QTreeWidget()
        t.setColumnCount(len(columns) + 1)
        t.setHeaderLabels([first] + [ptr(c[1]) for c in columns])
        t.setAlternatingRowColors(True)
        t.setUniformRowHeights(True)
        t.setRootIsDecorated(True)
        t.setIndentation(16)
        t.header().setStretchLastSection(True)
        t.header().setHighlightSections(False)
        for i, c in enumerate(columns, start=1):
            t.setColumnWidth(i, c[2])
        return t

    def _build_files(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar, h = self._toolbar()
        h.addWidget(Segmented([("groups", ptr("gui.files.by_group")), ("all", ptr("gui.files.all"))], self.list_mode))
        h.addStretch(1)
        self.recipe_btn = button(ptr("gui.btn.load_recipe"), on_click=self.ask_load_recipe)
        h.addWidget(self.recipe_btn)
        self.assign_btn = button(ptr("gui.btn.assign"), on_click=self._assign_calibration)
        h.addWidget(self.assign_btn)
        self.filters_btn = button(ptr("gui.btn.filters"), on_click=self._edit_filters)
        h.addWidget(self.filters_btn)
        self.kind_btn = QPushButton(ptr("gui.btn.set_kind"))
        menu = QMenu(self.kind_btn)
        for kind in KINDS:
            act = QAction(ptr(f"kind.{kind}"), menu)
            act.triggered.connect(lambda _c=False, k=kind: self._set_kind(k))
            menu.addAction(act)
        menu.addSeparator()
        act = QAction(ptr("gui.kind.auto"), menu)
        act.triggered.connect(lambda _c=False: self._set_kind(None))
        menu.addAction(act)
        self.kind_btn.setMenu(menu)
        h.addWidget(self.kind_btn)
        self.remove_btn = button(ptr("gui.btn.remove"), on_click=self._remove_files)
        h.addWidget(self.remove_btn)
        v.addWidget(bar)
        holder = QWidget()
        stack = QStackedLayout(holder)
        self.tree = self._tree(ptr("gui.col.file"), COLUMNS)
        self.tree.setColumnWidth(0, 300)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemExpanded.connect(lambda it: self._expanded.__setitem__(it.data(0, Qt.ItemDataRole.UserRole), True))
        self.tree.itemCollapsed.connect(lambda it: self._expanded.__setitem__(it.data(0, Qt.ItemDataRole.UserRole), False))
        self.empty = Hero(tr("step.stack"), tr("stack.welcome.sub"),
                          ptr("gui.empty.hint", shortcut="Ctrl+O") + "\n" + tr("drop.hint"))
        stack.addWidget(self.empty)
        stack.addWidget(self.tree)
        self.files_stack = stack
        v.addWidget(holder, 1)
        return w

    def _build_calibration(self) -> QWidget:
        split = QSplitter(Qt.Orientation.Vertical)
        upper = QWidget()
        uv = QVBoxLayout(upper)
        uv.setContentsMargins(0, 0, 0, 0)
        uv.setSpacing(0)
        bar, h = self._toolbar()
        h.addWidget(label(ptr("gui.cal.groups"), D.label, 9, bold=True))
        h.addStretch(1)
        uv.addWidget(bar)
        cols = [("n", "gui.col.count", 50, True), ("dark", "kind.dark", 190, False), ("bias", "kind.bias", 150, False),
                ("source", "gui.col.source", 70, False), ("flat", "kind.flat", 210, False),
                ("flat_sub", "gui.col.flat_sub", 170, False)]
        self.cal_tree = self._tree(ptr("gui.col.cal_group"), cols)
        self.cal_tree.setColumnWidth(0, 420)
        self.cal_tree.setRootIsDecorated(False)
        self.cal_tree.itemSelectionChanged.connect(self._fill_cal_editor)
        uv.addWidget(self.cal_tree, 1)
        editor = QWidget()
        editor.setStyleSheet(f"background: {D.card};")
        g = QGridLayout(editor)
        g.setContentsMargins(12, 8, 12, 10)
        g.setHorizontalSpacing(8)
        self.cal_editor_title = label("", D.secondary, 8.5)
        g.addWidget(self.cal_editor_title, 0, 0, 1, 5)
        self.cal_combos: dict[str, QComboBox] = {}
        for i, kind in enumerate(("dark", "bias", "flat", "flat_sub")):
            row, col = 1 + i // 2, (i % 2) * 2
            text = ptr("gui.cal.flat_sub") if kind == "flat_sub" else ptr(f"kind.{kind}")
            g.addWidget(label(text), row, col)
            cb = QComboBox()
            cb.activated.connect(lambda _i, k=kind: self._cal_combo_selected(k))
            g.addWidget(cb, row, col + 1)
            g.setColumnStretch(col + 1, 1)
            self.cal_combos[kind] = cb
        self.cal_reset_btn = button(ptr("gui.btn.auto"), on_click=self._cal_reset)
        g.addWidget(self.cal_reset_btn, 1, 4, 2, 1)
        uv.addWidget(editor)
        split.addWidget(upper)
        lower = QWidget()
        lv = QVBoxLayout(lower)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        bar, h = self._toolbar()
        h.addWidget(label(ptr("gui.cal.sets"), D.label, 9, bold=True))
        h.addStretch(1)
        lv.addWidget(bar)
        cols = [("kind", "gui.col.kind", 80, False), ("n", "gui.col.count", 54, True),
                ("master", "gui.col.master", 64, False), ("status", "gui.col.status", 420, False)]
        self.set_tree = self._tree(ptr("gui.col.cal_set"), cols)
        self.set_tree.setColumnWidth(0, 280)
        self.set_tree.setRootIsDecorated(False)
        self.set_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        lv.addWidget(self.set_tree, 1)
        split.addWidget(lower)
        split.setSizes([420, 220])
        return split

    def _build_result(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar, h = self._toolbar()
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(220)
        self.group_combo.activated.connect(lambda i: self.result_group.set(i))
        h.addWidget(self.group_combo)
        self.view_seg_holder = QHBoxLayout()
        h.addLayout(self.view_seg_holder)
        h.addWidget(button(ptr("gui.zoom.fit"), on_click=lambda: self.image.fit()))
        h.addWidget(button("100%", on_click=lambda: self.image.actual_size()))
        self.zoom_label = label("", D.label, mono=True)
        self.zoom_label.setFixedWidth(52)
        h.addWidget(self.zoom_label)
        h.addStretch(1)
        h.addWidget(label(ptr("gui.result.stretch_note"), D.secondary, 8.5))
        v.addWidget(bar)
        self.image = ImageView(ptr("gui.result.empty"))
        self.image.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z:.0%}"))
        v.addWidget(self.image, 1)
        self.qc_text = QPlainTextEdit()
        self.qc_text.setReadOnly(True)
        self.qc_text.setFixedHeight(120)
        v.addWidget(self.qc_text)
        return w

    def _build_log(self) -> QWidget:
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(fonts().mono(9))
        self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_text.setPlainText("\n".join(self.log_lines))
        return self.log_text

    def _build_inspector(self) -> Inspector:
        ins = Inspector()
        g = ins.add(self.group("result", ptr("gui.group.result"), ptr("gui.group.result.info")))
        self.metrics = {key: g.add(MetricRow(ptr(f"gui.metric.{key}")))
                        for key in ("frames", "integrated", "rejected", "outputs", "fwhm", "holes", "elapsed")}

        g = ins.add(self.group("calibration", ptr("gui.group.calibration"), ptr("gui.group.calibration.info")))
        self.cal_summary = g.add(label("", D.label, 8.5, wrap=True))
        g.add(ParameterSlider(ptr("gui.slider.temp_tol"), self.temp_tol, 0.5, 10, lambda v: f"±{v:.1f}°C", 0.5), 4)
        g.add(ParameterToggle(ptr("gui.toggle.split_nights"), self.split_nights))
        g.add(ParameterToggle(ptr("gui.toggle.flat_any_night"), self.flat_any_night))
        g.add_row(button(ptr("gui.btn.show_calibration"), on_click=lambda: self.view.set("calibration")))

        g = ins.add(self.group("integration", ptr("gui.group.integration"), ptr("gui.group.integration.info")))
        g.add(Segmented([("winsorized", "Winsorized"), ("sigma", "Sigma"), ("average", ptr("gui.rej.average")),
                         ("median", ptr("gui.rej.median"))], self.rejection, stretch=True))
        sec = lambda v: ptr("gui.value.sigma", v=v)  # noqa: E731
        self.low_slider = g.add(ParameterSlider(ptr("gui.slider.low"), self.low, 1.5, 8, sec, 0.1), 6)
        self.high_slider = g.add(ParameterSlider(ptr("gui.slider.high"), self.high, 1.5, 8, sec, 0.1))
        g.add(ParameterToggle(ptr("gui.toggle.weights"), self.weights))
        g.add(ParameterToggle(ptr("gui.toggle.local_norm"), self.local_norm))

        g = ins.add(self.group("output", ptr("gui.group.output"), ptr("gui.group.output.info")))
        self.align_row = QWidget()
        ar = QVBoxLayout(self.align_row)
        ar.setContentsMargins(0, 0, 0, 6)
        ar.addWidget(label(ptr("gui.output.apply_to"), D.secondary, 8.5))
        self.align_combo = QComboBox()
        self.align_combo.activated.connect(self._align_combo_selected)
        ar.addWidget(self.align_combo)
        g.add(self.align_row)
        g.add(label(ptr("gui.output.scale"), D.secondary, 8.5))
        g.add(Segmented([("0.5", "0.5×"), ("1", "1×"), ("2", "2×")], self.scale, stretch=True))
        self.drizzle_toggle = g.add(ParameterToggle(ptr("gui.toggle.drizzle"), self.drizzle), 4)
        self.pixfrac_slider = g.add(ParameterSlider(ptr("gui.slider.pixfrac"), self.pixfrac, 0.5, 1.0,
                                                    lambda v: f"{v:.2f}", 0.05))
        self.fill_toggle = g.add(ParameterToggle(ptr("gui.toggle.fill_holes"), self.fill_holes))
        g.add(ParameterToggle(ptr("gui.toggle.crop"), self.crop))
        self.cov_slider = g.add(ParameterSlider(ptr("gui.slider.min_coverage"), self.min_cov, 0.5, 1.0,
                                                lambda v: f"{v:.0%}", 0.05))
        self.merge_toggle = g.add(ParameterToggle(ptr("gui.toggle.merge"), self.merge))
        self.grid_row = QWidget()
        gr = QVBoxLayout(self.grid_row)
        gr.setContentsMargins(0, 0, 0, 4)
        gr.addWidget(label(ptr("gui.output.grid"), D.secondary, 8.5))
        self.grid_combo = QComboBox()
        self.grid_combo.activated.connect(lambda i: self.grid.set(self.grid_combo.itemText(i)))
        gr.addWidget(self.grid_combo)
        g.add(self.grid_row)
        g.add(ParameterToggle(ptr("gui.toggle.dual_band"), self.dual_band))
        row = QHBoxLayout()
        row.addWidget(label(ptr("gui.output.bits")), 1)
        row.addWidget(Segmented([("32", "32-bit"), ("16", "16-bit")], self.bits))
        g.lay.addLayout(row)
        g.add(label(ptr("gui.output.target"), D.secondary, 8.5), 8)
        self.target_edit = QLineEdit(self.target.get())
        self.target_edit.textChanged.connect(self.target.set)
        g.add(self.target_edit)
        g.add(label(ptr("gui.output.folder"), D.secondary, 8.5), 8)
        self.output_label = g.add(label("", D.label, 8.5, wrap=True))
        self.output_btn = button(ptr("gui.btn.change"), on_click=self._choose_output)
        self.reveal_btn = button(ptr("gui.btn.reveal"), on_click=self._reveal_output)
        g.add_row(self.output_btn, self.reveal_btn)

        g = ins.add(self.group("performance", ptr("gui.group.performance"), ptr("gui.group.performance.info")))
        self.gpu_toggle = g.add(ParameterToggle(ptr("gui.toggle.gpu"), self.gpu))
        self.gpu_label = g.add(label("", D.secondary, 8.5))
        g.add(ParameterSlider(ptr("gui.slider.workers"), self.workers, 1, max(2, os.cpu_count() or 2),
                              lambda v: f"{v:.0f}"), 4)
        g.add(ParameterSlider(ptr("gui.slider.memory"), self.memory, 1, 32, lambda v: f"{v:.0f} GB"))
        g.add(ParameterSlider(ptr("gui.slider.trial"), self.preview_n, 5, 60,
                              lambda v: ptr("gui.value.frames", n=int(v))))
        return ins

    # ------------------------------------------------------------------ 外殼

    def doc_title(self) -> str:
        return self.target.get().strip() or (self.plan.name if self.plan is not None else "")

    def rail_text(self) -> str:
        if self.is_busy() and self.progress is not None:
            return tr("rail.busy", p=round(self.progress * 100))
        if self.result is not None:
            return tr("rail.masters", n=len(self.result.outputs))
        if self.files:
            return tr("rail.files", n=len(self.files))
        return ""

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def close_page(self) -> None:
        self.cancel.set()
        self._plan_token += 1

    def ask_open(self) -> None:
        self.ask_add_files()

    def handle_drop(self, paths: list[Path]) -> None:
        recipes = [p for p in paths if p.name.endswith(".recipe.json")]
        if recipes:
            self._ask_recipe_mode(recipes[0])
        elif not self.is_busy():
            self.add_files(paths)

    def shortcut(self, name: str) -> None:
        if self.view.get() == "result":
            if name == "fit":
                self.image.fit()
            elif name == "actual":
                self.image.actual_size()

    # ------------------------------------------------------------------ 檔案

    def last_dir(self) -> str:
        return str(self.files[-1].parent) if self.files else photons_settings.load_settings().get("last_dir", "")

    def add_files(self, paths) -> None:
        """加入檔案（資料夾等同其中的影像，不往子資料夾找）；light 與校正檔都從這裡加入。"""
        new = [p for p in expand_inputs([Path(x) for x in paths]) if p not in self.files]
        if not new:
            return
        if not self.files:
            photons_settings.save_settings(last_dir=str(new[0].parent))
        self.files.extend(new)
        self._reference = None
        if not self.output_dir.get():
            base = new[0].parent.parent if new[0].parent.parent != new[0].parent else new[0].parent
            self.output_dir.set(str(base / "APU Photons"))
        self.result = None
        self._schedule_plan()
        self.refresh()

    def ask_add_files(self) -> None:
        if self.is_busy():
            return
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_SUFFIXES))
        paths, _ = QFileDialog.getOpenFileNames(self, ptr("gui.btn.add_files"), self.last_dir(),
                                                f"{ptr('gui.filetype.images')} ({exts});;* (*)")
        if paths:
            self.add_files(paths)

    def _selected_files(self) -> list[Path]:
        chosen: list[Path] = []

        def walk(item: QTreeWidgetItem) -> None:
            iid = item.data(0, Qt.ItemDataRole.UserRole) or ""
            if iid.startswith("f:"):
                chosen.append(Path(iid[2:]))
            for i in range(item.childCount()):
                walk(item.child(i))

        for item in self.tree.selectedItems():
            walk(item)
        out, seen = [], set()
        for p in chosen:
            if str(p) not in seen:
                seen.add(str(p))
                out.append(p)
        return out

    def _remove_files(self) -> None:
        if self.is_busy():
            return
        chosen = {str(p) for p in self._selected_files()}
        if not chosen:
            return
        self.files = [p for p in self.files if str(p) not in chosen]
        self._reference = None
        self.result = None
        self._schedule_plan()
        self.refresh()

    def _set_kind(self, kind: str | None) -> None:
        if self.is_busy():
            return
        for p in self._selected_files():
            if kind is None:
                self.kinds.pop(str(p), None)
            else:
                self.kinds[str(p)] = kind
        self.result = None
        self._schedule_plan()

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, ptr("gui.output.folder"), self.output_dir.get() or self.last_dir())
        if path:
            self.output_dir.set(path)
            self.refresh()

    def _reveal_output(self) -> None:
        folder = Path(self.output_dir.get())
        if folder.is_dir():
            if sys.platform == "win32":
                os.startfile(folder)  # noqa: S606
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", str(folder)], check=False)

    # ------------------------------------------------------------------ 濾鏡名稱、指定校正檔

    def _dialog(self, title: str, help_text: str) -> tuple[QDialog, QGridLayout]:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        g = QGridLayout(dlg)
        g.setContentsMargins(16, 14, 16, 14)
        g.setVerticalSpacing(6)
        g.addWidget(label(help_text, D.secondary, 8.5, wrap=True), 0, 0, 1, 2)
        return dlg, g

    def _dialog_buttons(self, dlg: QDialog, g: QGridLayout, row: int, extra: QPushButton | None = None) -> QPushButton:
        h = QHBoxLayout()
        h.addStretch(1)
        ok = button(tr("btn.ok"), "prominent", on_click=dlg.accept)
        if extra is not None:
            h.addWidget(extra)
        h.addWidget(ok)
        h.addWidget(button(tr("btn.cancel"), on_click=dlg.reject))
        g.addLayout(h, row, 0, 1, 2)
        return ok

    def _edit_filters(self) -> None:
        """濾鏡名稱：列出 header 裡出現過的寫法與自動歸併的結果，可以手動指定合併到哪個名稱。"""
        if self.plan is None:
            return
        names = filter_names(self.plan)
        spellings = sorted({s for v in names.values() for s in v} | set(self.filter_aliases))
        if not spellings:
            info(self, ptr("gui.filters.none"))
            return
        dlg, g = self._dialog(ptr("gui.filters.title"), ptr("gui.filters.help"))
        g.addWidget(label(ptr("gui.filters.header"), bold=True), 1, 0)
        g.addWidget(label(ptr("gui.filters.target"), bold=True), 1, 1)
        auto = ptr("gui.filters.auto")
        combos = {}
        for i, sp in enumerate(spellings, start=2):
            g.addWidget(label(sp), i, 0)
            cb = QComboBox()
            cb.setEditable(True)
            cb.addItems([auto] + sorted(names))
            cb.setCurrentText(self.filter_aliases.get(sp, auto))
            g.addWidget(cb, i, 1)
            combos[sp] = cb
        self._dialog_buttons(dlg, g, len(spellings) + 2)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        aliases = {}
        for sp, cb in combos.items():
            v = cb.currentText().strip()
            if v and v != auto and v != sp:
                aliases[sp] = v
        self.filter_aliases = aliases
        photons_settings.save_settings(filter_aliases=aliases)  # 記住：下次相同名稱自動套用
        self.result = None
        self._schedule_plan()

    def _assign_calibration(self) -> None:
        """「指定校正檔…」：選取的 light 改用指定的 dark／bias／flat／flat 扣除。"""
        if self.plan is None or self.is_busy():
            return
        lights = {str(f.path) for f in self.plan.lights}
        paths = [p for p in self._selected_files() if str(p) in lights]
        if not paths:
            info(self, ptr("gui.assign.no_lights"))
            return
        sets = self.plan.calibration_sets
        auto, none = ptr("gui.assign.auto"), ptr("gui.cal.none_option")
        current = [self.file_calib.get(str(p), {}) for p in paths]
        same = current[0] if all(c == current[0] for c in current) else {}
        dlg, g = self._dialog(ptr("gui.assign.title"), ptr("gui.assign.help", n=len(paths)))
        combos = {}
        for i, kind in enumerate(("dark", "bias", "flat", "flat_sub"), start=1):
            allowed = SUB_KINDS if kind == "flat_sub" else (kind,)
            ids = [sid for sid, cs in sets.items() if cs.kind in allowed]
            g.addWidget(label(ptr("gui.cal.flat_sub") if kind == "flat_sub" else ptr(f"kind.{kind}")), i, 0)
            cb = QComboBox()
            cb.addItems([auto, none] + ids)
            cb.setMinimumWidth(320)
            value = same.get(kind, "auto")
            cb.setCurrentText(auto if value == "auto" else (none if value is None else value))
            g.addWidget(cb, i, 1)
            combos[kind] = cb
        cleared = {"v": False}
        clear = button(ptr("gui.assign.clear"))
        clear.clicked.connect(lambda: (cleared.__setitem__("v", True), dlg.accept()))
        self._dialog_buttons(dlg, g, 6, clear)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        assignment = None
        if not cleared["v"]:
            assignment = {}
            for kind, cb in combos.items():
                v = cb.currentText()
                if v != auto:
                    assignment[kind] = None if v == none else v
        for p in paths:
            if assignment:
                self.file_calib[str(p)] = dict(assignment)
            else:
                self.file_calib.pop(str(p), None)
        self.result = None
        self._schedule_plan()

    # ------------------------------------------------------------------ 分組（背景讀 header）

    def _merge_trains(self) -> list[list[str]]:
        if not self.merge.get() or self.plan is None or len(self.plan.trains) < 2:
            return []
        return [list(self.plan.trains)]

    def _schedule_plan(self) -> None:
        """檔案或分組設定改了：在背景重新分類、分組、配對（RAW 要讀整個檔案才有 header）。"""
        self._plan_token += 1
        token = self._plan_token
        if not self.files:
            self.plan, self.plan_warnings, self._planning = None, [], False
            self.refresh()
            return
        self._planning = True
        files = list(self.files)
        args = dict(kinds=dict(self.kinds), filter_aliases=dict(self.filter_aliases),
                    temp_tolerance=float(self.temp_tol.get()), split_nights=bool(self.split_nights.get()),
                    flat_any_night=bool(self.flat_any_night.get()),
                    overrides=json.loads(json.dumps(self.calib_overrides)), merge_trains=self._merge_trains(),
                    file_calib=json.loads(json.dumps(self.file_calib)), name=None)
        post = self.bridge.post

        def work() -> None:
            warnings: list = []
            try:
                project = ingest(files, warnings, **args)
                post(("plan", token, project, warnings))
            except Exception as exc:  # noqa: BLE001
                post(("plan", token, None, [str(exc)]))

        run_thread(work, "photons-plan")
        self.set_status(lambda: ptr("gui.status.reading"))

    # ------------------------------------------------------------------ 校正配對

    def _fill_cal_editor(self) -> None:
        items = self.cal_tree.selectedItems()
        project = self.plan
        gk = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
        if project is None or gk not in project.calib_groups:
            self.cal_editor_title.setText(ptr("gui.cal.select"))
            for cb in self.cal_combos.values():
                cb.clear()
                cb.setEnabled(False)
            self.cal_reset_btn.setEnabled(False)
            return
        g = project.calib_groups[gk]
        self.cal_editor_title.setText(gk)
        none = ptr("gui.cal.none_option")
        master_flat = bool(g["flat"]) and project.calibration_sets[g["flat"]].is_master
        for kind, cb in self.cal_combos.items():
            allowed = SUB_KINDS if kind == "flat_sub" else (kind,)
            ids = [sid for sid, s in project.calibration_sets.items() if s.kind in allowed]
            cb.blockSignals(True)
            cb.clear()
            cb.addItems([none] + ids)
            cb.setCurrentText(g.get(kind) or none)
            if kind == "flat_sub" and master_flat:
                cb.addItem(ptr("gui.cal.flat_sub_master"))
                cb.setCurrentText(ptr("gui.cal.flat_sub_master"))
            cb.blockSignals(False)
            cb.setEnabled(not self.is_busy() and not (kind == "flat_sub" and (master_flat or not g["flat"])))
        self.cal_reset_btn.setEnabled(gk in self.calib_overrides and not self.is_busy())

    def _cal_combo_selected(self, kind: str) -> None:
        items = self.cal_tree.selectedItems()
        if not items:
            return
        gk = items[0].data(0, Qt.ItemDataRole.UserRole)
        value = self.cal_combos[kind].currentText()
        self.calib_overrides.setdefault(gk, {})[kind] = None if value == ptr("gui.cal.none_option") else value
        self.result = None
        self._schedule_plan()

    def _cal_reset(self) -> None:
        items = self.cal_tree.selectedItems()
        if items:
            self.calib_overrides.pop(items[0].data(0, Qt.ItemDataRole.UserRole), None)
            self.result = None
            self._schedule_plan()

    # ------------------------------------------------------------------ 設定 → 引擎

    @staticmethod
    def _engine_output(values: dict) -> dict:
        scale = values["scale"]
        drizzle = 2 if scale == "2" else (1 if scale == "1" and values["drizzle"] else 0)
        return {"drizzle": drizzle, "downsample": 0.5 if scale == "0.5" else 1.0,
                "pixfrac": round(float(values["pixfrac"]), 2), "fill_holes": bool(values["fill_holes"]),
                "crop_common": bool(values["crop"]), "crop_min_coverage": round(float(values["min_coverage"]), 2)}

    def settings(self, trial: bool = False) -> Settings:
        g = self._engine_output(self.global_output)
        align_output = {aid: self._engine_output({**self.global_output, **vals})
                        for aid, vals in self.align_output.items()}
        merge = self._merge_trains()
        grid = {" + ".join(merge[0]): self.grid.get()} if merge and self.grid.get() in merge[0] else {}
        return Settings(
            rejection=self.rejection.get(), low=round(float(self.low.get()), 2), high=round(float(self.high.get()), 2),
            weighting="snr2_over_fwhm2" if self.weights.get() else "none", downsample=g["downsample"],
            normalization="local" if self.local_norm.get() else "global", dual_band=bool(self.dual_band.get()),
            crop_common=g["crop_common"], crop_min_coverage=g["crop_min_coverage"],
            output_bits=int(self.bits.get()), memory_mb=int(self.memory.get()) * 1024,
            split_nights=bool(self.split_nights.get()), flat_any_night=bool(self.flat_any_night.get()),
            preview=int(self.preview_n.get()) if trial else 0, drizzle=g["drizzle"], pixfrac=g["pixfrac"],
            fill_holes=g["fill_holes"], gpu="auto" if self.gpu.get() else "cpu", workers=int(self.workers.get()),
            target=self.target.get().strip() or None, temp_tolerance=float(self.temp_tol.get()),
            kinds=dict(self.kinds), filter_aliases=dict(self.filter_aliases),
            calib_overrides=json.loads(json.dumps(self.calib_overrides)), merge_trains=merge,
            file_calib=json.loads(json.dumps(self.file_calib)), grid_train=grid, align_output=align_output)

    def _params_changed(self) -> None:
        self._update_output_rules()
        photons_settings.save_settings(params={
            "rejection": self.rejection.get(), "low": float(self.low.get()), "high": float(self.high.get()),
            "weights": bool(self.weights.get()), "local_norm": bool(self.local_norm.get()),
            "dual_band": bool(self.dual_band.get()), "split_nights": bool(self.split_nights.get()),
            "flat_any_night": bool(self.flat_any_night.get()), "temp_tolerance": float(self.temp_tol.get()),
            **dict(self.global_output), "bits": self.bits.get(), "gpu": bool(self.gpu.get()),
            "workers": int(self.workers.get()), "memory_gb": int(self.memory.get()),
            "preview_n": int(self.preview_n.get())})

    def _output_values(self) -> dict:
        return {"scale": self.scale.get(), "drizzle": bool(self.drizzle.get()), "pixfrac": float(self.pixfrac.get()),
                "fill_holes": bool(self.fill_holes.get()), "crop": bool(self.crop.get()),
                "min_coverage": float(self.min_cov.get())}

    def _output_changed(self) -> None:
        """輸出控制項改了：存到目前選的對齊組（或所有對齊組的預設）。"""
        if self._loading_output:
            return
        self._update_output_rules()
        target = self.align_target.get()
        if target == ALL_ALIGN:
            self.global_output = self._output_values()
            self._params_changed()
        else:
            self.align_output[target] = self._output_values()

    def _load_output_controls(self) -> None:
        target = self.align_target.get()
        values = self.global_output if target == ALL_ALIGN else {**self.global_output,
                                                                **self.align_output.get(target, {})}
        self._loading_output = True
        try:
            self.scale.set(values["scale"])
            self.drizzle.set(bool(values["drizzle"]))
            self.pixfrac.set(float(values["pixfrac"]))
            self.fill_holes.set(bool(values["fill_holes"]))
            self.crop.set(bool(values["crop"]))
            self.min_cov.set(float(values["min_coverage"]))
        finally:
            self._loading_output = False
        self._update_output_rules()

    def _align_combo_selected(self, i: int) -> None:
        ids = [ALL_ALIGN] + ([a.id for a in self.plan.align_groups] if self.plan else [])
        self.align_target.set(ids[i] if 0 <= i < len(ids) else ALL_ALIGN)

    def _update_output_rules(self) -> None:
        """2× 一定要 drizzle；0.5× 不能 drizzle；1× 可選。rejection 是平均／中位數時沒有 σ。"""
        scale = self.scale.get()
        if scale == "2" and not self.drizzle.get():
            self.drizzle.set(True)
        elif scale == "0.5" and self.drizzle.get():
            self.drizzle.set(False)
        if not hasattr(self, "drizzle_toggle"):
            return
        try:
            self.drizzle_toggle.set_enabled(scale == "1")
            on = bool(self.drizzle.get())
            self.pixfrac_slider.set_enabled(on)
            self.fill_toggle.set_enabled(on)
            self.cov_slider.set_enabled(bool(self.crop.get()))
            clip = self.rejection.get() in ("winsorized", "sigma")
            self.low_slider.set_enabled(clip)
            self.high_slider.set_enabled(clip)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ Recipe

    def ask_load_recipe(self) -> None:
        if self.is_busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, ptr("gui.recipe.title"), self.output_dir.get() or self.last_dir(),
                                              "Recipe (*.recipe.json *.json);;* (*)")
        if path:
            self._ask_recipe_mode(Path(path))

    def _ask_recipe_mode(self, path: Path) -> None:
        choice = ask_three(self, ptr("gui.recipe.ask"), [tr("btn.yes"), tr("btn.no"), tr("btn.cancel")])
        if choice is not None:
            self.load_recipe_file(path, reproduce=choice == 0)

    def load_recipe_file(self, path: Path, reproduce: bool) -> list[str]:
        try:
            rec = load_recipe(path, reproduce=reproduce, base=self.settings())
        except RecipeError as exc:
            error(self, str(exc))
            return []
        s = rec.settings
        self.rejection.set(s.rejection)
        self.low.set(float(s.low))
        self.high.set(float(s.high))
        self.weights.set(s.weighting != "none")
        self.local_norm.set(getattr(s, "normalization", "global") == "local")
        self.dual_band.set(bool(getattr(s, "dual_band", True)))
        self.split_nights.set(bool(s.split_nights))
        self.flat_any_night.set(bool(s.flat_any_night))
        self.temp_tol.set(float(s.temp_tolerance))
        self.bits.set(str(s.output_bits))
        scale = "2" if s.drizzle == 2 else ("0.5" if s.downsample == 0.5 else "1")
        self.global_output = {"scale": scale, "drizzle": s.drizzle >= 1, "pixfrac": s.pixfrac,
                              "fill_holes": s.fill_holes, "crop": s.crop_common, "min_coverage": s.crop_min_coverage}
        self.align_output = {}
        for aid, o in s.align_output.items():
            sc = "2" if o.get("drizzle") == 2 else ("0.5" if o.get("downsample") == 0.5 else "1")
            self.align_output[aid] = {"scale": sc, "drizzle": (o.get("drizzle") or 0) >= 1,
                                      "pixfrac": o.get("pixfrac", s.pixfrac),
                                      "fill_holes": o.get("fill_holes", s.fill_holes),
                                      "crop": o.get("crop_common", s.crop_common),
                                      "min_coverage": o.get("crop_min_coverage", s.crop_min_coverage)}
        self.align_target.set(ALL_ALIGN)
        self._load_output_controls()
        self.filter_aliases = dict(s.filter_aliases)
        self.merge.set(bool(s.merge_trains))
        if s.grid_train:
            self.grid.set(next(iter(s.grid_train.values())))
        if reproduce:
            self.files = list(rec.files)
            self.kinds = dict(s.kinds)
            self.calib_overrides = dict(s.calib_overrides)
            self.file_calib = dict(s.file_calib)
            self.target.set(s.target or "")
            if hasattr(self, "target_edit"):
                self.target_edit.setText(s.target or "")
            self._reference = s.reference
        self.result = None
        self._schedule_plan()
        self.refresh()
        for w in rec.warnings:
            self._append_log(f"⚠ {w}")
        if rec.warnings:
            warn(self, "\n".join(str(w) for w in rec.warnings[:12]))
        return [str(w) for w in rec.warnings]

    # ------------------------------------------------------------------ 執行

    def _detect_gpu(self) -> None:
        be = backend.select("auto")
        self.bridge.post(("gpu", be.device if be.gpu else None))

    def _stack_or_stop(self) -> None:
        if self.is_busy():
            self.cancel.set()
            self.set_status(lambda: ptr("gui.status.stopping"))
        else:
            self._start(trial=False)

    def _start(self, trial: bool) -> None:
        if self.is_busy() or not self.files:
            return
        if not self.output_dir.get():
            self._choose_output()
            if not self.output_dir.get():
                return
        self.cancel.clear()
        self.result = None
        self._shown_path = None
        self.image.set_image(None)
        self.log_lines = []
        self.log_text.setPlainText("")
        self._started = time.time()
        self._stage = None
        settings = self.settings(trial)
        settings.reference = self._reference
        out = Path(self.output_dir.get())
        out = out / "trial" if trial else out
        files, post, cancel = list(self.files), self.bridge.post, self.cancel

        def work() -> None:
            try:
                res = run(files, out, settings, echo=lambda line: post(("log", line)),
                          progress=lambda stage, d, t: post(("progress", stage, d, t)), cancel=cancel)
                post(("done", res))
            except Cancelled:
                post(("cancelled",))
            except Exception as exc:  # noqa: BLE001  顯示給使用者，不讓執行緒默默死掉
                post(("error", str(exc) or type(exc).__name__, traceback.format_exc()))

        self.progress = 0.0
        self.worker = run_thread(work, "photons-run")
        self.set_status(lambda: ptr("gui.status.starting"), 0.0)
        self.view.set("log")
        self.refresh()

    def _append_log(self, line: str) -> None:
        self.log_lines.append(line)
        try:
            self.log_text.appendPlainText(line)
        except RuntimeError:
            pass

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "log":
            self._append_log(event[1])
        elif kind == "plan":
            _, token, project, warnings = event
            if token != self._plan_token:
                return  # 已經有更新的分組在跑
            self._planning = False
            self.plan, self.plan_warnings = project, warnings
            if not self.is_busy():
                self.set_status(lambda: ptr("gui.status.start"))
            self.refresh()
            self.title_changed.emit()
        elif kind == "progress":
            _, stage, done, total = event
            if self._stage is None or self._stage[0] != stage:
                self._stage_started = time.time()
            self._stage = (stage, done, total)
            frac = self._overall_progress()
            self.set_status(self._progress_text, frac)
            self.rail_changed.emit()
        elif kind == "gpu":
            self._gpu_checked, self.gpu_name = True, event[1]
            self._update_gpu_view()
        elif kind == "done":
            self.worker = None
            self.progress = None
            self.result = event[1]
            self.result_view.set("drizzle" if any(o.stack for o in self.result.outputs) else "stack")
            self.result_group.set(0)
            elapsed = self._elapsed = time.time() - self._started
            self.set_status(lambda: ptr("gui.status.done", seconds=elapsed), None)
            self.refresh()
            self.view.set("result")
        elif kind == "cancelled":
            self.worker = None
            self.set_status(lambda: ptr("gui.status.cancelled"), None)
            self.refresh()
        elif kind == "error":
            self.worker = None
            self._append_log(event[2])
            message = event[1]
            self.set_status(lambda: ptr("gui.status.error", message=message), None)
            self.refresh()
            error(self, message)
        elif kind == "image":
            _, token, path, display = event
            if token == self._image_token:
                self.image.set_image(display, keep_view=True)
        elif kind == "image_error":
            if event[1] == self._image_token:
                self.image.set_image(None)
                self.image.empty_text = str(event[2])

    def _overall_progress(self) -> float:
        """整體進度：七個階段各占一段，階段內照完成比例。"""
        from apu_photons.engine import STAGES
        if self._stage is None:
            return 0.0
        stage, done, total = self._stage
        i = STAGES.index(stage) if stage in STAGES else 0
        within = done / total if total else 0.0
        return min(1.0, (i + within) / len(STAGES))

    def _progress_text(self) -> str:
        if self._stage is None:
            return ptr("gui.status.starting")
        stage, done, total = self._stage
        name = ptr(f"gui.stage.{stage}")
        if not total:
            return name
        text = ptr("gui.status.progress", stage=name, done=done, total=total)
        spent = time.time() - self._stage_started
        if 0 < done < total and spent > 3:
            text += ptr("gui.status.eta", seconds=spent / done * (total - done))
        return text

    # ------------------------------------------------------------------ 畫面更新

    def _update_gpu_view(self) -> None:
        if not hasattr(self, "gpu_label"):
            return
        try:
            if not self._gpu_checked:
                self.gpu_label.setText(ptr("gui.gpu.checking"))
                self.gpu_toggle.set_enabled(False)
            elif self.gpu_name:
                self.gpu_label.setText(self.gpu_name)
                self.gpu_toggle.set_enabled(True)
            else:
                self.gpu_label.setText(ptr("gui.gpu.none"))
                self.gpu_toggle.set_enabled(False, forced_off=True)
        except RuntimeError:
            pass

    def refresh(self) -> None:
        busy = self.is_busy()
        self.stack_btn.setText(ptr("gui.btn.stop") if busy else ptr("gui.btn.stack"))
        ready = bool(self.plan is not None and self.plan.lights and not self._planning)
        self.stack_btn.setEnabled(ready or busy)
        self.trial_btn.setEnabled(ready and not busy)
        self.send_btn.setEnabled(not busy and self.result is not None and bool(self.result.outputs))
        self.add_btn.setEnabled(not busy)
        self.output_btn.setEnabled(not busy)
        self.recipe_btn.setEnabled(not busy)
        for b in (self.remove_btn, self.filters_btn, self.assign_btn, self.kind_btn):
            b.setEnabled(not busy and bool(self.files))
        out = self.output_dir.get()
        self.reveal_btn.setEnabled(bool(out) and Path(out).is_dir())
        self.output_label.setText(short_path(out, 46) if out else ptr("gui.output.unset"))
        self.has_result = self.result is not None
        self._update_output_rules()
        self._update_gpu_view()
        self._fill_align_controls()
        self._fill_files()
        self._fill_calibration()
        self._fill_metrics()
        self._fill_result()
        p = self.plan
        if p is not None:
            n_cal = sum(len(s.frames) for s in p.calibration_sets.values())
            self.set_summary(lambda: ptr("gui.summary2", files=len(p.files), lights=len(p.lights), cal=n_cal,
                                         unknown=len(p.unknown)))
        else:
            n = len(self.files)
            self.set_summary(lambda: ptr("gui.summary_files", files=n) if n else "")
        self.rail_changed.emit()

    def _fill_align_controls(self) -> None:
        p = self.plan
        aligns = p.align_groups if p is not None else []
        if len(aligns) > 1:
            self.align_combo.clear()
            self.align_combo.addItems([ptr("gui.output.all_align")] + [a.id for a in aligns])
            ids = [ALL_ALIGN] + [a.id for a in aligns]
            target = self.align_target.get()
            self.align_combo.setCurrentIndex(ids.index(target) if target in ids else 0)
            self.align_row.show()
        else:
            self.align_row.hide()
            if self.align_target.get() != ALL_ALIGN:
                self.align_target.set(ALL_ALIGN)
        trains = list(p.trains) if p is not None else []
        if len(trains) > 1:
            self.merge_toggle.show()
            self.grid_combo.clear()
            self.grid_combo.addItems(trains)
            if self.grid.get() in trains:
                self.grid_combo.setCurrentText(self.grid.get())
            self.grid_row.setVisible(bool(self.merge.get()))
        else:
            self.merge_toggle.hide()
            self.grid_row.hide()

    def _shown_project(self) -> Project | None:
        return self.result.project if self.result is not None else self.plan

    def _file_values(self, f, accepted_n: int, refs: set) -> tuple[list[str], str | None]:
        kind = ptr(f"kind.{f.kind}") + (" ★" if f.is_master else "")
        exp, gain = _fmt(f.exposure), _fmt(f.gain)
        temp = _fmt(f.temp, 1) if f.temp is not None else ""
        night = f.night if f.night not in (None, "unknown", "all") else ""
        if self.result is None or f.kind != "light":
            color = D.beta if f.kind == UNKNOWN else (D.reject if not f.accepted else None)
            reason = ptr(f"reason.{f.reject_reason}") if f.reject_reason else (
                ptr("gui.kind.unknown_hint") if f.kind == UNKNOWN else "")
            return [kind, f.filter or "", exp, gain, temp, night, "", "", "", "", "", reason], color
        ok = f.accepted
        fwhm = f"{f.stars.fwhm:.2f}" if f.stars is not None and f.stars.fwhm == f.stars.fwhm else ""
        stars = str(len(f.stars)) if f.stars is not None else ""
        weight = f"{f.weight * accepted_n:.2f}" if ok and f.weight else ""
        rej = f"{f.rejected_pixel_fraction:.2%}" if ok and f.rejected_pixel_fraction is not None else ""
        reason = ptr(f"reason.{f.reject_reason}") if f.reject_reason else ""
        if id(f) in refs:
            reason = ptr("gui.reference")
        skipped = f.reject_reason == "not_in_preview"
        result = ptr("gui.result.ok") if ok else ("—" if skipped else ptr("gui.result.rejected"))
        color = None if ok else (D.secondary if skipped else D.reject)
        return [kind, f.filter or "", exp, gain, temp, night, result, fwhm, stars, weight, rej, reason], color

    def _item(self, parent, iid: str, text: str, values: list[str], color: str | None,
              expanded: bool | None = None) -> QTreeWidgetItem:
        it = QTreeWidgetItem([text] + values)
        it.setData(0, Qt.ItemDataRole.UserRole, iid)
        if color:
            for c in range(len(values) + 1):
                it.setForeground(c, QColor(color))
        for c, col in enumerate(COLUMNS, start=1):
            if col[3]:
                it.setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if isinstance(parent, QTreeWidget):
            parent.addTopLevelItem(it)
        else:
            parent.addChild(it)
        if expanded is not None:
            it.setExpanded(self._expanded.get(iid, expanded))
        return it

    def _fill_files(self) -> None:
        if not hasattr(self, "tree"):
            return
        selected = {it.data(0, Qt.ItemDataRole.UserRole) for it in self.tree.selectedItems()}
        self.tree.setUpdatesEnabled(False)
        self.tree.clear()
        try:
            if not self.files:
                self.files_stack.setCurrentIndex(0)
                return
            self.files_stack.setCurrentIndex(1)
            p = self._shown_project()
            blank = [""] * len(COLUMNS)
            if p is None:
                for path in self.files:
                    self._item(self.tree, f"f:{path}", path.name, ["…"] + blank[1:], D.secondary)
                return
            refs = {id(a.reference) for a in p.align_groups if a.reference is not None}
            accepted_n = len(p.accepted) or 1

            def add(parent, f) -> None:
                values, color = self._file_values(f, accepted_n, refs)
                it = self._item(parent, f"f:{f.path}", f.name, values, color)
                if it.data(0, Qt.ItemDataRole.UserRole) in selected:
                    it.setSelected(True)

            if self.list_mode.get() == "all":
                for f in sorted(p.files, key=lambda f: f.name.casefold()):
                    add(self.tree, f)
                return
            placed: set[int] = set()
            group_color = D.accent
            for a in p.align_groups:
                node = self._item(self.tree, f"a:{a.id}", a.id, blank, group_color, True)
                for i, g in enumerate(a.groups):
                    gnode = self._item(node, f"i:{a.id}:{i}", ptr("gui.files.group", filter=g.label, n=len(g.frames)),
                                       blank, group_color, True)
                    for f in g.frames:
                        add(gnode, f)
                        placed.add(id(f))
            if p.calibration_sets:
                cnode = self._item(self.tree, "c:", ptr("gui.files.calibration"), blank, group_color, True)
                for sid, s in p.calibration_sets.items():
                    text = f"{sid}（{len(s.frames)}）" if photons_language() == "zh" else f"{sid} ({len(s.frames)})"
                    snode = self._item(cnode, f"s:{sid}", text, blank, group_color if s.used_by else D.secondary,
                                       False)
                    for f in s.frames:
                        add(snode, f)
                        placed.add(id(f))
            rest = [f for f in p.files if id(f) not in placed]
            if rest:
                unk = any(f.kind == UNKNOWN for f in rest)
                node = self._item(self.tree, "u:", ptr("gui.files.unused", n=len(rest)), blank,
                                  D.beta if unk else D.secondary, True)
                for f in rest:
                    add(node, f)
        finally:
            self.tree.setUpdatesEnabled(True)

    def _fill_calibration(self) -> None:
        if not hasattr(self, "cal_tree"):
            return
        sel = [it.data(0, Qt.ItemDataRole.UserRole) for it in self.cal_tree.selectedItems()]
        self.cal_tree.clear()
        self.set_tree.clear()
        p = self.plan
        if p is None:
            self.cal_summary.setText(ptr("gui.cal.empty"))
            self.cal_summary.setStyleSheet(f"color: {D.secondary}; background: transparent;")
            self._fill_cal_editor()
            return
        missing_flat = 0
        for gk, g in p.calib_groups.items():
            src = (ptr("gui.cal.manual") if g.get("manual") else
                   ptr("gui.cal.source.user") if g["source"] == "user" else ptr("gui.cal.source.auto"))
            none = ptr("gui.cal.none_short")
            flat = g["flat"] or (ptr("gui.cal.flat_other") if g["flat_other_nights"] else none)
            if not g["flat"]:
                missing_flat += 1
            bias = g["bias"] or (ptr("gui.cal.bias_not_needed") if g["dark"] else none)
            color = D.accent if g["source"] == "user" else (D.beta if not g["flat"] or not g["dark"] else None)
            sub = g["flat_sub"] or (none if g["flat"] else "")
            it = QTreeWidgetItem([gk, str(len(g["frames"])), g["dark"] or none, bias, src, flat, sub])
            it.setData(0, Qt.ItemDataRole.UserRole, gk)
            if color:
                for c in range(7):
                    it.setForeground(c, QColor(color))
            self.cal_tree.addTopLevelItem(it)
            if gk in sel:
                it.setSelected(True)
        for sid, s in p.calibration_sets.items():
            status = ptr("gui.cal.used", n=len(s.used_by)) if s.used_by else str(unused_text(s))
            it = QTreeWidgetItem([sid, ptr(f"kind.{s.kind}"), str(len(s.frames)), "★" if s.is_master else "", status])
            if not s.used_by:
                for c in range(5):
                    it.setForeground(c, QColor(D.secondary))
            self.set_tree.addTopLevelItem(it)
        lines = [ptr("gui.cal.summary", groups=len(p.calib_groups), sets=len(p.calibration_sets))]
        if missing_flat:
            lines.append(ptr("gui.cal.missing_flat", n=missing_flat))
        unused = sum(1 for s in p.calibration_sets.values() if not s.used_by)
        if unused:
            lines.append(ptr("gui.cal.unused_sets", n=unused))
        self.cal_summary.setText("\n".join(lines))
        self.cal_summary.setStyleSheet(f"color: {D.label}; background: transparent;")
        self._fill_cal_editor()

    def _fill_metrics(self) -> None:
        m = self.metrics
        if self.result is None:
            for row in m.values():
                row.set("—")
            return
        q = self.result.qc
        m["frames"].set(str(q["frames_total"]))
        m["integrated"].set(str(q["frames_integrated"]))
        m["rejected"].set(str(sum(v for k, v in q["rejected_by_reason"].items() if k != "not_in_preview")))
        m["outputs"].set(str(len(self.result.outputs)))
        fw = [g["fwhm_px"]["median"] for g in q["groups"] if g.get("fwhm_px")]
        m["fwhm"].set(" / ".join(f"{v:.2f}" for v in fw) + " px" if fw else "—")
        holes = [v for g in q["groups"] for v in g.get("drizzle", {}).get("holes_fraction", [])]
        m["holes"].set(f"{max(holes):.1%}" if holes else "—")
        m["elapsed"].set(ptr("gui.value.seconds", v=self._elapsed) if self._elapsed else "—")

    def _result_label(self, o) -> str:
        multi = len({x.align for x in self.result.outputs}) > 1
        name = o.filter or ptr("gui.result.no_filter")
        return f"{o.align} / {name}" if multi else name

    def _fill_result(self) -> None:
        while self.view_seg_holder.count():
            w = self.view_seg_holder.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        outs = self.result.outputs if self.result is not None else []
        self.group_combo.clear()
        self.group_combo.addItems([self._result_label(o) for o in outs])
        if outs:
            self.group_combo.setCurrentIndex(min(max(int(self.result_group.get()), 0), len(outs) - 1))
        self.group_combo.setEnabled(bool(outs))
        if any(o.stack for o in outs):
            self.view_seg_holder.addWidget(Segmented([("drizzle", "Drizzle"), ("stack", ptr("gui.result.stack"))],
                                                     self.result_view))
        lines = []
        if self.result is not None:
            q = self.result.qc
            lines = [ptr("gui.qc.output", path=str(self.result.output_dir))]
            for g in q["groups"]:
                lines.append(ptr("gui.qc.group", name=g["output"], used=g["frames_integrated"],
                                 total=g["frames_total"],
                                 drizzle=ptr("gui.qc.drizzle_ok") if g["drizzle_suitable"] else ptr("gui.qc.drizzle_no")))
            lines += [f"⚠ {w}" for w in self.result.warnings]
        self.qc_text.setPlainText("\n".join(lines))
        self._show_preview()

    def _current_output(self):
        if self.result is None or not self.result.outputs:
            return None
        i = min(max(int(self.result_group.get() or 0), 0), len(self.result.outputs) - 1)
        return self.result.outputs[i]

    def _show_preview(self) -> None:
        if not hasattr(self, "image"):
            return
        o = self._current_output()
        path = None
        if o is not None:
            path = o.stack if (self.result_view.get() == "stack" and o.stack) else o.output
        if path == self._shown_path:
            return
        self._shown_path = path
        self._image_token += 1
        if path is None or not Path(path).is_file():
            self.image.set_image(None)
            return
        token, post = self._image_token, self.bridge.post

        def work() -> None:
            try:
                post(("image", token, path, load_master_display(Path(path))))
            except Exception as exc:  # noqa: BLE001
                post(("image_error", token, exc))

        run_thread(work, "photons-image")

    def rebuild(self) -> None:
        self._shown_path = None
        super().rebuild()

    # ------------------------------------------------------------------ 交棒

    def send_to_process(self) -> None:
        if self.result is None or not self.result.outputs:
            return
        self.shell.send_to_process(Path(self.result.recipe))


__all__ = ["StackPage", "QLabel"]
