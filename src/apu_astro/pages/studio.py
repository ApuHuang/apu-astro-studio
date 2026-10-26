"""後製工作室（整合版的第三步「後製」）：可以回頭修改的步驟鏈。

左邊步驟清單、中間影像（結果／對照前後／遮罩；去星後可以只看無星圖或星點層）、右邊這一步的設定（依工具自動產生）。
改任何設定：先用縮小的影像即時預覽，停手後背景算完整解析度；只重算改到的那一步之後。
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from apu_processing import compose as pcompose
from apu_processing import coverage as pcoverage
from apu_processing import display as pdisplay
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLineEdit,
                               QMenu, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from .. import icons
from ..settings import load_settings, save_settings
from ..studio import backend, chain as ch, export, masks, ops
from ..studio import tools as tools_mod
from ..studio.tools import T, Cancelled, RegionOutside
from ..theme import D
from ..ui.colorwheel import WheelBox
from ..ui.curves import CurveEditor
from ..ui.histogram import Histogram
from ..ui.imageview import ImageView
from ..ui.var import Var
from ..ui.widgets import (Hero, InfoButton, MetricRow, PanelGroup, ParameterSlider, ParameterToggle, Segmented,
                          button, delayed, error, hline, label)
from ..ui.worker import Bridge, run_thread
from .base import Page
from .studio_steps import StepPanel

PROJECT_SUFFIX = ".apuastro"
OPEN_FILTER = "影像 Images (*.xisf *.fit *.fits *.fts *.tif *.tiff *.png *.recipe.json *.apuastro);;* (*)"
MASK_MODES = [("all", T("整張", "Whole image")), ("bright", T("亮部", "Bright areas")), ("dark", T("暗部", "Dark areas")),
              ("range", T("亮度範圍", "Brightness range")), ("stars", T("星點", "Stars")),
              ("nostars", T("星點以外", "Everything but stars")), ("color", T("顏色範圍", "Colour range")),
              ("brush", T("手畫", "Painted"))]


class _Job:
    def __init__(self, gen: int, index: int, res: str):
        self.gen, self.index, self.res = gen, index, res
        self.cancelled = threading.Event()


class StudioPage(Page):
    key = "process"

    def __init__(self, shell):
        super().__init__(shell)
        st = load_settings()
        self.chain = ch.Chain(int(st.get("studio_memory_gb", 8)) * 2 ** 30)
        self.selected = -1
        self.view_mode = Var("result")
        self.layer_view = Var("combined")
        self.full_preview = Var(bool(st.get("studio_full_preview", True)))
        self.gpu_on = Var(bool(st.get("studio_gpu", True)))
        backend.enabled = bool(self.gpu_on.get())
        self.export_opts = export.ExportOptions(**{k: v for k, v in (st.get("studio_export") or {}).items()
                                                   if k in export.ExportOptions.__dataclass_fields__})
        self.project_path: Path | None = None
        self.loading = False
        self._job: _Job | None = None
        self._gen = 0
        self._busy_since = 0.0
        self._last: dict[str, ch.Result] = {}
        self._hover_text = ""
        self._tool_mode: str | None = None       # brush_exclude／brush_mask／pick_sel／pick_mask／crop／region
        self.bridge = Bridge(self._handle, self)
        self._quick_timer = delayed(self, 180, lambda: self._request("quick"))
        self._full_timer = delayed(self, 650, lambda: self._request(self.chain.region_res() or "full"))
        self._region_timer = delayed(self, 250, lambda: self._request(self.chain.region_res() or "full"))
        self._inset: tuple | None = None         # 最後一次顯示的預覽區 (步驟, 圖層, 影像, 位置)
        self.view_mode.subscribe(lambda _v: self.refresh_view())
        self.layer_view.subscribe(lambda _v: self.refresh_view())
        self.full_preview.subscribe(lambda v: save_settings(studio_full_preview=bool(v)))
        self.gpu_on.subscribe(self._gpu_changed)
        self._status_fn = lambda: str(T("開一張疊好的影像開始後製", "Open a stacked image to start"))

    # ------------------------------------------------------------------ 介面

    def build_ui(self) -> QWidget:
        self.layer_seg = Segmented([("combined", str(T("合成", "Combined"))), ("starless", str(T("無星圖", "Starless"))),
                                    ("stars", str(T("星點層", "Stars")))], self.layer_view)
        self.view_seg = Segmented([("result", str(T("結果", "Result"))), ("compare", str(T("對照前後", "Before/after"))),
                                   ("mask", str(T("遮罩", "Mask")))], self.view_mode,
                                  tooltips={"compare": str(T("左邊是這一步之前、右邊是之後，拖曳中線比較",
                                                             "Left is before this step, right is after; drag the line")),
                                            "mask": str(T("紅色是這一步作用的範圍", "Red shows where this step applies"))})
        self.open_btn = button(str(T("開啟", "Open")), tooltip=str(T("開啟疊好的影像（XISF、FITS、TIFF）、多張單色 master "
                                                                   "或疊圖紀錄（Ctrl+O）", "Open a stacked image, several "
                                                                   "mono masters or a stack record (Ctrl+O)")),
                               on_click=self.ask_open)
        self.project_btn = QPushButton(str(T("專案", "Project")))
        menu = QMenu(self.project_btn)
        menu.addAction(str(T("開啟專案…", "Open Project…")), self.ask_open_project)
        menu.addAction(str(T("儲存專案…", "Save Project…")), self.ask_save_project)
        menu.addSeparator()
        menu.addAction(str(T("重設為建議流程", "Reset to Recommended Steps")), self.reset_steps)
        self.project_btn.setMenu(menu)
        self.region_btn = button(str(T("預覽區", "Preview area")),
                                 tooltip=str(T("在影像上框一塊，只算那一塊的完整解析度（調反摺積、降噪、銳化等細節時很快）；"
                                               "再按一次回到整張", "Drag a box on the image to render just that area at "
                                               "full resolution (fast for tuning detail); click again to go back")),
                                 on_click=self._toggle_region)
        self.region_btn.setCheckable(True)
        self.export_btn = button(str(T("出圖", "Export")), "prominent",
                                 str(T("算完整解析度並存成 PNG／JPEG／TIFF／FITS（Ctrl+S）", "Render at full resolution and "
                                       "save as PNG/JPEG/TIFF/FITS (Ctrl+S)")), self.ask_export)
        self.steps_panel = StepPanel()
        self.steps_panel.build_add_menu(self.add_step)
        self.steps_panel.selected.connect(self.select)
        self.steps_panel.structure_changed.connect(self._structure_changed)
        self.steps_panel.dup_btn.clicked.connect(self.duplicate_step)
        self.steps_panel.up_btn.clicked.connect(lambda: self.move_step(-1))
        self.steps_panel.down_btn.clicked.connect(lambda: self.move_step(1))
        self.steps_panel.del_btn.clicked.connect(self.delete_step)

        center = QStackedWidget()
        self.hero = Hero(str(T("後製", "Process")), str(T("開一張疊好的影像（XISF、FITS、TIFF），或疊圖紀錄合成多個濾鏡",
                                                        "Open a stacked image (XISF, FITS, TIFF) or a stack record to "
                                                        "combine filters")),
                         str(T("Ctrl+O 開啟，也可以把檔案拖進視窗", "Ctrl+O to open, or drop files onto the window")))
        center.addWidget(self.hero)
        self.image = ImageView()
        self.image.hovered.connect(self._hovered)
        self.image.zoom_changed.connect(lambda _z: self._refresh_summary())
        self.image.stroke.connect(self._stroke)
        self.image.stroke_done.connect(self._stroke_done)
        self.image.picked.connect(self._picked)
        self.image.crop_changed.connect(self._crop_changed)
        center.addWidget(self.image)
        self.center = center

        self.inspector = QScrollArea()
        self.inspector.setFixedWidth(D.panel_width + 16)
        self.inspector.setWidgetResizable(True)
        self.inspector.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector.setStyleSheet(f"QScrollArea {{ background: {D.chrome}; border: none; "
                                     f"border-left: 1px solid {D.separator}; }}")
        body = QWidget()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self.steps_panel)
        h.addWidget(center, 1)
        h.addWidget(self.inspector)
        ui = self.layout([self.layer_seg, self.view_seg],
                         [self.region_btn, self.open_btn, self.project_btn, self.export_btn], body, None)
        self._fill_steps()
        self._build_inspector()
        return ui

    # ------------------------------------------------------------------ 外殼

    def doc_title(self) -> str:
        return self.chain.source.title() if self.chain.source else ""

    def rail_text(self) -> str:
        if self.progress is not None:
            return f"{round(self.progress * 100)}%"
        src = self.chain.source
        if src is None:
            return ""
        return src.compose["preset"] if src.compose else src.paths[0].stem

    def is_busy(self) -> bool:
        return self._job is not None or self.loading

    def close_page(self) -> None:
        if self._job is not None:
            self._job.cancelled.set()

    def ask_open(self) -> None:
        start = str(self.chain.source.paths[0].parent) if self.chain.source else load_settings().get("studio_dir", "")
        paths, _ = QFileDialog.getOpenFileNames(self, str(T("開啟", "Open")), start, OPEN_FILTER)
        if paths:
            self.open_paths([Path(p) for p in paths])

    def handle_drop(self, paths: list[Path]) -> None:
        self.open_paths(paths)

    def shortcut(self, name: str) -> None:
        if name == "save":
            self.ask_export()
        elif name == "fit":
            self.image.fit()
        elif name == "actual":
            self.image.actual_size()

    # ------------------------------------------------------------------ 開檔

    def open_paths(self, paths: list[Path]) -> None:
        paths = [Path(p) for p in paths]
        projects = [p for p in paths if p.suffix.lower() == PROJECT_SUFFIX]
        if projects:
            self.open_project(projects[0])
            return
        self._load([p for p in paths if p.is_file()], None, None, None)

    def _load(self, paths: list[Path], compose: dict | None, coverages: list | None, steps: list | None) -> None:
        if not paths:
            return
        self.loading = True
        names = ", ".join(p.name for p in paths[:3])
        self.set_status(lambda: str(T(f"開啟 {names}…", f"Opening {names}…")), 0.0)
        save_settings(studio_dir=str(paths[0].parent))
        post = self.bridge.post

        def work() -> None:
            try:
                src = ch.open_source(paths, compose, coverages)
                post(("loaded", src, steps))
            except Exception as exc:  # noqa: BLE001
                post(("error", -1, f"{exc}\n\n{traceback.format_exc()[-1200:]}"))

        run_thread(work, "studio-open")

    def _loaded(self, src: ch.Source, steps: list | None) -> None:
        self.loading = False
        self.chain.set_source(src)
        self.chain.set_region(None)
        self._inset = None
        self._sync_region_btn()
        self.chain.steps = steps if steps is not None else ch.default_steps(src.color, src.narrowband)
        self._last.clear()
        self.selected = len(self.chain.steps) - 1
        self._fill_steps()
        self._build_inspector()
        # 整頁狀態跟著「有圖了」更新（出圖、專案、檢視切換）：以前只切畫面，那幾顆按鍵停在還沒開圖時的停用狀態，
        # 要切換語言（整頁重建）才能按（0.1.3、0.1.4）
        self.refresh()
        self.image.set_image(None)
        self.title_changed.emit()
        shape = src.image.shape
        kind = str(T("彩色", "colour")) if src.color else str(T("單色", "mono"))
        self.set_status(lambda: str(T(f"已開啟：{shape[-1]} × {shape[-2]}（{kind}），先用縮小的影像處理…",
                                      f"Opened {shape[-1]} × {shape[-2]} ({kind}); processing a preview…")), 0.0)
        self._request("quick")

    def reset_steps(self) -> None:
        src = self.chain.source
        if src is None:
            return
        self.chain.steps = ch.default_steps(src.color, src.narrowband)
        self.selected = len(self.chain.steps) - 1
        self._structure_changed()

    # ------------------------------------------------------------------ 專案

    def ask_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, str(T("開啟專案", "Open Project")),
                                              load_settings().get("studio_dir", ""), f"APU Astro (*{PROJECT_SUFFIX})")
        if path:
            self.open_project(Path(path))

    def open_project(self, path: Path) -> None:
        try:
            source, steps = ch.load_project(path)
        except Exception as exc:  # noqa: BLE001
            error(self, str(exc))
            return
        self.project_path = path
        paths = [Path(p) for p in source.get("paths", [])]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            error(self, str(T("找不到專案的來源檔：\n", "Source files not found:\n")) + "\n".join(map(str, missing[:5])))
            return
        self._load(paths, source.get("compose"), source.get("coverages"), steps)

    def ask_save_project(self) -> None:
        if self.chain.source is None:
            return
        start = str(self.project_path or self.chain.source.paths[0].with_suffix(PROJECT_SUFFIX))
        path, _ = QFileDialog.getSaveFileName(self, str(T("儲存專案", "Save Project")), start,
                                              f"APU Astro (*{PROJECT_SUFFIX})")
        if path:
            p = Path(path)
            if p.suffix.lower() != PROJECT_SUFFIX:
                p = p.with_suffix(PROJECT_SUFFIX)
            ch.save_project(self.chain, p)
            self.project_path = p
            self.set_status(lambda: str(T(f"專案已存：{p.name}", f"Project saved: {p.name}")))

    # ------------------------------------------------------------------ 步驟清單

    def _fill_steps(self) -> None:
        if not hasattr(self, "steps_panel"):
            return
        src = self.chain.source
        sub = ""
        if src is not None:
            h, w = src.image.shape[-2:]
            sub = f"{w}×{h} · " + (str(T("彩色", "Colour")) if src.color else str(T("單色", "Mono")))
        warn = {i for i, r in self._step_warnings().items() if r}
        self.steps_panel.set_steps(self.chain.steps, src.title() if src else "", sub, self.selected, warn)

    def _step_warnings(self) -> dict[int, list]:
        out = {}
        for i in range(len(self.chain.steps)):
            r = self.chain.cached(i, "quick")
            if r is not None and r.state.warnings:
                out[i] = r.state.warnings
        return out

    def add_step(self, kind: str) -> None:
        if self.chain.source is None:
            return
        step = ch.new_step(kind)
        sep_before = self._separated_before(self.selected + 1)
        if sep_before and kind in ("denoise", "color_grade", "curves", "sharpen", "stretch"):
            step.layer = "starless"
        at = self.selected + 1
        self.chain.steps.insert(at, step)
        self.selected = at
        self._structure_changed()

    def duplicate_step(self) -> None:
        i = self.selected
        if 0 <= i < len(self.chain.steps):
            import copy
            s = copy.deepcopy(self.chain.steps[i])
            s.id = ch.Step(s.kind, {}).id
            self.chain.steps.insert(i + 1, s)
            self.selected = i + 1
            self._structure_changed()

    def move_step(self, d: int) -> None:
        i = self.selected
        j = i + d
        if 0 <= i < len(self.chain.steps) and 0 <= j < len(self.chain.steps):
            s = self.chain.steps
            s[i], s[j] = s[j], s[i]
            self.selected = j
            self._structure_changed()

    def delete_step(self) -> None:
        i = self.selected
        if 0 <= i < len(self.chain.steps):
            del self.chain.steps[i]
            self.selected = min(i, len(self.chain.steps) - 1)
            self._structure_changed()

    def _structure_changed(self) -> None:
        self._fill_steps()
        self._build_inspector()
        self.schedule()

    def _separated_before(self, index: int) -> bool:
        sep = False
        for s in self.chain.steps[:max(0, index)]:
            if not s.enabled:
                continue
            if s.kind == "star_split":
                sep = True
            elif s.kind == "star_merge":
                sep = False
        return sep

    def select(self, index: int) -> None:
        if index == self.selected and hasattr(self, "_built_for") and self._built_for == index:
            return
        self.selected = index
        self._set_tool(None)
        self._build_inspector()
        self.refresh_view()
        cached = self.chain.cached(index, "quick") if self.chain.source else None
        if self.chain.source is not None and cached is None:
            self._request("quick")
        else:
            self._detail_timer()

    def current_step(self) -> ch.Step | None:
        return self.chain.steps[self.selected] if 0 <= self.selected < len(self.chain.steps) else None

    # ------------------------------------------------------------------ 右側面板

    def _build_inspector(self) -> None:
        if not hasattr(self, "inspector"):
            return
        self._built_for = self.selected
        inner = QWidget()
        inner.setStyleSheet(f"background: {D.chrome};")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        self.histogram = Histogram()
        lay.addWidget(self.histogram)
        self._visibility: list[tuple[QWidget, object, dict]] = []
        step = self.current_step()
        if self.chain.source is None:
            lay.addWidget(label(str(T("還沒有開啟影像", "Nothing open yet")), D.secondary))
        elif step is None:
            self._build_source(lay)
        else:
            self._build_step(lay, step)
        lay.addWidget(self._build_settings())
        lay.addStretch(1)
        old = self.inspector.takeWidget()
        if old is not None:
            old.deleteLater()
        self.inspector.setWidget(inner)
        self._update_visibility()

    def _card(self, title: str, info: str | None = None, key: str | None = None) -> PanelGroup:
        return PanelGroup(f"studio.{key or title}", title, info, self.shell.panel_state, self.shell.save_panel)

    def _build_source(self, lay: QVBoxLayout) -> None:
        src = self.chain.source
        g = self._card(str(T("來源影像", "Source")), key="source")
        h, w = src.image.shape[-2:]
        g.add(MetricRow(str(T("尺寸", "Size")))).set(f"{w} × {h}")
        g.add(MetricRow(str(T("色彩", "Colour")))).set(str(T("彩色", "Colour")) if src.color else str(T("單色", "Mono")))
        header = src.header
        if header is not None:
            for key_, title in (("OBJECT", T("目標", "Object")), ("FILTER", T("濾鏡", "Filter")),
                                ("INSTRUME", T("相機", "Camera")), ("FOCALLEN", T("焦距", "Focal length"))):
                v = header.get(key_) if hasattr(header, "get") else None
                if v not in (None, ""):
                    g.add(MetricRow(str(title))).set(str(v))
        g.add(MetricRow(str(T("覆蓋率圖", "Coverage map")))).set(
            str(T("有（去光梯度只在覆蓋足夠處取樣）", "Yes (gradient sampling uses it)")) if src.coverage_mask is not None
            else "—")
        for p in src.paths:
            g.add(label(p.name, D.secondary, 8, wrap=True))
        lay.addWidget(g)
        if src.composer is not None and src.compose:
            lay.addWidget(self._build_compose())

    def _build_compose(self) -> PanelGroup:
        src = self.chain.source
        g = self._card(str(T("合成", "Channel combination")), str(T(
            "幾張單色 master 組成彩色：HOO（Ha→紅、OIII→綠藍）、SHO（SII→紅、Ha→綠、OIII→藍）、RGB。"
            "每個通道先扣自己的天空、對齊噪聲再組合；強度可以分別調。",
            "Builds colour from mono masters: HOO, SHO or RGB. Each channel is sky-subtracted and noise-matched; "
            "strengths can be set per channel.")), key="compose")
        preset = Var(src.compose["preset"])
        g.add(Segmented([(p, p) for p in pcompose.PRESETS], preset, stretch=True))
        preset.subscribe(lambda v: self._recompose({"preset": v, "channels": []}), g)
        choices = [f"{p.name}（{f}）" if f else p.name for p, f in zip(src.paths, src.filters)] + \
                  [str(T("不使用", "None"))]
        for i, (role, source, strength) in enumerate(src.compose["channels"]):
            weights = dict(pcompose.PRESETS[src.compose["preset"]])
            target = "+".join(c for c, wgt in zip("RGB", weights.get(role, (0, 0, 0))) if wgt)
            g.add(label(f"{role} → {target}", D.label, 9, bold=True), 6)
            combo = QComboBox()
            combo.addItems(choices)
            combo.setCurrentIndex(source if source >= 0 else len(choices) - 1)
            combo.activated.connect(lambda idx, k=i: self._compose_channel(k, source=idx if idx < len(src.paths)
                                                                           else -1))
            g.add(combo)
            var = Var(float(strength) * 100)
            g.add(ParameterSlider(str(T("強度", "Strength")), var, 0, 300, lambda v: f"{v:.0f}%", 5))
            var.subscribe(lambda v, k=i: self._compose_channel(k, strength=float(v) / 100), g)
        return g

    def _compose_channel(self, index: int, source: int | None = None, strength: float | None = None) -> None:
        c = {**self.chain.source.compose, "channels": [list(x) for x in self.chain.source.compose["channels"]]}
        if source is not None:
            c["channels"][index][1] = source
        if strength is not None:
            c["channels"][index][2] = strength
        self._recompose(c)

    def _recompose(self, compose: dict) -> None:
        src = self.chain.source
        if src is None or src.composer is None:
            return
        if not compose.get("channels"):
            s = pcompose.ComposeSettings.for_preset(compose["preset"], src.filters)
            compose = ch.settings_to_dict(s)
        post = self.bridge.post
        self.set_status(lambda: str(T("重新合成…", "Recombining…")), 0.0)

        def work() -> None:
            try:
                ch.recompose(src, compose)
                post(("recomposed", compose.get("preset")))
            except Exception as exc:  # noqa: BLE001
                post(("error", -1, str(exc)))

        run_thread(work, "studio-compose")

    def _build_step(self, lay: QVBoxLayout, step: ch.Step) -> None:
        tool = tools_mod.get(step.kind)
        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(4, 0, 4, 0)
        ic = label("")
        ic.setPixmap(icons.pixmap(tool.icon, D.label, 20))
        hl.addWidget(ic)
        hl.addWidget(label(str(tool.title), D.label, 11, bold=True), 1)
        hl.addWidget(InfoButton(str(tool.help)))
        lay.addWidget(head)
        r = self.chain.cached(self.selected, "quick")
        for w in (r.state.warnings if r else []):
            lay.addWidget(label("⚠ " + str(w), D.beta, 8.5, wrap=True))
        g = self._card(str(T("設定", "Settings")), key=f"tool.{step.kind}")
        if tool.maskable and self._separated_before(self.selected) and step.kind not in ("star_merge",):
            lv = Var(step.layer)
            g.add(label(str(T("作用在", "Apply to")), D.secondary, 8.5))
            g.add(Segmented([(k, str(t)) for k, t in tools_mod.LAYER_CHOICES], lv, stretch=True))
            lv.subscribe(lambda v: self._set_layer(step, v), g)
        elif step.kind == "stretch" and self._separated_before(self.selected):
            lv = Var(step.layer)
            g.add(label(str(T("作用在", "Apply to")), D.secondary, 8.5))
            g.add(Segmented([(k, str(t)) for k, t in tools_mod.LAYER_CHOICES[2:]], lv, stretch=True))
            lv.subscribe(lambda v: self._set_layer(step, v), g)
        for spec in tool.controls():
            w = self._control(spec, step)
            if w is not None:
                g.add(w)
                if spec.get("show"):
                    self._visibility.append((w, spec["show"], step.params))
        g.add(button(str(T("重設這一步", "Reset this step")), on_click=lambda: self._reset_step(step)), 8)
        lay.addWidget(g)
        if tool.maskable:
            lay.addWidget(self._build_mask(step))

    def _set_layer(self, step: ch.Step, layer: str) -> None:
        step.layer = layer
        self._fill_steps()
        self.schedule()

    def _reset_step(self, step: ch.Step) -> None:
        step.params = tools_mod.get(step.kind).defaults()
        step.mask = masks.default_mask()
        self._build_inspector()
        self._fill_steps()
        self.schedule()

    def _bind(self, step_params: dict, key: str, on_change=None) -> Var:
        var = Var(step_params.get(key))

        def changed(v) -> None:
            step_params[key] = v
            self._update_visibility()
            if on_change is not None:
                on_change(v)
            if key in ch.UI_ONLY:
                self.refresh_view()
                if key == "brush":
                    self.image.set_tool(self.image.tool, float(v))
            else:
                self.schedule()
                if key == "method":
                    self._fill_steps()

        var.subscribe(changed)
        return var

    def _control(self, spec: dict, step: ch.Step) -> QWidget | None:
        t = spec["type"]
        params = step.params
        tip = str(spec["help"]) if spec.get("help") else None
        if t == "slider":
            var = self._bind(params, spec["key"])
            fmt = spec["fmt"]
            return ParameterSlider(str(spec["label"]), var, spec["lo"], spec["hi"],
                                   lambda v, f=fmt: f.format(v), spec["step"], tooltip=tip)
        if t == "toggle":
            return ParameterToggle(str(spec["label"]), self._bind(params, spec["key"]), tooltip=tip)
        if t == "choice":
            var = self._bind(params, spec["key"])
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 4, 0, 2)
            v.setSpacing(3)
            v.addWidget(label(str(spec["label"]), D.secondary, 8.5))
            opts = [(k, str(lbl)) for k, lbl in spec["options"]]
            if len(opts) <= 3 and sum(len(o[1]) for o in opts) <= 16:
                v.addWidget(Segmented(opts, var, stretch=True))
            else:
                combo = QComboBox()
                for k, lbl in opts:
                    combo.addItem(lbl, k)
                combo.setCurrentIndex(max(0, [k for k, _ in opts].index(var.get()) if var.get() in
                                          [k for k, _ in opts] else 0))
                combo.activated.connect(lambda i, c=combo: var.set(c.itemData(i)))
                v.addWidget(combo)
            if tip:
                box.setToolTip(tip)
            return box
        if t == "section":
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 8, 0, 0)
            v.setSpacing(4)
            v.addWidget(hline())
            v.addWidget(label(str(spec["label"]), D.label, 9, bold=True))
            return box
        if t == "view":
            var = self._bind(params, spec["key"])
            seg = Segmented([("result", str(T("結果", "Result"))), ("samples", str(T("取樣", "Samples"))),
                             ("model", str(T("背景模型", "Model"))), ("removed", str(T("扣掉的", "Removed")))],
                            var, stretch=True,
                            tooltips={"samples": str(T("紅色是判斷為星雲、沒拿來取樣的地方", "Red: treated as nebula, "
                                                                                  "not sampled")),
                                      "removed": str(T("被當成光害扣掉的部分：看到星雲的形狀就表示扣到星雲了",
                                                       "What was removed as light pollution; nebula shapes here mean "
                                                       "nebula was removed"))})
            return seg
        if t == "brush":
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 2, 0, 2)
            v.addWidget(label(str(spec["label"]), D.secondary, 8.5, wrap=True))
            row = QHBoxLayout()
            paint = button(str(T("塗抹", "Paint")), on_click=lambda: self._toggle_tool("brush_exclude"))
            paint.setIcon(icons.icon("brush", D.label, 15))
            paint.setCheckable(True)
            paint.setChecked(self._tool_mode == "brush_exclude")
            self._paint_btn = paint
            clear = button(str(T("清除", "Clear")), on_click=lambda: self._clear_strokes(params, spec["key"]))
            row.addWidget(paint)
            row.addWidget(clear)
            row.addStretch(1)
            v.addLayout(row)
            return box
        if t == "eyedropper":
            b = button(str(spec["label"]), on_click=lambda: self._toggle_tool("pick_sel"))
            b.setIcon(icons.icon("dropper", D.label, 15))
            b.setCheckable(True)
            b.setChecked(self._tool_mode == "pick_sel")
            self._pick_btn = b
            self._pick_var = None
            return b
        if t == "wheels":
            return self._wheels(step)
        if t == "curves":
            ed = CurveEditor(lambda curves: self._curves_changed(step, curves))
            c = params.get("curves") or {}
            ed.set_curves(pdisplay.ToneCurves(**{k: [list(map(float, q)) for q in v] for k, v in c.items()}))
            self._curve_editor = ed
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(ed)
            v.addWidget(button(str(T("重設這條曲線", "Reset this curve")), on_click=ed.reset_channel))
            return box
        if t == "crop":
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 2, 0, 2)
            v.addWidget(label(str(spec["label"]), D.secondary, 8.5, wrap=True))
            row = QHBoxLayout()
            draw = button(str(T("畫裁切框", "Draw crop")), on_click=lambda: self._toggle_tool("crop"))
            draw.setCheckable(True)
            draw.setChecked(self._tool_mode == "crop")
            row.addWidget(draw)
            row.addWidget(button(str(T("清除", "Clear")), on_click=lambda: self._set_box(step, None)))
            row.addStretch(1)
            v.addLayout(row)
            src = self.chain.source
            if src is not None and src.coverage_mask is not None and self.selected == 0:
                v.addWidget(button(str(T("依疊圖覆蓋率建議", "Suggest from coverage")),
                                   on_click=lambda: self._suggest_crop(step)))
            return box
        return None

    def _wheels(self, step: ch.Step) -> QWidget:
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 2, 0, 6)
        grid.setHorizontalSpacing(4)
        names = (("sh", T("暗部", "Shadows")), ("mid", T("中間調", "Midtones")), ("hi", T("亮部", "Highlights")))
        p = step.params
        for col, (z, title) in enumerate(names):
            wb = WheelBox(str(title), float(p.get(f"{z}_h", 0)), float(p.get(f"{z}_a", 0)))

            def moved(h: float, a: float, z: str = z) -> None:
                p[f"{z}_h"], p[f"{z}_a"] = h, a
                self.schedule()

            wb.wheel.changed.connect(moved)
            grid.addWidget(wb, 0, col)
            var = self._bind(p, f"{z}_l")
            grid.addWidget(ParameterSlider("", var, -1, 1, lambda v: f"{v:+.2f}", 0.05, editable=False), 1, col)
        return box

    def _curves_changed(self, step: ch.Step, curves) -> None:
        step.params["curves"] = {k: [list(map(float, q)) for q in getattr(curves, k)]
                                 for k in ("master", "red", "green", "blue")}
        self.schedule()

    def _build_mask(self, step: ch.Step) -> PanelGroup:
        g = self._card(str(T("作用範圍（遮罩）", "Where it applies (mask)")), str(T(
            "只讓這一步作用在一部分畫面：亮部、暗部、星點、某個顏色，或自己畫。羽化讓邊緣平順。"
            "上方切到「遮罩」可以看到範圍（紅色）。",
            "Limit this step to part of the image: bright or dark areas, stars, a colour, or a painted area. "
            "Switch the view to Mask to see it in red.")), key="mask")
        m = step.mask
        mode = self._bind(m, "mode", lambda _v: self._fill_steps())
        combo = QComboBox()
        for k, lbl in MASK_MODES:
            combo.addItem(str(lbl), k)
        keys = [k for k, _ in MASK_MODES]
        combo.setCurrentIndex(keys.index(m.get("mode", "all")) if m.get("mode", "all") in keys else 0)
        combo.activated.connect(lambda i: mode.set(combo.itemData(i)))
        g.add(combo)
        rng = lambda p: p.get("mode") in ("bright", "dark", "range")  # noqa: E731
        for key, title, show in (("lo", T("亮度下限", "From"), rng),
                                 ("hi", T("亮度上限", "To"), lambda p: p.get("mode") == "range")):
            w = g.add(ParameterSlider(str(title), self._bind(m, key), 0, 1, lambda v: f"{v:.2f}", 0.01))
            self._visibility.append((w, show, m))
        col = lambda p: p.get("mode") == "color"  # noqa: E731
        pick = button(str(T("滴管：在畫面上點顏色", "Eyedropper: click a colour")),
                      on_click=lambda: self._toggle_tool("pick_mask"))
        pick.setIcon(icons.icon("dropper", D.label, 15))
        g.add(pick)
        self._visibility.append((pick, col, m))
        self._mask_hue_var = self._bind(m, "hue")
        for key, title, lo, hi, fmt, var in (("hue", T("色相", "Hue"), 0, 360, "{:.0f}°", self._mask_hue_var),
                                             ("width", T("範圍", "Range"), 10, 180, "{:.0f}°", None)):
            w = g.add(ParameterSlider(str(title), var or self._bind(m, key), lo, hi, lambda v, f=fmt: f.format(v), 1))
            self._visibility.append((w, col, m))
        brush = button(str(T("在畫面上畫範圍（右鍵擦）", "Paint the area (right-drag erases)")),
                       on_click=lambda: self._toggle_tool("brush_mask"))
        brush.setIcon(icons.icon("brush", D.label, 15))
        g.add(brush)
        self._visibility.append((brush, lambda p: p.get("mode") == "brush", m))
        g.add(ParameterSlider(str(T("羽化", "Feather")), self._bind(m, "feather"), 0, 100, lambda v: f"{v:.0f} px", 1))
        g.add(ParameterToggle(str(T("反轉", "Invert")), self._bind(m, "invert")))
        return g

    def _build_settings(self) -> PanelGroup:
        g = self._card(str(T("預覽與效能", "Preview & performance")), key="studio_settings")
        g.add(ParameterToggle(str(T("停手後算完整解析度", "Render full resolution when idle")), self.full_preview))
        g.add(ParameterToggle(str(T("使用 GPU", "Use GPU")), self.gpu_on))
        name = backend.device_name() if self.gpu_on.get() else None
        g.add(label(name or str(T("（沒有可用的 NVIDIA GPU，用 CPU）", "(no NVIDIA GPU; using CPU)")), D.secondary, 8))
        return g

    def _update_visibility(self) -> None:
        for w, show, params in getattr(self, "_visibility", []):
            try:
                w.setVisible(bool(show(params)))
            except RuntimeError:
                pass

    def _gpu_changed(self, on) -> None:
        backend.enabled = bool(on)
        save_settings(studio_gpu=bool(on))

    # ------------------------------------------------------------------ 工具（筆刷、滴管、裁切）

    def _toggle_tool(self, mode: str) -> None:
        self._set_tool(None if self._tool_mode == mode else mode)

    def _set_tool(self, mode: str | None) -> None:
        self._tool_mode = mode
        step = self.current_step()
        if mode in ("brush_exclude", "brush_mask"):
            r = float(step.params.get("brush", 150)) if step is not None else 150.0
            self.image.set_tool("brush", r)
        elif mode in ("pick_sel", "pick_mask"):
            self.image.set_tool("pick")
        else:
            self.image.set_tool(None)
        self.image.set_crop_mode(mode in ("crop", "region"))
        if mode == "crop" and step is not None and step.params.get("box"):
            self.image.crop_box = tuple(step.params["box"])
        self._sync_region_btn()
        for attr, m in (("_paint_btn", "brush_exclude"), ("_pick_btn", "pick_sel")):
            b = getattr(self, attr, None)
            try:
                if b is not None:
                    b.setChecked(mode == m)
            except RuntimeError:
                pass
        self.refresh_view()

    def _stroke(self, x: float, y: float, r: float, add: bool) -> None:
        step = self.current_step()
        if step is None:
            return
        target = step.params.setdefault("exclude", []) if self._tool_mode == "brush_exclude" else \
            step.mask.setdefault("strokes", [])
        target.append([round(x, 1), round(y, 1), round(r, 1), bool(add)])
        self._show_overlay()

    def _stroke_done(self) -> None:
        self.schedule()

    def _clear_strokes(self, params: dict, key: str) -> None:
        params[key] = []
        self._show_overlay()
        self.schedule()

    def _picked(self, fx: float, fy: float) -> None:
        r = self.chain.cached(self.selected - 1, "quick") if self.selected >= 0 else None
        if r is None:
            return
        disp = ch.display_of(r.state, "starless" if r.state.separated else "combined")
        if disp.shape[0] < 3:
            return
        s = self.chain.scale("quick")
        x, y = int(fx / s), int(fy / s)
        h, w = disp.shape[1:]
        patch = disp[:, max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)].reshape(3, -1).mean(axis=1)
        hue, _c = ops.oklab_hue(tuple(float(v) for v in patch))
        step = self.current_step()
        if self._tool_mode == "pick_mask":
            step.mask["hue"] = round(hue)
            if getattr(self, "_mask_hue_var", None) is not None:
                self._mask_hue_var.set(round(hue))
        else:
            step.params["sel_hue"] = round(hue)
            step.params["sel_on"] = True
            self._build_inspector()
        self._set_tool(None)
        self.schedule()

    def _crop_changed(self) -> None:
        if self._tool_mode == "region":
            self._region_drawn()
            return
        step = self.current_step()
        if step is None or self._tool_mode != "crop" or self.image.crop_box is None:
            return
        self._set_box(step, [round(v, 1) for v in self.image.crop_box], keep_tool=True)

    def _set_box(self, step: ch.Step, box, keep_tool: bool = False) -> None:
        step.params["box"] = box
        if not keep_tool:
            self._set_tool(None)
        self.schedule()

    def _suggest_crop(self, step: ch.Step) -> None:
        src = self.chain.source
        if src is None or src.coverage_mask is None:
            return
        box = pcoverage.suggest_crop(src.coverage_mask.astype(np.float32))
        if box:
            self._set_box(step, [float(v) for v in box])

    # ------------------------------------------------------------------ 預覽區

    def _toggle_region(self) -> None:
        if self.chain.source is None:
            self._sync_region_btn()
            return
        if self.chain.region is not None or self._tool_mode == "region":
            self.chain.set_region(None)
            self._inset = None
            self._set_tool(None)
            self.refresh_view()
            self._detail_timer()
            return
        self._set_tool("region")
        self.set_status(lambda: str(T("在影像上拖曳框出要看完整解析度的範圍", "Drag a box over the area to see at full "
                                      "resolution")), None)

    def _sync_region_btn(self) -> None:
        if hasattr(self, "region_btn"):
            self.region_btn.setChecked(self.chain.region is not None or self._tool_mode == "region")

    def _region_drawn(self) -> None:
        box = self.image.crop_box
        if box is None:
            return
        x0, y0, x1, y1 = box
        src_box = self.chain.map_to_source((y0, x0, y1, x1), self._view_index())
        if src_box is None:
            return
        self.chain.set_region(src_box)
        self._inset = None
        self._set_tool(None)
        self._full_timer.stop()
        self._region_timer.start()

    def _detail_timer(self) -> None:
        """快速預覽之後：有預覽區就算那一塊，沒有的話（設定開著時）算完整解析度。"""
        if self.chain.source is None or self._view_index() < 0:
            return
        if self.chain.region is not None:
            if self.chain.cached(self._view_index(), self.chain.region_res()) is None:
                self._region_timer.start()
        elif self.full_preview.get() and self.chain.cached(self._view_index(), "full") is None:
            self._full_timer.start()

    # ------------------------------------------------------------------ 計算

    def schedule(self) -> None:
        if self.chain.source is None:
            return
        self._full_timer.stop()
        self._region_timer.stop()
        self._quick_timer.start()

    def _view_index(self) -> int:
        """畫面要顯示第幾步的結果（裁切時顯示裁切前）。"""
        if self._tool_mode == "crop":
            return self.selected - 1
        return self.selected

    def _request(self, res: str) -> None:
        if self.chain.source is None:
            return
        index = max(self._view_index(), self.selected)
        if index < 0:
            self.refresh_view()
            return
        if self.chain.cached(index, res) is not None:
            self.refresh_view()
            if res == "quick":
                self._detail_timer()
            return
        if self._job is not None:
            self._job.cancelled.set()
        self._gen += 1
        job = _Job(self._gen, index, res)
        self._job = job
        steps = self.chain.snapshot()
        post = self.bridge.post
        self._busy_since = time.time()

        def work() -> None:
            try:
                r = self.chain.evaluate(index, res, steps, job.cancelled.is_set,
                                        lambda f, s: post(("progress", job.gen, f, s, res)))
                post(("result", job.gen, r))
            except Cancelled:
                post(("cancelled", job.gen))
            except RegionOutside:
                post(("region_outside", job.gen))
            except Exception:  # noqa: BLE001
                post(("error", job.gen, traceback.format_exc()))

        run_thread(work, f"studio-{res}")
        text = (T("預覽中…", "Previewing…") if res == "quick" else T("算完整解析度…", "Rendering full resolution…")
                if res == "full" else T("算預覽區…", "Rendering preview area…"))
        self.set_status(lambda: str(text), 0.0)

    def _handle(self, ev: tuple) -> None:
        kind = ev[0]
        if kind == "loaded":
            self._loaded(ev[1], ev[2])
        elif kind == "recomposed":
            self.chain.source_changed()
            self.title_changed.emit()
            self.rail_changed.emit()
            self.schedule()
        elif kind == "progress":
            if self._job is not None and ev[1] == self._job.gen:
                self.progress = float(ev[2])
                self.status_changed.emit()
                self.rail_changed.emit()
        elif kind == "result":
            r: ch.Result = ev[2]
            if self._job is not None and ev[1] == self._job.gen:
                self._job = None
            self._last[r.res] = r
            took = time.time() - self._busy_since
            res = r.res
            text = (T(f"預覽完成（{took:.1f} 秒）", f"Preview ready ({took:.1f} s)") if res == "quick"
                    else T(f"完整解析度完成（{took:.1f} 秒）", f"Full resolution ready ({took:.1f} s)") if res == "full"
                    else T(f"預覽區完成（{took:.1f} 秒）", f"Preview area ready ({took:.1f} s)"))
            self.set_status(lambda: str(text), None)
            self.has_result = True
            self.refresh_view()
            self._fill_steps_soft()
            if res == "quick" or (self.chain.region is not None and res != self.chain.region_res()):
                self._detail_timer()
            self.rail_changed.emit()
        elif kind == "cancelled":
            if self._job is not None and ev[1] == self._job.gen:
                self._job = None
        elif kind == "region_outside":
            if self._job is not None and ev[1] == self._job.gen:
                self._job = None
            self.chain.set_region(None)
            self._inset = None
            self._sync_region_btn()
            self.refresh_view()
            self.set_status(lambda: str(T("預覽區被裁切掉了，回到整張", "The preview area was cropped away; showing the "
                                          "whole image")), None)
        elif kind == "error":
            if ev[1] == -1:
                self.loading = False
            if self._job is not None and ev[1] == self._job.gen:
                self._job = None
            self.set_status(lambda: str(T("發生錯誤", "Something went wrong")), None)
            error(self, str(ev[2])[-2000:])

    def _fill_steps_soft(self) -> None:
        """有新的警告時才更新清單（重建清單會閃一下）。"""
        warn = {i for i, r in self._step_warnings().items() if r}
        if warn != self.steps_panel.warn:
            self._fill_steps()
            self._build_inspector()

    # ------------------------------------------------------------------ 顯示

    def _result_for(self, index: int) -> ch.Result | None:
        if self.chain.source is None:
            return None
        return self.chain.cached(index, "full") or self.chain.cached(index, "quick")

    def refresh_view(self) -> None:
        if not hasattr(self, "image") or self.chain.source is None:
            return
        index = self._view_index()
        r = self._result_for(index)
        if r is None:
            return
        step = self.current_step()
        scale = self.chain.scale(r.res)
        self.layer_seg.setVisible(r.state.separated)
        disp = ch.display_of(r.state, self.layer_view.get())
        if step is not None and step.kind == "gradient" and self._tool_mode != "crop":
            view = step.params.get("view", "result")
            if view in ("model", "removed") and r.views.get("model" if view == "model" else "removed") is not None:
                arr = r.views["model" if view == "model" else "removed"]
                if view == "model":
                    disp = ops.auto_display(ops.planes(arr))
                else:
                    a = ops.planes(arr)
                    lo, hi = float(a.min()), float(a.max())
                    disp = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        mode = self.view_mode.get()
        self.image.set_compare(None)
        self.image.set_overlay(None)
        if mode == "compare" and index >= 0:
            prev = self._result_for(index - 1)
            if prev is not None and prev.state.layers and \
                    next(iter(prev.state.layers.values())).shape[-2:] == next(iter(r.state.layers.values())).shape[-2:]:
                self.image.set_compare(ch.display_of(prev.state, self.layer_view.get()),
                                       (str(T("之前", "Before")), str(T("之後", "After"))),
                                       pixel_scale=self.chain.scale(prev.res))
        self.image.set_image(disp, keep_view=True, pixel_scale=scale)
        self._show_inset(index, mode, special_view=disp is not None and step is not None and step.kind == "gradient"
                         and step.params.get("view", "result") != "result" and self._tool_mode != "crop")
        if mode == "mask":
            self._show_mask_overlay()
        self._show_overlay()
        if step is not None and step.kind == "gradient" and step.params.get("view") == "samples" and                 r.views.get("sky_cells") is not None and self._tool_mode is None:
            cells, cell = r.views["sky_cells"]
            grid = (~cells).astype(np.float32)
            self.image.set_overlay(np.kron(grid, np.ones((cell, cell), np.float32)), "#ff5050", scale, 0.35)
        self.histogram.set_image(self.image.preview)
        if getattr(self, "_curve_editor", None) is not None:
            try:
                self._curve_editor.set_histogram(self.image.preview)
            except RuntimeError:
                self._curve_editor = None
        self._refresh_summary()

    def _show_inset(self, index: int, mode: str, special_view: bool) -> None:
        """預覽區：完整解析度的那一塊疊在快速預覽上。參數改了正在重算時先顯示上一次的（外框變黃色虛線）。"""
        res = self.chain.region_res()
        if res is None or special_view or mode == "mask" or self._tool_mode in ("crop", "region"):
            self.image.set_inset(None)
            return
        layer = self.layer_view.get()
        r = self.chain.cached(index, res)
        if r is not None:
            got = self.chain.region_display(r, layer)
            if got is None:
                self.image.set_inset(None)
                return
            before = None
            if mode == "compare":
                pr = self.chain.cached(index - 1, res) if index >= 0 else None
                pg = self.chain.region_display(pr, layer) if pr is not None else None
                if pg is not None and pg[1] == got[1] and pg[0].shape == got[0].shape:
                    before = pg[0]
            self._inset = (index, layer, got[0], got[1], before)
            self.image.set_inset(got[0], got[1], before, stale=False)
        elif self._inset is not None and self._inset[0] == index and self._inset[1] == layer:
            self.image.set_inset(self._inset[2], self._inset[3], self._inset[4] if mode == "compare" else None,
                                 stale=True)
        else:
            self.image.set_inset(None)

    def _show_overlay(self) -> None:
        """筆刷畫的排除區域（去光梯度）或遮罩範圍，畫成紅色疊在影像上。"""
        step = self.current_step()
        if step is None or self.chain.source is None:
            return
        strokes = None
        if step.kind == "gradient" and (self._tool_mode == "brush_exclude" or step.params.get("exclude")):
            strokes = step.params.get("exclude", [])
        elif self._tool_mode == "brush_mask":
            strokes = step.mask.get("strokes", [])
        if strokes is None:
            if self.view_mode.get() != "mask":
                self.image.set_overlay(None)
            return
        r = self._result_for(self._view_index())
        if r is None:
            return
        layer = next(iter(r.state.layers.values()))
        s = self.chain.scale("quick")
        h, w = int(round(layer.shape[-2] * self.chain.scale(r.res) / s)), int(round(layer.shape[-1] *
                                                                                   self.chain.scale(r.res) / s))
        self.image.set_overlay(masks.rasterize(strokes, (h, w), s), "#ff5050", s, 0.38)

    def _show_mask_overlay(self) -> None:
        step = self.current_step()
        if step is None:
            return
        prev = self.chain.cached(self.selected - 1, "quick")
        if prev is None:
            return
        tool = tools_mod.get(step.kind)
        names = tool.target_layers(prev.state, step.layer)
        img = ops.planes(prev.state.layers[names[0]])
        linear = prev.state.phase.get(names[0]) == "linear"
        s = self.chain.scale("quick")
        m = masks.build(step.mask, img, linear, s, {})
        if m is None:
            m = np.ones(img.shape[-2:], np.float32)
        self.image.set_overlay(m, "#ff4040", s, 0.5)

    def _hovered(self, fx: float, fy: float) -> None:
        text = ""
        if fx >= 0 and self.image.full is not None:
            p = self.image.full
            s = self.image.pixel_scale
            x, y = int(fx / s), int(fy / s)
            inset = self.image.inset_value(fx, fy)
            if inset is not None:
                text = f"x {int(fx)}, y {int(fy)}  ·  " + " ".join(f"{float(v):.3f}" for v in inset)
            elif 0 <= y < p.shape[1] and 0 <= x < p.shape[2]:
                vals = " ".join(f"{float(p[c, y, x]):.3f}" for c in range(p.shape[0]))
                text = f"x {int(fx)}, y {int(fy)}  ·  {vals}"
        self._hover_text = text
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        if self._hover_text:
            self.set_summary(self._hover_text)
        elif self.image.has_image:
            z = round(self.image.current_zoom() * self.image.devicePixelRatioF() * 100)
            res = self._result_for(self._view_index())
            tag = "" if res is None else (str(T(" · 預覽", " · preview")) if res.res == "quick" else
                                          str(T(" · 完整解析度", " · full resolution")))
            if self.image.inset is not None:
                tag += str(T(" · 預覽區是完整解析度", " · preview area at full resolution"))
            self.set_summary(f"{z}%{tag}")
        else:
            self.set_summary("")

    def refresh(self) -> None:
        has = self.chain.source is not None
        self.center.setCurrentIndex(1 if has else 0)
        self.export_btn.setEnabled(has)
        self.project_btn.setEnabled(has)
        self.view_seg.setEnabled(has)
        self.refresh_view()

    # ------------------------------------------------------------------ 出圖

    def ask_export(self) -> None:
        if self.chain.source is None or not self.chain.steps:
            return
        dlg = ExportDialog(self, self.export_opts)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.export_opts = dlg.options()
        save_settings(studio_export=self.export_opts.__dict__)
        src = self.chain.source
        default = src.paths[0].with_name(f"{src.paths[0].stem}_APU.{dlg.fmt()}")
        filt = {"png": "PNG (*.png)", "jpg": "JPEG (*.jpg)", "tif": "TIFF 16-bit (*.tif)",
                "fits": "FITS 32-bit (*.fits)"}[dlg.fmt()]
        path, _ = QFileDialog.getSaveFileName(self, str(T("出圖", "Export")), str(default), filt)
        if not path:
            return
        self._export(Path(path))

    def _export(self, path: Path) -> None:
        if self._job is not None:
            self._job.cancelled.set()
        self._gen += 1
        job = _Job(self._gen, len(self.chain.steps) - 1, "full")
        self._job = job
        steps = self.chain.snapshot()
        opts, header, post = self.export_opts, self.chain.source.header, self.bridge.post
        self._busy_since = time.time()

        def work() -> None:
            try:
                r = self.chain.evaluate(len(steps) - 1, "full", steps, job.cancelled.is_set,
                                        lambda f, s: post(("progress", job.gen, f * 0.95, s, "full")))
                disp = ch.display_of(r.state)
                lin = ops.planes(r.state.layers.get("main", disp))
                export.save(disp, path, opts, header, lin)
                post(("result", job.gen, r))
                post(("exported", path.name))
            except Cancelled:
                post(("cancelled", job.gen))
            except Exception:  # noqa: BLE001
                post(("error", job.gen, traceback.format_exc()))

        run_thread(work, "studio-export")
        self.set_status(lambda: str(T(f"出圖中：{path.name}…", f"Exporting {path.name}…")), 0.0)
        self.bridge.event.connect(self._exported_once)

    def _exported_once(self, ev: tuple) -> None:
        if ev[0] == "exported":
            name = ev[1]
            self.set_status(lambda: str(T(f"已出圖：{name}", f"Exported {name}")), None)
            try:
                self.bridge.event.disconnect(self._exported_once)
            except (RuntimeError, TypeError):
                pass

    def rebuild(self) -> None:
        super().rebuild()
        self.refresh_view()


class ExportDialog(QDialog):
    def __init__(self, parent: QWidget, opts: export.ExportOptions):
        super().__init__(parent)
        self.setWindowTitle(str(T("出圖", "Export")))
        self.setMinimumWidth(420)
        form = QFormLayout(self)
        form.setContentsMargins(18, 16, 18, 14)
        form.setVerticalSpacing(10)
        self.format = QComboBox()
        for key, text in (("png", "PNG（8-bit）"), ("jpg", "JPEG"), ("tif", "TIFF（16-bit）"),
                          ("fits", "FITS（32-bit）")):
            self.format.addItem(text, key)
        form.addRow(str(T("格式", "Format")), self.format)
        self.size = QComboBox()
        for v, text in ((0, str(T("原尺寸", "Original size"))), (4096, "4096 px"), (2048, str(T("2048 px（社群）",
                                                                                                "2048 px (social)"))),
                        (1600, "1600 px"), (1080, "1080 px")):
            self.size.addItem(text, v)
        self.size.setCurrentIndex(max(0, [0, 4096, 2048, 1600, 1080].index(opts.long_side)
                                      if opts.long_side in (0, 4096, 2048, 1600, 1080) else 0))
        form.addRow(str(T("長邊", "Long side")), self.size)
        self.sharpen = Var(float(opts.sharpen))
        form.addRow(str(T("縮圖銳化", "Output sharpening")),
                    ParameterSlider("", self.sharpen, 0, 1, lambda v: f"{v:.0%}", 0.05))
        self.icc = Var(bool(opts.icc))
        form.addRow("", ParameterToggle(str(T("嵌入 sRGB 色彩描述檔", "Embed sRGB profile")), self.icc))
        self.text = QLineEdit(opts.watermark)
        self.text.setPlaceholderText(str(T("例如：© 你的名字", "e.g. © Your Name")))
        form.addRow(str(T("浮水印", "Watermark")), self.text)
        self.position = QComboBox()
        for k, t in (("br", T("右下", "Bottom right")), ("bl", T("左下", "Bottom left")), ("tr", T("右上", "Top right")),
                     ("tl", T("左上", "Top left"))):
            self.position.addItem(str(t), k)
        self.position.setCurrentIndex(["br", "bl", "tr", "tl"].index(opts.position) if opts.position in
                                      ("br", "bl", "tr", "tl") else 0)
        form.addRow(str(T("位置", "Position")), self.position)
        self.opacity = Var(float(opts.opacity))
        form.addRow(str(T("不透明度", "Opacity")), ParameterSlider("", self.opacity, 0.1, 1, lambda v: f"{v:.0%}", 0.05))
        row = QHBoxLayout()
        row.addStretch(1)
        ok = button(str(T("出圖", "Export")), "prominent", on_click=self.accept)
        row.addWidget(ok)
        row.addWidget(button(str(T("取消", "Cancel")), on_click=self.reject))
        form.addRow(row)

    def fmt(self) -> str:
        return self.format.currentData()

    def options(self) -> export.ExportOptions:
        return export.ExportOptions(long_side=int(self.size.currentData()), sharpen=float(self.sharpen.get()),
                                    icc=bool(self.icc.get()), watermark=self.text.text(),
                                    position=self.position.currentData(), opacity=float(self.opacity.get()))


__all__ = ["StudioPage", "os"]
