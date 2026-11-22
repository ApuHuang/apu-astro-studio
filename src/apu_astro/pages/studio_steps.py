"""後製工作室左邊的步驟清單：來源在最上面，下面是一步一步的處理；可以新增、複製、上下移、停用、刪除。

兩種顯示（0.2.0）：
- 完整（阿噗流程、自訂）：去星之後到合成星點之前的步驟依分支分組（無星圖／星點／兩者），組內縮排、左邊一條分支顏色的線；
  合成星點標成匯合點。不再靠小字標「無星」「星點」
- 簡易（標準）：摺疊成 4 個階段（線性處理／去星與拉伸／調色與細節／合成）；點階段看它的關鍵滑桿，按右邊的箭頭展開細步驟
清單頂端的按鈕選預設集（標準、阿噗流程、自訂）。
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
                               QPushButton, QVBoxLayout, QWidget)

from .. import icons
from ..studio import presets
from ..studio import tools as tools_mod
from ..studio.chain import Step
from ..studio.tools import T
from ..theme import D, fonts
from ..ui.var import Var
from ..ui.widgets import Switch, label

BRANCH_TEXT = {"starless": T("無星圖分支", "Starless branch"), "stars": T("星點分支", "Stars branch"),
               "both": T("兩個分支都作用", "Both branches")}
BRANCH_COLOR = {"starless": "#5aa9ff", "stars": "#ffb347", "both": "#b58cff"}
GROUPS = [
    (T("拉伸前", "Linear"), ["geometry", "gradient", "color_cal", "deconv", "star_split", "ai_denoise", "denoise",
                            "stretch"]),
    (T("拉伸後", "Stretched"), ["color_grade", "curves", "sharpen", "denoise", "star_merge"]),
]


class StepRow(QWidget):
    toggled = Signal(bool)

    def __init__(self, number: str, icon: str, title: str, sub: str, enabled: bool | None, warn: bool = False,
                 branch: str | None = None, indent: int = 0):
        super().__init__()
        self.branch = branch
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8 + indent, 4, 8, 4)
        lay.setSpacing(8)
        n = label(number, D.tertiary, 8, mono=True)
        n.setFixedWidth(16)
        lay.addWidget(n)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon, D.label if enabled is not False else D.disabled, 18))
        ic.setStyleSheet("background: transparent;")
        lay.addWidget(ic)
        box = QVBoxLayout()
        box.setSpacing(0)
        t = label(title, D.label if enabled is not False else D.disabled, 9)
        box.addWidget(t)
        if sub:
            box.addWidget(label(sub, D.beta if warn else D.secondary, 7.5))
        lay.addLayout(box, 1)
        self.switch = None
        if enabled is not None:
            self.var = Var(bool(enabled))
            self.switch = Switch(self.var)
            self.switch.setFixedSize(26, 15)
            self.var.subscribe(self.toggled.emit, self)
            lay.addWidget(self.switch)

    def paintEvent(self, e) -> None:  # noqa: N802
        super().paintEvent(e)
        if self.branch in BRANCH_COLOR:   # 分支：左邊一條顏色的線
            p = QPainter(self)
            p.fillRect(6, 2, 3, self.height() - 4, QColor(BRANCH_COLOR[self.branch]))


class HeaderRow(QWidget):
    """分支的組名（不能選）。"""

    def __init__(self, branch: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 0)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background: {BRANCH_COLOR[branch]}; border-radius: 4px;")
        lay.addWidget(dot)
        lay.addWidget(label(str(BRANCH_TEXT[branch]), BRANCH_COLOR[branch], 8, bold=True), 1)


class StageRow(QWidget):
    """簡易顯示的一個階段：名稱、裡面的步驟、展開箭頭。"""
    expand = Signal()

    def __init__(self, number: int, title: str, sub: str, expanded: bool, enabled: bool):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 6, 6)
        lay.setSpacing(8)
        n = label(str(number), D.tertiary, 9, mono=True)
        n.setFixedWidth(16)
        lay.addWidget(n)
        box = QVBoxLayout()
        box.setSpacing(1)
        box.addWidget(label(title, D.label if enabled else D.disabled, 10, bold=True))
        if sub:
            s = label(sub, D.secondary, 7.5)
            s.setWordWrap(True)
            box.addWidget(s)
        lay.addLayout(box, 1)
        b = QPushButton()
        b.setIcon(icons.icon("down" if not expanded else "up", D.secondary, 14))
        b.setFixedSize(24, 24)
        b.setFlat(True)
        b.setToolTip(str(T("展開細步驟", "Show the detailed steps") if not expanded else T("收合", "Collapse")))
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(self.expand.emit)
        self.expand_btn = b
        lay.addWidget(b)


class StepPanel(QFrame):
    selected = Signal(int)            # -1＝來源
    stage_selected = Signal(int)      # 簡易顯示：點了第幾個階段
    structure_changed = Signal()      # 新增、刪除、移動、停用
    preset_action = Signal(str)       # "choose:<預設集>"、"save"、"delete:<名稱>"

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("StepPanel")
        self.setFixedWidth(250)
        self.setStyleSheet(f"QFrame#StepPanel {{ background: {D.chrome}; border-right: 1px solid {D.separator}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label(str(T("處理步驟", "Steps")), D.label, 9.5, bold=True), 1)
        self.preset_btn = QPushButton(str(presets.title_of(presets.STANDARD)))
        self.preset_btn.setToolTip(str(T("預設集：標準（4 個階段、關鍵滑桿）、阿噗流程（全部步驟），或自己存的",
                                         "Preset: Standard (4 stages, key sliders), APU workflow (every step), "
                                         "or your own")))
        self.preset_menu = QMenu(self.preset_btn)
        self.preset_btn.setMenu(self.preset_menu)
        head.addWidget(self.preset_btn)
        lay.addLayout(head)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; outline: none; }}"
            f"QListWidget::item {{ border-radius: 8px; margin: 1px 0; }}"
            f"QListWidget::item:selected {{ background: {D.selected_step}; }}"
            f"QListWidget::item:hover:!selected {{ background: #1c1c1e; }}")
        self.list.currentRowChanged.connect(self._row_changed)
        lay.addWidget(self.list, 1)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.add_btn = self._tool_button("plus", T("新增步驟", "Add step"))
        self.add_menu = QMenu(self.add_btn)
        self.add_btn.clicked.connect(lambda: self.add_menu.exec(self.add_btn.mapToGlobal(self.add_btn.rect().bottomLeft())))
        self.dup_btn = self._tool_button("copy", T("複製這一步", "Duplicate step"))
        self.up_btn = self._tool_button("up", T("往上移", "Move up"))
        self.down_btn = self._tool_button("down", T("往下移", "Move down"))
        self.del_btn = self._tool_button("trash", T("刪除這一步", "Delete step"))
        for b in (self.add_btn, self.dup_btn, self.up_btn, self.down_btn, self.del_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.steps: list[Step] = []
        self.source_title = ""
        self.source_sub = ""
        self.warn: set[int] = set()
        self.view = presets.VIEW_FULL
        self.expanded: set[int] = set()
        self.stage: int | None = None

    def _tool_button(self, icon: str, tip: T) -> QPushButton:
        b = QPushButton()
        b.setIcon(icons.icon(icon, D.label, 16))
        b.setIconSize(QSize(16, 16))
        b.setFixedSize(34, 28)
        b.setToolTip(str(tip))
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def build_add_menu(self, on_add) -> None:
        self.add_menu.clear()
        for title, kinds in GROUPS:
            sec = self.add_menu.addSection(str(title))
            sec.setEnabled(False)
            for kind in kinds:
                tool = tools_mod.get(kind)
                act = QAction(icons.icon(tool.icon, D.label, 16), str(tool.title), self.add_menu)
                act.triggered.connect(lambda _c=False, k=kind: on_add(k))
                self.add_menu.addAction(act)

    def set_presets(self, current: str | None, custom: list[str]) -> None:
        """預設集選單：內建的、自訂的、存目前的流程、刪除自訂的。"""
        self.preset_btn.setText(presets.title_of(current) if current else str(T("自訂", "Custom")))
        m = self.preset_menu
        m.clear()
        for key, title in presets.BUILTIN.items():
            act = m.addAction(str(title))
            act.setCheckable(True)
            act.setChecked(current == key)
            act.triggered.connect(lambda _c=False, k=key: self.preset_action.emit("choose:" + k))
        if custom:
            m.addSeparator()
            for name in custom:
                act = m.addAction(name)
                act.setCheckable(True)
                act.setChecked(current == presets.CUSTOM + name)
                act.triggered.connect(lambda _c=False, k=name: self.preset_action.emit("choose:" + presets.CUSTOM + k))
        m.addSeparator()
        m.addAction(str(T("把目前的流程存成預設集…", "Save current steps as a preset…")),
                    lambda: self.preset_action.emit("save"))
        if custom:
            sub = m.addMenu(str(T("刪除自訂預設集", "Delete a custom preset")))
            for name in custom:
                sub.addAction(name, lambda k=name: self.preset_action.emit("delete:" + k))

    # ------------------------------------------------------------------ 列

    def set_steps(self, steps: list[Step], source_title: str, source_sub: str, select: int,
                  warn: set[int] | None = None, view: str | None = None, stage: int | None = None,
                  expanded: set[int] | None = None) -> None:
        self.steps = steps
        self.source_title, self.source_sub = source_title, source_sub
        self.warn = warn or set()
        if view is not None:
            self.view = view
        if expanded is not None:
            self.expanded = set(expanded)
        self.stage = stage
        self.list.blockSignals(True)
        self.list.clear()
        self._add_row(StepRow("", "source", source_title or str(T("尚未開啟影像", "No image")), source_sub, None),
                      ("source", -1))
        if self.view == presets.VIEW_SIMPLE:
            self._fill_simple()
        else:
            self._fill_full()
        self.list.setCurrentRow(self._row_of(("stage", stage) if stage is not None else ("step", select)))
        self.list.blockSignals(False)
        self._update_buttons()

    def _step_row(self, i: int, branch: str | None, indent: int, merge: bool = False) -> StepRow:
        s = self.steps[i]
        tool = tools_mod.get(s.kind)
        title = s.label or str(tool.title)
        parts = []
        if merge:
            parts.append(str(T("匯合點：無星圖＋星點", "Merge point: starless + stars")))
        if s.kind == "stretch":
            parts.append(str({"auto": T("自動", "Auto"), "ghs": T("GHS", "GHS"), "arcsinh": T("保色", "Colour-preserving"),
                              "stf": T("STF", "STF")}.get(s.params.get("method", "auto"))))
        if s.mask.get("mode", "all") != "all":
            parts.append(str(T("有遮罩", "Masked")))
        row = StepRow(str(i + 1), tool.icon, title, " · ".join(parts), s.enabled, i in self.warn, branch, indent)
        row.toggled.connect(lambda on, k=i: self._toggle(k, on))
        return row

    def _fill_full(self) -> None:
        br = presets.branch_of(self.steps)
        for row in presets.display_order(self.steps):
            if row[0] == "header":
                self._add_row(HeaderRow(row[1]), ("header", row[1]), 24, selectable=False)
            else:
                i, branch = row[1], row[2]
                self._add_row(self._step_row(i, branch, 10 if branch else 0, br[i] == "merge"), ("step", i))

    def _fill_simple(self) -> None:
        stages = presets.stage_of(self.steps)
        for k, title in enumerate(presets.STAGES):
            members = [i for i, s in enumerate(stages) if s == k]
            if not members:
                continue
            br = presets.branch_of(self.steps)
            short = {"starless": T("無星圖", "starless"), "stars": T("星點", "stars")}
            names = "、".join((self.steps[i].label or str(tools_mod.get(self.steps[i].kind).title)) +
                             (f"（{short[br[i]]}）" if br[i] in short else "") for i in members if self.steps[i].enabled)
            on = any(self.steps[i].enabled for i in members)
            row = StageRow(k + 1, str(title), names or str(T("（都關著）", "(all off)")), k in self.expanded, on)
            row.expand.connect(lambda k=k: self._toggle_expand(k))
            self._add_row(row, ("stage", k), 72)
            if k in self.expanded:
                br = presets.branch_of(self.steps)
                for i in members:
                    b = br[i] if br[i] in BRANCH_COLOR else None
                    self._add_row(self._step_row(i, b, 18, br[i] == "merge"), ("step", i))

    def _add_row(self, row: QWidget, data: tuple, height: int = 44, selectable: bool = True) -> None:
        it = QListWidgetItem()
        it.setSizeHint(QSize(220, height))
        it.setData(Qt.ItemDataRole.UserRole, data)
        if not selectable:
            it.setFlags(Qt.ItemFlag.NoItemFlags)
        self.list.addItem(it)
        self.list.setItemWidget(it, row)

    def _row_of(self, data: tuple) -> int:
        for r in range(self.list.count()):
            if self.list.item(r).data(Qt.ItemDataRole.UserRole) == data:
                return r
        if data[0] == "step" and self.view == presets.VIEW_SIMPLE and 0 <= data[1] < len(self.steps):
            return self._row_of(("stage", presets.stage_of(self.steps)[data[1]]))
        return 0

    def _row_changed(self, r: int) -> None:
        if r < 0:
            return
        kind, value = self.list.item(r).data(Qt.ItemDataRole.UserRole)
        if kind == "stage":
            self.stage = value
            self.stage_selected.emit(value)
        elif kind in ("step", "source"):
            self.stage = None
            self.selected.emit(value)
        self._update_buttons()

    def _toggle_expand(self, k: int) -> None:
        self.expanded ^= {k}
        sel = self.current()
        self.set_steps(self.steps, self.source_title, self.source_sub, sel, self.warn, stage=self.stage)

    def current(self) -> int:
        it = self.list.currentItem()
        if it is None:
            return -1
        kind, value = it.data(Qt.ItemDataRole.UserRole)
        return value if kind in ("step", "source") else -1

    def _toggle(self, index: int, on: bool) -> None:
        if 0 <= index < len(self.steps):
            self.steps[index].enabled = on
            self.structure_changed.emit()

    def _update_buttons(self) -> None:
        i = self.current()
        has = 0 <= i < len(self.steps)
        self.dup_btn.setEnabled(has)
        self.del_btn.setEnabled(has)
        self.up_btn.setEnabled(has and presets.neighbour_in_branch(self.steps, i, -1) is not None)
        self.down_btn.setEnabled(has and presets.neighbour_in_branch(self.steps, i, 1) is not None)

    def select(self, index: int) -> None:
        self.list.setCurrentRow(self._row_of(("step", index)))
        self._update_buttons()

    def select_stage(self, stage: int) -> None:
        self.list.setCurrentRow(self._row_of(("stage", stage)))
        self._update_buttons()


__all__ = ["StepPanel", "QColor", "fonts"]
