"""後製工作室左邊的步驟清單：來源在最上面，下面是一步一步的處理；可以新增、複製、上下移、停用、刪除。"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
                               QPushButton, QVBoxLayout, QWidget)

from .. import icons
from ..studio import tools as tools_mod
from ..studio.chain import Step
from ..studio.tools import T
from ..theme import D, fonts
from ..ui.var import Var
from ..ui.widgets import Switch, label

LAYER_TEXT = {"starless": T("無星", "Starless"), "stars": T("星點", "Stars"), "main": T("整張", "All")}
# 去星／合成星點不在選單裡（2026-10-18 先不放：無星圖還不能單獨用）；舊專案檔裡有的照樣能開、能算
GROUPS = [
    (T("拉伸前", "Linear"), ["geometry", "gradient", "color_cal", "deconv", "ai_denoise", "denoise", "stretch"]),
    (T("拉伸後", "Stretched"), ["color_grade", "curves", "sharpen", "denoise"]),
]


class StepRow(QWidget):
    toggled = Signal(bool)

    def __init__(self, number: str, icon: str, title: str, sub: str, enabled: bool | None, warn: bool = False):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
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


class StepPanel(QFrame):
    selected = Signal(int)            # -1＝來源
    structure_changed = Signal()      # 新增、刪除、移動、停用

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
        lay.addLayout(head)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; outline: none; }}"
            f"QListWidget::item {{ border-radius: 8px; margin: 1px 0; }}"
            f"QListWidget::item:selected {{ background: {D.selected_step}; }}"
            f"QListWidget::item:hover:!selected {{ background: #1c1c1e; }}")
        self.list.currentRowChanged.connect(lambda r: self.selected.emit(r - 1))
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

    def set_steps(self, steps: list[Step], source_title: str, source_sub: str, select: int,
                  warn: set[int] | None = None) -> None:
        self.steps = steps
        self.source_title, self.source_sub = source_title, source_sub
        self.warn = warn or set()
        self.list.blockSignals(True)
        self.list.clear()
        self._add_row(StepRow("", "source", source_title or str(T("尚未開啟影像", "No image")), source_sub, None), -1)
        for i, s in enumerate(steps):
            tool = tools_mod.get(s.kind)
            title = s.label or str(tool.title)
            sub = ""
            if s.layer in LAYER_TEXT and s.layer != "auto":
                sub = str(LAYER_TEXT[s.layer])
            if s.kind == "stretch":
                method = {"auto": T("自動", "Auto"), "ghs": T("GHS", "GHS"), "arcsinh": T("保色", "Colour-preserving"),
                          "stf": T("STF", "STF")}.get(s.params.get("method", "auto"))
                sub = " · ".join(x for x in (sub, str(method)) if x)
            if s.mask.get("mode", "all") != "all":
                sub = " · ".join(x for x in (sub, str(T("有遮罩", "Masked"))) if x)
            row = StepRow(str(i + 1), tool.icon, title, sub, s.enabled, i in self.warn)
            row.toggled.connect(lambda on, k=i: self._toggle(k, on))
            self._add_row(row, i)
        self.list.setCurrentRow(select + 1)
        self.list.blockSignals(False)
        self._update_buttons()

    def _add_row(self, row: QWidget, index: int) -> None:
        it = QListWidgetItem()
        it.setSizeHint(QSize(220, 44))
        it.setData(Qt.ItemDataRole.UserRole, index)
        self.list.addItem(it)
        self.list.setItemWidget(it, row)

    def current(self) -> int:
        return self.list.currentRow() - 1

    def _toggle(self, index: int, on: bool) -> None:
        if 0 <= index < len(self.steps):
            self.steps[index].enabled = on
            self.structure_changed.emit()

    def _update_buttons(self) -> None:
        i = self.current()
        has = 0 <= i < len(self.steps)
        self.dup_btn.setEnabled(has)
        self.del_btn.setEnabled(has)
        self.up_btn.setEnabled(has and i > 0)
        self.down_btn.setEnabled(has and i < len(self.steps) - 1)

    def select(self, index: int) -> None:
        self.list.setCurrentRow(index + 1)
        self._update_buttons()


__all__ = ["StepPanel", "QColor", "fonts"]
