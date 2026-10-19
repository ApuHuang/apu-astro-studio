"""暗房主題：APU Astro 系列同一套色票，Qt 版再加上圓角、細分隔線與動畫。

固定深色、不跟系統設定：星空影像的背景亮度、色偏跟周圍環境有關，淺色視窗會拉走對比的判斷。
色票沿用 Pick／Photons／Processing 的 Darkroom（canvas、accent、prominent、beta、reject 相同），
Qt 版把介面底色分成幾個層次（rail < chrome < card），讓分區不靠粗框線也看得出來。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QWidget


class D:
    """色票與尺寸。"""
    canvas = "#000000"        # 影像底
    base = "#0e0e0f"          # 視窗底
    rail = "#121213"          # 左側步驟欄
    chrome = "#161617"        # 頂部列、狀態列、頁首
    card = "#1e1e20"          # 右側面板的卡片
    card_header = "#232326"
    control = "#2c2c2e"       # 按鈕、輸入框
    control_hover = "#3a3a3c"
    control_pressed = "#48484a"
    segment_bg = "#2a2a2c"
    segment_on = "#48484a"
    separator = "#2c2c2e"     # 細分隔線
    separator_strong = "#3a3a3c"
    label = "#ebebeb"
    secondary = "#8e8e93"
    tertiary = "#636366"
    disabled = "#5c5c60"
    accent = "#5cadff"
    prominent = "#0a84ff"
    prominent_hover = "#2a95ff"
    prominent_pressed = "#0064d2"
    beta = "#ffa842"
    reject = "#ff6961"
    success = "#30d158"
    selection = "#1f3d63"
    selected_step = "#1a2a3d"
    list_bg = "#121213"
    list_alt = "#151517"

    radius = 8
    top_bar_height = 46
    status_bar_height = 28
    header_height = 44
    rail_width = 76
    panel_width = 320


UI_FAMILIES = ("Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Helvetica Neue")
MONO_FAMILIES = ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono")


def _pick_family(candidates: tuple[str, ...]) -> str:
    families = set(QFontDatabase.families())
    return next((f for f in candidates if f in families), QApplication.font().family())


class Fonts:
    """字型：UI 用微軟正黑體（Mac 蘋方），數字用等寬字。點數與 Pick／Photons／Processing 相同。"""

    def __init__(self) -> None:
        self.ui_family = _pick_family(UI_FAMILIES)
        self.mono_family = _pick_family(MONO_FAMILIES)

    def ui(self, size: float = 9, bold: bool = False) -> QFont:
        f = QFont(self.ui_family)
        f.setPointSizeF(size)
        f.setBold(bold)
        return f

    def mono(self, size: float = 9) -> QFont:
        f = QFont(self.mono_family)
        f.setPointSizeF(size)
        f.setStyleHint(QFont.StyleHint.Monospace)
        return f


_fonts: Fonts | None = None


def fonts() -> Fonts:
    global _fonts
    if _fonts is None:
        _fonts = Fonts()
    return _fonts


ASSETS = str(Path(__file__).parent / "assets").replace("\\", "/")


def stylesheet() -> str:
    f = fonts()
    return f"""
* {{ font-family: "{f.ui_family}"; font-size: 9pt; color: {D.label}; }}
QMainWindow, QDialog {{ background: {D.base}; }}
QWidget#Page, QWidget#Body {{ background: {D.base}; }}
QToolTip {{ background: {D.card_header}; color: {D.label}; border: 1px solid {D.separator_strong};
            padding: 5px 8px; border-radius: 6px; }}

QPushButton {{ background: {D.control}; border: 1px solid {D.separator_strong}; border-radius: 7px;
               padding: 4px 14px; min-height: 18px; }}
QPushButton:hover {{ background: {D.control_hover}; }}
QPushButton:pressed {{ background: {D.control_pressed}; }}
QPushButton:checked {{ background: #1e3550; border-color: {D.prominent}; color: white; }}
QPushButton:checked:hover {{ background: #24405f; }}
QPushButton:disabled {{ background: #202022; color: {D.disabled}; border-color: {D.separator}; }}
QPushButton[kind="prominent"] {{ background: {D.prominent}; border-color: {D.prominent}; color: white; }}
QPushButton[kind="prominent"]:hover {{ background: {D.prominent_hover}; border-color: {D.prominent_hover}; }}
QPushButton[kind="prominent"]:pressed {{ background: {D.prominent_pressed}; }}
QPushButton[kind="prominent"]:disabled {{ background: #1d3550; border-color: #1d3550; color: #7f93a8; }}
QPushButton[kind="flat"] {{ background: transparent; border: none; padding: 4px 8px; }}
QPushButton[kind="flat"]:hover {{ background: {D.control}; }}
QPushButton::menu-indicator {{ subcontrol-position: right center; right: 6px; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {D.control}; border: 1px solid {D.separator_strong};
               border-radius: 6px; padding: 3px 8px; selection-background-color: {D.prominent}; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {D.prominent}; }}
QComboBox:disabled, QLineEdit:disabled {{ color: {D.disabled}; background: #202022; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: url({ASSETS}/chevron-down.svg); width: 12px; height: 12px; margin-right: 6px; }}
QComboBox QAbstractItemView {{ background: {D.card_header}; border: 1px solid {D.separator_strong};
                               selection-background-color: {D.prominent}; outline: none; padding: 4px; }}

QMenu {{ background: {D.card_header}; border: 1px solid {D.separator_strong}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {D.prominent}; color: white; }}
QMenu::separator {{ height: 1px; background: {D.separator_strong}; margin: 4px 6px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: #3f3f42; border-radius: 3px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: #56565a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTableView, QTreeView, QListView {{ background: {D.list_bg}; alternate-background-color: {D.list_alt};
               border: none; gridline-color: {D.separator}; selection-background-color: {D.selection};
               selection-color: {D.label}; outline: none; }}
QTableView::item, QTreeView::item {{ padding: 2px 6px; border: none; }}
QTreeView::item:hover, QTableView::item:hover {{ background: #1b1b1e; }}
QTreeView::item:selected, QTableView::item:selected {{ background: {D.selection}; }}
QHeaderView {{ background: {D.chrome}; border: none; }}
QHeaderView::section {{ background: {D.chrome}; color: {D.secondary}; border: none;
               border-right: 1px solid {D.separator}; border-bottom: 1px solid {D.separator}; padding: 5px 8px; }}
QHeaderView::section:hover {{ color: {D.label}; }}
QTableCornerButton::section {{ background: {D.chrome}; border: none; }}

QPlainTextEdit, QTextEdit {{ background: {D.list_bg}; border: none; padding: 10px 14px;
               selection-background-color: {D.selection}; }}
QProgressBar {{ background: {D.control}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {D.accent}; border-radius: 3px; }}
QSplitter::handle {{ background: {D.separator}; }}
QSplitter::handle:hover {{ background: {D.accent}; }}

QSlider::groove:horizontal {{ height: 4px; background: {D.control_hover}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {D.accent}; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: #3a3a3c; }}
QSlider::handle:horizontal {{ background: #f2f2f2; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: white; }}
QSlider::handle:horizontal:disabled {{ background: #6b6b6b; }}

QMessageBox {{ background: {D.card}; }}
QMessageBox QLabel {{ color: {D.label}; }}
QCheckBox {{ spacing: 6px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.ColorRole.Window, D.base), (QPalette.ColorRole.Base, D.list_bg),
                        (QPalette.ColorRole.AlternateBase, D.list_alt), (QPalette.ColorRole.Text, D.label),
                        (QPalette.ColorRole.WindowText, D.label), (QPalette.ColorRole.Button, D.control),
                        (QPalette.ColorRole.ButtonText, D.label), (QPalette.ColorRole.Highlight, D.prominent),
                        (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, D.card_header),
                        (QPalette.ColorRole.ToolTipText, D.label), (QPalette.ColorRole.PlaceholderText, D.tertiary),
                        (QPalette.ColorRole.Link, D.accent)):
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(D.disabled))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(D.disabled))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(D.disabled))
    app.setPalette(pal)
    app.setFont(fonts().ui())
    app.setStyleSheet(stylesheet())


def dark_title_bar(window: QWidget) -> None:
    """Windows 的系統標題列也用深色（Windows 10 20H1 以後），整個視窗才是一致的暗房。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(window.winId())
        value = ctypes.c_int(1)
        for attribute in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value),
                                                          ctypes.sizeof(value)) == 0:
                break
        # Windows 11：標題列底色跟頂部列一樣
        color = ctypes.c_int(0x00171616)  # COLORREF 是 0x00BBGGRR → #161617
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(color), ctypes.sizeof(color))
    except (AttributeError, OSError):
        pass
