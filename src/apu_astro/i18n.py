"""整合版自己的介面文字：繁體中文（預設）與英文。

各步驟原本就有的文字（按鈕、分組、說明）直接用各套件的 tr()，用詞跟獨立版一模一樣；
這裡只放整合版新增的文字（步驟名稱、交棒按鈕、外殼）。換語言時四個套件一起換。
"""

from __future__ import annotations

from astro_light_selector import i18n as pick_i18n
from apu_photons import i18n as photons_i18n
from apu_processing import i18n as processing_i18n

APP_NAME = "APU Astro"
APP_SUBTITLE = "Astrophotography Processing Utility"
LANGUAGES = {"zh": "繁體中文", "en": "English"}
DEFAULT_LANGUAGE = "zh"
_language = DEFAULT_LANGUAGE


def set_language(lang: str) -> None:
    """整合版與三個套件一起換語言。"""
    global _language
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang}")
    _language = lang
    for module in (pick_i18n, photons_i18n, processing_i18n):
        module.set_language(lang)


def get_language() -> str:
    return _language


def tr(key: str, **kw: object) -> str:
    text = _CATALOG[_language].get(key)
    if text is None:
        text = _CATALOG[DEFAULT_LANGUAGE][key]
    return text.format(**kw) if kw else text


_ZH = {
    "step.pick": "挑片",
    "step.stack": "疊圖",
    "step.process": "後製",
    "step.pick.help": "挑片：量測每張 light 的星點與背景，自動挑出要保留的片（APU Pick）",
    "step.stack.help": "疊圖：校正、對齊、整合（APU Photons）",
    "step.process.help": "後製：去光梯度、校色、降噪、縮星、拉伸與合成（APU Processing）",
    "rail.files": "{n} 個檔案",
    "rail.masters": "{n} 個 master",
    "rail.busy": "{p}%",
    "lang.zh": "繁中",
    "lang.en": "EN",
    "btn.send_stack": "送去疊圖",
    "btn.send_stack.n": "送去疊圖（{n} 張）",
    "btn.send_stack.help": "把保留的片加進疊圖（不用先搬檔）；校正檔（dark、flat…）請在疊圖那一步加入",
    "btn.send_process": "送去後製",
    "btn.send_process.help": "把這次疊好的 master 交給後製；同一套器材的多個濾鏡會直接進合成",
    "sent.stack": "已加入 {n} 張保留的片到疊圖",
    "sent.process": "已開啟疊圖結果：{name}",
    "send.nothing": "還沒有保留的片。",
    "close.confirm": "{steps}還在進行，確定要關閉嗎？",
    "close.sep": "、",
    "no_project": "尚未開啟",
    "view.compare": "對照",
    "view.compare.help": "左邊原圖、右邊目前結果，拖曳中間的線比較",
    "compare.left": "原圖",
    "compare.right": "成品",
    "pick.welcome.title": "挑片",
    "pick.welcome.sub": "量測星點、背景與追蹤，自動挑出要保留的 light frame",
    "stack.welcome.sub": "把 light 與 dark、flat、bias 一起加入，自動分類、配對、疊圖",
    "process.welcome.sub": "開一張疊好的影像，或疊圖紀錄（.recipe.json）合成多個濾鏡",
    "drop.hint": "也可以把檔案或資料夾拖進視窗",
    "zoom.fit": "符合視窗",
    "zoom.status": "縮放 {z}%",
    "pixel.value": "x {x}, y {y}  ·  {values}",
    "error.title": "發生錯誤",
    "slider.edit_hint": "點一下可以直接輸入數字",
    "info.tooltip": "說明",
    "btn.yes": "是",
    "btn.no": "否",
    "btn.ok": "確定",
    "btn.cancel": "取消",
    "pick.total_exposure": "保留片總曝光",
    "pick.keys.hint": "鍵盤：K 保留　X 淘汰　A 自動",
    "stack.empty.hint": "加入 light 與 dark、flat、bias 開始（Ctrl+O），或在挑片按「送去疊圖」",
}

_EN = {
    "step.pick": "Pick",
    "step.stack": "Stack",
    "step.process": "Process",
    "step.pick.help": "Pick: measure stars and background in every light frame and keep the good ones (APU Pick)",
    "step.stack.help": "Stack: calibrate, register and integrate (APU Photons)",
    "step.process.help": "Process: gradient removal, colour, noise, stars, stretch and channel combination "
                         "(APU Processing)",
    "rail.files": "{n} files",
    "rail.masters": "{n} masters",
    "rail.busy": "{p}%",
    "lang.zh": "繁中",
    "lang.en": "EN",
    "btn.send_stack": "Send to Stack",
    "btn.send_stack.n": "Send to Stack ({n})",
    "btn.send_stack.help": "Add the kept frames to stacking (no need to move files first); add darks, flats and "
                           "biases in the Stack step",
    "btn.send_process": "Send to Process",
    "btn.send_process.help": "Open this stack's masters for processing; filters from the same optics go straight "
                             "to channel combination",
    "sent.stack": "Added {n} kept frames to stacking",
    "sent.process": "Opened the stack result: {name}",
    "send.nothing": "No kept frames yet.",
    "close.confirm": "{steps} still running. Close anyway?",
    "close.sep": ", ",
    "no_project": "Nothing open",
    "view.compare": "Compare",
    "view.compare.help": "Original on the left, current result on the right; drag the line to compare",
    "compare.left": "Original",
    "compare.right": "Result",
    "pick.welcome.title": "Pick",
    "pick.welcome.sub": "Measure stars, background and tracking, and keep the best light frames",
    "stack.welcome.sub": "Add lights together with darks, flats and biases; they're sorted, matched and stacked",
    "process.welcome.sub": "Open a stacked image, or a stack record (.recipe.json) to combine filters",
    "drop.hint": "You can also drop files or folders onto the window",
    "zoom.fit": "Fit",
    "zoom.status": "Zoom {z}%",
    "pixel.value": "x {x}, y {y}  ·  {values}",
    "error.title": "Something went wrong",
    "slider.edit_hint": "Click to type a value",
    "info.tooltip": "Details",
    "btn.yes": "Yes",
    "btn.no": "No",
    "btn.ok": "OK",
    "btn.cancel": "Cancel",
    "pick.total_exposure": "Kept Exposure",
    "pick.keys.hint": "Keys: K keep · X reject · A auto",
    "stack.empty.hint": "Add lights with darks, flats and biases to begin (Ctrl+O), or use Send to Stack in Pick",
}

_CATALOG = {"zh": _ZH, "en": _EN}
