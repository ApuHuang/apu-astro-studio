# APU Astro（整合版）

挑片（APU Pick）→ 疊圖（APU Photons）→ 後製（APU Processing）整合成一套，Qt 介面。只在本機打包 Windows 版。

## 用法

- 左邊三個步驟：**挑片 › 疊圖 › 後製**（Alt+1／2／3 切換）。每一步都能單獨用。
- 挑片做完按 **送去疊圖**：保留的片直接加進疊圖（不用先搬檔），再加入 dark、flat、bias。
- 疊圖做完按 **送去後製**：同一套器材的多個濾鏡直接進合成（HOO／SHO／RGB）。
- 檔案或資料夾可以直接拖進視窗（交給目前這一步）。
- 快捷鍵：Ctrl+O 開啟／加入、Ctrl+S 存檔（後製）、Ctrl+0 符合視窗、Ctrl+1 100%。
- 滑桿右邊的數字點一下可以直接輸入。挑片清單：K 強制保留、X 強制淘汰、A 改回自動。
- 後製的「對照」：左邊原圖、右邊成品，拖曳中間的線比較；游標停在影像上，右下角顯示那個像素的線性值。

處理參數跟三個獨立版共用同一份設定（在獨立版調好的，整合版打開就一樣）。

## 開發

```
python -m venv .venv
.venv\Scripts\python -m pip install PySide6 pyqtgraph pytest pyinstaller matplotlib
.venv\Scripts\python -m pip install -e ..\astro-light-selector -e "..\ApuPhotonsUtility[gpu]" -e "..\APU Processing"
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m apu_astro          # 執行
.venv\Scripts\python -m pytest -q          # 測試
.venv\Scripts\python packaging\build_exe.py  # 打包 → dist\APUAstro-<版本>-win64.zip
```

三個原本的專案是用 editable 安裝的本機 repo：Photons 要在 `v0.2` 分支（打包前會檢查版本）。
matplotlib 只是打包時要有（astropy 的打包規則會載入它的繪圖模組，沒有就中斷）；打包會排除它，不會包進程式。
