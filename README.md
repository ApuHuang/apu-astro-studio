# APU Astro

挑片 → 疊圖 → 後製，一套做完的天文攝影軟體（Windows）。

[APU Pick](https://github.com/ApuHuang/apu-pick)（挑片）、[APU Photons](https://github.com/ApuHuang/apu-photons)（疊圖）、
[APU Processing](https://github.com/ApuHuang/apu-processing)（後製）的整合版：挑片、疊圖直接用這兩套的核心，
後製是新的「**後製工作室**」，每一步都能回頭改。給買不起付費軟體、插件的人：免費、順手、預設就好看。

## 特色

**一條龍**

- 挑片做完按「**送去疊圖**」，保留的片直接加進疊圖，不用先搬檔；疊圖做完按「**送去後製**」，
  同一套器材的多個濾鏡直接合成 HOO／SHO／RGB
- 三個步驟也都能單獨用；處理參數跟三個獨立版共用同一份設定

**後製工作室**

- **可以回頭改的步驟鏈**：建議流程是 去光梯度 → 校色 → 反摺積 → AI 降噪 → 去星 → 降噪 → 分開拉伸 → 調色 →
  銳化 → 再降噪 → 合成星點，每一步都能開關、調整、回頭修改，只重算後面的步驟
- **先看快速預覽，停手後換完整解析度**；框一塊「預覽區」，幾秒就用完整解析度看到那一塊的成品
- **去光梯度**：看得到扣掉了什麼（殘差圖），星雲用筆刷排除，平滑或彈性的背景模型
- **校色與調色**：滴管點選中性色、三色輪、自然飽和度、窄帶色盤（哈伯色調等）
- **反摺積**：量出畫面各區的星點形狀再修，星點變圓、亮星不會出現甜甜圈
- **AI 降噪**：用作者自己拍的資料訓練，**沒有使用任何付費軟體的輸出**。背景乾淨、不留一顆顆的點和色斑，暗星保留下來
- **去星、分開拉伸、合成星點**：無星圖可以單獨拉伸，補過的地方帶有這張圖自己的噪聲紋理，強拉伸也看不出補痕，
  星雲的纖維會保留；星點和星雲各自拉伸再合回去，縮星後露出來的地方也不會有一個個圓斑。
  去星不使用任何人的無星圖或付費軟體的輸出
- **遮罩**：亮部、暗部、亮度範圍、星點、顏色、手畫，可以羽化、反轉，套在任何一步
- 讀 XISF（WBPP 的 master，含壓縮格式）、FITS、TIFF；存 PNG／JPEG（附 sRGB 色彩描述）、16-bit TIFF、
  線性浮點 FITS（可以再拿去別的軟體處理），可以加浮水印
- 專案檔（`.apuastro`）存下整條步驟鏈，下次打開接著改
- 介面可以切換繁體中文 / English

## 下載

到 [Releases 頁面](../../releases/latest) 下載 `APUAstro-版本-win64.zip`（Windows 10 / 11，64 位元），不用另外安裝 Python。

在 zip 上按右鍵 →「全部解壓縮」，雙擊 `APUAstro.exe`。出現「Windows 已保護您的電腦」時，按「其他資訊」→「仍要執行」
（程式沒有購買數位簽章）。

有 NVIDIA 顯示卡時疊圖與後製會用它加速；AI 降噪可以用任何支援 DirectX 12 的顯示卡。沒有獨立顯示卡也能用，只是比較慢。

## 使用方式

- 左邊三個步驟：**挑片 › 疊圖 › 後製**（Alt+1／2／3 切換）
- 檔案或資料夾可以直接拖進視窗，交給目前這一步
- 後製：開檔後自動用建議流程處理；左邊選步驟、右邊調參數，停手後自動更新
- 「對照」：左邊原圖、右邊成品，拖曳中間的線比較；游標停在影像上，右下角顯示那個像素的線性值
- 滑桿右邊的數字點一下可以直接輸入；挑片清單 K 強制保留、X 強制淘汰、A 改回自動
- 快捷鍵：Ctrl+O 開啟／加入、Ctrl+S 存檔（後製）、Ctrl+0 符合視窗、Ctrl+1 100%

## 關於 AI 降噪

模型是用作者自己拍的照片訓練的（Noise2Noise）：同一個目標的 sub 分成幾組各自疊圖，同一片天空、不同的噪聲，
互相當答案，不需要「乾淨的標準答案」；再加上從實拍資料取出的真實噪聲，讓模型學到實際的噪聲質感。
不使用任何付費軟體（例如 BlurXTerminator、StarXTerminator、NoiseXTerminator）的輸出當訓練目標。
訓練與檢查的工具在 `tools/ai/`。

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

需要 APU Pick、APU Photons、APU Processing 三個 repo 放在同一層，用 editable 安裝；Photons 要在 `v0.2` 分支
（打包前會檢查版本）。matplotlib 只是打包時要有（astropy 的打包規則會載入它的繪圖模組），不會包進程式。

## 授權

MIT License
