"""內含的三個組件（APU Pick、APU Photons、APU Processing）的版號：「關於」對話框、打包、Release 說明共用。

打包與測試用的確切版本寫在 packaging/constraints.txt（apu-pick==…）；這裡回報的是實際載入的版本。
"""

from __future__ import annotations

REPO_URL = "https://github.com/ApuHuang/apu-astro-studio"
COMPONENTS = (("Pick", "apu-pick", "astro_light_selector", "https://github.com/ApuHuang/apu-pick"),
              ("Photons", "apu-photons", "apu_photons", "https://github.com/ApuHuang/apu-photons"),
              ("Processing", "apu-processing", "apu_processing", "https://github.com/ApuHuang/apu-processing"))


def versions() -> list[tuple[str, str, str]]:
    """[(名稱, 版號, GitHub 網址), …]，三個組件依流程順序。"""
    import importlib
    return [(name, str(importlib.import_module(module).__version__), url) for name, _dist, module, url in COMPONENTS]


def line() -> str:
    """一行文字，例如「Pick 0.9.0・Photons 0.4.0・Processing 0.2.4」（Release 說明、關於對話框用）。"""
    return "・".join(f"{name} {v}" for name, v, _url in versions())
