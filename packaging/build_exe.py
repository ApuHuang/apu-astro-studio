"""打包整合版：Windows 在這台機器，macOS 在 GitHub Actions（.github/workflows/macos.yml；PyInstaller 不能跨平台）。

    .venv\\Scripts\\python packaging\\build_exe.py [--dist 資料夾]

→ Windows：dist/APUAstro/APUAstro.exe 與 dist/APUAstro-<版本>-win64.zip
→ macOS：dist/APUAstro.app 與 dist/APUAstro-<版本>-macos-<arm64|x86_64>.zip

- 三個原本的專案用 editable 安裝的本機 repo（見 README）；打包前檢查版本：Photons 要 0.2 以上
  （整合版用 0.2 才有的「加入檔案」），版本不對就停下來，免得打包到另一個分支的程式。
- GPU（只有 Windows）：跟 APU Photons 一樣包進 CuPy 與 CUDA runtime／NVRTC，有 NVIDIA 顯示卡就用、沒有就用 CPU；
  用不到的 cuBLAS、cuFFT… 打包後刪掉（省 1.8 GB），由冒煙測試確認 GPU 疊圖仍正常。
  macOS 沒有 CUDA：疊圖用 CPU，AI 降噪用 onnxruntime 的 CPU（不包 CuPy）。
- 不包 tkinter、matplotlib（整合版的介面是 Qt，執行時不會載入它們）。
- 打包完用合成星場實際跑一次打包好的程式（視窗、挑片多行程量測、CPU／GPU 疊圖、後製），都正常才算成功。
- 舊的 dist/APUAstro 正在執行（開著 APUAstro.exe）時先停下來：PyInstaller 會先刪舊資料夾，刪到一半遇到鎖住的檔案才失敗，
  開著的程式就少了檔案（2026-10-17 發生過）。要保留開著的版本就用 --dist 打包到別的資料夾。
- 打包成功後刪掉 build/（PyInstaller 的中間檔）：裡面也有一個 APUAstro.exe，但沒有 _internal，
  直接開會出現「找不到指定的模組」（2026-10-18 使用者開錯過）。打包失敗時留著，方便查原因。
"""

from __future__ import annotations

import argparse
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from apu_astro import __version__  # noqa: E402

NAME = "APUAstro"
MAC = sys.platform == "darwin"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
EXCLUDES = ["pytest", "IPython", "PyQt5", "PyQt6", "PySide2", "notebook", "sphinx", "matplotlib", "tkinter",
            "_tkinter", "astro_light_selector.gui", "astro_light_selector.plot", "astro_light_selector.darkroom",
            "apu_photons.gui", "apu_photons.darkroom", "apu_photons.zoomview", "apu_processing.gui",
            "apu_processing.darkroom", "apu_processing.canvas", "apu_processing.curves"]
GPU_CUDA_LIBS = ["cuda_runtime", "cuda_nvrtc"]
UNUSED_CUDA = ("cublas", "cufft", "curand", "cusolver", "cusparse", "nvjitlink")
# 用不到的 Qt 模組（WebEngine 等很大）
QT_EXCLUDES = ["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
               "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtCharts",
               "PySide6.QtDataVisualization", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtDesigner",
               "PySide6.QtBluetooth", "PySide6.QtLocation", "PySide6.QtPositioning", "PySide6.QtSql",
               "PySide6.QtTest", "PySide6.QtWebSockets", "PySide6.QtNetwork"]


def check_versions() -> None:
    import apu_photons
    import apu_processing
    import astro_light_selector

    def major_minor(v: str) -> tuple[int, int]:
        parts = v.split(".")
        return int(parts[0]), int(parts[1])

    found = {"APU Pick": astro_light_selector.__version__, "APU Photons": apu_photons.__version__,
             "APU Processing": apu_processing.__version__}
    print("打包的版本：" + "、".join(f"{k} {v}" for k, v in found.items()))
    if major_minor(apu_photons.__version__) < (0, 2):
        raise SystemExit(f"APU Photons 是 {apu_photons.__version__}，整合版要 0.2 以上：請先更新 ApuPhotonsUtility")


def ensure_not_running(bundle: Path) -> None:
    """資料夾裡的程式正在執行時 Windows 不能改名：先試著改名再改回來。"""
    if not bundle.exists():
        return
    probe = bundle.with_name(bundle.name + ".check")
    try:
        os.rename(bundle, probe)
    except OSError:
        raise SystemExit(f"{bundle} 裡的程式正在執行：請先關掉 APUAstro.exe，或用 --dist 打包到別的資料夾") from None
    os.rename(probe, bundle)


def build() -> Path:
    import PyInstaller.__main__

    assets = ROOT / "src" / "apu_astro" / "assets"
    args = [
        str(ROOT / "packaging" / "gui_entry.py"),
        "--name", NAME, "--windowed", "--noconfirm", "--clean",
        "--paths", str(ROOT / "src"),
        "--add-data", f"{assets}{os.pathsep}apu_astro/assets",
        "--collect-submodules", "apu_astro",
        "--collect-submodules", "photutils",
        "--collect-all", "rawpy",
        "--collect-all", "onnxruntime",
        "--hidden-import", "lz4.block", "--hidden-import", "zstandard",
        "--copy-metadata", "photutils",
        "--distpath", str(DIST), "--workpath", str(BUILD), "--specpath", str(BUILD),
    ]
    # macOS 的圖示用 PNG（PyInstaller 用 Pillow 轉成 .icns）
    icon = assets / ("icon_256.png" if MAC else "app.ico")
    if icon.is_file():
        args += ["--icon", str(icon)]
    if MAC:
        args += ["--osx-bundle-identifier", "io.github.apuhuang.apuastro"]
        for mod in ("cupy", "cupyx", "cupy_backends", "cuda_pathfinder", "nvidia", "fastrlock"):
            args += ["--exclude-module", mod]
    else:
        args += gpu_args()
    for mod in EXCLUDES + QT_EXCLUDES:
        args += ["--exclude-module", mod]
    PyInstaller.__main__.run(args)
    if MAC:
        app = DIST / f"{NAME}.app"
        info = app / "Contents" / "Info.plist"
        with info.open("rb") as f:
            plist = plistlib.load(f)
        plist.update(CFBundleName="APU Astro", CFBundleDisplayName="APU Astro",
                     CFBundleShortVersionString=__version__, CFBundleVersion=__version__)
        with info.open("wb") as f:
            plistlib.dump(plist, f)
        # 改了 Info.plist 要重新簽（ad-hoc）；Apple 晶片上沒有簽章的程式根本不能執行
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
        return app / "Contents" / "MacOS" / NAME
    prune_cuda(DIST / NAME)
    return DIST / NAME / f"{NAME}.exe"


def gpu_args() -> list[str]:
    """Windows 的 GPU：CuPy、CUDA runtime／NVRTC 的 DLL 和標頭檔。"""
    import nvidia

    args = ["--collect-all", "cupy", "--collect-all", "cupyx", "--collect-all", "cupy_backends",
            "--collect-all", "cuda_pathfinder", "--hidden-import", "graphlib", "--copy-metadata", "cupy-cuda12x"]
    base = Path(list(nvidia.__path__)[0])
    for lib in GPU_CUDA_LIBS:
        for dll in (base / lib).rglob("*"):
            if dll.suffix.lower() in (".dll", ".so") or ".so." in dll.name:
                rel = dll.parent.relative_to(base.parent)
                args += ["--add-binary", f"{dll}{os.pathsep}{rel.as_posix()}"]
        args += ["--copy-metadata", f"nvidia-{lib.replace('_', '-')}-cu12"]
    # CuPy 用 NVRTC 編譯 kernel 要 CUDA 標頭檔，且必須與 NVRTC 同版。沒包的話會用使用者自己裝的 CUDA Toolkit，
    # 比 NVRTC 舊就編譯失敗（2026-10-26：__nv_fp8_e8m0 incomplete type）。gui_entry.py 會把 CUDA_PATH 指過來
    args += ["--add-data", f"{base / 'cuda_runtime' / 'include'}{os.pathsep}nvidia/cuda_runtime/include"]
    return args


def prune_cuda(bundle: Path) -> None:
    removed = 0
    for dll in (bundle / "_internal").rglob("*.dll"):
        if dll.name.lower().startswith(UNUSED_CUDA):
            removed += dll.stat().st_size
            dll.unlink()
    print(f"移除用不到的 CUDA 函式庫 {removed / 2**20:.0f} MB")


def make_lights(folder: Path) -> Path:
    """合成星場：用 APU Photons 測試裡的產生器（已知位移、熱像素），10 張彩色 CFA。"""
    photons = Path(__import__("apu_photons").__file__).resolve().parents[2]
    sys.path.insert(0, str(photons))
    from tests.test_drizzle import _dithered
    from tests.test_pipeline import _make

    light, _inputs = _make(folder, bayer="RGGB", with_cal=False, shifts=_dithered(10))
    return light


def smoke_test(exe: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        light = make_lights(Path(tmp))
        out = Path(tmp) / "smoke.txt"
        cmd = [str(exe), "--smoke-test", str(light), str(out)] + ([] if MAC else ["--expect-gpu"])
        proc = subprocess.run(cmd, timeout=1800)
        text = out.read_text(encoding="utf-8") if out.exists() else "(沒有輸出)"
        if proc.returncode != 0 or not text.startswith("ok "):
            raise SystemExit(f"打包好的程式測試失敗（exit {proc.returncode}）：\n{text}")
        print(f"打包好的程式測試通過：{text.strip()}")


def archive() -> Path:
    if MAC:
        # ditto 保留 .app 裡的符號連結、執行權限和簽章（shutil 的 zip 會弄壞）
        path = DIST / f"{NAME}-{__version__}-macos-{platform.machine()}.zip"
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(DIST / f"{NAME}.app"), str(path)],
                       check=True)
    else:
        zip_base = DIST / f"{NAME}-{__version__}-win64"
        path = Path(shutil.make_archive(str(zip_base), "zip", DIST, NAME))
    print(f"壓縮檔：{path}（{path.stat().st_size / 2**20:.0f} MB）")
    return path


def main() -> int:
    if sys.platform not in ("win32", "darwin"):
        raise SystemExit("整合版只打包 Windows 和 macOS 版")
    for stream in (sys.stdout, sys.stderr):  # 輸出導到檔案時預設是 Big5，中文訊息會變亂碼
        stream.reconfigure(encoding="utf-8")
    global DIST
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    DIST = ap.parse_args().dist.resolve()
    check_versions()
    ensure_not_running(DIST / (f"{NAME}.app" if MAC else NAME))
    exe = build()
    smoke_test(exe)
    archive()
    shutil.rmtree(BUILD, ignore_errors=True)
    print(f"已刪除中間檔 {BUILD}（要執行的是 {exe}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
