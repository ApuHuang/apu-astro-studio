"""APU Astro 打包用的進入點（PyInstaller）。

multiprocessing.freeze_support() 一定要最先呼叫：挑片的量測、疊圖的校正與整合都會開子行程，
打包後子行程跑的也是這個 exe，少了它每個子行程都會再開一個視窗。

--smoke-test <light 資料夾> <結果檔> [--expect-gpu]：確認打包好的程式三個步驟都能真的跑
（build_exe.py 打包完會自動執行）：
1. 視窗：三個步驟的頁面都建得起來、換語言重建不出錯（offscreen，不開視窗）
2. 挑片：多行程量測合成星場
3. 疊圖：CPU 疊一次；這台有 NVIDIA GPU 時再用 GPU 疊一次
4. 後製：把疊好的 master 跑完整的後製管線並存成 PNG
"""

import multiprocessing
import sys


def smoke_test(folder: str, out: str, expect_gpu: bool) -> int:
    import os
    import traceback
    from pathlib import Path

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = Path(out)
    lines = []
    try:
        import rawpy
        from PySide6.QtWidgets import QApplication

        assert rawpy.libraw_version
        app = QApplication.instance() or QApplication([])
        from apu_astro.i18n import set_language
        from apu_astro.theme import apply_theme
        from apu_astro.window import MainWindow

        apply_theme(app)
        win = MainWindow()
        for key in ("pick", "stack", "process"):
            win.step.set(key)
            app.processEvents()
        set_language("en")
        for page in win.pages.values():
            page.rebuild()
        set_language("zh")
        for page in win.pages.values():
            page.rebuild()
        lines.append("gui")

        from astro_light_selector.pipeline import collect_files, measure_files
        files, home = collect_files(Path(folder), None, False)
        frames = measure_files(files, 2, home_of=home)
        assert len(frames) == len(files) and all(f.error is None for f in frames), [f.error for f in frames]
        lines.append(f"pick={len(frames)}")

        from apu_photons import backend
        from apu_photons.engine import Settings, run
        has_gpu = backend.select("auto").gpu
        if not has_gpu and expect_gpu:
            assert backend.last_error and "ModuleNotFound" not in backend.last_error \
                and "ImportError" not in backend.last_error, f"CuPy 沒有打包好：{backend.last_error}"
            lines.append(f"gpu=none({backend.last_error})")
        master = None
        for mode in ["cpu"] + (["gpu"] if has_gpu else []):
            res = run([Path(folder)], result.parent / f"smoke_{mode}",
                      Settings(drizzle=0, workers=2, gpu=mode, crop_common=False), echo=None)
            master = res.output
            lines.append(f"{mode}:frames={res.qc['frames_integrated']}")

        # 後製工作室：預設流程（去光梯度 → 校色 → 反摺積 → AI 降噪 → 去星 → 降噪 → 分開拉伸 → 調色 → 銳化 → 合成星點）；
        # 有預設關著的步驟也打開一起跑
        from apu_astro.studio import chain as ch
        from apu_astro.studio import export, xisf
        from apu_astro.studio.tools import ai
        src = ch.open_source([master])
        c = ch.Chain()
        c.set_source(src)
        c.steps = ch.default_steps(src.color, src.narrowband)
        for s in c.steps:
            s.enabled = True
        r = c.evaluate(len(c.steps) - 1, "full")
        assert not any("AI" in str(w) for w in r.state.warnings), r.state.warnings
        export.save(ch.display_of(r.state), result.parent / "smoke.png", export.ExportOptions(long_side=512))
        lines.append(f"studio{'+ai' if ai.available() else ''}")
        assert xisf.SIGNATURE == b"XISF0100"
        import lz4.block  # noqa: F401  壓縮過的 XISF 要用
        import zstandard  # noqa: F401
        result.write_text("ok " + "; ".join(lines) + "\n", encoding="utf-8")
        return 0
    except Exception:  # noqa: BLE001
        result.write_text("; ".join(lines) + "\n" + traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) >= 4 and sys.argv[1] == "--smoke-test":
        sys.exit(smoke_test(sys.argv[2], sys.argv[3], "--expect-gpu" in sys.argv[4:]))
    from apu_astro.app import main

    sys.exit(main())
