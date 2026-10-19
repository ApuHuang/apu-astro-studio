"""開發用：不顯示視窗，照一個情境操作後把整個視窗存成 PNG（給 AI 協作者看畫面）。

python tools/shot.py <情境> <輸出.png> [參數...]
情境：empty、pick <資料夾>、stack <檔案或資料夾...>、process <影像或 recipe...>
設定寫到暫存的 APU Astro 設定檔，不動使用者的。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("APU_ASTRO_SETTINGS", str(Path(tempfile.gettempdir()) / "apu_astro_shot_settings.json"))


def main() -> int:
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("SHOT_DUMP", 0)) or 10 ** 6, exit=True)
    scenario, out = sys.argv[1], Path(sys.argv[2])
    args = sys.argv[3:]
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from apu_astro.i18n import set_language
    from apu_astro.theme import apply_theme
    from apu_astro.window import MainWindow

    lang = os.environ.get("SHOT_LANG", "zh")
    set_language(lang)
    apply_theme(app)
    win = MainWindow()
    win.resize(int(os.environ.get("SHOT_W", 1440)), int(os.environ.get("SHOT_H", 920)))
    win.show()

    def pump(seconds: float, until=None) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
            if until is not None and until():
                break
            time.sleep(0.02)

    pump(0.5)
    if scenario == "pick":
        win.step.set("pick")
        page = win.page("pick")
        page.open_folder(Path(args[0]))
        pump(2.0)
        view = os.environ.get("SHOT_VIEW")
        if view:
            page.view.set(view)
            pump(0.5)
            if view == "list" and page.table.rowCount():
                page.table.selectRow(int(os.environ.get("SHOT_ROW", 0)))
                pump(6.0, lambda: page._preview_key in page._previews)
                pump(0.5)
    elif scenario == "stack":
        win.step.set("stack")
        page = win.page("stack")
        page.add_files([Path(a) for a in args])
        pump(30.0, lambda: not page._planning)
        pump(0.5)
        view = os.environ.get("SHOT_VIEW")
        if view:
            page.view.set(view)
            pump(0.5)
    elif scenario == "process":
        win.step.set("process")
        page = win.page("process")
        page.open_paths([Path(a) for a in args])
        pump(60.0, lambda: page.chain.source is not None and not page.loading)
        want = os.environ.get("SHOT_RES", "quick")
        last = len(page.chain.steps) - 1
        pump(600.0, lambda: page.chain.cached(last, want) is not None and (want == "quick" or page._job is None))
        sel = os.environ.get("SHOT_STEP")
        if sel is not None:
            page.steps_panel.select(int(sel))
            pump(60.0, lambda: page.chain.cached(int(sel), "quick") is not None and page._job is None)
        param = os.environ.get("SHOT_PARAM")
        if param and page.current_step() is not None:
            k, v = param.split("=", 1)
            page.current_step().params[k] = v
            page._build_inspector()
            page.refresh_view()
        view = os.environ.get("SHOT_VIEW")
        if view:
            page.view_mode.set(view)
        layer = os.environ.get("SHOT_LAYER")
        if layer:
            page.layer_view.set(layer)
        region = os.environ.get("SHOT_REGION")     # 預覽區：畫面座標 x0,y0,x1,y1（完整尺寸）
        if region:
            x0, y0, x1, y1 = (float(v) for v in region.split(","))
            page._toggle_region()
            page.image.set_crop_box((x0, y0, x1, y1))
            idx = page._view_index()
            pump(300.0, lambda: page.chain.region_res() is not None
                 and page.chain.cached(idx, page.chain.region_res()) is not None and page._job is None)
            zoom = os.environ.get("SHOT_ZOOM")
            if zoom:
                page.image.set_zoom(float(zoom))
                page.image.cx, page.image.cy = (x0 + x1) / 2, (y0 + y1) / 2
                page.image.update()
        pump(1.5)
    elif scenario == "flow":
        # 疊圖 → 送去後製：flow <輸出資料夾> <檔案或資料夾...>；另外存 <輸出>_stack.png
        win.step.set("stack")
        page = win.page("stack")
        page.add_files([Path(a) for a in args[1:]])
        pump(60.0, lambda: not page._planning)
        page.output_dir.set(args[0])
        t0 = time.time()
        page._start(trial=False)
        pump(1800.0, lambda: not page.is_busy() and page.worker is None)
        print(f"stack took {time.time() - t0:.0f} s; outputs:",
              [str(o.output) for o in page.result.outputs] if page.result else page.status_text())
        pump(8.0, lambda: page.image.has_image)
        pump(0.5)
        win.grab().save(str(out.with_name(out.stem + "_stack.png")))
        page.send_to_process()
        proc = win.page("process")
        pump(120.0, lambda: proc.chain.source is not None and not proc.loading)
        last = len(proc.chain.steps) - 1
        pump(300.0, lambda: proc.chain.cached(last, "quick") is not None and proc._job is None)
        pump(1.0)
    elif scenario == "empty":
        win.step.set(args[0] if args else "pick")
        pump(0.5)
    pump(0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(out))
    print("saved", out)
    for page in win.pages.values():
        page.close_page()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
